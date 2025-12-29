# -*- coding: utf-8 -*-
"""
Illuminate an area, collecting onto the camera. Interleave a signal and control sequence
and plot the difference

Created on Fall 2023

@author: Saroj Chand
"""

import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from matplotlib.ticker import MaxNLocator
from scipy import ndimage
from scipy.interpolate import griddata
from scipy.optimize import curve_fit
from scipy.special import factorial
from scipy.stats import ks_2samp, poisson

from analysis.bimodal_histogram import (
    ProbDist,
    determine_threshold,
    fit_bimodal_histogram,
)
from majorroutines.widefield import base_routine
from utils import common, widefield
from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import positioning as pos
from utils import tool_belt as tb
from utils.constants import ChargeStateEstimationMode, NVSig, VirtualLaserKey

import numpy as np
import matplotlib.pyplot as plt

import numpy as np
from utils import widefield

def add_states_from_counts(data):
    if "states" in data:
        return data

    counts = np.asarray(data["counts"])          # (2, n_nv, n_run, n_step, n_rep)
    nv_list = data["nv_list"]

    states_by_exp = []
    for exp_ind in range(counts.shape[0]):
        counts_exp = counts[exp_ind]             # (n_nv, n_run, n_step, n_rep)
        states_exp = widefield.threshold_counts(nv_list, counts_exp)
        states_by_exp.append(states_exp)

    data["states"] = np.asarray(states_by_exp)   # (2, n_nv, n_run, n_step, n_rep)
    return data

def _states_to_by_nv(data, exp_ind=0):
    # Expect something like: (num_exps, num_nvs, num_runs, num_steps, num_reps)
    states = np.array(data["states"])
    if states.ndim < 3:
        raise ValueError(f"Unexpected states shape: {states.shape}")

    # Handle either [num_exps, ...] or already-selected
    if states.ndim == 5:
        s = states[exp_ind]              # (num_nvs, num_runs, num_steps, num_reps)
        s = s[:, :, 0, :]                # step=0 -> (num_nvs, num_runs, num_reps)
    elif states.ndim == 4:
        s = states[:, :, 0, :]           # (num_nvs, num_runs, num_reps)
    else:
        raise ValueError(f"Unsupported states shape: {states.shape}")

    num_nvs, num_runs, num_reps = s.shape
    return s, num_nvs, num_runs, num_reps

def process_detect_charge_events(data, exp_ind=0, scramble_trials=200, p_thresh=1e-6):
    s, num_nvs, num_runs, num_reps = _states_to_by_nv(data, exp_ind=exp_ind)

    # Build flip-count time series without mixing run boundaries
    flip_counts = []
    flip_masks = []  # which NVs flipped at each time point
    for r in range(num_runs):
        sr = s[:, r, :]                  # (num_nvs, num_reps)
        # flips = sr[:, 1:] != sr[:, :-1]  # (num_nvs, num_reps-1)
        flips = (sr[:, :-1] == 1) & (sr[:, 1:] == 0)   # only NV- -> NV0
        for t in range(num_reps - 1):
            mask = flips[:, t]
            flip_masks.append(mask)
            flip_counts.append(int(mask.sum()))
    flip_counts = np.array(flip_counts)  # length = num_runs*(num_reps-1)

    # Null by scrambling each NV’s flip train (break correlations, keep per-NV rate)
    flips_all = np.concatenate(
        [(s[:, r, 1:] != s[:, r, :-1]) for r in range(num_runs)],
        axis=1
    )  # (num_nvs, num_runs*(num_reps-1))

    null_samples = []
    T = flips_all.shape[1]
    rng = np.random.default_rng(0)
    for _ in range(scramble_trials):
        perm = np.empty_like(flips_all)
        for i in range(num_nvs):
            perm[i] = rng.permutation(flips_all[i])
        null_samples.append(perm.sum(axis=0))
    null_samples = np.concatenate(null_samples)

    # Empirical p-value for each observed point (right tail)
    # p_t = P_null(F >= F_obs)
    null_sorted = np.sort(null_samples)
    def p_right(x):
        # fraction >= x
        idx = np.searchsorted(null_sorted, x, side="left")
        return (len(null_sorted) - idx) / len(null_sorted)

    pvals = np.array([p_right(x) for x in flip_counts])
    event_inds = np.where(pvals < p_thresh)[0]

    # Plots
    fig1, ax1 = plt.subplots()
    ax1.hist(flip_counts, bins=np.arange(0, num_nvs+2)-0.5, alpha=0.7, label="Data (flip count)")
    ax1.hist(null_samples, bins=np.arange(0, num_nvs+2)-0.5, alpha=0.5, label="Scrambled null")
    ax1.set_xlabel("# NVs that flipped in this step")
    ax1.set_ylabel("Occurrences")
    ax1.set_yscale("log")
    ax1.legend()
    ax1.set_title(f"Flip-coincidence distribution (exp {exp_ind})")

    fig2, ax2 = plt.subplots()
    ax2.plot(flip_counts, lw=1)
    ax2.scatter(event_inds, flip_counts[event_inds], s=20)
    ax2.set_xlabel("Time index (run/rep collapsed, no run-boundary flips)")
    ax2.set_ylabel("# flips")
    ax2.set_title(f"Detected events: {len(event_inds)} (p < {p_thresh})")

    # Return candidate events with which NVs flipped
    events = []
    for k in event_inds:
        nv_inds = np.where(flip_masks[k])[0].tolist()
        events.append({"t_index": int(k), "flip_count": int(flip_counts[k]), "nv_inds": nv_inds, "p": float(pvals[k])})

    return (fig1, fig2), events



def detect_coincidence_events_old(data, exp_ind=1, scramble_trials=2000, p_thresh=1e-4, seed=0):
    """
    Old data: states shape (num_exps=2, num_nvs, num_runs, num_steps=1, num_reps)
    We detect shots where many NVs are in NV0 simultaneously.
    Null is built by circularly shifting each NV's time series by a random amount (breaks simultaneity).
    """
    states = np.asarray(data["states"])  # (2, n_nv, n_run, 1, n_rep)
    s = states[exp_ind, :, :, 0, :]      # (n_nv, n_run, n_rep)
    n_nv, n_run, n_rep = s.shape
    n_shots = n_run * n_rep

    # Flatten shots: (n_nv, n_shots)
    s_flat = s.reshape(n_nv, n_shots)

    # Coincidences = number of NVs in NV0 per shot.
    # Assumes state=1 means NV- and state=0 means NV0 (matches your existing code).
    coincid = n_nv - np.sum(s_flat, axis=0)  # (n_shots,)

    # Build scrambled null distribution of coincidences
    rng = np.random.default_rng(seed)
    null = np.empty((scramble_trials, n_shots), dtype=np.int16)
    for k in range(scramble_trials):
        scr = np.empty_like(s_flat)
        for i in range(n_nv):
            shift = rng.integers(0, n_shots)
            scr[i] = np.roll(s_flat[i], shift)
        null[k] = n_nv - np.sum(scr, axis=0)
    null_flat = null.ravel()

    # Empirical right-tail p-value for each observed shot
    null_sorted = np.sort(null_flat)
    def p_right(x):
        idx = np.searchsorted(null_sorted, x, side="left")
        return (len(null_sorted) - idx) / len(null_sorted)

    pvals = np.array([p_right(x) for x in coincid])
    event_shots = np.where(pvals < p_thresh)[0]

    # Make plots
    fig1, ax1 = plt.subplots()
    bins = np.arange(0, n_nv + 2) - 0.5
    ax1.hist(coincid, bins=bins, alpha=0.7, label="Data")
    ax1.hist(null_flat, bins=bins, alpha=0.5, label="Scrambled null")
    ax1.set_yscale("log")
    ax1.set_xlabel("# NVs in NV0 (per shot)")
    ax1.set_ylabel("Occurrences")
    ax1.set_title(f"Coincidence distribution (exp_ind={exp_ind})")
    ax1.legend()

    fig2, ax2 = plt.subplots()
    ax2.plot(coincid, lw=1)
    ax2.scatter(event_shots, coincid[event_shots], s=25)
    ax2.set_xlabel("Shot index (flattened run×rep)")
    ax2.set_ylabel("# NVs in NV0")
    ax2.set_title(f"Detected events: {len(event_shots)} (p < {p_thresh})")

    # Package events with run/rep + which NVs were NV0
    events = []
    for shot in event_shots:
        run_ind = shot // n_rep
        rep_ind = shot % n_rep
        nv0_inds = np.where(s_flat[:, shot] == 0)[0].tolist()
        events.append({
            "shot": int(shot),
            "run": int(run_ind),
            "rep": int(rep_ind),
            "nv0_count": int(coincid[shot]),
            "nv0_inds": nv0_inds,
            "p": float(pvals[shot]),
        })

    # Quick diagnostics
    print(f"[exp {exp_ind}] shots={n_shots}, mean NV0={coincid.mean():.3f}, max NV0={coincid.max()}")
    if len(events) == 0:
        # show the top few most extreme shots anyway
        top = np.argsort(coincid)[-10:][::-1]
        print("Top shots by NV0 count:")
        for t in top:
            print(f"  shot={t} run={t//n_rep} rep={t%n_rep} NV0={coincid[t]} p≈{p_right(coincid[t]):.2e}")

    return (fig1, fig2), events


import numpy as np
import matplotlib.pyplot as plt
from utils import widefield


def add_states_from_counts(data):
    """Create data['states'] from data['counts'] using widefield.threshold_counts."""
    if "states" in data:
        return data

    counts = np.asarray(data["counts"])          # (E, N, R, step, rep)
    nv_list = data["nv_list"]

    states_by_exp = []
    for exp_ind in range(counts.shape[0]):
        counts_exp = counts[exp_ind]             # (N, R, step, rep)
        states_exp = widefield.threshold_counts(nv_list, counts_exp)
        states_by_exp.append(states_exp)

    data["states"] = np.asarray(states_by_exp)   # (E, N, R, step, rep)
    return data


def _states_by_exp_step0(data):
    """Return states for step=0 as (E, N, R, rep)."""
    states = np.asarray(data["states"])
    if states.ndim != 5:
        raise ValueError(f"Expected states.ndim=5, got {states.shape}")
    return states[:, :, :, 0, :]  # (E, N, R, rep)


def _empirical_right_tail_pvals(obs, null_samples_flat):
    """Empirical p-values P_null(X >= obs)."""
    null_sorted = np.sort(null_samples_flat)

    def p_right(x):
        idx = np.searchsorted(null_sorted, x, side="left")
        return (len(null_sorted) - idx) / len(null_sorted)

    return np.array([p_right(x) for x in obs])


def detect_transition_bursts_3exp(
    data,
    from_exp,
    to_exp,
    transition="1->0",          # "1->0" (NV- to NV0) or "0->1"
    scramble_trials=2000,
    p_thresh=1e-4,
    scramble_mode="roll",        # "roll" preserves per-NV temporal structure better than "perm"
    seed=0,
):
    """
    Detect bursts of transitions across NVs between two readouts within the same rep.

    Returns:
      figs: (hist_fig, timeseries_fig)
      events: list of dicts with run/rep + NV indices that transitioned
    """
    data = add_states_from_counts(data)
    S = _states_by_exp_step0(data)  # (E, N, R, rep)
    E, N, R, REP = S.shape

    if to_exp >= E or from_exp >= E:
        raise ValueError(f"Requested exp {from_exp}->{to_exp}, but only E={E} exps saved.")

    A = S[from_exp]  # (N, R, rep)
    B = S[to_exp]    # (N, R, rep)

    if transition == "1->0":
        Tmask = (A == 1) & (B == 0)
    elif transition == "0->1":
        Tmask = (A == 0) & (B == 1)
    else:
        raise ValueError("transition must be '1->0' or '0->1'")

    # Flatten shots: (N, shots) where shots = R * REP
    shots = R * REP
    Tflat = Tmask.reshape(N, shots)

    # Observed transition counts per shot
    obs_counts = Tflat.sum(axis=0).astype(int)  # (shots,)

    # Scrambled null (break simultaneity, keep per-NV rate)
    rng = np.random.default_rng(seed)
    null_counts = np.empty((scramble_trials, shots), dtype=np.int16)

    for k in range(scramble_trials):
        scr = np.empty_like(Tflat)
        if scramble_mode == "roll":
            shifts = rng.integers(0, shots, size=N)
            for i in range(N):
                scr[i] = np.roll(Tflat[i], shifts[i])
        elif scramble_mode == "perm":
            for i in range(N):
                scr[i] = rng.permutation(Tflat[i])
        else:
            raise ValueError("scramble_mode must be 'roll' or 'perm'")
        null_counts[k] = scr.sum(axis=0)

    null_flat = null_counts.ravel()
    pvals = _empirical_right_tail_pvals(obs_counts, null_flat)
    event_shots = np.where(pvals < p_thresh)[0]

    # Plots
    bins = np.arange(0, N + 2) - 0.5

    fig1, ax1 = plt.subplots()
    ax1.hist(obs_counts, bins=bins, alpha=0.7, label="Data")
    ax1.hist(null_flat, bins=bins, alpha=0.5, label=f"Null ({scramble_mode})")
    ax1.set_yscale("log")
    ax1.set_xlabel(f"# transitions per shot ({transition})")
    ax1.set_ylabel("Occurrences")
    ax1.set_title(f"Transition bursts: exp {from_exp}->{to_exp}  (events={len(event_shots)})")
    ax1.legend()

    fig2, ax2 = plt.subplots()
    ax2.plot(obs_counts, lw=1)
    ax2.scatter(event_shots, obs_counts[event_shots], s=20)
    ax2.set_xlabel("Shot index (flattened run×rep)")
    ax2.set_ylabel(f"# transitions ({transition})")
    ax2.set_title(f"Detected events: {len(event_shots)} (p < {p_thresh})")

    # Package events with run/rep and which NVs transitioned
    events = []
    for shot in event_shots:
        run_ind = shot // REP
        rep_ind = shot % REP
        nv_inds = np.where(Tflat[:, shot])[0].tolist()
        events.append({
            "shot": int(shot),
            "run": int(run_ind),
            "rep": int(rep_ind),
            "n_trans": int(obs_counts[shot]),
            "nv_inds": nv_inds,
            "p": float(pvals[shot]),
            "from_exp": int(from_exp),
            "to_exp": int(to_exp),
            "transition": transition,
        })

    print(
        f"[exp {from_exp}->{to_exp}] shots={shots}, mean={obs_counts.mean():.3f}, "
        f"max={obs_counts.max()}, events={len(events)}"
    )
    return (fig1, fig2), events


def detect_state_coincidence_3exp(
    data,
    exp_ind,
    scramble_trials=2000,
    p_thresh=1e-4,
    seed=0,
):
    """
    Like your old coincidence detector, but works for any exp index (t0/t1/t2).
    Detect shots with unusually many NV0 simultaneously.
    """
    data = add_states_from_counts(data)
    S = _states_by_exp_step0(data)  # (E, N, R, rep)
    E, N, R, REP = S.shape
    if exp_ind >= E:
        raise ValueError(f"exp_ind={exp_ind}, but only E={E} exps saved.")

    s = S[exp_ind]  # (N, R, REP)
    shots = R * REP
    s_flat = s.reshape(N, shots)

    # NV0 count per shot (assuming 1=NV-, 0=NV0)
    nv0 = (N - np.sum(s_flat, axis=0)).astype(int)

    rng = np.random.default_rng(seed)
    null = np.empty((scramble_trials, shots), dtype=np.int16)
    for k in range(scramble_trials):
        scr = np.empty_like(s_flat)
        shifts = rng.integers(0, shots, size=N)
        for i in range(N):
            scr[i] = np.roll(s_flat[i], shifts[i])
        null[k] = N - np.sum(scr, axis=0)

    null_flat = null.ravel()
    pvals = _empirical_right_tail_pvals(nv0, null_flat)
    event_shots = np.where(pvals < p_thresh)[0]

    fig1, ax1 = plt.subplots()
    bins = np.arange(0, N + 2) - 0.5
    ax1.hist(nv0, bins=bins, alpha=0.7, label="Data")
    ax1.hist(null_flat, bins=bins, alpha=0.5, label="Null (roll)")
    ax1.set_yscale("log")
    ax1.set_xlabel("# NVs in NV0 (per shot)")
    ax1.set_ylabel("Occurrences")
    ax1.set_title(f"NV0 coincidence bursts at exp {exp_ind} (events={len(event_shots)})")
    ax1.legend()

    fig2, ax2 = plt.subplots()
    ax2.plot(nv0, lw=1)
    ax2.scatter(event_shots, nv0[event_shots], s=20)
    ax2.set_xlabel("Shot index (flattened run×rep)")
    ax2.set_ylabel("# NV0")
    ax2.set_title(f"Detected events: {len(event_shots)} (p < {p_thresh})")

    events = []
    for shot in event_shots:
        run_ind = shot // REP
        rep_ind = shot % REP
        nv0_inds = np.where(s_flat[:, shot] == 0)[0].tolist()
        events.append({
            "shot": int(shot),
            "run": int(run_ind),
            "rep": int(rep_ind),
            "nv0_count": int(nv0[shot]),
            "nv0_inds": nv0_inds,
            "p": float(pvals[shot]),
            "exp_ind": int(exp_ind),
        })

    print(f"[exp {exp_ind}] shots={shots}, mean NV0={nv0.mean():.3f}, max NV0={nv0.max()}, events={len(events)}")
    return (fig1, fig2), events


if __name__ == "__main__":
    kpl.init_kplotlib()

    # Load data
    # data = dm.get_raw_data(file_id=1695946921364)  # dark time 1000e9 (november data)
    # DATA IN jAN 2025
    # data = dm.get_raw_data(file_id=1756083081553)  # dark time 1e6
    # data = dm.get_raw_data(file_id=1756161618282)  # dark time 10e6
    # data = dm.get_raw_data(file_id=1757223169229)  # dark time 10e6
    # data = dm.get_raw_data(file_id=1757474735789)  # dark time 10e6
    # data = dm.get_raw_data(file_id=1756305080720)  # dark time 100e6
    # data = dm.get_raw_data(file_id=1756699202091)  # dark time 100e6
    # data = dm.get_raw_data(file_id=1755068762133)  # dark time 1000e9
    # inspect_raw_data(data)

    # data = dm.get_raw_data(file_id=1757562210411)  # dark time 10e6
    # data = dm.get_raw_data(file_id=1757223169229)  # dark time 10e6
    # data = dm.get_raw_data(file_id=1757883746286)  # dark time 10e6
    # data = dm.get_raw_data(file_id=1757904453004)  # dark time 10e6
    # data = dm.get_raw_data(file_id=1758180182062)  # dark time 1ms and 1s
    # data = dm.get_raw_data(file_id=1758336169797)  # dark time 1ms and 1s

    # counts = np.array(data["counts"])  # Extract signal counts
    # counts = np.array(counts).reshape((num_nvs, -1))
    # print(f"Counts shape: {counts.shape}")

    # Process data
    # (hist_fig, transition_fig, spatial_heatmap_fig, time_series_fig) = (
    #     process_detect_cosmic_rays(data)
    # )

    # file_ids = [1756083081553, 1756161618282, 1756305080720, 1755068762133]
    # dark_times = [1e6, 10e6, 100e6, 1000e6]
    # data_files = [1756161618282, 1756305080720, 1756699202091, 1755068762133]
    # dark_times = [10e6, 100e6, 500e6, 1000e6]

    # DATA FILES
    # file_ids = [1758180182062, 1758336169797] # johnson sample deep NVs
    # cannon sampel 147NVs
    # file_ids = [1766974557310] # 1ms and 1s dark time data (50ms readout)
    # file_ids = [1767157983900, 1767269375068, 1767395452581, 1767514737339]
    # rubin 105NVs
    # file_ids = [
    #     1798883656474,
    #     1798997502214,
    #     1799103957221,
    #     1799203370243,
    #     1799297505028,
    # ]

    # 8s wait time (your stems)

    # # file stems
    # file_ids = [
    #     "2025_03_10-23_06_34-rubin-nv0_2025_02_26",
    #     "2025_03_11-05_07_58-rubin-nv0_2025_02_26",
    #     "2025_03_11-11_10_04-rubin-nv0_2025_02_26",
    # ]
    

    data = dm.get_raw_data(file_stem="2025_12_24-22_48_09-johnson-nv0_2025_10_21")
    print(data.keys())
    # data = add_states_from_counts(data)
    print("counts:", np.asarray(data["counts"]).shape)
    # print("states:", np.asarray(data["states"]).shape)

    # Transition bursts (recommended):
    # exp 0 -> 1  : events during short wait t1
    (figs01, ev01) = detect_transition_bursts_3exp(
        data, from_exp=0, to_exp=1, transition="1->0",
        scramble_trials=5000, p_thresh=1e-4
    )

    # exp 1 -> 2  : events during long wait (t2 - t1)
    (figs12, ev12) = detect_transition_bursts_3exp(
        data, from_exp=1, to_exp=2, transition="1->0",
        scramble_trials=5000, p_thresh=1e-4
    )

    # Optional: coincidence bursts at each readout:
    (figs0, ev0) = detect_state_coincidence_3exp(data, exp_ind=0, scramble_trials=5000, p_thresh=1e-4)
    (figs1, ev1) = detect_state_coincidence_3exp(data, exp_ind=1, scramble_trials=5000, p_thresh=1e-4)
    (figs2, ev2) = detect_state_coincidence_3exp(data, exp_ind=2, scramble_trials=5000, p_thresh=1e-4)

    plt.show()


#     datas = [dm.get_raw_data(file_stem=f) for f in file_ids]

#     combined_data = datas[0].copy()
#     for d in datas[1:]:
#         combined_data["counts"] = np.concatenate([combined_data["counts"], d["counts"]], axis=2)

#     combined_data["num_runs"] = np.asarray(combined_data["counts"]).shape[2]

#     print("counts shape:", np.asarray(combined_data["counts"]).shape)

#     # ---- IMPORTANT: create states from counts ----
#     combined_data = add_states_from_counts(combined_data)
#     print("states shape:", np.asarray(combined_data["states"]).shape)

#     # # ---- Now run your event detector ----
#     (figs0, events0) = process_detect_charge_events(combined_data, exp_ind=0, scramble_trials=200, p_thresh=1e-6)
#     (figs1, events1) = process_detect_charge_events(combined_data, exp_ind=1, scramble_trials=200, p_thresh=1e-6)

# # 
#     print("num events exp0:", len(events0))
#     print("num events exp1:", len(events1))

    # # exp_ind=1 is your long wait (8s) in your old setup
    # (figs, events) = detect_coincidence_events_old(
    #     combined_data,
    #     exp_ind=0,
    #     scramble_trials=2000,
    #     p_thresh=1e-4
    # )

    # print("Num events:", len(events))


    # Process data
    # process_detect_cosmic_rays(combined_data)

    # data_files = [1756161618282, 1755068762133]
    # dark_times = [10e6, 1000e6]

    # plot_histogram_multi(data_files, dark_times)

    # nv_list = data["nv_list"]
    # num_nvs = len(nv_list)
    # counts = np.array(data["counts"])[0]  # Extract signal counts
    # states = widefield.threshold_counts(nv_list, counts, dynamic_thresh=True)
    # states = np.array(states).reshape((num_nvs, -1))
    # Parameters
    # window_size = 20000

    # Calculate correlations over time
    # correlations, time_windows = calculate_correlation_over_time(states, window_size)

    # Plot the correlations
    # plot_correlation_over_time(correlations, time_windows)
    kpl.show(block=True)
