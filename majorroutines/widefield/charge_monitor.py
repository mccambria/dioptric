# -*- coding: utf-8 -*-
"""
Illuminate an area, collecting onto the camera. Interleave a signal and control sequence
and plot the difference

Created on Fall 2023

@author: mccambria
"""

import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from scipy import ndimage
from scipy.optimize import curve_fit
from scipy.special import factorial
from scipy.stats import poisson

from majorroutines.widefield import base_routine, optimize
from utils import common, widefield
from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import positioning as pos
from utils import tool_belt as tb
from utils.constants import ChargeStateEstimationMode, NVSig, VirtualLaserKey


def detect_cosmic_rays(nv_list, num_reps, num_runs, dark_time):
    charge_prep = True
    main(
        nv_list,
        num_reps,
        num_runs,
        "detect_cosmic_rays",
        charge_prep,
        process_detect_cosmic_rays,
        dark_time=dark_time,
    )


def check_readout_fidelity(nv_list, num_reps, num_runs):
    charge_prep = False
    main(
        nv_list,
        num_reps,
        num_runs,
        "check_readout_fidelity",
        charge_prep,
        process_check_readout_fidelity,
    )


def charge_quantum_jump(nv_list, num_reps):
    num_runs = 1
    charge_prep = False
    main(
        nv_list,
        num_reps,
        num_runs,
        "charge_quantum_jump",
        charge_prep,
        process_detect_cosmic_rays,
    )


def process_detect_cosmic_rays(data):
    nv_list = data["nv_list"]
    num_nvs = len(nv_list)

    states = np.array(data["states"])[0]
    states_by_nv = np.array([states[nv_ind].flatten() for nv_ind in range(num_nvs)])

    for ind in range(2):
        if ind == 1:
            for nv_ind in range(num_nvs):
                states_by_nv[nv_ind] = np.roll(states_by_nv[nv_ind], 10 * nv_ind)
        coincidences = []
        num_shots = len(states_by_nv[0])
        for shot_ind in range(num_shots):
            coincidences.append(num_nvs - np.sum(states_by_nv[:, shot_ind]))
        coincidences = np.array(coincidences)
        hist_fig, ax = plt.subplots()
        kpl.histogram(ax, coincidences, label=f"Data ({num_nvs} NVs)")
        ax.set_xlabel("Number NVs found in NV0")
        ax.set_ylabel("Number of occurrences")
        x_vals = np.array(range(0, num_nvs + 1))
        expected_dist = num_shots * poisson.pmf(x_vals, np.mean(coincidences))
        kpl.plot_points(
            ax, x_vals, expected_dist, label="Poisson pmf", color=kpl.KplColors.RED
        )
        ax.legend()
        if ind == 0:
            ax.set_title("Unscrambled")
        elif ind == 1:
            ax.set_title("Scrambled")
        ax.set_yscale("log")

    im_fig, ax = plt.subplots()
    kpl.imshow(ax, states_by_nv, aspect="auto")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    return hist_fig, im_fig


def process_check_readout_fidelity(data, fidelity_ax=None):
    nv_list = data["nv_list"]
    num_nvs = len(nv_list)
    counts = np.array(data["counts"])
    num_runs = counts.shape[2]
    num_reps = counts.shape[4]
    sig_counts = counts[0]
    config = common.get_config_dict()
    charge_state_estimation_mode = config["charge_state_estimation_mode"]
    # charge_state_estimation_mode = ChargeStateEstimationMode.THRESHOLDING
    # charge_state_estimation_mode = ChargeStateEstimationMode.MLE
    if charge_state_estimation_mode == ChargeStateEstimationMode.THRESHOLDING:
        states = widefield.threshold_counts(nv_list, sig_counts)
    elif charge_state_estimation_mode == ChargeStateEstimationMode.MLE:
        states = np.array(data["states"])[0]

    figsize = kpl.figsize
    figsize[1] *= 1.5
    fig, axes_pack = plt.subplots(2, 1, sharex=True, figsize=figsize)
    labels = {0: "NV⁰", 1: "NV⁻"}
    probs = [[] for ind in range(2)]
    prob_errs = [[] for ind in range(2)]
    lookback = 2
    for init_state in [0, 1]:
        ax = axes_pack[init_state]
        for nv_ind in range(num_nvs):
            shots_list = []
            for run_ind in range(num_runs):
                for rep_ind in range(num_reps):
                    if rep_ind < lookback:
                        continue
                    prev_states = states[
                        nv_ind, run_ind, 0, rep_ind - lookback : rep_ind
                    ]
                    current_state = states[nv_ind, run_ind, 0, rep_ind]
                    if np.all([el == init_state for el in prev_states]):
                        shots_list.append(current_state == init_state)
            prob = np.mean(shots_list)
            err = np.std(shots_list, ddof=1) / np.sqrt(len(shots_list))
            nv_num = widefield.get_nv_num(nv_list[nv_ind])
            # kpl.plot_points(ax, nv_num, prob, yerr=err)
            kpl.plot_bars(ax, nv_num, prob, yerr=err)
            probs[init_state].append(prob)
            prob_errs[init_state].append(err)
        label = labels[init_state]
        ax.set_ylabel(f"P({label} | previous {lookback} shots {label})")
        ax.set_ylim((0.5, 1.0))

    ax.set_xlabel("NV index")
    # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xticks(range(num_nvs))

    if fidelity_ax is None:
        fig, fidelity_ax = plt.subplots()
    else:
        fig = None
    fidelities = []
    fidelity_errs = []
    for nv_ind in range(num_nvs):
        fidelity = (probs[0][nv_ind] + probs[1][nv_ind]) / 2
        fidelity_err = (
            np.sqrt(prob_errs[0][nv_ind] ** 2 + prob_errs[1][nv_ind] ** 2) / 2
        )
        fidelities.append(fidelity)
        fidelity_errs.append(fidelity_err)
        nv_num = widefield.get_nv_num(nv_list[nv_ind])
        # kpl.plot_points(ax, nv_num, fidelity, yerr=fidelity_err)
        kpl.plot_bars(fidelity_ax, nv_num, fidelity, yerr=fidelity_err)
    fidelity_ax.set_ylabel("Readout fidelity")
    fidelity_ax.set_xlabel("NV index")
    # fidelity_ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    fidelity_ax.set_xticks(range(num_nvs))
    fidelity_ax.set_ylim((0.5, 1.0))
    print(fidelities)
    print(fidelity_errs)

    return fig


def main(
    nv_list,
    num_reps,
    num_runs,
    caller_fn_name,
    charge_prep,
    data_processing_fn,
    dark_time=0,
):
    ### Some initial setup
    seq_file = "charge_monitor.py"

    tb.reset_cfm()
    pulse_gen = tb.get_server_pulse_gen()

    num_steps = 1
    num_exps_per_rep = 1

    ### Collect the data

    def run_fn(shuffled_step_inds):
        pol_coords_list = widefield.get_coords_list(nv_list, VirtualLaserKey.CHARGE_POL)
        seq_args = [pol_coords_list, charge_prep, dark_time]
        seq_args_string = tb.encode_seq_args(seq_args)
        pulse_gen.stream_load(seq_file, seq_args_string, num_reps)

    charge_prep_fn = base_routine.charge_prep_no_verification if charge_prep else None
    raw_data = base_routine.main(
        nv_list,
        num_steps,
        num_reps,
        num_runs,
        run_fn=run_fn,
        num_exps=num_exps_per_rep,
        charge_prep_fn=charge_prep_fn,
    )

    repr_nv_sig = widefield.get_repr_nv_sig(nv_list)
    repr_nv_name = repr_nv_sig.name
    timestamp = dm.get_time_stamp()
    raw_data |= {
        "timestamp": timestamp,
        "caller_fn_name": caller_fn_name,
    }

    try:
        fig = data_processing_fn(raw_data)
    except Exception:
        fig = None

    ### Save and clean up

    file_path = dm.get_file_path(__file__, timestamp, repr_nv_name)
    dm.save_raw_data(raw_data, file_path)
    if fig is not None:
        try:
            num_figs = len(fig)
            for fig_ind in range(num_figs):
                file_path = dm.get_file_path(
                    __file__, timestamp, f"{repr_nv_name}-{fig_ind}"
                )
                dm.save_figure(fig[fig_ind], file_path)
        except Exception:
            dm.save_figure(fig, file_path)

    tb.reset_cfm()


if __name__ == "__main__":
    kpl.init_kplotlib()

    # data = dm.get_raw_data(file_id=1537208030313)  # 50 ms
    data = dm.get_raw_data(file_id=1568108087044)  # 100 ms
    process_check_readout_fidelity(data)

    ###

    # data = dm.get_raw_data(file_id=1567772101718)
    # process_detect_cosmic_rays(data)

    kpl.show(block=True)
