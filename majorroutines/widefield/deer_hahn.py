# -*- coding: utf-8 -*-
"""
Pulsed deer haha on multiple NVs with spin-to-charge
conversion readout imaged onto a camera

Created on Coct 9th, 2025

@author: schand
"""

import os
import sys
import time
import traceback
from random import shuffle

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

from majorroutines.pulsed_resonance import fit_resonance, gaussian, norm_voigt, voigt
from majorroutines.widefield import base_routine
from utils import common
from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import positioning as pos
from utils import tool_belt as tb
from utils import widefield as widefield
from utils.constants import NVSig, NVSpinState
from utils.positioning import get_scan_1d as calculate_freqs


import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional
from scipy.optimize import curve_fit

@dataclass
class DeerResult:
    nv_index: int
    f0: float              # fitted center (GHz)
    amp: float             # fitted amplitude (contrast units)
    width: float           # fitted width (GHz); sigma for Gaussian
    chi2_red: float        # reduced chi^2
    peak_contrast: float   # max |contrast| in data (not fit)
    peak_freq: float       # freq at max |contrast|

def _mean_ste(a, axis=-1):
    """Return mean and standard error along an axis; keeps dims collapsed."""
    a = np.asarray(a, float)
    m = np.mean(a, axis=axis)
    # avoid division by zero for reps=1
    n = a.shape[axis]
    if n <= 1:
        return m, np.zeros_like(m)
    s = np.std(a, axis=axis, ddof=1) / np.sqrt(n)
    return m, s

def _gauss(x, A, sigma, x0, y0):
    return y0 + A * np.exp(-0.5 * ((x - x0) / sigma) ** 2)

def _lorentz(x, A, gamma, x0, y0):
    return y0 + A * (gamma**2) / ((x - x0) ** 2 + gamma**2)

def _fit_1d(x, y, yerr, model="gauss", x0_guess=None) -> Tuple[np.ndarray, np.ndarray, float]:
    """Weighted fit; returns (popt, pcov, chi2_red)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    yerr = np.asarray(yerr, float)
    yerr = np.where(yerr <= 0, np.median(yerr[yerr>0]) if np.any(yerr>0) else 1.0, yerr)

    if model == "gauss":
        fn = _gauss
        # crude guesses
        y0 = np.median(y)
        A = np.min(y) - y0 if np.abs(np.min(y) - y0) > np.abs(np.max(y) - y0) else np.max(y) - y0
        if x0_guess is None:
            x0_guess = x[np.argmax(np.abs(y - y0))]
        sigma = (np.max(x) - np.min(x)) / 10.0
        p0 = [A, sigma, x0_guess, y0]
        bounds = ([-np.inf, 0.0, np.min(x), -np.inf], [np.inf, (np.max(x)-np.min(x)), np.max(x), np.inf])
    else:
        fn = _lorentz
        y0 = np.median(y)
        A = (np.min(y) - y0) if np.abs(np.min(y) - y0) > np.abs(np.max(y) - y0) else (np.max(y) - y0)
        if x0_guess is None:
            x0_guess = x[np.argmax(np.abs(y - y0))]
        gamma = (np.max(x) - np.min(x)) / 20.0
        p0 = [A, gamma, x0_guess, y0]
        bounds = ([-np.inf, 0.0, np.min(x), -np.inf], [np.inf, (np.max(x)-np.min(x)), np.max(x), np.inf])

    popt, pcov = curve_fit(fn, x, y, p0=p0, sigma=yerr, absolute_sigma=True, bounds=bounds, maxfev=10000)
    yfit = fn(x, *popt)
    dof = max(1, len(x) - len(popt))
    chi2_red = np.sum(((y - yfit) / yerr) ** 2) / dof
    return popt, pcov, chi2_red

def split_on_off_interleaved(freqs_interleaved: np.ndarray, counts: np.ndarray):
    """
    Interleaved scheme: [on0, off0, on1, off1, ...]
    counts shape expected: (num_exps, num_nvs, num_runs, num_steps, num_reps)
    Returns:
        freqs_on (Nf), freqs_off (Nf),
        counts_on, counts_off with shape (num_nvs, num_runs, Nf, num_reps)
    """
    freqs_interleaved = np.asarray(freqs_interleaved, float)
    assert freqs_interleaved.ndim == 1 and freqs_interleaved.size % 2 == 0, "Interleaved freqs must be 1D and even length"
    # indices
    on_idx  = np.arange(0, freqs_interleaved.size, 2)
    off_idx = np.arange(1, freqs_interleaved.size, 2)
    freqs_on  = freqs_interleaved[on_idx]
    freqs_off = freqs_interleaved[off_idx]

    # collapse exp dimension (assume exp_ind=0), then gather steps
    # counts: (E, NV, R, S, rep) → use E=0
    E0 = counts[0] if counts.ndim == 5 else counts  # tolerate (NV,R,S,rep)
    # E0 shape now (NV, R, S, rep)
    counts_on  = E0[:, :, on_idx,  :]
    counts_off = E0[:, :, off_idx, :]
    return freqs_on, freqs_off, counts_on, counts_off

def deer_contrast(counts_on, counts_off, mode="frac_off"):
    """
    Compute contrast per NV, run, freq with STE across reps.
    counts_on/off shape: (NV, run, Nf, rep)
    Returns:
        mean_contrast (NV, Nf), ste_contrast (NV, Nf)
    """
    # average across reps first
    on_mean,  on_ste  = _mean_ste(counts_on,  axis=-1)   # (NV, run, Nf)
    off_mean, off_ste = _mean_ste(counts_off, axis=-1)

    if mode == "frac_off":
        # C = (ON - OFF)/OFF
        with np.errstate(divide='ignore', invalid='ignore'):
            C = (on_mean - off_mean) / off_mean
            # error propagation: var(C) ≈ (σ_on^2 + (ON/OFF)^2 σ_off^2)/OFF^2
            term_on  = (on_ste / off_mean)**2
            term_off = ((on_mean / (off_mean**2)) * off_ste)**2
            C_ste = np.sqrt(term_on + term_off)
            # handle zeros
            C = np.where(np.isfinite(C), C, 0.0)
            C_ste = np.where(np.isfinite(C_ste), C_ste, np.nan)
    elif mode == "diff":
        C = on_mean - off_mean
        C_ste = np.sqrt(on_ste**2 + off_ste**2)
    else:
        raise ValueError("mode must be 'frac_off' or 'diff'")

    # average across runs
    C_mean, C_ste_runs = _mean_ste(C, axis=1)
    # combine STE across runs and reps (conservative): sqrt(ste_runs^2 + mean(ste)^2)
    C_ste_mean = np.sqrt(C_ste_runs**2 + np.nanmean(C_ste, axis=1)**2)
    return C_mean, C_ste_mean  # (NV, Nf)
def postprocess_deer(raw_data: Dict[str, Any],
                     freqs_interleaved: np.ndarray,
                     fit_model: str = "gauss",
                     do_fit: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list, Optional[plt.Figure]]:
    """
    Args:
        raw_data: dict returned by base_routine.main (must contain 'counts')
        freqs_interleaved: 1D array used in acquisition (ON/OFF interleaved)
        fit_model: 'gauss' or 'lorentz'
        do_fit: if True, fit each NV's contrast spectrum

    Returns:
        freqs_on: (Nf,)
        C_mean: (NV, Nf) mean contrast
        C_ste:  (NV, Nf) STE of contrast
        fit_results: list[DeerResult] (possibly empty if do_fit=False)
        fig: overview matplotlib Figure (or None)
    """
    counts = np.asarray(raw_data["counts"])
    # Split ON/OFF along step dimension
    freqs_on, freqs_off, counts_on, counts_off = split_on_off_interleaved(freqs_interleaved, counts)

    # Sanity: ON/OFF freq sets should be identical up to constant delta
    if not np.allclose(freqs_off - freqs_on, freqs_off[0] - freqs_on[0], atol=1e-9):
        print("[warn] OFF detuning may be non-constant")

    # Build contrast vs the OFF reference
    C_mean, C_ste = deer_contrast(counts_on, counts_off, mode="frac_off")  # shapes (NV, Nf)
    NV, Nf = C_mean.shape

    # Optional: fit each NV's curve
    fit_results = []
    if do_fit:
        for nv_i in range(NV):
            y   = C_mean[nv_i]
            ye  = np.where(C_ste[nv_i] <= 0, np.nanmedian(C_ste[nv_i][C_ste[nv_i]>0]) if np.any(C_ste[nv_i]>0) else 1.0, C_ste[nv_i])
            # use the strongest excursion as initial x0
            x0_guess = freqs_on[np.nanargmax(np.abs(y))]
            popt, pcov, chi2 = _fit_1d(freqs_on, y, ye, model=fit_model, x0_guess=x0_guess)
            # unpack
            if fit_model == "gauss":
                A, sigma, x0, y0 = popt
                width = sigma
            else:
                A, gamma, x0, y0 = popt
                width = gamma
            # data peak
            idx = np.nanargmax(np.abs(y))
            fit_results.append(DeerResult(
                nv_index=nv_i, f0=float(x0), amp=float(A), width=float(width),
                chi2_red=float(chi2), peak_contrast=float(y[idx]), peak_freq=float(freqs_on[idx])
            ))

    # Quick overview figure
    fig = plt.figure(figsize=(7.5, 4.5))
    ax = fig.add_subplot(111)
    # plot a few NVs to avoid clutter
    show = min(12, NV)
    for i in range(show):
        ax.errorbar(freqs_on, C_mean[i], C_ste[i], marker='o', lw=1, ms=3, alpha=0.8)
    ax.set_xlabel("RF frequency (GHz)")
    ax.set_ylabel("DEER contrast  (ON−OFF)/OFF")
    ax.axhline(0, ls="--", alpha=0.4)
    ax.set_title(f"DEER spectra (first {show} NVs of {NV})")
    return freqs_on, C_mean, C_ste, fit_results, fig


def main(
    nv_list: list[NVSig],
    num_steps,
    num_reps,
    num_runs,
    freqs,
    uwave_ind_list=[0, 1],
):
    ### Some initial setup
    pulse_gen = tb.get_server_pulse_gen()
    original_num_steps = num_steps
    num_steps *= 4  # For sig, ms=0 ref, and ms=+/-1 ref

    seq_file = "deer_hahn.py"

    ### Collect the data
    # Assume freqs is a 1D array (GHz) for RF sweep (e.g., 0.120–0.150 GHz)
    delta = 0.060  # 60 MHz detuning for OFF
    freqs_on = np.asarray(freqs, float)
    freqs_off = freqs_on + delta

    # Interleave [on0, off0, on1, off1, ...]
    freqs_interleaved = np.empty(2 * len(freqs_on), dtype=float)
    freqs_interleaved[0::2] = freqs_on
    freqs_interleaved[1::2] = freqs_off

    original_num_steps = len(freqs_interleaved)
    num_steps = original_num_steps  # no need for ×4 in DEER

    def run_fn(step_inds):
        seq_args = [widefield.get_base_scc_seq_args(nv_list, uwave_ind_list), step_inds]
        seq_args_string = tb.encode_seq_args(seq_args)
        pulse_gen.stream_load(seq_file, seq_args_string, num_reps)
        # print(seq_args)

    def step_fn(step_ind):
        # MW (NV) chain: fixed at NV transition; ON for pulses
        mw_ind = uwave_ind_list[0]
        mw_dict = tb.get_virtual_sig_gen_dict(mw_ind)
        mw = tb.get_server_sig_gen(mw_ind)
        mw.set_amp(mw_dict["uwave_power"])
        mw.set_freq(mw_dict["frequency"])  # NV MW freq (GHz)
        mw.uwave_on()

        # RF chain: set frequency per step (interleaved on/off)
        rf_ind = uwave_ind_list[1]
        rf = tb.get_server_sig_gen(rf_ind)
        rf_dict = tb.get_virtual_sig_gen_dict(rf_ind)
        rf.set_amp(rf_dict["uwave_power"])  # RF power (dBm) for π_RF
        rf.set_freq(freqs_interleaved[step_ind])  # GHz
        rf.uwave_on()  # leave RF CW; OPX gates it with RF_GATE TTL

    raw_data = base_routine.main(
        nv_list,
        num_steps,
        num_reps,
        num_runs,
        run_fn,
        step_fn,
        uwave_ind_list=uwave_ind_list,
        save_images=False,
        num_exps=1,
        ref_by_rep_parity=False,
        # load_iq=True,
    )

    ### Process and plot

    try:
        counts = raw_data["counts"]
        reformatted_counts = reformat_counts(counts)
        sig_counts = reformatted_counts[0]
        ref_counts = reformatted_counts[1]

        avg_counts, avg_counts_ste, norms = widefield.process_counts(
            nv_list, sig_counts, ref_counts, threshold=True
        )

        # raw_fig = create_raw_data_figure(nv_list, freqs, avg_counts, avg_counts_ste)
    except Exception:
        print(traceback.format_exc())
        raw_fig = None
        fit_fig = None

    ### Clean up and return

    tb.reset_cfm()
    kpl.show()

    timestamp = dm.get_time_stamp()
    raw_data |= {
        "timestamp": timestamp,
        "freqs": freqs,
        "freq-units": "GHz",
        # "freq_range": freq_range,
        # "freq_center": freq_center,
    }

    repr_nv_sig = widefield.get_repr_nv_sig(nv_list)
    repr_nv_name = repr_nv_sig.name
    file_path = dm.get_file_path(__file__, timestamp, repr_nv_name)
    if "img_arrays" in raw_data:
        keys_to_compress = ["img_arrays"]
    else:
        keys_to_compress = None
    dm.save_raw_data(raw_data, file_path, keys_to_compress)
    if raw_fig is not None:
        dm.save_figure(raw_fig, file_path)
    if fit_fig is not None:
        file_path = dm.get_file_path(__file__, timestamp, repr_nv_name + "-fit")
        dm.save_figure(fit_fig, file_path)


if __name__ == "__main__":
    kpl.init_kplotlib()

    # --- Load saved raw ---
    file_id = "2025_10_09-19_03_56-rubin-nv0_2025_09_08"
    data = dm.get_raw_data(file_stem=file_id, load_npz=True, use_cache=True)

    nv_list  = data["nv_list"]
    num_nvs  = len(nv_list)
    num_steps = data["num_steps"]
    num_runs  = data["num_runs"]
    num_reps  = data["num_reps"]
    freqs     = np.asarray(data["freqs"], float)  # ON frequencies you scanned
    counts    = np.asarray(data["counts"])

    # --- Build the same interleaved vector used during acquisition ---
    delta = 0.060 
    freqs_on  = freqs
    freqs_on  = np.asarray(freqs_on, float)
    freqs_off = freqs_on + float(delta)
    Nf = freqs_on.size

    on_idx  = np.arange(0, 2*Nf, 2)
    off_idx = np.arange(1, 2*Nf, 2)

    E0 = np.asarray(counts)[0]              # (NV, runs, steps=2*Nf, reps)
    sig_counts = E0[:, :, on_idx, :]        # (NV, runs, Nf, reps)
    ref_counts = E0[:, :, off_idx, :]       # (NV, runs, Nf, reps)
    sig_counts, ref_counts = widefield.threshold_counts(
            nv_list, sig_counts, ref_counts, dynamic_thresh=True
        )
    ### Report the results

    avg_sig_counts, avg_sig_counts_ste, _ = widefield.average_counts(sig_counts)
    avg_ref_counts, avg_ref_counts_ste, _ = widefield.average_counts(ref_counts)

    avg_snr, avg_snr_ste = widefield.calc_snr(sig_counts, ref_counts)
    avg_contrast, avg_contrast_ste = widefield.calc_contrast(sig_counts, ref_counts)

    
    # Loop through NVs one by one
    for nv_i in range(num_nvs):
        fig, ax = plt.subplots()
        ax.errorbar(freqs_on,
                    avg_snr[nv_i],
                    yerr=avg_snr_ste[nv_i],
                    marker='o', ms=4, lw=1, color="C0")

        ax.set_title(f"NV {nv_i} DEER Contrast")
        ax.set_xlabel("RF frequency (GHz)")
        ax.set_ylabel("Contrast (ON−OFF)/OFF")

        # plt.show(block=True)   # wait until you close the figure before next one
    sys.exit()
    # --- Process & fit (DEER: ON/OFF interleaved) ---
    try:
        freqs_on_out, C_mean, C_ste, fit_results, deer_fig = postprocess_deer(
            {"counts": counts}, freqs_interleaved, fit_model="gauss", do_fit=True
        )
    except Exception:
        print(traceback.format_exc())
        freqs_on_out, C_mean, C_ste, fit_results, deer_fig = None, None, None, [], None

    kpl.show(block=True)