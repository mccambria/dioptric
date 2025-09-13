# -*- coding: utf-8 -*-
"""
Pulsed electron spin resonance on multiple NVs with spin-to-charge
conversion readout imaged onto a camera

Created on November 19th, 2023

@author: mccambria
updated by @schand on February 6th, 2025
@author: sbchand
"""

import os
import sys
import time
from random import shuffle

import matplotlib.pyplot as plt
import numpy as np

from majorroutines.pulsed_resonance import fit_resonance, voigt, voigt_split
from majorroutines.widefield import base_routine
from utils import common
from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import positioning as pos
from utils import tool_belt as tb
from utils import widefield as widefield
from utils.constants import NVSig, NVSpinState
from utils.positioning import get_scan_1d as calculate_powers


def get_lower_left_ax(axes_pack):
    """Helper function to find the lower-left axis from axes_pack."""
    if isinstance(axes_pack, dict):
        # Assuming the axes_pack dictionary has keys indicating positions (like a mosaic)
        # Let's extract the keys and find the one in the lower left
        lower_left_key = min(
            axes_pack.keys()
        )  # Assuming keys represent positions and lower-left is smallest
        return axes_pack[lower_left_key]
    else:
        # If it's a list or something else, return the last axis
        return axes_pack[-1]


def create_raw_data_figure(data):
    nv_list = data["nv_list"]
    num_nvs = len(nv_list)
    powers = data["powers"]
    counts = np.array(data["counts"])
    sig_counts, ref_counts = counts[0], counts[1]

    norm_counts, norm_counts_ste = widefield.process_counts(
        nv_list, sig_counts, ref_counts, threshold=True
    )
    # norm_counts = avg_counts - norms[0][:, np.newaxis]
    norm_counts_ste = abs(norm_counts_ste)

    fig, ax_raw = plt.subplots(figsize=(6, 4))

    for nv_idx, nv in enumerate(nv_list):
        ax_raw.errorbar(
            powers,
            norm_counts[nv_idx],
            yerr=norm_counts_ste[nv_idx],
            fmt="o",
            label=f"NV {nv_idx + 1}",
        )

    ax_raw.set_xlabel("Microwave power (dBm)")
    ax_raw.set_ylabel("Normalized NV- population")
    ax_raw.set_title("Raw ESR Data")
    ax_raw.legend()
    ax_raw.grid(True)
    plt.show()
    return fig

    # return fig


import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


def lorentzian(x, a, x0, gamma, c):
    return a * gamma**2 / ((x - x0) ** 2 + gamma**2) + c


def create_raw_data_figure_with_fit(data):
    nv_list = data["nv_list"]
    num_nvs = len(nv_list)
    powers = np.array(data["powers"])
    counts = np.array(data["counts"])
    sig_counts, ref_counts = counts[0], counts[1]

    # Normalize
    norm_counts, norm_counts_ste = widefield.process_counts(
        nv_list, sig_counts, ref_counts, threshold=True
    )
    norm_counts_ste = np.abs(norm_counts_ste)

    fig, (ax_raw, ax_median) = plt.subplots(1, 2, figsize=(12, 4))

    # Plot individual NVs
    for nv_idx, nv in enumerate(nv_list):
        ax_raw.errorbar(
            powers,
            norm_counts[nv_idx],
            yerr=norm_counts_ste[nv_idx],
            fmt="o",
            alpha=0.4,
            label=f"NV {nv_idx + 1}",
        )

    ax_raw.set_xlabel("Microwave power (dBm)")
    ax_raw.set_ylabel("Normalized NV⁻ population")
    ax_raw.set_title("Raw ESR Data")
    ax_raw.legend(fontsize="x-small")
    ax_raw.grid(True)

    # Compute median and std error
    median_vals = np.median(norm_counts, axis=0)
    std_error = np.std(norm_counts, axis=0) / np.sqrt(num_nvs)

    # Fit the median to Lorentzian
    try:
        popt, _ = curve_fit(
            lorentzian,
            powers,
            median_vals,
            p0=[-1, powers[np.argmin(median_vals)], 5, 1],
        )
        fitted_curve = lorentzian(powers, *popt)
        optimal_power = popt[1]
    except RuntimeError:
        fitted_curve = None
        optimal_power = None

    # Plot median and fit
    ax_median.errorbar(
        powers,
        median_vals,
        yerr=std_error,
        fmt="o",
        color="black",
        label="Median across NVs",
    )
    if fitted_curve is not None:
        ax_median.plot(
            powers,
            fitted_curve,
            "--",
            color="red",
            label=f"Fit (min at {optimal_power:.2f} dBm)",
        )
        ax_median.axvline(optimal_power, color="red", linestyle=":", alpha=0.6)

    ax_median.set_xlabel("Microwave power (dBm)")
    ax_median.set_ylabel("Median NV⁻ population")
    ax_median.set_title("Median ESR Response & Fit")
    ax_median.legend()
    ax_median.grid(True)

    plt.tight_layout()
    plt.show()

    return fig, optimal_power


def create_median_snr_vs_power_figure(data):
    powers = np.array(data["powers"])
    sig_counts = np.array(data["counts"][0])
    ref_counts = np.array(data["counts"][1])
    nv_list = data["nv_list"]
    num_nvs = len(nv_list)
    avg_snr, avg_snr_ste = widefield.calc_snr(sig_counts, ref_counts)
    # Compute median and standard error of the median
    median_snr = np.median(avg_snr, axis=0)
    std_err_median = np.median(avg_snr_ste, axis=0)

    # Fit to inverted Lorentzian
    # Fit the median to Lorentzian
    try:
        popt, _ = curve_fit(
            lorentzian,
            powers,
            median_snr,
            p0=[-1, powers[np.argmax(median_snr)], 5, 1],
        )
        fit_curve = lorentzian(powers, *popt)
        optimal_power = popt[1]
    except RuntimeError:
        fit_curve = None
        optimal_power = None

    # Plot
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(
        powers,
        median_snr,
        yerr=std_err_median,
        fmt="o",
        color="black",
        label="Median SNR",
    )

    if fit_curve is not None:
        ax.plot(
            powers,
            fit_curve,
            "--",
            color="red",
            label=f"Fit (max at {optimal_power:.2f} dBm)",
        )
        ax.axvline(optimal_power, color="red", linestyle=":", alpha=0.6)

    ax.set_xlabel("Microwave power (dBm)")
    ax.set_ylabel("Median SNR across NVs")
    ax.set_title("Median SNR vs Microwave Power")
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.show()

    return fig, optimal_power


def main(
    nv_list: list[NVSig],
    num_steps,
    num_reps,
    num_runs,
    powers,
    uwave_ind_list=[0, 1],
):
    ### Some initial setup

    pulse_gen = tb.get_server_pulse_gen()
    seq_file = "power_rabi.py"

    ### Collect the data
    def run_fn(step_inds):
        seq_args = [widefield.get_base_scc_seq_args(nv_list, uwave_ind_list), step_inds]
        # print(seq_args)
        seq_args_string = tb.encode_seq_args(seq_args)
        pulse_gen.stream_load(seq_file, seq_args_string, num_reps)

    def step_fn(step_ind):
        power = powers[step_ind]
        for uwave_ind in uwave_ind_list:
            uwave_dict = tb.get_virtual_sig_gen_dict(uwave_ind)
            sig_gen = tb.get_server_sig_gen(uwave_ind)
            uwave_power = uwave_dict["uwave_power"]
            sig_gen = tb.get_server_sig_gen(uwave_ind)
            sig_gen.set_amp(round(uwave_power + power, 3))

    data = base_routine.main(
        nv_list,
        num_steps,
        num_reps,
        num_runs,
        run_fn,
        step_fn,
        uwave_ind_list=uwave_ind_list,
        load_iq=True,
    )
    # uwave_power = uwave_dict["uwave_power"]
    # uwave_powers = uwave_power + power
    ### save the data
    timestamp = dm.get_time_stamp()
    data |= {
        "timestamp": timestamp,
        "power-units": "GHz",
        "powers": powers,
    }

    repr_nv_sig = widefield.get_repr_nv_sig(nv_list)
    repr_nv_name = repr_nv_sig.name
    file_path = dm.get_file_path(__file__, timestamp, repr_nv_name)
    dm.save_raw_data(data, file_path)
    ### Process and plot

    data["powers"] = powers
    try:
        raw_fig = create_raw_data_figure(data)
    except Exception as exc:
        print(exc)
        raw_fig = None

    ### Clean up and return
    tb.reset_cfm()
    kpl.show()

    if raw_fig is not None:
        dm.save_figure(raw_fig, file_path)


if __name__ == "__main__":
    kpl.init_kplotlib()
    data = dm.get_raw_data(file_id=1832854285676)
    # raw_fig = create_raw_data_figure(data)
    create_median_snr_vs_power_figure(data)
    kpl.show(block=True)
