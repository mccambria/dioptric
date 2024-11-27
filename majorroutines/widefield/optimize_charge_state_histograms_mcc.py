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
import traceback

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from scipy.optimize import curve_fit
from scipy.special import factorial

from analysis import bimodal_histogram
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
from utils.constants import NVSig, VirtualLaserKey

# region Process and plotting functions


# def process_and_plot(raw_data):
#     nv_list = raw_data["nv_list"]
#     num_nvs = len(nv_list)
#     min_step_val = raw_data["min_step_val"]
#     max_step_val = raw_data["max_step_val"]
#     num_steps = raw_data["num_steps"]
#     step_vals = np.linspace(min_step_val, max_step_val, num_steps)
#     optimize_pol_or_readout = raw_data["optimize_pol_or_readout"]
#     optimize_duration_or_amp = raw_data["optimize_duration_or_amp"]

#     counts = np.array(raw_data["counts"])
#     # [nv_ind, run_ind, steq_ind, rep_ind]
#     ref_exp_ind = 1
#     condensed_counts = [
#         [
#             counts[ref_exp_ind, nv_ind, :, step_ind, :].flatten()
#             for step_ind in range(num_steps)
#         ]
#         for nv_ind in range(num_nvs)
#     ]
#     condensed_counts = np.array(condensed_counts)

#     prob_dist = ProbDist.COMPOUND_POISSON

#     readout_fidelity_arr = np.empty((num_nvs, num_steps))
#     prep_fidelity_arr = np.empty((num_nvs, num_steps))
#     for nv_ind in range(num_nvs):
#         for step_ind in range(num_steps):
#             popt = fit_bimodal_histogram(condensed_counts[nv_ind, step_ind], prob_dist)
#             if popt is None:
#                 readout_fidelity = np.nan
#                 prep_fidelity = np.nan
#             else:
#                 threshold, readout_fidelity = determine_threshold(
#                     popt, prob_dist, dark_mode_weight=0.5, ret_fidelity=True
#                 )
#                 prep_fidelity = 1 - popt[0]
#             readout_fidelity_arr[nv_ind, step_ind] = readout_fidelity
#             prep_fidelity_arr[nv_ind, step_ind] = prep_fidelity

#     ### Plotting

#     if optimize_pol_or_readout:
#         if optimize_duration_or_amp:
#             x_label = "Polarization duration"
#         else:
#             x_label = "Polarization amplitude"
#     else:
#         if optimize_duration_or_amp:
#             x_label = "Readout duration"
#         else:
#             x_label = "Readout amplitude"

#     print("Plotting results for first 10 NVs")
#     for nv_ind in range(10):
#         fig, ax = plt.subplots()
#         kpl.plot_points(ax, step_vals, readout_fidelity_arr[nv_ind, :])
#         ax.set_xlabel(x_label)
#         ax.set_ylabel("Readout fidelity")
#         ax.set_title(f"NV{nv_ind}")
#         fig, ax = plt.subplots()
#         kpl.plot_points(ax, step_vals, prep_fidelity_arr[nv_ind, :])
#         ax.set_xlabel(x_label)
#         ax.set_ylabel("Charge polarization fidelity")
#         ax.set_title(f"NV{nv_ind}")


def process_and_plot(raw_data):
    nv_list = raw_data["nv_list"]
    num_nvs = len(nv_list)
    min_step_val = raw_data["min_step_val"]
    max_step_val = raw_data["max_step_val"]
    num_steps = raw_data["num_steps"]
    step_vals = np.linspace(min_step_val, max_step_val, num_steps)
    optimize_pol_or_readout = raw_data["optimize_pol_or_readout"]
    optimize_duration_or_amp = raw_data["optimize_duration_or_amp"]

    counts = np.array(raw_data["counts"])
    # [nv_ind, run_ind, steq_ind, rep_ind]
    ref_exp_ind = 1
    condensed_counts = [
        [
            counts[ref_exp_ind, nv_ind, :, step_ind, :].flatten()
            for step_ind in range(num_steps)
        ]
        for nv_ind in range(num_nvs)
    ]
    condensed_counts = np.array(condensed_counts)

    prob_dist = ProbDist.COMPOUND_POISSON

    readout_fidelity_arr = np.empty((num_nvs, num_steps))
    prep_fidelity_arr = np.empty((num_nvs, num_steps))
    separation_metric_arr = np.empty((num_nvs, num_steps))
    goodness_of_fit_arr = np.empty((num_nvs, num_steps))

    for nv_ind in range(num_nvs):
        for step_ind in range(num_steps):
            # Fit the histogram
            popt, r_squared = fit_bimodal_histogram(
                condensed_counts[nv_ind, step_ind], prob_dist
            )
            if popt is None:
                readout_fidelity = np.nan
                prep_fidelity = np.nan
                inter_class_variance = np.nan
                separation_metric = np.nan
            else:
                # Threshold and readout fidelity
                threshold, readout_fidelity = determine_threshold(
                    popt, prob_dist, dark_mode_weight=0.5, ret_fidelity=True
                )
                prep_fidelity = 1 - popt[0]

                # Calculate inter-class and intra-class variance
                w_dark = popt[0]
                mu_dark = popt[1]
                mu_bright = popt[2]
                sigma_dark = np.sqrt(mu_dark)
                sigma_bright = np.sqrt(mu_bright)

                inter_class_variance = (
                    w_dark * (1 - w_dark) * (mu_bright - mu_dark) ** 2
                )

                intra_class_variance = (
                    w_dark * sigma_dark**2 + (1 - w_dark) * sigma_bright**2
                )

                # Calculate separation metric
                if intra_class_variance > 0:
                    separation_metric = inter_class_variance / intra_class_variance
                else:
                    separation_metric = np.nan

            readout_fidelity_arr[nv_ind, step_ind] = readout_fidelity
            prep_fidelity_arr[nv_ind, step_ind] = prep_fidelity
            separation_metric_arr[nv_ind, step_ind] = separation_metric
            goodness_of_fit_arr[nv_ind, step_ind] = r_squared

    ### Plotting
    if optimize_pol_or_readout:
        if optimize_duration_or_amp:
            x_label = "Polarization duration"
        else:
            x_label = "Polarization amplitude"
    else:
        if optimize_duration_or_amp:
            x_label = "Readout duration"
        else:
            x_label = "Readout amplitude"

    # Determine step values corresponding to maximum separation and goodness of fit for each NV
    print("Plotting results and calculating optima for all NVs")
    optimal_amplitudes = []  # To store optimal separation and step values
    optimal_goodness = []  # To store maximum goodness of fit and step values

    for nv_ind in range(num_nvs):
        try:
            # Find the maximum separation and the corresponding step value
            max_separation = np.nanmax(separation_metric_arr[nv_ind, :])
            optimal_step_val_sep = step_vals[
                np.nanargmax(separation_metric_arr[nv_ind, :])
            ]

            # Find the maximum goodness of fit (R²) and its corresponding step value
            max_goodness = np.nanmax(goodness_of_fit_arr[nv_ind, :])
            optimal_step_val_goodness = step_vals[
                np.nanargmax(goodness_of_fit_arr[nv_ind, :])
            ]

            optimal_amplitudes.append((nv_ind, max_separation, optimal_step_val_sep))
            optimal_goodness.append((nv_ind, max_goodness, optimal_step_val_goodness))
        except Exception as e:
            print(f"Failed to process NV{nv_ind}: {e}")
            optimal_amplitudes.append((nv_ind, np.nan, np.nan))
            optimal_goodness.append((nv_ind, np.nan, np.nan))
            continue

        # Plotting
        fig, ax1 = plt.subplots(figsize=(6, 5))  # Adjust figure size

        # Primary y-axis for readout fidelity and prep fidelity
        ax1.scatter(
            step_vals,
            readout_fidelity_arr[nv_ind, :],
            label="Readout Fidelity",
        )
        ax1.scatter(
            step_vals,
            prep_fidelity_arr[nv_ind, :],
            label="Charge Polarization Fidelity",
        )
        ax1.set_xlabel(x_label)
        ax1.set_ylabel("Fidelity")
        ax1.tick_params(axis="y")
        ax1.legend(loc="upper left", fontsize=9)

        # Secondary y-axis for separation metric
        ax2 = ax1.twinx()
        ax2.plot(
            step_vals,
            separation_metric_arr[nv_ind, :],
            color="purple",
            label="Separation Metric",
        )
        ax2.scatter(
            optimal_step_val_sep,
            max_separation,
            color="red",
            label=f"Max Separation: {max_separation:.2f}",
            zorder=5,
        )
        ax2.set_ylabel("Separation", color="purple")
        ax2.tick_params(axis="y", labelcolor="purple")

        # Overlay goodness of fit as a scatter plot on a new axis
        ax3 = ax1.twinx()
        ax3.spines["right"].set_position(("outward", 60))  # Offset the third axis
        ax3.scatter(
            step_vals,
            goodness_of_fit_arr[nv_ind, :],
            color="green",
            label="Goodness of Fit (R²)",
            alpha=0.7,
        )
        ax3.scatter(
            optimal_step_val_goodness,
            max_goodness,
            color="blue",
            s=100,
            label=f"Max R²: {max_goodness:.2f}",
            zorder=5,
        )
        ax3.set_ylabel("Goodness of Fit (R²)", color="green")
        ax3.tick_params(axis="y", labelcolor="green")
        ax3.legend(loc="upper right", fontsize=9)

        # Title and layout
        ax1.set_title(
            f"NV{nv_ind} - Max Sep: {max_separation:.2f}, Step Val: {optimal_step_val_sep:.2f} | Max R²: {max_goodness:.2f}, Step Val: {optimal_step_val_goodness:.2f}"
        )
        fig.tight_layout()
        plt.show()

    # Save results to a file
    with open("optimal_separation_and_goodness.txt", "w") as f:
        f.write(
            "NV Index, Max Separation, Step Val (Separation), Max R², Step Val (R²)\n"
        )
        for (nv_index, max_sep, opt_step_sep), (_, max_good, opt_step_good) in zip(
            optimal_amplitudes, optimal_goodness
        ):
            f.write(
                f"{nv_index}, {max_sep:.6f}, {opt_step_sep:.6f}, {max_good:.6f}, {opt_step_good:.6f}\n"
            )


# endregion


def optimize_pol_duration(
    nv_list, num_steps, num_reps, num_runs, min_duration, max_duration
):
    return _main(
        nv_list, num_steps, num_reps, num_runs, min_duration, max_duration, True, True
    )


def optimize_pol_amp(nv_list, num_steps, num_reps, num_runs, min_amp, max_amp):
    return _main(nv_list, num_steps, num_reps, num_runs, min_amp, max_amp, True, False)


def optimize_readout_duration(
    nv_list, num_steps, num_reps, num_runs, min_duration, max_duration
):
    return _main(
        nv_list, num_steps, num_reps, num_runs, min_duration, max_duration, False, True
    )


def optimize_readout_amp(nv_list, num_steps, num_reps, num_runs, min_amp, max_amp):
    return _main(nv_list, num_steps, num_reps, num_runs, min_amp, max_amp, False, False)


def _main(
    nv_list,
    num_steps,
    num_reps,
    num_runs,
    min_step_val,
    max_step_val,
    optimize_pol_or_readout,
    optimize_duration_or_amp,
):
    ### Initial setup
    seq_file = "optimize_charge_state_histograms.py"
    step_vals = np.linspace(min_step_val, max_step_val, num_steps)

    pulse_gen = tb.get_server_pulse_gen()

    ### Collect the data

    def run_fn(shuffled_step_inds):
        shuffled_step_vals = step_vals[shuffled_step_inds].tolist()
        pol_coords_list, pol_duration_list, pol_amp_list = (
            widefield.get_pulse_parameter_lists(nv_list, VirtualLaserKey.CHARGE_POL)
        )
        ion_coords_list = widefield.get_coords_list(nv_list, VirtualLaserKey.ION)
        seq_args = [
            pol_coords_list,
            pol_duration_list,
            pol_amp_list,
            ion_coords_list,
            shuffled_step_vals,
            optimize_pol_or_readout,
            optimize_duration_or_amp,
        ]
        # print(seq_args)
        seq_args_string = tb.encode_seq_args(seq_args)
        pulse_gen.stream_load(seq_file, seq_args_string, num_reps)

    raw_data = base_routine.main(nv_list, num_steps, num_reps, num_runs, run_fn=run_fn)

    ### Processing

    timestamp = dm.get_time_stamp()
    repr_nv_sig = widefield.get_repr_nv_sig(nv_list)
    repr_nv_name = repr_nv_sig.name

    raw_data |= {
        "timestamp": timestamp,
        "min_step_val": min_step_val,
        "max_step_val": max_step_val,
        "optimize_pol_or_readout": optimize_pol_or_readout,
        "optimize_duration_or_amp": optimize_duration_or_amp,
    }

    try:
        process_and_plot(raw_data)

    except Exception:
        print(traceback.format_exc())

    try:
        del raw_data["img_arrays"]
    except Exception:
        pass

    ### Save and clean up

    file_path = dm.get_file_path(__file__, timestamp, repr_nv_name)
    dm.save_raw_data(raw_data, file_path)

    tb.reset_cfm()

    return raw_data


if __name__ == "__main__":
    kpl.init_kplotlib()
    # raw_data = dm.get_raw_data(file_id=1705172140093, load_npz=False)
    raw_data = dm.get_raw_data(file_id=1709868774004, load_npz=False)
    # data = dm.get_raw_data(file_id=1691569540529, load_npz=False)
    process_and_plot(raw_data)
    kpl.show(block=True)
