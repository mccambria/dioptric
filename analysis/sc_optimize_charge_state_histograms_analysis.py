# -*- coding: utf-8 -*-
"""
Illuminate an area, collecting onto the camera. Interleave a signal and control sequence
and plot the difference
Created on Fall 2024
@author: saroj chand
"""

import os
import sys
import time
import traceback
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from matplotlib import font_manager as fm
from matplotlib import rcParams

# Specify the path to the Arial font file
arial_font_path = r"C:\Windows\Fonts\arial.ttf"
arial_font = fm.FontProperties(fname=arial_font_path)
rcParams["font.family"] = arial_font.get_name()

from analysis.bimodal_histogram import (
    ProbDist,
    determine_threshold,
    fit_bimodal_histogram,
)
from utils import common, widefield
from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import positioning as pos
from utils import tool_belt as tb
from utils.constants import NVSig, VirtualLaserKey


def find_optimal_value_geom_mean(
    step_vals, prep_fidelity, readout_fidelity, goodness_of_fit, weights=(1, 1, 1)
):
    """
    Finds the optimal step value using a weighted geometric mean of fidelities and goodness of fit.

    """
    w1, w2, w3 = weights

    # Normalize metrics
    norm_prep_fidelity = (prep_fidelity - np.nanmin(prep_fidelity)) / (
        np.nanmax(prep_fidelity) - np.nanmin(prep_fidelity)
    )
    norm_readout_fidelity = (readout_fidelity - np.nanmin(readout_fidelity)) / (
        np.nanmax(readout_fidelity) - np.nanmin(readout_fidelity)
    )
    norm_goodness = (goodness_of_fit - np.nanmin(goodness_of_fit)) / (
        np.nanmax(goodness_of_fit) - np.nanmin(goodness_of_fit)
    )
    inverted_goodness = 1 - norm_goodness  # Minimize goodness of fit

    # Compute weighted geometric mean
    combined_score = (
        (norm_readout_fidelity**w1) * (norm_prep_fidelity**w2) * (inverted_goodness**w3)
    ) ** (1 / (w1 + w2 + w3))

    # Find the step value corresponding to the maximum combined score
    max_index = np.nanargmax(combined_score)
    max_combined_score = combined_score[max_index]
    optimal_step_val = step_vals[max_index]
    optimal_prep_fidality = prep_fidelity[max_index]
    optimal_readout_fidality = readout_fidelity[max_index]
    return (
        optimal_step_val,
        optimal_prep_fidality,
        optimal_readout_fidality,
        max_combined_score,
    )


def process_and_plot(raw_data):
    nv_list = raw_data["nv_list"]
    num_nvs = len(nv_list)
    min_step_val = raw_data["min_step_val"]
    max_step_val = raw_data["max_step_val"]
    num_steps = raw_data["num_steps"]
    step_vals = np.linspace(min_step_val, max_step_val, num_steps)
    optimize_pol_or_readout = raw_data["optimize_pol_or_readout"]
    optimize_duration_or_amp = raw_data["optimize_duration_or_amp"]
    # get yellow amplitude
    yellow_charge_readout_amp = raw_data["opx_config"]["waveforms"][
        "yellow_charge_readout"
    ]["sample"]
    green_aod_cw_charge_pol_amp = raw_data["opx_config"]["waveforms"][
        "green_aod_cw-charge_pol"
    ]["sample"]

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

    # Function to process a single NV and step
    def process_nv_step(nv_ind, step_ind):
        counts_data = condensed_counts[nv_ind, step_ind]
        popt, pcov, chi_squared = fit_bimodal_histogram(counts_data, prob_dist)

        if popt is None:
            return np.nan, np.nan, np.nan, np.nan

        # Threshold, prep and readout fidelity
        threshold, readout_fidelity = determine_threshold(
            popt, prob_dist, dark_mode_weight=0.5, ret_fidelity=True
        )
        prep_fidelity = 1 - popt[0]  # Population weight of dark state

        return readout_fidelity, prep_fidelity, chi_squared

    # Parallel processingv -->  n_jobs :  Defaults to using all available cores (-1).
    results = Parallel(n_jobs=-1)(
        delayed(process_nv_step)(nv_ind, step_ind)
        for nv_ind in range(num_nvs)
        for step_ind in range(num_steps)
    )

    # Reshape results into arrays
    results = np.array(results).reshape(num_nvs, num_steps, 4)
    readout_fidelity_arr = results[:, :, 0]
    prep_fidelity_arr = results[:, :, 1]
    goodness_of_fit_arr = results[:, :, 3]

    # Save results
    timestamp = dm.get_time_stamp()
    file_name = f"optimal_steps_{file_id}"
    file_path = dm.get_file_path(__file__, timestamp, file_name)

    # Prepare data to save
    processed_data = {
        "timestamp": timestamp,
        "nv_data": [
            {
                "nv_index": nv_index,
                "step_values": step_vals.tolist(),
                "readout_fidelity": readout_fidelity_arr[nv_index].tolist(),
                "prep_fidelity": prep_fidelity_arr[nv_index].tolist(),
                "goodness_of_fit": goodness_of_fit_arr[nv_index].tolist(),
            }
            for nv_index in range(num_nvs)
        ],
        "metadata": {
            "num_nvs": num_nvs,
            "num_steps": num_steps,
            "min_step_val": min_step_val,
            "max_step_val": max_step_val,
            "optimize_pol_or_readout": optimize_pol_or_readout,
            "optimize_duration_or_amp": optimize_duration_or_amp,
            "yellow_charge_readout_amp": yellow_charge_readout_amp,
            "green_aod_cw_charge_pol_amp": green_aod_cw_charge_pol_amp,
            "file_id": file_id,
            "file_name": file_name,
        },
    }

    dm.save_raw_data(processed_data, file_path)
    print(f"Processed data saved to '{file_path}'.")

    ### Plotting
    if optimize_pol_or_readout:
        if optimize_duration_or_amp:
            step_vals *= 1e-3
            x_label = "Polarization duration (us)"
        else:
            step_vals *= green_aod_cw_charge_pol_amp
            x_label = "Polarization amplitude"
    else:
        if optimize_duration_or_amp:
            step_vals *= 1e-6
            x_label = "Readout duration (ms)"
        else:
            step_vals *= yellow_charge_readout_amp
            x_label = "Readout amplitude"
            # step_vals = a * (step_vals**b) + c
            # x_label = "Readout amplitude (uW)"
    # print(step_vals)
    # Optimal values
    optimal_values = []
    for nv_ind in range(num_nvs):
        try:
            # Calculate the optimal step value
            (
                optimal_step_val,
                optimal_prep_fidality,
                optimal_readout_fidality,
                max_combined_score,
            ) = find_optimal_value_geom_mean(
                step_vals,
                readout_fidelity_arr[nv_ind],
                prep_fidelity_arr[nv_ind],
                goodness_of_fit_arr[nv_ind],
                weights=(1, 1, 1),
            )
            optimal_values.append(
                (
                    nv_ind,
                    optimal_step_val,
                    optimal_prep_fidality,
                    optimal_readout_fidality,
                    max_combined_score,
                )
            )
        except Exception as e:
            print(f"Failed to process NV{nv_ind}: {e}")
            optimal_values.append((nv_ind, np.nan, np.nan))
            continue

        # # Plotting
        fig, ax1 = plt.subplots(figsize=(7, 5))
        # Plot readout fidelity
        ax1.plot(
            step_vals,
            readout_fidelity_arr[nv_ind],
            label="Readout Fidelity",
            color="blue",
        )
        ax1.plot(
            step_vals,
            prep_fidelity_arr[nv_ind],
            label="Prep Fidelity",
            linestyle="--",
            color="blue",
        )
        ax1.set_xlabel(x_label)
        ax1.set_ylabel("Fidelity")
        ax1.tick_params(axis="y", labelcolor="blue")
        ax1.grid(True, linestyle="--", alpha=0.6)

        # Plot Goodness of Fit ()
        ax2 = ax1.twinx()
        ax2.plot(
            step_vals,
            goodness_of_fit_arr[nv_ind],
            color="green",
            label=r"Goodness of Fit ($\chi^2_{\text{reduced}}$)",
            alpha=0.7,
        )
        ax2.set_ylabel(r"Goodness of Fit ($\chi^2_{\text{reduced}}$)", color="green")
        ax2.tick_params(axis="y", labelcolor="green")

        # Highlight optimal step value
        ax1.axvline(
            optimal_step_val,
            color="red",
            linestyle="--",
            label=f"Optimal Step Val: {optimal_step_val:.3f}",
        )
        ax2.axvline(
            optimal_step_val,
            color="red",
            linestyle="--",
        )

        # Combine legends
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc="upper left", fontsize=11)
        ax1.set_title(f"NV{nv_ind} - Optimal Step Val: {optimal_step_val:.3f}")
        plt.tight_layout()
        plt.show()

    ### Calculate Averages
    avg_readout_fidelity = np.nanmean(readout_fidelity_arr, axis=0)
    avg_prep_fidelity = np.nanmean(prep_fidelity_arr, axis=0)
    avg_goodness_of_fit = np.nanmean(goodness_of_fit_arr, axis=0)

    # Calculate the optimal step value
    (
        optimal_step_val,
        optimal_prep_fidelity,
        optimal_readout_fidelity,
        max_combined_score,
    ) = find_optimal_value_geom_mean(
        step_vals,
        avg_readout_fidelity,
        avg_prep_fidelity,
        avg_goodness_of_fit,
        weights=(1, 1, 1),
    )
    # Plot average readout and prep fidelity
    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax1.plot(
        step_vals,
        avg_readout_fidelity,
        label="Avg. Readout Fidelity",
        color="orange",
    )
    ax1.plot(
        step_vals,
        avg_prep_fidelity,
        label="Avg. Prep Fidelity",
        color="green",
    )
    ax1.set_xlabel(x_label)
    ax1.set_ylabel("Fidelity")
    ax1.tick_params(axis="y")

    # Plot average Goodness of Fit (reduced chi-squared))
    ax2 = ax1.twinx()
    ax2.plot(
        step_vals,
        avg_goodness_of_fit,
        color="gray",
        linestyle="--",
        label=r"Goodness of Fit ($\chi^2_{\text{reduced}}$)",
    )
    ax2.set_ylabel(r"Goodness of Fit ($\chi^2_{\text{reduced}}$)", color="gray")

    ax2.tick_params(axis="y", labelcolor="gray")
    ax2.axvline(
        optimal_step_val,
        color="red",
        linestyle="--",
    )
    optimal_text = (
        f"Optimal {x_label}: {optimal_step_val:.3f}\n"
        f"Optimal Prep Fidelity: {optimal_prep_fidelity:.3f}\n"
        f"Optimal Readout Fidelity: {optimal_readout_fidelity:.3f}"
    )
    ax1.text(
        0.01,
        0.99,
        optimal_text,
        transform=ax1.transAxes,
        fontsize=8,
        verticalalignment="top",
        bbox=dict(
            boxstyle="round,pad=0.1", alpha=0.5, edgecolor="gray", facecolor="white"
        ),
    )
    # Combine legends
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right", fontsize=8)

    # Title and layout
    ax1.set_title(f"Average Metrics Across All NVs ({file_id})", fontsize=16)
    fig.tight_layout()
    plt.show()
    ### Calculate Medians
    median_readout_fidelity = np.nanmedian(readout_fidelity_arr, axis=0)
    median_prep_fidelity = np.nanmedian(prep_fidelity_arr, axis=0)
    median_goodness_of_fit = np.nanmedian(goodness_of_fit_arr, axis=0)

    (
        optimal_step_val,
        optimal_prep_fidelity,
        optimal_readout_fidelity,
        max_combined_score,
    ) = find_optimal_value_geom_mean(
        step_vals,
        median_readout_fidelity,
        median_prep_fidelity,
        median_goodness_of_fit,
        weights=(1, 1, 1),
    )

    # Plot average and median readout and prep fidelity
    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax1.plot(
        step_vals,
        median_readout_fidelity,
        label="Median Readout Fidelity",
        color="orange",
    )
    ax1.plot(
        step_vals,
        median_prep_fidelity,
        label="Median Prep Fidelity",
        color="green",
    )
    ax1.set_xlabel(x_label)
    ax1.set_ylabel("Fidelity")
    ax1.tick_params(axis="y")

    # Plot average and median Goodness of Fit (reduced chi-squared))
    ax2 = ax1.twinx()
    ax2.plot(
        step_vals,
        median_goodness_of_fit,
        color="gray",
        linestyle="--",
        label=r"Goodness of Fit ($\chi^2_{\text{reduced}}$)",
    )
    ax2.set_ylabel(r"Goodness of Fit ($\chi^2_{\text{reduced}}$)", color="gray")

    ax2.tick_params(axis="y", labelcolor="gray")
    ax2.axvline(
        optimal_step_val,
        color="red",
        linestyle="--",
    )
    # Add a box with optimal values on the left
    optimal_text = (
        f"Optimal {x_label}: {optimal_step_val:.3f}\n"
        f"Optimal Prep Fidelity: {optimal_prep_fidelity:.3f}\n"
        f"Optimal Readout Fidelity: {optimal_readout_fidelity:.3f}"
    )
    ax1.text(
        0.01,
        0.99,
        optimal_text,
        transform=ax1.transAxes,
        fontsize=8,
        verticalalignment="top",
        bbox=dict(
            boxstyle="round,pad=0.1", alpha=0.5, edgecolor="black", facecolor="white"
        ),
    )
    # Combine legends
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right", fontsize=8)

    # Title and layout
    ax1.set_title(f"Median Metrics Across All NVs ({file_id})", fontsize=16)
    fig.tight_layout()
    plt.show()


# endregion

if __name__ == "__main__":
    kpl.init_kplotlib()
    # file_id = 1710843759806
    file_id = 1712782503640  # yellow ampl var 50ms
    file_id = 1723230400688  # yellow ampl var 30ms
    # file_id = 1717056176426  # yellow duration var
    # file_id = 1711618252292  # green ampl var
    # file_id = 1712421496166  # green ampl var
    # file_id = 1720970373150  # yellow ampl var iter_1
    # raw_data = dm.get_raw_data(file_id=1709868774004, load_npz=False) #yellow ampl var
    # raw_data = dm.get_raw_data(file_id=file_id, load_npz=False)  # yellow amp var
    # process_and_plot(raw_data)
    # kpl.show(block=True)
    print(dm.get_file_name(1717056176426))
