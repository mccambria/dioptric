# -*- coding: utf-8 -*-
"""
Illuminate an area, collecting onto the camera. Interleave a signal and control sequence
and plot the difference
Created on Fall 2024
@author: sbchand
"""
import os
import sys
import time
import traceback
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed


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

import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.family'] = 'DejaVu Sans'
rcParams['font.family'] = 'Roboto'

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
        (norm_readout_fidelity**w1)
        * (norm_prep_fidelity**w2)
        * (inverted_goodness**w3)
    ) ** (1 / (w1 + w2 + w3))

    # Find the step value corresponding to the maximum combined score
    max_index = np.nanargmax(combined_score)
    optimal_step_val = step_vals[max_index]
    max_combined_score = combined_score[max_index]

    return optimal_step_val, max_combined_score

def process_and_plot(raw_data):
    nv_list = raw_data["nv_list"]
    num_nvs = len(nv_list)
    min_step_val = raw_data["min_step_val"]
    max_step_val = raw_data["max_step_val"]
    num_steps = raw_data["num_steps"]
    step_vals = np.linspace(min_step_val, max_step_val, num_steps) * 0.39
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

    # Function to process a single NV and step
    def process_nv_step(nv_ind, step_ind):
        counts_data = condensed_counts[nv_ind, step_ind]
        popt, chi_squared = fit_bimodal_histogram(counts_data, prob_dist)

        if popt is None:
            return np.nan, np.nan, np.nan, np.nan

        # Threshold, prep and readout fidelity
        threshold, readout_fidelity = determine_threshold(
            popt, prob_dist, dark_mode_weight=0.5, ret_fidelity=True
        )
        prep_fidelity = 1 - popt[0]  # Population weight of dark state

        # Calculate inter-class and intra-class variance for separation metric
        w_dark, mu_dark, mu_bright = popt[0], popt[1], popt[2]
        sigma_dark = np.sqrt(mu_dark)
        sigma_bright = np.sqrt(mu_bright)
        inter_class_variance = w_dark * (1 - w_dark) * (mu_bright - mu_dark) ** 2
        intra_class_variance = w_dark * sigma_dark**2 + (1 - w_dark) * sigma_bright**2
        separation_metric = (
            inter_class_variance / intra_class_variance
            if intra_class_variance > 0
            else np.nan
        )

        return readout_fidelity, prep_fidelity, separation_metric, chi_squared

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
    # separation_metric_arr = results[:, :, 2]
    goodness_of_fit_arr = results[:, :, 3]

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
            # x_label = "Pol. amplitude"

    # Optimal values
    optimal_values = []
    for nv_ind in range(num_nvs):
        try:
            # Calculate the optimal step value
            optimal_step_val, max_combined_score = find_optimal_value_geom_mean(
                step_vals,
                readout_fidelity_arr[nv_ind],
                prep_fidelity_arr[nv_ind],
                goodness_of_fit_arr[nv_ind],
            )
            optimal_values.append((nv_ind, optimal_step_val, max_combined_score))
        except Exception as e:
            print(f"Failed to process NV{nv_ind}: {e}")
            optimal_values.append((nv_ind, np.nan, np.nan))
            continue

        # Plotting
        fig, ax1 = plt.subplots(figsize=(6, 5))

        # Plot readout fidelity
        ax1.plot(
            step_vals,
            # readout_fidelity_arr[nv_ind],
            prep_fidelity_arr[nv_ind],
            # label="Readout Fidelity",
            label="Prep Fidelity",
            color="blue",
        )
        ax1.set_xlabel(x_label)
        ax1.set_ylabel("Readout Fidelity")
        ax1.set_ylabel("Prep Fidelity")
        ax1.tick_params(axis="y", labelcolor="blue")
        ax1.legend(loc="upper left", fontsize=9)

        # Plot goodness of fit (R²)
        ax2 = ax1.twinx()
        ax2.plot(
            step_vals,
            goodness_of_fit_arr[nv_ind],
            color="green",
            label="Goodness of Fit (R²)",
            alpha=0.7,
        )
        ax2.set_ylabel("Goodness of Fit (R²)", color="green")
        ax2.tick_params(axis="y", labelcolor="green")

        # Highlight optimal step value
        ax1.axvline(
            optimal_step_val,
            color="red",
            linestyle="--",
            label=f"Optimal Step Val: {optimal_step_val:.2f}",
        )
        ax2.axvline(
            optimal_step_val,
            color="red",
            linestyle="--",
            label=f"Optimal Step Val: {optimal_step_val:.2f}",
        )

        # Add legends for both y-axes
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax2.legend(
            lines_1 + lines_2, labels_1 + labels_2, loc="upper right", fontsize=9
        )

        # Title and layout
        ax1.set_title(f"NV{nv_ind} - Optimal Step Val: {optimal_step_val:.2f}")
        fig.tight_layout()
        plt.show()

    # Save results to a file
    with open("optimal_combined_values.txt", "w") as f:
        f.write("NV Index, Optimal Step Value, Max Combined Score\n")
        for nv_index, opt_step, max_score in optimal_values:
            f.write(f"{nv_index}, {opt_step:.6f}, {max_score:.6f}\n")
    print("Optimal combined values saved to 'optimal_combined_values.txt'.")


    ### Calculate Averages
    avg_readout_fidelity = np.nanmean(readout_fidelity_arr, axis=0)
    avg_prep_fidelity = np.nanmean(prep_fidelity_arr, axis=0)
    avg_goodness_of_fit = np.nanmean(goodness_of_fit_arr, axis=0)

    # Calculate the optimal step value
    optimal_step_val, max_combined_score = find_optimal_value_geom_mean(
        step_vals,
        avg_readout_fidelity,
        avg_prep_fidelity,
        avg_goodness_of_fit,
    )

    # Plot average readout and prep fidelity
    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax1.plot(
        step_vals,
        avg_readout_fidelity,
        label="Avg. Readout Fidelity",
        color="blue",
    )
    ax1.plot(
        step_vals,
        avg_prep_fidelity,
        label="Avg. Prep Fidelity",
        color="orange",
    )
    ax1.set_xlabel(x_label)
    ax1.set_ylabel("Fidelity")
    ax1.tick_params(axis="y")

    # Plot average goodness of fit (R²)
    ax2 = ax1.twinx()
    ax2.plot(
        step_vals,
        avg_goodness_of_fit,
        color="green",
        linestyle="--",
        label="Avg. Goodness of Fit (chi-squared)",
    )
    ax2.set_ylabel("Goodness of Fit (chi-squared)", color="green")

    ax2.tick_params(axis="y", labelcolor="green")
    ax2.axvline(
        optimal_step_val,
        color="red",
        linestyle="--",
        label=f"Optimal Step Val: {optimal_step_val:.2f}",
    )
    # Combine legends
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right", fontsize=9)

    # Title and layout
    ax1.set_title("Average Metrics Across All NVs", fontsize=12)
    fig.tight_layout()
    plt.show()
# endregion

if __name__ == "__main__":
    kpl.init_kplotlib()
    # raw_data = dm.get_raw_data(file_id=1709868774004, load_npz=False) #yellow ampl var
    raw_data = dm.get_raw_data(file_id=1710843759806, load_npz=False) #yellow amp var
    # raw_data = dm.get_raw_data(file_id=1711618252292, load_npz=False) #green ampl var
    process_and_plot(raw_data)
    kpl.show(block=True)
