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
from matplotlib.ticker import MaxNLocator
from scipy import ndimage
from scipy.optimize import curve_fit
from scipy.special import factorial
from scipy.stats import poisson
from scipy.interpolate import griddata

from majorroutines.widefield import base_routine
from utils import common, widefield
from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import positioning as pos
from utils import tool_belt as tb
from utils.constants import ChargeStateEstimationMode, NVSig, VirtualLaserKey
from analysis.bimodal_histogram import (
    ProbDist,
    determine_threshold,
    fit_bimodal_histogram,
)


def process_check_readout_fidelity(data, fidelity_ax=None):
    nv_list = data["nv_list"]
    num_nvs = len(nv_list)
    counts = np.array(data["counts"])
    num_runs = counts.shape[2]
    num_reps = counts.shape[4]
    sig_counts = counts[0]
    config = common.get_config_dict()
    # charge_state_estimation_mode = config["charge_state_estimation_mode"]
    # charge_state_estimation_mode = ChargeStateEstimationMode.THRESHOLDING
    # charge_state_estimation_mode = ChargeStateEstimationMode.MLE
    # if charge_state_estimation_mode == ChargeStateEstimationMode.THRESHOLDING:
    #     states = widefield.threshold_counts(nv_list, sig_counts)
    # elif charge_state_estimation_mode == ChargeStateEstimationMode.MLE:
    #     states = np.array(data["states"])[0]1

    states = widefield.threshold_counts(nv_list, sig_counts)
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


# def process_detect_cosmic_rays(data):
#     """
#     Process cosmic ray detection data with additional analyses.

#     Parameters:
#         data (dict): Collected raw data.

#     Returns:
#         tuple: Figures for the histogram, image plots, transition times, and spatial heatmaps.
#     """
#     nv_list = data["nv_list"]
#     num_reps = data["num_reps"]
#     num_runs = data["num_runs"]
#     num_nvs = len(nv_list)
#     # Example usage
#     timestamps = generate_timestamps(
#         num_reps,
#         num_runs,
#         dark_time_ns=1e9,
#         deadtime_ns=12e6,
#         readout_time_ns=50e6,
#     )
#     states = np.array(data["states"]).reshape((num_nvs, -1))
#     coincidences = num_nvs - states.sum(axis=0)
#     # 1. Histogram of Coincidences
#     hist_fig, ax = plt.subplots()
#     kpl.histogram(ax, coincidences, label=f"Data ({num_nvs} NVs)")
#     ax.set_xlabel("Number NVs found in NV0")
#     ax.set_ylabel("Number of occurrences")
#     ax.set_title("Cosmic Ray Detection")
#     ax.legend()
#     x_vals = np.arange(num_nvs + 1)
#     expected_dist = len(coincidences) * poisson.pmf(x_vals, np.mean(coincidences))
#     kpl.plot_points(
#         ax, x_vals, expected_dist, label="Poisson PMF", color=kpl.KplColors.RED
#     )
#     ax.set_yscale("log")

#     # 2. State Visualization
#     im_fig, ax = plt.subplots()
#     kpl.imshow(ax, states, aspect="auto", cmap="viridis")
#     ax.yaxis.set_major_locator(MaxNLocator(integer=True))
#     ax.set_title("NV State Visualization")

#     # 3. Transition Time Analysis
#     transition_fig, ax = plt.subplots()
#     transition_times = analyze_transition_times(states, timestamps)
#     for nv_ind, times in enumerate(transition_times):
#         ax.hist(
#             times[~np.isnan(times)],
#             bins=50,
#             alpha=0.5,
#             label=f"NV {nv_ind}",
#             histtype="step",
#         )
#     ax.set_title("Transition Time Distribution")
#     ax.set_xlabel("Time (s)")
#     ax.set_ylabel("Frequency")
#     ax.legend()

#     # 4. Heatmap of NV Activity
#     activity_heatmap_fig, ax = plt.subplots()
#     activity_counts = states.sum(axis=1)
#     kpl.imshow(ax, activity_counts.reshape(num_nvs, 1), aspect="auto", cmap="plasma")
#     ax.set_title("NV Activity Heatmap")
#     ax.set_xlabel("Activity")
#     ax.set_ylabel("NV Index")

#     # 5. Temporal Clustering
#     temporal_corr_fig, ax = plt.subplots()
#     corr_matrix = np.corrcoef(states)
#     kpl.imshow(ax, corr_matrix, aspect="auto", cmap="coolwarm")
#     ax.set_title("Temporal Transition Correlation Matrix")
#     ax.set_xlabel("NV Index")
#     ax.set_ylabel("NV Index")

#     return hist_fig, im_fig, transition_fig, activity_heatmap_fig, temporal_corr_fig


# def analyze_transition_times(states, timestamps):
#     """
#     Analyze the timing of charge state transitions for NV centers.

#     Parameters:
#         states (ndarray): Binary state matrix (NV x Time).
#         timestamps (ndarray): Array of timestamps for each state.

#     Returns:
#         list: Transition times for each NV.
#     """
#     transition_times = []
#     for nv_states in states:
#         transitions = np.diff(nv_states)
#         transition_indices = np.where(transitions != 0)[0]
#         times = timestamps[transition_indices + 1]  # Match indices with timestamps
#         padded_times = np.full_like(nv_states, np.nan, dtype=np.float64)
#         padded_times[transition_indices] = times
#         transition_times.append(padded_times)
#     return transition_times


# def calculate_transition_probabilities(states):
#     """
#     Calculate transition probabilities for each NV.

#     Parameters:
#         states (ndarray): NV states array (num_nvs x num_timesteps).

#     Returns:
#         list: Transition probabilities for each NV.
#     """
#     transition_probs = []
#     for nv_states in states:
#         transitions = np.diff(nv_states)
#         prob = np.sum(transitions != 0) / len(transitions)
#         transition_probs.append(prob)
#     return transition_probs


# def generate_timestamps(num_reps, num_runs, dark_time_ns, deadtime_ns, readout_time_ns):
#     """
#     Generate timestamps for all repetitions based on provided timing info.

#     Parameters:
#         num_reps (int): Number of repetitions per run.
#         num_runs (int): Total number of runs.
#         dark_time_ns (float): Dark time in nanoseconds.
#         deadtime_ns (float): Deadtime in nanoseconds.
#         readout_time_ns (float): Readout time in nanoseconds.

#     Returns:
#         np.ndarray: Array of timestamps in seconds for all repetitions.
#     """
#     # Total cycle time in nanoseconds
#     cycle_time_ns = dark_time_ns + deadtime_ns + readout_time_ns

#     # Generate timestamps
#     total_reps = num_reps * num_runs
#     timestamps_ns = np.arange(total_reps) * cycle_time_ns

#     # Convert to seconds for readability
#     return timestamps_ns / 1e9


# def visualize_raw_counts(counts):
#     plt.figure()
#     for nv_ind, nv_counts in enumerate(counts):
#         plt.hist(nv_counts.flatten(), bins=50, alpha=0.5, label=f"NV {nv_ind}")
#     plt.xlabel("Counts")
#     plt.ylabel("Frequency")
#     plt.title("Raw Count Distribution")
#     plt.legend()
#     plt.show()


# if __name__ == "__main__":
#     kpl.init_kplotlib()

#     # Load data
#     # data = dm.get_raw_data(file_id=1568108087044)  # Replace with relevant ID
#     data = dm.get_raw_data(file_id=1695946921364)  # Replace with relevant ID
#     counts = np.array(data["counts"])
#     states = np.array(data["states"])
#     print(f"counts.shape={counts.shape}")
#     print(f"states.shape={states.shape}")
#     # visualize_raw_counts(counts)
#     # print(data.keys())
#     # (
#     #     hist_fig,
#     #     im_fig,
#     #     transition_fig,
#     #     activity_heatmap_fig,
#     #     temporal_corr_fig,
#     # ) = process_detect_cosmic_rays(data)

#     kpl.show(block=True)


def process_detect_cosmic_rays(data, prob_dist: ProbDist = ProbDist.COMPOUND_POISSON):
    """
    Process cosmic ray detection data with additional analyses, including counts visualization.

    Parameters:
        data (dict): Collected raw data.

    Returns:
        tuple: Figures for the histogram, image plots, transition times, spatial heatmaps, and counts.
    """
    nv_list = data["nv_list"]
    num_reps = data["num_reps"]
    num_runs = data["num_runs"]
    counts_initial = np.array(data["counts"])
    print(f"counts.shape={counts_initial.shape}")
    # fmt: off
    selected_indices = [0, 1, 3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 29, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 64, 67, 68, 69, 70, 71, 73, 74, 75, 76, 77, 78, 79, 80, 82, 84, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 98, 101, 103, 104, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 118, 119, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 149, 150, 151, 152, 153, 154, 157, 158, 159]
    # fmt: on
    nv_list = [nv_list[ind] for ind in selected_indices]
    num_nvs = len(nv_list)
    counts = counts_initial[:, selected_indices, :, :, :]
    sig_counts = counts[0]
    # ref_counts = counts[1]
    # Extract counts and states
    states = widefield.threshold_counts(nv_list, sig_counts, dynamic_thresh=True)
    states = np.array(states).reshape((num_nvs, -1))
    # states = np.array(data["counts"]).reshape((num_nvs, -1))

    ### Histograms and thresholding

    threshold_list = []
    readout_fidelity_list = []
    prep_fidelity_list = []
    red_chi_sq_list = []

    for ind in range(num_nvs):
        ref_counts_list = sig_counts[ind]

        # Only use ref counts for threshold determination
        popt, _, red_chi_sq = fit_bimodal_histogram(
            ref_counts_list, prob_dist, no_print=True
        )
        threshold, readout_fidelity = determine_threshold(
            popt, prob_dist, dark_mode_weight=0.5, do_print=True, ret_fidelity=True
        )
        threshold_list.append(threshold)
        readout_fidelity_list.append(readout_fidelity)
        if popt is not None:
            prep_fidelity = 1 - popt[0]
        else:
            prep_fidelity = np.nan
        # prep_fidelity = (
        #     np.count_nonzero(np.array(ref_counts_list) > threshold) / num_shots
        # )  # MCC
        prep_fidelity_list.append(prep_fidelity)
        red_chi_sq_list.append(red_chi_sq)

    # Convert lists to arrays for easier manipulation
    prep_fidelity_array = np.array(prep_fidelity_list)
    readout_fidelity_array = np.array(readout_fidelity_list)

    # Calculate fidelity weights
    fidelity_weights = prep_fidelity_array * readout_fidelity_array
    # fidelity_weights = np.nan_to_num(fidelity_weights, nan=0.0)  # Handle NaNs
    # Weight the states by fidelity
    weighted_states = states * fidelity_weights[:, np.newaxis]
    # Generate timestamps
    timestamps = generate_timestamps(
        num_reps,
        num_runs,
        dark_time_ns=1e9,
        deadtime_ns=100e6,
        readout_time_ns=50e6,
    )

    # Coincidences
    coincidences = num_nvs - states.sum(axis=0)
    # coincidences = num_nvs - weighted_states.sum(axis=0)
    # coincidences = states.sum(axis=0)

    # 1. Histogram of Coincidences
    hist_fig, ax = plt.subplots()
    # Compute histogram data
    bin_edges = np.arange(-0.5, num_nvs + 1.5, 1)
    hist_values, _ = np.histogram(coincidences, bins=bin_edges)

    # Plot histogram as a bar chart for better readability
    ax.bar(
        bin_edges[:-1] + 0.5,  # Center bins
        hist_values,
        width=1.0,
        align="center",
        alpha=0.6,
        label=f"Observed ({num_nvs} NVs)",
        color="blue",
    )

    # Add Poisson PMF for comparison
    x_vals = np.arange(num_nvs + 1)
    expected_dist = len(coincidences) * poisson.pmf(x_vals, np.mean(coincidences))
    ax.plot(
        x_vals,
        expected_dist,
        marker="o",
        linestyle="--",
        label="Poisson PMF (Expected Dist.)",
        color="red",
        alpha=0.6,
        markersize=2,
    )

    # Set y-axis limits dynamically based on data range
    # ax.set_ylim(1, max(max(hist_values), max(expected_dist)) * 1.0)

    # Add labels, title, and legend
    ax.set_xlabel("Number of NVs affected (Coincidences)", fontsize=15)
    ax.set_ylabel("Number of occurrences", fontsize=15)
    ax.set_title("Charge Jump - Coincidence Histogram", fontsize=15)
    ax.legend(fontsize=12, loc="upper right")

    # Use logarithmic scale for y-axis
    # ax.set_yscale("log")
    ax.grid(True, which="both", linestyle="--", alpha=0.6)
    ax.tick_params(axis="both", which="major", labelsize=15)

    # 2. Time-Resolved Histogram
    time_resolved_fig, ax = plt.subplots()
    time_bins = np.linspace(timestamps.min(), timestamps.max(), 6)  # 10 time intervals
    for start, end in zip(time_bins[:-1], time_bins[1:]):
        mask = (timestamps >= start) & (timestamps < end)
        time_coincidences = coincidences[mask]
        hist_values, _ = np.histogram(time_coincidences, bins=bin_edges)
        ax.plot(
            bin_edges[:-1] + 0.5,
            hist_values,
            label=f"{start:.2f}s - {end:.2f}s",
            marker="o",
            markersize=2,
        )
    ax.set_xlabel("Number of NVs affected (Coincidences)", fontsize=15)
    ax.set_ylabel("Frequency", fontsize=15)
    ax.set_title("Time-Resolved Coincidence Histogram", fontsize=15)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.6)

    # 3. Spatial Heatmap of NV Activity
    activity_counts = states.sum(axis=1)
    x_coords = [nv.coords["pixel"][0] for nv in nv_list]
    y_coords = [nv.coords["pixel"][1] for nv in nv_list]

    # Create the spatial heatmap without interpolation
    spatial_heatmap_fig, ax = plt.subplots()
    scatter = ax.scatter(
        x_coords,
        y_coords,
        c=activity_counts,
        cmap="coolwarm",
        edgecolors="black",
        s=60,
        label="NV Positions",
    )
    plt.colorbar(scatter, ax=ax, label="Activity Count (Sum)")
    ax.set_xlabel("X Coordinate (Pixels)")
    ax.set_ylabel("Y Coordinate (Pixels)")
    ax.set_title("Spatial map of NV Activity")
    ax.legend(fontsize=10)
    plt.show()

    ### Plot 4: Single Shot Spatial Heatmaps
    import random

    num_shots_to_plot = 6
    shot_indices = np.linspace(0, states.shape[1] - 1, num_shots_to_plot, dtype=int)
    num_shots_to_plot = 6
    shot_indices = random.sample(range(states.shape[1]), num_shots_to_plot)
    single_shot_figs = []
    for idx in shot_indices:
        single_shot_fig, ax = plt.subplots()
        scatter = ax.scatter(
            x_coords,
            y_coords,
            c=states[:, idx],
            cmap="coolwarm",
            edgecolors="black",
            s=60,
        )
        plt.colorbar(scatter, ax=ax, label="Activity Count (Single Shot)")
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        ax.set_title(f"Spatial map - Shot {idx + 1}")
        single_shot_figs.append(single_shot_fig)

    return hist_fig, time_resolved_fig, spatial_heatmap_fig, single_shot_figs


def analyze_transition_times(states, timestamps):
    """
    Analyze the timing of charge state transitions for NV centers.

    Parameters:
        states (ndarray): Binary state matrix (NV x Time).
        timestamps (ndarray): Array of timestamps for each state.

    Returns:
        list: Transition times for each NV.
    """
    transition_times = []
    for nv_states in states:
        transitions = np.diff(nv_states)
        transition_indices = np.where(transitions != 0)[0]
        times = timestamps[transition_indices + 1]  # Match indices with timestamps
        padded_times = np.full_like(nv_states, np.nan, dtype=np.float64)
        padded_times[transition_indices] = times
        transition_times.append(padded_times)
    return transition_times


def generate_timestamps(num_reps, num_runs, dark_time_ns, deadtime_ns, readout_time_ns):
    """
    Generate timestamps for all repetitions based on provided timing info.

    Parameters:
        num_reps (int): Number of repetitions per run.
        num_runs (int): Total number of runs.
        dark_time_ns (float): Dark time in nanoseconds.
        deadtime_ns (float): Deadtime in nanoseconds.
        readout_time_ns (float): Readout time in nanoseconds.

    Returns:
        np.ndarray: Array of timestamps in seconds for all repetitions.
    """
    # Total cycle time in nanoseconds
    cycle_time_ns = dark_time_ns + deadtime_ns + readout_time_ns

    # Generate timestamps
    total_reps = num_reps * num_runs
    timestamps_ns = np.arange(total_reps) * cycle_time_ns

    # Convert to seconds for readability
    return timestamps_ns / 1e9


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson, skew


def plot_histogram_multi(data_files, dark_times):
    """
    Plot comparison histograms with expected Poisson distribution for multiple datasets.
    Additionally, compute and plot skewness vs. dark time.

    Parameters:
        data_files (list): List of data files containing coincidence data.
        dark_times (list): List of corresponding dark times for labeling.
    """
    skewness_values = []  # To store skewness for each dataset
    plt.figure(figsize=(10, 7))

    for data_file, dark_time in zip(data_files, dark_times):
        # Load the data
        data = dm.get_raw_data(file_id=data_file)
        nv_list = data["nv_list"]
        num_nvs = len(nv_list)
        counts = np.array(data["counts"])[0]  # Extract signal counts
        states = widefield.threshold_counts(nv_list, counts, dynamic_thresh=True)
        states = np.array(states).reshape((num_nvs, -1))
        coincidences = num_nvs - states.sum(axis=0)

        # Compute histogram data
        bin_edges = np.arange(-0.5, num_nvs + 1.5, 1)
        hist_values, _ = np.histogram(coincidences, bins=bin_edges)

        # Compute skewness
        distribution_skewness = skew(coincidences)
        skewness_values.append(distribution_skewness)

        # Plot histogram as a line plot
        plt.plot(
            bin_edges[:-1] + 0.5,  # Center bins
            hist_values,
            marker="o",
            linestyle="-",
            label=f"Observed - Dark Time {dark_time:.0e} ns\nSkew: {distribution_skewness:.2f}",
            alpha=0.7,
        )

        # Compute and plot expected Poisson distribution
        x_vals = np.arange(num_nvs + 1)
        expected_dist = len(coincidences) * poisson.pmf(x_vals, np.mean(coincidences))
        plt.plot(
            x_vals,
            expected_dist,
            marker="x",
            linestyle="--",
            label=f"Expected - Dark Time {dark_time:.0e} ns",
            alpha=0.7,
        )

    # Customize the histogram plot
    plt.xlabel("Number of NVs affected (Coincidences)", fontsize=14)
    plt.ylabel("Number of occurrences", fontsize=14)
    plt.title("Charge Jump - Coincidence Histograms", fontsize=16)
    plt.legend(fontsize=10, loc="upper right")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()

    # Plot skewness vs. dark time
    plt.figure(figsize=(8, 6))
    plt.plot(dark_times, skewness_values, marker="o", linestyle="-", color="blue")
    plt.xlabel("Dark Time (ns)", fontsize=14)
    plt.ylabel("Skewness", fontsize=14)
    plt.title("Skewness vs. Dark Time", fontsize=16)
    plt.xscale("log")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()


def calculate_correlation_over_time(data, window_size):
    """
    Calculate correlation of NV states over time.

    Parameters:
        states (ndarray): Binary matrix of shape (num_nvs, num_timesteps).
        window_size (int): Size of the time window for correlation calculation.

    Returns:
        correlations (list): List of correlation matrices for each time window.
        time_windows (list): List of start times for each time window.
    """
    num_nvs, num_timesteps = states.shape
    correlations = []
    time_windows = []

    for start in range(0, num_timesteps - window_size + 1, window_size):
        window_data = states[:, start : start + window_size]
        correlation_matrix = np.corrcoef(window_data)
        correlations.append(correlation_matrix)
        time_windows.append(start)

    return correlations, time_windows


def plot_correlation_over_time(correlations, time_windows):
    """
    Plot the correlation matrices for multiple time windows.

    Parameters:
        correlations (np.ndarray): 3D array of correlations (time_windows x NVs x NVs).
        time_windows (list): List of time window labels.
    """
    correlations = np.array(correlations)
    num_windows = correlations.shape[0]

    for i in range(num_windows):
        fig, ax = plt.subplots(figsize=(8, 6))

        # Extract the correlation matrix for the current time window
        correlation_matrix = correlations[i]

        # Set diagonal to NaN for better visualization (if square matrix)
        if correlation_matrix.shape[0] == correlation_matrix.shape[1]:
            np.fill_diagonal(correlation_matrix, np.nan)

        # Calculate vmin and vmax dynamically
        mean = np.nanmean(correlation_matrix)
        sdt = np.nanstd(correlation_matrix)
        vmin = mean - 2 * sdt
        vmax = mean + 2 * sdt
        # vmin = max(vmin, -1)  # Ensure the range is within valid correlation values
        # vmax = min(vmax, 1)

        # Plot the heatmap
        im = ax.imshow(
            correlation_matrix,
            cmap="coolwarm",
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )

        # Add colorbar and labels
        plt.colorbar(im, ax=ax, label="Correlation")
        ax.set_title(f"Correlation Matrix for Time Window {time_windows[i]}")
        ax.set_xlabel("NV Index")
        ax.set_ylabel("NV Index")
        plt.show()


if __name__ == "__main__":
    kpl.init_kplotlib()

    # Load data
    # data = dm.get_raw_data(file_id=1695946921364)  # dark time 1000e9 (november data)
    # DATA IN jAN 2025
    # data = dm.get_raw_data(file_id=1756083081553)  # dark time 1e6
    # data = dm.get_raw_data(file_id=1756161618282)  # dark time 10e6
    # data = dm.get_raw_data(file_id=1756305080720)  # dark time 100e6
    # data = dm.get_raw_data(file_id=1756699202091)  # dark time 100e6
    # data = dm.get_raw_data(file_id=1755068762133)  # dark time 1000e9

    # Process data
    # (
    #     hist_fig,
    #     transition_fig,
    #     spatial_heatmap_fig,
    #     single_shot_figs,
    # ) = process_detect_cosmic_rays(data)

    # data_files = [1756083081553, 1756161618282, 1756305080720, 1755068762133]
    # dark_times = [1e6, 10e6, 100e6, 1000e6]
    data_files = [1756161618282, 1756305080720, 1756699202091, 1755068762133]
    dark_times = [10e6, 100e6, 500e6, 1000e6]

    # data_files = [1756161618282, 1755068762133]
    # dark_times = [10e6, 1000e6]

    plot_histogram_multi(data_files, dark_times)

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
