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
    selected_indices = list(range(len(nv_list)))
    # fmt: on
    nv_list = [nv_list[ind] for ind in selected_indices]
    num_nvs = len(nv_list)
    counts = counts_initial[:, selected_indices, :, :, :]
    ref_counts = counts[0]
    sig_counts = counts[1]
    print(f"sig_coutns_shape = {sig_counts.shape}")
    # ref_counts = counts[1]
    # Extract counts and states
    states_0 = widefield.threshold_counts(nv_list, sig_counts, dynamic_thresh=True)
    states_1 = widefield.threshold_counts(nv_list, ref_counts, dynamic_thresh=True)
    ref_states = np.array(states_0).reshape((num_nvs, -1))
    sig_states = np.array(states_1).reshape((num_nvs, -1))
    states = ref_states
    # states = sig_states
    # states = np.array(states).reshape((num_nvs, -1))
    print(f"states.shape = {states.shape}")
    # states = np.array(data["counts"]).reshape((num_nvs, -1))

    # ### Histograms and thresholding

    # threshold_list = []
    # readout_fidelity_list = []
    # prep_fidelity_list = []
    # red_chi_sq_list = []

    # for ind in range(num_nvs):
    #     ref_counts_list = sig_counts[ind]

    #     # Only use ref counts for threshold determination
    #     popt, _, red_chi_sq = fit_bimodal_histogram(
    #         ref_counts_list, prob_dist, no_print=True
    #     )
    #     threshold, readout_fidelity = determine_threshold(
    #         popt, prob_dist, dark_mode_weight=0.5, do_print=True, ret_fidelity=True
    #     )
    #     threshold_list.append(threshold)
    #     readout_fidelity_list.append(readout_fidelity)
    #     if popt is not None:
    #         prep_fidelity = 1 - popt[0]
    #     else:
    #         prep_fidelity = np.nan
    #     # prep_fidelity = (
    #     #     np.count_nonzero(np.array(ref_counts_list) > threshold) / num_shots
    #     # )  # MCC
    #     prep_fidelity_list.append(prep_fidelity)
    #     red_chi_sq_list.append(red_chi_sq)

    # # Convert lists to arrays for easier manipulation
    # prep_fidelity_array = np.array(prep_fidelity_list)
    # readout_fidelity_array = np.array(readout_fidelity_list)

    # # Calculate fidelity weights
    # fidelity_weights = prep_fidelity_array * readout_fidelity_array
    # fidelity_weights = np.nan_to_num(fidelity_weights, nan=0.0)  # Handle NaNs
    # Weight the states by fidelity
    # weighted_states = states * fidelity_weights[:, np.newaxis]
    # Generate timestamps

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

    # ax.vlines(
    #     78,
    #     0,
    #     max(hist_values),
    #     color="red",
    #     linestyle="--",
    #     label="Mean",
    # )

    # Add Poisson PMF for comparison
    # x_vals = np.arange(num_nvs + 1)
    # expected_dist = len(coincidences) * poisson.pmf(x_vals, np.mean(coincidences))
    # ax.plot(
    #     x_vals,
    #     expected_dist,
    #     marker="o",
    #     linestyle="--",
    #     label="Poisson PMF (Expected Dist.)",
    #     color="red",
    #     alpha=0.6,
    #     markersize=2,
    # )

    # Set y-axis limits dynamically based on data range
    # ax.set_yscale("log")
    ax.set_ylim(0, 4)

    # Add labels, title, and legend
    ax.set_xlabel("Number of NVs in NV⁰ State", fontsize=15)
    ax.set_ylabel("Number of occurrences", fontsize=15)
    ax.set_title("Charge Jump - Coincidence Histogram (Dark Time: 1ms)", fontsize=15)
    ax.legend(fontsize=12, loc="upper right")

    # Use logarithmic scale for y-axis
    # ax.set_yscale("log")
    ax.grid(True, which="both", linestyle="--", alpha=0.6)
    ax.tick_params(axis="both", which="major", labelsize=15)

    # 2. Time-Resolved Histogram
    time_resolved_fig, ax = plt.subplots()
    # time_bins = np.linspace(timestamps.min(), timestamps.max(), 6)  # 10 time intervals
    # for start, end in zip(time_bins[:-1], time_bins[1:]):
    #     mask = (timestamps >= start) & (timestamps < end)
    #     time_coincidences = coincidences[mask]
    #     hist_values, _ = np.histogram(time_coincidences, bins=bin_edges)
    #     ax.plot(
    #         bin_edges[:-1] + 0.5,
    #         hist_values,
    #         label=f"{start:.2f}s - {end:.2f}s",
    #         marker="o",
    #         markersize=2,
    #     )
    # ax.set_xlabel("Number of NVs affected (Coincidences)", fontsize=15)
    # ax.set_ylabel("Frequency", fontsize=15)
    # ax.set_title("Time-Resolved Coincidence Histogram", fontsize=15)
    # ax.legend(fontsize=10)
    # ax.grid(True, linestyle="--", alpha=0.6)
    # Filter points where coincidences > 80

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
        cmap="viridis",
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

    # Plot the time-dependent NV⁰ counts
    nv0_counts = states.shape[0] - states.sum(axis=0)
    time_series_fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the data
    ax.plot(
        nv0_counts,
        marker="o",
        markersize=2,
        linestyle="-",
        alpha=0.7,
        label="NV⁰ Count",
    )

    # Add labels, title, grid, and legend
    ax.set_xlabel("Number of Shots", fontsize=14)  # Corrected: Use ax.set_xlabel
    ax.set_ylabel("Number of NVs in NV⁰ State", fontsize=14)
    ax.set_title("Time-Series NV⁰ Count (dark time: 1ms)", fontsize=16)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(fontsize=12)

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()
    # Reshape coincidences to match the NV count (assuming 15,000 = num_shots x num_nvs)
    # Unpack relevant variables

    # Number of NVs
    num_nvs = len(nv_list)

    # Compute the sum across all NVs for each shot
    states = np.array(states_1).reshape((num_nvs, -1))
    shot_sums = num_nvs - states.sum(axis=0)

    # Find indices of shots where the sum is greater than 80
    high_activity_indices = np.where(shot_sums > 78)[0]  # Get 1D array of indices
    print(f"Selected shots: {len(high_activity_indices)}")

    # Extract only the selected shots
    selected_states = states[:, high_activity_indices]

    # Extract spatial coordinates
    x_coords = [nv.coords["pixel"][0] for nv in nv_list]
    y_coords = [nv.coords["pixel"][1] for nv in nv_list]

    # Compute activity counts for each NV
    high_activity_counts = selected_states.sum(
        axis=1
    )  # Sum over selected shots for each NV

    # Plot the spatial heatmap for high-activity NVs
    high_activity_map_fig, ax = plt.subplots()
    scatter = ax.scatter(
        x_coords,
        y_coords,
        c=high_activity_counts,
        cmap="viridis",
        edgecolors="black",
        s=60,
    )
    plt.colorbar(scatter, ax=ax, label="Activity Count (Sum)")
    ax.set_xlabel("X Coordinate (Pixels)", fontsize=15)
    ax.set_ylabel("Y Coordinate (Pixels)", fontsize=15)
    ax.set_title("NVs with High-Activity Shots in NV⁰ State (>78)", fontsize=15)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.show()

    print(f"High-activity shot indices: {high_activity_indices}")

    return (
        hist_fig,
        high_activity_map_fig,
        spatial_heatmap_fig,
        time_series_fig,
    )


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


def determine_and_threshold_fidelity_multi(
    data_files,
    prob_dist: ProbDist = ProbDist.COMPOUND_POISSON,
    prep_thresh=0.6,
    readout_thresh=0.8,
):
    """
    Plot comparison histograms with expected Poisson distribution for multiple datasets.
    Additionally, compute and plot skewness vs. dark time, filtering NVs based on fidelity.

    Parameters:
        data_files (list): List of data files containing coincidence data.
        dark_times (list): List of corresponding dark times for labeling.
        prep_thresh (float): Threshold for preparation fidelity.
        readout_thresh (float): Threshold for readout fidelity.
        prob_dist: Probability distribution for threshold determination (optional).
    """
    # Load the first dataset to initialize variables
    first_data = dm.get_raw_data(file_id=data_files[0])
    nv_list = first_data["nv_list"]
    num_nvs = len(nv_list)
    counts = np.array(first_data["counts"])[0]

    # Calculate fidelity for filtering
    prep_fidelity_list = []
    readout_fidelity_list = []
    threshold_list = []

    for nv_idx in range(num_nvs):
        for file_id in data_files[1:]:
            new_data = dm.get_raw_data(file_id=file_id)
            new_counts = np.array(new_data["counts"])[0]
            combined_counts = np.append(
                counts[nv_idx].flatten(), new_counts[nv_idx].flatten()
            )

        counts = np.array(first_data["counts"])[0]
        ref_counts = combined_counts[nv_idx]
        popt, _, _ = fit_bimodal_histogram(ref_counts, prob_dist, no_print=True)
        threshold, readout_fidelity = determine_threshold(
            popt, prob_dist, dark_mode_weight=0.5, do_print=False, ret_fidelity=True
        )
        prep_fidelity = 1 - popt[0] if popt else np.nan
        prep_fidelity_list.append(prep_fidelity)
        readout_fidelity_list.append(readout_fidelity)
        threshold_list.append(threshold)

    # Convert to arrays for filtering
    prep_fidelity_array = np.array(prep_fidelity_list)
    readout_fidelity_array = np.array(readout_fidelity_list)

    valid_indices = [
        idx
        for idx, (prep_fid, readout_fid) in enumerate(
            zip(prep_fidelity_array, readout_fidelity_array)
        )
        if prep_fid >= prep_thresh and readout_fid >= readout_thresh
    ]
    return (
        valid_indices,
        prep_fidelity_array[valid_indices],
        readout_fidelity_array[valid_indices],
    )


def plot_histogram_multi(data_files, dark_times):
    """
    Plot comparison histograms with expected Poisson distribution for multiple datasets.
    Additionally, compute and plot skewness vs. dark time.

    Parameters:
        data_files (list): List of data files containing coincidence data.
        dark_times (list): List of corresponding dark times for labeling.
    """
    # Filter NVs based on fidelity thresholds
    (filtered_indices, filtered_prep_fidelity, filtered_readout_fidelity) = (
        determine_and_threshold_fidelity_multi(
            data_files, prep_thresh=0.6, readout_thresh=0.8
        )
    )
    num_nvs_filtered = len(filtered_indices)
    print(f"Filtered NV count: {num_nvs_filtered}")

    skewness_values = []
    plt.figure(figsize=(10, 7))
    for data_file, dark_time in zip(data_files, dark_times):
        # Load the data
        data = dm.get_raw_data(file_id=data_file)
        nv_list = data["nv_list"]

        counts = np.array(data["counts"])[0]
        print(f"counts.shape= {counts.shape}")
        counts = counts[filtered_indices, :, :, :]

        nv_list = [nv_list[idx] for idx in filtered_indices]

        num_nvs = len(nv_list)
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


def inspect_raw_data(data, num_samples=10):
    """
    Inspect raw counts data before thresholding.

    Parameters:
        data (dict): Raw data dictionary containing NV counts and metadata.
        num_samples (int): Number of NVs to randomly sample for inspection.
    """
    nv_list = data["nv_list"]
    num_reps = data["num_reps"]
    num_runs = data["num_runs"]
    num_nvs = len(nv_list)
    counts = np.array(data["counts"])[0]  # Extract signal counts
    counts = np.array(counts).reshape((num_nvs, -1))
    print(f"Counts shape: {counts.shape}")

    # Generate timestamps for the dataset
    timestamps = generate_timestamps(
        num_reps=num_reps,
        num_runs=num_runs,
        dark_time_ns=10e6,
        deadtime_ns=100e6,
        readout_time_ns=50e6,
    )

    # Validate `num_samples` to avoid errors
    num_samples = min(num_samples, num_nvs)

    # Sample a subset of NVs for visualization
    sampled_indices = np.random.choice(num_nvs, num_samples, replace=False)
    sampled_counts = counts[sampled_indices]
    sampled_nv_names = [nv_list[i].name for i in sampled_indices]

    # Plot raw counts over time for sampled NVs
    plt.figure(figsize=(12, 8))
    for i, (nv_name, nv_counts) in enumerate(zip(sampled_nv_names, sampled_counts)):
        plt.plot(
            timestamps,
            nv_counts,
            label=f"NV {sampled_indices[i]}",
            alpha=0.7,
        )
    plt.xlabel("Time (s)", fontsize=14)
    plt.ylabel("Raw Counts", fontsize=14)
    plt.title("Raw Counts Over Time for Sampled NVs", fontsize=16)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()

    # Plot histograms of raw counts for sampled NVs
    fig, axes = plt.subplots(num_samples, 1, figsize=(10, 3 * num_samples), sharex=True)
    for i, (nv_name, nv_counts) in enumerate(zip(sampled_nv_names, sampled_counts)):
        axes[i].hist(nv_counts, bins=50, alpha=0.7, label=f"NV {sampled_indices[i]})")
        axes[i].set_ylabel("Frequency", fontsize=12)
        axes[i].legend(fontsize=10)
        axes[i].grid(True, linestyle="--", alpha=0.6)
    axes[-1].set_xlabel("Raw Counts", fontsize=14)
    fig.suptitle("Raw Counts Distribution for Sampled NVs", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


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
    data = dm.get_raw_data(file_id=1758336169797)  # dark time 1ms and 1s

    counts = np.array(data["counts"])  # Extract signal counts
    # counts = np.array(counts).reshape((num_nvs, -1))
    print(f"Counts shape: {counts.shape}")

    # Process data
    # (hist_fig, transition_fig, spatial_heatmap_fig, time_series_fig) = (
    #     process_detect_cosmic_rays(data)
    # )

    # data_files = [1756083081553, 1756161618282, 1756305080720, 1755068762133]
    # dark_times = [1e6, 10e6, 100e6, 1000e6]
    data_files = [1756161618282, 1756305080720, 1756699202091, 1755068762133]
    dark_times = [10e6, 100e6, 500e6, 1000e6]

    # DATA FILES
    file_ids = [1758180182062, 1758336169797]
    combined_data = dm.get_raw_data(file_id=file_ids[0])
    for file_id in file_ids[1:]:
        new_data = dm.get_raw_data(file_id=file_id)
        combined_data["num_runs"] += new_data["num_runs"]
        combined_data["counts"] = np.append(
            combined_data["counts"], new_data["counts"], axis=2
        )
    # Process data
    (hist_fig, transition_fig, spatial_heatmap_fig, time_series_fig) = (
        process_detect_cosmic_rays(combined_data)
    )
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
