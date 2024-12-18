# -*- coding: utf-8 -*-
"""
Created on Fall, 2024
@author: Saroj Chand
"""

import os
import sys
import time
import traceback
from datetime import datetime
from random import shuffle

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Circle
from pykrige import OrdinaryKriging

# from pykrige.ok import OrdinaryKriging
from scipy.interpolate import Rbf
from scipy.optimize import curve_fit, least_squares
from sklearn.cluster import KMeans

from majorroutines.pulsed_resonance import fit_resonance, norm_voigt, voigt, voigt_split
from majorroutines.widefield import base_routine
from utils import common
from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import positioning as pos
from utils import tool_belt as tb
from utils import widefield as widefield
from utils.constants import NVSig, NVSpinState
from utils.positioning import get_scan_1d as calculate_freqs


def voigt_with_background(
    freq, amp1, amp2, center1, center2, width, bg_offset, bg_slope
):
    """Voigt profile for two peaks with a linear background."""
    freq = np.array(freq)  # Ensure freq is a NumPy array for element-wise operations
    return (
        amp1 * norm_voigt(freq, width, width, center1)
        + amp2 * norm_voigt(freq, width, width, center2)
        + bg_offset
        + bg_slope * freq
    )


def residuals_fn(params, freq, nv_counts, nv_counts_ste):
    """Compute residuals for least_squares optimization."""
    amp1, amp2, center1, center2, width, bg_offset, bg_slope = params
    fit_vals = voigt_with_background(
        freq, amp1, amp2, center1, center2, width, bg_offset, bg_slope
    )
    return (nv_counts - fit_vals) / nv_counts_ste  # Weighted residuals


def calculate_contrast(signal, bg_offset):
    """Calculate contrast based on fit parameters and chi-squared value."""
    contrast = signal / bg_offset
    return contrast


def calculate_snr(residuals, nv_counts_ste, signal):
    """Calculate SNR based on the fitted Voigt profile and residuals."""
    noise = np.std(residuals / nv_counts_ste)
    return signal / noise if noise > 0 else 0


def cluster_and_print_average_center_frequencies(center_freqs):
    # Fit KMeans with the desired number of clusters (e.g., 4)
    kmeans = KMeans(n_clusters=4, random_state=42).fit(center_freqs)
    labels = kmeans.labels_

    # Dynamically initialize clusters dictionary based on unique labels
    unique_labels = np.unique(labels)
    clusters = {label: [] for label in unique_labels}

    # Populate clusters
    for idx, label in enumerate(labels):
        clusters[label].append(center_freqs[idx])

    # Print average lower and upper center frequencies and counts for each cluster
    for label, freqs in clusters.items():
        lower_freqs = [freq[0] for freq in freqs]
        upper_freqs = [freq[1] for freq in freqs]
        avg_lower_freq = np.mean(lower_freqs)
        avg_upper_freq = np.mean(upper_freqs)
        count = len(freqs)
        print(
            f"Cluster {label}: Count = {count}, Average Lower Frequency = {avg_lower_freq:.3f}, Average Upper Frequency = {avg_upper_freq:.2f}"
        )


def plot_nv_resonance_fits_and_residuals(
    nv_list,
    freqs,
    sig_counts,
    ref_counts,
    file_id,
    num_cols=6,
    threshold_method=None,
    chi_sq_threshold=0,
    contrast_threshold=0.05,
):
    """
    Plot NV resonance data with fitted Voigt profiles (including background), with residuals and contrast values.

    Args:
        nv_list: List of NV signatures.
        freqs: Frequency data.
        avg_counts: Averaged counts for NVs.
        avg_counts_ste: Standard error of averaged counts.
        file_id: ID for saving the file.
        num_cols: Number of columns for the grid layout.
        threshold_method: Optional method for thresholding.
        chi_sq_threshold: Threshold for filtering NVs based on chi-squared value.
        contrast_threshold: Threshold for filtering NVs based on contrast.
    """
    avg_counts, avg_counts_ste, norms = widefield.process_counts(
        nv_list, sig_counts, ref_counts, threshold=False
    )
    num_nvs = len(nv_list)
    chi_squared_list = []
    contrast_list = []
    center_freqs = []
    center_freq_differences = []
    avg_peak_widths = []
    avg_peak_amplitudes = []
    fit_data = []
    filtered_indices = []
    snrs = []
    for nv_idx in range(num_nvs):
        nv_counts = avg_counts[nv_idx]
        nv_counts_ste = avg_counts_ste[nv_idx]

        low_freq_guess = freqs[np.argmax(avg_counts[nv_idx][: len(freqs) // 2])]
        high_freq_guess = freqs[
            np.argmax(avg_counts[nv_idx][len(freqs) // 2 :]) + len(freqs) // 2
        ]
        max_amp = np.max(nv_counts)

        guess_params = [
            max_amp,
            max_amp,
            low_freq_guess,
            high_freq_guess,
            5,
            np.min(nv_counts),
            0,
        ]
        bounds = (
            [0, 0, min(freqs), min(freqs), 0, -np.inf, -np.inf],  # Lower bounds
            [
                np.inf,
                np.inf,
                max(freqs),
                max(freqs),
                np.inf,
                np.inf,
                np.inf,
            ],  # Upper bounds
        )

        result = least_squares(
            residuals_fn,
            guess_params,
            args=(freqs, nv_counts, nv_counts_ste),
            bounds=bounds,
            max_nfev=20000,
        )
        popt = result.x

        # Compute chi-squared value
        fit_data.append(voigt_with_background(freqs, *popt))
        residuals = nv_counts - fit_data[nv_idx]
        chi_squared = np.sum((residuals / nv_counts_ste) ** 2)
        chi_squared_list.append(chi_squared)

        # Calculate contrast
        amp1 = popt[0]
        amp2 = popt[1]
        bg_offset = popt[5]

        # Store center frequencies and frequency difference
        center_freqs.append((popt[2], popt[3]))
        center_freq_differences.append(abs(popt[3] - popt[2]))

        # Average peak widths and amplitudes
        avg_peak_widths.append((popt[4] + popt[4]) / 2)
        avg_peak_amplitude = (amp1 + amp2) / 2
        avg_peak_amplitudes.append(avg_peak_amplitude)
        # snr calculatin
        combined_counts = np.append(
            sig_counts[nv_idx].flatten(), ref_counts[nv_idx].flatten()
        )
        noise = np.std(combined_counts)
        snr = avg_peak_amplitude / noise
        # signal = avg_peak_amplitude
        # snr = calculate_snr(residuals, nv_counts_ste, signal)
        snrs.append(snr)
        print(f"NV {nv_idx}: SNR = {snr:.2f}")
        # Filter based on chi-squared and contrast thresholds
        # if chi_squared < chi_sq_threshold and contrast > contrast_threshold:
        # if chi_squared > chi_sq_threshold:
        #     filtered_indices.append(nv_idx)
    median_snr = np.median(snrs)
    print(f"median snr:{median_snr:.2f}")
    # Remove outliers from a data array using the IQR method.
    Q1 = np.percentile(snrs, 25)
    Q3 = np.percentile(snrs, 75)
    IQR = Q3 - Q1
    lower_bound = Q1
    upper_bound = Q3 + 1.5 * IQR
    print(f"lower bound:{lower_bound}")
    print(f"upper_bound:{upper_bound}")
    # Identify and filter out outliers
    cleaned_snrs = [val for val in snrs if lower_bound <= val <= upper_bound]
    outlier_indices = [
        i for i, val in enumerate(snrs) if val < lower_bound or val > upper_bound
    ]

    median_snr_cleaned = np.median(cleaned_snrs)
    cleaned_nv_indices = np.arange(len(cleaned_snrs))
    print(f"snrs IQR: {IQR}")
    print(f"median snr cleaned:{median_snr_cleaned}")
    print(f"Number of nvs removed : {len(outlier_indices)}")
    print(f"removed indices: {outlier_indices}")

    # Scatter plot
    plt.figure(figsize=(8, 5))
    plt.scatter(cleaned_nv_indices, cleaned_snrs, color="blue", alpha=0.6, label="SNR")

    # Add reference lines for median, Q1, and Q3
    plt.axhline(
        median_snr_cleaned,
        color="green",
        linestyle="--",
        label=f"Median SNR = {median_snr_cleaned:.3f}",
    )
    plt.axhline(
        Q1,
        color="orange",
        linestyle="--",
        label=f"Q1 (25th Percentile) = {Q1:.3f}",
    )
    plt.axhline(
        Q3,
        color="blue",
        linestyle="--",
        label=f"Q3 (75th Percentile) = {Q3:.3f}",
    )
    # Add labels and legend
    plt.title("SNR Across NV Centers (50ms Readout)")
    plt.xlabel("NV Index")
    plt.ylabel("SNR")
    plt.legend(loc="upper right")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()
    return
    # print(f"contrast_list:{contrast_list}")
    # print(f"hi_squared_list:{chi_squared_list}")
    # filtered_avg_peak_widths_1 = [avg_peak_widths[idx] for idx in filtered_indices]

    # filtered_indices = [
    #     idx for idx, value in enumerate(filtered_avg_peak_widths_1) if value > 1
    # ]
    # filtered_avg_peak_amplitudes = [
    #     avg_peak_amplitudes[idx] for idx in filtered_indices
    # ]
    # # Calculate IQR for filtered_avg_peak_widths
    # q1 = np.percentile(avg_peak_amplitudes, 25)  # 1st quartile
    # q3 = np.percentile(avg_peak_amplitudes, 75)  # 3rd quartile
    # iqr = q3 - q1  # Interquartile range

    # # Define bounds for outlier detection
    # lower_bound = q1 - 1.5 * iqr
    # upper_bound = q3 + 1.0 * iqr

    # # Filter indices based on the IQR bounds
    # filtered_indices = [
    #     idx
    #     for idx, value in enumerate(filtered_avg_peak_amplitudes)
    #     if lower_bound <= value <= upper_bound
    # ]
    # List of target peak values for filtering
    target_peak_values = [0.041, 0.069, 0.147, 0.175]
    # target_peak_values = [0.041, 0.147]
    tolerance = 0.01  # Set a tolerance for matching

    # Filter indices based on proximity to target peak differences with plus/minus bound
    filtered_indices = [
        idx
        for idx, freq_diff in enumerate(center_freq_differences)
        if any(
            target - tolerance <= freq_diff <= target + tolerance
            for target in target_peak_values
        )
    ]
    # Exclude manual indices from filtered indices
    manual_indices = [4, 9, 61, 81, 87, 119, 130, 144, 148]
    filtered_indices = [idx for idx in filtered_indices if idx not in manual_indices]
    # Find indices that do not match the criteria
    non_matching_indices = [
        idx
        for idx in range(len(center_freq_differences))
        if idx not in filtered_indices
    ]
    for idx in non_matching_indices:
        print(f"NV: {idx}")

    print(f"filtered_indices:{len(filtered_indices)}")
    print(f"filtered_indices:{len(filtered_indices)}")
    print(f"filtered_indices:{len(non_matching_indices)}")

    # Initialize a dictionary to store indices for each orientation
    orientation_indices = {value: [] for value in target_peak_values}

    # Filter and group indices based on proximity to target peak differences
    for idx, freq_diff in enumerate(center_freq_differences):
        for target in target_peak_values:
            if target - tolerance <= freq_diff <= target + tolerance:
                orientation_indices[target].append(idx)

    # Find non-matching indices
    # filtered_indices = [
    #     idx for indices in orientation_indices.values() for idx in indices
    # ]
    # non_matching_indices = [
    #     idx
    #     for idx in range(len(center_freq_differences))
    #     if idx not in filtered_indices
    # ]

    # # Combine all results into a single dictionary
    # results = {
    #     "orientation_indices": {
    #         target: {
    #             "nv_indices": indices,
    #             "count": len(indices),
    #         }
    #         for target, indices in orientation_indices.items()
    #     },
    #     "non_matching_indices": {
    #         "nv_indices": non_matching_indices,
    #         "count": len(non_matching_indices),
    #     },
    # }

    # Save the combined results to a single file
    file_name = dm.get_file_name(file_id=file_id)
    file_path = dm.get_file_path(__file__, file_name, f"{file_id}_all_results")
    # dm.save_raw_data(results, file_path)
    # Print the NV indices for each orientation
    for orientation, indices in orientation_indices.items():
        print(f"Orientation centered around {orientation} GHz:")
        print(f"NV Indices: {indices}")
        print(f"Number of NVs: {len(indices)}\n")

    # Filter NVs for plotting
    filtered_nv_list = [nv_list[idx] for idx in filtered_indices]
    filtered_avg_counts = [avg_counts[idx] for idx in filtered_indices]
    filtered_avg_counts_ste = [avg_counts_ste[idx] for idx in filtered_indices]
    filtered_center_freqs = [center_freqs[idx] for idx in filtered_indices]
    filtered_freq_differences = [
        center_freq_differences[idx] for idx in filtered_indices
    ]
    filtered_avg_peak_widths = [avg_peak_widths[idx] for idx in filtered_indices]
    filtered_avg_peak_amplitudes = [
        avg_peak_amplitudes[idx] for idx in filtered_indices
    ]
    filtered_contrast_list = [contrast_list[idx] for idx in filtered_indices]
    filtered_chi_squared_list = [chi_squared_list[idx] for idx in filtered_indices]
    filtered_fitted_data = [fit_data[idx] for idx in filtered_indices]

    # Print
    # cluster_and_print_average_center_frequencies(filtered_center_freqs)
    # Plot filtered resonance fits
    sns.set(style="whitegrid", palette="muted")
    num_filtered_nvs = len(filtered_nv_list)
    colors = sns.color_palette("deep", num_filtered_nvs)
    num_rows = int(np.ceil(num_filtered_nvs / num_cols))
    fig_fitting, axes_fitting = plt.subplots(
        num_rows,
        num_cols,
        figsize=(num_cols * 3, num_rows * 1),
        sharex=True,
        sharey=False,
    )
    axes_fitting = axes_fitting.flatten()

    for nv_idx, ax in enumerate(axes_fitting):
        if nv_idx < num_filtered_nvs:
            sns.lineplot(
                x=freqs,
                y=filtered_avg_counts[nv_idx],
                ax=ax,
                color=colors[nv_idx % len(colors)],
                lw=2,
                marker="o",
                markersize=5,
                label=f"{filtered_indices[nv_idx]+1}",
            )
            ax.legend(fontsize="small")
            ax.errorbar(
                freqs,
                filtered_avg_counts[nv_idx],
                yerr=filtered_avg_counts_ste[nv_idx],
                fmt="none",
                ecolor="gray",
                alpha=0.6,
            )
            ax.grid(True, which="both", linestyle="--", linewidth=0.5)
            # Plot fitted data on the same subplot
            ax.plot(
                freqs,
                filtered_fitted_data[nv_idx],
                "-",
                color=colors[nv_idx % len(colors)],
                label="Fit",
                lw=2,
            )
            # Y-tick labels for the leftmost column
            # if nv_idx % num_cols == 0:
            #     ax.set_yticks(ax.get_yticks())
            # else:
            #     ax.set_yticklabels([])
            ax.set_yticklabels([])

            fig_fitting.text(
                0.1,
                0.5,
                "NV$^{-}$ Population",
                va="center",
                rotation="vertical",
                fontsize=12,
            )
            # Set custom tick locations in x axis
            if nv_idx >= (num_rows - 1) * num_cols:  # Bottom row
                ax.set_xlabel("Frequency (GHz)")
                ax.set_xticks(np.linspace(min(freqs), max(freqs), 5))
            for col in range(num_cols):
                bottom_row_idx = num_rows * num_cols - num_cols + col
                if bottom_row_idx < len(axes_fitting):
                    ax = axes_fitting[bottom_row_idx]
                    tick_positions = np.linspace(min(freqs), max(freqs), 5)
                    ax.set_xticks(tick_positions)
                    ax.set_xticklabels(
                        [f"{tick:.2f}" for tick in tick_positions],
                        rotation=45,
                        fontsize=9,
                    )
                    ax.set_xlabel("Frequency (GHz)")
                else:
                    ax.set_xticklabels([])

            ax.grid(True, which="both", linestyle="--", linewidth=0.5)
        else:
            ax.axis("off")
    fig_fitting.suptitle(
        f"NV Resonance Fits (Readout: 30ms) file id {file_id}", fontsize=16
    )
    plt.subplots_adjust(
        left=0.1, right=0.95, top=0.95, bottom=0.1, hspace=0.01, wspace=0.01
    )
    # plt.tight_layout()
    now = datetime.now()
    date_time_str = now.strftime("%Y%m%d_%H%M%S")
    file_name = dm.get_file_name(file_id=file_id)
    file_path = dm.get_file_path(__file__, file_name, f"{file_id}_{date_time_str}")
    kpl.show(block=True)
    # dm.save_figure(fig_fitting, file_path)
    # plt.close(fig_fitting)

    # Plot histograms and scatter plots
    plots_data = [
        (
            "Histogram of Frequency Splitting",
            filtered_freq_differences,
            None,
            "Frequency Splitting (GHz)",
            "Count",
            "teal",
        ),
        (
            "Frequency Splitting vs Contrast",
            filtered_freq_differences,
            filtered_contrast_list,
            "Frequency Splitting (GHz)",
            "Contrast",
            "purple",
        ),
        (
            "Frequency Splitting vs Average Peak Width",
            filtered_freq_differences,
            filtered_avg_peak_widths,
            "Frequency Splitting (GHz)",
            "Average Peak Width",
            "orange",
        ),
        (
            "Frequency Splitting vs Average Peak Amplitude",
            filtered_freq_differences,
            filtered_avg_peak_amplitudes,
            "Frequency Splitting (GHz)",
            "Average Peak Amplitude",
            "green",
        ),
        (
            "NV Index vs Frequency Difference",
            list(range(len(filtered_freq_differences))),  # NV indices
            filtered_freq_differences,
            "NV Index",
            "Frequency Difference (GHz)",
            "blue",
        ),
    ]

    for title, x_data, y_data, xlabel, ylabel, color in plots_data:
        fig = plt.figure(figsize=(6, 5))
        if isinstance(y_data, list):
            plt.scatter(x_data, y_data, color=color, alpha=0.7, edgecolors="k")
        else:
            plt.hist(x_data, bins=5, color=color, alpha=0.7, edgecolor="black")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True, linestyle="--", alpha=0.6)
        file_path = dm.get_file_path(
            __file__, file_name, f"{file_id}_{date_time_str}_{title}"
        )
        kpl.show(block=True)
        # dm.save_figure(fig, file_path)
        # plt.close(fig)

    # plot_list = ["chi_squared_]
    # Plot histograms and scatter plots
    # fig_freq_diff = plt.figure(figsize=(6, 5))
    # plt.hist(
    #     filtered_freq_differences, bins=30, color="purple", alpha=0.7, edgecolor="black"
    # )
    # plt.xlabel("Frequency Splitting (GHz)")
    # plt.ylabel("Count")
    # plt.title("Histogram of Frequency Splitting")
    # plt.show(block=False)
    # file_path = dm.get_file_path(
    #     __file__, file_name, f"{file_id}_{date_time_str}_freq_splitting"
    # )
    # dm.save_figure(fig_freq_diff, file_path)
    # plt.close(fig_freq_diff)

    # Plot histograms and scatter plots
    # fig = plt.figure(figsize=(8, 6))
    # plt.scatter(
    #     filtered_freq_differences,
    #     filtered_contrast_list,
    #     color="purple",
    #     alpha=0.7,
    #     edgecolors="k",
    # )
    # plt.xlabel("Frequency Splitting (GHz)")
    # plt.ylabel("Contrast")
    # plt.title("Frequency Splitting vs Contrast")
    # plt.grid(True, linestyle="--", alpha=0.6)
    # file_path = dm.get_file_path(
    #     __file__, file_name, f"{file_id}_{date_time_str}_freq_splitting_vs_contrast"
    # )
    # dm.save_figure(fig, file_path)
    # plt.close(fig)

    # # Plot  scatter plots
    # fig = plt.figure(figsize=(8, 6))
    # plt.scatter(
    #     filtered_freq_differences,
    #     filtered_avg_peak_widths,
    #     color="orange",
    #     alpha=0.7,
    #     edgecolors="k",
    # )
    # plt.xlabel("Frequency Splitting (GHz)")
    # plt.ylabel("Average Peak Width")
    # plt.title("Frequency Splitting vs Average Peak Width")
    # plt.grid(True, linestyle="--", alpha=0.6)
    # file_path = dm.get_file_path(
    #     __file__,
    #     file_name,
    #     f"{file_id}_{date_time_str}_freq_splitting_vs_avg_peak_width",
    # )
    # dm.save_figure(fig, file_path)
    # plt.close(fig)

    # # Plot  scatter plots
    # fig = plt.figure(figsize=(8, 6))
    # plt.scatter(
    #     filtered_freq_differences,
    #     filtered_avg_peak_amplitudes,
    #     color="green",
    #     alpha=0.7,
    #     edgecolors="k",
    # )
    # plt.xlabel("Frequency Splitting (GHz)")
    # plt.ylabel("Average Peak Amplitude")
    # plt.title("Frequency Splitting vs Average Peak Amplitude")
    # plt.grid(True, linestyle="--", alpha=0.6)
    # file_path = dm.get_file_path(
    #     __file__, file_name, f"{file_id}_{date_time_str}_freq_splitting_vs_avg_amp"
    # )
    # dm.save_figure(fig, file_path)
    # plt.close(fig)
    # plt.show(block=False)


# def plot_nv_resonance_fits_and_residuals(
#     nv_list, freqs, sig_counts, ref_counts, file_id, num_cols=3, threshold_method=None
# ):
#     """
#     Plot NV resonance data with fitted Voigt profiles (including background), with residuals and contrast values.

#     Args:
#         nv_list: List of NV signatures.
#         freqs: Frequency data.
#         avg_counts: Averaged counts for NVs.
#         avg_counts_ste: Standard error of averaged counts.
#         file_id: ID for saving the file.
#         num_cols: Number of columns for the grid layout.
#         threshold_method: Optional method for thresholding.
#     """
#     avg_counts, avg_counts_ste, norms = widefield.process_counts(
#         nv_list, sig_counts, ref_counts, threshold=False
#     )
#     # # Normalize counts from 0 to 1
#     # avg_counts = [(ac) / (max(ac)) for ac in avg_counts]

#     # # Indices to remove
#     # indices_to_remove = {9, 26, 33, 35, 36, 45, 47, 53, 56, 58, 59, 60, 61, 62, 63}

#     # # Filter nv_list, avg_counts, and avg_counts_ste based on indices to keep
#     # nv_list = [nv for idx, nv in enumerate(nv_list) if idx not in indices_to_remove]
#     # avg_counts = [
#     #     ac for idx, ac in enumerate(avg_counts) if idx not in indices_to_remove
#     # ]
#     # avg_counts_ste = [
#     #     ste for idx, ste in enumerate(avg_counts_ste) if idx not in indices_to_remove
#     # ]

#     # Update num_nvs and re-calculate number of rows
#     num_nvs = len(nv_list)
#     num_rows = int(np.ceil(num_nvs / num_cols))
#     sns.set(style="whitegrid", palette="muted")
#     colors = sns.color_palette("deep", num_nvs)

#     # Initialize fitting variables
#     num_nvs = len(nv_list)
#     chi_squared_list = []
#     contrast_list = []
#     center_freqs = []
#     center_freq_differences = []
#     avg_peak_widths = []
#     avg_peak_amplitudes = []

#     filtered_indices = []

#     ### 1. Fitting and Contrast Figure ###
#     fig_fitting, axes_fitting = plt.subplots(
#         num_rows,
#         num_cols,
#         figsize=(num_cols * 3, num_rows * 1),
#         sharex=True,
#         sharey=False,
#     )
#     axes_fitting = axes_fitting.flatten()

#     for nv_idx, ax in enumerate(axes_fitting):
#         if nv_idx < num_nvs:
#             sns.lineplot(
#                 x=freqs,
#                 y=avg_counts[nv_idx],
#                 ax=ax,
#                 color=colors[nv_idx % len(colors)],
#                 lw=2,
#                 marker="o",
#                 markersize=5,
#                 label=f"NV {nv_idx+1}",
#             )
#             ax.errorbar(
#                 freqs,
#                 avg_counts[nv_idx],
#                 yerr=avg_counts_ste[nv_idx],
#                 fmt="none",
#                 ecolor="gray",
#                 alpha=0.6,
#             )

#             nv_counts = avg_counts[nv_idx]
#             nv_counts_ste = avg_counts_ste[nv_idx]

#             # Initial guess: amplitudes, centers, shared width, and background (linear)
#             low_freq_guess = freqs[np.argmax(avg_counts[nv_idx][: len(freqs) // 2])]
#             high_freq_guess = freqs[
#                 np.argmax(avg_counts[nv_idx][len(freqs) // 2 :]) + len(freqs) // 2
#             ]
#             max_amp = np.max(nv_counts)

#             guess_params = [
#                 max_amp,
#                 max_amp,
#                 low_freq_guess,
#                 high_freq_guess,
#                 5,
#                 np.min(nv_counts),
#                 0,
#             ]
#             bounds = (
#                 [0, 0, min(freqs), min(freqs), 0, -np.inf, -np.inf],  # Lower bounds
#                 [
#                     np.inf,
#                     np.inf,
#                     max(freqs),
#                     max(freqs),
#                     np.inf,
#                     np.inf,
#                     np.inf,
#                 ],  # Upper bounds
#             )

#             # Visualize initial guess
#             # fit_guess = voigt_with_background(freqs, *guess_params)
#             # ax.plot(freqs, fit_guess, '--', label="Initial Guess", color='gray')

#             # Perform the least_squares fitting
#             result = least_squares(
#                 residuals_fn,
#                 guess_params,
#                 args=(freqs, nv_counts, nv_counts_ste),
#                 bounds=bounds,
#                 max_nfev=20000,
#             )
#             popt = result.x
#             # Store center frequencies (two peaks for each NV)
#             center_freqs.append((popt[2], popt[3]))
#             # Track parameters for plotting and residuals
#             fit_fns.append(voigt_with_background)
#             popts.append(popt)
#             # Plot fitted data on the same subplot
#             fit_data = voigt_with_background(freqs, *popt)
#             ax.plot(
#                 freqs,
#                 fit_data,
#                 "-",
#                 color=colors[nv_idx % len(colors)],
#                 label="Fit",
#                 lw=2,
#             )
#             residuals = nv_counts - fit_data
#             residuals_list.append(residuals)

#             # Compute chi-squared value
#             chi_squared = np.sum((residuals / nv_counts_ste) ** 2)
#             chi_squared_list.append(chi_squared)

#             # Calculate contrast
#             amp1 = popt[0]  # First peak amplitude
#             amp2 = popt[1]  # Second peak amplitude
#             bg_offset = popt[5]  # Background level (constant offset)
#             # contrast = calculate_contrast(amp1, amp2, bg_offset)
#             contrast = calculate_contrast(amp1, amp2, bg_offset, chi_squared)
#             # snr = calculate_snr(sig_counts, ref_counts)
#             # contrast,_ = widefield.calc_contrast(sig_counts, ref_counts)
#             # contrast, contrast_ste = widefield.calc_contrast(
#             # sig_counts[nv_idx][np.newaxis, ...],
#             # ref_counts[nv_idx][np.newaxis, ...]
#             # )
#             # snr, snr_ste = widefield.calc_snr(
#             # sig_counts[nv_idx][np.newaxis, ...],
#             # ref_counts[nv_idx][np.newaxis, ...]
#             # )
#             # Ensure contrast is a scalar value
#             # if isinstance(contrast, np.ndarray):
#             #     contrast = np.mean(contrast)

#             # contrast, contrast_ste = widefield.calc_contrast(sig_counts[nv_idx], ref_counts[nv_idx])
#             # Add contrast to the plot
#             # ax.text(
#             #     0.05,
#             #     0.85,
#             #     f"Contrast: {contrast:.2f}",
#             #     transform=ax.transAxes,
#             #     fontsize=10,
#             #     bbox=dict(facecolor="white", alpha=0.6),
#             # )

#             # Y-tick labels for the leftmost column
#             if nv_idx % num_cols == 0:
#                 ax.set_yticks(ax.get_yticks())
#             else:
#                 ax.set_yticklabels([])

#             fig_fitting.text(
#                 0.04,
#                 0.5,
#                 "NV$^{-}$ Population",
#                 va="center",
#                 rotation="vertical",
#                 fontsize=12,
#             )

#             # if nv_idx >= (num_rows - 1) * num_cols:  # Bottom row
#             #     ax.set_xlabel("Frequency (GHz)")
#             #     ax.set_xticks(np.linspace(min(freqs), max(freqs), 5))  # Set custom tick locations
#             # Add frequency values at the bottom of each column
#             for col in range(num_cols):
#                 bottom_row_idx = (
#                     num_rows * num_cols - num_cols + col
#                 )  # Index of the bottom row in each column
#                 if bottom_row_idx < len(
#                     axes_fitting
#                 ):  # Ensure the index is within bounds
#                     ax = axes_fitting[bottom_row_idx]
#                     tick_positions = np.linspace(min(freqs), max(freqs), 5)
#                     ax.set_xticks(tick_positions)
#                     ax.set_xticklabels(
#                         [f"{tick:.2f}" for tick in tick_positions],
#                         rotation=45,
#                         fontsize=9,
#                     )
#                     ax.set_xlabel("Frequency (GHz)")
#                 else:
#                     ax.set_xticklabels([])

#             ax.grid(True, which="both", linestyle="--", linewidth=0.5)
#         else:
#             ax.axis("off")

#     if threshold_method is not None:
#         title = f"NV Resonance Fitting and Contrast (data_id = {file_id})"
#         fig_fitting.suptitle(title, fontsize=16, y=0.97)

#     plt.subplots_adjust(
#         left=0.1, right=0.95, top=0.95, bottom=0.1, hspace=0.01, wspace=0.01
#     )

#     now = datetime.now()
#     date_time_str = now.strftime("%Y%m%d_%H%M%S")
#     file_name = dm.get_file_name(file_id=file_id)
#     file_path = dm.get_file_path(__file__, file_name, f"{file_id}_{date_time_str}")
#     dm.save_figure(fig_fitting, file_path)
#     plt.close(fig_fitting)


### 3. Plot  ###


def plot_selected_nv_resonance_fits_comparison(
    nv_list, freqs, sig_counts, ref_counts, file_id, selected_nv_indices
):
    """
    Plot resonance data and fitted Voigt profiles for selected NVs in a single column for comparison,
    focusing on peak differences.

    Args:
        nv_list: List of NV signatures.
        freqs: Frequency data.
        sig_counts: Signal counts for NVs.
        ref_counts: Reference counts for NVs.
        file_id: ID for saving the file.
        selected_nv_indices: List of indices of NVs to plot.
    """
    # Filter NVs based on selected indices
    nv_list = [nv_list[i] for i in selected_nv_indices]
    avg_counts, avg_counts_ste, norms = widefield.process_counts(
        nv_list, sig_counts, ref_counts, threshold=False
    )

    num_nvs = len(nv_list)
    sns.set(style="whitegrid", palette="muted")
    colors = sns.color_palette("deep", num_nvs)

    fig, axes = plt.subplots(
        num_nvs, 1, figsize=(6, 3 * num_nvs), sharex=True, sharey=False
    )

    if num_nvs == 1:
        axes = [axes]  # Ensure axes is always iterable

    fit_fns = []
    popts = []
    center_freqs_all = []

    for plot_idx, nv_idx in enumerate(selected_nv_indices):
        ax = axes[plot_idx]

        sns.lineplot(
            x=freqs,
            y=avg_counts[plot_idx],
            ax=ax,
            color=colors[plot_idx % len(colors)],
            lw=2,
            marker="o",
            markersize=5,
            label=f"NV {nv_idx+1}",
        )
        ax.errorbar(
            freqs,
            avg_counts[plot_idx],
            yerr=avg_counts_ste[plot_idx],
            fmt="none",
            ecolor="gray",
            alpha=0.6,
        )

        nv_counts = avg_counts[plot_idx]
        nv_counts_ste = avg_counts_ste[plot_idx]

        # Initial guesses and fitting logic
        low_freq_guess = freqs[np.argmax(avg_counts[plot_idx][: len(freqs) // 2])]
        high_freq_guess = freqs[
            np.argmax(avg_counts[plot_idx][len(freqs) // 2 :]) + len(freqs) // 2
        ]
        max_amp = np.max(nv_counts)
        guess_params = [
            max_amp,
            max_amp,
            low_freq_guess,
            high_freq_guess,
            5,
            np.min(nv_counts),
            0,
        ]
        bounds = (
            [0, 0, min(freqs), min(freqs), 0, -np.inf, -np.inf],
            [np.inf, np.inf, max(freqs), max(freqs), np.inf, np.inf, np.inf],
        )

        result = least_squares(
            residuals_fn,
            guess_params,
            args=(freqs, nv_counts, nv_counts_ste),
            bounds=bounds,
            max_nfev=20000,
        )
        popt = result.x
        center_freqs_all.append([popt[2], popt[3]])
        fit_fns.append(voigt_with_background)
        popts.append(popt)

        fit_data = voigt_with_background(freqs, *popt)
        ax.plot(
            freqs,
            fit_data,
            "-",
            color=colors[plot_idx % len(colors)],
            label="Fit",
            lw=2,
        )

        ax.set_title(f"NV {nv_idx + 1}")
        ax.set_ylabel("NV$^{-}$ Population")
        ax.grid(True, which="both", linestyle="--", linewidth=0.5)

        peak_diff = abs(popt[3] - popt[2])
        print(
            f"NV {nv_idx + 1} fitted peaks: {popt[2]:.2f} GHz, {popt[3]:.2f} GHz (Difference: {peak_diff:.3f} GHz)"
        )

    axes[-1].set_xlabel("Frequency (GHz)")
    fig.suptitle(
        f"Selected NV Resonance Fits Comparison (data_id = {file_id})", fontsize=16
    )
    plt.tight_layout()
    plt.show()


def estimate_magnetic_field_direction(
    field_splitting, gyromagnetic_ratio=28.0, threshold=0.06
):
    """
    Estimate the direction of the magnetic field based on resonance frequency splitting in NV centers,
    and return the relative magnetic field direction in degrees for small, medium, and large splitting cases.

    Args:
        field_splitting: List of frequency splitting values.
        threshold1: First threshold to classify small splitting (default: 0.12 GHz).
        threshold2: Second threshold to classify medium splitting (default: 0.6 GHz).

    Returns:
        result: A dictionary containing the magnetic field directions in degrees for small, medium, and large splitting cases.
    """
    # Separate small, medium, and large splitting cases
    small_splitting_nv = [split for split in field_splitting if split <= threshold]
    # medium_splitting_nv = [split for split in field_splitting if threshold1 < split <= threshold2]
    large_splitting_nv = [split for split in field_splitting if split > threshold]

    # Compute average splittings for small, medium, and large splitting cases
    avg_small_split = np.mean(small_splitting_nv) if small_splitting_nv else 0
    # avg_medium_split = np.mean(medium_splitting_nv) if medium_splitting_nv else 0
    avg_large_split = np.mean(large_splitting_nv) if large_splitting_nv else 0

    # Known NV orientation vectors in the diamond lattice (for 3 NV orientations)
    # nv_orientations = np.array([[-1, 1, 1],[1, 1, 1]]) / np.sqrt(3) #no good
    # nv_orientations = np.array([[1, -1, 1], [1, 1, 1]]) / np.sqrt(3) #no good
    # nv_orientations = np.array([[1, 1, -1],[1, 1, 1]]) / np.sqrt(3) #no good
    # nv_orientations = np.array([[1, -1, 1],[-1, 1, 1]]) / np.sqrt(3) #good
    # nv_orientations = np.array([[-1, 1, 1], [1, 1, -1]]) / np.sqrt(3) #good
    nv_orientations = np.array([[1, -1, 1], [1, 1, -1]]) / np.sqrt(3)
    # Initialize a result dictionary
    avg_splitting = np.array([avg_small_split, avg_large_split])
    # Convert splitting into an array and scale by gyromagnetic ratio
    B_proj = avg_splitting / gyromagnetic_ratio  # Magnetic field projections in Tesla

    # Solve the system of linear equations to estimate the magnetic field components
    B_components, _, _, _ = np.linalg.lstsq(nv_orientations, B_proj, rcond=None)

    # Calculate the magnitude of the magnetic field
    B_magnitude = np.linalg.norm(B_components)

    # Compute the angle between the magnetic field vector and each NV orientation
    B_direction_deg = []
    for nv_orientation in nv_orientations:
        cos_theta = np.dot(B_components, nv_orientation) / B_magnitude
        theta_deg = np.degrees(np.arccos(cos_theta))
        B_direction_deg.append(theta_deg)

    # Return the magnetic field direction in degrees and the magnitude of the magnetic field
    print(B_direction_deg)
    return B_direction_deg


def calculate_magnetic_fields(
    nv_list,
    field_splitting,
    zero_field_splitting=0.0,
    gyromagnetic_ratio=28.0,
    threshold=0.09,
):
    """
    Calculate magnetic fields for each NV center based on frequency splitting and adjust based on the magnetic field direction.

    Args:
        nv_list: List of NV center identifiers.
        field_splitting: List of frequency splitting values corresponding to the NV centers.
        zero_field_splitting: Zero-field splitting (D) in GHz.
        gyromagnetic_ratio: Gyromagnetic ratio (28 GHz/T).
        threshold1: First threshold to classify small splitting (default: 0.06 GHz).
        threshold2: Second threshold to classify medium splitting (default: 0.12 GHz).

    Returns:
        result: A list of magnetic field values for each NV center, in the same order as nv_list.
    """
    # Initialize a list to store the magnetic field values, maintaining order
    magnetic_fields = []

    # Get magnetic field directions for small, medium, and large splitting cases
    magnetic_field_directions = estimate_magnetic_field_direction(
        field_splitting, threshold
    )
    # Extract angles for each category
    theta_deg_small = magnetic_field_directions[0]
    # theta_deg_medium = magnetic_field_directions[1]
    theta_deg_large = magnetic_field_directions[1]
    # Iterate over each NV center and its corresponding frequency splitting, maintaining the order
    for split in field_splitting:
        if split > threshold:
            # Large splitting (orientation 3)
            B_3 = (split - zero_field_splitting) / gyromagnetic_ratio
            B_3 = abs(
                B_3 / np.cos(np.deg2rad(theta_deg_large))
            )  # Adjust by direction angle for large splitting
            magnetic_fields.append(B_3)
        # elif threshold1 < split <= threshold2:
        #     # Medium splitting (orientation 2)
        #     B_2 = (split - zero_field_splitting) / gyromagnetic_ratio
        #     B_2 = abs(B_2 / np.cos(np.deg2rad(theta_deg_medium)))  # Adjust by direction angle for medium splitting
        #     magnetic_fields.append(B_2)
        else:
            # Small splitting (orientation 1)
            B_1 = (split - zero_field_splitting) / gyromagnetic_ratio
            B_1 = abs(
                B_1 / np.cos(np.deg2rad(theta_deg_small))
            )  # Adjust by direction angle for small splitting
            magnetic_fields.append(B_1)

    # Return the magnetic fields in the same order as the input NV list
    return magnetic_fields


def estimate_magnetic_field_from_fitting(
    nv_list,
    field_splitting,
    zero_field_splitting=2.87,
    gyromagnetic_ratio=28.0,
    threshold=0.05,
):
    """
    Estimate magnetic fields at each NV based on resonance frequency splitting.

    Args:
        nv_list: List of NV signatures.
        field_splitting: Frequency splitting values.
        zero_field_splitting: Zero-field splitting (D) in GHz.
        gyromagnetic_ratio: Gyromagnetic ratio (28 GHz/T).
        threshold: Threshold to classify splitting into orientations.

    Returns:
        magnetic_fields: Magnetic fields for each NV, reordered by NV index.
    """
    # Initialize a list to store magnetic fields
    magnetic_fields = []

    # Iterate over each NV and its frequency splitting
    for nv_idx, split in enumerate(field_splitting):
        if split > threshold:
            # Large splitting (orientation 2)
            B_2 = (split - zero_field_splitting) / gyromagnetic_ratio
            # B_2 = abs(B_2 * np.cos(np.deg2rad(109.47)))
            magnetic_fields.append(abs(B_2))
        else:
            # Small splitting (orientation 1)
            B_1 = (split - zero_field_splitting) / gyromagnetic_ratio
            B_1 = abs(B_1 / np.cos(np.deg2rad(109.47)))  # Adjust by angle if needed
            magnetic_fields.append(B_1)

    return magnetic_fields


def remove_outliers(B_values, nv_list):
    """
    Remove outliers using the IQR (Interquartile Range) method and corresponding NV centers.

    Args:
        B_values: Array of magnetic field values.
        nv_list: List of NV centers.

    Returns:
        Cleaned B_values, cleaned nv_list with outliers removed.
    """
    # Calculate Q1 (25th percentile) and Q3 (75th percentile)
    Q1 = np.percentile(B_values, 25)
    Q3 = np.percentile(B_values, 75)

    # Calculate IQR
    IQR = Q3 - Q1

    # Define outlier bounds
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Filter out the outliers in B_values and corresponding nv_list
    mask = (B_values >= lower_bound) & (B_values <= upper_bound)

    # Return cleaned data
    return B_values[mask], [nv for i, nv in enumerate(nv_list) if mask[i]]


def generate_2d_magnetic_field_map_kriging(
    nv_list, magnetic_fields, dist_conversion_factor, grid_size=100
):
    """
    Generate a 2D map of the magnetic field using Kriging interpolation.

    Args:
        nv_list: List of NV centers, each having 'pixel_coords' attributes.
        magnetic_fields: Calculated magnetic fields for each NV.
        dist_conversion_factor: Conversion factor from pixels to real-world distance (e.g., micrometers per pixel).
        grid_size: Size of the output grid (resolution of the 2D map).

    Returns:
        X, Y: Coordinates of the grid.
        Z: Interpolated magnetic field values over the grid.
    """
    # Convert magnetic fields from Tesla to Gauss
    B_values = (
        np.array(magnetic_fields) * 1e4
    )  # Convert Tesla to Gauss (1 Tesla = 10,000 Gauss)
    # Remove outliers and corresponding NV centers
    B_values, nv_list = remove_outliers(B_values, nv_list)
    # Extract NV positions (convert pixel coordinates to real-world distance)
    x_coords = (
        np.array([nv.coords["pixel"][0] for nv in nv_list]) * dist_conversion_factor
    )
    y_coords = (
        np.array([nv.coords["pixel"][1] for nv in nv_list]) * dist_conversion_factor
    )

    # Create a grid for interpolation
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    X_grid, Y_grid = np.meshgrid(
        np.linspace(x_min, x_max, grid_size), np.linspace(y_min, y_max, grid_size)
    )

    # Perform Kriging interpolation
    kriging_interpolator = OrdinaryKriging(
        x_coords, y_coords, B_values, variogram_model="linear"
    )
    Z_grid, _ = kriging_interpolator.execute(
        "grid",
        np.linspace(x_min, x_max, grid_size),
        np.linspace(y_min, y_max, grid_size),
    )
    # Plot the 2D magnetic field map using matplotlib
    plt.figure(figsize=(8, 6))
    contour = plt.contourf(X_grid, Y_grid, Z_grid, levels=100, cmap="plasma")
    plt.colorbar(contour, label="Magnetic Field (G)")

    # for x, y in zip(x_coords, y_coords):
    #     circle = Circle((x, y), radius=3, facecolor=None, edgecolor="lightblue")
    #     ax.add_patch(circle)
    # Scatter the NV positions and label their magnetic field values
    # plt.scatter(x_coords, y_coords, edgecolor='lightblue', s=50)
    plt.scatter(
        x_coords, y_coords, facecolor="none", edgecolor="lightblue", s=30, linewidth=1.0
    )
    # plt.colorbar(scatter, label='Magnetic Field (G)')

    # for i, (x, y, b) in enumerate(zip(x_coords, y_coords, B_values)):
    #     plt.text(x, y, f'{b:.2f} G', fontsize=8, color='white', ha='center', va='center')

    plt.title("2D Magnetic Field Map (Kriging Interpolation)")
    plt.xlabel("X Position (µm)")
    plt.ylabel("Y Position (µm)")
    plt.xticks(np.linspace(x_min, x_max, 5))
    plt.yticks(np.linspace(y_min, y_max, 5))
    plt.show()

    return X_grid, Y_grid, Z_grid


if __name__ == "__main__":
    # file_id = 1663484946120
    # file_id = 1695092317631
    # file_id = 1698088573367
    # file_id =1699853891683
    # file_id = 1701152211845  # 50ms readout
    file_id = 1726476640278  # 30ms readout
    # file_id = 1725055024398 # 30ms readout
    data = dm.get_raw_data(file_id=file_id, load_npz=False, use_cache=True)
    nv_list = data["nv_list"]
    num_nvs = len(nv_list)
    counts = np.array(data["counts"])[0]
    num_nvs = len(nv_list)
    num_steps = data["num_steps"]
    num_runs = data["num_runs"]
    num_reps = data["num_reps"]
    freqs = data["freqs"]
    adj_num_steps = num_steps // 4
    sig_counts_0 = counts[:, :, 0:adj_num_steps, :]
    sig_counts_1 = counts[:, :, adj_num_steps : 2 * adj_num_steps, :]
    sig_counts = np.append(sig_counts_0, sig_counts_1, axis=3)
    ref_counts_0 = counts[:, :, 2 * adj_num_steps : 3 * adj_num_steps, :]
    ref_counts_1 = counts[:, :, 3 * adj_num_steps :, :]
    ref_counts = np.empty((num_nvs, num_runs, adj_num_steps, 2 * num_reps))
    ref_counts[:, :, :, 0::2] = ref_counts_0
    ref_counts[:, :, :, 1::2] = ref_counts_1
    #
    now = datetime.now()
    date_time_str = now.strftime("%Y%m%d_%H%M%S")
    file_name = dm.get_file_name(file_id=file_id)
    file_path = dm.get_file_path(__file__, file_name, f"{file_id}_{date_time_str}")

    thresh_method = "otsu"
    plot_nv_resonance_fits_and_residuals(
        nv_list,
        freqs,
        sig_counts,
        ref_counts,
        file_id,
        num_cols=9,
        threshold_method=thresh_method,
    )
    # print(f"Plot saved to {file_path}")

    # selected_nv_indices = [5, 11, 22, 60]
    # plot_selected_nv_resonance_fits_comparison(
    #     nv_list, freqs, sig_counts, ref_counts, file_id, selected_nv_indices
    # )

    # plt.show()
    # freq_splitting = nv_resonance_splitting(nv_list, freqs, avg_counts, avg_counts_ste)

    # magnetic_fields = calculate_magnetic_fields(
    #     nv_list, field_splitting=freq_splitting, zero_field_splitting=0.0,
    #     gyromagnetic_ratio=28.0, threshold=0.09
    # )

    # # Print or visualize the magnetic fields
    # for i, B in enumerate(magnetic_fields):
    #     print(f"NV {i+1}: Magnetic Field: {B:.4f} T")

    # Generate a 2D magnetic field map
    dist_conversion_factor = 0.072
    # generate_2d_magnetic_field_map_rbf(nv_list, magnetic_fields, dist_conversion_factor, grid_size=100)
    # generate_2d_magnetic_field_map_kriging(nv_list, magnetic_fields, dist_conversion_factor, grid_size=100)

    # # List of file IDs to process
    # file_ids = [1647377018086,
    #             1651762931005,
    #             1652859661831,
    #             1654734385295]  # Add more file IDs as needed

    # # Iterate over each file_id
    # for file_id in file_ids:
    #     # Load raw data
    #     data = dm.get_raw_data(file_id=file_id, load_npz=False, use_cache=True)

    #     nv_list = data["nv_list"]
    #     num_nvs = len(nv_list)
    #     counts = np.array(data["counts"])[0]
    #     num_steps = data["num_steps"]
    #     num_runs = data["num_runs"]
    #     num_reps = data["num_reps"]
    #     freqs = data["freqs"]

    #     adj_num_steps = num_steps // 4
    #     sig_counts_0 = counts[:, :, 0:adj_num_steps, :]
    #     sig_counts_1 = counts[:, :, adj_num_steps : 2 * adj_num_steps, :]
    #     sig_counts = np.append(sig_counts_0, sig_counts_1, axis=3)

    #     ref_counts_0 = counts[:, :, 2 * adj_num_steps : 3 * adj_num_steps, :]
    #     ref_counts_1 = counts[:, :, 3 * adj_num_steps :, :]
    #     ref_counts = np.empty((num_nvs, num_runs, adj_num_steps, 2 * num_reps))
    #     ref_counts[:, :, :, 0::2] = ref_counts_0
    #     ref_counts[:, :, :, 1::2] = ref_counts_1

    #     avg_counts = np.mean(sig_counts, axis=1)
    #     avg_counts_ste = np.std(sig_counts, axis=1) / np.sqrt(num_runs)
