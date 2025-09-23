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
from joblib import Parallel, delayed
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle

# from pykrige import OrdinaryKriging
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


def calculate_contrast(signal, bg_offset):
    """Calculate contrast based on fit parameters and chi-squared value."""
    contrast = signal / bg_offset
    return contrast


def calculate_snr(residuals, nv_counts_ste, signal):
    """Calculate SNR based on the fitted Voigt profile and residuals."""
    noise = np.std(residuals / nv_counts_ste)
    return signal / noise if noise > 0 else 0


def voigt_with_background(
    freq,
    amp1,
    amp2,
    center1,
    center2,
    width,
    bg_offset,
    # bg_slope,
):
    """Voigt profile for two peaks with a linear background."""
    freq = np.array(freq)  # Ensure freq is a NumPy array for element-wise operations
    return (
        amp1 * norm_voigt(freq, width, width, center1)
        + amp2 * norm_voigt(freq, width, width, center2)
        + bg_offset
        # + bg_slope * freq
    )


def residuals_fn(params, freq, nv_counts, nv_counts_ste):
    """Compute residuals for least_squares optimization."""
    fit_vals = voigt_with_background(freq, *params)
    return (nv_counts - fit_vals) / nv_counts_ste  # Weighted residuals


def create_esr_blink_movie(
    freqs, avg_counts, fitted_data, nv_positions, nv_images, output_file="esr_movie.mp4"
):
    """
    Creates a movie showing ESR resonance fits and blinking NVs.

    Parameters:
    - freqs: Array of frequency steps.
    - avg_counts: 2D array of average counts [NV index, frequency step].
    - fitted_data: 2D array of fitted ESR data [NV index, frequency step].
    - nv_positions: List of (x, y) positions for NVs in the image.
    - nv_images: 3D array of NV images [time step, height, width].
    - output_file: Path to save the output movie.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    # Axes for resonance data and NV blinking
    ax_esr, ax_nv = axes
    # Initialize ESR plot
    (esr_line,) = ax_esr.plot([], [], "o-", label="ESR Data", color="blue")
    (esr_fit,) = ax_esr.plot([], [], "-", label="Fit", color="red")
    ax_esr.set_xlim(min(freqs), max(freqs))
    ax_esr.set_ylim(0, np.max(avg_counts) * 1.2)
    ax_esr.set_xlabel("Frequency (GHz)")
    ax_esr.set_ylabel("Counts")
    ax_esr.legend()
    ax_esr.grid()

    # Initialize NV blinking image
    nv_image = ax_nv.imshow(
        nv_images[0], cmap="gray", aspect="auto", interpolation="nearest"
    )
    ax_nv.set_title("NV Blinking")
    ax_nv.axis("off")

    # Update function for animation
    def update(frame):
        # Update ESR data
        nv_idx = frame % len(avg_counts)  # Loop through NV indices
        esr_line.set_data(freqs, avg_counts[nv_idx])
        esr_fit.set_data(freqs, fitted_data[nv_idx])
        ax_esr.set_title(f"ESR Resonance (NV Index: {nv_idx + 1})")

        # Update NV image
        nv_image.set_data(nv_images[frame])
        return esr_line, esr_fit, nv_image

    # Create animation
    anim = FuncAnimation(fig, update, frames=len(nv_images), blit=True, repeat=True)
    # Save the movie
    anim.save(output_file, fps=6, writer="ffmpeg")
    plt.close(fig)


def plot_nv_resonance_fits_and_residuals(
    nv_list,
    freqs,
    sig_counts,
    ref_counts,
    file_id,
    num_cols=6,
):
    """
    Plot NV resonance data with fitted Voigt profiles (including background), with residuals and contrast values.
    """

    avg_counts, avg_counts_ste = widefield.process_counts(
        nv_list, sig_counts, ref_counts, threshold=True
    )

    avg_snr, avg_snr_ste = widefield.calc_snr(sig_counts, ref_counts)
    num_nvs = len(nv_list)
    freqs_dense = np.linspace(min(freqs), max(freqs), 60)

    def process_nv(nv_idx):
        nv_counts = avg_counts[nv_idx]
        nv_counts_ste = avg_counts_ste[nv_idx]
        # Normalize counts between 0 and 1
        # min_count = np.min(nv_counts)
        # max_count = np.max(nv_counts)
        # nv_counts = (nv_counts - min_count) / (max_count - min_count)
        # nv_counts_ste = nv_counts_ste / (max_count - min_count)

        low_freq_guess = freqs[np.argmax(nv_counts[: len(freqs) // 2])]
        high_freq_guess = freqs[
            np.argmax(nv_counts[len(freqs) // 2 :]) + len(freqs) // 2
        ]
        max_amp = np.max(nv_counts)

        guess_params = [
            max_amp,
            max_amp,
            low_freq_guess,
            high_freq_guess,
            5,  # Width guess
            np.min(nv_counts),
        ]
        bounds = (
            [0, 0, min(freqs), min(freqs), 0, -np.inf],  # Lower bounds
            [np.inf, np.inf, max(freqs), max(freqs), np.inf, np.inf],  # Upper bounds
        )

        result = least_squares(
            residuals_fn,
            guess_params,
            args=(freqs, nv_counts, nv_counts_ste),
            bounds=bounds,
            max_nfev=20000,
        )
        popt = result.x
        fit_fns = voigt_with_background(freqs_dense, *popt)
        # Compute fit
        fit_curve = voigt_with_background(freqs, *popt)
        residuals = nv_counts - fit_curve
        chi_squared = np.sum((residuals / nv_counts_ste) ** 2)

        # Extract parameters
        amp1, amp2, f1, f2, width, bg_offset = popt
        contrast = (amp1 + amp2) / 2
        freq_diff = abs(f2 - f1)

        # SNR Calculation
        # Flatten SNR array for current NV
        snrs_1d = np.reshape(avg_snr[nv_idx], len(freqs))

        # Interpolate to get SNR at the two resonance frequencies
        from scipy.interpolate import interp1d

        snr_interp = interp1d(freqs, snrs_1d, kind="linear", fill_value="extrapolate")
        snr1 = snr_interp(f1)
        snr2 = snr_interp(f2)
        snr = (snr1 + snr2) / 2
        print(
            f"NV {nv_idx}: SNR = {snr:.2f}, (f1,f2) = ({f1:.4f}, {f2:.4f}), freq_diff = {freq_diff:.3f}"
        )

        return (
            fit_fns,
            fit_curve,
            chi_squared,
            (f1, f2),
            freq_diff,
            width,
            contrast,
            snr,
        )

    # Run in parallel
    fit_results = Parallel(n_jobs=-1)(
        delayed(process_nv)(nv_idx) for nv_idx in range(num_nvs)
    )
    # return
    # # Unpacking results correctly
    (
        fit_fns,
        fit_data,
        chi_squared_list,
        center_freqs,
        center_freq_differences,
        avg_peak_widths,
        avg_peak_amplitudes,
        snrs,
    ) = zip(*fit_results)

    # Convert results to lists (since zip returns tuples)
    fit_fns = list(fit_fns)
    chi_squared_list = list(chi_squared_list)
    center_freqs = list(center_freqs)
    center_freq_differences = list(center_freq_differences)
    avg_peak_widths = list(avg_peak_widths)
    avg_peak_amplitudes = list(avg_peak_amplitudes)
    snrs = list(snrs)

    # Plot SNR vs frequency for all NVs
    fig_snr, ax_snr = plt.subplots(figsize=(6.5, 5))
    for nv_idx in range(num_nvs):
        snrs = np.reshape(avg_snr[nv_idx], len(freqs))
        ax_snr.plot(freqs, snrs, linewidth=0.5)

    ax_snr.set_xlabel("Microwave Frequency (MHz)", fontsize=15)
    ax_snr.set_ylabel("Signal-to-Noise Ratio (SNR)", fontsize=15)
    ax_snr.set_title("SNR vs Frequency for All NVs", fontsize=15)
    ax_snr.legend(fontsize="small", ncol=2)
    ax_snr.grid(True)
    ax_snr.tick_params(axis="both", labelsize=14)
    plt.tight_layout()
    # plt.show(block=True)
    # sys.exit()'
    # return
    ### snrs
    median_snr = np.median(snrs)
    print(f"median snr:{median_snr:.2f}")
    # Remove outliers from a data array using the IQR method.
    Q1 = np.percentile(snrs, 25)
    Q3 = np.percentile(snrs, 75)
    IQR = Q3 - Q1
    # lower_bound = Q1 - 2 * IQR
    # upper_bound = Q3 + 2 * IQR
    lower_bound = 0.0
    upper_bound = 0.6
    print(f"lower bound:{lower_bound}")
    print(f"upper_bound:{upper_bound}")
    # Identify and filter out outliers
    cleaned_nv_indices = [
        i for i, val in enumerate(snrs) if lower_bound <= val <= upper_bound
    ]
    outlier_indices = [
        i for i, val in enumerate(snrs) if val < lower_bound or val > upper_bound
    ]

    # cleaned_snrs = [val for val in snrs if val in cleaned_nv_indices]
    # Remove index 144 from cleaned_nv_indices first
    # indices_to_remove = [4, 144]
    # cleaned_nv_indices = [
    #     idx for idx in cleaned_nv_indices if idx not in indices_to_remove
    # ]

    cleaned_snrs = [snrs[idx] for idx in cleaned_nv_indices]
    median_snr_cleaned = np.median(cleaned_snrs)
    print(f"snrs IQR: {IQR}")
    print(f"median snr cleaned:{median_snr_cleaned}")
    print(f"Number of nvs removed : {len(outlier_indices)}")
    print(f" removed nvs : {outlier_indices}")

    print(f"Number of nvs after filtering : {len(cleaned_nv_indices)}")
    print(f"cleaned indices: {cleaned_nv_indices}")
    # return
    # Scatter plot
    plt.figure()
    plt.scatter(
        cleaned_nv_indices,
        cleaned_snrs,
        color="blue",
        marker="o",
        alpha=0.6,
        label="SNRs",
    )
    for i, (nv_index, snr) in enumerate(zip(cleaned_nv_indices, cleaned_snrs)):
        plt.annotate(
            f"{nv_index}",
            (nv_index, snr),
            textcoords="offset points",
            xytext=(0, 2),
            ha="center",
            fontsize=6,
        )
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
        label=f"Q1 = {Q1:.3f}",
    )
    plt.axhline(
        Q3,
        color="red",
        linestyle="--",
        label=f"Q3 = {Q3:.3f}",
    )
    # Add labels and legend
    plt.title(f"SNRs Across {num_nvs} Shallow NVs")
    plt.xlabel("NV Index")
    plt.ylabel("SNR")
    plt.legend(loc="upper right", fontsize=9)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    # return

    filter_nvs = True
    # filter_nvs = False
    if filter_nvs:
        # target_peak_values = [0.025, 0.068, 0.146, 0.185]
        target_peak_values = [0.068, 0.185]
        target_peak_values = [0.290]
        tolerance = 0.01
        # Filter indices based on proximity to target peak differences
        filtered_indices = [
            idx
            for idx, freq_diff in enumerate(center_freq_differences)
            if any(
                target - tolerance <= freq_diff <= target + tolerance
                for target in target_peak_values
            )
        ]
        # Find non-matching indices
        non_matching_indices = [
            idx
            for idx in range(len(center_freq_differences))
            if idx not in filtered_indices
        ]
        if non_matching_indices:
            print(f"Non-matching NVs: {non_matching_indices}")

        # Initialize a dictionary to store indices for each orientation
        orientation_indices = {value: [] for value in target_peak_values}

        for idx, freq_diff in enumerate(center_freq_differences):
            for target in target_peak_values:
                if target - tolerance <= freq_diff <= target + tolerance:
                    orientation_indices[target].append(idx)

        avg_center_freqs = {}
        for orientation, indices in orientation_indices.items():
            if indices:
                freqs_selected_by_orienation = [center_freqs[idx] for idx in indices]
                avg_peak1 = np.median(
                    [freq[0] for freq in freqs_selected_by_orienation]
                )
                avg_peak2 = np.median(
                    [freq[1] for freq in freqs_selected_by_orienation]
                )
                avg_center_freqs[orientation] = (avg_peak1, avg_peak2)
        # Print the results
        for orientation, indices in orientation_indices.items():
            print(f"Orientation centered around {orientation} GHz:")
            print(f"NV Indices: {indices}")
            print(f"Number of NVs: {len(indices)}\n")
        print(f"All SNRs: {snrs}")
        print(f"Median SNR: {np.median(snrs)}")

        for orientation, avg_freqs in avg_center_freqs.items():
            print(f"Orientation {orientation} GHZ")
            print(f"Avearge Peak 1: {avg_freqs[0]:.6f} GHz")
            print(f"Avearge Peak 1: {avg_freqs[1]:.6f} GHz")

    else:
        filtered_indices = list(range(num_nvs))

    # mannual removal of indices
    # indices_to_remove_manually = []
    # filtered_indices = [
    #     filtered_indices[idx]
    #     for idx in range(num_nvs)
    #     if idx not in indices_to_remove_manually
    # ]
    # return
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
    # filtered_contrast_list = [contrast_list[idx] for idx in filtered_indices]
    filtered_chi_squared_list = [chi_squared_list[idx] for idx in filtered_indices]
    # filtered_fitted_data = [fit_data[idx] for idx in filtered_indices]
    filtered_fitted_data = [fit_fns[idx] for idx in filtered_indices]

    # return
    # Set plot style
    for nv_ind in range(num_nvs):
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        ax.plot(freqs, avg_counts[nv_ind], "o", color="steelblue")
        ax.plot(freqs_dense, fit_fns[nv_ind], "-", color="red")
        ax.set_xlabel("Frequency (GHz)")
        ax.set_ylabel("Norm. NV- Population")
        ax.set_title(f"NV Index: {nv_ind}")
        ax.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        # plt.show()
        plt.show(block=True)
    return

    # Plot histograms and scatter plots
    plots_data = [
        (
            "Histogram of Frequency Splitting",
            filtered_freq_differences,
            None,
            "Freq Splitting (GHz)",
            "Count",
            "teal",
            "MHz",
        ),
        (
            "ESR Freq Splitting vs Average Peak Width",
            filtered_freq_differences,
            filtered_avg_peak_widths,
            "Freq Splitting (GHz)",
            "Average Peak Width",
            "orange",
            "MHz",
        ),
        (
            "ESR Peak Freqs vs Avg Peak Amps",
            filtered_freq_differences,
            filtered_avg_peak_amplitudes,
            "Freq Splitting (GHz)",
            "Avg Peak Amp",
            "green",
            "arb. unit",
        ),
        (
            "NV Index vs Freq. Splitting.",
            list(range(len(filtered_freq_differences))),  # NV indices
            filtered_freq_differences,
            "NV Index",
            "Freq. Splitting (GHz)",
            "blue",
            "GHz",
        ),
    ]

    for title, x_data, y_data, xlabel, ylabel, color, unit in plots_data:
        kpl.init_kplotlib()
        plt.figure()
        if isinstance(y_data, list):
            plt.scatter(x_data, y_data, color=color, alpha=0.7, edgecolors="k")
            # plt.figtext(
            #     0.97,
            #     0.9,
            #     f"Median Width {np.median(y_data):.2f} {unit}",
            #     fontsize=11,
            #     ha="right",
            #     va="top",
            #     bbox=dict(
            #         facecolor="white", edgecolor="black", boxstyle="round,pad=0.3"
            #     ),
            # )
        else:
            plt.hist(x_data, bins=9, color=color, alpha=0.7, edgecolor="black")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True, linestyle="--", alpha=0.6)
        # file_path = dm.get_file_path(
        #     __file__, file_name, f"{file_id}_{date_time_str}_{title}"
        # )
        kpl.show()
        # dm.save_figure(fig, file_path)
        # plt.close(fig)
    # Plot filtered resonance fits
    sns.set(style="whitegrid", palette="muted")
    num_filtered_nvs = len(filtered_nv_list)
    colors = sns.color_palette("deep", num_filtered_nvs)
    num_rows = int(np.ceil(num_filtered_nvs / num_cols))
    fig_fitting, axes_fitting = plt.subplots(
        num_rows,
        num_cols,
        figsize=(num_cols * 2, num_rows * 1),
        sharex=True,
        sharey=False,
        constrained_layout=True,
    )
    axes_fitting = axes_fitting.flatten()

    for nv_idx, ax in enumerate(axes_fitting):
        if nv_idx < num_filtered_nvs:
            sns.lineplot(
                x=freqs,
                y=filtered_avg_counts[nv_idx],
                ax=ax,
                color=colors[nv_idx % len(colors)],
                lw=0,
                marker="o",
                markersize=2,
                label=f"{filtered_indices[nv_idx]}",
            )
            ax.legend(fontsize="xx-small")
            ax.errorbar(
                freqs,
                filtered_avg_counts[nv_idx],
                # yerr=filtered_avg_counts_ste[nv_idx],
                yerr=np.abs(filtered_avg_counts_ste[nv_idx]),
                fmt="none",
                ecolor="gray",
                alpha=0.6,
            )
            ax.grid(True, which="both", linestyle="--", linewidth=0.5)
            # Plot fitted data on the same subplot
            ax.plot(
                freqs_dense,
                filtered_fitted_data[nv_idx],
                "-",
                color=colors[nv_idx % len(colors)],
                label="Fit",
                lw=1,
            )
            # Y-tick labels for the leftmost column
            # if nv_idx % num_cols == 0:
            #     ax.set_yticks(ax.get_yticks())
            # else:
            #     ax.set_yticklabels([])
            ax.set_yticklabels([])
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
            # # Check if the last row has any valid plots
            # last_row_start_idx = (num_rows - 1) * num_cols
            # last_row_indices = range(last_row_start_idx, last_row_start_idx + num_cols)
            # last_row_has_plots = any(
            #     idx < len(axes_fitting) for idx in last_row_indices
            # )

            # # Set custom tick locations and labels
            # if (
            #     not last_row_has_plots
            # ):  # Only handle the second-to-last row if the last row is empty
            #     second_last_row_start_idx = (num_rows - 2) * num_cols
            #     for col in range(num_cols):
            #         col_idx = second_last_row_start_idx + col
            #         if col_idx < len(axes_fitting):  # Ensure index is within bounds
            #             ax = axes_fitting[col_idx]
            #             tick_positions = np.linspace(min(freqs), max(freqs), 5)
            #             ax.set_xticks(tick_positions)
            #             ax.set_xticklabels(
            #                 [f"{tick:.2f}" for tick in tick_positions],
            #                 rotation=45,
            #                 fontsize=9,
            #             )
            #             ax.set_xlabel("Frequency (GHz)")

            ax.grid(True, which="both", linestyle="--", linewidth=0.5)
        else:
            ax.axis("off")
    plt.subplots_adjust(
        left=0.1, right=0.95, top=0.95, bottom=0.1, hspace=0.001, wspace=0.001
    )
    fig_fitting.text(
        -0.005,
        0.5,
        "NV$^{-}$ Population",
        va="center",
        rotation="vertical",
        fontsize=12,
    )
    # file_name = dm.get_file_name(file_id=file_id)
    fig_fitting.suptitle(f"ESR {file_id}", fontsize=16)
    # plt.tight_layout()
    # now = datetime.now()
    # date_time_str = now.strftime("%Y%m%d_%H%M%S")
    # # file_path = dm.get_file_path(__file__, file_name, f"{file_id}_{date_time_str}")
    plt.show()
    # dm.save_figure(fig_fitting, file_path)
    # plt.close(fig_fitting)
    # return


# def estimate_magnetic_field_direction(
#     field_splitting, gyromagnetic_ratio=28.0, threshold=0.06
# ):
#     """
#     Estimate the direction of the magnetic field based on resonance frequency splitting in NV centers,
#     and return the relative magnetic field direction in degrees for small, medium, and large splitting cases.

#     Args:
#         field_splitting: List of frequency splitting values.
#         threshold1: First threshold to classify small splitting (default: 0.12 GHz).
#         threshold2: Second threshold to classify medium splitting (default: 0.6 GHz).

#     Returns:
#         result: A dictionary containing the magnetic field directions in degrees for small, medium, and large splitting cases.
#     """
#     # Separate small, medium, and large splitting cases
#     small_splitting_nv = [split for split in field_splitting if split <= threshold]
#     # medium_splitting_nv = [split for split in field_splitting if threshold1 < split <= threshold2]
#     large_splitting_nv = [split for split in field_splitting if split > threshold]

#     # Compute average splittings for small, medium, and large splitting cases
#     avg_small_split = np.mean(small_splitting_nv) if small_splitting_nv else 0
#     # avg_medium_split = np.mean(medium_splitting_nv) if medium_splitting_nv else 0
#     avg_large_split = np.mean(large_splitting_nv) if large_splitting_nv else 0

#     # Known NV orientation vectors in the diamond lattice (for 3 NV orientations)
#     # nv_orientations = np.array([[-1, 1, 1],[1, 1, 1]]) / np.sqrt(3) #no good
#     # nv_orientations = np.array([[1, -1, 1], [1, 1, 1]]) / np.sqrt(3) #no good
#     # nv_orientations = np.array([[1, 1, -1],[1, 1, 1]]) / np.sqrt(3) #no good
#     # nv_orientations = np.array([[1, -1, 1],[-1, 1, 1]]) / np.sqrt(3) #good
#     # nv_orientations = np.array([[-1, 1, 1], [1, 1, -1]]) / np.sqrt(3) #good
#     nv_orientations = np.array([[1, -1, 1], [1, 1, -1]]) / np.sqrt(3)
#     # Initialize a result dictionary
#     avg_splitting = np.array([avg_small_split, avg_large_split])
#     # Convert splitting into an array and scale by gyromagnetic ratio
#     B_proj = avg_splitting / gyromagnetic_ratio  # Magnetic field projections in Tesla

#     # Solve the system of linear equations to estimate the magnetic field components
#     B_components, _, _, _ = np.linalg.lstsq(nv_orientations, B_proj, rcond=None)

#     # Calculate the magnitude of the magnetic field
#     B_magnitude = np.linalg.norm(B_components)

#     # Compute the angle between the magnetic field vector and each NV orientation
#     B_direction_deg = []
#     for nv_orientation in nv_orientations:
#         cos_theta = np.dot(B_components, nv_orientation) / B_magnitude
#         theta_deg = np.degrees(np.arccos(cos_theta))
#         B_direction_deg.append(theta_deg)

#     # Return the magnetic field direction in degrees and the magnitude of the magnetic field
#     print(B_direction_deg)
#     return B_direction_deg


# def calculate_magnetic_fields(
#     nv_list,
#     field_splitting,
#     zero_field_splitting=0.0,
#     gyromagnetic_ratio=28.0,
#     threshold=0.09,
# ):
#     """
#     Calculate magnetic fields for each NV center based on frequency splitting and adjust based on the magnetic field direction.

#     Args:
#         nv_list: List of NV center identifiers.
#         field_splitting: List of frequency splitting values corresponding to the NV centers.
#         zero_field_splitting: Zero-field splitting (D) in GHz.
#         gyromagnetic_ratio: Gyromagnetic ratio (28 GHz/T).
#         threshold1: First threshold to classify small splitting (default: 0.06 GHz).
#         threshold2: Second threshold to classify medium splitting (default: 0.12 GHz).

#     Returns:
#         result: A list of magnetic field values for each NV center, in the same order as nv_list.
#     """
#     # Initialize a list to store the magnetic field values, maintaining order
#     magnetic_fields = []

#     # Get magnetic field directions for small, medium, and large splitting cases
#     magnetic_field_directions = estimate_magnetic_field_direction(
#         field_splitting, threshold
#     )
#     # Extract angles for each category
#     theta_deg_small = magnetic_field_directions[0]
#     # theta_deg_medium = magnetic_field_directions[1]
#     theta_deg_large = magnetic_field_directions[1]
#     # Iterate over each NV center and its corresponding frequency splitting, maintaining the order
#     for split in field_splitting:
#         if split > threshold:
#             # Large splitting (orientation 3)
#             B_3 = (split - zero_field_splitting) / gyromagnetic_ratio
#             B_3 = abs(
#                 B_3 / np.cos(np.deg2rad(theta_deg_large))
#             )  # Adjust by direction angle for large splitting
#             magnetic_fields.append(B_3)
#         # elif threshold1 < split <= threshold2:
#         #     # Medium splitting (orientation 2)
#         #     B_2 = (split - zero_field_splitting) / gyromagnetic_ratio
#         #     B_2 = abs(B_2 / np.cos(np.deg2rad(theta_deg_medium)))  # Adjust by direction angle for medium splitting
#         #     magnetic_fields.append(B_2)
#         else:
#             # Small splitting (orientation 1)
#             B_1 = (split - zero_field_splitting) / gyromagnetic_ratio
#             B_1 = abs(
#                 B_1 / np.cos(np.deg2rad(theta_deg_small))
#             )  # Adjust by direction angle for small splitting
#             magnetic_fields.append(B_1)

#     # Return the magnetic fields in the same order as the input NV list
#     return magnetic_fields


# def estimate_magnetic_field_from_fitting(
#     nv_list,
#     field_splitting,
#     zero_field_splitting=2.87,
#     gyromagnetic_ratio=28.0,
#     threshold=0.05,
# ):
#     """
#     Estimate magnetic fields at each NV based on resonance frequency splitting.

#     Args:
#         nv_list: List of NV signatures.
#         field_splitting: Frequency splitting values.
#         zero_field_splitting: Zero-field splitting (D) in GHz.
#         gyromagnetic_ratio: Gyromagnetic ratio (28 GHz/T).
#         threshold: Threshold to classify splitting into orientations.

#     Returns:
#         magnetic_fields: Magnetic fields for each NV, reordered by NV index.
#     """
#     # Initialize a list to store magnetic fields
#     magnetic_fields = []

#     # Iterate over each NV and its frequency splitting
#     for nv_idx, split in enumerate(field_splitting):
#         if split > threshold:
#             # Large splitting (orientation 2)
#             B_2 = (split - zero_field_splitting) / gyromagnetic_ratio
#             # B_2 = abs(B_2 * np.cos(np.deg2rad(109.47)))
#             magnetic_fields.append(abs(B_2))
#         else:
#             # Small splitting (orientation 1)
#             B_1 = (split - zero_field_splitting) / gyromagnetic_ratio
#             B_1 = abs(B_1 / np.cos(np.deg2rad(109.47)))  # Adjust by angle if needed
#             magnetic_fields.append(B_1)

#     return magnetic_fields


# def remove_outliers(B_values, nv_list):
#     """
#     Remove outliers using the IQR (Interquartile Range) method and corresponding NV centers.

#     Args:
#         B_values: Array of magnetic field values.
#         nv_list: List of NV centers.

#     Returns:
#         Cleaned B_values, cleaned nv_list with outliers removed.
#     """
#     # Calculate Q1 (25th percentile) and Q3 (75th percentile)
#     Q1 = np.percentile(B_values, 25)
#     Q3 = np.percentile(B_values, 75)

#     # Calculate IQR
#     IQR = Q3 - Q1

#     # Define outlier bounds
#     lower_bound = Q1 - 1.5 * IQR
#     upper_bound = Q3 + 1.5 * IQR

#     # Filter out the outliers in B_values and corresponding nv_list
#     mask = (B_values >= lower_bound) & (B_values <= upper_bound)

#     # Return cleaned data
#     return B_values[mask], [nv for i, nv in enumerate(nv_list) if mask[i]]


# def generate_2d_magnetic_field_map_kriging(
#     nv_list, magnetic_fields, dist_conversion_factor, grid_size=100
# ):
#     """
#     Generate a 2D map of the magnetic field using Kriging interpolation.

#     Args:
#         nv_list: List of NV centers, each having 'pixel_coords' attributes.
#         magnetic_fields: Calculated magnetic fields for each NV.
#         dist_conversion_factor: Conversion factor from pixels to real-world distance (e.g., micrometers per pixel).
#         grid_size: Size of the output grid (resolution of the 2D map).

#     Returns:
#         X, Y: Coordinates of the grid.
#         Z: Interpolated magnetic field values over the grid.
#     """
#     # Convert magnetic fields from Tesla to Gauss
#     B_values = (
#         np.array(magnetic_fields) * 1e4
#     )  # Convert Tesla to Gauss (1 Tesla = 10,000 Gauss)
#     # Remove outliers and corresponding NV centers
#     B_values, nv_list = remove_outliers(B_values, nv_list)
#     # Extract NV positions (convert pixel coordinates to real-world distance)
#     x_coords = (
#         np.array([nv.coords["pixel"][0] for nv in nv_list]) * dist_conversion_factor
#     )
#     y_coords = (
#         np.array([nv.coords["pixel"][1] for nv in nv_list]) * dist_conversion_factor
#     )

#     # Create a grid for interpolation
#     x_min, x_max = min(x_coords), max(x_coords)
#     y_min, y_max = min(y_coords), max(y_coords)
#     X_grid, Y_grid = np.meshgrid(
#         np.linspace(x_min, x_max, grid_size), np.linspace(y_min, y_max, grid_size)
#     )

#     # Perform Kriging interpolation
#     kriging_interpolator = OrdinaryKriging(
#         x_coords, y_coords, B_values, variogram_model="linear"
#     )
#     Z_grid, _ = kriging_interpolator.execute(
#         "grid",
#         np.linspace(x_min, x_max, grid_size),
#         np.linspace(y_min, y_max, grid_size),
#     )
#     # Plot the 2D magnetic field map using matplotlib
#     plt.figure(figsize=(8, 6))
#     contour = plt.contourf(X_grid, Y_grid, Z_grid, levels=100, cmap="plasma")
#     plt.colorbar(contour, label="Magnetic Field (G)")

#     # for x, y in zip(x_coords, y_coords):
#     #     circle = Circle((x, y), radius=3, facecolor=None, edgecolor="lightblue")
#     #     ax.add_patch(circle)
#     # Scatter the NV positions and label their magnetic field values
#     # plt.scatter(x_coords, y_coords, edgecolor='lightblue', s=50)
#     plt.scatter(
#         x_coords, y_coords, facecolor="none", edgecolor="lightblue", s=30, linewidth=1.0
#     )
#     # plt.colorbar(scatter, label='Magnetic Field (G)')

#     # for i, (x, y, b) in enumerate(zip(x_coords, y_coords, B_values)):
#     #     plt.text(x, y, f'{b:.2f} G', fontsize=8, color='white', ha='center', va='center')

#     plt.title("2D Magnetic Field Map (Kriging Interpolation)")
#     plt.xlabel("X Position (µm)")
#     plt.ylabel("Y Position (µm)")
#     plt.xticks(np.linspace(x_min, x_max, 5))
#     plt.yticks(np.linspace(y_min, y_max, 5))
#     plt.show()

#     return X_grid, Y_grid, Z_grid

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter


def create_movie(data, output_filename="movie.gif", nv_index=0, fps=5):
    """
    Create a movie of the NV images along step indices.
    """
    # Extract img_arrays and validate
    # img_arrays = np.array(
    #     data["img_arrays"]
    # )  # Shape: [nv_ind, step_ind, height, width]

    img_arrays = data["img_arrays"]
    print("Type of img_arrays:", type(img_arrays))
    print("Contents of img_arrays:", img_arrays)

    if img_arrays.ndim != 4:
        raise ValueError(
            "img_arrays must have the structure [nv_ind, step_ind, height, width]"
        )

    num_steps = img_arrays.shape[1]

    # Set up the figure
    fig, ax = plt.subplots()
    img = ax.imshow(
        img_arrays[nv_index, 0, :, :], cmap="viridis", interpolation="nearest"
    )
    ax.set_title(f"NV {nv_index} - Step 0")
    plt.colorbar(img, ax=ax)

    # Update function for animation
    def update(frame):
        img.set_data(img_arrays[nv_index, frame, :, :])
        ax.set_title(f"NV {nv_index} - Step {frame}")
        return (img,)

    # Create animation
    ani = FuncAnimation(fig, update, frames=num_steps, interval=1000 // fps, blit=True)

    # Save as GIF
    writer = PillowWriter(fps=fps)
    ani.save(output_filename, writer=writer)
    print(f"Movie saved to {output_filename}")


def create_movie(data, output_filename="movie.gif", nv_index=0, fps=5):
    """
    Generate a movie of NV images along step indices and save it as a GIF.

    Parameters:
        data (dict): A dictionary containing the 'img_arrays' key with image data.
        output_filename (str): The path to save the output movie (GIF format).
        nv_index (int): The index of the NV center to visualize.
        fps (int): Frames per second for the movie.
    """
    # Extract img_arrays from the data dictionary
    img_arrays = data.get("img_arrays")
    if img_arrays is None:
        raise ValueError("The 'img_arrays' key is missing in the data dictionary.")

    # Validate img_arrays structure
    if not isinstance(img_arrays, np.ndarray):
        raise ValueError("img_arrays must be a numpy array.")
    if img_arrays.ndim != 4:
        raise ValueError(
            "img_arrays must have the shape [nv_ind, step_ind, height, width]."
        )

    num_steps = img_arrays.shape[1]

    # Set up the figure for visualization
    fig, ax = plt.subplots()
    img_display = ax.imshow(
        img_arrays[nv_index, 0, :, :], cmap="viridis", interpolation="nearest"
    )
    ax.set_title(f"NV {nv_index} - Step 0")
    plt.colorbar(img_display, ax=ax)

    # Define the update function for animation
    def update(frame):
        img_display.set_data(img_arrays[nv_index, frame, :, :])
        ax.set_title(f"NV {nv_index} - Step {frame}")
        return (img_display,)

    # Create the animation
    ani = FuncAnimation(fig, update, frames=num_steps, interval=1000 // fps, blit=True)

    # Save the animation as a GIF
    writer = PillowWriter(fps=fps)
    ani.save(output_filename, writer=writer)
    print(f"Movie successfully saved to {output_filename}")


if __name__ == "__main__":
    kpl.init_kplotlib()
    # file_id = 1663484946120
    # file_id = 1695092317631
    # file_id = 1698088573367
    # file_id =1699853891683
    # file_id = 1701152211845  # 50ms readout
    # file_id = 1725055024398  # 30ms readout
    # file_id = 1726476640278  # 30ms readout all variabe
    # file_id = 1729834552723  # 50ms readout mcc
    # file_id = 1732403187814  # 50ms readout 117NVs movies
    # file_id = 1768622780958  # 50ms readout 148 shallow NVs dataset 1
    # file_id = 1768928898711  # 50ms readout 148 shallow NVs dataset 2
    # file_id = 1769412747779  # 50ms readout 148 shallow NVs dataset 3
    # # file_id = 1733307847194
    # file_id = 1771614901873
    # data = dm.get_raw_data(file_id=file_id, load_npz=False, use_cache=True)
    # print(data.keys())
    # # create_movie(data, output_filename="nv_movie.gif", nv_index=0, fps=5)
    # # sys.exit()
    # # readout_duration = data["config"]["Optics"]["VirtualLasers"][
    # #     "VirtualLaserKey.WIDEFIELD_CHARGE_READOUT"
    # # ]["duration"]
    # vls = data["config"]["Optics"]["VirtualLasers"]
    # # print(vls.keys())
    # # print(f"reaout_duaration:{readout_duration}")
    # image_arrays = data["img_arrays"]
    # nv_list = data["nv_list"]
    # num_nvs = len(nv_list)
    # counts = np.array(data["counts"])[0]
    # num_nvs = len(nv_list)
    # num_steps = data["num_steps"]
    # num_runs = data["num_runs"]
    # num_reps = data["num_reps"]
    # freqs = data["freqs"]
    # adj_num_steps = num_steps // 4
    # sig_counts_0 = counts[:, :, 0:adj_num_steps, :]
    # sig_counts_1 = counts[:, :, adj_num_steps : 2 * adj_num_steps, :]
    # sig_counts = np.append(sig_counts_0, sig_counts_1, axis=3)
    # ref_counts_0 = counts[:, :, 2 * adj_num_steps : 3 * adj_num_steps, :]
    # ref_counts_1 = counts[:, :, 3 * adj_num_steps :, :]
    # ref_counts = np.empty((num_nvs, num_runs, adj_num_steps, 2 * num_reps))
    # ref_counts[:, :, :, 0::2] = ref_counts_0
    # ref_counts[:, :, :, 1::2] = ref_counts_1
    # #
    # now = datetime.now()
    # date_time_str = now.strftime("%Y%m%d_%H%M%S")
    # file_name = dm.get_file_name(file_id=file_id)
    # file_path = dm.get_file_path(__file__, file_name, f"{file_id}_{date_time_str}")

    # thresh_method = "otsu"
    # plot_nv_resonance_fits_and_residuals(
    #     nv_list,
    #     freqs,
    #     sig_counts,
    #     ref_counts,
    #     file_id,
    #     num_cols=9,
    # )
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
    dist_conversion_factor = 0.130
    # generate_2d_magnetic_field_map_rbf(nv_list, magnetic_fields, dist_conversion_factor, grid_size=100)
    # generate_2d_magnetic_field_map_kriging(nv_list, magnetic_fields, dist_conversion_factor, grid_size=100)

    # # List of file IDs to process
    # List of file IDs to process
    # file_ids = [1771614901873, 1771932659040]  # before pi pulse optimization
    # file_ids = [1773214393869, 1773497843179]  # uwave power 6, period 80
    # file_ids = [1779130115960, 1779242979662]  # uwave power 6dB, rabi peridod 80ns
    # file_ids = [1780148547772]  # uwave power 2dB, period 96ns
    # file_ids = [1782289026588]  # uwave power 2dB, period 96ns pulse optimization
    # file_ids = [
    #     1783133120931
    # ]  # uwave power 2dB, period 96ns pulse optimization 2 oreintation

    # rubin 140NVs
    file_ids = [1795016507394]
    file_ids = [1796261430133]
    # rubin 107NVs
    file_ids = [1801725762770, 1801943804484]
    # after remoutnig the sample
    # rubin 304NVs
    file_ids = [1803870882950]
    # rubin 154NVs
    file_ids = [1806862148858]
    # rubib 81
    file_ids = [1809016009780]
    # rubib 75
    file_ids = [1810826711017]
    # rubib 75 after change magnet position
    file_ids = [1826522639984]
    # rubib 154 after change magnet position
    file_ids = [1827020564514]
    # # rubib 75 after change magnet position (new position)
    # file_ids = [1829782989309]
    # file_ids = [1830447165544]
    # file_ids = [1831411242534]
    file_ids = [1832069584608]
    file_ids = [1836425531438]

    file_ids = [
        "2025_09_22-22_56_46-rubin-nv0_2025_09_08",
    ]

    # fmt: off
    # fmt: on
    # print(len(reference_pixel_coords))
    # sys.exit()
    # Load the first dataset as a base
    combined_data = dm.get_raw_data(
        file_stem=file_ids[0], load_npz=True, use_cache=True
    )

    combined_sig_counts = None
    combined_ref_counts = None

    if combined_data:
        nv_list = combined_data["nv_list"]
        freqs = combined_data["freqs"]
        num_steps = combined_data["num_steps"]
        num_reps = combined_data["num_reps"]
        num_runs = combined_data["num_runs"]
        counts = np.array(combined_data["counts"])[0]

        # # Extract pixel coordinates from nv_list
        # nv_pixel_coords = np.array([nv.coords["pixel"][:2] for nv in nv_list])
        # # Compute the index mapping by finding closest matches in pixel coordinates
        # ordered_indices = []
        # for ref_coord in reference_pixel_coords:

        #     # Find the index of the closest match
        #     distances = np.linalg.norm(nv_pixel_coords - ref_coord, axis=1)
        #     closest_index = np.argmin(distances)
        #     ordered_indices.append(closest_index)

        # # Reorder nv_list based on ordered indices
        # print(ordered_indices)
        # nv_list = [nv_list[i] for i in ordered_indices]
        # counts = counts[ordered_indices]

        adj_num_steps = num_steps // 4
        sig_counts_0 = counts[:, :, 0:adj_num_steps, :]
        sig_counts_1 = counts[:, :, adj_num_steps : 2 * adj_num_steps, :]
        combined_sig_counts = np.append(sig_counts_0, sig_counts_1, axis=3)

        ref_counts_0 = counts[:, :, 2 * adj_num_steps : 3 * adj_num_steps, :]
        ref_counts_1 = counts[:, :, 3 * adj_num_steps :, :]
        combined_ref_counts = np.empty(
            (len(nv_list), num_runs, adj_num_steps, 2 * num_reps)
        )
        combined_ref_counts[:, :, :, 0::2] = ref_counts_0
        combined_ref_counts[:, :, :, 1::2] = ref_counts_1

        # Process remaining files
        for file_id in file_ids[1:]:
            try:
                print(f"Processing file: {file_id}")
                new_data = dm.get_raw_data(
                    file_id=file_id, load_npz=False, use_cache=True
                )
                if not new_data:
                    print(f"Skipping file {file_id}: Data not found.")
                    continue

                new_counts = np.array(new_data["counts"])[0]

                new_sig_counts_0 = new_counts[:, :, 0:adj_num_steps, :]
                new_sig_counts_1 = new_counts[
                    :, :, adj_num_steps : 2 * adj_num_steps, :
                ]
                new_sig_counts = np.append(new_sig_counts_0, new_sig_counts_1, axis=3)

                new_ref_counts_0 = new_counts[
                    :, :, 2 * adj_num_steps : 3 * adj_num_steps, :
                ]
                new_ref_counts_1 = new_counts[:, :, 3 * adj_num_steps :, :]
                new_ref_counts = np.empty(
                    (len(nv_list), num_runs, adj_num_steps, 2 * num_reps)
                )
                new_ref_counts[:, :, :, 0::2] = new_ref_counts_0
                new_ref_counts[:, :, :, 1::2] = new_ref_counts_1

                # Append new data
                combined_sig_counts = np.append(
                    combined_sig_counts, new_sig_counts, axis=1
                )
                combined_ref_counts = np.append(
                    combined_ref_counts, new_ref_counts, axis=1
                )
                combined_data["num_runs"] += new_data["num_runs"]

            except Exception as e:
                print(f"Error processing file {file_id}: {e}")

        # Generate unique filename
        now = datetime.now()
        date_time_str = now.strftime("%Y%m%d_%H%M%S")
        combined_file_id = "_".join(map(str, file_ids))
        file_name = f"combined_{combined_file_id}_{date_time_str}.png"
        file_path = dm.get_file_path(__file__, "combined_plot", file_name)
        print(f"Combined plot saved to {file_path}")
        # Plot combined data
        plot_nv_resonance_fits_and_residuals(
            nv_list,
            freqs,
            combined_sig_counts,
            combined_ref_counts,
            file_id=combined_file_id,
            num_cols=7,
        )
    else:
        print("No valid data available for plotting.")

    kpl.show(block=True)
