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
        min_count = np.min(nv_counts)
        max_count = np.max(nv_counts)
        nv_counts = (nv_counts - min_count) / (max_count - min_count)
        nv_counts_ste = nv_counts_ste / (max_count - min_count)

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

        return (
            fit_fns,
            fit_curve,
            chi_squared,
            (f1, f2),
            freq_diff,
            width,
            contrast,
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
    ) = zip(*fit_results)

    # Convert results to lists (since zip returns tuples)
    fit_fns = list(fit_fns)
    chi_squared_list = list(chi_squared_list)
    center_freqs = list(center_freqs)
    center_freq_differences = list(center_freq_differences)
    avg_peak_widths = list(avg_peak_widths)
    avg_peak_amplitudes = list(avg_peak_amplitudes)
    # snrs = list(snrs)

    # Plot SNR vs frequency for all NVs
    fig_snr, ax_snr = plt.subplots(figsize=(7, 5))

    for nv_idx in range(num_nvs):
        snrs = np.reshape(avg_snr[nv_idx], len(freqs))
        # Replace NaNs/infs with 0 so the plot still shows
        snrs = np.nan_to_num(snrs, nan=0.0, posinf=0.0, neginf=0.0)

        ax_snr.plot(freqs, snrs, linewidth=1, label=f"NV{nv_idx}")

    ax_snr.set_xlabel("Microwave Frequency (MHz)", fontsize=15)
    ax_snr.set_ylabel("Signal-to-Noise Ratio (SNR)", fontsize=15)
    ax_snr.set_title("SNR vs Frequency for All NVs", fontsize=15)

    ax_snr.legend(fontsize="small", ncol=2)
    ax_snr.grid(True, linestyle="--", alpha=0.6)
    ax_snr.tick_params(axis="both", labelsize=14)

    plt.tight_layout()
    plt.show(block=True)

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

    # filter_nvs = True
    filter_nvs = False
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
    # writer = PillowWriter(fps=fps)
    # ani.save(output_filename, writer=writer)
    # print(f"Movie successfully saved to {output_filename}")


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

    # file_ids = [
    #     "2025_09_24-09_33_36-rubin-nv0_2025_09_08",
    # ]
    file_ids = [
        "2025_10_03-06_59_37-rubin-nv0_2025_09_08",
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
