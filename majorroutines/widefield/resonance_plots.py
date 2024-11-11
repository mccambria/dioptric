# -*- coding: utf-8 -*-
"""
Created on October 17th, 2024
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

# from pykrige.ok import OrdinaryKriging
from scipy.interpolate import Rbf
from scipy.optimize import curve_fit, least_squares

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


def process_data(data):
    """
    Process the raw data for NV centers, including averaging counts, normalization, and
    setting up the parameters for fitting and visualization.

    Args:
        data: The raw data dictionary containing NVs, counts, frequencies, and other metadata.

    Returns:
        avg_counts: Averaged counts for NVs.
        avg_counts_ste: Standard error of averaged counts.
        norms: Normalization data for NVs.
        freqs: Frequency data.
        nv_list: List of NV signatures.
    """
    nv_list = data["nv_list"]
    freqs = data["freqs"]
    counts = np.array(data["counts"])

    # Process the counts using the widefield utilities
    avg_counts, avg_counts_ste, norms = widefield.process_counts(
        nv_list, counts, threshold=False
    )

    return avg_counts, avg_counts_ste, norms, freqs, nv_list


def plot_data(nv_list, freqs, avg_counts, avg_counts_ste):
    """
    Plot the raw data for NV centers.

    Args:
        nv_list: List of NV signatures.
        freqs: Frequency data.
        avg_counts: Averaged counts for NVs.
        avg_counts_ste: Standard error of averaged counts.

    Returns:
        fig: The matplotlib figure object containing the raw data plot.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    for nv_idx, nv_sig in enumerate(nv_list):
        ax.errorbar(
            freqs,
            avg_counts[nv_idx],
            yerr=avg_counts_ste[nv_idx],
            label=f"NV {nv_idx+1}",
        )

    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("Normalized NV$^{-}$ Population")
    ax.legend(loc="best", ncol=2, fontsize=8)  # Add legend with NV labels

    plt.title("Raw Data for NV Centers")

    # Manually adjust the layout
    plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.1)

    return fig


def plot_fit_data(nv_list, freqs, avg_counts, avg_counts_ste, norms):
    """
    Plot the fitted data for NV centers.

    Args:
        nv_list: List of NV signatures.
        freqs: Frequency data.
        avg_counts: Averaged counts for NVs.
        avg_counts_ste: Standard error of averaged counts.
        norms: Normalization data.

    Returns:
        fig: The matplotlib figure object containing the fitted data plot.
    """
    fig, axes_pack = plt.subplots(
        len(nv_list), 1, figsize=(10, len(nv_list) * 2), sharex=True
    )

    for nv_idx, ax in enumerate(axes_pack):
        ax.errorbar(
            freqs,
            avg_counts[nv_idx],
            yerr=avg_counts_ste[nv_idx],
            fmt="o",
            label=f"NV {nv_idx+1}",
        )
        # Plot fitted function (placeholder, you can replace with actual fitting code)
        ax.plot(freqs, avg_counts[nv_idx], label="Fit")
        ax.set_ylabel("Norm. NV$^{-}$ Pop.")
        ax.legend(loc="best", fontsize=8)

    ax.set_xlabel("Frequency (GHz)")

    # Adjust the layout manually
    plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.05, hspace=0.4)

    return fig


def visualize_large_nv_data(
    nv_list, freqs, avg_counts, avg_counts_ste, norms, batch_size=25
):
    """
    Visualize large NV datasets by batching NVs and plotting them in separate figures or subplots.

    Args:
        nv_list: List of NV signatures.
        freqs: Frequency data.
        avg_counts: Averaged counts for NVs.
        avg_counts_ste: Standard error of averaged counts.
        norms: Normalization data.
        batch_size: Number of NVs to plot in each batch.

    Returns:
        fig_list: List of figure objects for each batch.
    """
    num_nvs = len(nv_list)
    fig_list = []

    for i in range(0, num_nvs, batch_size):
        batch_nv_list = nv_list[i : i + batch_size]
        batch_avg_counts = avg_counts[i : i + batch_size]
        batch_avg_counts_ste = avg_counts_ste[i : i + batch_size]

        fig = plot_fit_data(
            batch_nv_list, freqs, batch_avg_counts, batch_avg_counts_ste, norms
        )
        fig_list.append(fig)

    return fig_list


def plot_nv_resonance_data(
    nv_list, freqs, avg_counts, avg_counts_ste, file_path, num_cols=4
):
    """
    Plot the NV resonance data in multiple panels (grid) in the same figure and save the figure.

    Args:
        nv_list: List of NV signatures.
        freqs: Frequency data.
        avg_counts: Averaged counts for NVs.
        avg_counts_ste: Standard error of averaged counts.
        file_path: Path where the figure will be saved.
        num_cols: Number of columns for the grid layout.
    """
    # Number of NVs and rows/columns
    num_nvs = len(nv_list)
    num_rows = int(np.ceil(num_nvs / num_cols))  # Calculate the number of rows needed

    # Create a figure with a grid of subplots (num_rows x num_cols)
    fig, axes_pack = plt.subplots(
        num_rows,
        num_cols,
        figsize=(num_cols * 3, num_rows * 1.0),  # Increase height slightly
        sharex=True,
    )
    axes_pack = axes_pack.flatten()  # Flatten the axes array for easy indexing

    # Plot each NV in the corresponding subplot
    for nv_idx, ax in enumerate(axes_pack):
        if nv_idx < num_nvs:
            ax.errorbar(
                freqs,
                avg_counts[nv_idx],
                yerr=avg_counts_ste[nv_idx],
                fmt="o",
                label=f"NV {nv_idx+1}",
                markersize=4,
                color="blue",
            )
            # Auto-scaling of y-axis to fit data better
            ax.set_ylim([min(avg_counts[nv_idx]) - 0.05, max(avg_counts[nv_idx]) + 0.3])

            # Only label the y-axis for the leftmost column
            if nv_idx % num_cols == 0:
                ax.set_ylabel("Norm. NV$^{-}$ Pop.")
            else:
                ax.set_yticklabels([])  # Remove y-axis ticks for non-leftmost subplots

            # Only label the x-axis for the bottom row
            if nv_idx >= (num_rows - 1) * num_cols:
                ax.set_xlabel("Frequency (GHz)")
            else:
                ax.set_xticklabels([])  # Remove x-axis ticks for non-bottom subplots
        else:
            # Hide any unused subplots if the number of NVs is less than the grid size
            ax.axis("off")

    # Adjust layout to ensure nothing overlaps and reduce vertical gaps
    plt.subplots_adjust(
        left=0.1, right=0.95, top=0.95, bottom=0.03, hspace=0.03, wspace=0.03
    )

    # Save the figure to the specified file path
    plt.savefig(file_path, bbox_inches="tight")

    # Close the figure to free up memory
    plt.close(fig)


def plot_nv_resonance_data_sns_with_freq_labels(
    nv_list,
    freqs,
    avg_counts,
    avg_counts_ste,
    file_id,
    file_path,
    num_cols=3,
    threshold_method=None,
):
    """
    Plot the NV resonance data using Seaborn aesthetics in multiple panels (grid) in the same figure,
    add frequency values at the bottom of each column, and save the figure.

    Args:
        nv_list: List of NV signatures.
        freqs: Frequency data.
        avg_counts: Averaged counts for NVs.
        avg_counts_ste: Standard error of averaged counts.
        file_path: Path where the figure will be saved.
        num_cols: Number of columns for the grid layout.
    """
    # Use Seaborn style
    sns.set(style="whitegrid", palette="muted")

    # Number of NVs and rows/columns
    num_nvs = len(nv_list)
    num_rows = int(np.ceil(num_nvs / num_cols))  # Calculate the number of rows needed

    # Create a figure with a grid of subplots (num_rows x num_cols)
    fig, axes_pack = plt.subplots(
        num_rows,
        num_cols,
        figsize=(
            num_cols * 3,
            num_rows * 1,
        ),  # Adjust the figure size for better visibility
        sharex=True,
        sharey=False,  # Allow y-axis to scale individually for each subplot
    )
    axes_pack = axes_pack.flatten()  # Flatten the axes array for easy indexing

    # Set a color palette
    colors = sns.color_palette("deep", num_nvs)

    # Plot each NV in the corresponding subplot
    for nv_idx, ax in enumerate(axes_pack):
        if nv_idx < num_nvs:
            # Use Seaborn's smooth lines with matplotlib markers
            sns.lineplot(
                x=freqs,
                y=avg_counts[nv_idx],
                ax=ax,
                color=colors[nv_idx % len(colors)],
                lw=2,
                marker="o",
                markersize=5,
                label=f"NV {nv_idx+1}",
            )
            # Add error bars manually using matplotlib
            ax.errorbar(
                freqs,
                avg_counts[nv_idx],
                yerr=avg_counts_ste[nv_idx],
                fmt="none",
                ecolor="gray",
                alpha=0.6,
            )

            # Auto-scale y-axis for better view of resonance
            ax.set_ylim([min(avg_counts[nv_idx]), max(avg_counts[nv_idx]) + 0.03])

            # Only set y-tick labels for the leftmost column
            if nv_idx % num_cols == 0:
                ax.set_yticks(
                    ax.get_yticks()
                )  # Keep the default y-tick labels for the leftmost column
            else:
                ax.set_yticklabels([])
            #  Add a single y-axis label for the entire figure
            fig.text(
                0.04,
                0.5,
                "Norm. NV$^{-}$ Pop.",
                va="center",
                rotation="vertical",
                fontsize=12,
            )
            # Only label the x-axis for the bottom row
            if nv_idx >= (num_rows - 1) * num_cols:
                ax.set_xlabel("Frequency (GHz)")
            else:
                ax.set_xticklabels([])  # Remove x-axis ticks for non-bottom subplots

            # Add grid for better visualization
            ax.grid(True, which="both", linestyle="--", linewidth=0.5)
        else:
            # Hide any unused subplots if the number of NVs is less than the grid size
            ax.axis("off")

    # Add frequency values at the bottom of each column
    for col in range(num_cols):
        bottom_row_idx = (
            num_rows * num_cols - num_cols + col
        )  # Index of the bottom row in each column

        if bottom_row_idx < len(axes_pack):  # Ensure the index is within bounds
            ax = axes_pack[bottom_row_idx]
            # Set fewer x-ticks (num_ticks) using np.linspace
            tick_positions = np.linspace(min(freqs), max(freqs), 5)
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(
                [f"{tick:.2f}" for tick in tick_positions], rotation=45, fontsize=9
            )
            # Set the x-ticks to the frequencies for the bottom row of each column
            # ax.set_xticks(freqs)
            # ax.set_xticklabels([f"{f:.2f}" for f in freqs], rotation=45, fontsize=9)
        # Set title for the entire figure
    if threshold_method is not None:
        title = f"NV Resonance Data (Threshold: {threshold_method}, data_id = {file_id}"
        fig.suptitle(title, fontsize=16, y=0.97)
    # Adjust layout to ensure nothing overlaps and reduce vertical gaps
    plt.subplots_adjust(
        left=0.1, right=0.95, top=0.95, bottom=0.1, hspace=0.01, wspace=0.01
    )

    # Save the figure to the specified file path
    # plt.savefig(file_path, bbox_inches="tight")
    if fig is not None:
        dm.save_figure(fig, file_path)
    # Automatically save the plot using the same approach for file paths
    # dm.save_figure(fig, file_path)
    # Close the figure to free up memory
    plt.close(fig)


def plot_nv_resonance_data_sns_with_fit(
    nv_list,
    freqs,
    avg_counts,
    avg_counts_ste,
    file_id,
    file_path,
    num_cols=3,
    threshold_method=None,
):
    """
    Plot the NV resonance data using Seaborn aesthetics in multiple panels (grid) in the same figure,
    add frequency values at the bottom of each column, and generate a separate figure with fit lines.

    Args:
        nv_list: List of NV signatures.
        freqs: Frequency data.
        avg_counts: Averaged counts for NVs.
        avg_counts_ste: Standard error of averaged counts.
        file_path: Path where the figure will be saved.
        num_cols: Number of columns for the grid layout.
    """
    # Normalize counts from 0 to 1
    avg_counts = [(ac - min(ac)) / (max(ac) - min(ac)) for ac in avg_counts]

    # Use Seaborn style
    sns.set(style="whitegrid", palette="muted")

    # Number of NVs and rows/columns
    num_nvs = len(nv_list)
    num_rows = int(np.ceil(num_nvs / num_cols))  # Calculate the number of rows needed

    # Create a figure with a grid of subplots (num_rows x num_cols)
    fig, axes_pack = plt.subplots(
        num_rows,
        num_cols,
        figsize=(
            num_cols * 3,
            num_rows * 1,
        ),  # Adjust the figure size for better visibility
        sharex=True,
        sharey=False,  # Allow y-axis to scale individually for each subplot
    )
    axes_pack = axes_pack.flatten()  # Flatten the axes array for easy indexing

    # Set a color palette
    colors = sns.color_palette("deep", num_nvs)

    fit_fns = []
    popts = []
    center_freqs = []

    # Plot each NV in the corresponding subplot
    for nv_idx, ax in enumerate(axes_pack):
        if nv_idx < num_nvs:
            # Use Seaborn's smooth lines with matplotlib markers
            sns.lineplot(
                x=freqs,
                y=avg_counts[nv_idx],
                ax=ax,
                color=colors[nv_idx % len(colors)],
                lw=2,
                marker="o",
                markersize=5,
                label=f"NV {nv_idx+1}",
            )
            # Add error bars manually using matplotlib
            ax.errorbar(
                freqs,
                avg_counts[nv_idx],
                yerr=avg_counts_ste[nv_idx],
                fmt="none",
                ecolor="gray",
                alpha=0.6,
            )

            # Smart way to set ylim
            ymin = min(avg_counts[nv_idx]) - 0.05 * (
                max(avg_counts[nv_idx]) - min(avg_counts[nv_idx])
            )
            ymax = max(avg_counts[nv_idx]) + 0.05 * (
                max(avg_counts[nv_idx]) - min(avg_counts[nv_idx])
            )
            ax.set_ylim([ymin, ymax])
            ax.set_ylim([0, 1])
            # Fitting part
            num_resonances = 2  # Assuming 2 resonances
            low_freq_guess = freqs[np.argmax(avg_counts[nv_idx][: len(freqs) // 2])]
            high_freq_guess = freqs[
                np.argmax(avg_counts[nv_idx][len(freqs) // 2 :]) + len(freqs) // 2
            ]
            guess_params = [5, 5, low_freq_guess, 5, 5, high_freq_guess]
            bounds = [[0] * len(guess_params), [np.inf] * len(guess_params)]
            for ind in [0, 1, 3, 4]:
                bounds[1][ind] = 10

            def fit_fn(freq, *args):
                return norm_voigt(freq, *args[:3]) + norm_voigt(freq, *args[3:])

            _, popt, _ = fit_resonance(
                freqs,
                avg_counts[nv_idx],
                avg_counts_ste[nv_idx],
                fit_func=fit_fn,
                guess_params=guess_params,
                bounds=bounds,
            )

            # Save fit function and parameters for the fit figure
            fit_fns.append(fit_fn)
            popts.append(popt)
            center_freqs.append((popt[2], popt[5]))

            # Plot fitted data on the same subplot
            # fit_data = fit_fn(freqs, *popt)
            # ax.plot(freqs, fit_data, "--", color="black", label="Fit", lw=1.5)
            fit_data = fit_fn(freqs, *popt)
            ax.plot(
                freqs,
                fit_data,
                "-",
                color=colors[nv_idx % len(colors)],
                label="Fit",
                lw=2,
            )
            # Only set y-tick labels for the leftmost column
            if nv_idx % num_cols == 0:
                ax.set_yticks(
                    ax.get_yticks()
                )  # Keep the default y-tick labels for the leftmost column
            else:
                ax.set_yticklabels([])

            # Add a single y-axis label for the entire figure
            fig.text(
                0.04,
                0.5,
                "Norm. NV$^{-}$ Pop.",
                va="center",
                rotation="vertical",
                fontsize=12,
            )

            # Only label the x-axis for the bottom row
            if nv_idx >= (num_rows - 1) * num_cols:
                ax.set_xlabel("Frequency (GHz)")
            else:
                ax.set_xticklabels([])  # Remove x-axis ticks for non-bottom subplots

            # Add grid for better visualization
            ax.grid(True, which="both", linestyle="--", linewidth=0.5)
        else:
            # Hide any unused subplots if the number of NVs is less than the grid size
            ax.axis("off")

    # Add frequency values at the bottom of each column
    for col in range(num_cols):
        bottom_row_idx = (
            num_rows * num_cols - num_cols + col
        )  # Index of the bottom row in each column
        if bottom_row_idx < len(axes_pack):  # Ensure the index is within bounds
            ax = axes_pack[bottom_row_idx]
            tick_positions = np.linspace(min(freqs), max(freqs), 5)
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(
                [f"{tick:.2f}" for tick in tick_positions], rotation=45, fontsize=9
            )

    # Set title for the entire figure
    if threshold_method is not None:
        title = (
            f"NV Resonance Data (Threshold: {threshold_method}, data_id = {file_id})"
        )
        fig.suptitle(title, fontsize=16, y=0.97)

    # Adjust layout to ensure nothing overlaps and reduce vertical gaps
    plt.subplots_adjust(
        left=0.1, right=0.95, top=0.95, bottom=0.1, hspace=0.01, wspace=0.01
    )

    # Save the figure to the specified file path
    if fig is not None:
        dm.save_figure(fig, file_path)

    # Close the figure to free up memory
    plt.close(fig)


# def plot_nv_resonance_fits_and_residuals(
#     nv_list,
#     freqs,
#     avg_counts,
#     avg_counts_ste,
#     file_id,
#     num_cols=3,
#     threshold_method=None,
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
#     # Use Seaborn style
#     sns.set(style="whitegrid", palette="muted")

#     # Number of NVs and rows/columns
#     num_nvs = len(nv_list)
#     num_rows = int(np.ceil(num_nvs / num_cols))  # Calculate the number of rows needed

#     # Set up colors
#     colors = sns.color_palette("deep", num_nvs)

#     fit_fns = []
#     popts = []
#     pcovs = []
#     residuals_list = []
#     chi_squared_list = []
#     center_freqs = []
#     center_freq_errs = []

#     ### 1. Fitting and Contrast Figure ###
#     fig_fitting, axes_fitting = plt.subplots(
#         num_rows,
#         num_cols,
#         figsize=(num_cols * 3, num_rows * 1),
#         sharex=True,
#         sharey=False,
#     )
#     axes_fitting = axes_fitting.flatten()  # Flatten the axes array for easy indexing

#     # # Define the Voigt function with a background and shared width for both peaks
#     # def voigt_with_background(freq, amp1, amp2, center1, center2, width, bg_offset, bg_slope):
#     #     """Voigt profile for two peaks with a linear background."""
#     #     return (amp1 * norm_voigt(freq, width, width, center1) +
#     #             amp2 * norm_voigt(freq, width, width, center2) +
#     #             bg_offset + bg_slope * freq)

#     def voigt_with_background(
#         freq, amp1, amp2, center1, center2, width, bg_offset, bg_slope
#     ):
#         """Voigt profile for two peaks with a linear background."""
#         freq = np.array(
#             freq
#         )  # Ensure freq is a NumPy array for element-wise operations
#         return (
#             amp1 * norm_voigt(freq, width, width, center1)
#             + amp2 * norm_voigt(freq, width, width, center2)
#             + bg_offset
#             + bg_slope * freq
#         )

#     # Perform the fitting and plot the contrast
#     for nv_idx, ax in enumerate(axes_fitting):
#         if nv_idx < num_nvs:
#             # Plot raw data
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
#             # Add error bars manually using matplotlib
#             ax.errorbar(
#                 freqs,
#                 avg_counts[nv_idx],
#                 yerr=avg_counts_ste[nv_idx],
#                 fmt="none",
#                 ecolor="gray",
#                 alpha=0.6,
#             )

#             # Get NV data and guess initial parameters for fitting
#             nv_counts = avg_counts[nv_idx]
#             nv_counts_ste = avg_counts_ste[nv_idx]

#             # # Initial guess: amplitudes, centers, shared width, and background (linear)
#             # low_freq_guess = freqs[np.argmax(avg_counts[nv_idx][:len(freqs) // 2])]
#             # high_freq_guess = freqs[np.argmax(avg_counts[nv_idx][len(freqs) // 2:]) + len(freqs) // 2]
#             # guess_params = [np.max(nv_counts), np.max(nv_counts), low_freq_guess, high_freq_guess, 5, np.min(nv_counts), 0]

#             # # Bounds for fitting: amplitudes and widths are positive, centers within range
#             # bounds = (
#             #     [0, 0, min(freqs), min(freqs), 0, -np.inf, -np.inf],  # Lower bounds
#             #     [np.inf, np.inf, max(freqs), max(freqs), np.inf, np.inf, np.inf]  # Upper bounds
#             # )

#             # # Fit the Voigt profile with background
#             # popt, pcov = curve_fit(
#             #     voigt_with_background, freqs, nv_counts, p0=guess_params, sigma=nv_counts_ste, bounds=bounds
#             # )

#             # Improved initial guess and relaxed bounds
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

#             # Increased maxfev for more iterations
#             popt, pcov = curve_fit(
#                 voigt_with_background,
#                 freqs,
#                 nv_counts,
#                 p0=guess_params,
#                 sigma=nv_counts_ste,
#                 bounds=bounds,
#                 maxfev=10000,
#             )

#             # Track parameters for plotting and residuals
#             fit_fns.append(voigt_with_background)
#             popts.append(popt)
#             pcovs.append(pcov)

#             # Extract center frequencies for the two peaks
#             center_freqs.append((popt[2], popt[3]))

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

#             # Calculate residuals
#             residuals = nv_counts - fit_data
#             residuals_list.append(residuals)

#             # Compute chi-squared value
#             chi_squared = np.sum((residuals / nv_counts_ste) ** 2)
#             chi_squared_list.append(chi_squared)

#             # Add text annotation for chi-squared value
#             ax.text(
#                 0.05,
#                 0.85,
#                 f"Chi-sq: {chi_squared:.2f}",
#                 transform=ax.transAxes,
#                 fontsize=10,
#                 bbox=dict(facecolor="white", alpha=0.6),
#             )

#             # Y-tick labels for the leftmost column
#             if nv_idx % num_cols == 0:
#                 ax.set_yticks(
#                     ax.get_yticks()
#                 )  # Keep default y-tick labels for leftmost column
#             else:
#                 ax.set_yticklabels([])

#             # Add a single y-axis label for the entire figure
#             fig_fitting.text(
#                 0.04,
#                 0.5,
#                 "NV$^{-}$ Population",
#                 va="center",
#                 rotation="vertical",
#                 fontsize=12,
#             )

#             # Label x-axis for the bottom row
#             if nv_idx >= (num_rows - 1) * num_cols:
#                 ax.set_xlabel("Frequency (GHz)")
#             else:
#                 ax.set_xticklabels([])

#             # Add grid for better visualization
#             ax.grid(True, which="both", linestyle="--", linewidth=0.5)
#         else:
#             ax.axis("off")

#     # Set title for the fitting figure
#     if threshold_method is not None:
#         title = f"NV Resonance Fitting and Contrast (data_id = {file_id})"
#         fig_fitting.suptitle(title, fontsize=16, y=0.97)

#     # Adjust layout for the fitting figure
#     plt.subplots_adjust(
#         left=0.1, right=0.95, top=0.95, bottom=0.1, hspace=0.01, wspace=0.01
#     )
#     now = datetime.now()
#     date_time_str = now.strftime("%Y%m%d_%H%M%S")
#     file_name = dm.get_file_name(file_id=file_id)
#     file_path = dm.get_file_path(__file__, file_name, f"{file_id}_{date_time_str}")
#     # Save the fitting and contrast figure
#     dm.save_figure(fig_fitting, file_path)

#     # Close the fitting figure to free up memory
#     plt.close(fig_fitting)

#     ### 2. Residuals and Contrast Figure ###
#     fig_residuals, axes_residuals = plt.subplots(
#         num_rows,
#         num_cols,
#         figsize=(num_cols * 3, num_rows * 1),
#         sharex=True,
#         sharey=False,
#     )
#     axes_residuals = axes_residuals.flatten()

#     # Plot the residuals and contrast
#     for nv_idx, ax in enumerate(axes_residuals):
#         if nv_idx < num_nvs:
#             # Plot residuals for each NV
#             sns.lineplot(
#                 x=freqs,
#                 y=residuals_list[nv_idx],
#                 ax=ax,
#                 color=colors[nv_idx % len(colors)],
#                 lw=2,
#                 marker="o",
#                 markersize=5,
#                 label=f"NV {nv_idx+1} Residuals",
#             )

#             # Add grid for better visualization
#             ax.grid(True, which="both", linestyle="--", linewidth=0.5)

#             # Add a single y-axis label for the entire figure
#             fig_residuals.text(
#                 0.04,
#                 0.5,
#                 "Residuals",
#                 va="center",
#                 rotation="vertical",
#                 fontsize=12,
#             )

#             # Only label the x-axis for the bottom row
#             if nv_idx >= (num_rows - 1) * num_cols:
#                 ax.set_xlabel("Frequency (GHz)")
#             else:
#                 ax.set_xticklabels([])

#             # Add y-tick labels for the leftmost column only
#             if nv_idx % num_cols == 0:
#                 ax.set_yticks(ax.get_yticks())
#             else:
#                 ax.set_yticklabels([])

#         else:
#             # Hide any unused subplots if there are fewer NVs than grid size
#             ax.axis("off")

#     # Set title for the residuals figure
#     fig_residuals.suptitle(
#         f"NV Resonance Residuals and Contrast (data_id = {file_id})",
#         fontsize=16,
#         y=0.97,
#     )

#     # Adjust layout for the residuals figure
#     plt.subplots_adjust(
#         left=0.1, right=0.95, top=0.95, bottom=0.1, hspace=0.01, wspace=0.01
#     )
#     file_path = dm.get_file_path(
#         __file__, file_name, f"{file_id}_{date_time_str}_residuals"
#     )
#     # Save the residuals and contrast figure
#     dm.save_figure(fig_residuals, file_path)

#     # Close the residuals figure to free up memory
#     plt.close(fig_residuals)


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


def calculate_contrast(amp1, amp2, bg_offset, chi_squared):
    """
    Calculate contrast for an NV center signal.

    Args:
        amp1: Amplitude of the first peak.
        amp2: Amplitude of the second peak.
        bg_offset: Fitted background level (offset).

    Returns:
        Contrast value between 0 and 1.
    """
    alpha0 = (amp1 + amp2) / 2  # Average peak amplitude (bright state)
    alpha1 = bg_offset  # Background level (dark state)

    # Ensure contrast lies between 0 and 1
    # Set contrast to 0 if it exceeds 1
    contrast = max(0, (alpha0 + alpha1) / (alpha1) - 1) if alpha0 > 0 else 0
    if contrast > 0.4:
        contrast = 0
    if chi_squared < 39.0:
        contrast = 0
    # if snr < 1.0:
    #     contrast = 0
    return contrast


def calculate_snr(sig_counts, ref_counts):
    """Calculate the signal-to-noise ratio (SNR)."""
    avg_sig = np.mean(sig_counts)
    avg_ref = np.mean(ref_counts)
    noise = np.sqrt(np.std(sig_counts) ** 2 + np.std(ref_counts) ** 2)
    return (avg_sig - avg_ref) / noise if noise != 0 else 0


def plot_nv_resonance_fits_and_residuals(
    nv_list, freqs, sig_counts, ref_counts, file_id, num_cols=3, threshold_method=None
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
    """
    avg_counts, avg_counts_ste, norms = widefield.process_counts(
        nv_list, sig_counts, ref_counts, threshold=False, method=thresh_method
    )
    # Normalize counts from 0 to 1
    # avg_counts = [(ac) / (max(ac)) for ac in avg_counts]
    from sklearn.cluster import KMeans

    # List to store NV orientations based on fitted peak centers
    orientation_clusters = []
    center_freqs_all = []  # Store center frequencies for clustering
    # Indices to remove
    # indices_to_remove = {9, 26, 33, 35, 36, 45, 47, 53, 56, 58, 59, 60, 61, 62, 63}

    # Filter nv_list, avg_counts, and avg_counts_ste based on indices to keep
    # nv_list = [nv for idx, nv in enumerate(nv_list) if idx not in indices_to_remove]
    # avg_counts = [
    #     ac for idx, ac in enumerate(avg_counts) if idx not in indices_to_remove
    # ]
    # avg_counts_ste = [
    #     ste for idx, ste in enumerate(avg_counts_ste) if idx not in indices_to_remove
    # ]

    # num_nvs = len(nv_list)
    # num_rows = int(np.ceil(num_nvs / num_cols))  # Calculate the number of rows needed

    # Update num_nvs and re-calculate number of rows
    num_nvs = len(nv_list)
    num_rows = int(np.ceil(num_nvs / num_cols))  # Recalculate rows
    sns.set(style="whitegrid", palette="muted")

    colors = sns.color_palette("deep", num_nvs)
    fit_fns = []
    popts = []
    pcovs = []
    residuals_list = []
    chi_squared_list = []
    center_freqs = []
    center_freq_errs = []
    ### 1. Fitting and Contrast Figure ###
    fig_fitting, axes_fitting = plt.subplots(
        num_rows,
        num_cols,
        figsize=(num_cols * 3, num_rows * 1),
        sharex=True,
        sharey=False,
    )
    axes_fitting = axes_fitting.flatten()

    for nv_idx, ax in enumerate(axes_fitting):
        if nv_idx < num_nvs:
            sns.lineplot(
                x=freqs,
                y=avg_counts[nv_idx],
                ax=ax,
                color=colors[nv_idx % len(colors)],
                lw=2,
                marker="o",
                markersize=5,
                label=f"NV {nv_idx+1}",
            )
            ax.errorbar(
                freqs,
                avg_counts[nv_idx],
                yerr=avg_counts_ste[nv_idx],
                fmt="none",
                ecolor="gray",
                alpha=0.6,
            )

            nv_counts = avg_counts[nv_idx]
            nv_counts_ste = avg_counts_ste[nv_idx]

            # Initial guess: amplitudes, centers, shared width, and background (linear)
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

            # Visualize initial guess
            # fit_guess = voigt_with_background(freqs, *guess_params)
            # ax.plot(freqs, fit_guess, '--', label="Initial Guess", color='gray')

            # Perform the least_squares fitting
            result = least_squares(
                residuals_fn,
                guess_params,
                args=(freqs, nv_counts, nv_counts_ste),
                bounds=bounds,
                max_nfev=20000,
            )
            popt = result.x

            # Store center frequencies (two peaks for each NV)
            center_freqs_all.append([popt[2], popt[3]])

            # Track parameters for plotting and residuals
            fit_fns.append(voigt_with_background)
            popts.append(popt)

            center_freqs.append((popt[2], popt[3]))

            # Plot fitted data on the same subplot
            fit_data = voigt_with_background(freqs, *popt)
            ax.plot(
                freqs,
                fit_data,
                "-",
                color=colors[nv_idx % len(colors)],
                label="Fit",
                lw=2,
            )

            residuals = nv_counts - fit_data
            residuals_list.append(residuals)

            # Compute chi-squared value
            chi_squared = np.sum((residuals / nv_counts_ste) ** 2)
            chi_squared_list.append(chi_squared)

            # Calculate contrast
            amp1 = popt[0]  # First peak amplitude
            amp2 = popt[1]  # Second peak amplitude
            bg_offset = popt[5]  # Background level (constant offset)
            # contrast = calculate_contrast(amp1, amp2, bg_offset)
            contrast = calculate_contrast(amp1, amp2, bg_offset, chi_squared)
            # snr = calculate_snr(sig_counts, ref_counts)
            # contrast,_ = widefield.calc_contrast(sig_counts, ref_counts)
            # contrast, contrast_ste = widefield.calc_contrast(
            # sig_counts[nv_idx][np.newaxis, ...],
            # ref_counts[nv_idx][np.newaxis, ...]
            # )
            # snr, snr_ste = widefield.calc_snr(
            # sig_counts[nv_idx][np.newaxis, ...],
            # ref_counts[nv_idx][np.newaxis, ...]
            # )
            # Ensure contrast is a scalar value
            # if isinstance(contrast, np.ndarray):
            #     contrast = np.mean(contrast)

            # contrast, contrast_ste = widefield.calc_contrast(sig_counts[nv_idx], ref_counts[nv_idx])
            # Add contrast to the plot
            # ax.text(
            #     0.05,
            #     0.85,
            #     f"Contrast: {contrast:.2f}",
            #     transform=ax.transAxes,
            #     fontsize=10,
            #     bbox=dict(facecolor="white", alpha=0.6),
            # )

            # Y-tick labels for the leftmost column
            if nv_idx % num_cols == 0:
                ax.set_yticks(ax.get_yticks())
            else:
                ax.set_yticklabels([])

            fig_fitting.text(
                0.04,
                0.5,
                "NV$^{-}$ Population",
                va="center",
                rotation="vertical",
                fontsize=12,
            )

            # if nv_idx >= (num_rows - 1) * num_cols:  # Bottom row
            #     ax.set_xlabel("Frequency (GHz)")
            #     ax.set_xticks(np.linspace(min(freqs), max(freqs), 5))  # Set custom tick locations
            # Add frequency values at the bottom of each column
            for col in range(num_cols):
                bottom_row_idx = (
                    num_rows * num_cols - num_cols + col
                )  # Index of the bottom row in each column
                if bottom_row_idx < len(
                    axes_fitting
                ):  # Ensure the index is within bounds
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

    if threshold_method is not None:
        title = f"NV Resonance Fitting and Contrast (data_id = {file_id})"
        fig_fitting.suptitle(title, fontsize=16, y=0.97)

    plt.subplots_adjust(
        left=0.1, right=0.95, top=0.95, bottom=0.1, hspace=0.01, wspace=0.01
    )

    now = datetime.now()
    date_time_str = now.strftime("%Y%m%d_%H%M%S")
    file_name = dm.get_file_name(file_id=file_id)
    file_path = dm.get_file_path(__file__, file_name, f"{file_id}_{date_time_str}")
    dm.save_figure(fig_fitting, file_path)
    plt.close(fig_fitting)

    ### 2. Residuals and Contrast Figure ###
    fig_residuals, axes_residuals = plt.subplots(
        num_rows,
        num_cols,
        figsize=(num_cols * 3, num_rows * 1),
        sharex=True,
        sharey=False,
    )
    axes_residuals = axes_residuals.flatten()

    for nv_idx, ax in enumerate(axes_residuals):
        if nv_idx < num_nvs:
            sns.lineplot(
                x=freqs,
                y=residuals_list[nv_idx],
                ax=ax,
                color=colors[nv_idx % len(colors)],
                lw=2,
                marker="o",
                markersize=5,
                label=f"NV {nv_idx+1} Residuals",
            )

            ax.grid(True, which="both", linestyle="--", linewidth=0.5)

            fig_residuals.text(
                0.04, 0.5, "Residuals", va="center", rotation="vertical", fontsize=12
            )

            ax.text(
                0.05,
                0.85,
                f"Chi-sq: {chi_squared_list[nv_idx]:.2f}",
                transform=ax.transAxes,
                fontsize=10,
                bbox=dict(facecolor="white", alpha=0.6),
            )

            for col in range(num_cols):
                bottom_row_idx = (
                    num_rows * num_cols - num_cols + col
                )  # Index of the bottom row in each column
                if bottom_row_idx < len(
                    axes_residuals
                ):  # Ensure the index is within bounds
                    ax = axes_residuals[bottom_row_idx]
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
            # if nv_idx >= (num_rows - 1) * num_cols:
            #     ax.set_xlabel("Frequency (GHz)")
            #     ax.set_xticks(np.linspace(min(freqs), max(freqs), 5))  # Set custom tick locations
            else:
                ax.set_xticklabels([])

            if nv_idx % num_cols == 0:
                ax.set_yticks(ax.get_yticks())
            else:
                ax.set_yticklabels([])

        else:
            ax.axis("off")

    fig_residuals.suptitle(
        f"NV Resonance Residuals and Chi-Square (data_id = {file_id})",
        fontsize=16,
        y=0.97,
    )

    plt.subplots_adjust(
        left=0.1, right=0.95, top=0.95, bottom=0.1, hspace=0.01, wspace=0.01
    )
    file_path = dm.get_file_path(
        __file__, file_name, f"{file_id}_{date_time_str}_residuals"
    )
    dm.save_figure(fig_residuals, file_path)
    plt.close(fig_residuals)

    # Calculate differences between the two center frequencies for each NV
    center_freq_differences = [abs(cf[1] - cf[0]) for cf in center_freqs_all]
    fig_freq_diff =   plt.figure(figsize=(6, 5))
    plt.scatter(
    range(len(center_freq_differences)),
    center_freq_differences,
    color='teal',
    alpha=0.8,
    edgecolors='k',
    s=120  # Larger point size for emphasis
)
    plt.xlabel('NV Index', fontsize=12, labelpad=10)
    plt.ylabel('Center Frequency Difference (GHz)', fontsize=12, labelpad=10)
    plt.title('Center Frequency Differences for NV Resonance Fits', fontsize=14, pad=15)

    # Customizing x-axis ticks using tick_positions
    tick_positions = np.linspace(0, len(center_freq_differences) - 1, num=10, dtype=int)  # 5 tick positions evenly spaced
    plt.xticks(
        tick_positions,
        [f'NV {int(i+1)}' for i in tick_positions],
        rotation=45,
        fontsize=10
    )

    plt.yticks(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    file_path = dm.get_file_path(
        __file__, file_name, f"{file_id}_{date_time_str}_center_freq"
    )
    dm.save_figure(fig_freq_diff, file_path)
    plt.close(fig_freq_diff)

def nv_resonance_splitting(
    nv_list, freqs, avg_counts, avg_counts_ste, threshold=0.06, bins=30
):
    """
    Calculate the frequency splitting between two NV resonances by fitting a Voigt profile to two resonance peaks,
    and plot the histogram of frequency splitting values with the given threshold.

    Args:
        nv_list: List of NV signatures.
        freqs: Frequency data.
        avg_counts: Averaged counts for NVs.
        avg_counts_ste: Standard error of averaged counts.
        threshold: Threshold for classifying splitting (default: 0.06 GHz).
        bins: Number of bins for the histogram (default: 30).

    Returns:
        freq_splitting: Frequency splitting values for all NVs.
        fitted_peaks: List of tuples with the two fitted resonance frequencies for each NV.
    """
    # Normalize counts from 0 to 1
    avg_counts = [(ac - min(ac)) / (max(ac) - min(ac)) for ac in avg_counts]

    # Initialize lists to store fit functions, fit parameters, and center frequencies
    fit_fns = []
    popts = []
    center_freqs = []

    for nv_idx in range(len(nv_list)):
        # Fitting part using the provided method
        num_resonances = 2  # Assuming 2 resonances
        low_freq_guess = freqs[np.argmax(avg_counts[nv_idx][: len(freqs) // 2])]
        high_freq_guess = freqs[
            np.argmax(avg_counts[nv_idx][len(freqs) // 2 :]) + len(freqs) // 2
        ]

        # Guess parameters and bounds
        guess_params = [5, 5, low_freq_guess, 5, 5, high_freq_guess]
        bounds = [[0] * len(guess_params), [np.inf] * len(guess_params)]
        for ind in [0, 1, 3, 4]:
            bounds[1][ind] = 10

        def fit_fn(freq, *args):
            return norm_voigt(freq, *args[:3]) + norm_voigt(freq, *args[3:])

        # Fit the resonance frequencies using the Voigt profile
        _, popt, _ = fit_resonance(
            freqs,
            avg_counts[nv_idx],
            avg_counts_ste[nv_idx],
            fit_func=fit_fn,
            guess_params=guess_params,
            bounds=bounds,
        )

        # Save fit function and parameters for the fit figure
        fit_fns.append(fit_fn)
        popts.append(popt)
        center_freqs.append(
            (popt[2], popt[5])
        )  # Extract the two center frequencies for each NV

    # Calculate frequency splitting for each NV
    freq_splitting = [abs(f[1] - f[0]) for f in center_freqs]
    # plot_histogram_with_threshold(freq_splitting, threshold, bins)
    return freq_splitting


def plot_histogram_with_threshold(freq_splitting, threshold, bins=30):
    """
    Plot a histogram of frequency splitting values with a vertical line indicating the threshold.

    Args:
        freq_splitting: List of frequency splitting values for NVs.
        threshold: Threshold value for classification.
        bins: Number of bins for the histogram (default: 30).
    """
    # Create a histogram of the frequency splitting values
    plt.figure(figsize=(8, 6))
    plt.hist(freq_splitting, bins=bins, color="blue", alpha=0.7, edgecolor="black")

    # Plot a vertical line at the threshold
    plt.axvline(
        x=threshold,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Threshold = {threshold}",
    )

    # Add labels and title
    plt.title("Histogram of Frequency Splitting with Threshold", fontsize=14)
    plt.xlabel("Frequency Splitting (GHz)", fontsize=12)
    plt.ylabel("Count", fontsize=12)

    # Add a legend
    plt.legend()

    # Display the plot
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
    file_id = 1663484946120
    file_id = 1695092317631
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

    # fig = plot_nv_resonance_data_sns_with_freq_labels(
    #     nv_list, freqs, avg_counts, avg_counts_ste, file_id, file_path, num_cols=5,threshold_method=thresh_method)
    # fig = plot_nv_resonance_data_sns_with_fit(
    #     nv_list, freqs, avg_counts, avg_counts_ste, file_id, file_path, num_cols=5,threshold_method=thresh_method
    #     )
    thresh_method = "otsu"
    plot_nv_resonance_fits_and_residuals(
        nv_list,
        freqs,
        sig_counts,
        ref_counts,
        file_id,
        num_cols=6,
        threshold_method=thresh_method,
    )
    print(f"Plot saved to {file_path}")
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

    #     # Assuming avg_counts and avg_counts_ste are calculated or loaded elsewhere
    #     avg_counts = np.mean(sig_counts, axis=1)
    #     avg_counts_ste = np.std(sig_counts, axis=1) / np.sqrt(num_runs)
