# -*- coding: utf-8 -*-
"""
Spin Echo Analysis and Visualization

Created on December 22nd, 2024

@author: Saroj Chand
"""

import time
import traceback
import sys
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.optimize import curve_fit
from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import widefield as widefield


# Define a decay model for spin echo fitting
def quartic_decay(tau, baseline, revival_time, decay_time, amp1, amp2, freq):
    num_revivals = 3
    value = baseline
    for i in range(num_revivals):
        exp_decay = np.exp(-(((tau - i * revival_time) / decay_time) ** 2))
        mod = amp1 * np.cos(2 * np.pi * freq * tau)
        value -= exp_decay * mod
    return value


def generate_initial_guess_and_bounds(tau, counts):
    baseline_guess = np.mean(counts[-10:])  # Use the end of the data for baseline
    revival_time_guess = np.mean(np.diff(tau)) * 2  # Estimated period
    decay_time_guess = (tau[-1] - tau[0]) / 3  # Approximate decay time
    amp1_guess = (max(counts) - min(counts)) / 2
    amp2_guess = amp1_guess / 2

    # Frequency guess based on FFT
    fft_freqs = np.fft.rfftfreq(len(tau), d=(tau[1] - tau[0]))
    fft_spectrum = np.abs(np.fft.rfft(counts - baseline_guess))
    freq_guess = fft_freqs[np.argmax(fft_spectrum)]

    initial_guess = [
        baseline_guess,
        revival_time_guess,
        decay_time_guess,
        amp1_guess,
        amp2_guess,
        freq_guess,
    ]

    # Adjust bounds if needed
    bounds = (
        [0, tau[1] - tau[0], 0, 0, 0, 0],  # Lower bounds
        [1, tau[-1], np.inf, np.inf, np.inf, np.inf],  # Upper bounds
    )

    return initial_guess, bounds


# Combine data from multiple file IDs
def process_multiple_files(file_ids):
    """
    Load and combine data from multiple file IDs.

    Args:
        file_ids (list): List of file IDs to process.

    Returns:
        dict: Combined data.
    """
    combined_data = dm.get_raw_data(file_id=file_ids[0])
    for file_id in file_ids[1:]:
        new_data = dm.get_raw_data(file_id=file_id)
        combined_data["num_runs"] += new_data["num_runs"]
        combined_data["counts"] = np.append(
            combined_data["counts"], new_data["counts"], axis=1
        )
    return combined_data


# Analyze and visualize spin echo data
def analyze_spin_echo(nv_list, taus, norm_counts, norm_counts_ste):
    fit_params = []
    chi_squared_values = []
    parameters = []
    sns.set(style="whitegrid", palette="muted")
    num_nvs = len(nv_list)
    colors = sns.color_palette("deep", num_nvs)
    num_cols = 9
    num_rows = int(np.ceil(len(nv_list) / num_cols))

    # Full plot
    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=(num_cols * 3, num_rows * 1),
        sharex=True,
        sharey=False,
    )
    axes = axes.flatten()

    # Zoomed-in plot
    zoom_fig, zoom_axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=(num_cols * 3, num_rows * 1),
        sharex=True,
        sharey=False,
    )
    zoom_axes = zoom_axes.flatten()

    for nv_idx, (ax, zoom_ax) in enumerate(zip(axes, zoom_axes)):
        if nv_idx >= len(nv_list):
            ax.axis("off")
            zoom_ax.axis("off")
            continue

        nv_tau = taus
        nv_counts = norm_counts[nv_idx]
        try:
            initial_guess, bounds = generate_initial_guess_and_bounds(nv_tau, nv_counts)
            popt, pcov = curve_fit(
                quartic_decay, nv_tau, nv_counts, p0=initial_guess, bounds=bounds
            )
            fit_params.append(popt)

            # Compute residuals and chi-squared
            residuals = nv_counts - quartic_decay(nv_tau, *popt)
            chi_sq = np.sum((residuals / np.std(residuals)) ** 2)
            degrees_of_freedom = len(nv_tau) - len(popt)
            # if degrees_of_freedom > 0:
            red_chi_sq = chi_sq / degrees_of_freedom
            chi_squared_values.append(red_chi_sq)
            # Extract meaningful parameters
            baseline, revival_time, decay_time, amp1, amp2, freq = popt
            parameters.append(
                {
                    "NV Index": nv_idx,
                    "Baseline": baseline,
                    "Revival Time (µs)": revival_time,
                    "Decay Time (µs)": decay_time,
                    "Amplitude 1": amp1,
                    "Amplitude 2": amp2,
                    "Frequency (Hz)": freq,
                    "Chi-Squared": red_chi_sq,
                }
            )

            # Plot data and fit on full plot
            sns.lineplot(
                x=nv_tau,
                y=nv_counts,
                ax=ax,
                color=colors[nv_idx % len(colors)],
                lw=2,
                marker="o",
                markersize=3,
                label=f"NV {nv_idx}",
            )
            ax.plot(
                nv_tau,
                quartic_decay(nv_tau, *popt),
                "-",
                color=colors[nv_idx % len(colors)],
                label="Fit",
                lw=2,
            )
            ax.errorbar(
                nv_tau,
                norm_counts[nv_idx],
                yerr=norm_counts_ste[nv_idx],
                fmt="none",
                ecolor="gray",
                alpha=0.6,
            )
            ax.legend(fontsize="small")
            ax.grid(True, which="both", linestyle="--", linewidth=0.5)
            ax.set_yticklabels([])

            # Plot zoomed-in data and fit
            sns.lineplot(
                x=nv_tau,
                y=nv_counts,
                ax=zoom_ax,
                color=colors[nv_idx % len(colors)],
                lw=2,
                marker="o",
                markersize=3,
                label=f"NV {nv_idx}",
            )
            zoom_ax.plot(
                nv_tau,
                quartic_decay(nv_tau, *popt),
                "-",
                color=colors[nv_idx % len(colors)],
                label="Fit",
                lw=2,
            )
            zoom_ax.set_xlim(min(nv_tau) + 5e3, min(nv_tau) + 1e4)  # Example zoom range
            # Dynamically adjust the zoom range for the first echo (e.g., around the first revival period)
            zoom_range_start = min(nv_tau) + 5e3  # Start range for zoom
            zoom_range_end = min(nv_tau) + 51.5e3  # Use revival time to estimate range
            zoom_ax.set_xlim(zoom_range_start, zoom_range_end)
            zoom_ax.errorbar(
                nv_tau,
                norm_counts[nv_idx],
                yerr=norm_counts_ste[nv_idx],
                fmt="none",
                ecolor="gray",
                alpha=0.6,
            )
            zoom_ax.legend(fontsize="small")
            zoom_ax.grid(True, which="both", linestyle="--", linewidth=0.5)
            zoom_ax.set_yticklabels([])

        except Exception as e:
            print(f"Fit failed for NV {nv_idx}: {e}")
            fit_params.append(None)
            chi_squared_values.append(np.inf)

    print("Extracted meaningful parameters for each NV center:")
    for params in parameters:
        print(params)

    fig.text(
        0.08,
        0.5,
        "NV$^{-}$ Population",
        va="center",
        rotation="vertical",
        fontsize=12,
    )
    zoom_fig.text(
        0.08,
        0.5,
        "NV$^{-}$ Population (Zoomed)",
        va="center",
        rotation="vertical",
        fontsize=12,
    )

    for col in range(num_cols):
        bottom_row_idx = num_rows * num_cols - num_cols + col
        if bottom_row_idx < len(axes):
            ax = axes[bottom_row_idx]
            tick_positions = np.linspace(min(taus), max(taus), 5)
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(
                [f"{tick:.2f}" for tick in tick_positions],
                rotation=45,
                fontsize=9,
            )
            ax.set_xlabel("Time (µs)")

            zoom_ax = zoom_axes[bottom_row_idx]
            zoom_ax.set_xticks(tick_positions)
            zoom_ax.set_xticklabels(
                [f"{tick:.2f}" for tick in tick_positions],
                rotation=45,
                fontsize=9,
            )
            zoom_ax.set_xlabel("Time (µs)")
        else:
            ax.set_xticklabels([])
            zoom_ax.set_xticklabels([])

    fig.suptitle("Spin Echo Fits", fontsize=16)
    zoom_fig.suptitle("Zoomed Spin Echo Fits", fontsize=16)

    plt.subplots_adjust(
        left=0.1, right=0.95, top=0.95, bottom=0.1, hspace=0.01, wspace=0.01
    )
    plt.show()
    return parameters


def plot_analysis_parameters(meaningful_parameters):

    import pandas as pd

    params_df = pd.DataFrame(meaningful_parameters)

    # Define the parameters to plot
    plot_columns = [
        "Revival Time (µs)",
        "Decay Time (µs)",
        "Frequency (Hz)",
        "Chi-Squared",
    ]

    # Remove outliers using IQR
    def remove_outliers(data):
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        return data[(data >= lower_bound) & (data <= upper_bound)]

    # Filter outliers for each parameter
    filtered_params = params_df.copy()
    for param in plot_columns:
        filtered_params[param] = remove_outliers(params_df[param])

    # Set up the figure
    sns.set(style="whitegrid", palette="muted")
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.flatten()

    # Create scatter plots for each parameter
    for i, param in enumerate(plot_columns):
        if i < len(axes):
            ax = axes[i]
            sns.scatterplot(
                x=filtered_params.index,
                y=filtered_params[param],
                ax=ax,
                marker="o",
                edgecolor="w",
                s=50,
            )
            ax.set_title(param)
            ax.set_xlabel("NV Index")
            ax.set_ylabel(param)

    # Adjust layout
    plt.tight_layout()
    plt.suptitle(
        "Scatter Plots of Fitted Parameters (Outliers Removed)", fontsize=16, y=1.02
    )
    plt.subplots_adjust(top=0.9)
    plt.show()


if __name__ == "__main__":
    kpl.init_kplotlib()

    # Define the file IDs to process
    file_ids = [
        1734158411844,
        1734273666255,
        1734371251079,
        1734461462293,
    ]
    # Process and analyze data from multiple files
    try:
        data = process_multiple_files(file_ids)
        nv_list = data["nv_list"]
        taus = np.array(data["taus"])
        counts = np.array(data["counts"])
        sig_counts, ref_counts = counts[0], counts[1]
        norm_counts, norm_counts_ste = widefield.process_counts(
            nv_list, sig_counts, ref_counts, threshold=False
        )
        nv_num = len(nv_list)
        ids_num = len(file_ids)
        norm_counts_adjusted = np.mean(norm_counts.reshape(nv_num, ids_num, -1), axis=1)
        norm_counts_ste_adjusted = np.mean(
            norm_counts_ste.reshape(nv_num, ids_num, -1), axis=1
        )
        parameters = analyze_spin_echo(
            nv_list, taus, norm_counts_adjusted, norm_counts_ste_adjusted
        )
        plot_analysis_parameters(parameters)
    except Exception as e:
        print(f"Error occurred: {e}")
        print(traceback.format_exc())

    kpl.show(block=True)
