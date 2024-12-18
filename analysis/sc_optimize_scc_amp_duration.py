# -*- coding: utf-8 -*-
"""
Optimize SCC parameters

Created on December 6th, 2023

@author: mccambria
"""

import traceback

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.optimize import curve_fit

from majorroutines.widefield import base_routine
from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import tool_belt as tb
from utils import widefield as widefield
from utils.constants import NVSig, VirtualLaserKey


def process_and_plot(data):
    """
    Process and save average counts and standard errors for the SCC optimization experiment.

    Parameters
    ----------
    data : dict
        Dictionary containing experiment data.
    selected_orientations : list
        List of selected orientations (e.g., ["0.041", "0.147"]).

    Returns
    -------
    processed_data : dict
        Processed data for further analysis.
    figs : list
        List of matplotlib figures generated during plotting.
    """
    # Parse input data
    nv_list = data["nv_list"]
    counts = np.array(data["counts"])
    sig_counts = counts[0]
    ref_counts = counts[1]
    step_vals = np.array(data["step_vals"])
    duration_vals = np.unique(step_vals[:, 0])
    amp_vals = np.unique(step_vals[:, 1])

    # Filter NVs by selected orientations
    orientation_data = dm.get_raw_data(file_id=1723161184641)
    orientation_indices = orientation_data["orientation_indices"]
    selected_orientations = ["0.041", "0.147"]
    selected_indices = []
    for orientation in selected_orientations:
        if str(orientation) in orientation_indices:
            selected_indices.extend(orientation_indices[str(orientation)]["nv_indices"])
    selected_indices = list(set(selected_indices))  # Remove duplicates

    # Filter counts and NV list
    nv_list = [nv_list[i] for i in selected_indices]
    sig_counts = sig_counts[selected_indices, :, :, :]
    ref_counts = ref_counts[selected_indices, :, :, :]

    # Standard errors for signal and reference counts
    # avg_sig_counts, avg_sig_counts_ste = widefield.average_counts(sig_counts)
    # avg_ref_counts, avg_ref_counts_ste = widefield.average_counts(ref_counts)
    avg_snr, avg_snr_ste = widefield.calc_snr(sig_counts, ref_counts)
    avg_counts, avg_counts_ste, norms = widefield.average_counts(sig_counts, ref_counts)
    # Reshape data into 2D grids
    num_dur_steps = len(duration_vals)
    num_amp_steps = len(amp_vals)
    avg_counts_grid = avg_counts.reshape(
        len(selected_indices), num_dur_steps, num_amp_steps
    )
    avg_coutns_ste_grid = avg_counts_ste.reshape(
        len(selected_indices), num_dur_steps, num_amp_steps
    )

    avg_snr_grid = avg_snr.reshape(len(selected_indices), num_dur_steps, num_amp_steps)
    avg_snr_ste_grid = avg_snr_ste.reshape(
        len(selected_indices), num_dur_steps, num_amp_steps
    )

    # Save all processed data
    processed_data = {
        "nv_list": nv_list,
        "step_vals": step_vals,
        "avg_counts_grid": avg_counts_grid,
        "avg_coutns_ste_grid": avg_coutns_ste_grid,
        "avg_snr_grid": avg_snr_grid,
        "avg_snr_ste_grid": avg_snr_ste_grid,
        "amp_vals": amp_vals,
        "duration_vals": duration_vals,
    }

    # # Save data to a file
    # timestamp = dm.get_time_stamp()
    # file_name = dm.get_file_name(file_id=1723161184641)
    # file_path = dm.get_file_path(__file__, timestamp, f"{file_name}_processed")
    # dm.save_raw_data(processed_data, file_path)
    # print(f"Processed data saved to: {file_path}")

    median_snr_grid = np.median(avg_snr_grid, axis=0)
    # Plot the heatmap for the median SNR
    fig, ax = plt.subplots()
    cax = ax.imshow(
        median_snr_grid,
        extent=(
            amp_vals.min(),
            amp_vals.max(),
            duration_vals.min(),
            duration_vals.max(),
        ),
        aspect="auto",
        cmap="coolwarm",
        origin="lower",
    )
    ax.set_title("Median SCC SNR Across NVs")
    ax.set_xlabel("SCC Amplitude")
    ax.set_ylabel("SCC Duration (ns)")
    fig.colorbar(cax, label="Median SCC SNR")
    plt.show()
    # Visualization (optional)
    # figs = []
    # for nv_idx, snr_2d in enumerate(avg_snr_grid):
    #     fig, ax = plt.subplots()
    #     cax = ax.imshow(
    #         snr_2d,
    #         extent=(
    #             amp_vals.min(),
    #             amp_vals.max(),
    #             duration_vals.min(),
    #             duration_vals.max(),
    #         ),
    #         aspect="auto",
    #         cmap="viridis",
    #     )
    #     ax.set_title(f"NV {nv_idx} - SNR Heatmap")
    #     ax.set_xlabel("Amplitude")
    #     ax.set_ylabel("Duration")
    #     fig.colorbar(cax, label="SNR")
    #     plt.show()
    #     figs.append(fig)

    return processed_data, fig


def analyze_and_visualize(processed_data):
    """
    Perform additional analysis and generate advanced visualizations.

    Parameters
    ----------
    processed_data : dict
        Dictionary containing processed data.
    """
    # Extract data from the dictionary
    avg_coutns_grid = processed_data["avg_coutns_grid"]
    avg_coutns_ste_grid = processed_data["avg_coutns_ste_grid"]
    avg_snr_grid = processed_data["avg_snr_grid"]
    avg_snr_ste_grid = processed_data["avg_snr_ste_grid"]
    amp_vals = np.array(processed_data["amp_vals"])
    duration_vals = np.array(processed_data["duration_vals"])

    # Compute the median SNR across NVs
    median_snr_grid = np.median(avg_snr_grid, axis=0)
    mean_snr_grid = np.mean(avg_snr_grid, axis=0)

    # Plot the heatmap for the median SNR
    fig, ax = plt.subplots()
    cax = ax.imshow(
        median_snr_grid,
        extent=(
            amp_vals.min(),
            amp_vals.max(),
            duration_vals.min(),
            duration_vals.max(),
        ),
        aspect="auto",
        cmap="coolwarm",
        origin="lower",
    )
    ax.set_title("Median SCC SNR Across NVs")
    ax.set_xlabel("SCC Amplitude")
    ax.set_ylabel("SCC Duration (ns)")
    fig.colorbar(cax, label="Median SCC SNR")
    plt.show()

    # Plot Median SNR vs. SCC Amplitude for each duration
    fig1, ax1 = plt.subplots()
    for i, duration in enumerate(duration_vals):
        ax1.plot(amp_vals, median_snr_grid[i, :], label=f"{duration}", marker="o")
    ax1.set_xlabel("SCC Amplitude")
    ax1.set_ylabel("Median SCC SNR")
    ax1.set_title("Median SCC SNR Across NVs")
    ax1.legend(title="Durations (ns)", fontsize=9, title_fontsize=10)
    plt.show()

    # Plot Median SNR vs. SCC Duration for each amplitude
    fig2, ax2 = plt.subplots()
    for j, amplitude in enumerate(amp_vals):
        ax2.plot(
            duration_vals,
            median_snr_grid[:, j],
            label=f"{amplitude:.2f}",
            marker="o",
        )
    ax2.set_xlabel("SCC Duration")
    ax2.set_ylabel("Median SCC SNR")
    ax2.set_title("Median SCC SNR Across NVs")
    ax2.legend(title="Amplitude (relative)", fontsize=9, title_fontsize=10)
    plt.show()


# Individual NV SNR line plots
# for nv_ind, nv_snr in enumerate(avg_snr):
#     plt.figure(figsize=(8, 6))
#     plt.errorbar(
#         duration_vals,
#         nv_snr,
#         yerr=avg_snr_ste[nv_ind],
#         label=f"NV {nv_ind + 1}",
#         fmt="o",
#     )
#     plt.xlabel("Pulse Duration (ns)")
#     plt.ylabel("SNR")
#     plt.title(f"NV {nv_ind + 1} SNR vs. Pulse Duration")
#     plt.grid(alpha=0.3)
#     plt.legend()
#     plt.show()


if __name__ == "__main__":
    kpl.init_kplotlib()
    file_id = 1722903695939
    # file_id = 1727140766217
    data = dm.get_raw_data(file_id=file_id)
    process_and_plot(data)
    # # processed data analysis
    # processed_data_id = 1723819842491
    # processed_data_id = 1723904775619
    # processed_data = dm.get_raw_data(file_id=processed_data_id)
    # analyze_and_visualize(processed_data)
    plt.show(block=True)
