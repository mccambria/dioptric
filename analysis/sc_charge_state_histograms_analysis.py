# -*- coding: utf-8 -*-
"""
Illuminate an area, collecting onto the camera. Interleave a signal and control sequence
and plot the difference, while fitting a bimodal distribution to NV charge states.

Created on Fall 2024

@author: saroj chand
"""

import os
import sys
import time
import traceback

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import ndimage

from majorroutines.widefield import base_routine
from utils import common, widefield
from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import positioning as pos
from utils import tool_belt as tb
from utils.constants import NVSig, VirtualLaserKey
from analysis import bimodal_histogram
from analysis.bimodal_histogram import (
    ProbDist,
    determine_threshold,
    fit_bimodal_histogram,
)


# Update the plot_histograms function for better visualization
def plot_histograms(
    sig_counts_list,
    ref_counts_list,
    no_title=True,
    no_text=None,
    ax=None,
    density=False,
    nv_index=None,
):
    """Plot histograms for signal and reference counts with enhanced visualization."""
    sns.set_theme(style="whitegrid")  # Use a Seaborn theme for improved aesthetics

    laser_key = VirtualLaserKey.WIDEFIELD_CHARGE_READOUT
    laser_dict = tb.get_virtual_laser_dict(laser_key)
    readout = laser_dict["duration"]
    readout_ms = int(readout / 1e6)
    readout_s = readout / 1e9

    ### Histograms
    num_reps = len(ref_counts_list)
    labels = ["With ionization pulse", "Without ionization pulse"]
    colors = sns.color_palette("husl", 2)  # Use Seaborn color palette
    counts_lists = [sig_counts_list, ref_counts_list]

    if ax is None:
        fig, ax = plt.subplots()  # Larger figure size for clarity
    else:
        fig = None

    if not no_title:
        ax.set_title(
            f"Charge Prep Histogram ({num_reps} reps)", fontsize=14, weight="bold"
        )

    ax.set_xlabel("Integrated Counts", fontsize=12)
    ax.set_ylabel("Probability" if density else "Occurrences", fontsize=12)

    for ind, counts_list in enumerate(counts_lists):
        sns.histplot(
            counts_list,
            kde=False,
            stat="density" if density else "count",
            bins=50,
            ax=ax,
            label=labels[ind],
            color=colors[ind],
            alpha=0.7,
        )

    ax.legend(title="Pulse Type", fontsize=10, loc="upper right", title_fontsize=12)

    if fig is not None:
        return fig


# Update scatter plot aesthetics
def scatter_plot(x_data, y_data, xlabel, ylabel, title):
    """Create a scatter plot with purple markers and transparent filling."""
    plt.figure(figsize=(6, 5))
    plt.scatter(
        x_data,
        y_data,
        edgecolors="darkblue",  #  circle outlines
        facecolors="skyblue",  #  fill color
        alpha=0.8,  # Transparency for the filling
        s=60,  # Marker size
        linewidth=0.8,  # Outline thickness
    )
    plt.title(title, fontsize=18)
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()


# Update the image plotting function for improved visuals
def plot_images(img_arrays, readout_laser, readout_ms, title_suffixes):
    """Plot images with improved Seaborn style."""
    sns.set_theme(style="darkgrid")  # Change to a darker grid style for images
    img_figs = []

    for ind, img_array in enumerate(img_arrays):
        title_suffix = title_suffixes[ind]
        fig, ax = plt.subplots()
        sns.heatmap(
            img_array,
            ax=ax,
            cmap="viridis",
            cbar_kws={"label": "Photons"},
            annot=False,
        )
        ax.set_title(
            f"{readout_laser}, {readout_ms:.2f} ms, {title_suffix}", fontsize=14
        )
        img_figs.append(fig)

    return img_figs


# # Process and plot function and Set Seaborn theme globally for consistent styling
sns.set_theme(style="whitegrid")


def process_and_plot(
    raw_data, do_plot_histograms=False, prob_dist: ProbDist = ProbDist.COMPOUND_POISSON
):
    ### Setup
    nv_list = raw_data["nv_list"]
    num_nvs = len(nv_list)
    counts = np.array(raw_data["counts"])
    sig_counts_lists = [counts[0, nv_ind].flatten() for nv_ind in range(num_nvs)]
    ref_counts_lists = [counts[1, nv_ind].flatten() for nv_ind in range(num_nvs)]
    num_reps = raw_data["num_reps"]
    num_runs = raw_data["num_runs"]
    num_shots = num_reps * num_runs

    ### Histograms and thresholding
    threshold_list = []
    readout_fidelity_list = []
    prep_fidelity_list = []
    hist_figs = []

    for ind in range(num_nvs):
        sig_counts_list = sig_counts_lists[ind]
        ref_counts_list = ref_counts_lists[ind]

        # Only use ref counts for threshold determination
        popt, _, red_chi_sq = fit_bimodal_histogram(
            ref_counts_list, prob_dist, no_print=True
        )
        threshold, readout_fidelity = determine_threshold(
            popt, prob_dist, dark_mode_weight=0.5, do_print=False, ret_fidelity=True
        )
        threshold_list.append(threshold)
        readout_fidelity_list.append(readout_fidelity)
        if popt is not None:
            prep_fidelity = 1 - popt[0]
        else:
            prep_fidelity = np.nan
        prep_fidelity_list.append(prep_fidelity)

        # Plot histograms
        if do_plot_histograms:
            fig = plot_histograms(sig_counts_list, ref_counts_list, density=True)
            if fig is not None:
                hist_figs.append(fig)

    # Report averages
    avg_readout_fidelity = np.nanmean(readout_fidelity_list)
    avg_prep_fidelity = np.nanmean(prep_fidelity_list)
    print(f"Average Readout Fidelity: {avg_readout_fidelity:.3f}")
    print(f"Average NV- Preparation Fidelity: {avg_prep_fidelity:.3f}")

    # Scatter plot: Readout fidelity vs Prep fidelity
    scatter_plot(
        readout_fidelity_list,
        prep_fidelity_list,
        xlabel="Readout Fidelity",
        ylabel="NV- Preparation Fidelity",
        title="Readout vs Prep Fidelity",
    )

    # Scatter plot: Distance from center vs Prep fidelity
    coords_key = "laser_INTE_520_aod"
    distances = [
        np.sqrt(
            (110 - pos.get_nv_coords(nv, coords_key, drift_adjust=False)[0]) ** 2
            + (110 - pos.get_nv_coords(nv, coords_key, drift_adjust=False)[1]) ** 2
        )
        for nv in nv_list
    ]
    scatter_plot(
        distances,
        prep_fidelity_list,
        xlabel="Distance from Center (MHz)",
        ylabel="NV- Preparation Fidelity",
        title="Prep Fidelity vs Distance",
    )

    # Image plotting
    if "img_arrays" not in raw_data:
        return

    laser_key = VirtualLaserKey.WIDEFIELD_CHARGE_READOUT
    laser_dict = tb.get_virtual_laser_dict(laser_key)
    readout_laser = laser_dict["physical_name"]
    readout_ms = laser_dict["duration"] / 10**6

    img_arrays = raw_data["img_arrays"]
    mean_img_arrays = np.mean(img_arrays, axis=(1, 2, 3))
    sig_img_array = mean_img_arrays[0]
    ref_img_array = mean_img_arrays[1]
    diff_img_array = sig_img_array - ref_img_array
    img_arrays_to_save = [sig_img_array, ref_img_array, diff_img_array]
    title_suffixes = ["Signal", "Reference", "Difference"]

    img_figs = plot_images(
        img_arrays_to_save, readout_laser, readout_ms, title_suffixes
    )

    return img_arrays_to_save, img_figs, hist_figs


if __name__ == "__main__":
    kpl.init_kplotlib()
    data = dm.get_raw_data(file_id=1713224279642, load_npz=False)
    process_and_plot(data, do_plot_histograms=False)
    kpl.show(block=True)
    plt.show(block=True)
