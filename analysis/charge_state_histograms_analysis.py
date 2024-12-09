# -*- coding: utf-8 -*-
"""
Illuminate an area, collecting onto the camera. Interleave a signal and control sequence
and plot the difference, while fitting a bimodal distribution to NV charge states.

Created on Fall 2023

@author: mccambria
"""

import os
import sys
import time
import traceback

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
from scipy.special import factorial

from majorroutines.widefield import base_routine
from utils import common, widefield
from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import positioning as pos
from utils import tool_belt as tb
from utils.constants import NVSig, VirtualLaserKey
from analysis.bimodal_histogram import (
    ProbDist,
    determine_threshold,
    fit_bimodal_histogram,
)

# region Process and fitting functions


def bimodal_gaussian(x, mu1, sigma1, mu2, sigma2, w_nv_minus):
    """Model for bimodal Gaussian distribution."""
    w_nv_zero = 1 - w_nv_minus  # Ensure weights sum to 1
    g1 = (
        w_nv_zero
        * np.exp(-0.5 * ((x - mu1) / sigma1) ** 2)
        / (sigma1 * np.sqrt(2 * np.pi))
    )
    g2 = (
        w_nv_minus
        * np.exp(-0.5 * ((x - mu2) / sigma2) ** 2)
        / (sigma2 * np.sqrt(2 * np.pi))
    )
    return g1 + g2


def fit_bimodal_distribution(counts):
    """Fit a bimodal Gaussian distribution to the provided counts."""
    counts = np.array(counts)
    smoothed_counts = gaussian_filter1d(counts, sigma=2)  # Smooth data to reduce noise

    # Generate histogram data
    y_data, bin_edges = np.histogram(smoothed_counts, bins=50, density=True)
    x_data = 0.5 * (bin_edges[:-1] + bin_edges[1:])  # Bin centers

    # Initial guesses: mu1, sigma1, mu2, sigma2, w_nv_minus
    initial_guess = [
        np.mean(counts) - 10,
        np.std(counts) / 2,
        np.mean(counts) + 10,
        np.std(counts),
        0.7,
    ]
    bounds = (
        [0, 0, 0, 0, 0.5],
        [np.inf, np.inf, np.inf, np.inf, 0.9],
    )  # Constrain weights

    try:
        params, _ = curve_fit(
            bimodal_gaussian, x_data, y_data, p0=initial_guess, bounds=bounds
        )
        return params
    except RuntimeError as e:
        print(f"Fitting failed: {e}")
        return None


def plot_histograms(
    sig_counts_list,
    ref_counts_list,
    no_title=True,
    no_text=None,
    ax=None,
    density=False,
    nv_index=None,  # Add NV index as an optional parameter
):
    """Plot histograms for signal and reference counts."""
    laser_key = VirtualLaserKey.WIDEFIELD_CHARGE_READOUT
    laser_dict = tb.get_virtual_laser_dict(laser_key)
    readout = laser_dict["duration"]
    readout_ms = int(readout / 1e6)
    readout_s = readout / 1e9

    ### Histograms
    num_reps = len(ref_counts_list)
    labels = ["With ionization pulse", "Without ionization pulse"]
    colors = [kpl.KplColors.RED, kpl.KplColors.GREEN]
    counts_lists = [sig_counts_list, ref_counts_list]

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = None
    if not no_title:
        ax.set_title(f"Charge prep hist, {num_reps} reps")
    ax.set_xlabel("Integrated counts")
    if density:
        ax.set_ylabel("Probability")
    else:
        ax.set_ylabel("Number of occurrences")

    for ind in range(2):
        counts_list = counts_lists[ind]
        label = labels[ind]
        color = colors[ind]
        kpl.histogram(ax, counts_list, label=label, color=color, density=density)

    ax.legend()

    if fig is not None:
        return fig


def process_and_plot(raw_data, do_plot_histograms=True):
    """Process data, fit histograms, and plot results."""
    nv_list = raw_data["nv_list"]
    num_nvs = len(nv_list)
    counts = np.array(raw_data["counts"])
    sig_counts_lists = [counts[0, nv_ind].flatten() for nv_ind in range(num_nvs)]
    ref_counts_lists = [counts[1, nv_ind].flatten() for nv_ind in range(num_nvs)]

    threshold_list = []
    readout_fidelity_list = []
    prep_fidelity_list = []
    hist_figs = []

    for ind in range(num_nvs):
        sig_counts_list = sig_counts_lists[ind]
        ref_counts_list = ref_counts_lists[ind]

        # Fit bimodal distribution
        params = fit_bimodal_distribution(ref_counts_list)
        if params is not None:
            mu1, sigma1, mu2, sigma2, w_nv_minus = params
            print(
                f"Fitted parameters for NV{ind}: mu1={mu1}, mu2={mu2}, w_nv_minus={w_nv_minus}"
            )

        threshold, readout_fidelity = determine_threshold(
            ref_counts_list, nvn_ratio=0.5, no_print=True, ret_fidelity=True
        )
        threshold_list.append(threshold)
        readout_fidelity_list.append(readout_fidelity)
        popt = fit_histogram(ref_counts_list, no_print=True)
        if popt is not None:
            prep_fidelity = 1 - popt[0]
        else:
            prep_fidelity = np.nan
        prep_fidelity_list.append(prep_fidelity)

        if do_plot_histograms:
            fig = plot_histograms(sig_counts_list, ref_counts_list, density=True)
            ax = fig.gca()
            if popt is not None:
                x_vals = np.linspace(0, np.max(ref_counts_list), 1000)
                kpl.plot_line(ax, x_vals, tb.bimodal_skew_gaussian(x_vals, *popt))

            try:
                if threshold is not None:
                    ax.axvline(threshold, color=kpl.KplColors.GRAY, ls="dashed")
            except TypeError as e:
                print(f"Could not add threshold line due to: {e}")

            snr_str = f"NV{ind}\nReadout fidelity: {round(readout_fidelity, 3)}\nCharge prep. fidelity {round(prep_fidelity, 3)}"
            kpl.anchored_text(ax, snr_str, "center right", size=kpl.Size.SMALL)
            kpl.show()

            if fig is not None:
                hist_figs.append(fig)

    avg_readout_fidelity = np.nanmean(readout_fidelity_list)
    avg_prep_fidelity = np.nanmean(prep_fidelity_list)
    print(f"Average readout fidelity: {avg_readout_fidelity:.3f}")
    print(f"Average NV- preparation fidelity: {avg_prep_fidelity:.3f}")

    return hist_figs


def plot_bimodal_fit(x_data, y_data, params):
    """
    Visualize the histogram data with the fitted bimodal Gaussian distribution.

    Parameters:
    - x_data: Array of x-values (bin centers)
    - y_data: Histogram y-values (density or count)
    - params: Fitted parameters [mu1, sigma1, mu2, sigma2, w_nv_minus]
    """
    # Extract fitted parameters
    mu1, sigma1, mu2, sigma2, w_nv_minus = params
    w_nv_zero = 1 - w_nv_minus

    # Define the individual Gaussian components
    def gaussian(x, mu, sigma, weight):
        return (
            weight
            * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
            / (sigma * np.sqrt(2 * np.pi))
        )

    # Generate fit values
    y_fit = bimodal_gaussian(x_data, mu1, sigma1, mu2, sigma2, w_nv_minus)
    y_gauss1 = gaussian(x_data, mu1, sigma1, w_nv_zero)
    y_gauss2 = gaussian(x_data, mu2, sigma2, w_nv_minus)

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.bar(
        x_data,
        y_data,
        width=(x_data[1] - x_data[0]),
        alpha=0.5,
        label="Histogram Data",
        color="lightgray",
    )
    plt.plot(x_data, y_fit, label="Bimodal Fit", color="red", linewidth=2)
    plt.plot(
        x_data,
        y_gauss1,
        "--",
        label=f"Gaussian 1 (mu={mu1:.2f}, sigma={sigma1:.2f}, w={w_nv_zero:.2f})",
        color="blue",
    )
    plt.plot(
        x_data,
        y_gauss2,
        "--",
        label=f"Gaussian 2 (mu={mu2:.2f}, sigma={sigma2:.2f}, w={w_nv_minus:.2f})",
        color="green",
    )

    # Annotations
    plt.title("Bimodal Gaussian Fit to NV Charge States")
    plt.xlabel("Integrated Counts")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(alpha=0.3)

    # Annotate key fitted parameters
    plt.text(
        0.95,
        0.7,
        f"mu1 = {mu1:.2f}\nsigma1 = {sigma1:.2f}\nmu2 = {mu2:.2f}\nsigma2 = {sigma2:.2f}\nw(NV-) = {w_nv_minus:.2f}",
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(facecolor="white", alpha=0.6, edgecolor="gray"),
    )

    plt.show()


if __name__ == "__main__":
    kpl.init_kplotlib()
    data = dm.get_raw_data(file_id=1713224279642, load_npz=False)
    # data = dm.get_raw_data(file_id=1691569540529, load_npz=False)
    process_and_plot(data, do_plot_histograms=False)
    kpl.show(block=True)
