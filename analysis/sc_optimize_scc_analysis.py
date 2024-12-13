# -*- coding: utf-8 -*-
"""
Optimize SCC parameters

Created on December 6th, 2023

@author: mccambria
"""

import traceback

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from analysis.bimodal_histogram import (
    ProbDist,
    determine_threshold,
    fit_bimodal_histogram,
)
from majorroutines.widefield import base_routine
from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import tool_belt as tb
from utils import widefield as widefield


import traceback
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
from utils import widefield, data_manager as dm, kplotlib as kpl, tool_belt as tb
from majorroutines.widefield import base_routine


def multi_fit_snr(taus, snr_data, snr_ste, model="exponential"):
    """Fit SNR data to a chosen model."""

    # Define the models
    def exp_decay(tau, delay, slope, decay):
        tau = np.array(tau) - delay
        return slope * tau * np.exp(-tau / decay)

    def hill_exp_decay(tau, A, K, n, tau_decay):
        hill = (A * tau**n) / (K**n + tau**n)
        return hill * np.exp(-tau / tau_decay)

    def hill_gaussian(tau, A, K, n, mu, sigma):
        hill = (A * tau**n) / (K**n + tau**n)
        return hill * np.exp(-((tau - mu) ** 2) / (2 * sigma**2))

    def hill_logistic(tau, A, K, n, r, c):
        hill = (A * tau**n) / (K**n + tau**n)
        return hill / (1 + np.exp(-r * (tau - c)))

    # Map models to their respective functions and initial guesses
    models = {
        "exponential": (exp_decay, [taus[0], np.max(snr_data), taus[-1]]),
        "hill_exp": (hill_exp_decay, [np.max(snr_data), np.median(taus), 2, taus[-1]]),
        "hill_gaussian": (
            hill_gaussian,
            [np.max(snr_data), np.median(taus), 2, taus.mean(), taus.std()],
        ),
        "hill_logistic": (
            hill_logistic,
            [np.max(snr_data), np.median(taus), 2, 1, taus.mean()],
        ),
    }

    # Select the model and initial guesses
    fit_fn, guess_params = models.get(
        model, (exp_decay, [taus[0], np.max(snr_data), taus[-1]])
    )

    # Attempt to fit the model
    try:
        popt, _ = curve_fit(
            fit_fn,
            taus,
            snr_data,
            p0=guess_params,
            sigma=snr_ste,
            absolute_sigma=True,
            maxfev=10000,  # Increase the max number of iterations
        )
    except Exception as e:
        print(f"Fitting failed for model '{model}': {e}")
        popt = guess_params  # Use initial guesses as fallback parameters

    return popt, fit_fn


def fit_snr(taus, avg_snr_nv, avg_snr_ste_nv):
    """Fit SNR data to a custom function."""

    def fit_fn(tau, delay, slope, decay):
        tau = np.array(tau) - delay
        return slope * tau * np.exp(-tau / decay)

    guess_params = [taus[0], np.max(avg_snr_nv), taus[-1]]
    try:
        popt, _ = curve_fit(
            fit_fn,
            taus,
            avg_snr_nv,
            p0=guess_params,
            sigma=avg_snr_ste_nv,
            absolute_sigma=True,
            maxfev=10000,  # Increase the max number of iterations
        )
    except Exception as e:
        print(f"Fitting failed for this NV: {e}")
        popt = [taus[0], 0, taus[-1]]  # Default fallback parameters
    return popt, fit_fn


def plot_individual_nv_fits(nv_list, taus, avg_snr, avg_snr_ste):
    """Create separate figures for individual NV SNR fits."""
    figs = []  # Store all figures for later reference

    for nv_ind in range(len(nv_list)):
        fig, ax = plt.subplots(figsize=(8, 6))  # Create a new figure for each NV
        popt, fit_fn = fit_snr(taus, avg_snr[nv_ind], avg_snr_ste[nv_ind])
        tau_linspace = np.linspace(min(taus), max(taus), 1000)

        # Plot the fit curve
        sns.lineplot(
            x=tau_linspace,
            y=fit_fn(tau_linspace, *popt),
            label="Fit",
            ax=ax,
        )

        # Plot the data points
        sns.scatterplot(
            x=taus,
            y=avg_snr[nv_ind],
            ax=ax,
            label="Data",
            s=60,
        )

        # Customize the plot
        nv_num = widefield.get_nv_num(nv_list[nv_ind])
        ax.set_title(f"NV {nv_num} SNR Fit")
        ax.set_xlabel("SCC Pulse Duration (ns)")
        ax.set_ylabel("SNR")
        ax.legend()
        ax.grid(True)

        figs.append(fig)  # Append the created figure to the list

    return figs


def process_and_plot(nv_list, taus, sig_counts, ref_counts, duration_or_amp):
    """Process and plot data for signal, reference, and SNR."""
    num_nvs = len(nv_list)
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
    # Average counts and calculate metrics
    # avg_sig_counts, avg_sig_counts_ste, _ = widefield.average_counts(sig_counts)
    # avg_ref_counts, avg_ref_counts_ste, _ = widefield.average_counts(ref_counts)
    avg_snr, avg_snr_ste = widefield.calc_snr(sig_counts, ref_counts)

    # Average and Median SNR
    avg_snr_all = np.mean(avg_snr, axis=0)
    median_snr_all = np.median(avg_snr, axis=0)
    avg_snr_ste_all = np.mean(avg_snr_ste, axis=0)

    fig_avg_snr, ax_avg_snr = plt.subplots()
    sns.lineplot(x=taus, y=avg_snr_all, ax=ax_avg_snr, label="Average SNR")
    sns.lineplot(
        x=taus, y=median_snr_all, ax=ax_avg_snr, label="Median SNR", linestyle="--"
    )
    ax_avg_snr.fill_between(
        taus,
        avg_snr_all - avg_snr_ste_all,
        avg_snr_all + avg_snr_ste_all,
        alpha=0.2,
        label="Error Bounds",
    )
    ax_avg_snr.set_xlabel("SCC Duration (ns)")
    ax_avg_snr.set_ylabel("SNR")
    ax_avg_snr.legend()
    ax_avg_snr.grid(True)
    plt.title("Avg and Median SNR across NVs")

    # fig_snr_fits = plot_individual_nv_fits(nv_list, taus, avg_snr, avg_snr_ste)

    return fig_avg_snr


if __name__ == "__main__":
    kpl.init_kplotlib()

    # Load data
    data = dm.get_raw_data(file_id=1722305531191)
    nv_list = data["nv_list"]
    taus = data["taus"]
    counts = np.array(data["counts"])
    sig_counts = counts[0]
    ref_counts = counts[1]
    # Process and plot
    figs = process_and_plot(nv_list, taus, sig_counts, ref_counts, duration_or_amp=True)

    # Show plots
    plt.show(block=True)
