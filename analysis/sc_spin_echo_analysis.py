# -*- coding: utf-8 -*-
"""
Spin Echo Analysis and Visualization

Created on December 22nd, 2024

@author: Saroj cHAND
"""

import time
import traceback

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.optimize import curve_fit
from scipy.signal import lombscargle
from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import tool_belt as tb
from utils import widefield as widefield
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.stats import iqr
from datetime import datetime


# Define a decay model for spin echo fitting
def quartic_decay(tau, baseline, revival_time, decay_time, amp1, amp2):
    num_revivals = 3
    value = baseline
    for i in range(num_revivals):
        exp_decay = np.exp(-(((tau - i * revival_time) / decay_time) ** 2))
        mod = amp1 * np.cos(2 * np.pi * amp2 * tau)
        value -= exp_decay * mod
    return value


# Filter NV centers based on SNR and fit quality
def filter_nv_centers(
    snrs, contrasts, chi_squared, snr_threshold=5, contrast_threshold=0.1
):
    filtered_indices = [
        i
        for i, (snr, contrast, chi_sq) in enumerate(zip(snrs, contrasts, chi_squared))
        if snr > snr_threshold and contrast > contrast_threshold and chi_sq < 50
    ]
    return filtered_indices


# Function to provide smart guesses and bounds for curve fitting
def generate_initial_guess_and_bounds(tau, counts):
    baseline_guess = np.mean(counts[-10:])  # Use the end of the data for baseline
    revival_time_guess = tau[np.argmax(counts)] * 2  # Estimate revival period
    decay_time_guess = (tau[-1] - tau[0]) / 5  # Broad guess for decay time
    amp1_guess = (max(counts) - min(counts)) / 2
    amp2_guess = amp1_guess / 2

    # Frequency guess based on simple FFT
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

    bounds = (
        [0, tau[1] - tau[0], 0, 0, 0, 0],  # Lower bounds
        [1, tau[-1], np.inf, np.inf, np.inf, np.inf],  # Upper bounds
    )

    return initial_guess, bounds


# Analyze and visualize spin echo data
def analyze_spin_echo(data):
    nv_list = data["nv_list"]
    num_nvs = len(nv_list)
    num_nvs = len(nv_list)
    taus = np.array(data["taus"])
    num_steps = data["num_steps"]
    total_evolution_times = 2 * np.array(taus) / 1e3
    counts = np.array(data["counts"])

    sig_counts = counts[0]
    ref_counts = counts[1]
    norm_counts, norm_counts_ste = widefield.process_counts(
        nv_list, sig_counts, ref_counts, threshold=False
    )

    fit_params = []
    snrs = []
    contrasts = []
    chi_squared_values = []

    # Prepare figure for visualization
    num_cols = 4
    num_rows = int(np.ceil(num_nvs / num_cols))
    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=(num_cols * 4, num_rows * 3),
        sharex=True,
        sharey=False,
    )
    axes = axes.flatten()
    # Visualization settings
    sns.set(style="whitegrid", palette="muted")
    for nv_idx, ax in enumerate(axes):
        if nv_idx >= num_nvs:
            ax.axis("off")
            continue

        nv_tau = taus[nv_idx]
        nv_counts = norm_counts[nv_idx]
        initial_guess, bounds = generate_initial_guess_and_bounds(nv_tau, nv_counts)
        try:
            # Curve fitting
            popt, pcov = curve_fit(
                quartic_decay, nv_tau, nv_counts, p0=initial_guess, bounds=bounds
            )
            fit_params.append(popt)

            # Calculate residuals and chi-squared
            residuals = nv_counts - quartic_decay(nv_tau, *popt)
            chi_sq = np.sum((residuals / np.std(residuals)) ** 2)
            chi_squared_values.append(chi_sq)

            # Calculate SNR
            signal_peak = max(nv_counts) - min(nv_counts)
            noise_level = np.std(nv_counts)
            snr = signal_peak / noise_level
            snrs.append(snr)

            # Calculate contrast
            contrast = (max(nv_counts) - min(nv_counts)) / max(nv_counts)
            contrasts.append(contrast)

            # Plot data and fit
            sns.lineplot(x=nv_tau, y=nv_counts, ax=ax, label=f"NV {nv_idx}", lw=2)
            sns.lineplot(
                x=nv_tau,
                y=quartic_decay(nv_tau, *popt),
                ax=ax,
                linestyle="--",
                label="Fit",
            )

            ax.legend(fontsize="small")
            ax.set_title(f"NV {nv_idx}", fontsize=10)

        except Exception as e:
            print(f"Fit failed for NV {nv_idx}: {e}")
            fit_params.append(None)
            snrs.append(0)
            contrasts.append(0)
            chi_squared_values.append(np.inf)

    plt.tight_layout()
    plt.suptitle(f"Spin Echo - {file_id}", fontsize=16, y=1.02)
    plt.show()

    # Filter NV centers based on thresholds
    filtered_indices = filter_nv_centers(snrs, contrasts, chi_squared_values)

    print(f"Filtered NVs: {filtered_indices}")

    return fit_params, snrs, contrasts, chi_squared_values, filtered_indices


if __name__ == "__main__":
    kpl.init_kplotlib()
    file_id = 1548381879624
    data = dm.get_raw_data(file_id=file_id)
    analyze_spin_echo(data)
    kpl.show(block=True)
