# -*- coding: utf-8 -*-
"""
Spin Echo Analysis and Visualization

Created on December 22nd, 2024
Enhanced and Parallelized on January 25th, 2025

@author: Saroj Chand
"""

import sys
import time
import traceback
from datetime import datetime
import concurrent.futures
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter, find_peaks, hilbert
from scipy.optimize import curve_fit
import pywt
import pymc as pm
import random
from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import widefield as widefield


def focused_revival_model(
    tau,
    baseline,
    rev_time_1,
    decay_time_1,
    amp_1,
    freq_1,
    phase_1,
    rev_time_2,
    decay_time_2,
    amp_2,
    freq_2,
    phase_2,
):
    """
    Model with two Gaussian envelopes for first and second revivals:
    - Each Gaussian envelope is modulated with its own oscillation term.
    - Includes a quadratic background decay.
    """
    # First revival Gaussian envelope
    gauss_env_1 = amp_1 * np.exp(-((tau - rev_time_1) ** 2) / (2 * decay_time_1**2))
    oscillation_1 = np.cos(2 * np.pi * freq_1 * tau + phase_1)

    # Second revival Gaussian envelope
    gauss_env_2 = amp_2 * np.exp(-((tau - rev_time_2) ** 2) / (2 * decay_time_2**2))
    oscillation_2 = np.cos(2 * np.pi * freq_2 * tau + phase_2)

    # Background decay
    background = baseline * (1 - 0.001 * tau)

    return background - gauss_env_1 * oscillation_1 - gauss_env_2 * oscillation_2


def enhanced_initial_guess(
    taus, signal, first_rev_range=(40, 60), second_rev_range=(80, 110)
):
    """
    Generate robust initial guesses for the full fitting model, considering both revival regions.
    """
    # Normalize the signal
    signal = (signal - np.nanmin(signal)) / (np.nanmax(signal) - np.nanmin(signal))
    baseline_guess = np.percentile(signal, 10)

    # Estimate parameters for each revival region
    def estimate_revival_params(taus, signal, rev_range):
        mask = (taus >= rev_range[0]) & (taus <= rev_range[1])
        rev_taus = taus[mask]
        rev_signal = signal[mask]
        peaks, props = find_peaks(rev_signal, prominence=0.1, width=5)
        if len(peaks) > 0:
            main_peak = peaks[np.argmax(props["prominences"])]
            revival_time_guess = rev_taus[main_peak]
            decay_guess = props["widths"][np.argmax(props["prominences"])] * (
                taus[1] - taus[0]
            )
        else:  # Fallback
            revival_time_guess = np.mean(rev_range)
            decay_guess = 0.2 * (rev_range[1] - rev_range[0])
        return revival_time_guess, decay_guess

    # Get guesses for both revival regions
    rev_time_1, decay_time_1 = estimate_revival_params(taus, signal, first_rev_range)
    rev_time_2, decay_time_2 = estimate_revival_params(taus, signal, second_rev_range)

    # Amplitude guess
    amp_guess = np.max(signal) - baseline_guess

    # Frequency guess
    def estimate_frequency(taus, signal):
        sample_spacing = taus[1] - taus[0]
        freqs = np.fft.rfftfreq(len(signal), d=sample_spacing)
        fft_magnitude = np.abs(np.fft.rfft(signal))
        dominant_freq_idx = np.argmax(fft_magnitude[: len(freqs)])
        return freqs[dominant_freq_idx]

    freq_guess = estimate_frequency(taus, signal)

    # Phase guess
    analytic_signal = hilbert(signal - baseline_guess)
    phase_guess = np.angle(analytic_signal[np.argmax(np.abs(analytic_signal))])

    # Return combined guess
    return [
        baseline_guess,
        rev_time_1,
        decay_time_1,
        amp_guess,
        freq_guess,
        phase_guess,
        rev_time_2,
        decay_time_2,
        amp_guess * 0.5,  # Assume second revival is weaker
        freq_guess,
        phase_guess,
    ]


def analyze_nv_center(nv_idx, taus, processed_signal):
    """Analyze a single NV center's signal for the full range."""
    try:
        # Enhanced initial guessing
        initial_guess = enhanced_initial_guess(taus, processed_signal)

        # Fit using scipy's curve_fit
        popt, pcov = curve_fit(
            focused_revival_model,
            taus,
            processed_signal,
            p0=initial_guess,
            bounds=(
                [0, 30, 5, 0.05, 0.1, -np.pi, 70, 5, 0.01, 0.1, -np.pi],  # Lower bounds
                [1, 50, 20, 0.5, 5.0, np.pi, 100, 30, 0.5, 5.0, np.pi],  # Upper bounds
            ),
        )

        return nv_idx, {"popt": popt, "pcov": pcov}

    except Exception as e:
        print(f"Error in NV {nv_idx}: {str(e)}")
        return nv_idx, None


def parallel_fit_processor(taus, all_signals, max_workers=None):
    """Parallel processing for full-range fitting."""
    results = [None] * len(all_signals)

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(analyze_nv_center, nv_idx, taus, signal): nv_idx
            for nv_idx, signal in enumerate(all_signals)
        }

        for future in concurrent.futures.as_completed(futures):
            nv_idx = futures[future]
            try:
                results[nv_idx] = future.result()
            except Exception as e:
                print(f"NV {nv_idx} failed: {str(e)}")
                results[nv_idx] = (nv_idx, None)

    return [result[1] for result in sorted(results, key=lambda x: x[0])]


def plot_full_fit(taus, smooth_counts, fit_results, num_to_plot=10):
    """Plot full fits for a random subset of NV centers."""
    num_nvs = len(smooth_counts)
    random_indices = random.sample(range(num_nvs), min(num_to_plot, num_nvs))

    for nv_idx in random_indices:
        smooth_counts_single = smooth_counts[nv_idx, :]
        fit_result = fit_results[nv_idx]
        popt = fit_result.get("popt", None)

        plt.figure(figsize=(10, 6))
        plt.plot(
            taus, smooth_counts_single, "o", label=f"NV {nv_idx} Signal", markersize=4
        )
        if popt is not None:
            plt.plot(
                taus,
                focused_revival_model(taus, *popt),
                "-",
                linewidth=2,
                label="Fitted Model",
            )
        plt.xlabel("Time (µs)")
        plt.ylabel("Normalized Counts")
        plt.title(f"Full Fit for NV {nv_idx}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


def process_multiple_files(file_ids):
    """
    Load and combine data from multiple file IDs.
    """
    combined_data = dm.get_raw_data(file_id=file_ids[0])
    for file_id in file_ids[1:]:
        new_data = dm.get_raw_data(file_id=file_id)
        combined_data["num_runs"] += new_data["num_runs"]
        combined_data["counts"] = np.append(
            combined_data["counts"], new_data["counts"], axis=2
        )
    return combined_data


if __name__ == "__main__":
    # Initialize kplotlib
    kpl.init_kplotlib()

    # Define file IDs to process
    file_ids = [
        1734158411844,
        1734273666255,
        1734371251079,
        1734461462293,
        1734569197701,
    ]

    # Step 1: Process data from multiple files
    data = process_multiple_files(file_ids)
    taus = 2 * np.array(data["taus"]) / 1e3  # Convert to µs
    counts = np.array(data["counts"])
    sig_counts, ref_counts = counts[0], counts[1]

    # Step 2: Normalize counts and

    norm_counts, _ = widefield.process_counts(
        data["nv_list"], sig_counts, ref_counts, threshold=True
    )
    # Perform fitting
    fit_results = parallel_fit_processor(taus, norm_counts, max_workers=4)

    # Save results
    now = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    file_name = f"{now}_spin_echo_fit"
    # dm.save_raw_data(
    #     {"taus": taus, "norm_counts": norm_counts, "fit_results": fit_results},
    #     file_name,
    # )

    # Plot results
    plot_full_fit(taus, norm_counts, fit_results, num_to_plot=15)
    # Show plots
    kpl.show(block=True)
