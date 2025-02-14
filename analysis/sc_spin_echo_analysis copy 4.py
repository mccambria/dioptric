# -*- coding: utf-8 -*-
"""
Enhanced Spin Echo Analysis with Focused First Revival Characterization
"""

import sys
import time
from datetime import datetime
import traceback
import concurrent.futures
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
import pywt
import pymc as pm
import random
from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import widefield as widefield
from scipy.signal import hilbert  # Correct import at top of file


def focused_revival_model(tau, baseline, revival_time, decay_time, amp, freq, phase):
    """
    Optimized model for first revival characterization with:
    - Single focused Gaussian envelope for first revival
    - Oscillation term with phase control
    - Quadratic background decay
    """
    gauss_env = amp * np.exp(-((tau - revival_time) ** 2) / (2 * decay_time**2))
    oscillation = np.cos(2 * np.pi * freq * tau + phase)
    background = baseline * (1 - 0.001 * tau)  # Small linear background decay
    return background - gauss_env * oscillation


def enhanced_initial_guess(taus, signal, first_rev_range=(40, 60)):
    """
    Generate robust initial guesses using multi-modal analysis:
    1. Wavelet-based frequency estimation
    2. Peak finding with prominence detection
    3. Envelope characteristic time analysis
    """
    # Basic signal conditioning
    signal = (signal - np.nanmin(signal)) / (np.nanmax(signal) - np.nanmin(signal))

    # Focus on first revival region
    mask = (taus >= first_rev_range[0]) & (taus <= first_rev_range[1])
    rev_taus = taus[mask]
    rev_signal = signal[mask]

    # 1. Find dominant frequency using wavelet transform
    scales = np.arange(1, 20)
    coeffs, freqs = pywt.cwt(rev_signal, scales, "cmor1.5-1.0")
    power = np.abs(coeffs).mean(axis=1)
    freq_guess = freqs[np.argmax(power)] / (taus[1] - taus[0])  # Convert to Hz/μs

    # 2. Find peak with prominence detection
    peaks, props = find_peaks(rev_signal, prominence=0.1, width=5)
    if len(peaks) > 0:
        main_peak = peaks[np.argmax(props["prominences"])]
        revival_time_guess = rev_taus[main_peak]
        decay_guess = props["widths"][np.argmax(props["prominences"])] * (
            taus[1] - taus[0]
        )
    else:  # Fallback to center of range
        revival_time_guess = np.mean(first_rev_range)
        decay_guess = 0.2 * (first_rev_range[1] - first_rev_range[0])

    # 3. Estimate baseline and amplitude
    baseline_guess = np.median(signal[:10])  # Use initial points for baseline
    amp_guess = np.max(rev_signal) - baseline_guess

    # 4. Phase estimation using Hilbert transform
    analytic_signal = hilbert(rev_signal - baseline_guess)
    phase_guess = np.angle(analytic_signal[np.argmax(np.abs(analytic_signal))])

    return [
        np.clip(baseline_guess, 0.1, 0.9),
        np.clip(revival_time_guess, *first_rev_range),
        np.clip(decay_guess, 5, 20),
        np.clip(amp_guess, 0.05, 0.5),
        np.clip(freq_guess, 0.1, 5.0),
        phase_guess,
    ]


# def bayesian_first_revival_fit(taus, signal, initial_guess):
#     """Bayesian fitting with physics-informed priors and focused convergence"""
#     with pm.Model() as model:
#         # Priors centered on initial guesses with constrained ranges
#         baseline = pm.TruncatedNormal(
#             "baseline", mu=initial_guess[0], sigma=0.1, lower=0.1, upper=0.9
#         )
#         revival_time = pm.TruncatedNormal(
#             "revival_time", mu=initial_guess[1], sigma=5, lower=40, upper=60
#         )
#         decay_time = pm.TruncatedNormal(
#             "decay_time", mu=initial_guess[2], sigma=5, lower=5, upper=30
#         )
#         amp = pm.HalfNormal("amp", sigma=0.5, initval=initial_guess[3])
#         freq = pm.TruncatedNormal(
#             "freq", mu=initial_guess[4], sigma=1.0, lower=0.1, upper=5.0
#         )
#         phase = pm.VonMises("phase", mu=initial_guess[5], kappa=2)

#         # Forward model
#         model_pred = focused_revival_model(
#             taus, baseline, revival_time, decay_time, amp, freq, phase
#         )

#         # Likelihood with robust noise estimation
#         sigma = pm.HalfNormal("sigma", sigma=0.1)
#         pm.StudentT(
#             "obs",
#             nu=4,  # Heavy-tailed for robustness
#             mu=model_pred,
#             sigma=sigma,
#             observed=signal,
#         )

#         # Sampling with advanced settings
#         trace = pm.sample(
#             draws=800,
#             tune=1000,
#             chains=4,
#             cores=4,
#             target_accept=0.95,
#             init="adapt_diag",
#         )

#     # Return posterior means as best estimates
#     return [
#         trace.posterior[var].mean().item() for var in trace.posterior.data_vars
#     ], trace


def bayesian_first_revival_fit(taus, signal, initial_guess):
    """Enhanced Bayesian fitting with better sampling settings"""
    with pm.Model() as model:
        # Priors (unchanged)
        baseline = pm.TruncatedNormal(
            "baseline", mu=initial_guess[0], sigma=0.1, lower=0.1, upper=0.9
        )
        revival_time = pm.TruncatedNormal(
            "revival_time", mu=initial_guess[1], sigma=5, lower=40, upper=60
        )
        decay_time = pm.TruncatedNormal(
            "decay_time", mu=initial_guess[2], sigma=5, lower=5, upper=30
        )
        amp = pm.HalfNormal("amp", sigma=0.5, initval=initial_guess[3])
        freq = pm.TruncatedNormal(
            "freq", mu=initial_guess[4], sigma=1.0, lower=0.1, upper=5.0
        )
        phase = pm.VonMises("phase", mu=initial_guess[5], kappa=2)

        # Model prediction
        model_pred = focused_revival_model(
            taus, baseline, revival_time, decay_time, amp, freq, phase
        )

        # Likelihood
        sigma = pm.HalfNormal("sigma", sigma=0.1)
        pm.StudentT("obs", nu=4, mu=model_pred, sigma=sigma, observed=signal)

        # Enhanced sampling settings
        trace = pm.sample(
            draws=1000,  # Reduced from 800 to 1000
            tune=2000,  # Increased tuning steps
            chains=4,
            cores=4,
            target_accept=0.99,  # Increased target acceptance
            max_treedepth=15,  # Increased tree depth
            init="adapt_diag",
            return_inferencedata=True,  # Ensure compatibility with ArviZ
        )

    # Extract posterior means safely
    posterior_means = {
        param: (
            trace.posterior[param].mean().item()
            if hasattr(trace.posterior[param].mean(), "item")
            else float(trace.posterior[param].mean())
        )
        for param in trace.posterior.data_vars
    }

    return list(posterior_means.values()), trace


def analyze_nv_center(nv_idx, taus, processed_signal):
    """Robust single NV analysis pipeline"""
    try:
        # Enhanced initial guessing
        initial_guess = enhanced_initial_guess(taus, processed_signal)

        # Bayesian fitting
        popt, trace = bayesian_first_revival_fit(taus, processed_signal, initial_guess)

        # Quality checks
        max_rhat = az.rhat(trace).max().values.item()
        if max_rhat > 1.05:
            print(f"NV {nv_idx}: Potential convergence issues (Rhat={max_rhat:.2f})")

        return nv_idx, (popt, trace)

    except Exception as e:
        print(f"Error in NV {nv_idx}: {str(e)}")
        return nv_idx, (None, None)


def parallel_fit_processor(taus, all_signals, max_workers=None):
    """Parallel processing with resource optimization"""
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
                results[nv_idx] = (nv_idx, (None, None))

    # Sort by NV index and extract fit results
    return [result[1] for result in sorted(results, key=lambda x: x[0])]


# Visualization functions remain similar but use focused_revival_model
def plot_first_revival(taus, smooth_counts, popts, num_to_plot=10):
    """
    Plot spin echo data and fitted curves for a random subset of NV centers.
    Focuses on the first revival region.
    """
    num_nvs = len(smooth_counts)
    random_indices = random.sample(range(num_nvs), min(num_to_plot, num_nvs))

    for nv_idx in random_indices:
        smooth_counts_single = smooth_counts[nv_idx, :]
        popt = popts[nv_idx]  # Extract optimized parameters for this NV

        plt.figure(figsize=(8, 5))
        first_revival_mask = (taus >= 40) & (taus <= 60)
        plt.plot(
            taus[first_revival_mask],
            smooth_counts_single[first_revival_mask],
            "o",
            label=f"NV {nv_idx}",
            markersize=4,
        )
        if popt is not None:
            plt.plot(
                taus[first_revival_mask],
                focused_revival_model(taus[first_revival_mask], *popt),
                "-",
                linewidth=2,
                label="Fitted Model",
            )
        plt.xlabel("Time (µs)")
        plt.ylabel("Normalized Counts")
        plt.title(f"First Revival Fit for NV {nv_idx}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


# Preprocessing functions should include per-NV normalization
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
    kpl.init_kplotlib()

    # try:
    #     # Data loading and preprocessing
    #     file_ids = [
    #         1734158411844,
    #         1734273666255,
    #         1734371251079,
    #         1734461462293,
    #         1734569197701,
    #     ]
    #     data = process_multiple_files(file_ids)
    #     taus = 2 * np.array(data["taus"]) / 1e3  # Convert to µs
    #     # Enhanced preprocessing with per-NV normalization
    #     counts = np.array(data["counts"])
    #     sig_counts, ref_counts = counts[0], counts[1]

    #     # Step 2: Normalize counts and      # Step 3: Preprocess data

    #     norm_counts, _ = widefield.process_counts(
    #         data["nv_list"], sig_counts, ref_counts, threshold=True
    #     )
    #     # Parallel fitting
    #     fit_results = parallel_fit_processor(taus, norm_counts, max_workers=2)

    #     # Visualization focused on first revival
    #     # plot_first_revival(taus, norm_counts, fit_results, num_to_plot=15)

    #     # Save results for later analysis)
    #     now = datetime.now()
    #     date_time_str = now.strftime("%Y_%m_%d-%H_%M_%S")
    #     source_file_name = dm.get_file_name(file_id=file_ids[0])
    #     file_name = f"{date_time_str}_spin_echo_analysis"
    #     file_path = dm.get_file_path(__file__, file_name, source_file_name)
    #     analysed_result = {
    #         "taus": taus,
    #         "norm_counts": norm_counts,
    #         "fit_results": fit_results,
    #         "source_files": file_ids,
    #     }
    #     dm.save_raw_data(analysed_result, file_path)
    # except Exception as e:
    #     print(f"Main execution failed: {str(e)}")
    #     traceback.print_exc()

    # load analysed data and plot
    file_id = 1760373999845
    analysed_data = dm.get_raw_data(file_id=file_id)
    taus = np.array(analysed_data["taus"])
    norm_counts = np.array(analysed_data["norm_counts"])
    fit_results = np.array(analysed_data["fit_results"])
    for i, fit_result in enumerate(fit_results[:5]):  # Print first 5 entries
        print(f"NV {i} popt:", fit_result[0])

    print(norm_counts.shape)
    print(fit_results.shape)
    popts = [fit_result[0] for fit_result in fit_results]  # Extract popt
    traces = [fit_result[1] for fit_result in fit_results]  # Extract trace
    popts = np.array(popts)
    print(popts.shape)
    print("taus:", taus[:10])  # Print first 10 values of taus
    print("Signal shape:", norm_counts.shape)  # Shape of normalized counts
    print("Signal for NV 0:", norm_counts[0, :10])  # First 10 signal points for NV 0

    plot_first_revival(taus, norm_counts, fit_results, num_to_plot=15)
    kpl.show(block=True)
