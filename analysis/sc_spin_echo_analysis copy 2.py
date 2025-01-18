# -*- coding: utf-8 -*-
"""
Spin Echo Analysis and Visualization

Created on December 22nd, 2024

@author: Saroj Chand
"""

import sys
import time
import traceback
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.optimize import curve_fit

from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import widefield as widefield

# import pip as azn

# from tensorflow.keras import layers, models
import pymc as pm
from scipy.signal import savgol_filter
import pywt
from scipy.signal import find_peaks


# Combine data from multiple file IDspi
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


def preprocess_spin_echo_data(taus, norm_counts):
    """
    Preprocess spin echo data by normalizing and smoothing.

    Parameters:
    taus (array): Array of evolution times (in microseconds).
    norm_counts (array): Array of normalized spin echo counts for each NV.

    Returns:
    smooth_counts (array): Smoothed counts using Savitzky-Golay filter.
    """
    # Normalize counts (if not already normalized)
    norm_counts = (norm_counts - np.min(norm_counts)) / (
        np.max(norm_counts) - np.min(norm_counts)
    )

    # Apply Savitzky-Golay filter for smoothing (window size and polynomial order can be adjusted)
    window_length = min(
        11, len(taus) - (len(taus) % 2 == 0)
    )  # Ensure odd window length
    polyorder = 3  # Polynomial order for smoothing
    smooth_counts = savgol_filter(
        norm_counts, window_length=window_length, polyorder=polyorder
    )

    return smooth_counts


def detect_revivals(taus, smooth_counts):
    """
    Detect revival times for all NV centers using Continuous Wavelet Transform (CWT) and peak detection.

    Parameters:
    taus (array): Array of evolution times (in microseconds).
    smooth_counts (array): Smoothed spin echo counts (2D array, shape [num_nvs, num_taus]).

    Returns:
    revival_times_all (list): List of arrays, where each array contains detected revival times for an NV center.
    """
    taus = np.array(taus)
    wavelet_name = "cmor1.5-1.0"
    scales = np.arange(1, 128)
    revival_times_all = []

    for nv_idx in range(smooth_counts.shape[0]):  # Loop over each NV center
        smooth_counts_single = smooth_counts[
            nv_idx, :
        ]  # Select signal for the current NV

        # Perform CWT using the Morlet wavelet
        coefficients, _ = pywt.cwt(
            smooth_counts_single,
            scales,
            wavelet_name,
            sampling_period=(taus[1] - taus[0]),
        )

        # Compute the energy at each scale
        energy = np.sum(np.abs(coefficients) ** 2, axis=1)

        # Select the scale with the maximum total energy
        best_scale_idx = np.argmax(energy)
        revival_indicator = np.abs(
            coefficients[best_scale_idx]
        )  # Use the best scale for peak detection

        # Detect peaks in the revival indicator
        peaks, _ = find_peaks(revival_indicator, height=np.mean(revival_indicator))
        # print(f"Peaks: {peaks}")
        revival_times = taus[
            peaks
        ]  # Use the entire array of peak indices to get revival times
        revival_times_all.append(revival_times)

    return revival_times_all


# def generate_initial_guess(taus, smooth_counts_single, revival_times_single):
#     """
#     Generate initial guess for spin echo fitting for a single NV center.

#     Parameters:
#     taus (array): Array of evolution times (in microseconds).
#     smooth_counts_single (array): Smoothed spin echo counts for a single NV.
#     revival_times_single (array): Detected revival times for a single NV.

#     Returns:
#     initial_guess (list): Initial guess for [baseline, revival_time, decay_time, amp1, amp2, freq].
#     """
#     baseline_guess = np.mean(smooth_counts_single)
#     revival_time_guess = (
#         np.mean(np.diff(revival_times_single))
#         if len(revival_times_single) > 1
#         else taus[-1] / 3
#     )
#     decay_time_guess = taus[-1] / 3  # Rough estimate of decay time
#     amp1_guess = (max(smooth_counts_single) - min(smooth_counts_single)) / 2
#     amp2_guess = amp1_guess / 2

#     # Estimate frequency using FFT
#     fft_freqs = np.fft.rfftfreq(len(taus), d=(taus[1] - taus[0]))
#     fft_spectrum = np.abs(np.fft.rfft(smooth_counts_single - baseline_guess))
#     freq_guess = fft_freqs[np.argmax(fft_spectrum)]

#     return [
#         baseline_guess,
#         revival_time_guess,
#         decay_time_guess,
#         amp1_guess,
#         amp2_guess,
#         freq_guess,
#     ]


def generate_initial_guess(taus, smooth_counts_single, revival_times_single):
    """
    Generate an enhanced initial guess for spin echo fitting using FFT and peak detection.

    Parameters:
    taus (array): Array of evolution times (in microseconds).
    smooth_counts_single (array): Smoothed spin echo counts for a single NV.
    revival_times_single (array): Detected revival times for a single NV.

    Returns:
    initial_guess (list): Initial guess for [baseline, revival_time, decay_time, amp1, amp2, freq].
    """
    baseline_guess = np.mean(smooth_counts_single)
    revival_time_guess = (
        np.mean(np.diff(revival_times_single))
        if len(revival_times_single) > 1
        else taus[-1] / 3
    )
    decay_time_guess = taus[-1] / 3  # Rough estimate of decay time
    amp1_guess = (max(smooth_counts_single) - min(smooth_counts_single)) / 2
    amp2_guess = amp1_guess / 2

    # Estimate frequency using FFT, focusing on the first revival range
    fft_range = (taus >= 40) & (taus <= 60)
    fft_freqs = np.fft.rfftfreq(len(taus[fft_range]), d=(taus[1] - taus[0]))
    fft_spectrum = np.abs(np.fft.rfft(smooth_counts_single[fft_range] - baseline_guess))
    freq_guess = fft_freqs[np.argmax(fft_spectrum)]

    return [
        baseline_guess,
        revival_time_guess,
        decay_time_guess,
        amp1_guess,
        amp2_guess,
        freq_guess,
    ]


def quartic_decay(tau, baseline, revival_time, decay_time, amp1, amp2, freq):
    """
    Quartic decay model for spin echo fitting.

    Parameters:
    tau (array): Evolution times (in microseconds).
    baseline, revival_time, decay_time, amp1, amp2, freq: Model parameters.

    Returns:
    value (array): Fitted spin echo signal.
    """
    num_revivals = 3
    value = baseline
    for i in range(num_revivals):
        exp_decay = np.exp(-(((tau - i * revival_time) / decay_time) ** 2))
        mod = amp1 * np.cos(2 * np.pi * freq * tau)
        value -= exp_decay * mod
    return value


# def advanced_quartic_decay(
#     tau, baseline, revival_time, decay_time, amp1, amp2, freq, phase
# ):
#     """
#     Advanced quartic decay model with oscillations and a phase shift for spin echo fitting.

#     Parameters:
#     tau (array): Evolution times (in microseconds).
#     baseline, revival_time, decay_time, amp1, amp2, freq, phase: Model parameters.

#     Returns:
#     value (array): Fitted spin echo signal.
#     """
#     num_revivals = (
#         int(tau[-1] // revival_time) + 1
#     )  # Dynamically determine number of revivals
#     value = baseline
#     for i in range(num_revivals):
#         exp_decay = np.exp(-(((tau - i * revival_time) / decay_time) ** 2))
#         mod = amp1 * np.cos(2 * np.pi * freq * tau + phase) + amp2 * np.sin(
#             2 * np.pi * freq * tau + phase
#         )
#         value -= exp_decay * mod
#     return value


def advanced_spin_echo_model(
    tau,
    baseline,
    slope,
    flat_start,
    decay_time,
    amp1,
    freq1,
    phase1,
    revival_time2,
    amp2,
    dip_width,
):
    """
    Advanced spin echo model with an initial linear rise, oscillations, and a second revival dip.

    Parameters:
    tau (array): Evolution times (in microseconds).
    baseline (float): Baseline signal level.
    slope (float): Slope of the initial linear rise.
    flat_start (float): Time at which the signal becomes flat after the initial rise.
    decay_time (float): Characteristic decay time for the oscillations.
    amp1 (float): Amplitude of the first oscillations.
    freq1 (float): Frequency of the first oscillations.
    phase1 (float): Phase of the first oscillations.
    revival_time2 (float): Time of the second revival (dip).
    amp2 (float): Amplitude of the second dip.
    dip_width (float): Width of the second dip.

    Returns:
    value (array): Fitted spin echo signal.
    """
    # Initial linear rise followed by a flat region
    rise = np.minimum(slope * tau, baseline)  # Linear rise capped at the baseline
    flat_region = (
        np.heaviside(tau - flat_start, 0.5) * baseline
    )  # Flat region after flat_start

    # First oscillatory region with Gaussian decay envelope
    gaussian_envelope1 = np.exp(-(((tau - flat_start) / decay_time) ** 2))
    oscillations1 = amp1 * np.cos(2 * np.pi * freq1 * tau + phase1)

    # Second revival (dip) modeled as a Gaussian dip
    dip = amp2 * np.exp(-(((tau - revival_time2) / dip_width) ** 2))

    # Combine all components
    value = rise + flat_region - gaussian_envelope1 * oscillations1 - dip
    return value


def advanced_quartic_decay(
    tau, baseline, revival_time, decay_time, amp1, amp2, freq, phase
):
    """
    Advanced quartic decay model with oscillations and a phase shift for spin echo fitting.

    Parameters:
    tau (array): Evolution times (in microseconds).
    baseline, revival_time, decay_time, amp1, amp2, freq, phase: Model parameters.

    Returns:
    value (array): Fitted spin echo signal.
    """
    value = baseline
    gaussian_envelope = np.exp(-(((tau % revival_time) / decay_time) ** 2))
    modulation = amp1 * np.cos(2 * np.pi * freq * tau + phase) + amp2 * np.sin(
        2 * np.pi * freq * tau + phase
    )
    value -= gaussian_envelope * modulation
    return value


def bayesian_fit_spin_echo(taus, smooth_counts_single, initial_guess):
    """
    Fit spin echo data using a Bayesian approach with Markov Chain Monte Carlo (MCMC).

    Parameters:
    taus (array): Evolution times (in microseconds).
    smooth_counts_single (array): Smoothed spin echo counts for a single NV.
    initial_guess (list): Initial guess for fitting parameters.

    Returns:
    popt (array): Optimal parameters for the fit.
    pcov (2D array): Covariance matrix of the parameters.
    trace (InferenceData): Trace object containing sampled parameter values.
    """
    with pm.Model() as model:
        # Priors for initial rise and flat region
        baseline = pm.Uniform("baseline", lower=0, upper=1, testval=initial_guess[0])
        slope = pm.HalfNormal("slope", sigma=10, testval=initial_guess[1])
        flat_start = pm.Uniform(
            "flat_start", lower=30, upper=50, testval=initial_guess[2]
        )

        # Priors for oscillations in the first revival
        decay_time = pm.HalfNormal("decay_time", sigma=20, testval=initial_guess[3])
        amp1 = pm.HalfNormal("amp1", sigma=0.5, testval=initial_guess[4])
        freq1 = pm.Uniform("freq1", lower=0, upper=10, testval=initial_guess[5])
        phase1 = pm.Uniform(
            "phase1", lower=0, upper=2 * np.pi, testval=initial_guess[6]
        )

        # Priors for second revival (dip)
        revival_time2 = pm.Uniform(
            "revival_time2", lower=70, upper=90, testval=initial_guess[7]
        )
        amp2 = pm.HalfNormal("amp2", sigma=0.5, testval=initial_guess[8])
        dip_width = pm.HalfNormal("dip_width", sigma=10, testval=initial_guess[9])

        # Likelihood
        mu = advanced_spin_echo_model(
            taus,
            baseline,
            slope,
            flat_start,
            decay_time,
            amp1,
            freq1,
            phase1,
            revival_time2,
            amp2,
            dip_width,
        )
        sigma = pm.HalfNormal("sigma", sigma=0.1)
        likelihood = pm.Normal(
            "likelihood", mu=mu, sigma=sigma, observed=smooth_counts_single
        )

        # Inference with multiple cores and chains
        trace = pm.sample(600, chains=4, cores=4, target_accept=0.9, max_treedepth=15)

    # Extract posterior means as optimal parameters
    popt = [
        np.mean(trace.posterior[param].values)
        for param in [
            "baseline",
            "slope",
            "flat_start",
            "decay_time",
            "amp1",
            "freq1",
            "phase1",
            "revival_time2",
            "amp2",
            "dip_width",
        ]
    ]

    # Compute covariance matrix of parameters
    params = np.vstack(
        [
            trace.posterior[param].values.flatten()
            for param in [
                "baseline",
                "slope",
                "flat_start",
                "decay_time",
                "amp1",
                "freq1",
                "phase1",
                "revival_time2",
                "amp2",
                "dip_width",
            ]
        ]
    )
    pcov = np.cov(params)

    return popt, pcov, trace


# def bayesian_fit_spin_echo(taus, smooth_counts_single, initial_guess):
#     """
#     Fit spin echo data using a Bayesian approach with Markov Chain Monte Carlo (MCMC).

#     Parameters:
#     taus (array): Evolution times (in microseconds).
#     smooth_counts_single (array): Smoothed spin echo counts for a single NV.
#     initial_guess (list): Initial guess for fitting parameters.

#     Returns:
#     popt (array): Optimal parameters for the fit.
#     pcov (2D array): Covariance matrix of the parameters.
#     """
#     with pm.Model() as model:
#     # Priors for initial rise and flat region
#     baseline = pm.Uniform("baseline", lower=0, upper=1)
#     slope = pm.HalfNormal("slope", sigma=10)
#     flat_start = pm.Uniform("flat_start", lower=30, upper=50)

#     # Priors for oscillations
#     decay_time = pm.HalfNormal("decay_time", sigma=20)
#     amp1 = pm.HalfNormal("amp1", sigma=0.5)
#     freq1 = pm.Uniform("freq1", lower=0, upper=10)
#     phase1 = pm.Uniform("phase1", lower=0, upper=2 * np.pi)

#     # Priors for second revival (dip)
#     revival_time2 = pm.Uniform("revival_time2", lower=70, upper=90)
#     amp2 = pm.HalfNormal("amp2", sigma=0.5)
#     dip_width = pm.HalfNormal("dip_width", sigma=10)

#     # Likelihood
#     mu = advanced_spin_echo_model(
#         taus,
#         baseline,
#         slope,
#         flat_start,
#         decay_time,
#         amp1,
#         freq1,
#         phase1,
#         revival_time2,
#         amp2,
#         dip_width,
#     )
#     sigma = pm.HalfNormal("sigma", sigma=0.1)
#     likelihood = pm.Normal(
#         "likelihood", mu=mu, sigma=sigma, observed=smooth_counts_single
#     )

#     # Inference with multiple cores and chains
#     trace = pm.sample(600, chains=4, cores=4, target_accept=0.9, max_treedepth=15)
#     with pm.Model() as model:
#         # Priors for unknown parameters
#         baseline = pm.Uniform("baseline", lower=0, upper=1)
#         revival_time = pm.Uniform("revival_time", lower=0, upper=taus[-1])
#         decay_time = pm.HalfNormal("decay_time", sigma=taus[-1] / 3)
#         amp1 = pm.HalfNormal("amp1", sigma=0.5)
#         amp2 = pm.HalfNormal("amp2", sigma=0.5)
#         freq = pm.Uniform("freq", lower=0, upper=10)
#         phase = pm.Uniform("phase", lower=0, upper=2 * np.pi)

#         # Likelihood
#         mu = advanced_quartic_decay(
#             taus, baseline, revival_time, decay_time, amp1, amp2, freq, phase
#         )
#         sigma = pm.HalfNormal("sigma", sigma=0.1)
#         likelihood = pm.Normal(
#             "likelihood", mu=mu, sigma=sigma, observed=smooth_counts_single
#         )

#         # Inference
#         # trace = pm.sample(1000, return_inferencedata=False, cores=1)
#         # Inference with multiple cores and chains
#         # trace = pm.sample(1000, chains=4, cores=4, return_inferencedata=False)
#         # trace = pm.sample(500, chains=2, cores=2, target_accept=0.9, max_treedepth=15)
#         trace = pm.sample(
#             600, chains=4, cores=4, target_accept=0.9, max_treedepth=15, method="slice"
#         )

#     popt = [
#         np.mean(trace[param])
#         for param in [
#             "baseline",
#             "revival_time",
#             "decay_time",
#             "amp1",
#             "amp2",
#             "freq",
#             "phase",
#         ]
#     ]
#     pcov = np.cov(
#         [
#             trace[param]
#             for param in [
#                 "baseline",
#                 "revival_time",
#                 "decay_time",
#                 "amp1",
#                 "amp2",
#                 "freq",
#                 "phase",
#             ]
#         ]
#     )
#     return popt, pcov, trace


# def fit_spin_echo(taus, smooth_counts_single, initial_guess):
#     """
#     Fit spin echo data for all NV centers using a quartic decay model with enhanced initial guess.

#     Parameters:
#     taus (array): Evolution times (in microseconds).
#     smooth_counts (2D array): Smoothed spin echo counts (one row per NV).
#     initial_guess (list): Initial guess for fitting parameters.

#     Returns:
#     popt (array): Optimal parameters for the fit.
#     pcov (2D array): Covariance matrix of the parameters.
#     """

#     # Define bounds for curve fitting
#     bounds = (
#         [0, 0, 1e-6, 0, 0, 0],  # Lower bounds
#         [1, taus[-1], np.inf, np.inf, np.inf, np.inf],  # Upper bounds
#     )

#     try:
#         # Fit the spin echo data using curve_fit with more robust settings
#         popt, pcov = curve_fit(
#             quartic_decay,
#             taus,
#             smooth_counts_single,
#             p0=initial_guess,
#             bounds=bounds,
#             maxfev=10000,  # Increase maximum function evaluations
#             method="trf",  # Use Trust Region Reflective algorithm for better handling of bounds
#         )

#     except Exception as e:
#         print(f"Fit failed: {e}")
#         popt, pcov = None, None

#     return popt, pcov


def plot_spin_echo_fit(taus, smooth_counts, fit_results):
    """
    Plot spin echo data and fitted curves for each NV center in separate figures.

    Parameters:
    taus (array): Evolution times (in microseconds).
    smooth_counts (2D array): Smoothed spin echo counts (one row per NV).
    fit_results (list): List of tuples (popt, pcov) for each NV center.
    """
    num_nvs = len(smooth_counts)

    for nv_idx in range(num_nvs):
        smooth_counts_single = smooth_counts[nv_idx, :]
        popt, _ = fit_results[nv_idx]

        plt.figure(figsize=(8, 5))
        plt.plot(taus, smooth_counts_single, "o", label=f"{nv_idx}", markersize=4)

        if popt is not None:
            plt.plot(taus, quartic_decay(taus, *popt), "-", linewidth=2)

        plt.xlabel("Time (µs)")
        plt.ylabel("Normalized Counts")
        plt.title(f"Spin Echo Fit for NV {nv_idx}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    kpl.init_kplotlib()

    # Define the file IDs to process
    file_ids = [
        1734158411844,
        1734273666255,
        1734371251079,
        1734461462293,
        1734569197701,
    ]

    try:
        # Process and analyze data from multiple files
        data = process_multiple_files(file_ids)
        nv_list = data["nv_list"]
        taus = data["taus"]
        taus = 2 * np.array(taus) / 1e3
        counts = np.array(data["counts"])
        sig_counts, ref_counts = counts[0], counts[1]

        # Normalize and preprocess counts
        norm_counts, norm_counts_ste = widefield.process_counts(
            nv_list, sig_counts, ref_counts, threshold=True
        )
        smooth_counts = preprocess_spin_echo_data(taus, norm_counts)
        # smooth_counts = norm_counts
        print(f"Smooth counts shape: {smooth_counts.shape}")
        # Detect revival times for all NV centers
        revival_times_all = detect_revivals(taus, smooth_counts)
        # Initialize lists to store results
        fit_results = []
        all_traces = []
        # Loop over each NV center
        for nv_idx in range(len(smooth_counts)):
            smooth_counts_single = smooth_counts[nv_idx, :]
            revival_times_single = revival_times_all[nv_idx]
            # Generate initial guess for the current NV center
            initial_guess = generate_initial_guess(
                taus, smooth_counts_single, revival_times_single
            )
            # Fit the spin echo data for the current NV center
            try:
                # popt, pcov = fit_spin_echo(taus, smooth_counts_single, initial_guess)
                popt, pcov, trace = bayesian_fit_spin_echo(
                    taus, smooth_counts_single, initial_guess
                )
                fit_results.append((popt, pcov))
                all_traces.append(trace)
            except Exception as e:
                print(f"Fit failed for NV {nv_idx}: {e}")
                all_traces.append(None)
                fit_results.append((None, None))

        # Combine all traces into a single InferenceData object
        combined_trace = az.concat(all_traces, dim="nv_idx")
        az.to_netcdf(combined_trace, "combined_trace.nc")
        # Plot the spin echo fits for all NV centers
        plot_spin_echo_fit(taus, smooth_counts, fit_results)

    except Exception as e:
        print(f"Error occurred: {e}")
        print(traceback.format_exc())

    kpl.show(block=True)
