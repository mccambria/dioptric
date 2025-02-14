# -*- coding: utf-8 -*-
"""
Spin Echo Analysis and Visualization

Created on December 22nd, 2024

@author: Saroj Chand
"""

import sys
import time
import traceback

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.optimize import curve_fit

from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import widefield as widefield

import pymc3 as pm
import arviz as az
import pymc3 as pm
import arviz as az
import tensorflow as tf
from tensorflow.keras import layers, models
from scipy.signal import savgol_filter


# Define a decay model for spin echo fitting
def quartic_decay(tau, baseline, revival_time, decay_time, amp1, amp2, freq):
    num_revivals = 3
    value = baseline
    for i in range(num_revivals):
        exp_decay = np.exp(-(((tau - i * revival_time) / decay_time) ** 2))
        mod = amp1 * np.cos(2 * np.pi * freq * tau)
        value -= exp_decay * mod
    return value


# Combine data from multiple file IDs
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


def simulate_spin_echo(tau, baseline, revival_time, decay_time, amp1, amp2, freq):
    num_revivals = 3
    value = baseline
    for i in range(num_revivals):
        exp_decay = np.exp(-(((tau - i * revival_time) / decay_time) ** 2))
        mod = amp1 * np.cos(2 * np.pi * freq * tau)
        value -= exp_decay * mod
    return value


def preprocess_data(tau, counts):
    """
    Apply Savitzky-Golay filter to smooth the counts data.

    Parameters:
    tau (array): Time axis (in microseconds).
    counts (array): Spin echo signal counts.

    Returns:
    smooth_counts (array): Smoothed counts data.
    """
    window_length = min(11, len(tau) - (len(tau) % 2 == 0))  # Ensure odd window length
    polyorder = 3  # Polynomial order for smoothing
    smooth_counts = savgol_filter(
        counts, window_length=window_length, polyorder=polyorder
    )
    return smooth_counts


def build_cnn_model(input_shape):
    """
    Build a CNN model for parameter estimation.

    Parameters:
    input_shape (tuple): Shape of the input data (length of tau, 1).

    Returns:
    model (tf.keras.Model): Compiled CNN model.
    """
    model = models.Sequential(
        [
            layers.Conv1D(
                64, kernel_size=5, activation="relu", input_shape=input_shape
            ),
            layers.MaxPooling1D(pool_size=2),
            layers.Conv1D(128, kernel_size=5, activation="relu"),
            layers.GlobalAveragePooling1D(),
            layers.Dense(64, activation="relu"),
            layers.Dense(
                6
            ),  # Output: 6 parameters (baseline, revival_time, decay_time, amp1, amp2, freq)
        ]
    )
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


def predict_initial_guess(tau, counts):
    """
    Predict initial parameters using the pre-trained CNN model.

    Parameters:
    tau (array): Time axis (in microseconds).
    counts (array): Spin echo signal counts.

    Returns:
    initial_guess (list): Predicted initial parameters [baseline, revival_time, decay_time, amp1, amp2, freq].
    """
    smooth_counts = preprocess_data(tau, counts)
    input_data = smooth_counts.reshape(1, -1, 1)  # Reshape for CNN input
    initial_guess = cnn_model.predict(input_data)[0]
    return initial_guess


def bayesian_inference(tau, counts, initial_guess):
    """
    Perform Bayesian inference using MCMC to estimate spin echo parameters.

    Parameters:
    tau (array): Time axis (in microseconds).
    counts (array): Spin echo signal counts.
    initial_guess (list): Initial guess for [baseline, revival_time, decay_time, amp1, amp2, freq].

    Returns:
    trace (pm.backends.base.MultiTrace): MCMC trace containing posterior samples.
    """
    (
        baseline_guess,
        revival_time_guess,
        decay_time_guess,
        amp1_guess,
        amp2_guess,
        freq_guess,
    ) = initial_guess

    with pm.Model() as model:
        # Priors for unknown parameters
        baseline = pm.Normal("baseline", mu=baseline_guess, sigma=0.1)
        revival_time = pm.Normal("revival_time", mu=revival_time_guess, sigma=0.1)
        decay_time = pm.Normal("decay_time", mu=decay_time_guess, sigma=0.1)
        amp1 = pm.Normal("amp1", mu=amp1_guess, sigma=0.1)
        amp2 = pm.Normal("amp2", mu=amp2_guess, sigma=0.1)
        freq = pm.Normal("freq", mu=freq_guess, sigma=1e6)

        # Likelihood (assuming Gaussian noise)
        sigma = pm.HalfNormal("sigma", sigma=0.05)
        mu = quartic_decay(tau, baseline, revival_time, decay_time, amp1, amp2, freq)
        likelihood = pm.Normal("counts", mu=mu, sigma=sigma, observed=counts)

        # Inference using MCMC
        trace = pm.sample(1000, tune=500, cores=2, return_inferencedata=True)

    # Plot posterior distributions
    az.plot_trace(trace)
    az.summary(trace, hdi_prob=0.95)
    return trace


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
    # Process and analyze data from multiple files
    try:
        data = process_multiple_files(file_ids)
        nv_list = data["nv_list"]
        taus = data["taus"]
        total_evolution_times = 2 * np.array(taus) / 1e3
        counts = np.array(data["counts"])
        sig_counts, ref_counts = counts[0], counts[1]
        norm_counts, norm_counts_ste = widefield.process_counts(
            nv_list, sig_counts, ref_counts, threshold=True
        )
        # Example usage:
        input_shape = (len(taus), 1)
        cnn_model = build_cnn_model(input_shape)
        cnn_model.summary()
        # Simulate 100,000 synthetic spin echo signals
        num_samples = 100000
        synthetic_params = np.random.uniform(
            [0, 1e-6, 1e-6, 0.01, 0.01, 1e6],
            [1, 1e-3, 1e-3, 1, 1, 1e7],
            size=(num_samples, 6),
        )
        synthetic_signals = np.array(
            [simulate_spin_echo(taus, *params) for params in synthetic_params]
        )

        # Reshape data for CNN input
        X_train = synthetic_signals.reshape(-1, len(taus), 1)
        y_train = synthetic_params

        # Train the CNN model
        cnn_model.fit(X_train, y_train, epochs=20, batch_size=64, validation_split=0.1)

        # parameters = analyze_spin_echo(
        #     nv_list, total_evolution_times, norm_counts, norm_counts_ste
        # )

        # Step 1: Predict initial guess using CNN
        initial_guess = predict_initial_guess(taus, norm_counts)

        # Step 2: Perform Bayesian inference
        trace = bayesian_inference(taus, norm_counts, initial_guess)

        # Step 3: Extract and display results
        posterior_params = az.summary(trace, hdi_prob=0.95)
        print(posterior_params)
        # plot_analysis_parameters(parameters)
    except Exception as e:
        print(f"Error occurred: {e}")
        print(traceback.format_exc())

    kpl.show(block=True)
