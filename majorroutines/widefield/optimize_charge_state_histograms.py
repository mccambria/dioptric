# -*- coding: utf-8 -*-
"""
Real-time acquisition, histogram analysis, and SLM weight adjustment.

Created on Oct 26, 2024

@author: sbcahnd
"""

import os
import traceback
from datetime import datetime
from time import sleep  # For real-time updates

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

from majorroutines.widefield import base_routine
from slmsuite.hardware.cameras.thorlabs import ThorCam
from slmsuite.hardware.cameraslms import FourierSLM
from slmsuite.hardware.slms.thorlabs import ThorSLM
from slmsuite.holography.algorithms import SpotHologram
from utils import data_manager as dm
from utils.tool_belt import determine_charge_state_threshold


# Connect devices
def connect_devices():
    try:
        slm = ThorSLM(serialNumber="00429430")
        cam = ThorCam(serial="26438", verbose=True)
        fs = FourierSLM(cam, slm)
        print("Connected to SLM and camera.")
        return slm, cam, fs
    except Exception as e:
        print(f"Error connecting devices: {e}")
        raise


# Update SLM phase based on weights
def update_slm_phase(slm, fs, coords, weights):
    try:
        hologram = SpotHologram(
            shape=(2048, 2048), spot_vectors=coords, spot_amp=weights, cameraslm=fs
        )
        hologram.optimize("WGS-Kim", maxiter=30, feedback="computational_spot")
        phase = hologram.extract_phase()
        slm.write(phase, settle=True)
    except Exception as e:
        print(f"SLM phase update error: {e}")


# Capture and plot feedback from the camera
def capture_and_plot(cam):
    img = cam.get_image()
    plt.imshow(img, cmap="gray")
    plt.title("Camera Feedback")
    plt.show()


# Process histograms and calculate fidelity
def process_histograms(sig_counts, ref_counts):
    noise = np.sqrt(np.var(sig_counts) + np.var(ref_counts))
    signal = np.mean(ref_counts) - np.mean(sig_counts)
    snr = round(signal / noise, 3)

    fidelity = np.sum(sig_counts < np.mean(ref_counts)) / len(sig_counts)
    print(f"SNR: {snr}, Fidelity: {fidelity}")
    return fidelity, snr


# Adjust weights based on fidelity
def adjust_weights(weights, fidelity, threshold=0.9):
    if fidelity < threshold:
        weights *= 1.1  # Increase weight for low-fidelity NVs
    return weights / np.max(weights)


# Save data from each step
def save_step_data(step, data, timestamp):
    file_path = dm.get_file_path(__file__, timestamp, f"step_{step}_data")
    dm.save_raw_data(data, file_path)
    print(f"Step {step} data saved.")


# Apply new weights to the SLM
def apply_slm_weights(slm, coords, weights):
    hologram = SpotHologram(shape=(4096, 2048), spot_vectors=coords.T, spot_amp=weights)
    hologram.optimize("WGS-Kim", maxiter=30)
    slm.write(hologram.extract_phase(), settle=True)


# Load NV coordinates and weights
def load_nv_coords(
    file_path="slmsuite/nv_blob_detection/nv_blob_filtered_128nvs_updated.npz",
):
    data = np.load(file_path, allow_pickle=True)
    return data["nv_coordinates"], data["spot_weights"]


# Adjust total power configuration
def adjust_total_power(target_power):
    config_path = "path/to/purcell_config.py"
    with open(config_path, "r") as f:
        config = f.read()

    new_config = config.replace(
        '"yellow_charge_readout": {"type": "constant", "sample": 0.36}',
        f'"yellow_charge_readout": {{"type": "constant", "sample": {target_power}}}',
    ).replace(
        '"green_aod_cw-charge_pol": {"type": "constant", "sample": 0.11}',
        f'"green_aod_cw-charge_pol": {{"type": "constant", "sample": {target_power}}}',
    )

    with open(config_path, "w") as f:
        f.write(new_config)
    print(f"Updated power to {target_power}")


# Real-time acquisition and optimization loop
def real_time_acquisition(slm, nv_list, steps=10, target_power=0.4):
    timestamp = dm.get_time_stamp()
    coords, weights = load_nv_coords()
    fixed_weights = np.zeros(len(nv_list), dtype=bool)

    for step in range(steps):
        print(f"Step {step + 1}/{steps}")

        # Acquire data
        raw_data = base_routine.main(nv_list, num_reps=10, num_runs=5)
        sig_counts, ref_counts = raw_data["counts"][0], raw_data["counts"][1]

        # Process histograms
        for i, ref_list in enumerate(ref_counts):
            _, fidelity = process_histograms(sig_counts[i], ref_list)
            if fidelity > 0.9:
                fixed_weights[i] = True  # Lock weight if fidelity is high

        # Adjust weights
        weights = adjust_weights(weights, fidelity=0.9)

        # Save step data
        raw_data["weights"] = weights
        save_step_data(step, raw_data, timestamp)

        # Update SLM with new weights
        apply_slm_weights(slm, nv_list, weights)

        hologram = SpotHologram(shape=(4096, 2048), spot_vectors=coords.T, spot_amp=weights)
        hologram.optimize("WGS-Kim", maxiter=30)
        slm.write(hologram.extract_phase(), settle=True)


        # Plot histograms in real time
        for i, (sig, ref) in enumerate(zip(sig_counts, ref_counts)):
            plt.figure()
            plt.hist(sig, bins=50, alpha=0.5, label="Signal")
            plt.hist(ref, bins=50, alpha=0.5, label="Reference")
            plt.title(f"NV{i} Histogram")
            plt.legend()
            plt.show()

        # Adjust total power if necessary
        # if np.sum(weights) < 0.8:
        #     adjust_total_power(target_power)

        sleep(1)  # Pause for real-time updates


# Main function to connect devices and run optimization
def main(nv_list):
    try:
        slm, cam, fs = connect_devices()
        fs.load_fourier_calibration("slmsuite/fourier_calibration/26438-SLM-fourier-calibration_00003.h5")
        real_time_acquisition(slm, nv_list)
    finally:
        print("Closing devices...")
        slm.close_window()
        slm.close_device()
        cam.close()


if __name__ == "__main__":
    main()
