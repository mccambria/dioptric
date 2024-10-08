# -*- coding: utf-8 -*-
"""
Pulsed electron spin resonance on multiple NVs with spin-to-charge
conversion readout imaged onto a camera

Created on November 19th, 2023

@author: mccambria
"""

import os
import sys
import time
from random import shuffle

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.optimize import curve_fit

from majorroutines.pulsed_resonance import fit_resonance, voigt, voigt_split
from majorroutines.widefield import base_routine, optimize
from utils import common
from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import positioning as pos
from utils import tool_belt as tb
from utils import widefield as widefield
from utils.constants import NVSig, NVSpinState
from utils.positioning import get_scan_1d as calculate_powers


def create_raw_data_figure(data):
    nv_list = data["nv_list"]
    num_nvs = len(nv_list)
    powers = data["powers"]
    counts = np.array(data["states"])


# Define a Gaussian function to fit the resonance peak
def gaussian(x, a, x0, sigma):
    return a * np.exp(-((x - x0) ** 2) / (2 * sigma**2))


def find_optimal_power(powers, norm_counts):
    """
    Find the optimal microwave power by fitting the data to a Gaussian and identifying the peak.

    Parameters:
    powers (ndarray): Array of microwave power values.
    norm_counts (ndarray): Array of normalized NV- population counts.

    Returns:
    optimal_power (float): The optimal microwave power at the peak of the fitted curve.
    popt (ndarray): The optimized parameters of the fitted Gaussian.
    """
    # Initial guess for the Gaussian fit parameters: amplitude, center, and width
    initial_guess = [np.max(norm_counts), powers[np.argmax(norm_counts)], 1.0]

    try:
        # Perform Gaussian curve fitting
        popt, _ = curve_fit(gaussian, powers, norm_counts, p0=initial_guess)
        optimal_power = popt[
            1
        ]  # Extract the optimal power (center of the Gaussian curve)
    except RuntimeError:
        # If fitting fails, return None for both values
        optimal_power = np.nan
        popt = None

    return optimal_power, popt


def plot_all_nv_data(nv_list, powers, norm_counts_list, num_cols=3):
    """
    Plot the NV resonance data for all NVs using Seaborn aesthetics and return the optimal power for each NV.

    Parameters:
    nv_list (list): List of NVs to plot.
    powers (ndarray): Array of microwave power values.
    norm_counts_list (ndarray): List of normalized NV population data for each NV.
    num_cols (int): Number of columns in the subplot grid.

    Returns:
    optimal_powers (list): List of optimal microwave powers for each NV.
    """
    sns.set(style="whitegrid", palette="muted")

    num_nvs = len(nv_list)
    num_rows = int(np.ceil(num_nvs / num_cols))  # Calculate number of rows needed

    fig, axes_pack = plt.subplots(
        num_rows, num_cols, figsize=(num_cols * 5, num_rows * 4), sharex=True
    )
    axes_pack = axes_pack.flatten()

    optimal_powers = []  # List to store the optimal power for each NV

    for nv_idx, ax in enumerate(axes_pack):
        if nv_idx < num_nvs:
            # Get normalized counts for this NV
            norm_counts = norm_counts_list[nv_idx]

            # Find the optimal microwave power using the Gaussian fit
            optimal_power, popt = find_optimal_power(powers, norm_counts)
            optimal_powers.append(optimal_power)

            # Plot the raw data and the fitted Gaussian curve
            sns.lineplot(
                x=powers, y=norm_counts, ax=ax, lw=2, marker="o", label=f"NV {nv_idx+1}"
            )
            if not np.isnan(optimal_power):
                ax.plot(powers, gaussian(powers, *popt), "r-", label="Gaussian fit")
                ax.axvline(
                    optimal_power,
                    color="green",
                    linestyle="--",
                    label=f"Opt. Power = {optimal_power:.2f} dBm",
                )

            # Set labels and grid
            ax.set_xlabel("Microwave Power (dBm)")
            ax.set_ylabel("Normalized NV- Population")
            ax.grid(True, linestyle="--", linewidth=0.5)
            ax.legend()

        else:
            ax.axis("off")  # Hide unused subplots if number of NVs < grid size

    plt.tight_layout()
    plt.show()

    return optimal_powers


def main(file_id=1661020621314, num_cols=3):
    """
    Main function to load the data, process it, and plot NVs.

    Parameters:
    file_id: ID or path of the data file.
    num_cols: Number of columns for the grid layout.
    """
    # Load data using dm.get_raw_data
    data = dm.get_raw_data(file_id=file_id)

    # Extract necessary information from the data
    nv_list = data["nv_list"]
    powers = data["powers"]  # Assuming powers correspond to microwave power sweep
    counts = np.array(data["states"])
    sig_counts, ref_counts = counts[0], counts[1]

    # Process counts (replace widefield.process_counts with correct function)
    avg_counts, avg_counts_ste, norms = widefield.process_counts(
        nv_list, sig_counts, ref_counts, threshold=False
    )
    norm_counts_list = avg_counts - norms[0][:, np.newaxis]

    # Plot all NV data and return the optimal powers
    optimal_powers = plot_all_nv_data(
        nv_list, powers, norm_counts_list, num_cols=num_cols
    )

    # Print the optimal powers for each NV
    for idx, nv in enumerate(nv_list):
        print(f"Optimal power for NV {nv.name}: {optimal_powers[idx]:.2f} dBm")


if __name__ == "__main__":
    main(num_cols=7)
