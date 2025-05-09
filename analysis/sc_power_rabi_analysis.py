# -*- coding: utf-8 -*-
"""
Pulsed electron spin resonance on multiple NVs with spin-to-charge
conversion readout imaged onto a camera

Created on Fall, 2024

@author: saroj chand
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
from majorroutines.widefield import base_routine
from utils import common
from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import positioning as pos
from utils import tool_belt as tb
from utils import widefield as widefield
from utils.constants import NVSig, NVSpinState
from utils.positioning import get_scan_1d as calculate_powers

# -*- coding: utf-8 -*-
"""
Analysis of NV Resonance Data for 160 NVs with Two Signal Generators

Created on November 19th, 2023

"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from majorroutines.pulsed_resonance import voigt
from utils import data_manager as dm


def find_optimal_uwave_power(powers, norm_counts):
    """
    Find the optimal microwave power by analyzing the resonance data.

    Args:
        powers: List of microwave powers used.
        norm_counts: Normalized NV population counts corresponding to each power.

    Returns:
        optimal_power_1: Optimal microwave power value for the first center.
        optimal_power_2: Optimal microwave power value for the second center.
    """

    def voigt_fit(x, amp, cen, width, offset):
        # Example with a simplified Voigt function
        return voigt(x, amp, cen, width, offset)  # Adjust the parameters as needed

    # Adjusted initial guess according to the expected parameters for voigt_fit
    initial_guess = [
        np.max(norm_counts),  # Amplitude
        np.mean(powers),  # Center of resonance
        1,  # Width (arbitrary initial guess)
        np.min(norm_counts),  # Offset
    ]
    popt, _ = curve_fit(voigt_fit, powers, norm_counts, p0=initial_guess)

    # Assuming the optimal power is represented by the fitted center
    optimal_power = popt[1]
    return optimal_power, optimal_power  # Assuming two outputs for compatibility


def plot_optimal_power(powers, norm_counts, optimal_power_1, optimal_power_2, nv_idx):
    """
    Plot the resonance data and indicate the optimal microwave power for two signal generators.

    Args:
        powers: List of microwave powers used.
        norm_counts: Normalized NV population counts.
        optimal_power_1: Optimal microwave power value for the first signal generator.
        optimal_power_2: Optimal microwave power value for the second signal generator.
        nv_idx: Index of the NV being analyzed.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(
        powers,
        norm_counts,
        marker="o",
        linestyle="-",
        color="b",
        label="Resonance Data",
    )
    plt.axvline(
        optimal_power_1,
        color="r",
        linestyle="--",
        label=f"Optimal Power 1: {optimal_power_1:.2f} dBm",
    )
    plt.axvline(
        optimal_power_2,
        color="g",
        linestyle="--",
        label=f"Optimal Power 2: {optimal_power_2:.2f} dBm",
    )
    plt.xlabel("Microwave Power (dBm)")
    plt.ylabel("Normalized NV- Population")
    plt.title(f"Optimal Microwave Power for NV {nv_idx + 1}")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()


def main():
    data = dm.get_raw_data(
        file_id=1696792280595  # Replace with actual file_id as needed
    )

    nv_list = data["nv_list"]
    sig_counts, ref_counts = np.array(data["states"])[0], np.array(data["states"])[1]
    powers = data["powers"]

    # Process counts (assuming 160 NVs)
    avg_counts, avg_counts_ste, norms = widefield.process_counts(
        nv_list, sig_counts, ref_counts, threshold=False
    )
    norm_counts = avg_counts - norms[0][:, np.newaxis]
    print(f"powers shape: {len(powers)}, norm_counts shape: {norm_counts.shape}")

    # Calculate optimal power for each NV and each signal generator (0 and 1)
    for nv_idx in range(len(nv_list)):
        # Ensure correct shape extraction for norm_counts_nv
        norm_counts_nv = norm_counts[nv_idx]  # This should be a 1D array
        if norm_counts_nv.ndim != 1:
            norm_counts_nv = norm_counts_nv.mean(
                axis=0
            )  # Average across dimensions if necessary

        if norm_counts_nv.shape != (len(powers),):
            print(f"Unexpected shape for norm_counts_nv: {norm_counts_nv.shape}")
            continue  # Skip if the shape is still incorrect

        optimal_power_1, optimal_power_2 = find_optimal_uwave_power(
            powers, norm_counts_nv
        )
        plot_optimal_power(
            powers, norm_counts_nv, optimal_power_1, optimal_power_2, nv_idx
        )
        print(
            f"NV {nv_idx + 1}: Optimal Power 1 = {optimal_power_1:.2f} dBm, Optimal Power 2 = {optimal_power_2:.2f} dBm"
        )


if __name__ == "__main__":
    main()


# def create_raw_data_figure(data):
#     nv_list = data["nv_list"]
#     num_nvs = len(nv_list)
#     powers = data["powers"]
#     counts = np.array(data["states"])


# # Define a Gaussian function to fit the resonance peak
# def gaussian(x, a, x0, sigma):
#     return a * np.exp(-((x - x0) ** 2) / (2 * sigma**2))


# def find_optimal_power(powers, norm_counts):
#     """
#     Find the optimal microwave power by fitting the data to a Gaussian and identifying the peak.

#     Parameters:
#     powers (ndarray): Array of microwave power values.
#     norm_counts (ndarray): Array of normalized NV- population counts.

#     Returns:
#     optimal_power (float): The optimal microwave power at the peak of the fitted curve.
#     popt (ndarray): The optimized parameters of the fitted Gaussian.
#     """
#     # Initial guess for the Gaussian fit parameters: amplitude, center, and width
#     initial_guess = [np.max(norm_counts), powers[np.argmax(norm_counts)], 1.0]

#     try:
#         # Perform Gaussian curve fitting
#         popt, _ = curve_fit(gaussian, powers, norm_counts, p0=initial_guess)
#         optimal_power = popt[
#             1
#         ]  # Extract the optimal power (center of the Gaussian curve)
#     except RuntimeError:
#         # If fitting fails, return None for both values
#         optimal_power = np.nan
#         popt = None

#     return optimal_power, popt


# def plot_all_nv_data(nv_list, powers, norm_counts_list, num_cols=3):
#     """
#     Plot NV resonance data for all NVs using Seaborn aesthetics and return the optimal power for each NV.

#     Parameters:
#     nv_list (list): List of NVs to plot.
#     powers (ndarray): Array of microwave power values.
#     norm_counts_list (ndarray): Normalized NV population data for each NV.
#     num_cols (int): Number of columns in the subplot grid.

#     Returns:
#     optimal_powers (list): List of optimal microwave powers for each NV.
#     """
#     sns.set(style="whitegrid", palette="muted")

#     num_nvs = len(nv_list)
#     num_rows = int(np.ceil(num_nvs / num_cols))  # Calculate number of rows needed

#     fig, axes = plt.subplots(
#         num_rows, num_cols, figsize=(num_cols * 5, num_rows * 4), sharex=True
#     )
#     axes = axes.flatten()

#     optimal_powers = []  # List to store the optimal power for each NV

#     for nv_idx, ax in enumerate(axes):
#         if nv_idx < num_nvs:
#             # Get normalized counts for this NV
#             norm_counts = norm_counts_list[nv_idx]

#             # Find the optimal microwave power
#             optimal_power, max_population = extract_optimal_power(norm_counts, powers)
#             optimal_powers.append(optimal_power)

#             # Plot the raw data
#             sns.lineplot(
#                 x=powers, y=norm_counts, ax=ax, lw=2, marker="o", label=f"NV {nv_idx+1}"
#             )

#             # If the optimal power was found, plot the Gaussian fit and the optimal line
#             if not np.isnan(optimal_power):
#                 ax.axvline(
#                     optimal_power,
#                     color="green",
#                     linestyle="--",
#                     label=f"Opt. Power = {optimal_power:.2f} dBm",
#                 )

#             # Set labels, grid, and legend
#             ax.set_xlabel("Microwave Power (dBm)")
#             ax.set_ylabel("Normalized NV- Population")
#             ax.grid(True, linestyle="--", linewidth=0.5)
#             ax.set_ylabel("Normalized NV Population")
#             ax.grid(True, linestyle="--", linewidth=0.5)
#             ax.legend()
#         else:
#             ax.axis("off")  # Hide unused subplots if number of NVs < grid size

#     plt.tight_layout()
#     plt.show()

#     return optimal_powers


# def extract_optimal_power(norm_counts, powers):
#     """
#     Extract the optimal microwave power based on NV population data.

#     Parameters:
#     norm_counts (ndarray): Normalized NV population for different powers.
#     powers (ndarray): Array of microwave power values.

#     Returns:
#     optimal_power (float): The microwave power where the NV population is maximized.
#     max_population (float): The maximum NV population value at the optimal power.
#     """
#     # Average the population over all NVs if needed
#     avg_population = (
#         np.mean(norm_counts, axis=0) if norm_counts.ndim > 1 else norm_counts
#     )

#     # Find the index of the maximum population
#     optimal_index = np.argmax(avg_population)

#     # Get the corresponding power value
#     optimal_power = powers[optimal_index]

#     return optimal_power, avg_population[optimal_index]


# if __name__ == "__main__":
#     # Load data using dm.get_raw_data
#     file_id = 1661020621314
#     file_id = 1666943042304
#     data = dm.get_raw_data(file_id=file_id)

#     # Extract necessary information from the data
#     nv_list = data["nv_list"]
#     powers = data["powers"]  # Assuming powers correspond to microwave power sweep
#     counts = np.array(data["states"])
#     sig_counts, ref_counts = counts[0], counts[1]

#     # Process counts (replace widefield.process_counts with the correct function)
#     avg_counts, avg_counts_ste, norms = widefield.process_counts(
#         nv_list, sig_counts, ref_counts, threshold=False
#     )
#     norm_counts_list = avg_counts - norms[0][:, np.newaxis]

#     # Plot all NV data and return the optimal powers
#     optimal_powers = plot_all_nv_data(nv_list, powers, norm_counts_list)

#     # Print the optimal powers for each NV
#     for idx, nv in enumerate(nv_list):
#         print(f"Optimal power for NV {nv.name}: {optimal_powers[idx]:.2f} dBm")

#     print(f"Optimal Powers for each NV: {optimal_powers}")
