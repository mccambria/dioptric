# -*- coding: utf-8 -*-
"""
Optimize SCC parameters

Created on December 6th, 2023

@author: mccambria
"""

import traceback
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.optimize import curve_fit
from mpl_toolkits.mplot3d import Axes3D
from majorroutines.widefield import base_routine
from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import tool_belt as tb
from utils import widefield as widefield
from utils.constants import NVSig, VirtualLaserKey
from scipy.ndimage import gaussian_filter


def process_and_plot(data):
    """
    Process and save average counts and standard errors for the SCC optimization experiment.

    Parameters
    ----------
    data : dict
        Dictionary containing experiment data.
    selected_orientations : list
        List of selected orientations (e.g., ["0.041", "0.147"]).

    Returns
    -------
    processed_data : dict
        Processed data for further analysis.
    figs : list
        List of matplotlib figures generated during plotting.
    """
    # Parse input data
    nv_list = data["nv_list"]
    counts = np.array(data["counts"])
    sig_counts = counts[0]
    ref_counts = counts[1]
    step_vals = np.array(data["step_vals"])
    duration_vals = np.unique(step_vals[:, 0])
    amp_vals = np.unique(step_vals[:, 1])

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

    # Standard errors for signal and reference counts
    # avg_sig_counts, avg_sig_counts_ste = widefield.average_counts(sig_counts)
    # avg_ref_counts, avg_ref_counts_ste = widefield.average_counts(ref_counts)
    avg_counts, avg_counts_ste, _ = widefield.average_counts(sig_counts, ref_counts)
    avg_snr, avg_snr_ste = widefield.calc_snr(sig_counts, ref_counts)
    avg_counts, avg_counts_ste, norms = widefield.average_counts(sig_counts, ref_counts)
    # Reshape data into 2D grids
    num_dur_steps = len(duration_vals)
    num_amp_steps = len(amp_vals)
    avg_counts_grid = avg_counts.reshape(
    avg_counts_grid = avg_counts.reshape(
        len(selected_indices), num_dur_steps, num_amp_steps
    )
    avg_counts_ste_grid = avg_counts_ste.reshape(
        len(selected_indices), num_dur_steps, num_amp_steps
    )
    avg_snr_grid = avg_snr.reshape(len(selected_indices), num_dur_steps, num_amp_steps)
    avg_snr_ste_grid = avg_snr_ste.reshape(
        len(selected_indices), num_dur_steps, num_amp_steps
    )

    # Save all processed data
    processed_data = {
        "nv_list": nv_list,
        "step_vals": step_vals,
        "avg_counts_grid": avg_counts_grid,
        "avg_counts_ste_grid": avg_counts_ste_grid,
        "avg_snr_grid": avg_snr_grid,
        "avg_snr_ste_grid": avg_snr_ste_grid,
        "amp_vals": amp_vals,
        "duration_vals": duration_vals,
    }

    # # Save data to a file
    # timestamp = dm.get_time_stamp()
    # file_name = dm.get_file_name(file_id=1723161184641)
    # file_path = dm.get_file_path(__file__, timestamp, f"{file_name}_processed")
    # dm.save_raw_data(processed_data, file_path)
    # print(f"Processed data saved to: {file_path}")

    # Visualization (optional)
    # figs = []
    # for nv_idx, snr_2d in enumerate(avg_snr_grid):
    #     fig, ax = plt.subplots()
    #     cax = ax.imshow(
    #         snr_2d,
    #         extent=(
    #             amp_vals.min(),
    #             amp_vals.max(),
    #             duration_vals.min(),
    #             duration_vals.max(),
    #         ),
    #         aspect="auto",
    #         cmap="viridis",
    #     )
    #     ax.set_title(f"NV {nv_idx} - SNR Heatmap")
    #     ax.set_xlabel("Amplitude")
    #     ax.set_ylabel("Duration")
    #     fig.colorbar(cax, label="SNR")
    #     plt.show()
    #     figs.append(fig)
    return processed_data


def fit_2d_snr(duration_vals, amp_vals, snr_grid, snr_ste_grid):
    """
    Perform 2D fitting on SNR data over duration and amplitude, including uncertainties.

    """
    # atten the grid for fitting
    # snr_grid = gaussian_filter(snr_grid, sigma=2)
    # Crop the SNR grid and standard error grid
    # edge_pixels = 1

    # def crop_grid(grid, edge_pixels):
    #     return grid[edge_pixels:-edge_pixels, edge_pixels:-edge_pixels]

    # snr_grid = crop_grid(snr_grid, edge_pixels)
    # snr_ste_grid = crop_grid(snr_ste_grid, edge_pixels)

    # # Update duration and amplitude values to match the cropped grid
    # duration_vals = duration_vals[edge_pixels:-edge_pixels]
    # amp_vals = amp_vals[edge_pixels:-edge_pixels]

    x, y = np.meshgrid(duration_vals, amp_vals, indexing="ij")
    x_flat = x.flatten()
    y_flat = y.flatten()
    z_flat = snr_grid.flatten()
    sigma_flat = snr_ste_grid.flatten()  # Standard errors

    # # Define the 2D SNR fit function
    def snr_fit_fn(xy, A, x_delay, y_delay, tau_x, tau_y):
        x, y = xy
        x_shifted = x - x_delay
        y_shifted = y - y_delay
        return (
            A * x_shifted * y_shifted * np.exp(-x_shifted / tau_x - y_shifted / tau_y)
        )

    # Define the 2D SNR fit function
    # def snr_fit_fn(xy, A, x_delay, y_delay, tau_x, tau_y):
    #     x, y = xy
    #     x_shifted = np.clip(x - x_delay, 0, None)  # Ensure non-negative
    #     y_shifted = np.clip(y - y_delay, 0, None)  # Ensure non-negative
    #     return (
    #         A * x_shifted * y_shifted * np.exp(-x_shifted / tau_x - y_shifted / tau_y)
    #     )

    # Initial guesses for parameters
    guess_params = [
        np.max(snr_grid),  # Amplitude (A)
        duration_vals[0],  # x_delay (duration delay)
        amp_vals[0],  # y_delay (amplitude delay)
        duration_vals[-1],  # tau_x (duration decay)
        amp_vals[-1],  # tau_y (amplitude decay)
    ]
    # Initial guesses for parameters
    # guess_params = [
    #     np.max(snr_grid),
    #     np.median(duration_vals),
    #     np.median(amp_vals),
    #     duration_vals.ptp() / 4,
    #     amp_vals.ptp() / 4,
    # ]

    try:
        popt, pcov = curve_fit(
            snr_fit_fn,
            (x_flat, y_flat),
            z_flat,
            p0=guess_params,
            sigma=sigma_flat,
            absolute_sigma=True,
            maxfev=20000,
            # bounds=bounds,
        )

        # Calculate residuals and reduced chi-squared
        fit_vals = snr_fit_fn((x_flat, y_flat), *popt)
        residuals = z_flat - fit_vals
        dof = len(z_flat) - len(popt)  # Degrees of freedom
        red_chi_sq = np.sum((residuals / sigma_flat) ** 2) / dof  # Reduced chi-squared

    except Exception as e:
        print(f"2D fitting failed: {e}")
        popt = guess_params
        pcov = np.zeros((len(guess_params), len(guess_params)))  # Default covariance
        red_chi_sq = np.inf

    # Define the fitted 2D function
    def fit_fn(x, y):
        return snr_fit_fn((x, y), *popt)

    # Compute optimal duration and amplitude
    x_lin = np.linspace(duration_vals.min(), duration_vals.max(), 100)
    y_lin = np.linspace(amp_vals.min(), amp_vals.max(), 100)
    x_mesh, y_mesh = np.meshgrid(x_lin, y_lin, indexing="ij")
    snr_fit_surface = fit_fn(x_mesh, y_mesh)
    optimal_idx = np.unravel_index(np.argmax(snr_fit_surface), snr_fit_surface.shape)
    optimal_duration = x_lin[optimal_idx[0]]
    optimal_amplitude = y_lin[optimal_idx[1]]

    return popt, red_chi_sq, fit_fn, (optimal_duration, optimal_amplitude)


def analyze_and_visualize(processed_data):
    """
    Perform additional analysis and generate advanced visualizations.

    """
    # Extract data from the dictionary
    nv_list = processed_data["nv_list"]
    step_vals = processed_data["step_vals"]
    avg_counts_grid = processed_data["avg_counts_grid"]
    avg_counts_ste_grid = processed_data["avg_counts_ste_grid"]
    avg_snr_grid = processed_data["avg_snr_grid"]
    avg_snr_ste_grid = processed_data["avg_snr_ste_grid"]
    amp_vals = np.array(processed_data["amp_vals"])
    duration_vals = np.array(processed_data["duration_vals"])
    # Compute the median SNR across NVs
    median_snr_grid = np.median(avg_snr_grid, axis=0)
    median_snr_grid_ste = np.median(avg_snr_ste_grid, axis=0)
    # median_snr_grid_ste *= np.sqrt(400)
    # Perform 2D fitting
    popt, red_chi_sq, fit_fn, optimal_values = fit_2d_snr(
        duration_vals, amp_vals, median_snr_grid, median_snr_grid_ste
    )
    # Extract optimal values
    optimal_duration, optimal_amplitude = optimal_values
    print(f"Optimal SCC Duration: {optimal_duration:.2f} ns")
    print(f"Optimal SCC Amplitude: {optimal_amplitude:.2f}")
    print(f"Reduced Chi-Squared: {red_chi_sq:.3f}")

    # Visualize the 2D fit and experimental data
    # fig = plt.figure(figsize=(8, 6))
    # ax = fig.add_subplot(111, projection="3d")
    # Plot experimental SNR grid
    # X, Y = np.meshgrid(duration_vals, amp_vals, indexing="ij")
    # ax.scatter(X, Y, median_snr_grid, color="r", label="Experimental Data", alpha=0.6)
    # # Plot fitted surface
    # X_fit = np.linspace(duration_vals.min(), duration_vals.max(), 100)
    # Y_fit = np.linspace(amp_vals.min(), amp_vals.max(), 100)
    # X_fit_mesh, Y_fit_mesh = np.meshgrid(X_fit, Y_fit, indexing="ij")
    # Z_fit = fit_fn(X_fit_mesh, Y_fit_mesh)

    # # Mark the optimal values
    # optimal_snr = fit_fn(optimal_duration, optimal_amplitude)
    # ax.scatter(
    #     optimal_duration,
    #     optimal_amplitude,
    #     optimal_snr,
    #     color="blue",
    #     label=f"Optimal (Dur: {optimal_duration:.0f} ns, Amp: {optimal_amplitude:.3f})",
    #     s=60,
    #     edgecolors="black",
    #     # zorder=10,
    # )
    # ax.plot_surface(X_fit_mesh, Y_fit_mesh, Z_fit, cmap="viridis", alpha=0.7)

    # Add text for reduced chi-squared
    # ax.text2D(
    #     0.05,
    #     0.95,
    #     f"Reduced Chi-Squared: {red_chi_sq:.3f}",
    #     transform=ax.transAxes,
    #     fontsize=10,
    #     color="blue",
    # )

    # Set tight axes
    # ax.set_xlim(duration_vals.min(), duration_vals.max())
    # ax.set_ylim(amp_vals.min(), amp_vals.max())
    # ax.set_zlim(median_snr_grid.min(), median_snr_grid.max())
    # ax.set_box_aspect([1, 1, 0.8])  # Proportionally scale axes
    # # Customize titles and labels
    # ax.set_title("Fit of Median SCC SNR Grid", fontsize=14)
    # ax.set_xlabel("Duration (ns)", fontsize=12)
    # ax.set_ylabel("Amplitude", fontsize=12)
    # ax.set_zlabel("SNR", fontsize=12)
    # ax.tick_params(axis="x", labelsize=10)
    # ax.tick_params(axis="y", labelsize=10)
    # ax.tick_params(axis="z", labelsize=10)
    # ax.legend(fontsize=10, loc="best")
    # plt.tight_layout()
    # plt.show()

    # Plot the heatmap for the median SNR
    # fig, ax = plt.subplots()
    # cax = ax.imshow(
    #     median_snr_grid,
    #     extent=(
    #         amp_vals.min(),
    #         amp_vals.max(),
    #         duration_vals.min(),
    #         duration_vals.max(),
    #     ),
    #     aspect="auto",
    #     cmap="coolwarm",
    #     origin="lower",
    # )
    # ax.set_title("Median SCC SNR Across NVs")
    # ax.set_xlabel("SCC Amplitude")
    # ax.set_ylabel("SCC Duration (ns)")
    # fig.colorbar(cax, label="Median SCC SNR")
    # plt.show()

    # Individual NV SNR fitting
    optimal_durations = []
    optimal_amplitudes = []
    red_chi_sqs = []
    for nv_ind, nv_snr_grid in enumerate(avg_snr_grid):
        nv_snr_grid = np.array(nv_snr_grid)
        nv_snr_ste_grid = np.array(avg_snr_ste_grid[nv_ind])
        popt, red_chi_sq, fit_fn, optimal_values = fit_2d_snr(
            duration_vals, amp_vals, nv_snr_grid, nv_snr_ste_grid
        )
        optimal_duration, optimal_amplitude = optimal_values
        optimal_durations.append(optimal_duration)
        optimal_amplitudes.append(optimal_amplitude)
        red_chi_sqs.append(red_chi_sq)

        print(
            f"NV {nv_ind}: Optimal Values: ({optimal_values[0]:.0f}, {optimal_values[1]:.3f}), Red. Chi-Sq: {red_chi_sq:.3f}"
        )
        # Scatter plot for optimal durations vs. optimal amplitudes
    plt.figure(figsize=(6.5, 5))
    plt.scatter(
        optimal_durations,
        optimal_amplitudes,
        c=red_chi_sqs,
        cmap="viridis",
        s=60,
        edgecolors="black",
    )
    plt.colorbar(label="Reduced Chi-Squared")
    plt.xlabel("Optimal Duration (ns)", fontsize=12)
    plt.ylabel("Optimal Amplitude", fontsize=12)
    plt.title("Optimal SCC Parameters Across NVs", fontsize=14)
    plt.grid(alpha=0.3)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()


# # Plot Median SNR vs. SCC Amplitude for each duration
# fig1, ax1 = plt.subplots()
# for i, duration in enumerate(duration_vals):
#     ax1.plot(amp_vals, median_snr_grid[i, :], label=f"{duration}", marker="o")
# ax1.set_xlabel("SCC Amplitude")
# ax1.set_ylabel("Median SCC SNR")
# ax1.set_title("Median SCC SNR Across NVs")
# ax1.legend(title="Durations (ns)", fontsize=9, title_fontsize=10)
# plt.show()

# # Plot Median SNR vs. SCC Duration for each amplitude
# fig2, ax2 = plt.subplots()
# for j, amplitude in enumerate(amp_vals):
#     ax2.plot(
#         duration_vals,
#         median_snr_grid[:, j],
#         label=f"{amplitude:.2f}",
#         marker="o",
#     )
# ax2.set_xlabel("SCC Duration")
# ax2.set_ylabel("Median SCC SNR")
# ax2.set_title("Median SCC SNR Across NVs")
# ax2.legend(title="Amplitude (relative)", fontsize=9, title_fontsize=10)
# plt.show()


if __name__ == "__main__":
    kpl.init_kplotlib()
    # file_id = 1728131481474
    # data = dm.get_raw_data(file_id=file_id)
    # process_and_plot(data)
    # processed data analysis
    # processed_data_id = 1723819842491
    processed_data_id = 1728147590280
    processed_data = dm.get_raw_data(file_id=processed_data_id)
    analyze_and_visualize(processed_data)
    plt.show(block=True)
