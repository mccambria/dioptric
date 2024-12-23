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
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit

from majorroutines.widefield import base_routine
from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import tool_belt as tb
from utils import widefield as widefield
from utils.constants import NVSig, VirtualLaserKey


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
    # Filter counts and NV list
    sig_counts, ref_counts = widefield.threshold_counts(
        nv_list, sig_counts, ref_counts, dynamic_thresh=True
    )
    # Standard errors for signal and reference counts
    avg_counts, avg_counts_ste, norms = widefield.average_counts(sig_counts, ref_counts)
    avg_snr, avg_snr_ste = widefield.calc_snr(sig_counts, ref_counts)
    # Reshape data into 2D grids
    num_dur_steps = len(duration_vals)
    num_amp_steps = len(amp_vals)
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
        "norms ": norms,
        "avg_counts_ste_grid": avg_counts_ste_grid,
        "avg_snr_grid": avg_snr_grid,
        "avg_snr_ste_grid": avg_snr_ste_grid,
        "amp_vals": amp_vals,
        "duration_vals": duration_vals,
    }

    # # Save data to a file
    timestamp = dm.get_time_stamp()
    file_name = dm.get_file_name(file_id=1723161184641)
    file_path = dm.get_file_path(__file__, timestamp, f"{file_name}_processed")
    dm.save_raw_data(processed_data, file_path)
    print(f"Processed data saved to: {file_path}")

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

    # Initial guesses for parameters
    guess_params = [
        np.max(snr_grid),  # Amplitude (A)
        duration_vals[0],  # x_delay (duration delay)
        amp_vals[0],  # y_delay (amplitude delay)
        duration_vals[-1],  # tau_x (duration decay)
        amp_vals[-1],  # tau_y (amplitude decay)
    ]

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
    print(f"Reduced Chi-Squared: {red_chi_sq:.2f}")

    # Visualize the 2D fit and experimental data
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    # Plot experimental SNR grid
    X, Y = np.meshgrid(duration_vals, amp_vals, indexing="ij")
    ax.scatter(X, Y, median_snr_grid, color="r", label="Experimental Data", alpha=0.6)
    # Plot fitted surface
    X_fit = np.linspace(duration_vals.min(), duration_vals.max(), 100)
    Y_fit = np.linspace(amp_vals.min(), amp_vals.max(), 100)
    X_fit_mesh, Y_fit_mesh = np.meshgrid(X_fit, Y_fit, indexing="ij")
    Z_fit = fit_fn(X_fit_mesh, Y_fit_mesh)

    # Mark the optimal values
    optimal_snr = fit_fn(optimal_duration, optimal_amplitude)

    ax.scatter(
        optimal_duration,
        optimal_amplitude,
        optimal_snr,
        color="blue",
        label=f"Optimal (Dur: {optimal_duration:.0f} ns, Amp: {optimal_amplitude:.3f})",
        s=60,
        edgecolors="black",
        # zorder=10,
    )
    ax.plot_surface(X_fit_mesh, Y_fit_mesh, Z_fit, cmap="viridis", alpha=0.6)

    # Add text for reduced chi-squared
    ax.text2D(
        0.05,
        0.95,
        f"Reduced Chi-Squared: {red_chi_sq:.3f}",
        transform=ax.transAxes,
        fontsize=10,
        color="blue",
    )

    # Set tight axes
    ax.set_xlim(duration_vals.min(), duration_vals.max())
    ax.set_ylim(amp_vals.min(), amp_vals.max())
    ax.set_zlim(median_snr_grid.min(), median_snr_grid.max())
    ax.set_box_aspect([1, 1, 0.8])  # Proportionally scale axes
    # Customize titles and labels
    ax.set_title("Fit of Median SCC SNR Grid", fontsize=14)
    ax.set_xlabel("Duration (ns)", fontsize=12)
    ax.set_ylabel("Amplitude", fontsize=12)
    ax.set_zlabel("SNR", fontsize=12)
    ax.tick_params(axis="x", labelsize=10)
    ax.tick_params(axis="y", labelsize=10)
    ax.tick_params(axis="z", labelsize=10)
    ax.legend(fontsize=10, loc="best")
    plt.tight_layout()
    plt.show()

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
    optimal_snrs = []
    for nv_ind, nv_snr_grid in enumerate(avg_snr_grid):
        nv_snr_grid = np.array(nv_snr_grid)
        nv_snr_ste_grid = np.array(avg_snr_ste_grid[nv_ind])
        popt, red_chi_sq, fit_fn, optimal_values = fit_2d_snr(
            duration_vals, amp_vals, nv_snr_grid, nv_snr_ste_grid
        )
        optimal_duration, optimal_amplitude = optimal_values
        optimal_snr = fit_fn(optimal_duration, optimal_amplitude)
        optimal_durations.append(optimal_duration)
        optimal_amplitudes.append(optimal_amplitude)
        red_chi_sqs.append(red_chi_sq)
        optimal_snrs.append(optimal_snr)
        print(
            f"NV {nv_ind}: snr: {optimal_snr:.2f} (dur,amp): ({optimal_values[0]:.0f}, {optimal_values[1]:.3f}), red chi: {red_chi_sq:.3f}"
        )
        # Scatter plot for optimal durations vs. optimal amplitudes
    plt.figure(figsize=(6.5, 5))
    plt.scatter(
        optimal_durations,
        optimal_amplitudes,
        c=red_chi_sqs,
        cmap="viridis",
        s=40,
        edgecolors="black",
    )
    for i, (x, y) in enumerate(zip(optimal_durations, optimal_amplitudes)):
        optimal_snr = optimal_snrs[i]
        plt.annotate(f"{optimal_snr:.2f}", (x, y), fontsize=6, alpha=0.6, color="red")
    colorbar = plt.colorbar(label="Reduced Chi-Squared")
    colorbar.ax.tick_params(labelsize=12)
    colorbar.set_label("Reduced Chi-Squared", fontsize=12)
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


def plot_scc_amp_duration(file_id):
    # fmt: off
    snr_list = [0.207, 0.206, 0.211, 0.183, 0.08, 0.224, 0.095, 0.078, 0.136, 0.038, 0.034, 0.026, 0.039, 0.165, 0.13, 0.18, 0.153, 0.074, 0.08, 0.028, 0.053, 0.142, 0.188, 0.077, 0.121, 0.137, 0.085, 0.067, 0.157, 0.135, 0.036, 0.075, 0.135, 0.168, 0.045, 0.067, 0.158, 0.12, 0.074, 0.167, 0.073, 0.046, 0.149, 0.054, 0.135, 0.064, 0.119, 0.193, 0.104, 0.091, 0.04, 0.127, 0.125, 0.105, 0.054, 0.069, 0.139, 0.151, 0.119, 0.068, 0.134, 0.054, 0.11, 0.096, 0.105, 0.133, 0.149, 0.057, 0.102, 0.083, 0.097, 0.175, 0.096, 0.058, 0.161, 0.158, 0.048, 0.1, 0.093, 0.132, 0.131, 0.055, 0.028, 0.083, 0.05, 0.061, 0.06, 0.082, 0.114, 0.065, 0.144, 0.142, 0.116, 0.095, 0.143, 0.121, 0.116, 0.102, 0.032, 0.061, 0.113, 0.087, 0.061, 0.119, 0.027, 0.119, 0.131, 0.144, 0.122, 0.087, 0.087, 0.067, 0.089, 0.068, 0.089, 0.043, 0.131, 0.05, 0.075, 0.039, 0.09, 0.085, 0.099, 0.123, 0.133, 0.097, 0.083, 0.04, 0.097, 0.032, 0.043, 0.148, 0.092, 0.037, 0.118, 0.051, 0.078, 0.053, 0.081, 0.056, 0.112, 0.119, 0.05, 0.044, 0.131, 0.137, 0.133, 0.074, 0.049, 0.06, 0.043, 0.063, 0.106, 0.165, 0.16, 0.05, 0.132, 0.088, 0.081, 0.062]
    # scc_duration_list = [304, 304, 304, 156, 304, 148, 244, 100, 304, 60, 304, 76, 88, 304, 112, 304, 144, 304, 304, 48, 76, 140, 144, 88, 304, 304, 304, 112, 304, 172, 304, 96, 72, 168, 128, 48, 304, 112, 124, 304, 48, 304, 304, 48, 304, 304, 168, 144, 304, 304, 60, 304, 108, 304, 48, 304, 164, 160, 304, 268, 240, 196, 304, 112, 304, 48, 264, 304, 152, 304, 184, 148, 304, 52, 160, 112, 104, 304, 88, 116, 56, 304, 68, 304, 304, 112, 52, 304, 304, 96, 304, 120, 304, 140, 304, 304, 156, 48, 304, 64, 304, 304, 132, 124, 304, 148, 304, 148, 80, 136, 124, 148, 108, 132, 132, 68, 124, 132, 304, 92, 80, 64, 304, 152, 136, 304, 48, 96, 304, 48, 64, 304, 64, 304, 216, 304, 304, 144, 176, 140, 304, 136, 104, 304, 56, 136, 76, 112, 304, 120, 164, 304, 88, 104, 128, 152, 132, 112, 100, 304]
    # scc_amp_list = [1.107, 1.071, 1.214, 1.179, 1.036, 1.179, 0.75, 1.214, 0.857, 0.857, 1.036, 0.821, 0.857, 1.25, 1.036, 1.179, 1.25, 0.786, 1.179, 1.036, 1.25, 1.0, 1.071, 1.25, 1.25, 0.857, 1.036, 1.071, 1.036, 1.143, 1.036, 0.75, 1.214, 0.964, 1.036, 0.75, 0.786, 0.964, 1.107, 0.857, 1.179, 0.857, 1.214, 1.143, 1.071, 1.25, 1.143, 0.857, 1.214, 1.143, 0.786, 0.929, 0.75, 1.071, 0.857, 0.75, 1.036, 1.071, 0.786, 1.107, 1.071, 1.214, 0.964, 0.929, 1.107, 1.143, 1.214, 1.071, 1.036, 1.214, 0.893, 1.071, 0.75, 0.786, 1.25, 1.107, 0.929, 0.786, 0.929, 1.25, 1.107, 1.036, 1.0, 0.893, 1.0, 0.964, 1.107, 1.143, 1.25, 1.214, 0.821, 0.929, 1.107, 1.107, 1.25, 1.214, 0.75, 1.214, 1.0, 1.25, 0.964, 0.857, 0.929, 1.25, 0.893, 1.0, 0.75, 1.179, 1.25, 1.214, 1.036, 0.821, 1.214, 1.0, 1.179, 1.214, 1.107, 1.25, 0.929, 1.036, 1.143, 0.821, 0.893, 1.179, 1.143, 0.893, 1.25, 1.071, 0.786, 1.25, 1.107, 1.179, 0.929, 1.0, 1.25, 0.964, 1.036, 1.036, 1.25, 1.179, 1.143, 1.143, 1.143, 1.179, 1.179, 1.143, 1.214, 1.107, 0.893, 1.25, 1.143, 0.964, 1.25, 1.036, 0.857, 1.107, 1.179, 1.0, 1.214, 0.786]
    # fmt: on
    scc_data = dm.get_raw_data(file_id=1725870710271)
    scc_optimal_durations = scc_data["optimal_durations"]
    scc_optimal_amplitudes = scc_data["optimal_amplitudes"]
    scc_duration_list = list(scc_optimal_durations.values())
    scc_amp_list = list(scc_optimal_amplitudes.values())
    snr_list = np.array(snr_list)
    exclude_list = np.argwhere(snr_list >= 0.07).flatten()
    optimal_durations = [scc_duration_list[i] for i in exclude_list]
    optimal_amplitudes = [scc_amp_list[i] for i in exclude_list]
    optimal_snrs = [snr_list[i] for i in exclude_list]
    num_snrs = len(optimal_snrs)
    fig = plt.figure(figsize=(6.5, 5))
    plt.scatter(
        optimal_durations,
        optimal_amplitudes,
        c=optimal_snrs,
        cmap="viridis",
        s=40,
        edgecolors="black",
    )

    # for i, (x, y) in enumerate(zip(optimal_durations, optimal_amplitudes)):
    #     optimal_snr = optimal_snrs[i]
    #     plt.annotate(f"{optimal_snr:.2f}", (x, y), fontsize=6, alpha=0.6, color="red")

    colorbar = plt.colorbar(label="Reduced Chi-Squared")
    colorbar.ax.tick_params(labelsize=12)
    colorbar.set_label("SNR", fontsize=12)
    plt.xlabel("Optimal Duration (ns)", fontsize=12)
    plt.ylabel("Optimal Amplitude", fontsize=12)
    plt.title(f"Optimal SCC Parameters Across NVs ({num_snrs}NVs)", fontsize=14)
    plt.grid(alpha=0.3)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()
    # file_name = dm.get_file_name(file_id)
    # file_path = dm.get_file_path(__file__, file_name, f"{file_id}_amp_vs_duration")
    # dm.save_figure(fig, file_path)


if __name__ == "__main__":
    kpl.init_kplotlib()
    file_id = 1728131481474
    # data = dm.get_raw_data(file_id=file_id)
    # process_and_plot(data)
    # # processed data analysis
    # processed_data_id = 1723819842491
    # processed_data_id = 1728147590280
    processed_data_id = 1729123064963
    processed_data = dm.get_raw_data(file_id=processed_data_id)
    analyze_and_visualize(processed_data)
    # plot_scc_amp_duration(file_id)
    # file_name = dm.get_file_name(file_id)
    # print(dm.get_file_name(1728131481474))
    plt.show(block=True)
