# -*- coding: utf-8 -*-
"""
Widefield Rabi experiment - Refactored

Created on November 29th, 2023

"""

import time
import traceback
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from majorroutines.widefield import base_routine
from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import widefield as widefield
from utils.constants import NVSig
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
from utils import kplotlib as kpl

import warnings
from scipy.optimize import OptimizeWarning
# Fitting function for Rabi data
# def cos_decay(tau, freq, decay, tau_phase):
#     amp = 0.5
#     envelope = np.exp(-tau / abs(decay)) * amp
#     cos_part = np.cos(2 * np.pi * freq * (tau - tau_phase))
#     return amp - (envelope * cos_part)
def cos_decay(tau, freq, decay, tau_phase):
    tau = np.array(tau)  # Convert tau to a NumPy array if it isn't already
    amp = 0.5
    envelope = np.exp(-tau / abs(decay)) * amp
    cos_part = np.cos(2 * np.pi * freq * (tau - tau_phase))
    return amp - (envelope * cos_part)


def process_rabi_data(nv_list, taus, counts, counts_ste, norms):
    """
    Process and fit the Rabi experiment data for each NV.

    Args:
        nv_list: List of NV signatures.
        taus: Pulse durations (ns).
        counts: Measured counts for NVs.
        counts_ste: Standard error of counts.
        norms: Normalization data for NVs.

    Returns:
        fit_fns: List of fitted functions for each NV.
        popts: List of optimized parameters for each NV fit.
        norm_counts: Normalized counts for NVs.
        norm_counts_ste: Standard error for normalized counts.
    """
    taus = np.array(taus)
    num_nvs = len(nv_list)
    tau_step = taus[1] - taus[0]
    num_steps = len(taus)

    # Normalize counts
    norms_ms0_newaxis = norms[0][:, np.newaxis]
    norms_ms1_newaxis = norms[1][:, np.newaxis]
    contrast = norms_ms1_newaxis - norms_ms0_newaxis
    norm_counts = (counts - norms_ms0_newaxis) / contrast
    norm_counts_ste = counts_ste / contrast
    print("Contrast values:", contrast)
    print("Norm counts:", norm_counts)
    print("Norm counts ste:", norm_counts_ste)

    fit_fns = []
    popts = []

    for nv_ind in range(num_nvs):
        nv_counts = norm_counts[nv_ind]
        nv_counts_ste = norm_counts_ste[nv_ind]

        transform = np.fft.rfft(nv_counts)
        freqs = np.fft.rfftfreq(num_steps, d=tau_step)
        transform_mag = np.absolute(transform)
        max_ind = np.argmax(transform_mag[1:])  # Exclude DC component
        freq_guess = freqs[max_ind + 1]
        tau_phase_guess = -np.angle(transform[max_ind + 1]) / (2 * np.pi * freq_guess)
        guess_params = [freq_guess, 1000, tau_phase_guess]

        try:
            popt, _ = curve_fit(
                cos_decay,
                taus,
                nv_counts,
                p0=guess_params,
                sigma=nv_counts_ste,
                absolute_sigma=True,
            )
        except Exception:
            popt = None

        fit_fns.append(cos_decay)
        popts.append(popt)

    return fit_fns, popts, norm_counts, norm_counts_ste


def plot_rabi_fit(nv_list, taus, norm_counts, norm_counts_ste, fit_fns, popts):
    """
    Plot the fitted Rabi data for NVs.

    Args:
        nv_list: List of NV signatures.
        taus: Pulse durations (ns).
        norm_counts: Normalized counts for NVs.
        norm_counts_ste: Standard error of normalized counts.
        fit_fns: List of fitted functions for each NV.
        popts: List of optimized parameters for each NV fit.
    """
    num_nvs = len(nv_list)
    layout = kpl.calc_mosaic_layout(num_nvs, num_rows=2)
    fig, axes_pack = plt.subplot_mosaic(layout, figsize=[6.5, 5.0], sharex=True, sharey=True)
    axes_pack_flat = list(axes_pack.values())

    widefield.plot_fit(
        axes_pack_flat,
        nv_list,
        taus,
        norm_counts,
        norm_counts_ste,
        fit_fns,
        popts,
        xlim=[0, None],
        no_legend=True,
    )

    ax = axes_pack[layout[-1][0]]
    kpl.set_shared_ax_xlabel(ax, "Pulse duration (ns)")
    kpl.set_shared_ax_ylabel(ax, "Norm. NV$^{-}$ population")

    plt.tight_layout()
    plt.show()


def create_raw_rabi_plot(nv_list, taus, counts, counts_ste):
    """
    Plot the raw Rabi data.

    Args:
        nv_list: List of NV signatures.
        taus: Pulse durations (ns).
        counts: Measured counts for NVs.
        counts_ste: Standard error of counts.
    """
    fig, ax = plt.subplots()
    widefield.plot_raw_data(ax, nv_list, taus, counts, counts_ste)
    ax.set_xlabel("Pulse duration (ns)")
    ax.set_ylabel("Fraction in NV$^{-}$")
    plt.show()

import seaborn as sns

def plot_rabi_fit_seaborn(nv_list, taus, norm_counts, norm_counts_ste, fit_fns, popts, file_path, num_cols=5):
    """
    Plot the fitted Rabi data for NVs using Seaborn in a grid layout for better visualization of many NVs.

    Args:
        nv_list: List of NV signatures.
        taus: Pulse durations (ns).
        norm_counts: Normalized counts for NVs.
        norm_counts_ste: Standard error of normalized counts.
        fit_fns: List of fitted functions for each NV.
        popts: List of optimized parameters for each NV fit.
        file_path: Path where the figure will be saved.
        num_cols: Number of columns for the grid layout.
    """
    num_nvs = len(nv_list)
    num_rows = int(np.ceil(num_nvs / num_cols))  # Calculate the number of rows needed

    # Set up the Seaborn style and palette
    sns.set(style="whitegrid")
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 2, num_rows * 3), sharex=True, sharey=True, constrained_layout=False)
    axes = axes.flatten()  # Flatten the grid for easy access

    # Color palette for different NVs
    colors = sns.color_palette("husl", num_nvs)

    # Ensure no negative values in yerr
    norm_counts_ste = np.abs(norm_counts_ste)

    for nv_idx, ax in enumerate(axes):
        if nv_idx < num_nvs:
            # Use Seaborn's dot plot for visualizing data points
            sns.scatterplot(
                x=taus,
                y=norm_counts[nv_idx],
                ax=ax,
                color=colors[nv_idx % len(colors)],
                s=15,  # Size of the dots
                label=f"NV {nv_idx+1}"
            )
            ax.legend(fontsize=8)
            # if popts[nv_idx] is not None:
            #     # Fit line using standard matplotlib plot
            #     ax.plot(taus, fit_fns[nv_idx](taus, *popts[nv_idx]), color="r", label="Fit")
            #     # Calculate and print the Rabi period
            #     rabi_freq = popts[nv_idx][0]  # The first parameter should be the frequency
            #     rabi_period = 1 / rabi_freq
            #     print(f"NV {nv_idx + 1} Rabi period: {rabi_period:.2f} ns")

            # # Set the title for each NV
            # ax.set_title(f"NV {nv_idx + 1}", fontsize=10)

            # Dynamically adjust y-axis limits for better view of Rabi oscillations
            y_min = min(norm_counts[nv_idx])   # Small buffer below the min value
            y_max = max(norm_counts[nv_idx])  # Small buffer above the max value
            # ax.set_ylim([y_min, y_max])
            if np.isfinite(y_min) and np.isfinite(y_max):
                ax.set_ylim([y_min, y_max])

        else:
            # Hide unused subplots if the number of NVs is less than the grid size
            ax.axis("off")

    # Set common labels for the entire figure
    fig.text(0.5, 0.04, "Pulse duration (ns)", ha="center", va="center")
    fig.text(0.04, 0.5, "Norm. NV$^{-}$ Pop.", ha="center", va="center", rotation="vertical")

    # Adjust layout to ensure nothing overlaps and reduce vertical gaps
    plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.01, hspace=0.01, wspace=0.01)

    # Save the figure to the specified file path
    plt.savefig(file_path, bbox_inches="tight")

    # Close the figure to free up memory
    plt.close(fig)

import numpy as np
from scipy.optimize import curve_fit

def filter_rabi_periods(popts, period_range=(95, 105)):
    """
    Filter NVs that have a Rabi period within a given range and print the NV indices and the average Rabi period.

    Args:
        popts: List of optimized parameters for each NV fit.
        period_range: Tuple specifying the lower and upper bounds for the Rabi period.

    Returns:
        filtered_nv_indices: List of indices of NVs with Rabi periods within the specified range.
        average_filtered_period: Average Rabi period of the filtered NVs.
    """
    filtered_nv_indices = []
    filtered_periods = []

    for idx, popt in enumerate(popts):
        if popt is not None:
            rabi_period = 1 / popt[0]  # Rabi period is the inverse of the frequency
            if period_range[0] <= rabi_period <= period_range[1]:
                filtered_nv_indices.append(idx)
                filtered_periods.append(rabi_period)

    # Calculate the average Rabi period for the filtered NVs
    if filtered_periods:
        average_filtered_period = np.mean(filtered_periods)
    else:
        average_filtered_period = None

    return filtered_nv_indices, average_filtered_period


# def process_rabi_data(nv_list, taus, avg_counts, avg_counts_ste, norms):
#     """
#     Process the Rabi data to fit each NV's Rabi oscillation, and filter NVs with Rabi periods within a specific range.

#     Args:
#         nv_list: List of NV signatures.
#         taus: Pulse durations (ns).
#         avg_counts: Averaged counts for NVs.
#         avg_counts_ste: Standard error of averaged counts.
#         norms: Normalization data for NVs.

#     Returns:
#         fit_fns: List of fitted functions for each NV.
#         popts: List of optimized parameters for each NV fit.
#         norm_counts: Normalized counts for NVs.
#         norm_counts_ste: Standard error of normalized counts.
#     """
#     num_nvs = len(nv_list)
#     fit_fns = []
#     popts = []
#     norm_counts = []
#     norm_counts_ste = []

#     for nv_idx in range(num_nvs):
#         # Normalized counts and standard error
#         norm_count = (avg_counts[nv_idx] - norms[0][nv_idx]) / (norms[1][nv_idx] - norms[0][nv_idx])
#         norm_count_ste = avg_counts_ste[nv_idx] / (norms[1][nv_idx] - norms[0][nv_idx])
#         norm_counts.append(norm_count)
#         norm_counts_ste.append(norm_count_ste)

#         # Define the cosine decay function
#         def cos_decay(tau, freq, decay, tau_phase):
#             amp = 0.5
#             envelope = np.exp(-tau / abs(decay)) * amp
#             cos_part = np.cos(2 * np.pi * freq * (tau - tau_phase))
#             return amp - envelope * cos_part

#         # Initial guess for fitting: frequency, decay time, and phase offset
#         guess_params = [0.005, 1000, 0]  # Example values

#         try:
#             popt, _ = curve_fit(cos_decay, taus, norm_count, p0=guess_params, sigma=norm_count_ste, absolute_sigma=True)
#         except Exception as e:
#             popt = None

#         fit_fns.append(cos_decay if popt is not None else None)
#         popts.append(popt)

#     # Filter NVs with Rabi periods within 100 ns ± 10 ns
#     filtered_nv_indices, avg_filtered_period = filter_rabi_periods(popts, period_range=(90, 110))

#     # Print the filtered NV indices and their average Rabi period
#     print("Filtered NV indices with Rabi period 100 ± 10 ns:", filtered_nv_indices)
#     if avg_filtered_period:
#         print(f"Average Rabi period of filtered NVs: {avg_filtered_period:.2f} ns")
#     else:
#         print("No NVs found within the specified Rabi period range.")

#     return fit_fns, popts, norm_counts, norm_counts_ste


# if __name__ == "__main__":
#     # Load raw data
#     data = dm.get_raw_data(file_id=1652952342615, load_npz=True, use_cache=False)
#     nv_list = data["nv_list"]
#     taus = data["taus"]
#     counts = np.array(data["states"])

#     sig_counts = counts[0]
#     ref_counts = counts[1]

#     # Process Rabi data for NVs
#     avg_counts, avg_counts_ste, norms = widefield.process_counts(nv_list, sig_counts, ref_counts, threshold=False)
#     fit_fns, popts, norm_counts, norm_counts_ste = process_rabi_data(nv_list, taus, avg_counts, avg_counts_ste, norms)


if __name__ == "__main__":
    kpl.init_kplotlib()

    # Suppress warnings for covariance estimation issues
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=OptimizeWarning)
        # data = dm.get_raw_data(file_id=1652952342615, load_npz=True, use_cache=False)
        # data = dm.get_raw_data(file_id=1697111534197, load_npz=True, use_cache=False)
        data = dm.get_raw_data(file_id=1698301105306, load_npz=True, use_cache=False)
        
        nv_list = data["nv_list"]
        counts = np.array(data["states"])
        taus = data["taus"]

        sig_counts = counts[0]
        ref_counts = counts[1]

        avg_counts, avg_counts_ste, norms = widefield.process_counts(nv_list, sig_counts, ref_counts, threshold=True)

        # Process the Rabi data
        fit_fns, popts, norm_counts, norm_counts_ste = process_rabi_data(nv_list, taus, avg_counts, avg_counts_ste, norms)

        file_path = "nv_rabi_plot.png"
        # Plot using seaborn
        plot_rabi_fit_seaborn(nv_list, taus, norm_counts, norm_counts_ste, fit_fns, popts, file_path, num_cols=6)
