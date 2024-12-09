# -*- coding: utf-8 -*-
"""
Widefield Rabi experiment - Enhanced

Created on Fall, 2024
@auhtor
"""
import traceback
import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.optimize import curve_fit, least_squares

from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import widefield as widefield


def process_rabi_data(nv_list, taus, avg_counts, avg_counts_ste, norms, epsilon=1e-10):
    """
    Process the Rabi data to fit each NV's Rabi oscillation using a robust fitting method.

    Args:
        nv_list: List of NV signatures.
        taus: Pulse durations (ns).
        avg_counts: Averaged counts for NVs.
        avg_counts_ste: Standard error of averaged counts.
        norms: Normalization data for NVs.
        epsilon: Small value to prevent division by zero.

    Returns:
        fit_fns: List of fitted functions for each NV.
        popts: List of optimized parameters for each NV fit.
        norm_counts: Normalized counts for NVs.
        norm_counts_ste: Standard error of normalized counts.
    """
    num_nvs = len(nv_list)
    fit_fns = []
    popts = []

    # Define the cosine decay function for fitting
    def cos_decay(tau, freq, decay, tau_phase):
        amp = 0.5
        envelope = np.exp(-tau / abs(decay)) * amp
        cos_part = np.cos((2 * np.pi * freq * (tau - tau_phase)))
        return amp - envelope * cos_part

    for nv_idx in range(num_nvs):
        print(f"Fitting NV {nv_idx}]")
        num_steps = len(taus)
        tau_step = taus[1] - taus[0]
        transform = np.fft.rfft(avg_counts[nv_idx])  # Use avg_counts[nv_idx]
        freqs = np.fft.rfftfreq(num_steps, d=tau_step)
        transform_mag = np.absolute(transform)
        max_ind = np.argmax(transform_mag[1:]) + 1  # Adjusted index
        max_ind = min(max_ind, len(freqs) - 1)  # Ensure index is within bounds
        freq_guess = freqs[max_ind]
        angular_freq_guess = 2 * np.pi * freq_guess
        tau_phase_guess = -np.angle(transform[max_ind]) / angular_freq_guess
        decay_guess = max(taus) / 5  # Example decay guess
        guess_params = [freq_guess, decay_guess, tau_phase_guess]

        try:
            popt, _ = curve_fit(
                cos_decay,
                taus,
                avg_counts[nv_idx],  # Use specific NV data
                p0=guess_params,
                sigma=avg_counts_ste[nv_idx],  # Use specific NV error
                absolute_sigma=True,
                maxfev=10000,
            )
            fit_fns.append(lambda tau, p=popt: cos_decay(tau, *p))
            popts.append(popt)
        except RuntimeError as e:
            print(f"Fitting failed for NV {nv_idx + 1}: {e}")
            fit_fns.append(None)
            popts.append(None)
        except Exception as e:
            print(f"Unexpected error for NV {nv_idx + 1}: {e}")
            fit_fns.append(None)
            popts.append(None)

    return fit_fns, popts


def plot_rabi_fits(
    nv_list, taus, avg_counts, avg_counts_ste, fit_fns, popts, file_id, num_cols=6
):
    """
    Plot the fitted Rabi oscillation data for each NV center alongside the scatter data.

    Args:
        nv_list: List of NV signatures.
        taus: Pulse durations (ns).
        avg_counts: Normalized counts for NVs.
        avg_counts_ste: Standard error of normalized counts.
        fit_fns: List of fitted functions for each NV.
        popts: List of optimized parameters for each NV fit.
        file_id: ID for saving the file.
        num_cols: Number of columns for subplot layout.
    """
    num_nvs = len(nv_list)
    num_rows = int(np.ceil(num_nvs / num_cols))
    sns.set(style="whitegrid")
    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=(num_cols * 2, num_rows * 3),
        sharex=True,
        sharey=True,
    )
    axes = axes.flatten()
    colors = sns.color_palette("husl", num_nvs)
    # taus = np.array(taus).flatten()
    taus = np.array(taus)

    for nv_idx, ax in enumerate(axes):
        if nv_idx < num_nvs:
            # Scatter plot of the data
            # print(f"Length of taus: {len(taus)}")
            print(f"plotting NV {nv_idx}")
            sns.scatterplot(
                x=taus,
                y=avg_counts[nv_idx],
                ax=ax,
                color=colors[nv_idx % len(colors)],
                s=15,
                label=f"NV {nv_idx + 1}",
            )
            # Error bars
            ax.errorbar(
                taus,
                avg_counts[nv_idx],
                yerr=avg_counts_ste[nv_idx],
                fmt="none",
                ecolor="gray",
                alpha=0.6,
            )
            # Plot the fitted function if available
            if fit_fns[nv_idx] is not None:
                ax.plot(
                    taus,
                    fit_fns[nv_idx](taus),
                    "-",
                    color=colors[nv_idx % len(colors)],
                    label="Fit",
                    lw=2,
                )
            # ax.set_title(f"NV {nv_idx + 1}", fontsize=10)
            ax.grid(True, which="both", linestyle="--", linewidth=0.5)

            # Only show y-ticks on the leftmost column
            if nv_idx % num_cols == 0:
                ax.set_yticks(ax.get_yticks())
            else:
                ax.set_yticklabels([])

            # Set x-axis label for bottom row plots
            if nv_idx >= (num_rows - 1) * num_cols:
                ax.set_xlabel("Pulse Duration (ns)")
                ax.set_xticks(np.linspace(min(taus), max(taus), 5))

        else:
            ax.axis("off")

    # Common y-label for the entire figure
    fig.text(
        0.04,
        0.5,
        "Normalized NV$^{-}$ Population",
        va="center",
        rotation="vertical",
        fontsize=12,
    )

    # Save the figure
    now = datetime.now()
    date_time_str = now.strftime("%Y%m%d_%H%M%S")
    file_name = dm.get_file_name(file_id=file_id)
    file_path = dm.get_file_path(__file__, file_name, f"{file_id}_{date_time_str}")
    dm.save_figure(fig, file_path)
    plt.close(fig)


if __name__ == "__main__":
    kpl.init_kplotlib()
    # file_id = 1566322671967
    file_id = 1697111534197
    # file_id = 1654509086036

    data = dm.get_raw_data(file_id=file_id, load_npz=True, use_cache=False)
    nv_list = data["nv_list"]
    taus = data["taus"]
    counts = np.array(data["states"])
    sig_counts, ref_counts = counts[0], counts[1]
    avg_counts, avg_counts_ste, norms = widefield.process_counts(
        nv_list, sig_counts, ref_counts, threshold=True
    )

    # Call to process Rabi data without normalization (can still perform fitting)
    fit_fns, popts = process_rabi_data(nv_list, taus, avg_counts, avg_counts_ste, norms)

    # Plotting without normalization
    plot_rabi_fits(
        nv_list, taus, avg_counts, avg_counts_ste, fit_fns, popts, file_id, num_cols=5
    )
