# -*- coding: utf-8 -*-
"""
Widefield Rabi experiment - Enhanced

Created on Fall, 2024
@auhtor
"""

import traceback
import warnings
import numpy as np
from datetime import datetime
from joblib import Parallel, delayed
import seaborn as sns
from scipy.optimize import curve_fit, least_squares
from utils import data_manager as dm
from utils import kplotlib as kpl
import matplotlib.pyplot as plt
from utils import widefield as widefield


def fit_rabi_data(nv_list, taus, avg_counts, avg_counts_ste, epsilon=1e-10):
    """
    Process the Rabi data to fit each NV's Rabi oscillation using a robust fitting method.
    """
    num_nvs = len(nv_list)

    # Define the cosine decay function for fitting
    def cos_decay(tau, amp, freq, decay, tau_phase, baseline):
        envelope = np.exp(-tau / abs(decay)) * amp
        cos_part = np.cos((2 * np.pi * freq * (tau - tau_phase)))
        return amp - envelope * cos_part + baseline

    # Fit a single NV
    def fit_single_nv(nv_idx):
        print(f"Fitting NV {nv_idx}...")

        num_steps = len(taus)
        tau_step = taus[1] - taus[0]

        # Compute FFT to estimate initial frequency
        transform = np.fft.rfft(avg_counts[nv_idx])
        freqs = np.fft.rfftfreq(num_steps, d=tau_step)
        transform_mag = np.abs(transform)

        # Identify the peak frequency
        max_ind = np.argmax(transform_mag[1:]) + 1  # Skip DC component
        max_ind = min(max_ind, len(freqs) - 1)  # Ensure index is within bounds
        freq_guess = freqs[max_ind] if freqs[max_ind] > epsilon else epsilon

        # Estimate other initial parameters
        angular_freq_guess = 2 * np.pi * freq_guess
        tau_phase_guess = (
            -np.angle(transform[max_ind]) / angular_freq_guess
            if angular_freq_guess > epsilon
            else 0
        )
        decay_guess = max(taus) / 2  # Arbitrary initial guess for decay time
        amp_guess = np.max(avg_counts[nv_idx] - np.mean(avg_counts[nv_idx]))
        baseline_guess = np.min(avg_counts[nv_idx])
        guess_params = [
            amp_guess,
            freq_guess,
            decay_guess,
            tau_phase_guess,
            baseline_guess,
        ]

        try:
            popt, _ = curve_fit(
                cos_decay,
                taus,
                avg_counts[nv_idx],
                p0=guess_params,
                sigma=avg_counts_ste[nv_idx] + epsilon,  # Avoid zero division
                absolute_sigma=True,
                maxfev=10000,
            )
            fit_fn = lambda tau: cos_decay(tau, *popt)  # Return function handle

            # Compute and print Rabi period
            rabi_freq = popt[1]
            rabi_period = 1 / rabi_freq if rabi_freq > epsilon else None
            if rabi_period:
                print(f"NV {nv_idx}: Rabi Period = {rabi_period:.3f} µs")
            else:
                print(f"NV {nv_idx}: Invalid Rabi frequency (f = {rabi_freq})")

        except RuntimeError as e:
            print(f"Fitting failed for NV {nv_idx}: {e}")
            fit_fn = None  # Assign None if fitting fails
            popt = [None] * 3  # Avoid breaking later processing

        return fit_fn, popt

    # Parallel execution of fitting for each NV
    results = Parallel(n_jobs=-1)(
        delayed(fit_single_nv)(nv_idx) for nv_idx in range(num_nvs)
    )

    fit_fns, popts = zip(*results)

    return fit_fns, popts


def plot_rabi_fits(
    nv_list,
    taus,
    avg_counts,
    avg_counts_ste,
    fit_fns,
    popts,
):
    """
    Plot the fitted Rabi oscillation data for each NV center separately.

    """
    num_nvs = len(nv_list)
    taus = np.array(taus)

    for nv_ind in range(num_nvs):
        fig, ax = plt.subplots()

        # Scatter plot with error bars
        ax.errorbar(
            taus,
            avg_counts[nv_ind],
            yerr=np.abs(avg_counts_ste[nv_ind]),
            fmt="o",
        )

        # Plot the fitted curve if available
        tau_dense = np.linspace(0, taus.max(), 300)
        if fit_fns[nv_ind] is not None:
            ax.plot(tau_dense, fit_fns[nv_ind](tau_dense), "-")
        rabi_freq = popts[nv_ind][1]
        if rabi_freq is not None:
            rabi_period = round((1 / rabi_freq) / 4) * 4
            title = f"NV {nv_ind} (Rabi Period: {rabi_period}ns)"
        else:
            title = f"NV {nv_ind} (Rabi Period: N/A)"
        ax.set_title(title)
        ax.set_xlabel("Pulse Duration (ns)")
        ax.set_ylabel("Norm. NV- Population")
        ax.grid(True)
        fig.tight_layout()
        # Save or show the plot
        kpl.show(block=True)
        # Print Rabi periods for each NV center
        # for i, popt in enumerate(popts):
        #     rabi_freq = popt[1]
        #     if rabi_freq > 0:
        #         rabi_period = 1 / rabi_freq
        #         print(f"NV {nv_list[i]}: Rabi Period = {rabi_period:.3f} µs")
        #     else:
        #         print(f"NV {nv_list[i]}: Invalid Rabi frequency (f = {rabi_freq})")

    # num_rows = int(np.ceil(num_nvs / num_cols))
    # sns.set(style="whitegrid")
    # fig, axes = plt.subplots(
    #     num_rows,
    #     num_cols,
    #     figsize=(num_cols * 2, num_rows * 3),
    #     sharex=True,
    #     sharey=True,
    # )
    # axes = axes.flatten()
    # colors = sns.color_palette("husl", num_nvs)
    # # taus = np.array(taus).flatten()

    # for nv_idx, ax in enumerate(axes):
    #     if nv_idx < num_nvs:
    #         # Scatter plot of the data
    #         # print(f"Length of taus: {len(taus)}")
    #         print(f"plotting NV {nv_idx}")
    #         sns.scatterplot(
    #             x=taus,
    #             y=avg_counts[nv_idx],
    #             ax=ax,
    #             color=colors[nv_idx % len(colors)],
    #             s=15,
    #             label=f"NV {nv_idx + 1}",
    #         )
    #         # Error bars
    #         ax.errorbar(
    #             taus,
    #             avg_counts[nv_idx],
    #             yerr=avg_counts_ste[nv_idx],
    #             fmt="none",
    #             ecolor="gray",
    #             alpha=0.6,
    #         )
    #         # Plot the fitted function if available
    #         if fit_fns[nv_idx] is not None:
    #             ax.plot(
    #                 taus,
    #                 fit_fns[nv_idx](taus),
    #                 "-",
    #                 color=colors[nv_idx % len(colors)],
    #                 label="Fit",
    #                 lw=2,
    #             )
    #         # ax.set_title(f"NV {nv_idx + 1}", fontsize=10)
    #         ax.grid(True, which="both", linestyle="--", linewidth=0.5)

    #         # Only show y-ticks on the leftmost column
    #         if nv_idx % num_cols == 0:
    #             ax.set_yticks(ax.get_yticks())
    #         else:
    #             ax.set_yticklabels([])

    #         # Set x-axis label for bottom row plots
    #         if nv_idx >= (num_rows - 1) * num_cols:
    #             ax.set_xlabel("Pulse Duration (ns)")
    #             ax.set_xticks(np.linspace(min(taus), max(taus), 5))

    #     else:
    #         ax.axis("off")

    # # Common y-label for the entire figure
    # fig.text(
    #     0.04,
    #     0.5,
    #     "Normalized NV$^{-}$ Population",
    #     va="center",
    #     rotation="vertical",
    #     fontsize=12,
    # )

    # # Save the figure
    # now = datetime.now()
    # date_time_str = now.strftime("%Y%m%d_%H%M%S")
    # file_name = dm.get_file_name(file_id=file_id)
    # file_path = dm.get_file_path(__file__, file_name, f"{file_id}_{date_time_str}")
    # dm.save_figure(fig, file_path)
    # plt.close(fig)


if __name__ == "__main__":
    kpl.init_kplotlib()
    # file_id = 1772297872545 # two orientations with freqs      # "frequency": 2.7801,  # shallow NVs O1
    # "rabi_period": 104, # shallow NVs O1
    file_id = 1772755741220

    data = dm.get_raw_data(file_id=file_id, load_npz=False, use_cache=False)
    nv_list = data["nv_list"]
    taus = data["taus"]
    counts = np.array(data["counts"])
    sig_counts, ref_counts = counts[0], counts[1]
    avg_counts, avg_counts_ste = widefield.process_counts(
        nv_list, sig_counts, ref_counts, threshold=True
    )

    # Call to process Rabi data without normalization (can still perform fitting)
    fit_fns, popts = fit_rabi_data(nv_list, taus, avg_counts, avg_counts_ste)

    # Plotting without normalization
    plot_rabi_fits(
        nv_list,
        taus,
        avg_counts,
        avg_counts_ste,
        fit_fns,
        popts,
    )
    kpl.show(block=True)
