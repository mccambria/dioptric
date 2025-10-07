# -*- coding: utf-8 -*-
"""
Widefield Rabi experiment - Enhanced

Created on Fall, 2024
@auhtor : Saroj Chand
"""

import sys
import traceback
import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from joblib import Parallel, delayed
from numpy.linalg import lstsq
from scipy.optimize import curve_fit, least_squares
from scipy.stats import pearsonr

from utils import _cloud_box as box_cloud
from utils import data_manager as dm
from utils import kplotlib as kpl
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
                maxfev=20000,
                # bounds=bounds,
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

    def fit_median_trace():
        print("Fitting median NV trace...")
        median_trace = np.median(avg_counts, axis=0)
        median_ste = np.sqrt(np.sum(avg_counts_ste**2, axis=0)) / len(nv_list)
        # Estimate frequency via FFT
        num_steps = len(taus)
        tau_step = taus[1] - taus[0]
        transform = np.fft.rfft(median_trace)
        freqs = np.fft.rfftfreq(num_steps, d=tau_step)
        transform_mag = np.abs(transform)
        max_ind = np.argmax(transform_mag[1:]) + 1
        max_ind = min(max_ind, len(freqs) - 1)
        freq_guess = freqs[max_ind] if freqs[max_ind] > epsilon else epsilon

        angular_freq_guess = 2 * np.pi * freq_guess
        tau_phase_guess = (
            -np.angle(transform[max_ind]) / angular_freq_guess
            if angular_freq_guess > epsilon
            else 0
        )
        decay_guess = max(taus) / 2
        amp_guess = np.max(median_trace - np.mean(median_trace))
        baseline_guess = np.min(median_trace)
        guess_params = [
            amp_guess,
            freq_guess,
            decay_guess,
            tau_phase_guess,
            baseline_guess,
        ]

        try:
            median_popt, _ = curve_fit(
                cos_decay,
                taus,
                median_trace,
                p0=guess_params,
                sigma=median_ste + epsilon,
                absolute_sigma=True,
                maxfev=20000,
            )
            median_fit_fn = lambda tau: cos_decay(tau, *median_popt)
            print(f"Median Rabi Period = {1 / median_popt[1]:.3f} µs")
        except RuntimeError as e:
            print(f"Median fitting failed: {e}")
            median_fit_fn = None
            median_popt = [None] * 5

        return median_fit_fn, median_popt

    median_fit_fn, median_popt = fit_median_trace()

    return fit_fns, popts, median_fit_fn, median_popt


import math


def plot_rabi_fits(
    nv_list,
    taus,
    avg_counts,
    avg_counts_ste,
    fit_fns,
    popts,
    median_fit_fn=None,
    median_popt=None,
    num_cols=9,
    period_bin_width=8,  # ns, choose a multiple of 4 if you want
    period_round_to=4,  # ns
    period_keep_range=(100, 200),  # ns, set to None to disable range filter
):
    """
    Plot fitted Rabi oscillations for each NV and summarize Rabi periods.

    Assumes popts[nv] has popts[nv][1] = rabi_freq in 1/ns (or Hz if taus in s).
    """

    taus = np.asarray(taus, dtype=float)
    num_nvs = len(nv_list)

    epsilon = 1e-10
    rabi_periods = []
    amps = []
    kept_indices = []

    # --- Collect period/amp per NV (with guards) ---
    for nv_ind in range(num_nvs):
        try:
            popt = popts[nv_ind]
            amp = float(popt[0])
            rabi_freq = abs(float(popt[1]))
        except (TypeError, ValueError, IndexError):
            continue

        if not np.isfinite(rabi_freq) or rabi_freq <= epsilon:
            continue

        rabi_period = 1.0 / rabi_freq  # units consistent with taus
        if not np.isfinite(rabi_period) or rabi_period <= 0:
            continue

        # round to nearest multiple (optional aesthetic)
        if period_round_to is not None and period_round_to > 0:
            rabi_period = int(np.round(rabi_period / period_round_to)) * period_round_to

        rabi_periods.append(rabi_period)
        amps.append(float(amp))
        kept_indices.append(nv_ind)

    print(f"Raw Rabi Periods (ns): {rabi_periods}")

    if len(rabi_periods) == 0:
        print("No valid Rabi periods; skipping histogram.")
    else:
        # --- Paired outlier removal via IQR on periods ---
        p = np.asarray(rabi_periods, dtype=float)
        a = np.asarray(amps, dtype=float)
        idx = np.asarray(kept_indices, dtype=int)

        Q1, Q3 = np.percentile(p, [25, 75])
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        print(f"Rabi Period IQR filter: {Q1=}, {Q3=}, {IQR=}, {lower=}, {upper=}")
        mask = (p >= lower) & (p <= upper)

        # Optional range filter (e.g., 100–200 ns)
        if period_keep_range is not None:
            lo, hi = period_keep_range
            mask &= (p >= lo) & (p <= hi)

        p_f = p[mask]
        a_f = a[mask]
        
        idx_f = idx[mask]

        print(f"Filtered Rabi Periods (ns): {p_f.tolist()}")
        print(f"Filtered idx: {idx_f.tolist()}")
        print(f"len before/after filtering: {len(p)}/{len(p_f)}")
        if p_f.size > 0:
            print(f"Median Rabi Period (ns): {float(np.median(p_f)):.3f}")
        else:
            print("No periods remain after filtering.")

        # --- Histogram ---
        if p_f.size > 0:
            nbins = max(
                5, int(np.ceil((p_f.max() - p_f.min()) / max(1, period_bin_width)))
            )
            fig_h, ax_h = plt.subplots(figsize=(6, 5))
            ax_h.hist(p_f, bins=nbins)
            ax_h.set_title("Rabi Periods", fontsize=15)
            ax_h.set_xlabel("Rabi Period (ns)", fontsize=15)
            ax_h.set_ylabel("Number of Occurrence", fontsize=15)
            ax_h.tick_params(axis="both", labelsize=12)
            ax_h.grid(True)
            plt.show(block=True)
    return
    # --- Grid of per-NV plots ---
    if num_cols is None or num_cols < 1:
        num_cols = 9
    num_rows = math.ceil(num_nvs / num_cols)

    # If many NVs, consider smaller figsize per panel
    fig_g, axes = plt.subplots(
        num_rows, num_cols, figsize=(3.6 * num_cols, 2.8 * num_rows), squeeze=False
    )
    tau_dense = np.linspace(0, float(taus.max()), 300)

    for nv_ind in range(num_nvs):
        r = nv_ind // num_cols
        c = nv_ind % num_cols
        ax = axes[r, c]

        # data + errors
        y = np.asarray(avg_counts[nv_ind], dtype=float)
        yerr = np.asarray(avg_counts_ste[nv_ind], dtype=float)
        ax.errorbar(taus, y, yerr=np.abs(yerr), fmt="o", ms=3)

        # fit curve
        if fit_fns[nv_ind] is not None:
            try:
                ax.plot(tau_dense, fit_fns[nv_ind](tau_dense), "-")
            except Exception:
                pass

        # title with period if valid
        period_str = "N/A"
        try:
            rf = float(popts[nv_ind][1])
            if np.isfinite(rf) and rf > epsilon:
                rp = 1.0 / rf
                if (
                    period_round_to is not None
                    and period_round_to > 0
                    and np.isfinite(rp)
                    and rp > 0
                ):
                    rp = int(np.round(rp / period_round_to)) * period_round_to
                period_str = f"{rp:.0f} ns"
        except Exception:
            pass

        ax.set_title(f"NV {nv_ind} (Rabi: {period_str})", fontsize=10)
        ax.set_xlabel("Pulse Duration (ns)", fontsize=9)
        ax.set_ylabel("Norm. NV- Population", fontsize=9)
        ax.grid(True, alpha=0.4)

        # Optional: median fit overlay if provided
        if callable(median_fit_fn) and (median_popt is not None):
            try:
                ax.plot(tau_dense, median_fit_fn(tau_dense, *median_popt), "--")
            except Exception:
                pass

    # Hide any empty subplots
    total_panels = num_rows * num_cols
    for k in range(num_nvs, total_panels):
        r = k // num_cols
        c = k % num_cols
        axes[r, c].axis("off")

    fig_g.tight_layout()
    plt.show(block=True)
    # plotting median acroos all NVs

    fig, ax = plt.subplots()
    # Scatter plot with error bars
    median_counts = np.median(avg_counts, axis=0)
    median_counts_ste = np.median(avg_counts_ste, axis=0)
    # median_counts_ste = np.sqrt(np.sum(avg_counts_ste**2, axis=0)) / len(nv_list)

    ax.errorbar(
        taus,
        median_counts,
        yerr=np.abs(median_counts_ste),
        fmt="o",
    )
    # Plot the fitted curve if available
    tau_dense = np.linspace(0, taus.max(), 300)
    if fit_fns[nv_ind] is not None:
        # fit_fns_median = median_fit_fn(tau_dense)
        ax.plot(tau_dense, median_fit_fn(tau_dense), "-")
    title = f"Median across {len(nv_list)} NVs"
    ax.set_title(title)
    ax.set_xlabel("Pulse Duration (ns)")
    ax.set_ylabel("Norm. NV- Population")
    ax.grid(True)
    fig.tight_layout()
    plt.show(block=True)

    # Save or show the plot
    # Print Rabi periods for each NV center
    # for i, popt in enumerate(popts):
    #     rabi_freq = popt[1]
    #     if rabi_freq > 0:
    #         rabi_period = 1 / rabi_freq
    #         print(f"NV {nv_list[i]}: Rabi Period = {rabi_period:.3f} µs")
    #     else:
    #         print(f"NV {nv_list[i]}: Invalid Rabi frequency (f = {rabi_freq})")

    ### Plotiitg all NVs
    # num_rows = int(np.ceil(num_nvs / num_cols))
    # sns.set(style="whitegrid", palette="muted")
    # fig, axes = plt.subplots(
    #     num_rows,
    #     num_cols,
    #     figsize=(num_cols * 3, num_rows * 3),
    #     sharex=True,
    #     sharey=True,
    # )
    # axes = axes.flatten()
    # colors = sns.color_palette("deep", num_nvs)
    # # taus = np.array(taus).flatten()
    # for nv_idx, ax in enumerate(axes):
    #     if nv_idx < num_nvs:
    #         sns.lineplot(
    #             x=taus,
    #             y=avg_counts[nv_idx],
    #             ax=ax,
    #             color=colors[nv_idx % len(colors)],
    #             lw=0,
    #             marker="o",
    #             markersize=2,
    #             label=f"{nv_idx}",
    #         )
    #         ax.legend(fontsize="xx-small")
    #         # Error bars
    #         ax.errorbar(
    #             taus,
    #             avg_counts[nv_idx],
    #             # yerr=filtered_avg_counts_ste[nv_idx],
    #             yerr=np.abs(avg_counts_ste[nv_idx]),
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
    #                 lw=1,
    #             )
    #         ax.set_yticklabels([])
    #         ax.set_xlim([0, max(taus)])
    #         ax.set_ylim(
    #             [np.min(avg_counts[nv_idx]) - 1.0, np.max(avg_counts[nv_idx]) + 1.0]
    #         )
    #         fig.text(
    #             0.0,
    #             0.5,
    #             "NV$^{-}$ Population",
    #             va="center",
    #             rotation="vertical",
    #             fontsize=12,
    #         )
    #         # Set custom tick locations in x axis
    #         if nv_idx >= (num_rows - 1) * num_cols:  # Bottom row
    #             ax.set_xlabel("Time (ns)")
    #             ax.set_xticks(np.linspace(min(taus), max(taus), 6))
    #         for col in range(num_cols):
    #             bottom_row_idx = num_rows * num_cols - num_cols + col
    #             if bottom_row_idx < len(axes):
    #                 ax = axes[bottom_row_idx]
    #                 tick_positions = np.linspace(min(taus), max(taus), 5)
    #                 ax.set_xticks(tick_positions)
    #                 ax.set_xticklabels(
    #                     [f"{tick:.2f}" for tick in tick_positions],
    #                     rotation=45,
    #                     fontsize=9,
    #                 )
    #                 ax.set_xlabel("Time (ns)")
    #             else:
    #                 ax.set_xticklabels([])
    #         ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    #     else:
    #         ax.axis("off")

    # # Save the figure
    # now = datetime.now()
    # date_time_str = now.strftime("%Y%m%d_%H%M%S")
    # file_name = dm.get_file_name(file_id=file_id)
    # print(f"{file_name}_{file_id}__{date_time_str}")
    # file_path = dm.get_file_path(__file__, file_name, f"{file_id}_{date_time_str}")
    # kpl.show()
    # dm.save_figure(fig, file_path)
    # plt.close(fig)


def remove_outliers(data):
    data = np.array(data)
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - IQR
    upper_bound = Q3 + 1.5 * IQR
    print(f"({Q1, Q3, IQR, lower_bound, upper_bound})")
    return (data >= lower_bound) & (data <= upper_bound)


if __name__ == "__main__":
    kpl.init_kplotlib()
    # file_id = 1772297872545  # two orientations with freqsa round 2.79
    # file_id = 1772755741220  # two orientations with freqs aroud 2.84
    # file_id = 1774582403511  # all four orientation measured with two frequency tone per sig gen
    # file_id = 1775776922337  # all four orientation measured with two frequency tone per sig gen with offset pulses both microwaave ()
    # rubin
    # file_id = 1775776922337  # all four orientation measured with two frequency tone per sig gen with offset pulses both microwaave ()
    # file_id = 1779670263899
    # rubin sample
    # file_id = 1795718888560
    # file_id = 1796958071866
    # file_id = 1795718888560
    # file_id = 1796958071866
    # 300 NVs
    # file_id = 1803593992080
    # file_id = 1804466558303
    # file_id = 1817818887926  # 75NVs iq modulation test 68Mhz orientation

    # After changing magnet postion
    # file_id = 1833635613442  # i channel
    # 75NVs iq modulation both degenerate orientation
    # file_id = 1832587019842  # q channel
    # file_id = 1842383067959  # i channel
    # file_stem = box_cloud.get_file_stem_from_file_id(file_id)
    # data = dm.get_raw_data(file_id=file_id, load_npz=False, use_cache=False)
    # file_stem = "2025_04_30-07_09_33-rubin-nv0_2025_02_26"
    # file_stem = "2025_09_21-04_35_06-rubin-nv0_2025_09_08"
    # file_stem = "2025_10_02-05_57_27-rubin-nv0_2025_09_08"
    # file_stem = ["2025_10_05-20_06_59-rubin-nv0_2025_09_08"]
    file_stem = ["2025_10_06-03_26_08-rubin-nv0_2025_09_08"]
    data = dm.get_raw_data(file_stem=file_stem, load_npz=True, use_cache=False)
    nv_list = data["nv_list"]
    taus = data["taus"]
    counts = np.array(data["counts"])
    sig_counts, ref_counts = counts[0], counts[1]
    avg_counts, avg_counts_ste = widefield.process_counts(
        nv_list, sig_counts, ref_counts, threshold=True
    )
    # file_name = dm.get_file_name(file_id=file_id)
    # print(f"{file_name}_{file_id}")
    # Call to process Rabi data without normalization (can still perform fitting)
    fit_fns, popts, median_fit_fn, median_popt = fit_rabi_data(
        nv_list, taus, avg_counts, avg_counts_ste
    )
    # Plotting without normalization
    plot_rabi_fits(
        nv_list,
        taus,
        avg_counts,
        avg_counts_ste,
        fit_fns,
        popts,
        median_fit_fn,
        median_popt,
    )

kpl.show(block=True)
