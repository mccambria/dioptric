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

import numpy as np
from numpy.linalg import lstsq
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed
from scipy.stats import pearsonr
from scipy.optimize import curve_fit, least_squares

from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import widefield as widefield
from utils import _cloud_box as box_cloud


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
            print(f"Median Rabi Period = {1/median_popt[1]:.3f} µs")
        except RuntimeError as e:
            print(f"Median fitting failed: {e}")
            median_fit_fn = None
            median_popt = [None] * 5

        return median_fit_fn, median_popt

    median_fit_fn, median_popt = fit_median_trace()

    return fit_fns, popts, median_fit_fn, median_popt


def plot_rabi_fits(
    nv_list,
    taus,
    avg_counts,
    avg_counts_ste,
    fit_fns,
    popts,
    median_fit_fn,
    median_popt,
    num_cols=9,
):
    """
    Plot the fitted Rabi oscillation data for each NV center separately.

    """
    num_nvs = len(nv_list)
    taus = np.array(taus)
    # scatter rabi period
    epsilon = 1e-10
    rabi_periods = []
    amps = []
    for nv_ind in range(num_nvs):
        popt = popts[nv_ind]
        amp = popt[0]  # FIXED: Use actual amplitude
        rabi_freq = popt[1]

        if rabi_freq > epsilon:
            rabi_period = 1 / rabi_freq
            rabi_period = round(rabi_period / 4) * 4  # Keep nearest multiple of 4
            rabi_periods.append(rabi_period)
            amps.append(amp)

    # print(f"Raw Rabi Periods: {rabi_periods}")

    # Remove outliers using IQR method
    def remove_outliers(data):
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return [val for val in data if lower_bound <= val <= upper_bound]

    filtered_rabi_periods = remove_outliers(rabi_periods)
    filtered_amps = remove_outliers(amps)

    print(f"Filtered Rabi Periods: {filtered_rabi_periods}")
    print(f"Median Rabi Periods: {np.median(filtered_rabi_periods)}")
    # print(f"Filtered Amplitudes: {filtered_amps}")
    # Ensure lists remain the same length after filtering
    filtered_data = [
        (rabi, amp)
        for rabi, amp in zip(rabi_periods, amps)
        if rabi in filtered_rabi_periods and amp in filtered_amps
    ]
    filtered_rabi_periods, filtered_amps = (
        zip(*filtered_data) if filtered_data else ([], [])
    )

    # histogam Plotting
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.hist(filtered_rabi_periods, bins=10)
    ax.set_title("Median Rabi Periods (I-Channel)", fontsize=15)
    ax.set_xlabel("Rabi Period (ns)", fontsize=15)
    ax.set_ylabel("Number of Occurance", fontsize=15)
    ax.tick_params(axis="both", labelsize=14)
    ax.grid(True)

    # plt.show()
    # for nv_ind in range(num_nvs):
    #     fig, ax = plt.subplots()
    #     # Scatter plot with error bars
    #     ax.errorbar(
    #         taus,
    #         avg_counts[nv_ind],
    #         yerr=np.abs(avg_counts_ste[nv_ind]),
    #         fmt="o",
    #     )
    #     # Plot the fitted curve if available
    #     tau_dense = np.linspace(0, taus.max(), 300)
    #     if fit_fns[nv_ind] is not None:
    #         ax.plot(tau_dense, fit_fns[nv_ind](tau_dense), "-")
    #     rabi_freq = popts[nv_ind][1]
    #     if rabi_freq is not None:
    #         rabi_period = round((1 / rabi_freq) / 4) * 4
    #         title = f"NV {nv_ind} (Rabi Period: {rabi_period}ns)"
    #     else:
    #         title = f"NV {nv_ind} (Rabi Period: N/A)"
    #     ax.set_title(title)
    #     ax.set_xlabel("Pulse Duration (ns)")
    #     ax.set_ylabel("Norm. NV- Population")
    #     ax.grid(True)
    #     fig.tight_layout()
    #     plt.show(block=True)

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
    file_stem = "2025_04_30-07_09_33-rubin-nv0_2025_02_26"
    data = dm.get_raw_data(file_stem=file_stem, load_npz=False, use_cache=False)
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
sys.exit()


# plot rabi periods for i and q channels
file_ids = [1832587019842, 1842383067959]  # [Q, I]
# file_ids = [1832587019842]  # [Q, I]
rabi_periods_all = []
pi_pulses_all = []
pi_half_pulses_all = []
# Loop over both channels
for file_id in file_ids:
    file_stem = box_cloud.get_file_stem_from_file_id(file_id)
    # file_stem = "2025_04_30-07_09_33-rubin-nv0_2025_02_26"
    data = dm.get_raw_data(file_stem=file_stem, load_npz=False, use_cache=False)
    nv_list = data["nv_list"]
    taus = data["taus"]
    counts = np.array(data["counts"])
    sig_counts, ref_counts = counts[0], counts[1]

    avg_counts, avg_counts_ste = widefield.process_counts(
        nv_list, sig_counts, ref_counts, threshold=True
    )

    # Fit the Rabi data
    fit_fns, popts, median_fit_fn, median_popt = fit_rabi_data(
        nv_list, taus, avg_counts, avg_counts_ste
    )

    # Compute Rabi periods from frequency: T = 1 / f
    # rabi_periods = np.array(
    #     [1.0 / p[1] if p is not None and p[1] > 0 else np.nan for p in popts]
    # )
    # rabi_periods_all.append(rabi_periods)

    rabi_periods = []
    pi_pulses = []
    pi_half_pulses = []

    for p in popts:
        # if p is not None and p[1] > 0:
        freq = p[1]
        tau_phase = p[3]
        rabi_period = 1.0 / freq
        rabi_periods.append(rabi_period)
        pi_pulses.append(tau_phase + 0.5 * rabi_period)
        pi_half_pulses.append(tau_phase + 0.25 * rabi_period)
        # else:
        # rabi_periods.append(np.nan)
        # pi_pulses.append(np.nan)
        # pi_half_pulses.append(np.nan)
    print("median rabi period", np.median(rabi_periods))
    print("median pi pulse", np.median(pi_pulses))
    print("median pi/2 pulse", np.median(pi_half_pulses))
    rabi_periods_all.append(np.array(rabi_periods))
    pi_pulses_all.append(np.array(pi_pulses))
    pi_half_pulses_all.append(np.array(pi_half_pulses))


# Extract Q and I
rabi_q, rabi_i = rabi_periods_all
mask = ~np.isnan(rabi_q) & ~np.isnan(rabi_i)

# Apply outlier mask to both
combined = np.vstack([rabi_q[mask], rabi_i[mask]])
outlier_mask_q = remove_outliers(combined[0])
outlier_mask_i = remove_outliers(combined[1])
final_mask = outlier_mask_q & outlier_mask_i

rabi_q_filtered = combined[0][final_mask]
rabi_i_filtered = combined[1][final_mask]

# Fit line through origin: y = a * x
# a_fit, _, _, _ = lstsq(rabi_q_filtered.reshape(-1, 1), rabi_i_filtered, rcond=None)


def model(x, a):
    return a * x


a_fit, _ = curve_fit(model, rabi_q_filtered, rabi_i_filtered)

# Plot
plt.figure()
plt.scatter(rabi_q_filtered, rabi_i_filtered, alpha=0.7, label="Data")
plt.plot(
    rabi_q_filtered,
    # a_fit[0] * rabi_q_filtered,
    1.0 * rabi_q_filtered,
    "r--",
    # label=f"Fit: y = {a_fit[0]:.3f}x",
    label=f"Fit: y = {1.0}x",
)
plt.xlabel("Rabi Period (Q channel) [ns]")
plt.ylabel("Rabi Period (I channel) [ns]")
plt.title("Rabi Period Correlation (Outliers Removed)")
plt.grid(True)
plt.axis("equal")
plt.legend()
plt.show()

# Pearson correlation
r, p = pearsonr(rabi_q_filtered, rabi_i_filtered)
print(f"Pearson r = {r:.3f}, p = {p:.2e}")
print(f"Fitted slope (through origin): {a_fit[0]:.4f}")

# pi_q, pi_i = pi_pulses_all
# mask = ~np.isnan(pi_q) & ~np.isnan(pi_i)
# pi_q, pi_i = pi_q[mask], pi_i[mask]

# plt.figure()
# plt.scatter(pi_q, pi_i, alpha=0.7)
# plt.xlabel("π Pulse Time (Q Channel) [ns]")
# plt.ylabel("π Pulse Time (I Channel) [ns]")
# plt.title("π Pulse Time Correlation")
# plt.grid(True)
# plt.axis("equal")
# plt.show()

kpl.show(block=True)
