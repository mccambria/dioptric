# -*- coding: utf-8 -*-
"""
Spin Echo Analysis and Visualization

Created on December 22nd, 2024

@author: Saroj Chand
"""

import sys
import time
import traceback
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from joblib import Parallel, delayed
from scipy.optimize import curve_fit

from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import widefield as widefield

# def revival_model(
#     tau,
#     baseline,
#     rev_time_1,
#     decay_time_1,
#     amp1,
#     freq1,
#     phase1,
#     rev_time_2,
#     decay_time_2,
#     amp2,
#     freq2,
#     phase2,
# ):
#     """
#     Improved revival model for spin echo with Gaussian envelope and oscillations.
#     """
#     # Convert frequency from kHz to µs⁻¹ for correct oscillation
#     freq1_scaled = freq1 / 1e6  # MHz → µs⁻¹
#     freq2_scaled = freq2 / 1e6  # MHz → µs⁻¹

#     # First revival: Gaussian envelope * modulation
#     gauss_env_1 = amp1 * np.exp(-((tau - rev_time_1) ** 2) / (2 * decay_time_1**2))
#     mod_1 = np.cos(2 * np.pi * freq1_scaled * tau + phase1)

#     # Second revival: Gaussian envelope * modulation
#     gauss_env_2 = amp2 * np.exp(-((tau - rev_time_2) ** 2) / (2 * decay_time_2**2))
#     mod_2 = np.cos(2 * np.pi * freq2_scaled * tau + phase2)

#     return baseline - gauss_env_1 * mod_1 - gauss_env_2 * mod_2


# def generate_initial_guess_and_bounds(tau, counts):
#     """
#     Generate initial guesses and bounds for the revival model.
#     """
#     # Baseline (mean of the last few points, assuming flat decay at the end)
#     baseline_guess = np.mean(counts)

#     # First revival: Position of the first significant minimum
#     rev_time_1_guess = 51.5
#     rev_time_2_guess = 2 * rev_time_1_guess

#     # Decay times: Estimate as a fraction of the total time
#     decay_time_guess_1 = (tau[-1] - tau[0]) / 3
#     decay_time_guess_2 = (tau[-1] - tau[0]) / 4

#     # Amplitudes: Negative because revivals correspond to dips in the signal
#     amp1_guess = -np.abs((np.min(counts[: len(tau) // 2]) - baseline_guess))
#     amp2_guess = -np.abs((np.min(counts[len(tau) // 2 :]) - baseline_guess))

#     # Compute FFT to estimate dominant frequency
#     time_step = (tau[1] - tau[0]) * 1e-6  # Convert µs to seconds
#     fft_freqs = np.fft.rfftfreq(len(tau), d=time_step)  # Now in Hz
#     fft_spectrum = np.abs(np.fft.rfft(counts - baseline_guess))

#     # Find peak frequency in kHz
#     freq_guess_1 = fft_freqs[np.argmax(fft_spectrum)] / 1e3  # Convert Hz to kHz
#     freq_guess_2 = freq_guess_1  # Assume second revival shares the same frequency

#     # Print frequency result for debugging
#     print(f"FFT Peak Frequency: {freq_guess_1:.3f} kHz")

#     # Phases: Start with zero phase for both revivals
#     phase_guess_1 = 0
#     phase_guess_2 = 0

#     # Combine all guesses into a single list
#     initial_guess = [
#         baseline_guess,
#         rev_time_1_guess,
#         decay_time_guess_1,
#         amp1_guess,
#         freq_guess_1,  # Stored in kHz
#         phase_guess_1,
#         rev_time_2_guess,
#         decay_time_guess_2,
#         amp2_guess,
#         freq_guess_2,  # Stored in kHz
#         phase_guess_2,
#     ]

#     # Set bounds: Ensure parameters are physically meaningful (FREQUENCIES IN kHz)
#     bounds = (
#         [
#             0,  # baseline
#             40,  # rev_time_1
#             0,  # decay_time_1
#             -np.inf,  # amp1
#             1,  # Lower bound for freq1 (50 kHz)
#             -np.pi,  # phase1
#             0,  # rev_time_2
#             80,  # decay_time_2
#             -np.inf,  # amp2
#             1,  # Lower bound for freq2 (50 kHz)
#             -np.pi,  # phase2
#         ],
#         [
#             1,  # baseline
#             60,  # rev_time_1
#             np.inf,  # decay_time_1
#             np.inf,  # amp1
#             200,  # Upper bound for freq1 (200 kHz)
#             np.pi,  # phase1
#             tau[-1],  # rev_time_2
#             np.inf,  # decay_time_2
#             np.inf,  # amp2
#             200,  # Upper bound for freq2 (200 kHz)
#             np.pi,  # phase2
#         ],
#     )

#     return initial_guess, bounds


def revival_model(tau, baseline, T2, A1, f1, phi1, A2, f2, phi2):
    """
    Physically motivated model for spin echo revivals using damped oscillations.
    """
    freq1_scaled = f1 / 1e3  # Convert kHz to µs⁻¹
    freq2_scaled = f2 / 1e3  # Convert kHz to µs⁻¹

    envelope_1 = np.exp(-tau / T2)  # Exponential decay (T2 coherence time)
    oscillations_1 = A1 * np.cos(2 * np.pi * freq1_scaled * tau + phi1)
    envelope_2 = np.exp(-tau / T2)  # Exponential decay (T2 coherence time)
    oscillations_2 = A2 * np.cos(2 * np.pi * freq2_scaled * tau + phi2)

    return baseline - envelope_1 * oscillations_1 + envelope_2 * oscillations_2


def generate_initial_guess_and_bounds(tau, counts):
    """
    Generate initial guesses and bounds for the physically motivated model.
    """
    baseline_guess = np.mean(counts[-10:])  # Average last few points for baseline
    T2_guess = (tau[-1] - tau[0]) / 2  # Estimate of coherence time
    A1_guess = 0.5 * (np.max(counts) - np.min(counts))  # First amplitude
    A2_guess = A1_guess / 2  # Second amplitude is assumed smaller

    time_step = (tau[1] - tau[0]) * 1e-6  # Convert µs to seconds
    fft_freqs = np.fft.rfftfreq(len(tau), d=time_step)
    fft_spectrum = np.abs(np.fft.rfft(counts - baseline_guess))
    f1_guess = fft_freqs[np.argmax(fft_spectrum)] / 1e3  # Convert Hz to kHz
    f2_guess = f1_guess / 2  # Assume a secondary revival at half frequency

    phi1_guess, phi2_guess = 0, 0  # Assume zero initial phase

    initial_guess = [
        baseline_guess,
        T2_guess,
        A1_guess,
        f1_guess,
        phi1_guess,
        A2_guess,
        f2_guess,
        phi2_guess,
    ]

    bounds = (
        [0, 0, -np.inf, 1, -np.pi, -np.inf, 1, -np.pi],
        [1, np.inf, np.inf, 200, np.pi, np.inf, 200, np.pi],
    )

    return initial_guess, bounds


def single_revival_model(tau, baseline, T2, A1, f1, phi1):
    """
    Physically motivated model for spin echo revivals using damped oscillations.
    """
    freq1_scaled = f1 / 1e3  # Convert kHz to µs⁻¹
    envelope_1 = np.exp(-tau / T2)  # Exponential decay (T2 coherence time)
    oscillations_1 = A1 * np.cos(2 * np.pi * freq1_scaled * tau + phi1)

    return baseline - envelope_1 * oscillations_1


def generate_initial_guess_and_bounds_single(tau, counts):
    """
    Generate initial guesses and bounds for the physically motivated model.
    """
    baseline_guess = np.mean(counts[-10:])  # Average last few points for baseline
    T2_guess = (tau[-1] - tau[0]) / 2  # Estimate of coherence time
    A1_guess = 0.5 * (np.max(counts) - np.min(counts))  # First amplitude
    time_step = (tau[1] - tau[0]) * 1e-6  # Convert µs to seconds
    fft_freqs = np.fft.rfftfreq(len(tau), d=time_step)
    fft_spectrum = np.abs(np.fft.rfft(counts - baseline_guess))
    f1_guess = fft_freqs[np.argmax(fft_spectrum)] / 1e3  # Convert Hz to kHz

    phi1_guess = 0  # Assume zero initial phase

    initial_guess = [
        baseline_guess,
        T2_guess,
        A1_guess,
        f1_guess,
        phi1_guess,
    ]

    bounds = (
        [0.2, 0, -np.inf, 1, -np.pi, -np.inf, 1, -np.pi],
        [1.5, np.inf, np.inf, 200, np.pi, np.inf, 200, np.pi],
    )

    return initial_guess, bounds


# Analyze and visualize spin echo data
def fit_spin_echo(nv_list, taus, norm_counts, norm_counts_ste):
    num_nvs = len(nv_list)

    def fit_single_nv(nv_idx):
        print(f"Fitting NV {nv_idx}...")

        nv_tau = taus
        nv_counts = norm_counts[nv_idx]

        try:
            initial_guess, bounds = generate_initial_guess_and_bounds(nv_tau, nv_counts)

            # Ensure initial guess is within bounds
            initial_guess = np.clip(initial_guess, bounds[0], bounds[1])

            popt, pcov = curve_fit(
                revival_model,
                nv_tau,
                nv_counts,
                p0=initial_guess,
                bounds=bounds,
                maxfev=10000,
            )

            fit_fn = lambda tau: revival_model(tau, *popt)

        except ValueError as e:
            print(f"ValueError for NV {nv_idx}: {e}")
            print(f"Initial Guess: {initial_guess}")
            print(f"Bounds: {bounds}")
            fit_fn = None
            popt = [None] * len(initial_guess)

        except RuntimeError as e:
            print(f"RuntimeError for NV {nv_idx}: {e}")
            fit_fn = None
            popt = [None] * len(initial_guess)

        return fit_fn, popt

    # Parallel execution of fitting for each NV
    results = Parallel(n_jobs=-1)(
        delayed(fit_single_nv)(nv_idx) for nv_idx in range(num_nvs)
    )

    fit_fns, popts = zip(*results)

    return fit_fns, popts


def plot_spin_echo_fits(
    nv_list,
    taus,
    norm_counts,
    norm_counts_ste,
    fit_fns=None,
    popts=None,
):
    """
    Plot the fitted spin echo data for each NV center separately and print fitting quality metrics.
    """
    num_nvs = len(nv_list)
    taus = np.array(taus)

    for nv_ind in range(num_nvs):
        fig, ax = plt.subplots()
        # Scatter plot with error bars
        ax.errorbar(
            taus,
            norm_counts[nv_ind],
            yerr=np.abs(norm_counts_ste[nv_ind]),
            fmt="o",
            label="Data",
        )

        # # Compute fitting quality metrics
        # if fit_fns[nv_ind] is not None:
        #     tau_dense = np.linspace(0, taus.max(), 300)
        #     fit_values = fit_fns[nv_ind](taus)
        #     residuals = norm_counts[nv_ind] - fit_values

        #     # Residual Sum of Squares (RSS)
        #     rss = np.sum(residuals**2)

        #     # Reduced Chi-Square (assuming errors are std errors in counts)
        #     degrees_of_freedom = len(taus) - len(popts[nv_ind])
        #     chi_squared_red = (
        #         np.sum((residuals / np.abs(norm_counts_ste[nv_ind])) ** 2)
        #         / degrees_of_freedom
        #         if degrees_of_freedom > 0
        #         else np.nan
        #     )

        #     # Coefficient of Determination (R^2)
        #     ss_total = np.sum((norm_counts[nv_ind] - np.mean(norm_counts[nv_ind])) ** 2)
        #     r_squared = 1 - (rss / ss_total) if ss_total > 0 else np.nan

        #     print(
        #         f"NV {nv_ind}: RSS = {rss:.4f}, Chi-Squared_red = {chi_squared_red:.4f}, R^2 = {r_squared:.4f}"
        #     )

        #     # Plot the fitted curve
        #     ax.plot(tau_dense, fit_fns[nv_ind](tau_dense), "-", label="Fit")

        ax.set_title(f"NV {nv_ind}")
        ax.set_xlabel("Time (us)")
        ax.set_ylabel("Norm. NV- Population")
        ax.legend()
        ax.grid(True)
        plt.show(block=True)


def plot_analysis_parameters(meaningful_parameters):
    import pandas as pd

    params_df = pd.DataFrame(meaningful_parameters)

    # Define the parameters to plot
    plot_columns = [
        "Revival Time (µs)",
        "Decay Time (µs)",
        "Frequency (Hz)",
        "Chi-Squared",
    ]

    # Remove outliers using IQR
    def remove_outliers(data):
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        return data[(data >= lower_bound) & (data <= upper_bound)]

    # Filter outliers for each parameter
    filtered_params = params_df.copy()
    for param in plot_columns:
        filtered_params[param] = remove_outliers(params_df[param])

    # Set up the figure
    sns.set(style="whitegrid", palette="muted")
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.flatten()

    # Create scatter plots for each parameter
    for i, param in enumerate(plot_columns):
        if i < len(axes):
            ax = axes[i]
            sns.scatterplot(
                x=filtered_params.index,
                y=filtered_params[param],
                ax=ax,
                marker="o",
                edgecolor="w",
                s=50,
            )
            ax.set_title(param)
            ax.set_xlabel("NV Index")
            ax.set_ylabel(param)

    # Adjust layout
    plt.tight_layout()
    plt.suptitle(
        "Scatter Plots of Fitted Parameters (Outliers Removed)", fontsize=16, y=1.02
    )
    plt.subplots_adjust(top=0.9)
    plt.show()


def plot_spin_echo_all(nv_list, taus, norm_counts, norm_counts_ste):
    kpl.init_kplotlib()
    sns.set(style="whitegrid", palette="muted")
    num_nvs = len(nv_list)
    colors = sns.color_palette("deep", num_nvs)
    num_cols = 7
    num_rows = int(np.ceil(len(nv_list) / num_cols))

    # Full plot
    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=(num_cols * 1.5, num_rows * 3),
        sharex=True,
        sharey=False,
        constrained_layout=True,
        gridspec_kw={"wspace": 0.0, "hspace": 0.0},
    )
    axes = axes.flatten()

    for nv_idx, ax in enumerate(axes):
        if nv_idx >= len(nv_list):
            ax.axis("off")
            continue

        nv_tau = taus  # Convert to µs
        nv_counts = norm_counts[nv_idx]
        # Plot data and fit on full plot
        sns.lineplot(
            x=nv_tau,
            y=nv_counts,
            ax=ax,
            color=colors[nv_idx % len(colors)],
            lw=0,
            marker="o",
            markersize=4,
            # label=f"NV {nv_idx}",
        )
        ax.errorbar(
            nv_tau,
            norm_counts[nv_idx],
            yerr=abs(norm_counts_ste[nv_idx]),
            fmt="none",
            ecolor=colors[nv_idx % len(colors)],
            alpha=0.9,
        )
        # ax.legend(fontsize="xx-small")
        ax.grid(True, which="both", linestyle="--", linewidth=0.5)
        ax.tick_params(labelleft=False)

    for col in range(num_cols):
        bottom_row_idx = num_rows * num_cols - num_cols + col
        if bottom_row_idx < len(axes):
            ax = axes[bottom_row_idx]
            tick_positions = np.linspace(min(taus) + 2, max(taus) - 2, 6)
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(
                [f"{tick:.2f}" for tick in tick_positions],
                rotation=45,
                fontsize=9,
                y=0.00,
            )
            ax.set_xlim(min(taus), max(taus))
            ax.set_xlabel("Time (µs)")
        else:
            ax.set_xticklabels([])
    fig.text(
        0.000,
        0.5,
        "NV$^{-}$ Population",
        va="center",
        rotation="vertical",
        fontsize=12,
    )
    fig.suptitle(f"Spin Echo {all_file_ids_str}", fontsize=12, y=0.99)
    fig.tight_layout(pad=0.2, rect=[0.02, 0.01, 0.99, 0.99])
    kpl.show()


if __name__ == "__main__":
    kpl.init_kplotlib()
    # Define the file IDs to process (66 shallow NVs)
    # file_ids = [1783227865584, 1783326666625, 1783448620086, 1783572765304]
    # file_ids = [
    #     1785346398683,
    #     1785254038960,
    #     1785155665809,
    #     1785057603893,
    #     1784917916514,
    #     1784779506353,
    # ]

    # 117 deep NVs
    # file_ids = [
    #     1734158411844,
    #     1734273666255,
    #     1734371251079,
    #     1734461462293,
    #     1734569197701,
    # ]
    # rubin75 NVs
    # file_ids = [1809864601542, 1810050697942, 1810230561491, 1810371359284]
    # rubin75 NVs
    # file_ids = [1811334050314, 1811401206447, 1811464617147, 1811540653210]
    # rubin75 NVs after making both orientation degenerate
    # file_ids = [1835778335625, 1836023279415]

    file_ids = [1837153340732, 1837462226158]
    all_file_ids_str = "_".join(map(str, file_ids))
    now = datetime.now()
    date_time_str = now.strftime("%Y%m%d_%H%M%S")
    file_name = dm.get_file_name(file_id=file_ids[0])
    timestamp = dm.get_time_stamp()
    file_path = dm.get_file_path(
        __file__, file_name, f"{all_file_ids_str}_{date_time_str}"
    )
    print(f"File path: {file_path}")
    # Process and analyze data from multiple files
    try:
        data = widefield.process_multiple_files(file_ids)
        nv_list = data["nv_list"]
        taus = data["taus"]
        total_evolution_times = 2 * np.array(taus) / 1e3
        counts = np.array(data["counts"])
        sig_counts, ref_counts = counts[0], counts[1]
        norm_counts, norm_counts_ste = widefield.process_counts(
            nv_list, sig_counts, ref_counts, threshold=True
        )
        nv_num = len(nv_list)
        ids_num = len(file_ids)
        # fit_fns, popts = fit_spin_echo(
        #     nv_list, total_evolution_times, norm_counts, norm_counts_ste
        # )
        # plot_spin_echo_fits(
        #     nv_list, total_evolution_times, norm_counts, norm_counts_ste
        # )
        plot_spin_echo_all(nv_list, total_evolution_times, norm_counts, norm_counts_ste)
    except Exception as e:
        print(f"Error occurred: {e}")
        print(traceback.format_exc())

    kpl.show(block=True)
