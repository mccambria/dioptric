# -*- coding: utf-8 -*-
"""
Widefield Rabi experiment

Created on November 29th, 2023

@author: mccambria
"""

import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from majorroutines.widefield import base_routine
from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import tool_belt as tb
from utils import widefield as widefield


def quartic_decay(tau, amplitude, revival_time, quartic_decay_time, T2):
    baseline = 1 + amplitude
    val = baseline
    # print(len(amplitudes))
    envelope = np.exp(-((tau / T2) ** 3))
    num_revivals = 3
    for ind in range(num_revivals):
        exp_part = np.exp(-(((tau - ind * revival_time) / quartic_decay_time) ** 4))
        val -= amplitude * envelope * exp_part
    return val


def constant(tau):
    norm = 1
    if isinstance(tau, list):
        return [norm] * len(tau)
    elif type(tau) == np.ndarray:
        return np.array([norm] * len(tau))
    else:
        return norm


# def create_raw_data_figure(nv_list, taus, counts, counts_ste):
#     total_evolution_times = 2 * np.array(taus) / 1e3
#     for ind in range(len(nv_list)):
#         subset_inds = [ind]
#         fig, ax = plt.subplots()
#         widefield.plot_raw_data(
#             ax,
#             nv_list,
#             total_evolution_times,
#             counts,
#             counts_ste,
#             subset_inds=subset_inds,
#         )
#         ax.set_xlabel("Total evolution time (µs)")
#         ax.set_ylabel("Counts")
#     return fig


def create_raw_data_figure(nv_list, taus, counts, counts_ste):
    fig, ax = plt.subplots()
    total_evolution_times = 2 * np.array(taus) / 1e3
    widefield.plot_raw_data(ax, nv_list, total_evolution_times, counts, counts_ste)
    ax.set_xlabel("Total evolution time (µs)")
    ax.set_ylabel("Counts")
    return fig


def create_fit_figure(nv_list, taus, counts, counts_ste, norms):
    total_evolution_times = 2 * np.array(taus) / 1e3

    fit_fns = []
    popts = []

    num_nvs = len(nv_list)

    for nv_ind in range(num_nvs):
        nv_counts = counts[nv_ind] / norms[nv_ind]
        nv_counts_ste = counts_ste[nv_ind] / norms[nv_ind]

        try:
            if nv_ind != 1:
                fit_fn = quartic_decay
                amplitude_guess = np.quantile(nv_counts, 0.7)
                guess_params = [amplitude_guess, 175, 15, 500]
                popt, pcov = curve_fit(
                    fit_fn,
                    total_evolution_times,
                    nv_counts,
                    p0=guess_params,
                    sigma=nv_counts_ste,
                    absolute_sigma=True,
                    maxfev=10000,
                    bounds=(
                        (0, 100, 5, 100),
                        (100, 500, 30, 1000),
                    ),
                )
            else:
                fit_fn = constant
                popt = []
            fit_fns.append(fit_fn)
            popts.append(popt)
        except Exception as exc:
            print(exc)
            fit_fns.append(None)
            popts.append(None)

        residuals = fit_fn(total_evolution_times, *popt) - nv_counts
        chi_sq = np.sum((residuals / nv_counts_ste) ** 2)
        red_chi_sq = chi_sq / (len(nv_counts) - len(popt))
        print(f"Red chi sq: {round(red_chi_sq, 3)}")

    ### Make the figure

    # fig, ax = plt.subplots()
    fig, axes_pack = plt.subplots(
        nrows=3, ncols=2, sharex=True, sharey=True, figsize=[6.5, 6.0]
    )
    axes_pack = axes_pack.flatten()
    norm_counts = counts / norms[:, np.newaxis]
    norm_counts_ste = counts_ste / norms[:, np.newaxis]
    widefield.plot_fit(
        axes_pack,
        nv_list,
        total_evolution_times,
        norm_counts,
        norm_counts_ste,
        fit_fns,
        popts,
    )
    ax = axes_pack[-2]
    # ax.set_xlabel("Total evolution time (µs)")
    # ax.set_ylabel("Normalized fluorescence")
    ax.set_xlabel(" ")
    fig.text(0.55, 0.01, "Total evolution time (µs)", ha="center")
    ax.set_ylabel(" ")
    fig.text(0.01, 0.55, "Normalized fluorescence", va="center", rotation="vertical")
    # ax.set_ylim([0.9705, 1.1])
    # ax.set_yticks([1.0, 1.1])
    return fig


def create_correlation_figure(nv_list, taus, counts):
    total_evolution_times = 2 * np.array(taus) / 1e3

    # fig, ax = plt.subplots()
    fig, axes_pack = plt.subplots(
        nrows=5, ncols=5, sharex=True, sharey=True, figsize=[10, 10]
    )

    widefield.plot_correlations(axes_pack, nv_list, total_evolution_times, counts)

    ax = axes_pack[-1, 0]
    ax.set_xlabel(" ")
    fig.text(0.55, 0.01, "Total evolution time (µs)", ha="center")
    ax.set_ylabel(" ")
    fig.text(0.01, 0.55, "Correlation coefficient", va="center", rotation="vertical")
    return fig


def calc_T2_times(
    peak_total_evolution_times, peak_contrasts, peak_contrast_errs, baselines
):
    for nv_ind in range(len(peak_contrasts)):
        baseline = baselines[nv_ind]

        def envelope(total_evolution_time, T2):
            return (
                -(baseline - 1) * np.exp(-((total_evolution_time / T2) ** 3)) + baseline
            )

        guess_params = (400,)
        popt, pcov = curve_fit(
            envelope,
            peak_total_evolution_times[nv_ind],
            peak_contrasts[nv_ind],
            p0=guess_params,
            sigma=peak_contrast_errs[nv_ind],
            absolute_sigma=True,
        )
        pste = np.sqrt(np.diag(pcov))
        print(f"{round(popt[0])} +/- {round(pste[0])}")


def main(nv_list, num_steps, num_reps, num_runs, min_tau, max_tau):
    ### Some initial setup

    pulse_gen = tb.get_server_pulse_gen()
    seq_file = "spin_echo.py"
    taus = np.linspace(min_tau, max_tau, num_steps)

    ### Collect the data

    # MCC testing
    # tau = taus[0]
    # seq_args = widefield.get_base_scc_seq_args(nv_list)
    # seq_args.append(tau)
    # print(seq_args)
    # return

    def step_fn(tau_ind):
        tau = taus[tau_ind]
        seq_args = widefield.get_base_scc_seq_args(nv_list)
        seq_args.append(tau)
        # print(seq_args)
        seq_args_string = tb.encode_seq_args(seq_args)
        pulse_gen.stream_load(seq_file, seq_args_string, num_reps)

    counts, ref_counts, raw_data = base_routine.main(
        nv_list, num_steps, num_reps, num_runs, step_fn
    )

    ### Process and plot

    avg_counts, avg_counts_ste, norms = widefield.process_counts(counts, ref_counts)

    raw_fig = create_raw_data_figure(nv_list, taus, avg_counts, avg_counts_ste)
    try:
        fit_fig = create_fit_figure(nv_list, taus, avg_counts, avg_counts_ste, norms)
    except Exception as exc:
        print(exc)
        fit_fig = None

    ### Clean up and return

    tb.reset_cfm()
    kpl.show()

    timestamp = dm.get_time_stamp()
    raw_data |= {
        "timestamp": timestamp,
        "taus": taus,
        "tau-units": "ns",
        "min_tau": min_tau,
        "max_tau": max_tau,
    }

    repr_nv_sig = widefield.get_repr_nv_sig(nv_list)
    repr_nv_name = repr_nv_sig["name"]
    file_path = dm.get_file_path(__file__, timestamp, repr_nv_name)
    dm.save_raw_data(raw_data, file_path)
    dm.save_figure(raw_fig, file_path)
    if fit_fig is not None:
        file_path = dm.get_file_path(__file__, timestamp, repr_nv_name + "-fit")
        dm.save_figure(fit_fig, file_path)


if __name__ == "__main__":
    kpl.init_kplotlib()

    # file_name = ""
    # data = dm.get_raw_data(file_name)
    # data = dm.get_raw_data(file_id=1396164244162, no_npz=True)
    # data = dm.get_raw_data(file_id=1398135297223, no_npz=True)
    # data = dm.get_raw_data(file_id=1397700913905, no_npz=True)
    data = dm.get_raw_data(file_id=1409676402822, load_npz=True)

    nv_list = data["nv_list"]
    num_nvs = len(nv_list)
    num_steps = data["num_steps"]
    num_runs = data["num_runs"]
    taus = data["taus"]
    counts = np.array(data["counts"])
    ref_counts = np.array(data["ref_counts"])
    # counts = counts[:, : num_runs // 2, :, :]

    # data = dm.get_raw_data(file_id=1398135297223, no_npz=True)

    avg_counts, avg_counts_ste, norms = widefield.process_counts(counts, ref_counts)
    raw_fig = create_raw_data_figure(nv_list, taus, avg_counts, avg_counts_ste)
    fit_fig = create_fit_figure(nv_list, taus, avg_counts, avg_counts_ste, norms)
    correlation_fig = create_correlation_figure(nv_list, taus, counts)

    peak_total_evolution_times = [
        [2, 162, 335.0],
        [2, 162, 338.12],
        [2, 170, 335.24],
        [2, 162, 338.76],
        [2, 162, 335.8],
    ]
    peak_contrasts = [
        [1.0, 1.01101875, 1.010826],
        [1.0, 1.03949457, 1.024653],
        [1.0, 1.03255067, 1.031515],
        [1.0, 1.00638179, 1.011792],
        [1.0, 1.01289830, 1.020581],
    ]
    peak_contrast_errs = [
        [0.00606988, 0.006136653, 0.0066262],
        [0.00701808, 0.007445652, 0.0050701],
        [0.00695533, 0.007280518, 0.0062269],
        [0.00636465, 0.006256734, 0.0059204],
        [0.00516468, 0.005303820, 0.0046309],
    ]
    baselines = [1.0594120, 1.0891454, 1.0631568, 1.0335516, 1.0499098]
    calc_T2_times(
        peak_total_evolution_times, peak_contrasts, peak_contrast_errs, baselines
    )

    plt.show(block=True)
