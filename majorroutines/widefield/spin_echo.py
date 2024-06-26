# -*- coding: utf-8 -*-
"""
Widefield Rabi experiment

Created on November 29th, 2023

@author: mccambria
"""

import time
import traceback

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


def create_raw_data_figure(data):
    nv_list = data["nv_list"]
    taus = data["taus"]
    counts = np.array(data["states"])
    sig_counts, ref_counts = counts[0], counts[1]

    avg_counts, avg_counts_ste, norms = widefield.process_counts(
        nv_list, sig_counts, ref_counts, threshold=False
    )
    # avg_counts -= norms[:, np.newaxis]

    fig, ax = plt.subplots()
    total_evolution_times = 2 * np.array(taus) / 1e3
    widefield.plot_raw_data(
        ax, nv_list, total_evolution_times, avg_counts, avg_counts_ste
    )
    ax.set_xlabel("Total evolution time (µs)")
    ax.set_ylabel("Counts")
    return fig


def create_fit_figure(data, axes_pack=None, layout=None, no_legend=False):
    nv_list = data["nv_list"]
    taus = data["taus"]
    counts = np.array(data["states"])

    num_nvs = len(nv_list)
    total_evolution_times = 2 * np.array(taus) / 1e3
    # total_evolution_times = np.array(taus) / 1e3

    sig_counts = counts[0]
    ref_counts = counts[1]
    avg_counts, avg_counts_ste, norms = widefield.process_counts(
        nv_list, sig_counts, ref_counts, threshold=False
    )

    fit_fns = []
    popts = []

    # for nv_ind in range(num_nvs):
    #     nv_counts = counts[nv_ind] / norms[nv_ind]
    #     nv_counts_ste = counts_ste[nv_ind] / norms[nv_ind]

    #     try:
    #         if nv_ind != 1:
    #             fit_fn = quartic_decay
    #             amplitude_guess = np.quantile(nv_counts, 0.7)
    #             guess_params = [amplitude_guess, 175, 15, 500]
    #             popt, pcov = curve_fit(
    #                 fit_fn,
    #                 total_evolution_times,
    #                 nv_counts,
    #                 p0=guess_params,
    #                 sigma=nv_counts_ste,
    #                 absolute_sigma=True,
    #                 maxfev=10000,
    #                 bounds=(
    #                     (0, 100, 5, 100),
    #                     (100, 500, 30, 1000),
    #                 ),
    #             )
    #         else:
    #             fit_fn = constant
    #             popt = []
    #         fit_fns.append(fit_fn)
    #         popts.append(popt)
    #     except Exception as exc:
    #         print(exc)
    #         fit_fns.append(None)
    #         popts.append(None)

    #     residuals = fit_fn(total_evolution_times, *popt) - nv_counts
    #     chi_sq = np.sum((residuals / nv_counts_ste) ** 2)
    #     red_chi_sq = chi_sq / (len(nv_counts) - len(popt))
    #     print(f"Red chi sq: {round(red_chi_sq, 3)}")

    ### Make the figure

    if axes_pack is None:
        fig, axes_pack, layout = kpl.subplot_mosaic(num_nvs, num_rows=2)
    else:
        fig = None

    norms_ms0_newaxis = norms[0][:, np.newaxis]
    norms_ms1_newaxis = norms[1][:, np.newaxis]
    contrast = norms_ms1_newaxis - norms_ms0_newaxis
    norm_counts = (avg_counts - norms_ms0_newaxis) / contrast
    norm_counts_ste = avg_counts_ste / contrast

    widefield.plot_fit(
        axes_pack,
        nv_list,
        total_evolution_times,
        norm_counts,
        norm_counts_ste,
        # fit_fns,
        # popts,
        no_legend=no_legend,
    )
    ax = axes_pack[layout[-1, 0]]
    kpl.set_shared_ax_xlabel(ax, "Total evolution time (µs)")
    # kpl.set_shared_ax_ylabel(ax, "Change in $P($NV$^{-})$")
    kpl.set_shared_ax_ylabel(ax, "Norm. NV$^{-}$ population")
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

    # MCC x point manipulation
    taus = np.linspace(min_tau, max_tau, num_steps).tolist()
    revival_width = 5e3
    taus.extend(np.linspace(min_tau, min_tau + revival_width, 11).tolist())
    taus.extend(np.linspace(38e3 - revival_width, 38e3 + revival_width, 61).tolist())
    taus.extend(np.linspace(76e3 - revival_width, 76e3 + revival_width, 21).tolist())
    taus = [round(el / 4) * 4 for el in taus]
    num_steps = len(taus)
    # print(taus)

    uwave_ind_list = [0, 1]

    ### Collect the data

    def run_fn(shuffled_step_inds):
        shuffled_taus = [taus[ind] for ind in shuffled_step_inds]
        seq_args = [
            widefield.get_base_scc_seq_args(nv_list, uwave_ind_list),
            shuffled_taus,
        ]
        # print(seq_args)
        seq_args_string = tb.encode_seq_args(seq_args)
        pulse_gen.stream_load(seq_file, seq_args_string, num_reps)

    raw_data = base_routine.main(
        nv_list,
        num_steps,
        num_reps,
        num_runs,
        run_fn,
        uwave_ind_list=uwave_ind_list,
        save_images=False,
    )

    ### Process and plot

    timestamp = dm.get_time_stamp()
    raw_data |= {
        "timestamp": timestamp,
        "tau-units": "ns",
        "taus": taus,
        "min_tau": min_tau,
        "max_tau": max_tau,
    }

    try:
        raw_fig = create_raw_data_figure(data)
        fit_fig = create_fit_figure(data)
    except Exception:
        print(traceback.format_exc())
        raw_fig = None
        fit_fig = None

    ### Clean up and return

    tb.reset_cfm()
    kpl.show()

    repr_nv_sig = widefield.get_repr_nv_sig(nv_list)
    repr_nv_name = repr_nv_sig.name
    file_path = dm.get_file_path(__file__, timestamp, repr_nv_name)
    dm.save_raw_data(raw_data, file_path)
    if raw_fig is not None:
        dm.save_figure(raw_fig, file_path)
    if fit_fig is not None:
        file_path = dm.get_file_path(__file__, timestamp, repr_nv_name + "-fit")
        dm.save_figure(fit_fig, file_path)


if __name__ == "__main__":
    kpl.init_kplotlib()

    data = dm.get_raw_data(file_id=1548381879624)

    create_raw_data_figure(data)
    create_fit_figure(data)

    plt.show(block=True)
