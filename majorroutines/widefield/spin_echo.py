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


def quartic_decay(tau, baseline, revival_time, quartic_decay_time, T2_ms, env_exp):
    # baseline = 0.5
    amplitude = baseline
    val = baseline
    # print(len(amplitudes))
    T2_us = 1000 * T2_ms
    envelope = np.exp(-((tau / T2_us) ** env_exp))
    num_revivals = 3
    for ind in range(num_revivals):
        exp_part = np.exp(-(((tau - ind * revival_time) / quartic_decay_time) ** 2))
        val -= amplitude * envelope * exp_part
    return val


def quartic_decay_osc(
    tau,
    baseline,
    revival_time,
    quartic_decay_time,
    T2_ms,
    env_exp,
    osc_freq0,
    # osc_freq1,
):
    # # baseline = 0.5
    # env = quartic_decay(tau, baseline, revival_time, quartic_decay_time, T2_ms, env_exp)
    # return baseline * (1 + ((1 - env / baseline) * np.cos(2 * np.pi * osc_freq * tau)))
    #
    # baseline = 0.5
    amplitude = baseline
    val = baseline
    # print(len(amplitudes))
    T2_us = 1000 * T2_ms
    envelope = np.exp(-((tau / T2_us) ** env_exp))
    num_revivals = 3
    for ind in range(num_revivals):
        exp_part = np.exp(-(((tau - ind * revival_time) / quartic_decay_time) ** 2))
        coeff = (
            1 if ind == 0 else 0.5 * np.cos(2 * np.pi * osc_freq0 * tau)
            # * 0.5
            # * np.cos(2 * np.pi * osc_freq1 * tau)
        )
        val -= coeff * amplitude * envelope * exp_part
    return val


def quartic_decay_three_osc(
    tau,
    baseline,
    revival_time,
    quartic_decay_time,
    T2_ms,
    env_exp,
    osc_freq0,
    osc_freq1,
    osc_freq2,
):
    # # baseline = 0.5
    # env = quartic_decay(tau, baseline, revival_time, quartic_decay_time, T2_ms, env_exp)
    # return baseline * (1 + ((1 - env / baseline) * np.cos(2 * np.pi * osc_freq * tau)))
    #
    # baseline = 0.5
    amplitude = baseline
    val = baseline
    # print(len(amplitudes))
    T2_us = 1000 * T2_ms
    envelope = np.exp(-((tau / T2_us) ** env_exp))
    num_revivals = 3
    for ind in range(num_revivals):
        exp_part = np.exp(-(((tau - ind * revival_time) / quartic_decay_time) ** 2))
        coeff = (
            1
            if ind == 0
            else 0.5
            * np.cos(2 * np.pi * osc_freq0 * tau)
            * np.cos(2 * np.pi * osc_freq1 * tau)
            * np.cos(2 * np.pi * osc_freq2 * tau)
        )
        val -= coeff * amplitude * envelope * exp_part
    return val


def constant(tau):
    norm = 1
    if isinstance(tau, list):
        return [norm] * len(tau)
    elif type(tau) == np.ndarray:
        return np.array([norm] * len(tau))
    else:
        return norm


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
    num_nvs = len(nv_list)
    taus = np.array(data["taus"])
    counts = np.array(data["counts"])
    # counts = np.array(data["states"])

    sig_counts = counts[0]
    ref_counts = counts[1]
    avg_counts, avg_counts_ste, norms = widefield.process_counts(
        nv_list, sig_counts, ref_counts, threshold=True
    )

    norms_ms0_newaxis = norms[0][:, np.newaxis]
    norms_ms1_newaxis = norms[1][:, np.newaxis]
    contrast = norms_ms1_newaxis - norms_ms0_newaxis
    norm_counts = (avg_counts - norms_ms0_newaxis) / contrast
    norm_counts_ste = avg_counts_ste / contrast

    # Put everything in order to help curve_fit
    sorted_inds = taus.argsort()
    taus = taus[sorted_inds]
    total_evolution_times = 2 * np.array(taus) / 1e3
    norm_counts = np.array(
        [norm_counts[nv_ind, sorted_inds] for nv_ind in range(num_nvs)]
    )
    norm_counts_ste = np.array(
        [norm_counts_ste[nv_ind, sorted_inds] for nv_ind in range(num_nvs)]
    )

    fit_fns = []
    popts = []
    freq0_guesses = {0: 0.047, 1: 0.047, 2: 0.05, 4: 0.2, 5: 0.2, 8: 0.2}
    freq1_guesses = {2: 0.047, 4: 0.18, 5: 0.18}
    freq2_guesses = {2: 0.01, 4: 0.047, 5: 0.047}

    for nv_ind in range(num_nvs):
        nv_counts = norm_counts[nv_ind]
        nv_counts_ste = norm_counts_ste[nv_ind]

        try:
            if nv_ind in [2, 4, 5]:
                fit_fn = quartic_decay_three_osc
                freq0_guess = freq0_guesses[nv_ind]
                freq1_guess = freq1_guesses[nv_ind]
                freq2_guess = freq2_guesses[nv_ind]
                guess_params = [
                    0.5,
                    75,
                    10,
                    0.4,
                    2,
                    freq0_guess,
                    freq1_guess,
                    freq2_guess,
                ]
                bounds = (
                    (0, 0, 0, 0, 1, 0, 0, 0),
                    (1, np.inf, np.inf, np.inf, 5, np.inf, np.inf, np.inf),
                )
            elif nv_ind in [0, 1, 8]:
                fit_fn = quartic_decay_osc
                freq0_guess = freq0_guesses[nv_ind]
                guess_params = [0.5, 75, 10, 0.4, 2, freq0_guess]
                bounds = (
                    (0, 0, 0, 0, 1, 0),
                    (1, np.inf, np.inf, np.inf, 5, np.inf),
                )
            else:
                fit_fn = quartic_decay
                guess_params = [0.5, 75, 10, 0.4, 2]
                bounds = ((0, 0, 0, 0, 1), (1, np.inf, np.inf, np.inf, 5))
            popt, pcov, info, msg, ier = curve_fit(
                fit_fn,
                total_evolution_times,
                nv_counts,
                p0=guess_params,
                sigma=nv_counts_ste,
                absolute_sigma=True,
                full_output=True,
                bounds=bounds,
            )
        except Exception as exc:
            # pass
            print(exc)
            fit_fn = None
            popt = None
        fit_fns.append(fit_fn)
        popts.append(popt)

        if fit_fn is not None:
            residuals = fit_fn(total_evolution_times, *popt) - nv_counts
            chi_sq = np.sum((residuals / nv_counts_ste) ** 2)
            red_chi_sq = chi_sq / (len(nv_counts) - len(popt))
            print(f"Red chi sq: {round(red_chi_sq, 3)}")

    ### Make the figure

    if axes_pack is None:
        fig, axes_pack, layout = kpl.subplot_mosaic(num_nvs, num_rows=2)
    else:
        fig = None

    widefield.plot_fit(
        axes_pack,
        nv_list,
        total_evolution_times,
        norm_counts,
        norm_counts_ste,
        fit_fns,
        popts,
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

    # fig, ax = plt.subplots()
    # taus = np.linspace(0, 400, 1000)
    # ys = quartic_decay(taus, *[0.6, 75, 10, 400])
    # # ys = quartic_decay(taus, *[175, 15, 500])
    # kpl.plot_line(ax, taus, ys)

    data = dm.get_raw_data(file_id=1548381879624)

    # create_raw_data_figure(data)
    create_fit_figure(data)

    plt.show(block=True)
