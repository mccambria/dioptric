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
from scipy.signal import lombscargle

from majorroutines.widefield import base_routine
from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import tool_belt as tb
from utils import widefield as widefield


def quartic_decay_base(
    tau,
    baseline,
    revival_time,
    quartic_decay_time,
    amp1,
    amp2,
    osc_freqs=None,
):
    # baseline = 0.5
    amplitude = baseline
    val = 0
    # print(len(amplitudes))
    # T2_us = 1000 * T2_ms
    # envelope = np.exp(-((tau / T2_us) ** env_exp))
    envelope = 1
    num_revivals = 3
    amps = [1, amp1, amp2]
    for ind in range(num_revivals):
        exp_part = np.exp(-(((tau - ind * revival_time) / quartic_decay_time) ** 2))
        if ind == 0 or osc_freqs is None:
            mod = 1
        else:
            # freq_sum = osc_freqs[0] + osc_freqs[1]
            # freq_diff = np.abs(osc_freqs[0] - osc_freqs[1])
            # mod = (
            #     1
            #     - 2
            #     * np.sin(2 * np.pi * freq_sum * tau / 2) ** 2
            #     * np.sin(2 * np.pi * freq_diff * tau / 2) ** 2
            #     # * np.sin(2 * np.pi * osc_freqs[0] * tau / 2) ** 2
            #     # * np.sin(2 * np.pi * osc_freqs[1] * tau / 2) ** 2
            # )
            #
            mod = [np.cos(2 * np.pi * osc_freq * tau) for osc_freq in osc_freqs]
            mod = np.sum(mod, axis=0) / len(osc_freqs)
            #
            # mod = [np.cos(2 * np.pi * osc_freq * tau) for osc_freq in osc_freqs]
            # mod = np.prod(mod, axis=0)
        amp = amps[ind]
        val += amp * mod * exp_part
    val = baseline - amplitude * envelope * val
    return val


def quartic_decay(
    tau,
    baseline,
    revival_time,
    quartic_decay_time,
    T2_ms,
    env_exp,
):
    return quartic_decay_base(
        tau,
        baseline,
        revival_time,
        quartic_decay_time,
        T2_ms,
        env_exp,
    )


def quartic_decay_one_osc(
    tau,
    baseline,
    revival_time,
    quartic_decay_time,
    T2_ms,
    env_exp,
    osc_freq0,
):
    osc_freqs = [osc_freq0]
    return quartic_decay_base(
        tau,
        baseline,
        revival_time,
        quartic_decay_time,
        T2_ms,
        env_exp,
        osc_freqs,
    )


def quartic_decay_two_osc(
    tau,
    baseline,
    revival_time,
    quartic_decay_time,
    T2_ms,
    env_exp,
    osc_freq0,
    osc_freq1,
):
    osc_freqs = [osc_freq0, osc_freq1]
    return quartic_decay_base(
        tau,
        baseline,
        revival_time,
        quartic_decay_time,
        T2_ms,
        env_exp,
        osc_freqs,
    )


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
    osc_freqs = [osc_freq0, osc_freq1, osc_freq2]
    return quartic_decay_base(
        tau,
        baseline,
        revival_time,
        quartic_decay_time,
        T2_ms,
        env_exp,
        osc_freqs,
    )


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


def create_fit_figure(data, axes_pack=None, layout=None, no_legend=True):
    nv_list = data["nv_list"]
    num_nvs = len(nv_list)
    num_steps = data["num_steps"]
    taus = np.array(data["taus"])
    total_evolution_times = 2 * np.array(taus) / 1e3
    counts = np.array(data["counts"])

    sig_counts = counts[0]
    ref_counts = counts[1]
    norm_counts, norm_counts_ste = widefield.process_counts(
        nv_list, sig_counts, ref_counts, threshold=True
    )

    # Sort for plotting
    # total_evolution_times = 2 * np.array(taus) / 1e3
    inds = taus.argsort()
    total_evolution_times = 2 * np.array(taus)[inds] / 1e3
    norm_counts = np.array([norm_counts[nv_ind, inds] for nv_ind in range(num_nvs)])
    norm_counts_ste = np.array(
        [norm_counts_ste[nv_ind, inds] for nv_ind in range(num_nvs)]
    )

    do_fit = False
    if do_fit:
        fit_fns = []
        popts = []
        freq_guesses = [
            # 0
            (0.05, 0.09),
            # 1
            (0.048, 0.005),
            # 2
            (0.048, 0.099),
            # 3
            (),
            # 4
            (0.2, 0.248),
            # 5
            (0.048, 0.248),
            # 6
            (),
            # 7
            (),
            # 8
            (0.2, 0.248),
            # 9
            (),
        ]

        for nv_ind in range(num_nvs):
            nv_counts = norm_counts[nv_ind]
            nv_counts_ste = norm_counts_ste[nv_ind]
            # Contrast, revival period, quartic decay tc, amp1, amp2
            guess_params = [0.53, 75.5, 7, 0.4, 0.4]
            bounds = [[0, 73, 0, 0, 0], [1, 77, np.inf, 1, 1]]
            # guess_params = [75.5, 7, 0.4, 0.4]
            # bounds = [[73, 0, 0, 0], [77, np.inf, 1, 1]]

            # FFT to determine dominant frequency
            # even_counts = nv_counts[40:100] - 0.55
            # transform = np.fft.rfft(even_counts)
            # freqs = np.fft.rfftfreq(
            #     60, d=total_evolution_times[41] - total_evolution_times[40]
            # )
            # transform_mag = np.absolute(transform)
            # fig, ax = plt.subplots()
            # kpl.plot_points(ax, freqs[1:], transform_mag[1:])
            # ax.set_title(nv_ind)
            # kpl.show(block=True)

            try:
                freq_guess = freq_guesses[nv_ind]
                num_freqs = len(freq_guess)
                guess_params.extend(freq_guess)
                if num_freqs == 2:
                    fit_fn = quartic_decay_two_osc
                    bounds[0].extend([0, 0])
                    bounds[1].extend([np.inf, np.inf])
                elif num_freqs == 1:
                    fit_fn = quartic_decay_one_osc
                    bounds[0].extend([0])
                    bounds[1].extend([np.inf])
                else:
                    fit_fn = quartic_decay
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
                # popt[4] = popt[3]
            except Exception:
                # pass
                print(traceback.format_exc())
                fit_fn = None
                popt = None
            if nv_ind == 0:
                pass
                # popt = [0.557, 75.5, 6.279, 0.4, 0.4, 0.051]
                # popt = [0.557, 74.7, 6.279, 0.4, 0.4, 0.049]
            fit_fns.append(fit_fn)
            popts.append(popt)

            if fit_fn is not None:
                residuals = fit_fn(total_evolution_times, *popt) - nv_counts
                chi_sq = np.sum((residuals / nv_counts_ste) ** 2)
                red_chi_sq = chi_sq / (len(nv_counts) - len(popt))
                print(f"Red chi sq: {round(red_chi_sq, 3)}")

    ### Make the figure

    if axes_pack is None:
        figsize = [6.5, 5.0]
        figsize[0] *= 3
        figsize[1] *= 3
        fig, axes_pack, layout = kpl.subplot_mosaic(num_nvs, figsize=figsize)
    else:
        fig = None

    widefield.plot_fit(
        axes_pack,
        nv_list,
        total_evolution_times,
        norm_counts,
        norm_counts_ste,
        # fit_fns,
        # popts,
        no_legend=no_legend,
        # linestyle="solid",
    )
    ax = axes_pack[layout[-1, 0]]
    kpl.set_shared_ax_xlabel(ax, "Total evolution time (µs)")
    # kpl.set_shared_ax_ylabel(ax, "Change in $P($NV$^{-})$")
    # kpl.set_shared_ax_ylabel(ax, "Norm. NV$^{-}$ population")
    kpl.set_shared_ax_ylabel(ax, "Normalized NV$^{-}$ population")

    # figManager = plt.get_current_fig_manager()
    # figManager.window.showMaximized()

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


def main(nv_list, num_steps, num_reps, num_runs, min_tau=None, max_tau=None, taus=None):
    ### Some initial setup

    pulse_gen = tb.get_server_pulse_gen()
    seq_file = "spin_echo.py"

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

    # save data
    repr_nv_sig = widefield.get_repr_nv_sig(nv_list)
    repr_nv_name = repr_nv_sig.name
    file_path = dm.get_file_path(__file__, timestamp, repr_nv_name)
    dm.save_raw_data(raw_data, file_path)

    # creat fugure and save
    raw_fig = None
    try:
        # raw_fig = create_raw_data_figure(raw_data)
        fit_fig = create_fit_figure(raw_data)
    except Exception:
        print(traceback.format_exc())
        # raw_fig = None
        fit_fig = None

    ### Clean up and return
    tb.reset_cfm()
    kpl.show()

    if raw_fig is not None:
        dm.save_figure(raw_fig, file_path)
    if fit_fig is not None:
        file_path = dm.get_file_path(__file__, timestamp, repr_nv_name + "-fit")
        dm.save_figure(fit_fig, file_path)


if __name__ == "__main__":
    kpl.init_kplotlib()

    # data = dm.get_raw_data(file_id=1548381879624)

    file_ids = [1734158411844, 1734273666255]
    data = dm.get_raw_data(file_id=file_ids[0])
    for file_id in file_ids[1:]:
        new_data = dm.get_raw_data(file_id=file_id)
        data["num_runs"] += new_data["num_runs"]
        data["counts"] = np.append(data["counts"], new_data["counts"], axis=1)

    # create_raw_data_figure(data)
    create_fit_figure(data)

    plt.show(block=True)
