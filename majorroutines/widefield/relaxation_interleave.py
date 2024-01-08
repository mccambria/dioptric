# -*- coding: utf-8 -*-
"""
Widefield Rabi experiment

Created on November 29th, 2023

@author: mccambria
"""


import sys
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from majorroutines.widefield import base_routine
from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import tool_belt as tb
from utils import widefield as widefield
from utils.constants import NVSpinState


def process_rates(
    omega_exp_rates, omega_exp_rate_errs, gamma_exp_rates, gamma_exp_rate_errs
):
    num_nvs = len(omega_exp_rates)
    for ind in range(num_nvs):
        omega = omega_exp_rates[ind] / 3
        omega_err = omega_exp_rate_errs[ind] / 3
        print(
            f"Omega (x 10^3 s^-1): {tb.round_for_print(omega / 1000, omega_err / 1000)}"
        )

        gamma = (gamma_exp_rates[ind] - omega) / 2
        gamma_err = np.sqrt(gamma_exp_rate_errs[ind] ** 2 + omega_err**2) / 2
        print(
            f"gamma (x 10^3 s^-1): {tb.round_for_print(gamma / 1000, gamma_err / 1000)}"
        )
        print(round(gamma))
        print(round(gamma_err))


def create_raw_data_figure(
    nv_list, taus, counts, counts_ste, init_state, readout_state
):
    fig, ax = plt.subplots()
    taus_ms = np.array(taus) / 1e6
    widefield.plot_raw_data(ax, nv_list, taus_ms, counts, counts_ste)
    ax.set_xlabel("Relaxation time (ms)")
    ax.set_ylabel("Counts")
    state_str_dict = {
        NVSpinState.ZERO: "$\ket{0}$",
        NVSpinState.LOW: "$\ket{-1}$",
        NVSpinState.HIGH: "$\ket{+1}$",
    }
    init_state_str = state_str_dict[init_state]
    readout_state_str = state_str_dict[readout_state]
    ax.set_title(
        f"Initial state: {init_state_str}; readout state: {readout_state_str}",
        usetex=True,
    )
    return fig


def create_fit_figure(
    nv_list, taus, diff_counts, diff_counts_ste, Omega_or_gamma, nv1_norm=None
):
    # Do the fits

    taus_ms = np.array(taus) / 1e6

    def exp_decay(tau_ms, norm, rate):
        # def exp_decay(tau_ms, norm, rate, offset):
        offset = 0
        return norm * ((1 - offset) * np.exp(-rate * tau_ms / 1000) + offset)

    # def exp_decay(tau_ms, norm, decay, offset):
    #     return norm * ((1 - offset) * np.exp(-tau_ms / (1000 * decay)) + offset)

    def constant(tau_ms, norm):
        if isinstance(tau_ms, list):
            return [norm] * len(tau_ms)
        elif type(tau_ms) == np.ndarray:
            return np.array([norm] * len(tau_ms))
        else:
            return norm

    num_nvs = len(nv_list)

    fit_fns = []
    popts = []
    norms = []
    rates = []
    rate_errs = []
    offsets = []
    offset_errs = []
    for nv_ind in range(num_nvs):
        nv_counts = diff_counts[nv_ind]
        nv_counts_ste = diff_counts_ste[nv_ind]

        if nv_ind in [1]:
            fit_fn = constant
            guess_params = [np.average(nv_counts)]
        else:
            fit_fn = exp_decay
            # guess_params = [nv_counts[0], 70, 0]
            guess_params = [nv_counts[0], 70]
            # guess_params = [nv_counts[0], 0.01, 0]

        try:
            popt, pcov = curve_fit(
                fit_fn,
                taus_ms,
                nv_counts,
                p0=guess_params,
                sigma=nv_counts_ste,
                absolute_sigma=True,
            )
            fit_fns.append(fit_fn)
            popts.append(popt)
            pste = np.sqrt(np.diag(pcov))
            if nv_ind == 1:
                norms.append(nv1_norm)
            else:
                norms.append(popt[0])
                rates.append(popt[1])
                rate_errs.append(pste[1])
                # offsets.append(popt[2])
                # offset_errs.append(pste[2])
                # rate = 1 / popt[1]
                # rates.append(rate)
                # rate_errs.append(rate**2 * pste[1])
        except Exception as exc:
            fit_fns.append(None)
            popts.append(None)
            norms.append(None)

        residuals = fit_fn(taus_ms, *popt) - nv_counts
        chi_sq = np.sum((residuals / nv_counts_ste) ** 2)
        red_chi_sq = chi_sq / (len(nv_counts) - len(popt))
        print(f"Red chi sq: {round(red_chi_sq, 3)}")

    # Print the rates out
    print("rates")
    print(rates)
    print(rate_errs)
    # print("offsets")
    # print(offsets)
    # print(offset_errs)

    # Make the figure
    # fig, axes_pack = plt.subplots(nrows=6, sharex=True, figsize=[6.5, 6.0])
    fig, axes_pack = plt.subplots(
        nrows=3, ncols=2, sharex=True, sharey=True, figsize=[6.5, 6.0]
    )
    axes_pack = axes_pack.flatten()
    norms = None
    widefield.plot_fit(
        axes_pack,
        nv_list,
        taus_ms,
        diff_counts,
        diff_counts_ste,
        fit_fns,
        popts,
        norms,
        # skip_inds=[1],
    )

    axes_pack[-1].set_xlabel(" ")
    fig.text(0.55, 0.01, "Relaxation time (ms)", ha="center")
    axes_pack[0].set_ylabel(" ")
    fig.text(
        0.01,
        0.55,
        "Normalized fluorescence difference",
        va="center",
        rotation="vertical",
    )

    # axes_pack[-1].set_xlabel("Relaxation time (ms)")
    # ylabel = (
    #     "$F_{\Omega}$ (arb. units)" if Omega_or_gamma else "$F_{\gamma}$ (arb. units)"
    # )
    # ylabel = "Normalized fluorescence"
    # axes_pack[2].set_ylabel(ylabel)
    # axes_pack[-1].set_ylabel("Counts")
    # for ind in range(len(axes_pack)):
    #     ax = axes_pack[ind]
    #     if ind == 5:
    #         # ax.set_ylim([-1.2, +1.2])
    #         # ax.set_yticks([-1, 0, +1])
    #         ax.set_ylim([-0.8, +0.8])
    #         ax.set_yticks([-0.5, 0, +0.5])
    #     else:
    #         ax.set_ylim([-0.3, 1.35])
    #         ax.set_yticks([0, 1])
    ax = axes_pack[0]
    # ax.set_ylim([-0.253, 1.276])
    # ax.set_yticks([0, 1])
    return fig


def sq_relaxation(nv_list, num_steps, num_reps, num_runs, min_tau, max_tau):
    init_state_1 = NVSpinState.ZERO
    readout_state_1 = NVSpinState.ZERO
    init_state_2 = NVSpinState.ZERO
    readout_state_2 = NVSpinState.LOW
    base_args = [nv_list, num_steps, num_reps, num_runs, min_tau, max_tau]
    return main(
        *base_args, init_state_1, readout_state_1, init_state_2, readout_state_2
    )


def dq_relaxation(nv_list, num_steps, num_reps, num_runs, min_tau, max_tau):
    init_state_1 = NVSpinState.LOW
    readout_state_1 = NVSpinState.LOW
    init_state_2 = NVSpinState.LOW
    readout_state_2 = NVSpinState.HIGH
    base_args = [nv_list, num_steps, num_reps, num_runs, min_tau, max_tau]
    return main(
        *base_args, init_state_1, readout_state_1, init_state_2, readout_state_2
    )


def main(
    nv_list,
    num_steps,
    num_reps,
    num_runs,
    min_tau,
    max_tau,
    init_state_0=NVSpinState.ZERO,
    readout_state_0=NVSpinState.ZERO,
    init_state_1=NVSpinState.ZERO,
    readout_state_1=NVSpinState.LOW,
):
    ### Some initial setup

    pulse_gen = tb.get_server_pulse_gen()
    seq_file = "relaxation_interleave.py"

    # Get taus with a roughly even spacing on the y axis
    taus = np.geomspace(1 / num_steps, 1, num_steps)
    taus = (taus - taus[0]) / (taus[-1] - taus[0])  # Normalize to 0 to 1
    taus = (taus * (max_tau - min_tau)) + min_tau  # Normalize to mix/max tau
    taus = (taus // 4) * 4
    # taus = np.linspace(min_tau, max_tau, num_steps)

    # tau = taus[10]
    # seq_args = widefield.get_base_scc_seq_args(nv_list)
    # seq_args.extend(
    #     [tau, init_state_0, readout_state_0, init_state_1, readout_state_1]
    # )
    # seq_args_string = tb.encode_seq_args(seq_args)
    # print(seq_args)
    # print(seq_args_string)
    # return

    ### Collect the data

    def step_fn(tau_ind):
        tau = taus[tau_ind]
        seq_args = widefield.get_base_scc_seq_args(nv_list)
        seq_args.extend(
            [tau, init_state_0, readout_state_0, init_state_1, readout_state_1]
        )
        seq_args_string = tb.encode_seq_args(seq_args)
        pulse_gen.stream_load(seq_file, seq_args_string, num_reps)

    counts, raw_data = base_routine.main(
        nv_list,
        num_steps,
        num_reps,
        num_runs,
        step_fn,
        uwave_ind=[0, 1],
        num_exps_per_rep=2,
    )
    counts_0 = counts[0]
    counts_1 = counts[1]

    ### Process and plot

    avg_counts_0, avg_counts_0_ste = widefield.process_counts(counts_0)
    avg_counts_1, avg_counts_1_ste = widefield.process_counts(counts_1)

    raw_fig_0 = create_raw_data_figure(
        nv_list, taus, avg_counts_0, avg_counts_0_ste, init_state_0, readout_state_0
    )
    raw_fig_1 = create_raw_data_figure(
        nv_list, taus, avg_counts_1, avg_counts_1_ste, init_state_1, readout_state_1
    )

    ### Clean up and return

    tb.reset_cfm()
    kpl.show()

    timestamp = dm.get_time_stamp()
    raw_data |= {
        "timestamp": timestamp,
        "taus": taus,
        "tau-units": "ns",
        "min_tau": max_tau,
        "max_tau": max_tau,
        "init_state_0": init_state_0,
        "readout_state_0": readout_state_0,
        "init_state_1": init_state_1,
        "readout_state_1": readout_state_1,
    }

    repr_nv_sig = widefield.get_repr_nv_sig(nv_list)
    repr_nv_name = repr_nv_sig["name"]
    file_path = dm.get_file_path(__file__, timestamp, repr_nv_name)
    if "img_arrays" in raw_data:
        keys_to_compress = ["img_arrays"]
    else:
        keys_to_compress = None
    dm.save_raw_data(raw_data, file_path, keys_to_compress)
    file_path = dm.get_file_path(__file__, timestamp, repr_nv_name + "-0")
    dm.save_figure(raw_fig_0, file_path)
    file_path = dm.get_file_path(__file__, timestamp, repr_nv_name + "-1")
    dm.save_figure(raw_fig_1, file_path)


if __name__ == "__main__":
    ### Rate calculation
    # No offset
    # omega_exp_rates = [
    #     149.52876293233442,
    #     149.9076496706543,
    #     163.90784727283057,
    #     171.12691936163787,
    #     140.68822177145015,
    # ]
    # omega_exp_rate_errs = [
    #     13.071035712207378,
    #     11.193227235059075,
    #     13.205640033491687,
    #     19.12327121476871,
    #     13.316228407825507,
    # ]
    # gamma_exp_rates = [
    #     288.9371794219599,
    #     256.4689961920107,
    #     260.8659957292199,
    #     232.74941634806152,
    #     208.3639073269649,
    # ]
    # gamma_exp_rate_errs = [
    #     32.133818940701744,
    #     22.25996376863929,
    #     27.76090260093859,
    #     39.0883152024931,
    #     28.414155668518752,
    # ]
    # Offset
    # omega_exp_rates = [
    #     178.65927076759806,
    #     150.55961333102456,
    #     182.7383790247447,
    #     221.49601578708612,
    #     195.70649129132005,
    # ]
    # omega_exp_rate_errs = [
    #     32.42518984990736,
    #     25.278672433548234,
    #     29.12756202583012,
    #     45.32073245075222,
    #     38.16464756629071,
    # ]
    # gamma_exp_rates = [
    #     322.2588560877828,
    #     274.8856110109759,
    #     242.36811328021352,
    #     307.08797888419207,
    #     328.985126558155,
    # ]
    # gamma_exp_rate_errs = [
    #     57.36359787667976,
    #     41.97828859002894,
    #     46.743780796640415,
    #     91.47514044528685,
    #     82.42819543109252,
    # ]
    # process_rates(
    #     omega_exp_rates, omega_exp_rate_errs, gamma_exp_rates, gamma_exp_rate_errs
    # )
    # sys.exit()

    kpl.init_kplotlib()

    # file_name = ""
    # data = dm.get_raw_data(file_name)
    # data = dm.get_raw_data(file_id=1396784795732, no_npz=True)  # Omega
    # data = dm.get_raw_data(file_id=1396928132593, no_npz=True)  # gamma
    data = dm.get_raw_data(file_id=1407502794886, no_npz=True)  # Omega

    nv_list = data["nv_list"]
    num_nvs = len(nv_list)
    img_arrays = data["img_arrays"]
    num_steps = data["num_steps"]
    num_runs = data["num_runs"]
    # avg_img_arrays = np.average(img_arrays, axis=1)
    taus = data["taus"]
    counts = np.array(data["counts"])
    ref_counts = np.array(data["ref_counts"])
    init_state_0 = NVSpinState(data["init_state_0"])
    readout_state_0 = NVSpinState(data["readout_state_0"])
    init_state_1 = NVSpinState(data["init_state_1"])
    readout_state_1 = NVSpinState(data["readout_state_1"])

    avg_counts_0, avg_counts_ste_0, norms_0 = widefield.process_counts(
        counts[0], ref_counts
    )
    raw_fig = create_raw_data_figure(
        nv_list, taus, avg_counts_0, avg_counts_ste_0, init_state_0, readout_state_0
    )
    avg_counts_1, avg_counts_ste_1, norms_1 = widefield.process_counts(
        counts[1], ref_counts
    )
    raw_fig = create_raw_data_figure(
        nv_list, taus, avg_counts_1, avg_counts_ste_1, init_state_1, readout_state_1
    )

    # Calculate the differences and make the fit plot
    diff_counts = avg_counts_1 / norms_1 - avg_counts_0 / norms_0
    diff_counts_ste = np.sqrt(
        (avg_counts_ste_0 / norms_0) ** 2 + (avg_counts_ste_1 / norms_1) ** 2
    )
    Omega_or_gamma = (
        init_state_0 == NVSpinState.ZERO
        and readout_state_0 == NVSpinState.ZERO
        or init_state_1 == NVSpinState.ZERO
        and readout_state_1 == NVSpinState.ZERO
    )
    nv1_norm = np.mean(avg_counts_0[1] + avg_counts_1[1]) / 2
    fit_fig = create_fit_figure(
        nv_list, taus, diff_counts, diff_counts_ste, Omega_or_gamma, nv1_norm
    )

    plt.show(block=True)
