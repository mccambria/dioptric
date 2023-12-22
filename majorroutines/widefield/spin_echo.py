# -*- coding: utf-8 -*-
"""
Widefield Rabi experiment

Created on November 29th, 2023

@author: mccambria
"""


import time
import matplotlib.pyplot as plt
import numpy as np
from utils import tool_belt as tb
from utils import data_manager as dm
from utils import widefield as widefield
from utils import kplotlib as kpl
from utils import data_manager as dm
from scipy.optimize import curve_fit
from majorroutines.widefield import base_routine


def quartic_decay(
    tau,
    norm,
    amplitude,
    revival_time,
    quartic_decay_time,
    envelope_decay_time,
    envelope_exponent,
):
    baseline = norm
    val = baseline
    # print(len(amplitudes))
    envelope = np.exp(-((tau / envelope_decay_time) ** envelope_exponent))
    num_revivals = 3
    for ind in range(num_revivals):
        exp_part = np.exp(-(((tau - ind * revival_time) / quartic_decay_time) ** 4))
        val -= amplitude * envelope * exp_part
    return val


def constant(tau, norm):
    if type(tau) == list:
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
#         ax.set_xlabel("Total evolution time (us)")
#         ax.set_ylabel("Counts")
#     return fig


def create_raw_data_figure(nv_list, taus, counts, counts_ste):
    fig, ax = plt.subplots()
    total_evolution_times = 2 * np.array(taus) / 1e3
    widefield.plot_raw_data(ax, nv_list, total_evolution_times, counts, counts_ste)
    ax.set_xlabel("Total evolution time (us)")
    ax.set_ylabel("Counts")
    return fig


def create_fit_figure(nv_list, taus, counts, counts_ste):
    total_evolution_times = 2 * np.array(taus) / 1e3

    fit_fns = []
    popts = []
    norms = []

    for nv_ind in range(len(nv_list)):
        nv_counts = counts[nv_ind]
        nv_counts_ste = counts_ste[nv_ind]

        try:
            if nv_ind != 1:
                fit_fn = quartic_decay
                norm_guess = np.min(nv_counts)
                amplitude_guess = np.quantile(nv_counts, 0.7) - norm_guess
                T2_guess = 10 if nv_ind == 1 else 200
                guess_params = [norm_guess, amplitude_guess, 175, 15, T2_guess, 1.5]
                popt, pcov = curve_fit(
                    fit_fn,
                    total_evolution_times,
                    nv_counts,
                    p0=guess_params,
                    sigma=nv_counts_ste,
                    absolute_sigma=True,
                    maxfev=10000,
                    bounds=((10, 0, 100, 5, 10, 1), (100, 100, 500, 30, 500, 2.5)),
                )
            else:
                fit_fn = constant
                popt, pcov = curve_fit(
                    fit_fn,
                    total_evolution_times,
                    nv_counts,
                    p0=[np.mean(nv_counts)],
                    sigma=nv_counts_ste,
                    absolute_sigma=True,
                )
            # popt = guess_params
            fit_fns.append(fit_fn)
            popts.append(popt)
            norms.append(popt[0])
        except Exception as exc:
            print(exc)
            fit_fns.append(None)
            popts.append(None)
            norms.append(None)

        residuals = fit_fn(total_evolution_times, *popt) - nv_counts
        chi_sq = np.sum((residuals / nv_counts_ste) ** 2)
        red_chi_sq = chi_sq / (len(nv_counts) - len(popt))
        print(f"Red chi sq: {round(red_chi_sq, 3)}")

    ### Make the figure

    fig, ax = plt.subplots()
    # offset = 0.10
    offset = 0.07
    widefield.plot_fit(
        ax,
        nv_list,
        total_evolution_times,
        counts,
        counts_ste,
        fit_fns,
        popts,
        norms,
        offset=offset,
    )
    ax.set_xlabel("Total evolution time (us)")
    ax.set_ylabel("Normalized fluorescence")
    return fig


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

    counts, raw_data = base_routine.main(
        nv_list, num_steps, num_reps, num_runs, step_fn
    )

    ### Process and plot

    avg_counts, avg_counts_ste = widefield.process_counts(counts)

    raw_fig = create_raw_data_figure(nv_list, taus, avg_counts, avg_counts_ste)
    try:
        fit_fig = create_fit_figure(nv_list, taus, avg_counts, avg_counts_ste)
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
        "min_tau": max_tau,
        "max_tau": max_tau,
    }

    repr_nv_sig = widefield.get_repr_nv_sig(nv_list)
    repr_nv_name = repr_nv_sig["name"]
    file_path = dm.get_file_path(__file__, timestamp, repr_nv_name)
    if "img_arrays" in raw_data:
        keys_to_compress = ["img_arrays"]
    else:
        keys_to_compress = None
    dm.save_raw_data(raw_data, file_path, keys_to_compress)
    dm.save_figure(raw_fig, file_path)
    if fit_fig is not None:
        file_path = dm.get_file_path(__file__, timestamp, repr_nv_name + "-fit")
        dm.save_figure(fit_fig, file_path)


if __name__ == "__main__":
    kpl.init_kplotlib()

    # file_name = ""
    # data = dm.get_raw_data(file_name)
    data = dm.get_raw_data(file_id=1395732527176)

    nv_list = data["nv_list"]
    num_nvs = len(nv_list)
    num_steps = data["num_steps"]
    num_runs = data["num_runs"]
    taus = data["taus"]
    counts = np.array(data["counts"])

    avg_counts, avg_counts_ste = widefield.process_counts(counts)
    raw_fig = create_raw_data_figure(nv_list, taus, avg_counts, avg_counts_ste)
    fit_fig = create_fit_figure(nv_list, taus, avg_counts, avg_counts_ste)

    plt.show(block=True)
