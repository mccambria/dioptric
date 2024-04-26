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
from utils import common
from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import tool_belt as tb
from utils import widefield as widefield
from utils.constants import NVSig


def create_raw_data_figure(nv_list, taus, counts, counts_ste):
    fig, ax = plt.subplots()
    widefield.plot_raw_data(ax, nv_list, taus, counts, counts_ste)
    ax.set_xlabel("Pulse duration (ns)")
    ax.set_ylabel("Counts")
    return fig


def create_fit_figure(nv_list, taus, counts, counts_ste, norms):
    ### Do the fitting

    taus = np.array(taus)

    def cos_decay(tau, ptp_amp, freq, decay):
        amp = abs(ptp_amp) / 2
        envelope = np.exp(-tau / abs(decay)) * amp
        cos_part = np.cos((2 * np.pi * freq * tau))
        sign = np.sign(ptp_amp)
        return 1 + amp - sign * (envelope * cos_part)

    def constant(tau, norm):
        if isinstance(tau, list):
            return [norm] * len(tau)
        elif type(tau) == np.ndarray:
            return np.array([norm] * len(tau))
        else:
            return norm

    num_nvs = len(nv_list)
    tau_step = taus[1] - taus[0]
    num_steps = len(taus)

    fit_fns = []
    popts = []
    for nv_ind in range(num_nvs):
        nv_sig = nv_list[nv_ind]
        nv_counts = counts[nv_ind] / norms[nv_ind]
        nv_counts_ste = counts_ste[nv_ind] / norms[nv_ind]

        ptp_amp_guess = np.max(nv_counts) - np.min(nv_counts)
        if nv_sig.spin_flip:
            ptp_amp_guess *= -1
        transform = np.fft.rfft(nv_counts)
        freqs = np.fft.rfftfreq(num_steps, d=tau_step)
        transform_mag = np.absolute(transform)
        max_ind = np.argmax(transform_mag[1:])  # Exclude DC component
        freq_guess = freqs[max_ind + 1]
        guess_params = [ptp_amp_guess, freq_guess, 1000]
        fit_fn = cos_decay

        try:
            popt, pcov = curve_fit(
                fit_fn,
                taus,
                nv_counts,
                p0=guess_params,
                sigma=nv_counts_ste,
                absolute_sigma=True,
            )
            fit_fns.append(fit_fn)
            popts.append(popt)
        except Exception:
            fit_fns.append(None)
            popts.append(None)
            continue

        residuals = fit_fn(taus, *popt) - nv_counts
        chi_sq = np.sum((residuals / nv_counts_ste) ** 2)
        red_chi_sq = chi_sq / (len(nv_counts) - len(popt))
        print(f"Red chi sq: {round(red_chi_sq, 3)}")

    rabi_periods = [
        round(1 / el[2], 2) for el in popts if el is not None and len(el) > 1
    ]
    print(f"rabi_periods: {rabi_periods}")

    ### Make the figure

    layout = kpl.calc_mosaic_layout(num_nvs)
    fig, axes_pack = plt.subplot_mosaic(
        layout, figsize=[6.5, 6.0], sharex=True, sharey=True
    )
    axes_pack = list(axes_pack.values())

    norm_counts = np.array([counts[ind] / norms[ind] for ind in range(num_nvs)])
    norm_counts_ste = np.array([counts_ste[ind] / norms[ind] for ind in range(num_nvs)])
    widefield.plot_fit(
        axes_pack,
        nv_list,
        taus,
        norm_counts,
        norm_counts_ste,
        fit_fns,
        popts,
        xlim=[0, None],
    )
    ax = axes_pack[-2]
    ax.set_xlabel(" ")
    fig.text(0.55, 0.01, "Pulse duration (ns)", ha="center")
    ax.set_ylabel(" ")
    fig.text(0.01, 0.55, "Normalized fluorescence", va="center", rotation="vertical")
    # ax.set_ylim([0.966, 1.24])
    # ax.set_yticks([1.0, 1.2])
    return fig


def create_correlation_figure(nv_list, taus, counts):
    ### Make the figure

    # fig, ax = plt.subplots()
    fig, axes_pack = plt.subplots(
        nrows=5, ncols=5, sharex=True, sharey=True, figsize=[10, 10]
    )

    widefield.plot_correlations(axes_pack, nv_list, taus, counts)

    ax = axes_pack[-1, 0]
    ax.set_xlabel(" ")
    fig.text(0.55, 0.01, "Pulse duration (ns)", ha="center")
    ax.set_ylabel(" ")
    fig.text(0.01, 0.55, "Correlation coefficient", va="center", rotation="vertical")
    return fig


def main(nv_list, num_steps, num_reps, num_runs, min_tau, max_tau, uwave_ind_list):
    ### Some initial setup

    pulse_gen = tb.get_server_pulse_gen()
    seq_file = "rabi.py"
    taus = np.linspace(min_tau, max_tau, num_steps)

    ### Collect the data

    def run_fn(shuffled_step_inds):
        seq_args = widefield.get_base_scc_seq_args(nv_list, uwave_ind_list)
        shuffled_taus = [taus[ind] for ind in shuffled_step_inds]
        seq_args.append(shuffled_taus)
        seq_args_string = tb.encode_seq_args(seq_args)
        pulse_gen.stream_load(seq_file, seq_args_string, num_reps)

    raw_data = base_routine.main(
        nv_list,
        num_steps,
        num_reps,
        num_runs,
        run_fn=run_fn,
        uwave_ind_list=uwave_ind_list,
    )

    ### Process and plot

    counts = raw_data["counts"]
    sig_counts = counts[0]
    ref_counts = counts[1]
    avg_counts, avg_counts_ste, norms = widefield.process_counts(
        nv_list, sig_counts, ref_counts
    )

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
        "min_tau": max_tau,
        "max_tau": max_tau,
    }

    repr_nv_sig = widefield.get_repr_nv_sig(nv_list)
    repr_nv_name = repr_nv_sig.name
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

    data = dm.get_raw_data(file_id=1500286727132)

    nv_list = data["nv_list"]
    nv_list = [NVSig(**nv) for nv in nv_list]
    num_nvs = len(nv_list)
    num_steps = data["num_steps"]
    num_runs = data["num_runs"]
    taus = data["taus"]
    counts = np.array(data["counts"])
    sig_counts = counts[0]
    ref_counts = counts[1]

    avg_counts, avg_counts_ste, norms = widefield.process_counts(
        nv_list, sig_counts, ref_counts, no_threshold=True
    )
    raw_fig = create_raw_data_figure(nv_list, taus, avg_counts, avg_counts_ste)
    fit_fig = create_fit_figure(nv_list, taus, avg_counts, avg_counts_ste, norms)

    plt.show(block=True)
