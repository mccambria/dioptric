# -*- coding: utf-8 -*-
"""
Widefield Rabi experiment

Created on November 29th, 2023

@author: mccambria
"""


from random import shuffle
import sys
import matplotlib.pyplot as plt
import numpy as np
from majorroutines.widefield import optimize
from utils import tool_belt as tb
from utils import data_manager as dm
from utils import common
from utils import widefield as widefield
from utils.constants import LaserKey
from utils import kplotlib as kpl
from utils import positioning as pos
from utils import data_manager as dm
from utils.constants import NVSpinState
import os
import time
from utils.positioning import get_scan_1d as calculate_freqs
from scipy.optimize import curve_fit
from majorroutines.widefield import base_routine


def create_raw_data_figure(nv_list, taus, counts, counts_ste):
    fig, ax = plt.subplots()
    widefield.plot_raw_data(ax, nv_list, taus, counts, counts_ste)
    ax.set_xlabel("Pulse duration (ns)")
    ax.set_ylabel("Counts")
    return fig


def create_fit_figure(nv_list, taus, counts, counts_ste):
    ### Do the fitting

    def cos_decay(tau, norm, ptp_amp, freq, decay):
        amp = abs(ptp_amp) / 2
        envelope = np.exp(-tau / abs(decay)) * amp
        cos_part = np.cos((2 * np.pi * freq * tau))
        return abs(norm) + amp - (envelope * cos_part)

    def constant(tau, norm):
        if type(tau) == list:
            return [norm] * len(tau)
        elif type(tau) == np.ndarray:
            return np.array([norm] * len(tau))
        else:
            return norm

    num_nvs = len(nv_list)
    tau_step = taus[1] - taus[0]
    num_steps = len(taus)

    a0_list = []
    a1_list = []
    readout_noise_list = []

    fit_fns = []
    popts = []
    norms = []
    for nv_ind in range(num_nvs):
        nv_counts = counts[nv_ind]
        nv_counts_ste = counts_ste[nv_ind]

        if nv_ind in [1, 2, 3, 4, 8]:
            # Estimate fit parameters
            norm_guess = np.min(nv_counts)
            ptp_amp_guess = np.max(nv_counts) - norm_guess
            transform = np.fft.rfft(nv_counts)
            freqs = np.fft.rfftfreq(num_steps, d=tau_step)
            transform_mag = np.absolute(transform)
            max_ind = np.argmax(transform_mag[1:])  # Exclude DC component
            freq_guess = freqs[max_ind + 1]
            guess_params = [norm_guess, ptp_amp_guess, freq_guess, 1000]
            fit_fn = cos_decay
        else:
            fit_fn = constant
            guess_params = [np.average(nv_counts)]

        popt, pcov = curve_fit(
            fit_fn,
            taus,
            nv_counts,
            p0=guess_params,
            sigma=nv_counts_ste,
            absolute_sigma=True,
        )

        norm = popt[0]
        if len(popt) > 1:
            # SCC readout noise tracking
            contrast = popt[1]
            a0 = round((1 + contrast) * norm, 2)
            a1 = round(norm, 2)
            print(f"ms=+/-1: {a0}\nms=0: {a1}\n")
            a0_list.append(a0)
            a1_list.append(a1)
            readout_noise_list.append(np.sqrt(1 + 2 * (a0 + a1) / ((a0 - a1) ** 2)))

        # Tracking for plotting
        fit_fns.append(fit_fn)
        popts.append(popt)
        norms.append(norm)

    print(f"a0 average: {round(np.average(a0_list), 2)}")
    print(f"a1 average: {round(np.average(a1_list), 2)}")
    print(f"Average readout noise: {round(np.average(readout_noise_list), 2)}")
    print(f"Median readout noise: {round(np.median(readout_noise_list), 2)}")

    ### Make the figure

    fig, ax = plt.subplots()
    widefield.plot_fit(ax, nv_list, taus, counts, counts_ste, fit_fns, popts, norms)
    ax.set_xlabel("Pulse duration (ns)")
    ax.set_ylabel("Normalized fluorescence")
    return fig


def main(
    nv_list, uwave_list, uwave_ind, min_tau, max_tau, num_steps, num_reps, num_runs
):
    ### Some initial setup

    cxn = common.labrad_connect()

    pulse_gen = tb.get_server_pulse_gen(cxn)
    seq_file = "resonance.py"
    taus = np.linspace(min_tau, max_tau, num_steps)
    sig_gen = tb.get_server_sig_gen(cxn, uwave_ind)
    sig_gen_name = sig_gen.name

    ### Collect the data

    def step_fn(tau_ind):
        tau = taus[tau_ind]
        seq_args = widefield.get_base_scc_seq_args(nv_list)
        seq_args.extend([sig_gen_name, tau])
        seq_args_string = tb.encode_seq_args(seq_args)
        pulse_gen.stream_load(seq_file, seq_args_string, num_reps)

    counts, raw_data = base_routine.main(
        cxn, nv_list, uwave_list, num_steps, num_reps, num_runs, step_fn
    )

    ### Process and plot

    avg_counts, avg_counts_ste = widefield.process_counts(counts)

    kpl.init_kplotlib()
    raw_fig = create_raw_data_figure(nv_list, taus, avg_counts, avg_counts_ste)
    try:
        fit_fig = create_fit_figure(nv_list, taus, avg_counts, avg_counts_ste)
    except Exception as exc:
        print(exc)
        fit_fig = None

    ### Clean up and return

    tb.reset_cfm(cxn)

    timestamp = dm.get_time_stamp()
    raw_data |= {
        "timestamp": timestamp,
        "nv_list": nv_list,
        "uwave_list": uwave_list,
        "uwave_ind": uwave_ind,
        "num_reps": num_reps,
        "num_steps": num_steps,
        "num_runs": num_runs,
        "readout-units": "ms",
        "taus": taus,
        "tau-units": "ns",
        "min_tau": max_tau,
        "max_tau": max_tau,
        "counts": counts,
        "counts-units": "photons",
        "img_arrays": img_arrays,
        "img_array-units": "ADUs",
    }

    repr_nv_sig = widefield.get_repr_nv_sig(nv_list)
    repr_nv_name = repr_nv_sig["name"]
    file_path = dm.get_file_path(__file__, timestamp, repr_nv_name)
    # keys_to_compress = ["sig_img_arrays", "ref_img_arrays"]
    keys_to_compress = ["img_arrays"]
    dm.save_raw_data(raw_data, file_path, keys_to_compress=keys_to_compress)
    dm.save_figure(raw_fig, file_path)
    if fit_fig is not None:
        file_path = dm.get_file_path(__file__, timestamp, repr_nv_name + "-fit")
        dm.save_figure(fit_fig, file_path)


if __name__ == "__main__":
    kpl.init_kplotlib()

    # file_name = ""
    # data = dm.get_raw_data(file_name)
    data = dm.get_raw_data(file_id=1382892086081)

    nv_list = data["nv_list"]
    num_nvs = len(nv_list)
    img_arrays = data["img_arrays"]
    num_steps = data["num_steps"]
    num_runs = data["num_runs"]
    avg_img_arrays = np.average(img_arrays, axis=1)
    taus = data["taus"]
    counts = np.array(data["counts"])

    fig, ax = plt.subplots()
    kpl.histogram(ax, counts[1, :, 6, :].flatten(), nbins=100)
    print(np.count_nonzero(counts[1, :, 6, :] < 65))
    print(np.count_nonzero(counts[1, :, 6, :] >= 65))

    avg_counts, avg_counts_ste = widefield.process_counts(counts)
    raw_fig = create_raw_data_figure(nv_list, taus, avg_counts, avg_counts_ste)
    fit_fig = create_fit_figure(nv_list, taus, avg_counts, avg_counts_ste)

    plt.show(block=True)

# p  *(a0**2 + a0) - p**2  *a0**2 + (1-p)  *(a1**2 + a1) - (1-p)**2  *a1**2 - 2  *p  *a0  *(1-p) *a1
# 1669/2400, 56, 98
# 1765/2400, 48, 80
# p  *(a0**2 + a0) - p**2  *a0**2 + (1-p)  *(a1**2 + a1) - (1-p)**2  *a1**2 - 2  *p  *a0  *(1-p) *a1
