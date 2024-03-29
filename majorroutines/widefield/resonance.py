# -*- coding: utf-8 -*-
"""
Pulsed electron spin resonance on multiple NVs with spin-to-charge
conversion readout imaged onto a camera

Created on November 19th, 2023

@author: mccambria
"""

import os
import sys
import time
from random import shuffle

import matplotlib.pyplot as plt
import numpy as np

from majorroutines.pulsed_resonance import fit_resonance, voigt, voigt_split
from majorroutines.widefield import base_routine, optimize
from utils import common
from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import positioning as pos
from utils import tool_belt as tb
from utils import widefield as widefield
from utils.constants import NVSig, NVSpinState
from utils.positioning import get_scan_1d as calculate_freqs


def create_raw_data_figure(nv_list, freqs, counts, counts_errs):
    fig, ax = plt.subplots()
    widefield.plot_raw_data(ax, nv_list, freqs, counts, counts_errs)
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("Counts")
    return fig


def create_fit_figure(nv_list, freqs, counts, counts_ste, norms):
    # def create_fit_figure(nv_list, freqs, counts, counts_ste):
    ### Do the fitting

    num_nvs = len(nv_list)
    num_freqs = len(freqs)
    half_num_freqs = num_freqs // 2

    def constant(freq, norm):
        if isinstance(freq, list):
            return [norm] * len(freq)
        elif type(freq) == np.ndarray:
            return np.array([norm] * len(freq))
        else:
            return norm

    fit_fns = []
    popts = []
    # norms = []
    center_freqs = []

    for nv_ind in range(num_nvs):
        nv_counts = counts[nv_ind] / norms[nv_ind]
        nv_counts_ste = counts_ste[nv_ind] / norms[nv_ind]
        # nv_counts = counts[nv_ind]
        # nv_counts_ste = counts_ste[nv_ind]
        # norm_guess = np.median(nv_counts)
        # amp_guess = (np.max(nv_counts) - norm_guess) / norm_guess
        amp_guess = np.max(nv_counts) - 1

        # if nv_ind in [1]:
        #     num_resonances = 0
        # # elif nv_ind in [0, 1, 2, 4, 5, 7, 8, 9]:
        # #     num_resonances = 1
        # else:
        #     num_resonances = 2
        num_resonances = 2

        if num_resonances == 0:
            guess_params = []
            bounds = [[0], [np.inf]]
            fit_fn = constant
        elif num_resonances == 1:
            guess_params = [amp_guess, 5, 5, np.median(freqs)]
            bounds = [[0] * 4, [np.inf] * 4]
            # Limit linewidths
            for ind in [1, 2]:
                bounds[1][ind] = 10

            def fit_fn(freq, contrast, g_width, l_width, center):
                return 1 + voigt(freq, contrast, g_width, l_width, center)
        elif num_resonances == 2:
            # low_freq_guess = freqs[num_steps * 1 // 3]
            # high_freq_guess = freqs[num_steps * 2 // 3]

            # Find peaks in left and right halves
            low_freq_guess = freqs[np.argmax(nv_counts[:half_num_freqs])]
            high_freq_guess = freqs[
                np.argmax(nv_counts[half_num_freqs:]) + half_num_freqs
            ]

            # low_freq_guess = 2.85
            # high_freq_guess = 2.89

            guess_params = [
                amp_guess,
                5,
                5,
                low_freq_guess,
                amp_guess,
                5,
                5,
                high_freq_guess,
            ]
            num_params = len(guess_params)
            bounds = [[0] * num_params, [np.inf] * num_params]
            # Limit linewidths
            for ind in [1, 2, 5, 6]:
                bounds[1][ind] = 10

            def fit_fn(
                freq,
                contrast1,
                g_width1,
                l_width1,
                center1,
                contrast2,
                g_width2,
                l_width2,
                center2,
            ):
                # norm = 1
                return (
                    1
                    + voigt(freq, contrast1, g_width1, l_width1, center1)
                    + voigt(freq, contrast2, g_width2, l_width2, center2)
                )

        _, popt, pcov = fit_resonance(
            freqs,
            nv_counts,
            nv_counts_ste,
            fit_func=fit_fn,
            guess_params=guess_params,
            bounds=bounds,
        )

        # Tracking for plotting
        fit_fns.append(fit_fn)
        popts.append(popt)
        # norms.append(popt[-1])

        if num_resonances == 1:
            center_freqs.append(popt[4])
        elif num_resonances == 2:
            center_freqs.append((popt[3], popt[7]))

    print(center_freqs)

    ### Make the figured

    # nrows = 3
    # ncols = 5
    # fig, axes_pack = plt.subplots(
    #     nrows=nrows, ncols=ncols, sharex=True, sharey=True, figsize=[6.5, 6.0]
    # )

    layout = np.array(
        [
            ["a", "b", "c", ".", "."],
            ["f", "g", "h", "d", "e"],
            ["k", "l", "m", "i", "j"],
        ]
    )
    fig, axes_pack = plt.subplot_mosaic(
        layout, figsize=[6.5, 6.0], sharex=True, sharey=True
    )

    # axes_pack_flat = axes_pack.flatten()
    axes_pack_flat = list(axes_pack.values())
    norms = np.array(norms)

    widefield.plot_fit(
        axes_pack_flat, nv_list, freqs, counts, counts_ste, fit_fns, popts
    )

    ax = axes_pack[layout[-1, 0]]
    ax.set_xlabel(" ")
    fig.text(0.55, 0.01, "Frequency (GHz)", ha="center")
    ax.set_ylabel(" ")
    fig.text(0.01, 0.55, "Normalized fluorescence", va="center", rotation="vertical")
    # ax.set_ylim([0.945, 1.19])
    # ax.set_yticks([1.0, 1.1, 1.2])
    # ax.set_xticks([2.83, 2.87, 2.91])
    x_buffer = 0.05 * (np.max(freqs) - np.min(freqs))
    ax.set_xlim(np.min(freqs) - x_buffer, np.max(freqs) + x_buffer)
    norm_counts = counts / norms[:, np.newaxis]
    y_buffer = 0.05 * (np.max(norm_counts) - np.min(norm_counts))
    ax.set_ylim(np.min(norm_counts) - y_buffer, np.max(norm_counts) + y_buffer)
    return fig


def main(
    nv_list: list[NVSig],
    num_steps,
    num_reps,
    num_runs,
    freq_center,
    freq_range,
    uwave_ind=0,
):
    ### Some initial setup

    pulse_gen = tb.get_server_pulse_gen()
    sig_gen = tb.get_server_sig_gen()
    freqs = calculate_freqs(freq_center, freq_range, num_steps)

    seq_file = "resonance_ref.py"

    ### Collect the data

    def run_fn(step_inds):
        seq_args = widefield.get_base_scc_seq_args(nv_list)
        seq_args.append(uwave_ind)
        shuffled_freqs = [freqs[step_ind] for step_ind in step_inds]
        seq_args.append(shuffled_freqs)
        # MCC
        # if 2.835 < freq < 2.905:
        #     uwave_duration = 96 // 2
        # else:
        #     uwave_duration = 112 // 2
        # seq_args.append(uwave_duration)
        seq_args_string = tb.encode_seq_args(seq_args)
        pulse_gen.stream_load(seq_file, seq_args_string, num_reps)

    def step_fn(step_ind):
        freq = freqs[step_ind]
        sig_gen.set_freq(freq)

    counts, raw_data = base_routine.main(
        nv_list, num_steps, num_reps, num_runs, run_fn, step_fn, uwave_ind=uwave_ind
    )

    ### Process and plot

    sig_counts = counts[0]
    ref_counts = counts[1]

    avg_counts, avg_counts_ste, norms = widefield.average_counts(sig_counts, ref_counts)
    raw_fig = create_raw_data_figure(nv_list, freqs, avg_counts, avg_counts_ste)
    try:
        fit_fig = create_fit_figure(nv_list, freqs, avg_counts, avg_counts_ste, norms)
    except Exception as exc:
        print(exc)
        fit_fig = None

    ### Clean up and return

    tb.reset_cfm()
    kpl.show()

    timestamp = dm.get_time_stamp()
    raw_data |= {
        "timestamp": timestamp,
        "freqs": freqs,
        "freq-units": "GHz",
        "freq_range": freq_range,
        "freq_center": freq_center,
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

    # file_name = "2023_12_06-06_51_41-johnson-nv0_2023_12_04"
    # data = dm.get_raw_data(file_name)
    data = dm.get_raw_data(file_id=1470392816628, no_npz=True)  # Movie data

    nv_list = data["nv_list"]
    num_nvs = len(nv_list)
    num_steps = data["num_steps"]
    num_runs = data["num_runs"]
    num_reps = data["num_reps"]
    freqs = data["freqs"]

    # New style counts
    # counts = np.array(data["counts"])
    # sig_counts = counts[0]
    # ref_counts = counts[1]

    # Old style counts
    sig_counts = np.array(data["counts"])

    thresholds = [
        41.5,
        37.5,
        40.5,  # manual, 55.5 from bad fit
        44.5,
        37.5,
        40.5,
        41.5,
        40.5,
        40.5,
        42.5,
        38.5,
        36.5,
        37.5,
    ]
    temp_list = []
    for ind in range(num_nvs):
        nv_temp = nv_list[ind]
        threshold = thresholds[ind]
        threshold = None
        nv = NVSig(name=nv_temp["name"], threshold=threshold)
        temp_list.append(nv)
    nv_list = temp_list

    # ind = 12
    # sig_counts = sig_counts[[ind]]
    # nv_list = [nv_list[ind]]

    avg_counts, avg_counts_ste = widefield.process_counts(nv_list, sig_counts)
    # avg_counts, avg_counts_ste, norms = widefield.process_counts(
    #     nv_list, sig_counts, ref_counts
    # )
    # print(
    #     (max(avg_counts[0]) - np.median(avg_counts[0])) / np.median(avg_counts_ste[0])
    # )

    snrs = []
    for ind in range(num_nvs):
        snrs.append(
            (max(avg_counts[ind]) - np.median(avg_counts[ind]))
            / np.median(avg_counts_ste[ind])
        )
    print(np.mean(snrs))

    raw_fig = create_raw_data_figure(nv_list, freqs, avg_counts, avg_counts_ste)
    # fit_fig = create_fit_figure(nv_list, freqs, avg_counts, avg_counts_ste, norms)

    # img_arrays = np.array(data["img_arrays"])
    # img_arrays = np.mean(img_arrays, axis=0)
    # bottom = np.percentile(img_arrays, 30, axis=0)
    # img_arrays -= bottom

    # norms = norms[:, np.newaxis]
    # widefield.animate(
    #     freqs, nv_list, avg_counts / norms, avg_counts_ste / norms, img_arrays, 0, 3
    # )

    kpl.show(block=True)
