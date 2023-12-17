# -*- coding: utf-8 -*-
"""
Pulsed electron spin resonance on multiple NVs with spin-to-charge 
conversion readout imaged onto a camera

Created on November 19th, 2023

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
from utils import kplotlib as kpl
from utils import positioning as pos
from utils import data_manager as dm
from utils.constants import NVSpinState
import os
import time
from utils.positioning import get_scan_1d as calculate_freqs
from majorroutines.pulsed_resonance import fit_resonance, voigt_split, voigt
from majorroutines.widefield import base_routine


def create_raw_data_figure(nv_list, freqs, counts, counts_errs):
    fig, ax = plt.subplots()
    widefield.plot_raw_data(ax, nv_list, freqs, counts, counts_errs)
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("Counts")
    return fig


def create_fit_figure(nv_list, freqs, counts, counts_ste):
    ### Do the fitting

    num_nvs = len(nv_list)

    a0_list = []
    a1_list = []
    readout_noise_list = []

    fit_fns = []
    popts = []
    norms = []
    num_steps = len(freqs)

    center_freqs = []

    for nv_ind in range(num_nvs):
        nv_counts = counts[nv_ind]
        nv_counts_ste = counts_ste[nv_ind]
        norm_guess = np.median(nv_counts)
        amp_guess = (np.max(nv_counts) - norm_guess) / norm_guess

        single_resonance = False  # vs double resonance

        if single_resonance:
            guess_params = [norm_guess, amp_guess, 5, 5, np.median(freqs)]
            bounds = [[0] * 5, [np.inf] * 5]
            # Limit linewidths
            for ind in [2, 3]:
                bounds[1][ind] = 10
            fit_fn = lambda freq, norm, contrast, g_width, l_width, center: norm * (
                1 + voigt(freq, contrast, g_width, l_width, center)
            )
        else:  # Double
            low_freq_guess = freqs[num_steps * 1 // 3]
            high_freq_guess = freqs[num_steps * 2 // 3]
            if nv_ind == 8:
                low_freq_guess = 2.8164
                high_freq_guess = 2.93
            guess_params = [
                norm_guess,
                amp_guess,
                5,
                5,
                low_freq_guess,
                amp_guess,
                5,
                5,
                high_freq_guess,
            ]
            bounds = [[0] * 9, [np.inf] * 9]
            # Limit linewidths
            for ind in [2, 3, 6, 7]:
                bounds[1][ind] = 10
            fit_fn = (
                lambda freq, norm, contrast1, g_width1, l_width1, center1, contrast2, g_width2, l_width2, center2: norm
                * (
                    1
                    + voigt(freq, contrast1, g_width1, l_width1, center1)
                    + voigt(freq, contrast2, g_width2, l_width2, center2)
                )
            )

        _, popt, pcov = fit_resonance(
            freqs,
            nv_counts,
            nv_counts_ste,
            fit_func=fit_fn,
            guess_params=guess_params,
            bounds=bounds,
        )

        # SCC readout noise tracking
        norm = popt[0]
        contrast = popt[1]
        a0 = round((1 + contrast) * norm, 2)
        a1 = round(norm, 2)
        print(f"ms=+/-1: {a0}\nms=0: {a1}")
        a0_list.append(a0)
        a1_list.append(a1)
        readout_noise = np.sqrt(1 + 2 * (a0 + a1) / ((a0 - a1) ** 2))
        readout_noise_list.append(readout_noise)
        print(f"readout noise: {readout_noise}")
        print()

        # Tracking for plotting
        fit_fns.append(fit_fn)
        popts.append(popt)
        norms.append(popt[0])

        if single_resonance:
            center_freqs.append(popt[4])
        else:
            center_freqs.append((popt[4], popt[8]))

    print(f"a0 average: {round(np.average(a0_list), 2)}")
    print(f"a1 average: {round(np.average(a1_list), 2)}")
    print(f"Average readout noise: {round(np.average(readout_noise_list), 2)}")
    print(f"Median readout noise: {round(np.median(readout_noise_list), 2)}")
    r_readout_noise_list = [round(el, 2) for el in readout_noise_list]
    print(f"readout noise list: {r_readout_noise_list}")

    print(center_freqs)

    ### Make the figure

    fig, ax = plt.subplots()
    widefield.plot_fit(ax, nv_list, freqs, counts, counts_ste, fit_fns, popts, norms)
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("Normalized fluorescence")
    # ax.set_xlim(None, 3.01)
    return fig


def main(nv_list, num_steps, num_reps, num_runs, freq_center, freq_range, uwave_ind=0):
    ### Some initial setup

    pulse_gen = tb.get_server_pulse_gen()
    sig_gen = tb.get_server_sig_gen()
    freqs = calculate_freqs(freq_center, freq_range, num_steps)

    seq_file = "resonance.py"

    ### Collect the data

    def step_fn(freq_ind):
        freq = freqs[freq_ind]
        sig_gen.set_freq(freq)
        seq_args = widefield.get_base_scc_seq_args(nv_list)
        seq_args.append(uwave_ind)
        seq_args_string = tb.encode_seq_args(seq_args)
        pulse_gen.stream_load(seq_file, seq_args_string, num_reps)

    counts, raw_data = base_routine.main(
        nv_list, num_steps, num_reps, num_runs, step_fn, uwave_ind=uwave_ind
    )

    ### Process and plot

    avg_counts, avg_counts_ste = widefield.process_counts(counts)
    raw_fig = create_raw_data_figure(nv_list, freqs, avg_counts, avg_counts_ste)
    try:
        fit_fig = create_fit_figure(nv_list, freqs, avg_counts, avg_counts_ste)
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
    repr_nv_name = repr_nv_sig["name"]
    file_path = dm.get_file_path(__file__, timestamp, repr_nv_name)
    dm.save_raw_data(raw_data, file_path)
    dm.save_figure(raw_fig, file_path)
    if fit_fig is not None:
        file_path = dm.get_file_path(__file__, timestamp, repr_nv_name + "-fit")
        dm.save_figure(fit_fig, file_path)


if __name__ == "__main__":
    kpl.init_kplotlib()

    # file_name = "2023_12_06-06_51_41-johnson-nv0_2023_12_04"
    # data = dm.get_raw_data(file_name)
    # data = dm.get_raw_data(file_id=1388701699044)  # 90
    # data = dm.get_raw_data(file_id=1388679268107)  # 30
    # data = dm.get_raw_data(file_id=1388633807820)  # 0
    # data = dm.get_raw_data(file_id=1388633807820)  # large correlation
    # data = dm.get_raw_data(file_id=1389286042809)  # small correlation
    data = dm.get_raw_data(file_id=1390529504532)

    nv_list = data["nv_list"]
    num_nvs = len(nv_list)
    num_steps = data["num_steps"]
    num_runs = data["num_runs"]
    num_reps = data["num_reps"]
    freqs = data["freqs"]
    counts = np.array(data["counts"])

    # Spurious correlation testing
    # step_ind_master_list = np.array(data["step_ind_master_list"])
    # mean_inds = []
    # mean_corrs = []
    # mean_diffs = []
    # for step_ind in range(num_steps):
    #     step_inds = [el.tolist().index(step_ind) for el in step_ind_master_list]
    #     mean_inds.append(np.mean(step_inds))

    #     step_counts = [
    #         counts[nv_ind, :, step_ind, :].flatten()
    #         # for nv_ind in [1, 5]
    #         for nv_ind in range(num_nvs)
    #     ]
    #     corr = np.corrcoef(step_counts)
    #     mean_corrs.append(np.mean(corr, where=corr < 0.999))

    #     val = np.mean(
    #         [
    #             counts[nv_ind, :, step_ind, :] - np.mean(counts[nv_ind, :, step_ind, :])
    #             for nv_ind in range(num_nvs)
    #         ]
    #     )
    #     mean_diffs.append(val)
    # mean_corrs_runs = []
    # for run_ind in range(num_runs):
    #     run_counts = [
    #         counts[nv_ind, run_ind, :, :].flatten()
    #         # for nv_ind in [1, 5]
    #         for nv_ind in range(num_nvs)
    #     ]
    #     corr = np.corrcoef(run_counts)
    #     mean_corrs_runs.append(np.mean(corr, where=corr < 0.999))
    # print(mean_inds)
    # print([round(el, 3) for el in mean_corrs])
    # fig, ax = plt.subplots()
    # kpl.plot_points(ax, mean_inds, mean_corrs)
    # ax.set_xlabel("Mean step order")
    # # kpl.plot_points(ax, freqs, mean_corrs)
    # # kpl.plot_points(ax, range(num_steps), mean_corrs)
    # # ax.set_xlabel("Step index")
    # fig, ax = plt.subplots()
    # kpl.plot_points(ax, range(num_runs), mean_corrs_runs)
    # ax.set_xlabel("Run index")
    # # kpl.plot_points(ax, mean_inds, mean_diffs)

    # counts = counts[:, :, :, :5]
    avg_counts, avg_counts_ste = widefield.process_counts(counts)
    raw_fig = create_raw_data_figure(nv_list, freqs, avg_counts, avg_counts_ste)
    fit_fig = create_fit_figure(nv_list, freqs, avg_counts, avg_counts_ste)

    kpl.show(block=True)
