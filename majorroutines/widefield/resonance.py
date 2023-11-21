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
from utils.constants import LaserKey
from utils import kplotlib as kpl
from utils import positioning as pos
from utils import data_manager as dm
from utils.constants import NVSpinState
import os
import time
from utils.positioning import get_scan_1d as calculate_freqs
from majorroutines.pulsed_resonance import fit_resonance, voigt_split, voigt


def create_raw_data_figure(freqs, counts, counts_ste):
    num_nvs = counts.shape[0]
    fig, ax = plt.subplots()
    for ind in range(num_nvs):
        kpl.plot_points(ax, freqs, counts[ind], yerr=counts_ste[ind], label=ind)
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("Counts")
    min_freqs = min(freqs)
    max_freqs = max(freqs)
    excess = 0.08 * (max_freqs - min_freqs)
    ax.set_xlim(min_freqs - excess, max_freqs + excess)
    ax.legend(loc=kpl.Loc.LOWER_RIGHT)
    return fig


def create_fit_figure(freqs, counts, counts_ste, plot_residuals=False):
    num_nvs = counts.shape[0]
    fig, ax = plt.subplots()
    freq_linspace = np.linspace(min(freqs) - 0.001, max(freqs) + 0.001, 1000)
    offset_inds = [num_nvs - 1 - ind for ind in list(range(num_nvs))]
    shift_factor = 0.075
    # shuffle(offset_inds)
    for ind in range(num_nvs):
        nv_counts = counts[ind]
        nv_counts_ste = counts_ste[ind]
        guess_params = [nv_counts[0], 0.15, 2, 2, np.median(freqs)]
        fit_func = lambda freq, norm, contrast, g_width, l_width, center: norm * (
            1 + voigt(freq, contrast, g_width, l_width, center)
        )

        fit_func, popt, pcov = fit_resonance(
            freqs,
            nv_counts,
            nv_counts_ste,
            fit_func=fit_func,
            guess_params=guess_params,
        )
        pste = np.sqrt(np.diag(pcov))

        if plot_residuals:
            kpl.plot_points(
                ax,
                freqs,
                ((nv_counts - fit_func(freqs, *popt)) / nv_counts_ste),
                label=ind,
            )
        else:
            offset_ind = offset_inds[ind]
            norm = popt[0]
            kpl.plot_line(
                ax,
                freq_linspace,
                shift_factor * offset_ind + fit_func(freq_linspace, *popt) / norm,
            )
            kpl.plot_points(
                ax,
                freqs,
                shift_factor * offset_ind + nv_counts / norm,
                yerr=nv_counts_ste / norm,
                label=ind,
            )

        # Normalized residuals

        # Contrast in units of counts
        # contrast_counts = popt[0] * popt[1]
        # mean_err = np.mean(nv_counts_ste)
        # # print(contrast_counts)
        # print(mean_err)
        # print(contrast_counts / mean_err)
        # print(popt)
        # print(pste)
        # print()

    ax.set_xlabel("Frequency (GHz)")
    if plot_residuals:
        ax.set_ylabel("Normalized residuals")
    else:
        ax.set_ylabel("Normalized fluorescence")
    min_freqs = min(freqs)
    max_freqs = max(freqs)
    excess = 0.08 * (max_freqs - min_freqs)
    ax.set_xlim(min_freqs - excess, max_freqs + excess)
    ax.legend(loc=kpl.Loc.LOWER_RIGHT)
    return fig


def main(
    nv_list,
    freq_center,
    freq_range,
    num_steps,
    num_reps,
    num_runs,
    state=NVSpinState.LOW,
):
    with common.labrad_connect() as cxn:
        main_with_cxn(
            cxn,
            nv_list,
            freq_center,
            freq_range,
            num_steps,
            num_reps,
            num_runs,
            state,
        )


def main_with_cxn(
    cxn,
    nv_list,
    freq_center,
    freq_range,
    num_steps,
    num_reps,
    num_runs,
    state=NVSpinState.LOW,
):
    ### Some initial setup

    tb.reset_cfm(cxn)

    # First NV to represent the others
    repr_nv_ind = 0
    repr_nv_sig = nv_list[repr_nv_ind]
    pos.set_xyz_on_nv(cxn, repr_nv_sig)
    num_nvs = len(nv_list)

    camera = tb.get_server_camera(cxn)
    pulse_gen = tb.get_server_pulse_gen(cxn)
    sig_gen = tb.get_server_sig_gen(cxn, state)
    sig_gen_name = sig_gen.name

    freqs = calculate_freqs(freq_center, freq_range, num_steps)

    uwave_dict = repr_nv_sig[state]
    uwave_duration = tb.get_pi_pulse_dur(uwave_dict["rabi_period"])
    uwave_power = uwave_dict["uwave_power"]
    sig_gen.set_amp(uwave_power)

    ### Load the pulse generator

    seq_args = widefield.get_base_scc_seq_args(nv_list)
    seq_args.extend([sig_gen_name, uwave_duration])
    seq_args_string = tb.encode_seq_args(seq_args)
    seq_file = "resonance.py"

    # print(seq_args)
    # print(seq_file)
    # return

    ### Data tracking

    counts = np.empty((num_nvs, num_runs, num_steps, num_reps))
    img_arrays = np.empty((num_runs, num_steps), dtype=np.uint16)
    freq_ind_master_list = [[] for ind in range(num_runs)]
    freq_ind_list = list(range(0, num_steps))

    ### Collect the data

    pulse_gen.stream_load(seq_file, seq_args_string, num_reps)

    try:
        camera.arm()
        sig_gen.uwave_on()

        for run_ind in range(num_runs):
            shuffle(freq_ind_list)

            for freq_ind in freq_ind_list:
                pixel_coords_list = [
                    widefield.get_nv_pixel_coords(nv) for nv in nv_list
                ]

                freq_ind_master_list[run_ind].append(freq_ind)
                freq = freqs[freq_ind]
                sig_gen.set_freq(freq)

                pulse_gen.stream_start()

                for rep_ind in range(num_reps):
                    img_str = camera.read()
                    img_array = widefield.img_str_to_array(img_str)
                    if rep_ind == 0:
                        avg_img_array = np.copy(img_array)
                    else:
                        avg_img_array += img_array

                    for nv_ind in range(num_nvs):
                        pixel_coords = pixel_coords_list[nv_ind]
                        counts_val = widefield.integrate_counts_from_adus(
                            img_array, pixel_coords
                        )
                        counts[nv_ind, run_ind, freq_ind, rep_ind] = counts_val

                avg_img_array = avg_img_array / num_reps
                img_arrays[run_ind, freq_ind] = avg_img_array
                optimize.optimize_pixel_with_img_array(avg_img_array, repr_nv_sig)

    finally:
        camera.disarm()
        sig_gen.uwave_off()

    ### Process and plot

    avg_counts, avg_counts_ste = widefield.process_counts(counts)

    kpl.init_kplotlib()
    raw_fig = create_raw_data_figure(freqs, avg_counts, avg_counts_ste)
    try:
        fit_fig = create_fit_figure(freqs, avg_counts, avg_counts_ste)
    except Exception as exc:
        print(exc)
        fit_fig = None

    ### Clean up and return

    tb.reset_cfm(cxn)

    timestamp = dm.get_time_stamp()
    raw_data = {
        "timestamp": timestamp,
        "nv_list": nv_list,
        "num_reps": num_reps,
        "num_steps": num_steps,
        "num_runs": num_runs,
        "readout-units": "ms",
        "freqs": freqs,
        "freq_range": freq_range,
        "counts": counts,
        "counts-units": "photons",
        "img_arrays": img_arrays,
        "img_array-units": "ADUs",
    }

    nv_name = nv_sig["name"]
    file_path = dm.get_file_path(__file__, timestamp, nv_name)
    # keys_to_compress = ["sig_img_arrays", "ref_img_arrays"]
    keys_to_compress = ["img_arrays"]
    dm.save_raw_data(raw_data, file_path, keys_to_compress=keys_to_compress)
    dm.save_figure(raw_fig, file_path)
    if fit_fig is not None:
        file_path = dm.get_file_path(__file__, timestamp, nv_name + "-fit")
        dm.save_figure(fit_fig, file_path)


if __name__ == "__main__":
    kpl.init_kplotlib()

    file_name = "2023_11_20-23_45_03-johnson-nv0_2023_11_09"

    # data = dm.get_raw_data(file_name)
    data = dm.get_raw_data(file_id="1367909272526")
    nv_list = data["nv_list"]
    sig_img_arrays = data["sig_img_arrays"]
    # pixel_coords_list = data["pixel_coords_list"]
    # freqs = data["freqs"]
    counts = data["counts"]

    # counts = widefield.process_img_arrays(sig_img_arrays, pixel_coords_list)
    avg_counts, avg_counts_ste = widefield.process_counts(counts)

    kpl.init_kplotlib()
    freqs = calculate_freqs(2.87, 0.060, 20)
    raw_fig = create_raw_data_figure(freqs, avg_counts, avg_counts_ste)
    fit_fig = create_fit_figure(freqs, avg_counts, avg_counts_ste)

    plt.show(block=True)
