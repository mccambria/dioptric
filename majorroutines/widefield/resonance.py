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

    for nv_ind in range(num_nvs):
        nv_counts = counts[nv_ind]
        nv_counts_ste = counts_ste[nv_ind]
        norm_guess = np.median(nv_counts)
        amp_guess = (np.max(nv_counts) - norm_guess) / norm_guess
        guess_params = [norm_guess, amp_guess, 5, 5, np.median(freqs)]
        fit_fn = lambda freq, norm, contrast, g_width, l_width, center: norm * (
            1 + voigt(freq, contrast, g_width, l_width, center)
        )
        _, popt, pcov = fit_resonance(
            freqs,
            nv_counts,
            nv_counts_ste,
            fit_func=fit_fn,
            guess_params=guess_params,
        )

        # SCC readout noise tracking
        norm = popt[0]
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
        norms.append(popt[0])

    print(f"a0 average: {round(np.average(a0_list), 2)}")
    print(f"a1 average: {round(np.average(a1_list), 2)}")
    print(f"Average readout noise: {round(np.average(readout_noise_list), 2)}")
    print(f"Median readout noise: {round(np.median(readout_noise_list), 2)}")

    ### Make the figure

    fig, ax = plt.subplots()
    widefield.plot_fit(ax, nv_list, freqs, counts, counts_ste, fit_fns, popts, norms)
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("Normalized fluorescence")
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

    seq_file = "resonance.py"

    ### Data tracking

    counts = np.empty((num_nvs, num_runs, num_steps, num_reps))
    resolution = widefield._get_camera_resolution()
    img_arrays = np.empty((num_runs, num_steps, *resolution), dtype=np.uint16)
    freq_ind_master_list = [[] for ind in range(num_runs)]
    freq_ind_list = list(range(0, num_steps))

    ### Collect the data

    try:
        for run_ind in range(num_runs):
            shuffle(freq_ind_list)

            # Load the pulse gen from scratch so we have up-to-date AOD frequencies
            seq_args = widefield.get_base_scc_seq_args(nv_list)
            seq_args.extend([sig_gen_name, uwave_duration])
            seq_args_string = tb.encode_seq_args(seq_args)
            # print(seq_args)
            # print(seq_file)
            # return
            pulse_gen.stream_load(seq_file, seq_args_string, num_reps)

            camera.arm()
            sig_gen.uwave_on()

            for freq_ind in freq_ind_list:
                pixel_coords_list = [
                    widefield.get_nv_pixel_coords(nv) for nv in nv_list
                ]

                freq_ind_master_list[run_ind].append(freq_ind)
                freq = freqs[freq_ind]
                sig_gen.set_freq(freq)

                # Try 5 times then give up
                num_attempts = 5
                attempt_ind = 0
                while True:
                    try:
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
                        break
                    except Exception as exc:
                        print(exc)
                        camera.arm()
                        attempt_ind += 1
                        if attempt_ind == num_attempts:
                            raise RuntimeError("Maxed out number of attempts")
                if attempt_ind > 0:
                    print(f"{attempt_ind} crashes occurred")

                avg_img_array = avg_img_array / num_reps
                img_arrays[run_ind, freq_ind, :, :] = avg_img_array

            camera.disarm()
            sig_gen.uwave_off()

            optimize.optimize_pixel_with_cxn(cxn, repr_nv_sig)

    finally:
        camera.disarm()
        sig_gen.uwave_off()

    ### Process and plot

    avg_counts, avg_counts_ste = widefield.process_counts(counts)

    kpl.init_kplotlib()
    raw_fig = create_raw_data_figure(nv_list, freqs, avg_counts, avg_counts_ste)
    try:
        fit_fig = create_fit_figure(nv_list, freqs, avg_counts, avg_counts_ste)
    except Exception as exc:
        print(exc)
        fit_fig = None

    ### Clean up and return

    tb.reset_cfm(cxn)

    timestamp = dm.get_time_stamp()
    raw_data = {
        "timestamp": timestamp,
        "nv_list": nv_list,
        "state": state,
        "num_reps": num_reps,
        "num_steps": num_steps,
        "num_runs": num_runs,
        "readout-units": "ms",
        "freqs": freqs,
        "freq-units": "GHz",
        "freq_range": freq_range,
        "freq_center": freq_center,
        "freq_ind_master_list": freq_ind_master_list,
        "counts": counts,
        "counts-units": "photons",
        "img_arrays": img_arrays,
        "img_array-units": "ADUs",
    }

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

    # file_name = "2023_11_27-19_31_32-johnson-nv0_2023_11_25"
    # data = dm.get_raw_data(file_name)
    data = dm.get_raw_data(file_id=1378188362165)
    # data = dm.get_raw_data(file_id=1377535055998)  # large drift
    # data = dm.get_raw_data(file_id=1377603810907)  # No rf
    # data = dm.get_raw_data(file_id=1375345528278)
    # data = dm.get_raw_data(file_id=1377621937207)  # 2.5 GHz
    ### Drift fix
    # data = dm.get_raw_data(file_id=1377650545206)  # 8 runs
    # data = dm.get_raw_data(file_id=1377675224508)  # 32 runs
    # data = dm.get_raw_data(file_id=1377983214052)  # 64 runs, 200 MHz range

    nv_list = data["nv_list"]
    num_nvs = len(nv_list)
    img_arrays = data["img_arrays"]
    num_steps = data["num_steps"]
    num_runs = data["num_runs"]
    avg_img_arrays = np.average(img_arrays, axis=1)

    # for run_ind in range(num_steps):
    #     fig, ax = plt.subplots()
    #     kpl.imshow(ax, avg_img_arrays[run_ind])
    #     plt.show(block=True)

    freqs = data["freqs"]
    counts = np.array(data["counts"])
    print(counts.shape)
    fig, ax = plt.subplots()
    for ind in range(num_nvs):
        nv_counts = counts[ind]
        end_to_end = np.mean(nv_counts, (1, 2)).flatten()
        kpl.plot_line(ax, range(len(end_to_end)), end_to_end)
    for ind in range(num_steps):
        fig, ax = plt.subplots()
        end_to_end = counts[4, :, ind, :].flatten()
        kpl.histogram(ax, end_to_end, 100)
        ax.set_title(f"{ind}")
        ax.set_xlim(20, 130)
        ax.set_ylim(0, 45)
    # for ind in range(num_nvs):
    #     fig, ax = plt.subplots()
    #     end_to_end = counts[ind, 0 : num_runs // 2, :, :].flatten()
    #     kpl.histogram(ax, end_to_end, 100)
    #     ax.set_title(f"{ind} first half")
    #     fig, ax = plt.subplots()
    #     end_to_end = counts[ind, num_runs // 2 :, :, :].flatten()
    #     kpl.histogram(ax, end_to_end, 100)
    #     ax.set_title(f"{ind} second half")

    # counts_partial = counts[:, 0:16, :, :]
    # avg_counts, avg_counts_ste = widefield.process_counts(counts_partial)
    # raw_fig = create_raw_data_figure(nv_list, freqs, avg_counts, avg_counts_ste)
    # fit_fig = create_fit_figure(nv_list, freqs, avg_counts, avg_counts_ste)
    # counts_partial = counts[:, 16:32, :, :]
    # avg_counts, avg_counts_ste = widefield.process_counts(counts_partial)
    # raw_fig = create_raw_data_figure(nv_list, freqs, avg_counts, avg_counts_ste)
    # fit_fig = create_fit_figure(nv_list, freqs, avg_counts, avg_counts_ste)
    # counts_partial = counts[:, 32:48, :, :]
    # avg_counts, avg_counts_ste = widefield.process_counts(counts_partial)
    # raw_fig = create_raw_data_figure(nv_list, freqs, avg_counts, avg_counts_ste)
    # fit_fig = create_fit_figure(nv_list, freqs, avg_counts, avg_counts_ste)
    # counts_partial = counts[:, 48:, :, :]
    # avg_counts, avg_counts_ste = widefield.process_counts(counts_partial)
    # raw_fig = create_raw_data_figure(nv_list, freqs, avg_counts, avg_counts_ste)
    # fit_fig = create_fit_figure(nv_list, freqs, avg_counts, avg_counts_ste)

    avg_counts, avg_counts_ste = widefield.process_counts(counts)
    raw_fig = create_raw_data_figure(nv_list, freqs, avg_counts, avg_counts_ste)
    fit_fig = create_fit_figure(nv_list, freqs, avg_counts, avg_counts_ste)

    # for ind in range(7):
    #     counts_copy = counts[:, ind : ind + 1, :, :]
    #     avg_counts, avg_counts_ste = widefield.process_counts(counts_copy)
    #     raw_fig = create_raw_data_figure(nv_list, freqs, avg_counts, avg_counts_ste)
    #     # fit_fig = create_fit_figure(nv_list, freqs, avg_counts, avg_counts_ste)

    plt.show(block=True)
