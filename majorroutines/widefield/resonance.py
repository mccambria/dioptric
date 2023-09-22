# -*- coding: utf-8 -*-
"""
Widefield continuous wave electron spin resonance (CWESR) routine

Created on August 21st, 2023

@author: mccambria
"""


import matplotlib.pyplot as plt
import numpy as np
import time
import labrad
import majorroutines.optimize as optimize
from majorroutines.pulsed_resonance import fit_resonance, voigt_split, voigt
from utils.constants import ControlStyle
from utils import tool_belt as tb
from utils import common
from utils import widefield
from utils.constants import LaserKey, NVSpinState, CountFormat
from utils import kplotlib as kpl
from utils import positioning as pos
from utils.positioning import get_scan_1d as calculate_freqs
from random import shuffle
from cProfile import Profile
import itertools
from multiprocessing import Pool
import json


def process_counts(sig_counts):
    run_ax = 1

    # sig_counts = sig_counts.astype(float)
    # for ind in range(5):
    #     for jnd in range(20):
    #         low = np.percentile(sig_counts[ind, :, jnd], 20)
    #         sig_counts[ind, sig_counts[ind, :, jnd] < low, jnd] = np.nan
    # avg_counts = np.nanmean(sig_counts, axis=run_ax)
    # num_runs = np.count_nonzero(~np.isnan(sig_counts), axis=run_ax)
    # avg_counts_ste = np.nanstd(sig_counts, axis=run_ax, ddof=1) / np.sqrt(num_runs)

    avg_counts = np.mean(sig_counts, axis=run_ax)
    num_runs = sig_counts.shape[run_ax]
    avg_counts_ste = np.std(sig_counts, axis=run_ax, ddof=1) / np.sqrt(num_runs)

    # avg_counts = np.median(sig_counts, axis=run_ax)
    # dist_to_median = abs(sig_counts - avg_counts[:, np.newaxis, :])
    # avg_counts_ste = np.median(dist_to_median, axis=run_ax) / 10

    # avg_counts = np.max(sig_counts, axis=run_ax)
    # avg_counts = np.percentile(sig_counts, 75, axis=run_ax)
    # avg_counts_ste = np.sqrt(avg_counts)

    return avg_counts, avg_counts_ste


def process_img_arrays(img_arrays, nv_list, pixel_drifts, radius=None):
    num_nvs = len(nv_list)
    num_runs = img_arrays.shape[0]
    # num_runs = 8
    num_steps = img_arrays.shape[1]

    # Run through the images in parallel

    global process_img_arrays_sub  # Necessary for multiprocessing to pickle the function

    def process_img_arrays_sub(nv_ind, run_ind, freq_ind):
        pixel_coords = nv_list[nv_ind]["pixel_coords"]
        img_array = img_arrays[run_ind, freq_ind]
        pixel_drift = pixel_drifts[run_ind, freq_ind]
        opt_pixel_coords = optimize.optimize_pixel(
            pixel_coords=pixel_coords,
            img_array=img_array,
            set_scanning_drift=False,
            set_pixel_drift=False,
            pixel_drift=pixel_drift,
        )
        counts = widefield.counts_from_img_array(
            img_array, opt_pixel_coords, radius=radius, drift_adjust=False
        )
        return counts

    # List of Cartesian product of the indices
    nvs_range = range(num_nvs)
    runs_range = range(num_runs)
    steps_range = range(num_steps)
    index_list = itertools.product(nvs_range, runs_range, steps_range)

    with Pool(6) as p:
        sig_counts_list = p.starmap(process_img_arrays_sub, index_list, chunksize=100)

    sig_counts = np.reshape(sig_counts_list, (num_nvs, num_runs, num_steps))

    # Single threaded
    # sig_counts = [
    #     [[None] * num_steps for ind in range(num_runs)] for jnd in range(num_nvs)
    # ]
    # for nv_ind in range(num_nvs):
    #     for run_ind in range(num_runs):
    #         for freq_ind in range(num_steps):
    #             counts = process_img_arrays_sub(nv_ind, freq_ind, run_ind + 1)
    #             sig_counts[nv_ind][run_ind][freq_ind] = counts

    return sig_counts


def create_raw_data_figure(freqs, counts, counts_ste):
    kpl.init_kplotlib()
    num_nvs = counts.shape[0]
    fig, ax = plt.subplots()
    for ind in range(num_nvs):
        kpl.plot_points(ax, freqs, counts[ind], yerr=counts_ste[ind], label=ind)
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("ADUs")
    min_freqs = min(freqs)
    max_freqs = max(freqs)
    excess = 0.08 * (max_freqs - min_freqs)
    ax.set_xlim(min_freqs - excess, max_freqs + excess)
    ax.legend()
    return fig


def create_fit_figure(freqs, counts, counts_ste):
    kpl.init_kplotlib()
    num_nvs = counts.shape[0]
    fig, ax = plt.subplots()
    freq_linspace = np.linspace(min(freqs) - 0.001, max(freqs) + 0.001, 100)
    shift_factor = 0.25
    offset_inds = [num_nvs - 1 - ind for ind in list(range(num_nvs))]
    # shuffle(offset_inds)
    for ind in range(num_nvs):
        nv_counts = counts[ind]
        nv_counts_ste = counts_ste[ind]
        # if False:
        if ind in [3, 4]:
            # norm, contrast, g_width, l_width, center, splitting
            guess_params = [nv_counts[0], 0.15, 2, 2, 2.87, 5]
            fit_func = (
                lambda freq, norm, contrast, g_width, l_width, center, splitting: norm
                * (1 - voigt_split(freq, contrast, g_width, l_width, center, splitting))
            )
        else:
            # norm, contrast, g_width, l_width, center
            guess_params = [nv_counts[0], 0.15, 2, 2, 2.87]
            fit_func = lambda freq, norm, contrast, g_width, l_width, center: norm * (
                1 - voigt(freq, contrast, g_width, l_width, center)
            )

        fit_func, popt, pcov = fit_resonance(
            freqs,
            nv_counts,
            nv_counts_ste,
            fit_func=fit_func,
            guess_params=guess_params,
        )
        offset_ind = offset_inds[ind]
        norm = popt[0]
        # kpl.plot_line(ax, freqs, counts[ind])
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
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("Normalized fluorescence")
    min_freqs = min(freqs)
    max_freqs = max(freqs)
    excess = 0.08 * (max_freqs - min_freqs)
    ax.set_xlim(min_freqs - excess, max_freqs + excess)
    ax.legend()
    return fig


def main(
    nv_list,
    freq_center,
    freq_range,
    num_steps,
    num_reps,
    num_runs,
    uwave_power,
    state=NVSpinState.LOW,
    laser_filter=None,
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
            uwave_power,
            state,
            laser_filter,
        )


def main_with_cxn(
    cxn,
    nv_list,
    freq_center,
    freq_range,
    num_steps,
    num_reps,
    num_runs,
    uwave_power,
    state=NVSpinState.LOW,
    laser_filter=None,
):
    ### Some initial setup

    tb.reset_cfm(cxn)

    # Config stuff
    config = common.get_config_dict()
    config_positioning = config["Positioning"]
    delay = config_positioning["xy_delay"]
    control_style = pos.get_xy_control_style()
    count_format = config["count_format"]

    # Servers
    pos_server = pos.get_server_pos_xy(cxn)
    camera = tb.get_server_camera(cxn)
    pulse_gen = tb.get_server_pulse_gen(cxn)

    # Use one NV for some basic setup

    nv_sig, _ = widefield.get_widefield_calibration_nvs()
    optimize.prepare_microscope(cxn, nv_sig)
    laser_key = LaserKey.IMAGING
    laser_dict = nv_sig[laser_key]
    laser = laser_dict["name"]
    tb.set_filter(cxn, optics_name=laser, filter_name=laser_filter)
    readout_power = tb.set_laser_power(cxn, nv_sig, laser_key)
    readout = laser_dict["readout_dur"]
    readout_sec = readout / 10**9

    num_nvs = len(nv_list)
    nv_list_shuffle = nv_list.copy()

    # last_opt_time = time.time()
    last_opt_time = None
    opt_period = 5 * 60

    ### Load the pulse generator

    if control_style in [ControlStyle.STEP, ControlStyle.STREAM]:
        seq_args = [readout, state.name, laser, readout_power]
        seq_args_string = tb.encode_seq_args(seq_args)
        seq_file = "widefield-resonance.py"

    # print(seq_file)
    # print(seq_args)
    # return
    ret_vals = pulse_gen.stream_load(seq_file, seq_args_string)

    ### Set up the image display

    kpl.init_kplotlib(font_size=kpl.Size.SMALL)

    ### Microwave setup

    freqs = calculate_freqs(freq_center, freq_range, num_steps)
    sig_gen = tb.get_server_sig_gen(cxn, state)
    sig_gen.set_amp(uwave_power)

    ### Data tracking

    img_arrays = [[None] * num_steps for ind in range(num_runs)]
    freq_ind_master_list = [[] for ind in range(num_runs)]
    freq_ind_list = list(range(0, num_steps))
    pixel_drifts = [[None] * num_steps for ind in range(num_runs)]

    ### Collect the data

    tb.init_safe_stop()
    start_time = time.time()

    for run_ind in range(num_runs):
        shuffle(freq_ind_list)
        for freq_ind in freq_ind_list:
            print(f"{run_ind}, {freq_ind}")
            # print(run_ind)
            # print(freq_ind)
            # print()

            shuffle(nv_list_shuffle)

            # Optimize in z by scanning, then in xy with pixels again.
            # Note: Each optimization routine must be identical or else there
            # will be a potentially large source of variance between runs
            now = time.time()
            if (last_opt_time is None) or (now - last_opt_time > opt_period):
                # This first optimize is just for z
                optimize.main(nv_sig, set_to_opti_coords=False, only_z_opt=True)
                # The optimization
                optimize.optimize_pixel(nv_sig)
                # optimize.optimize_widefield_calibration(cxn)
                last_opt_time = time.time()

            # Reset the pulse streamer and laser filter
            tb.set_filter(cxn, optics_name=laser, filter_name=laser_filter)
            pulse_gen.stream_load(seq_file, seq_args_string)

            # Update the coordinates for drift
            adj_coords_list = [
                pos.adjust_coords_for_drift(nv_sig=nv, laser_name=laser)
                for nv in nv_list_shuffle
            ]
            if num_nvs == 1:
                coords = adj_coords_list[0]
                pos_server.write_xy(*coords[0:2])
            elif control_style == ControlStyle.STREAM:
                x_coords = [coords[0] for coords in adj_coords_list]
                y_coords = [coords[1] for coords in adj_coords_list]
                pos_server.load_stream_xy(x_coords, y_coords, True)

            freq_ind_master_list[run_ind].append(freq_ind)
            freq = freqs[freq_ind]
            sig_gen.set_freq(freq)
            sig_gen.uwave_on()

            # Record the image
            if control_style == ControlStyle.STEP:
                pass
            elif control_style == ControlStyle.STREAM:
                camera.arm()
                pulse_gen.stream_start(num_nvs * num_reps)
                img_array = camera.read()
                camera.disarm()

            img_arrays[run_ind][freq_ind] = img_array
            optimize.optimize_pixel(nv_sig)
            pixel_drifts[run_ind][freq_ind] = widefield.get_pixel_drift()

            sig_gen.uwave_off()

    ### Data processing and plotting

    img_arrays = np.array(img_arrays, dtype=int)
    pixel_drifts = np.array(pixel_drifts, dtype=float)
    radius = config["camera_spot_radius"]
    sig_counts = process_img_arrays(img_arrays, nv_list, pixel_drifts, radius=radius)
    avg_counts, avg_counts_ste = process_counts(sig_counts)
    raw_data_fig = create_raw_data_figure(freqs, avg_counts, avg_counts_ste)
    fit_fig = create_fit_figure(freqs, avg_counts, avg_counts_ste)

    ### Clean up and save the data

    tb.reset_cfm(cxn)
    pos.set_xyz_on_nv(cxn, nv_sig)

    timestamp = tb.get_time_stamp()
    raw_data = {
        "timestamp": timestamp,
        "nv_list": nv_list,
        "num_reps": num_reps,
        "readout": readout,
        "readout-units": "ns",
        "img_arrays": img_arrays,
        "img_arrays-units": "counts",
        # "sig_counts": sig_counts,
        # "avg_counts": avg_counts,
        # "avg_counts_ste": avg_counts_ste,
        "pixel_drifts": pixel_drifts,
        "pixel_drifts-units": "pixels",
        "freq_center": freq_center,
        "freq_center-units": "GHz",
        "freq_range": freq_range,
        "freq_range-units": "GHz",
        "freqs": freqs,
        "freqs-units": "GHz",
        "state": state,
        "num_steps": num_steps,
        "num_reps": num_reps,
        "num_runs": num_runs,
        "uwave_power": uwave_power,
        "uwave_power-units": "dBm",
        "readout": readout,
        "readout-units": "ns",
        "freq_ind_master_list": freq_ind_master_list,
    }

    nv_name = nv_list[0]["name"]
    file_path = tb.get_file_path(__file__, timestamp, nv_name)
    tb.save_figure(raw_data_fig, file_path)
    tb.save_raw_data(raw_data, file_path, keys_to_compress=["img_arrays"])
    file_path = tb.get_file_path(__file__, timestamp, nv_name + "-fit")
    tb.save_figure(fit_fig, file_path)


if __name__ == "__main__":
    kpl.init_kplotlib()

    file_name = "2023_09_22-12_01_45-johnson-nv0_2023_09_11"
    data = tb.get_raw_data(file_name)
    freqs = np.array(data["freqs"])
    img_arrays = np.array(data["img_arrays"], dtype=int)
    nv_list = data["nv_list"]
    pixel_drifts = np.array(data["pixel_drifts"], dtype=float)
    # radius = data["config"]["camera_spot_radius"]
    radius = 15
    freq_ind_master_list = data["freq_ind_master_list"]
    # sig_counts = np.array(data["sig_counts"])
    # avg_counts = np.array(data["avg_counts"])
    # avg_counts_ste = np.array(data["avg_counts_ste"])

    # Histograms
    # for ind in range(5):
    #     fig, ax = plt.subplots()
    #     nv_counts = sig_counts[ind, :, :]
    #     ax.hist(nv_counts.flatten(), 100)
    # plt.show(block=True)

    # Play the images back like a movie
    start_ind = 1
    img_arrays = img_arrays[start_ind:]
    pixel_drifts = pixel_drifts[start_ind:]
    num_nvs = len(nv_list)
    num_runs = img_arrays.shape[0]
    num_steps = img_arrays.shape[1]
    # for run_ind in range(num_runs):
    #     print(run_ind)
    #     for freq_ind in freq_ind_master_list[run_ind][-2:-1]:
    #         fig, ax = plt.subplots()
    #         kpl.imshow(ax, img_arrays[run_ind, freq_ind])
    #         plt.show(block=True)

    # Process images
    print("start")
    start = time.time()
    sig_counts = process_img_arrays(img_arrays, nv_list, pixel_drifts, radius=radius)
    stop = time.time()
    print("stop")
    print(f"Time elapsed: {stop - start}")
    avg_counts, avg_counts_ste = process_counts(sig_counts)

    create_raw_data_figure(freqs, avg_counts, avg_counts_ste)
    create_fit_figure(freqs, avg_counts, avg_counts_ste)

    # ordered_sig_counts = []
    # for run_ind in range(num_runs):
    #     for freq_ind in freq_ind_master_list[run_ind]:
    #         ordered_sig_counts.append(sig_counts[:, run_ind, freq_ind])
    # ordered_sig_counts = np.array(ordered_sig_counts)
    # fig, ax = plt.subplots()
    # ax.plot(ordered_sig_counts)

    # Update saved values
    # data["sig_counts"] = sig_counts.tolist()
    # data["avg_counts"] = avg_counts.tolist()
    # data["avg_counts_ste"] = avg_counts_ste.tolist()
    # data["img_arrays"] = f"pc_rabi/branch_master/resonance/2023_09/{file_name}.npz"
    # nvdata = common.get_nvdata_path()
    # full_path = nvdata / f"pc_rabi/branch_master/resonance/2023_09/{file_name}.txt"
    # with open(full_path, "w") as f:
    #     json.dump(data, f, indent=2)

    plt.show(block=True)
