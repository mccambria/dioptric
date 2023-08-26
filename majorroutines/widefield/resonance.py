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
from utils.constants import ControlStyle
from utils import tool_belt as tb
from utils import common
from utils import widefield
from utils.constants import LaserKey, NVSpinState, CountFormat
from utils import kplotlib as kpl
from utils import positioning as pos
from utils.positioning import get_scan_1d as calculate_freqs
from random import shuffle


def process_img_arrays(img_arrays, nv_list, pixel_drifts):
    num_nvs = len(nv_list)
    num_runs = img_arrays.shape[0]
    num_steps = img_arrays.shape[1]
    sig_counts = []
    for nv_ind in range(num_nvs):
        nv_sig = nv_list[nv_ind]
        pixel_coords = nv_sig["pixel_coords"]
        nv_counts = []
        for run_ind in range(num_runs):
            freq_counts = []
            for freq_ind in range(num_steps):
                img_array = img_arrays[run_ind, freq_ind]
                pixel_drift = pixel_drifts[run_ind, freq_ind]
                opt_pixel_coords = optimize.optimize_pixel(
                    img_array,
                    pixel_coords,
                    set_scanning_drift=False,
                    set_pixel_drift=False,
                    pixel_drift=pixel_drift,
                )
                counts = widefield.counts_from_img_array(
                    img_array, opt_pixel_coords, drift_adjust=False
                )
                freq_counts.append(counts)

                # Plot each img_array
                # if nv_ind == 0:
                #     fig, ax = plt.subplots()
                #     widefield.imshow(ax, img_array, count_format=CountFormat.RAW)

            nv_counts.append(freq_counts)
        nv_counts = np.array(nv_counts)
        sig_counts.append(np.average(nv_counts, axis=0))
    return np.array(sig_counts)


def create_figure(freqs, sig_counts):
    kpl.init_kplotlib()
    num_nvs = sig_counts.shape[0]
    fig, ax = plt.subplots()
    for ind in range(num_nvs):
        kpl.plot_line(ax, freqs, sig_counts[ind])
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("Counts")
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

    # Use first NV for some basic setup
    nv_sig = nv_list[0]
    optimize.prepare_microscope(cxn, nv_sig)
    laser_key = LaserKey.IMAGING
    laser_dict = nv_sig[laser_key]
    laser = laser_dict["name"]
    tb.set_filter(cxn, optics_name=laser, filter_name=laser_filter)
    readout_power = tb.set_laser_power(cxn, nv_sig, laser_key)
    readout = laser_dict["readout_dur"]
    readout_sec = readout / 10**9

    num_nvs = len(nv_list)

    last_opt_time = time.time()
    opt_period = 10 * 60

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
            print(run_ind)
            print(freq_ind)
            print()

            # Optimize
            # now = time.time()
            # if (last_opt_time is None) or (now - last_opt_time > opt_period):
            #     last_opt_time = now
            #     optimize.optimize_widefield_calibration(cxn)

            #     # Reset the pulse streamer and laser filter
            #     tb.set_filter(cxn, optics_name=laser, filter_name=laser_filter)
            #     pulse_gen.stream_load(seq_file, seq_args_string)

            # Update the coordinates for drift
            adj_coords_list = [
                pos.adjust_coords_for_drift(nv_sig=nv, laser_name=laser)
                for nv in nv_list
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
            optimize.optimize_pixel(img_array, nv_sig["pixel_coords"])
            pixel_drifts[run_ind][freq_ind] = widefield.get_pixel_drift()

            sig_gen.uwave_off()

    ### Data processing and plotting

    img_arrays = np.array(img_arrays, dtype=int)
    pixel_drifts = np.array(pixel_drifts, dtype=float)
    # sig_counts = process_img_arrays(img_arrays, nv_list, pixel_drifts)
    # fig = create_figure(freqs, sig_counts)

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

    filePath = tb.get_file_path(__file__, timestamp, nv_sig["name"])
    # tb.save_figure(fig, filePath)
    tb.save_raw_data(raw_data, filePath, keys_to_compress=["img_arrays"])


if __name__ == "__main__":
    kpl.init_kplotlib()

    file_name = "2023_08_23-15_22_42-johnson-nv0_2023_08_23"
    data = tb.get_raw_data(file_name)
    freqs = data["freqs"]
    img_arrays = np.array(data["img_arrays"], dtype=int)
    nv_list = data["nv_list"]
    pixel_drifts = np.array(data["pixel_drifts"], dtype=float)

    sig_counts = process_img_arrays(img_arrays, nv_list, pixel_drifts)
    create_figure(freqs, sig_counts)

    plt.show(block=True)
