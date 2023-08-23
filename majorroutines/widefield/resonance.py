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
from utils.constants import LaserKey
from utils.constants import NVSpinStates, NormStyle
from utils.constants import CollectionMode, CountFormat
from utils import kplotlib as kpl
from utils import positioning as pos
from utils.positioning import get_scan_1d as calculate_freqs
from random import shuffle


def main(
    nv_list,
    freq_center,
    freq_range,
    num_steps,
    num_reps,
    num_runs,
    uwave_power,
    state=NVSpinStates.LOW,
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
    state=NVSpinStates.LOW,
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
    readout_laser = laser_dict["laser"]
    tb.set_filter(cxn, nv_sig, laser_key)
    readout_power = tb.set_laser_power(cxn, nv_sig, laser_key)
    readout = laser_dict["readout_dur"]
    readout_sec = readout / 10**9
    num_reps = laser_dict["num_reps"]

    num_nvs = len(nv_list)

    ### Load the pulse generator

    if control_style in [ControlStyle.STEP, ControlStyle.STREAM]:
        seq_args = [readout, state.name, readout_laser, readout_power]
        seq_args_string = tb.encode_seq_args(seq_args)
        seq_file = "widefield-resonance.py"

    # print(seq_file)
    # print(seq_args)
    # return
    pulse_gen.stream_load(seq_file, seq_args_string)

    ### Set up the positioning server, either xy_server or xyz_server

    if num_nvs == 1:
        nv_sig = nv_list[0]
        coords = nv_sig["coords"]
        pos_server.write_xy(*coords[0:2])
    elif control_style == ControlStyle.STREAM:
        x_coords = [sig["coords"][0] for sig in nv_list]
        y_coords = [sig["coords"][1] for sig in nv_list]
        pos_server.load_stream_xy(x_coords, y_coords, True)

    ### Set up the image display

    kpl.init_kplotlib(font_size=kpl.Size.SMALL)

    ### Microwave setup

    freqs = calculate_freqs(freq_center, freq_range, num_steps)
    sig_gen = tb.get_server_sig_gen(cxn, state)
    sig_gen.set_amp(uwave_power)
    sig_gen.uwave_on()

    ### Data tracking

    img_arrays = [[] for ind in range(num_runs)]
    freq_ind_master_list = [[] for ind in range(num_runs)]
    freq_ind_list = list(range(0, num_steps))

    ### Collect the data

    tb.init_safe_stop()

    for run_ind in range(num_runs):
        shuffle(freq_ind_list)
        for freq_ind in freq_ind_list:
            freq_ind_master_list[run_ind].append(freq_ind)
            freq = freqs[freq_ind]
            sig_gen.set_freq(freq)

            if control_style == ControlStyle.STEP:
                pass

            elif control_style == ControlStyle.STREAM:
                camera.arm()
                pulse_gen.stream_start(num_nvs * num_reps)
                img_array = camera.read()
                camera.disarm()
                img_arrays[run_ind].append(img_array)

    ### Data processing and plotting

    img_arrays = np.array(img_arrays, dtype=int)
    # sig_counts = []
    # for ind in range(num_nvs):
    #     nv_sig_counts = [[] for ind in range(num_runs)]
    #     for run_ind in range(num_runs):
    #         for freq_ind in range(num_steps):
    #             img_array = img_arrays[run_ind]

    ### Clean up and save the data

    sig_gen.uwave_off()
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
    tb.save_raw_data(raw_data, filePath)

    return img_array


if __name__ == "__main__":
    file_name = "2023_08_22-21_19_41-johnson-nv0_2023_08_21"

    plt.show(block=True)
