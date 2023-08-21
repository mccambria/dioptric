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
from utils.constants import States, NormStyle
from utils.constants import CollectionMode, CountFormat
from utils import kplotlib as kpl
from utils import positioning as pos
from utils.positioning import get_scan_1d as calculate_freqs


def main(
    nv_list,
    freq_center,
    freq_range,
    num_steps,
    num_runs,
    uwave_power,
    state=States.LOW,
):
    with common.labrad_connect() as cxn:
        img_array, x_voltages, y_voltages = main_with_cxn(
            cxn,
            nv_list,
            freq_center,
            freq_range,
            num_steps,
            num_runs,
            uwave_power,
            state,
        )

    return img_array, x_voltages, y_voltages


def main_with_cxn(
    cxn,
    nv_list,
    freq_center,
    freq_range,
    num_steps,
    num_runs,
    uwave_power,
    state=States.LOW,
):
    ### Some initial setup

    tb.reset_cfm(cxn)

    config = common.get_config_dict()
    config_positioning = config["Positioning"]
    freqs = calculate_freqs(freq_center, freq_range, num_steps)
    nv_sig = nv_list[0]  # Use first NV for some basic setup
    control_style = pos.get_xy_control_style()

    center_coords = pos.set_xyz_on_nv(cxn, nv_sig)
    x_center, y_center, z_center = center_coords
    optimize.prepare_microscope(cxn, nv_sig)
    pos_server = pos.get_server_pos_xy(cxn)
    camera = tb.get_server_camera(cxn)
    pulse_gen = tb.get_server_pulse_gen(cxn)
    count_format = config["count_format"]

    laser_key = "imaging_laser"
    readout_laser = nv_sig[laser_key]
    tb.set_filter(cxn, nv_sig, laser_key)
    readout_power = tb.set_laser_power(cxn, nv_sig, laser_key)

    delay = config_positioning["xy_delay"]
    axis_units = config_positioning["xy_units"]

    num_nvs = len(nv_list)

    ### Load the pulse generator

    readout = nv_sig["imaging_readout_dur"]
    readout_us = readout / 10**3
    readout_sec = readout / 10**9
    seq_args = [delay, readout, readout_laser, readout_power]
    seq_args_string = tb.encode_seq_args(seq_args)
    seq_file = "widefield-resonance.py"

    # print(seq_file)
    # print(seq_args)
    # return
    ret_vals = pulse_gen.stream_load(seq_file, seq_args_string)
    period = ret_vals[0]

    ### Set up the positioning server, either xy_server or xyz_server

    x_coords = [sig["coords"][0] for sig in nv_list]
    y_coords = [sig["coords"][1] for sig in nv_list]
    pos_server = pos_server.load_stream_xy(x_coords, y_coords)

    ### Set up the image display

    kpl.init_kplotlib(font_size=kpl.Size.SMALL)
    hor_label = "X"
    ver_label = "Y"
    if count_format == CountFormat.RAW:
        cbar_label = "Counts"
    if count_format == CountFormat.KCPS:
        cbar_label = "Kcps"
    imshow_kwargs = {
        "x_label": hor_label,
        "y_label": ver_label,
        "cbar_label": cbar_label,
    }

    fig, ax = plt.subplots()

    ### Collect the data

    tb.init_safe_stop()

    if control_style == ControlStyle.STEP:
        pass

    elif control_style == ControlStyle.STREAM:
        camera.arm()
        pulse_gen.stream_start(num_nvs)
        img_array = camera.read()
        camera.disarm()
        if count_format == CountFormat.RAW:
            kpl.imshow(ax, img_array, **imshow_kwargs)
        elif count_format == CountFormat.KCPS:
            img_array_kcps = (np.copy(img_array) / 1000) / readout_sec
            kpl.imshow(ax, img_array_kcps, **imshow_kwargs)

    ### Clean up and save the data

    tb.reset_cfm(cxn)
    pos.set_xyz(cxn, center_coords)

    timestamp = tb.get_time_stamp()
    rawData = {
        "timestamp": timestamp,
        "nv_sig": nv_sig,
        "x_center": x_center,
        "y_center": y_center,
        "range_1": z_center,
        "x_range": range_1,
        "range_2": range_2,
        "num_steps": num_steps,
        "extent": extent,
        "scan_axes": scan_axes,
        "readout": readout,
        "readout-units": "ns",
        "title": title,
        "coords_1_1d": coords_1_1d.tolist(),
        "coords_1_1d-units": axis_1_units,
        "coords_2_1d": coords_1_1d.tolist(),
        "coords_2_1d-units": axis_2_units,
        "img_array": img_array.astype(int).tolist(),
        "img_array-units": "counts",
    }

    filePath = tb.get_file_path(__file__, timestamp, nv_sig["name"])
    tb.save_figure(fig, filePath)
    tb.save_raw_data(rawData, filePath)

    return img_array, coords_1_1d, coords_2_1d


if __name__ == "__main__":
    file_name = "2023_08_15-14_34_47-johnson-nvref"

    data = tb.get_raw_data(file_name)
    img_array = np.array(data["img_array"])
    readout = data["readout"]
    img_array_kcps = (img_array / 1000) / (readout * 1e-9)
    extent = data["extent"] if "extent" in data else None

    kpl.init_kplotlib()
    fig, ax = plt.subplots()
    im = kpl.imshow(ax, img_array, extent=extent)
    # ax.set_xlim([124.5 - 15, 124.5 + 15])
    # ax.set_ylim([196.5 + 15, 196.5 - 15])

    plt.show(block=True)
