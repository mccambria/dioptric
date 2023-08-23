# -*- coding: utf-8 -*-
"""
Illuminate a list of NVs under a single exposure. This routine is
the basis of most of the other widefield routines.

Created on August 21st, 2023

@author: mccambria
"""


import matplotlib.pyplot as plt
import numpy as np
import majorroutines.optimize as optimize
from utils.constants import ControlStyle, LaserKey
from utils import tool_belt as tb
from utils import common
from utils.constants import CountFormat
from utils import kplotlib as kpl
from utils import positioning as pos
from utils import widefield


def image_single_nv(nv_sig):
    nv_list = [nv_sig]
    with common.labrad_connect() as cxn:
        return main_with_cxn(cxn, nv_list)


def main(nv_list):
    with common.labrad_connect() as cxn:
        return main_with_cxn(cxn, nv_list)


def main_with_cxn(cxn, nv_list):
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
    laser = laser_dict["laser"]
    tb.set_filter(cxn, nv_sig, laser_key)
    readout_power = tb.set_laser_power(cxn, nv_sig, laser_key)
    readout = laser_dict["readout_dur"]
    readout_sec = readout / 10**9
    num_reps = laser_dict["num_reps"]

    num_nvs = len(nv_list)

    ### Load the pulse generator

    if control_style in [ControlStyle.STEP, ControlStyle.STREAM]:
        seq_args = [delay, readout, laser, readout_power]
        seq_args_string = tb.encode_seq_args(seq_args)
        seq_file = "widefield-simple_readout.py"

    # print(seq_file)
    # print(seq_args)
    # return
    pulse_gen.stream_load(seq_file, seq_args_string)

    ### Set up the positioning server, either xy_server or xyz_server

    # Update the coordinates for drift
    adj_coords_list = [
        pos.adjust_coords_for_drift(nv_sig=nv, laser_name=laser) for nv in nv_list
    ]
    if num_nvs == 1:
        coords = adj_coords_list[0]
        pos_server.write_xy(*coords[0:2])
    elif control_style == ControlStyle.STREAM:
        x_coords = [coords[0] for coords in adj_coords_list]
        y_coords = [coords[1] for coords in adj_coords_list]
        pos_server.load_stream_xy(x_coords, y_coords, True)

    ### Set up the image display

    kpl.init_kplotlib(font_size=kpl.Size.SMALL)
    fig, ax = plt.subplots()

    ### Collect the data

    tb.init_safe_stop()

    if control_style == ControlStyle.STEP:
        pass

    elif control_style == ControlStyle.STREAM:
        camera.arm()
        pulse_gen.stream_start(num_nvs * num_reps)
        img_array = camera.read()
        camera.disarm()
        if count_format == CountFormat.RAW:
            widefield.imshow(ax, img_array)
        elif count_format == CountFormat.KCPS:
            img_array_kcps = (np.copy(img_array) / 1000) / readout_sec
            widefield.imshow(ax, img_array_kcps)

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
        "img_array": img_array.astype(int).tolist(),
        "img_array-units": "counts",
    }

    filePath = tb.get_file_path(__file__, timestamp, nv_sig["name"])
    tb.save_figure(fig, filePath)
    tb.save_raw_data(raw_data, filePath)

    return img_array


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
