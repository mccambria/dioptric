# -*- coding: utf-8 -*-
"""
Illuminate the sample widefield and readout with a camera

Created on October 5th, 2023

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
from utils.constants import CollectionMode, CountFormat, LaserKey
from utils import kplotlib as kpl
from utils import positioning as pos
from scipy import ndimage
from enum import Enum, auto


def main(nv_sig):
    with common.labrad_connect() as cxn:
        img_array = main_with_cxn(cxn, nv_sig)

    return img_array


def main_with_cxn(cxn, nv_sig):
    ### Some initial setup

    config = common.get_config_dict()

    tb.reset_cfm(cxn)
    optimize.prepare_microscope(cxn, nv_sig)
    camera = tb.get_server_camera(cxn)
    pulse_gen = tb.get_server_pulse_gen(cxn)

    laser_key = LaserKey.IMAGING
    laser_dict = nv_sig[laser_key]
    readout_laser = laser_dict["name"]
    tb.set_filter(cxn, nv_sig, laser_key)

    ### Load the pulse generator

    readout = laser_dict["readout_dur"]
    readout_us = readout / 10**3
    readout_ms = round(readout / 10**6)
    readout_sec = readout / 10**9

    delay = 0
    seq_args = [delay, readout, readout_laser]
    seq_args_string = tb.encode_seq_args(seq_args)
    seq_file = "widefield-simple_readout.py"

    pulse_gen.stream_load(seq_file, seq_args_string)

    ### Set up the image display

    kpl.init_kplotlib(font_size=kpl.Size.SMALL)
    hor_label = "X"
    ver_label = "Y"
    cbar_label = "Counts"
    title = f"Widefield, {readout_laser}, {readout_ms} ms"
    imshow_kwargs = {
        "title": title,
        "x_label": hor_label,
        "y_label": ver_label,
        "cbar_label": cbar_label,
    }
    fig, ax = plt.subplots()

    ### Collect the data

    tb.init_safe_stop()

    camera.arm()
    pulse_gen.stream_start(1)
    img_array = camera.read()
    camera.disarm()
    kpl.imshow(ax, img_array, **imshow_kwargs)

    ### Clean up and save the data

    tb.reset_cfm(cxn)
    pos.set_xyz_on_nv(cxn, nv_sig)

    timestamp = tb.get_time_stamp()
    rawData = {
        "timestamp": timestamp,
        "nv_sig": nv_sig,
        "readout": readout,
        "readout-units": "ns",
        "title": title,
        "img_array": img_array.astype(int),
        "img_array-units": "counts",
    }

    filePath = tb.get_file_path(__file__, timestamp, nv_sig["name"])
    tb.save_figure(fig, filePath)
    tb.save_raw_data(rawData, filePath)

    return img_array


if __name__ == "__main__":
    file_name = "2023_10_11-17_02_33-johnson-nvref"
    data = tb.get_raw_data(file_name)
    img_array = np.array(data["img_array"])
    readout = data["readout"]
    img_array_kcps = (img_array / 1000) / (readout * 1e-9)

    kpl.init_kplotlib()
    fig, ax = plt.subplots()
    im = kpl.imshow(ax, img_array, cbar_label="counts")
    ax.set_xticks(range(0, 501, 100))
    # im = kpl.imshow(ax, img_array_kcps, extent=extent)
    # ax.set_xlim([124.5 - 15, 124.5 + 15])
    # ax.set_ylim([196.5 + 15, 196.5 - 15])

    # plot_coords = [
    #     [183.66, 201.62],
    #     [177.28, 233.34],
    #     [237.42, 314.84],
    #     [239.56, 262.84],
    #     [315.58, 203.56],
    # ]
    # cal_coords = [
    #     [139.5840657600651, 257.70994378810946],
    #     [324.4796398557366, 218.27466265286117],
    # ]
    # for coords in plot_coords:
    #     ax.plot(*coords, color="blue", marker="o", markersize=3)
    # for coords in cal_coords:
    #     ax.plot(*coords, color="green", marker="o", markersize=3)

    plt.show(block=True)
