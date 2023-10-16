# -*- coding: utf-8 -*-
"""
Scan over the designated area, collecting onto the camera

Created on April 9th, 2019

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


def main(nv_sig, range_1, range_2, num_steps, nv_minus_init=False):
    with common.labrad_connect() as cxn:
        ret_vals = main_with_cxn(
            cxn, nv_sig, range_1, range_2, num_steps, nv_minus_init
        )
    return ret_vals


def main_with_cxn(cxn, nv_sig, range_1, range_2, num_steps, nv_minus_init=False):
    ### Some initial setup

    config = common.get_config_dict()
    config_positioning = config["Positioning"]
    xy_units = config_positioning["xy_units"]
    axis_1_units = xy_units
    axis_2_units = xy_units

    tb.reset_cfm(cxn)
    center_coords = pos.adjust_coords_for_drift(nv_sig["coords"])
    x_center, y_center, z_center = center_coords
    optimize.prepare_microscope(cxn, nv_sig)
    camera = tb.get_server_camera(cxn)
    pulse_gen = tb.get_server_pulse_gen(cxn)

    laser_key = LaserKey.IMAGING
    laser_dict = nv_sig[laser_key]
    readout_laser = laser_dict["name"]
    tb.set_filter(cxn, nv_sig, laser_key)
    readout_power = tb.set_laser_power(cxn, nv_sig, laser_key)

    # Only support square grids at the moment
    num_steps_1 = num_steps
    num_steps_2 = num_steps
    total_num_samples = num_steps_1 * num_steps_2

    ### Set up the coordinates

    center_1 = x_center
    center_2 = y_center
    ret_vals = pos.get_scan_grid_2d(
        center_1, center_2, range_1, range_2, num_steps_1, num_steps_2
    )
    coords_1, coords_2, coords_1_1d, coords_2_1d, extent = ret_vals
    num_pixels = num_steps_1 * num_steps_2

    ### Load the pulse generator

    readout = laser_dict["readout_dur"]
    readout_us = readout / 10**3
    readout_ms = readout / 10**6
    readout_sec = readout / 10**9
    seq_args = [list(coords_1), list(coords_2), readout, readout_laser, readout_power]
    # print(seq_args)
    # return
    seq_args_string = tb.encode_seq_args(seq_args)
    seq_file = "widefield-scanning_image_sample.py"

    pulse_gen.stream_load(seq_file, seq_args_string)

    ### Set up the image display

    kpl.init_kplotlib(font_size=kpl.Size.SMALL)
    cbar_label = "Counts"
    title = f"Scanning, {readout_laser}, {readout_ms} ms"
    imshow_kwargs = {"title": title, "cbar_label": cbar_label}
    fig, ax = plt.subplots()

    ### Collect the data

    camera.arm()
    pulse_gen.stream_start(1)
    img_array = camera.read()
    camera.disarm()
    kpl.imshow(ax, img_array, **imshow_kwargs)

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
        "readout": readout,
        "readout-units": "ns",
        "title": title,
        "coords_1_1d": coords_1_1d,
        "coords_1_1d-units": axis_1_units,
        "coords_2_1d": coords_1_1d,
        "coords_2_1d-units": axis_2_units,
        "img_array": img_array.astype(int),
        "img_array-units": "counts",
    }

    filePath = tb.get_file_path(__file__, timestamp, nv_sig["name"])
    tb.save_figure(fig, filePath)
    tb.save_raw_data(rawData, filePath)

    return img_array, coords_1_1d, coords_2_1d


if __name__ == "__main__":
    file_name = "2023_09_11-13_52_01-johnson-nvref"

    data = tb.get_raw_data(file_name)
    img_array = np.array(data["img_array"])
    readout = data["readout"]
    img_array_kcps = (img_array / 1000) / (readout * 1e-9)
    extent = data["extent"] if "extent" in data else None

    kpl.init_kplotlib()
    fig, ax = plt.subplots()
    im = kpl.imshow(ax, img_array, cbar_label="ADUs")
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
