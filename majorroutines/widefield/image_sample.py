# -*- coding: utf-8 -*-
"""
Illuminate an area, collecting onto the camera

Created on April 9th, 2019

@author: mccambria
"""


import matplotlib.pyplot as plt
import numpy as np
import time
import labrad
import majorroutines.optimize as optimize
from utils import tool_belt as tb
from utils import common
from utils.constants import CollectionMode, CountFormat, LaserKey
from utils import kplotlib as kpl
from utils import positioning as pos
from scipy import ndimage


def single_nv(nv_sig):
    nv_list = [nv_sig]
    return nv_list_sub(nv_list, "single_nv")


def nv_list(nv_list):
    save_dict = {"nv_list": nv_list}
    return nv_list_sub(nv_list, "nv_list", save_dict)


def nv_list_sub(nv_list, caller_fn_name, save_dict=None):
    nv_sig = nv_list[0]
    laser_key = LaserKey.IMAGING
    laser_dict = nv_sig[laser_key]
    readout_laser = laser_dict["name"]
    adj_coords_list = [
        pos.adjust_coords_for_drift(nv_sig=nv, laser_name=readout_laser)
        for nv in nv_list
    ]
    x_coords = [coords[0] for coords in adj_coords_list]
    y_coords = [coords[1] for coords in adj_coords_list]
    num_reps = laser_dict["num_reps"]
    return main(nv_sig, x_coords, y_coords, caller_fn_name, num_reps, save_dict)


def widefield(nv_sig):
    laser_key = LaserKey.IMAGING
    laser_dict = nv_sig[laser_key]
    readout_laser = laser_dict["name"]
    center_coords = pos.adjust_coords_for_drift(
        nv_sig["coords"], laser_name=readout_laser
    )
    x_center, y_center, _ = center_coords
    x_coords = [x_center]
    y_coords = [y_center]
    return main(nv_sig, x_coords, y_coords, "widefield", 1)


def scanning(nv_sig, x_range, y_range, num_steps):
    center_coords = pos.adjust_coords_for_drift(nv_sig["coords"])
    x_center, y_center, _ = center_coords
    ret_vals = pos.get_scan_grid_2d(
        x_center, y_center, x_range, y_range, num_steps, num_steps
    )
    x_coords, y_coords, x_coords_1d, y_coords_1d, _ = ret_vals[0:2]
    x_coords = list(x_coords)
    y_coords = list(y_coords)
    save_dict = {
        "range_1": x_coords,
        "range_2": y_coords,
        "coords_1_1d": x_coords_1d,
        "coords_2_1d": y_coords_1d,
    }
    return main(nv_sig, x_coords, y_coords, "scanning", 1, save_dict)


def main(nv_sig, x_coords, y_coords, caller_fn_name, num_reps, save_dict=None):
    with common.labrad_connect() as cxn:
        ret_vals = main_with_cxn(
            cxn, nv_sig, x_coords, y_coords, caller_fn_name, num_reps, save_dict
        )
    return ret_vals


def main_with_cxn(
    cxn, nv_sig, x_coords, y_coords, caller_fn_name, num_reps, save_dict=None
):
    ### Some initial setup

    tb.reset_cfm(cxn)
    center_coords = pos.adjust_coords_for_drift(nv_sig["coords"])
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
    readout_ms = readout / 10**6
    readout_sec = readout / 10**9
    if caller_fn_name in ["scanning", "nv_list", "single_nv"]:
        seq_args = [list(x_coords), list(y_coords), readout, readout_laser]
        seq_file = "widefield-scanning_image_sample.py"
    elif caller_fn_name in ["widefield"]:
        seq_args = [readout, readout_laser]
        seq_file = "widefield-simple_readout.py"

    # print(seq_args)
    # return
    seq_args_string = tb.encode_seq_args(seq_args)
    pulse_gen.stream_load(seq_file, seq_args_string)

    ### Set up the image display

    kpl.init_kplotlib(font_size=kpl.Size.SMALL)
    cbar_label = "Counts"
    exposure = num_reps * readout_ms
    title = f"{caller_fn_name}, {readout_laser}, {exposure} ms"
    imshow_kwargs = {"title": title, "cbar_label": cbar_label}
    fig, ax = plt.subplots()

    ### Collect the data

    camera.arm()
    pulse_gen.stream_start(num_reps)
    img_array = camera.read()
    camera.disarm()
    kpl.imshow(ax, img_array, **imshow_kwargs)

    ### Clean up and save the data

    tb.reset_cfm(cxn)
    pos.set_xyz(cxn, center_coords)

    timestamp = tb.get_time_stamp()
    rawData = {
        "timestamp": timestamp,
        "caller_fn_name": caller_fn_name,
        "nv_sig": nv_sig,
        "center_coords": center_coords,
        "num_reps": num_reps,
        "readout": readout_ms,
        "readout-units": "ms",
        "title": title,
        "img_array": img_array.astype(int),
        "img_array-units": "counts",
    }
    if save_dict is not None:
        raw_data |= save_dict  # Add in the passed info to save

    filePath = tb.get_file_path(__file__, timestamp, nv_sig["name"])
    tb.save_figure(fig, filePath)
    tb.save_raw_data(rawData, filePath)

    return img_array


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
