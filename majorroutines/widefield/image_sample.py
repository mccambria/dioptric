# -*- coding: utf-8 -*-
"""
Illuminate an area, collecting onto the camera

Created on April 9th, 2019

@author: mccambria
"""


import matplotlib.pyplot as plt
import numpy as np
import majorroutines.optimize as optimize
from majorroutines.widefield.optimize_pixel_coords import prepare_microscope
from utils import tool_belt as tb
from utils import common
from utils import widefield as widefield_utils
from utils.constants import LaserKey
from utils import kplotlib as kpl
from utils import positioning as pos
from scipy import ndimage
import os
import time


def single_nv(nv_sig):
    nv_list = [nv_sig]
    return _nv_list_sub(nv_list, "single_nv")


def single_nv_ionization(nv_sig):
    nv_list = [nv_sig]
    num_runs = 1
    fn_name = "single_nv_ionization"
    for ind in range(num_runs):
        if ind == 0:
            control_img_array = _nv_list_sub(nv_list, fn_name, do_ionize=False)
            ionize_img_array = _nv_list_sub(nv_list, fn_name, do_ionize=True)
        else:
            control_img_array += _nv_list_sub(nv_list, fn_name, do_ionize=False)
            ionize_img_array += _nv_list_sub(nv_list, fn_name, do_ionize=True)
    control_img_array = control_img_array / num_runs
    ionize_img_array = ionize_img_array / num_runs
    fig, ax = plt.subplots()
    kpl.imshow(
        ax,
        ionize_img_array - control_img_array,
        title="Difference",
        cbar_label="Counts",
    )
    fig, ax = plt.subplots()
    kpl.imshow(
        ax, ionize_img_array / control_img_array, title="Contrast", cbar_label="Counts"
    )


def nv_list(nv_list):
    save_dict = {"nv_list": nv_list}
    return _nv_list_sub(nv_list, "nv_list", save_dict)


def _nv_list_sub(nv_list, caller_fn_name, save_dict=None, do_ionize=False):
    nv_sig = nv_list[0]
    if caller_fn_name == "single_nv_ionization":
        laser_key = LaserKey.IONIZATION
    else:
        laser_key = LaserKey.IMAGING
    laser_dict = nv_sig[laser_key]
    laser_name = laser_dict["name"]
    adj_coords_list = [pos.get_nv_coords(nv, laser_name) for nv in nv_list]
    x_coords = [coords[0] for coords in adj_coords_list]
    y_coords = [coords[1] for coords in adj_coords_list]
    num_reps = nv_sig[LaserKey.IMAGING]["num_reps"]
    return main(
        nv_sig, caller_fn_name, num_reps, x_coords, y_coords, save_dict, do_ionize
    )


def widefield(nv_sig):
    return main(nv_sig, "widefield")


def scanning(nv_sig, x_range, y_range, num_steps):
    laser_key = LaserKey.IMAGING
    laser_dict = nv_sig[laser_key]
    laser_name = laser_dict["name"]
    center_coords = pos.get_nv_coords(nv_sig, laser_name)
    x_center, y_center = center_coords[0:2]
    ret_vals = pos.get_scan_grid_2d(
        x_center, y_center, x_range, y_range, num_steps, num_steps
    )
    x_coords, y_coords, x_coords_1d, y_coords_1d, _ = ret_vals
    x_coords = list(x_coords)
    y_coords = list(y_coords)
    save_dict = {
        "x_range": x_range,
        "y_range": y_range,
        "x_coords_1d": x_coords_1d,
        "y_coords_1d": y_coords_1d,
    }
    num_reps = 1
    return main(nv_sig, "scanning", num_reps, x_coords, y_coords, save_dict)


def main(
    nv_sig,
    caller_fn_name,
    num_reps=1,
    x_coords=None,
    y_coords=None,
    save_dict=None,
    do_ionize=False,
):
    with common.labrad_connect() as cxn:
        ret_vals = main_with_cxn(
            cxn,
            nv_sig,
            caller_fn_name,
            num_reps,
            x_coords,
            y_coords,
            save_dict,
            do_ionize,
        )
    return ret_vals


def main_with_cxn(
    cxn,
    nv_sig,
    caller_fn_name,
    num_reps=1,
    x_coords=None,
    y_coords=None,
    save_dict=None,
    do_ionize=False,
):
    ### Some initial setup

    tb.reset_cfm(cxn)
    laser_key = LaserKey.IMAGING
    optimize.prepare_microscope(cxn, nv_sig)
    camera = tb.get_server_camera(cxn)
    pulse_gen = tb.get_server_pulse_gen(cxn)

    laser_dict = nv_sig[laser_key]
    readout_laser = laser_dict["name"]
    tb.set_filter(cxn, nv_sig, laser_key)

    pos.set_xyz_on_nv(cxn, nv_sig)

    ### Load the pulse generator

    readout = laser_dict["readout_dur"]
    readout_us = readout / 10**3
    readout_ms = readout / 10**6
    readout_sec = readout / 10**9

    if caller_fn_name in ["scanning", "nv_list", "single_nv"]:
        seq_args = [readout, readout_laser, list(x_coords), list(y_coords)]
        seq_file = "simple_readout-scanning.py"

    elif caller_fn_name == "single_nv_ionization":
        ionization_laser = nv_sig[LaserKey.IONIZATION]["name"]
        ion_coords = pos.get_nv_coords(nv_sig, coords_suffix=ionization_laser)

        polarization_laser = nv_sig[LaserKey.POLARIZATION]["name"]
        pol_coords = pos.get_nv_coords(nv_sig, coords_suffix=polarization_laser)

        seq_args = [
            readout,
            readout_laser,
            do_ionize,
            ionization_laser,
            ion_coords,
            polarization_laser,
            pol_coords,
        ]
        # print(seq_args)
        # return
        seq_file = "simple_readout-ionization.py"

    elif caller_fn_name == "widefield":
        seq_args = [readout, readout_laser]
        seq_file = "simple_readout-widefield.py"

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

    if caller_fn_name == "single_nv_ionization":
        num_runs = 1000
    else:
        num_runs = 1
    camera.arm()
    for ind in range(num_runs):
        # start = time.time()
        pulse_gen.stream_start()
        if ind == 0:
            img_array = camera.read()
        else:
            img_array += camera.read()
    camera.disarm()
    img_array = img_array / num_runs
    kpl.imshow(ax, img_array, **imshow_kwargs)

    ### Clean up and save the data

    tb.reset_cfm(cxn)

    timestamp = tb.get_time_stamp()
    raw_data = {
        "timestamp": timestamp,
        "caller_fn_name": caller_fn_name,
        "nv_sig": nv_sig,
        "num_reps": num_reps,
        "readout": readout_ms,
        "readout-units": "ms",
        "title": title,
        "img_array": img_array.astype(int),
        "img_array-units": "counts",
    }
    if save_dict is not None:
        raw_data |= save_dict  # Add in the passed info to save

    file_path = tb.get_file_path(__file__, timestamp, nv_sig["name"])
    tb.save_figure(fig, file_path)
    tb.save_raw_data(raw_data, file_path)

    return img_array


if __name__ == "__main__":
    # file_name = "2023_11_01-10_43_50-johnson-nv0_2023_10_30"
    # data = tb.get_raw_data(file_name)
    # img_array = np.array(data["img_array"])

    kpl.init_kplotlib()
    # fig, ax = plt.subplots()
    # im = kpl.imshow(ax, img_array, cbar_label="Counts")

    file_name = "2023_11_02-22_48_17-johnson-nv0_2023_11_02"
    data = tb.get_raw_data(file_name)
    control_img_array = np.array(data["img_array"])

    file_name = "2023_11_02-22_50_58-johnson-nv0_2023_11_02"
    data = tb.get_raw_data(file_name)
    ionize_img_array = np.array(data["img_array"])

    fig, ax = plt.subplots()
    kpl.imshow(
        ax,
        ionize_img_array - control_img_array,
        title="Difference",
        cbar_label="Counts",
    )

    plt.show(block=True)
