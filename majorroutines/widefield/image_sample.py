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


def single_nv(nv_sig, num_reps=1):
    nv_list = [nv_sig]
    return _nv_list_sub(nv_list, "single_nv", num_reps=num_reps)


def single_nv_ionization(nv_sig, num_reps=1):
    caller_fn_name = "single_nv_ionization"
    control_img_array = _charge_state_prep(
        nv_sig, caller_fn_name, num_reps, do_ionize=False
    )
    ionize_img_array = _charge_state_prep(
        nv_sig, caller_fn_name, num_reps, do_ionize=True
    )
    fig, ax = plt.subplots()
    diff = ionize_img_array - control_img_array
    kpl.imshow(ax, diff, title="Difference", cbar_label="Counts")


def single_nv_polarization(nv_sig, num_reps=1):
    caller_fn_name = "single_nv_polarization"
    control_img_array = _charge_state_prep(
        nv_sig, caller_fn_name, num_reps, do_polarize=False
    )
    polarize_img_array = _charge_state_prep(
        nv_sig, caller_fn_name, num_reps, do_polarize=True
    )
    fig, ax = plt.subplots()
    diff = polarize_img_array - control_img_array
    kpl.imshow(ax, diff, title="Difference", cbar_label="Counts")


def _charge_state_prep(
    nv_sig,
    caller_fn_name,
    num_reps=1,
    save_dict=None,
    do_polarize=False,
    do_ionize=False,
):
    return main(
        nv_sig,
        caller_fn_name,
        num_reps=num_reps,
        save_dict=save_dict,
        do_polarize=do_polarize,
        do_ionize=do_ionize,
    )


def nv_list(nv_list, num_reps=1):
    save_dict = {"nv_list": nv_list}
    return _nv_list_sub(nv_list, "nv_list", save_dict, num_reps)


def _nv_list_sub(nv_list, caller_fn_name, save_dict=None, num_reps=1):
    nv_sig = nv_list[0]
    laser_key = LaserKey.IMAGING
    laser_dict = nv_sig[laser_key]
    laser_name = laser_dict["name"]
    adj_coords_list = [pos.get_nv_coords(nv, laser_name) for nv in nv_list]
    x_coords = [coords[0] for coords in adj_coords_list]
    y_coords = [coords[1] for coords in adj_coords_list]
    return main(nv_sig, caller_fn_name, num_reps, x_coords, y_coords, save_dict)


def widefield(nv_sig, num_reps=1):
    return main(nv_sig, "widefield", num_reps)


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
    do_polarize=False,
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
            do_polarize,
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
    do_polarize=False,
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

    elif caller_fn_name == "widefield":
        seq_args = [readout, readout_laser]
        seq_file = "simple_readout-widefield.py"

    elif caller_fn_name in ["single_nv_ionization", "single_nv_polarization"]:
        ionization_laser = nv_sig[LaserKey.IONIZATION]["name"]
        ion_coords = pos.get_nv_coords(nv_sig, coords_suffix=ionization_laser)
        polarization_laser = nv_sig[LaserKey.POLARIZATION]["name"]
        pol_coords = pos.get_nv_coords(nv_sig, coords_suffix=polarization_laser)
        seq_args = [
            readout,
            readout_laser,
            do_polarize,
            do_ionize,
            ionization_laser,
            ion_coords,
            polarization_laser,
            pol_coords,
        ]
        seq_file = "simple_readout-charge_state_prep.py"

    # print(seq_args)
    # print(seq_file)
    # return

    ### Set up the image display

    kpl.init_kplotlib(font_size=kpl.Size.SMALL)
    cbar_label = "Counts"
    title = f"{caller_fn_name}, {readout_laser}, {readout_ms} ms"
    imshow_kwargs = {"title": title, "cbar_label": cbar_label}

    ### Collect the data

    camera.arm()
    seq_args_string = tb.encode_seq_args(seq_args)
    pulse_gen.stream_load(seq_file, seq_args_string, num_reps)
    pulse_gen.stream_start()
    for ind in range(num_reps):
        img_str = camera.read()
        sub_img_array = widefield_utils.img_str_to_array(img_str)
        if ind == 0:
            img_array = np.copy(sub_img_array)
        else:
            img_array += sub_img_array
    camera.disarm()

    img_array = img_array / num_reps
    fig, ax = plt.subplots()
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
