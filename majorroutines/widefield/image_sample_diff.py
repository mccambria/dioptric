# -*- coding: utf-8 -*-
"""
Illuminate an area, collecting onto the camera. Interleave a signal and control sequence
and plot the difference

Created on April 9th, 2019

@author: mccambria
"""


import matplotlib.pyplot as plt
import numpy as np
import majorroutines.optimize as optimize
from utils import tool_belt as tb
from utils import common
from utils import widefield as widefield_utils
from utils.constants import LaserKey
from utils import kplotlib as kpl
from utils import positioning as pos
from scipy import ndimage
import os
import time
from majorroutines.widefield.optimize_pixel_coords import (
    main_with_img_array as optimize_pixel_coords,
)


def single_nv_ionization(nv_sig, num_reps=1):
    caller_fn_name = "single_nv_ionization"
    return main(nv_sig, caller_fn_name, num_reps)


def single_nv_polarization(nv_sig, num_reps=1):
    caller_fn_name = "single_nv_polarization"
    return main(nv_sig, caller_fn_name, num_reps)


def main(nv_sig, caller_fn_name, num_reps=1, save_dict=None):
    with common.labrad_connect() as cxn:
        ret_vals = main_with_cxn(cxn, nv_sig, caller_fn_name, num_reps, save_dict)
    return ret_vals


def main_with_cxn(cxn, nv_sig, caller_fn_name, num_reps=1, save_dict=None):
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

    readout = laser_dict["duration"]
    readout_ms = readout / 10**6

    if caller_fn_name in ["single_nv_ionization", "single_nv_polarization"]:
        do_polarize = caller_fn_name == "single_nv_polarization"
        do_ionize = caller_fn_name == "single_nv_ionization"

        # Polarization
        pol_laser_dict = nv_sig[LaserKey.POLARIZATION]
        pol_laser = pol_laser_dict["name"]
        pol_coords = pos.get_nv_coords(nv_sig, coords_suffix=pol_laser)
        pol_duration = pol_laser_dict["duration"]

        # Ionization
        ion_laser_dict = nv_sig[LaserKey.IONIZATION]
        ion_laser = ion_laser_dict["name"]
        ion_coords = pos.get_nv_coords(nv_sig, coords_suffix=ion_laser)
        ion_duration = ion_laser_dict["duration"]

        seq_args = [
            readout,
            readout_laser,
            do_polarize,
            pol_laser,
            pol_coords,
            pol_duration,
            do_ionize,
            ion_laser,
            ion_coords,
            ion_duration,
        ]
        seq_file = "simple_readout-charge_state_prep-diff.py"

    # print(seq_args)
    # print(seq_file)
    # return

    ### Collect the data

    camera.arm()

    seq_args_string = tb.encode_seq_args(seq_args)
    pulse_gen.stream_load(seq_file, seq_args_string, num_reps)
    pulse_gen.stream_start()

    for ind in range(num_reps):
        img_str = camera.read()
        sub_img_array = widefield_utils.img_str_to_array(img_str)
        sig_img_array = (
            np.copy(sub_img_array) if ind == 0 else sig_img_array + sub_img_array
        )

        img_str = camera.read()
        sub_img_array = widefield_utils.img_str_to_array(img_str)
        ref_img_array = (
            np.copy(sub_img_array) if ind == 0 else ref_img_array + sub_img_array
        )

    camera.disarm()

    ### Process and plot

    kpl.init_kplotlib(font_size=kpl.Size.SMALL)

    diff_img_array = sig_img_array - ref_img_array
    sig_img_array = sig_img_array / num_reps
    ref_img_array = ref_img_array / num_reps
    diff_img_array = diff_img_array / num_reps

    img_arrays = [sig_img_array, ref_img_array, diff_img_array]
    title_suffices = ["sig", "ref", "diff"]
    figs = []
    for ind in range(3):
        img_array = img_arrays[ind]
        title_suffix = title_suffices[ind]
        fig, ax = plt.subplots()
        title = f"{caller_fn_name}, {readout_laser}, {readout_ms} ms, {title_suffix}"
        kpl.imshow(ax, img_array, title=title, cbar_label="Counts")
        figs.append(fig)

    ### Get counts

    bg_offset = [10, 10]
    img_arrays = [sig_img_array, ref_img_array]
    titles = ["Signal", "Reference"]

    for ind in range(2):
        img_array = img_arrays[ind]
        title = titles[ind]

        nv_pixel_coords = optimize_pixel_coords(
            img_array,
            nv_sig,
            set_scanning_drift=False,
            set_pixel_drift=False,
            pixel_drift_adjust=False,
        )
        nv_counts = widefield_utils.counts_from_img_array(
            img_array, nv_pixel_coords, drift_adjust=False
        )
        bg_pixel_coords = [
            nv_pixel_coords[0] + bg_offset[0],
            nv_pixel_coords[1] + bg_offset[1],
        ]
        bg_counts = widefield_utils.counts_from_img_array(
            img_array, bg_pixel_coords, drift_adjust=False
        )

        print(title)
        print(f"nv_counts: {nv_counts}")
        print(f"bg_counts: {bg_counts}")
        print(f"diff: {nv_counts - bg_counts}")
        print()

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
        "sig_img_array": sig_img_array.astype(int),
        "ref_img_array": ref_img_array.astype(int),
        "diff_img_array": diff_img_array.astype(float),
        "img_array-units": "counts",
    }
    if save_dict is not None:
        raw_data |= save_dict  # Add in the passed info to save

    nv_name = nv_sig["name"]
    file_path = tb.get_file_path(__file__, timestamp, nv_name)
    keys_to_compress = ["sig_img_array", "ref_img_array", "diff_img_array"]
    tb.save_raw_data(raw_data, file_path, keys_to_compress=keys_to_compress)
    for ind in range(3):
        fig = figs[ind]
        title_suffix = title_suffices[ind]
        name = f"{nv_name}-{title_suffix}"
        file_path = tb.get_file_path(__file__, timestamp, name)
        tb.save_figure(fig, file_path)

    return img_array


if __name__ == "__main__":
    # file_name = "2023_11_01-10_43_50-johnson-nv0_2023_10_30"
    # data = tb.get_raw_data(file_name)
    # img_array = np.array(data["img_array"])

    kpl.init_kplotlib()
    # fig, ax = plt.subplots()
    # im = kpl.imshow(ax, img_array, cbar_label="Counts")

    file_name = "2023_11_07-17_13_43-johnson-nv2_2023_11_07"
    data = tb.get_raw_data(file_name)
    signal_img_array = np.array(data["img_array"])
    pixel_coords = optimize_pixel_coords(
        signal_img_array,
        data["nv_sig"],
        set_scanning_drift=False,
        set_pixel_drift=False,
        pixel_drift_adjust=False,
    )
    print(
        widefield_utils.counts_from_img_array(
            signal_img_array, pixel_coords, drift_adjust=False
        )
    )

    file_name = "2023_11_07-17_14_32-johnson-nv2_2023_11_07"
    data = tb.get_raw_data(file_name)
    control_img_array = np.array(data["img_array"])
    pixel_coords = optimize_pixel_coords(
        control_img_array,
        data["nv_sig"],
        set_scanning_drift=False,
        set_pixel_drift=False,
        pixel_drift_adjust=False,
    )
    print(
        widefield_utils.counts_from_img_array(
            control_img_array, pixel_coords, drift_adjust=False
        )
    )

    bg_coords = [pixel_coords[0] + 10, pixel_coords[1] + 10]
    print(
        widefield_utils.counts_from_img_array(
            control_img_array, bg_coords, drift_adjust=False
        )
    )

    diff = signal_img_array - control_img_array

    fig, ax = plt.subplots()
    kpl.imshow(ax, diff, title="Difference", cbar_label="Counts")
    # kpl.imshow(ax, signal_img_array, title="Signal", cbar_label="Counts")
    # kpl.imshow(ax, control_img_array, title="Control", cbar_label="Counts")

    plt.show(block=True)
