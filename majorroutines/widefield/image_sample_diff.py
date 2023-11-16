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


def charge_state_histogram(nv_sig, num_reps=100):
    ### Setup

    caller_fn_name = "single_nv_ionization"
    pixel_coords = nv_sig["pixel_coords"]
    readout_s = nv_sig[LaserKey.IMAGING]["duration"] * 1e-9

    kpl.init_kplotlib()

    ### Collect the data

    with common.labrad_connect() as cxn:
        sig_img_array_list, ref_img_array_list, file_path, num_reps = main_with_cxn(
            cxn, nv_sig, caller_fn_name, num_reps, separate_images=True
        )

    ### Turn it into counts

    sig_counts_list = []
    ref_counts_list = []

    for ind in range(num_reps):
        sig_img_array = sig_img_array_list[ind]
        sig_counts = widefield_utils.counts_from_img_array(
            sig_img_array, pixel_coords, drift_adjust=False
        )
        ref_img_array = ref_img_array_list[ind]
        ref_counts = widefield_utils.counts_from_img_array(
            ref_img_array, pixel_coords, drift_adjust=False
        )
        sig_counts_list.append(sig_counts)
        ref_counts_list.append(ref_counts)

    ### Histograms

    num_bins = 50

    labels = ["sig", "ref"]
    counts_lists = [sig_counts_list, ref_counts_list]
    fig, ax = plt.subplots()
    ax.set_title(f"Ionization histogram, {num_bins} bins, {num_reps} reps")
    ax.set_xlabel(f"Integrated counts")
    ax.set_ylabel("Number of occurrences")
    for ind in range(2):
        kpl.histogram(
            ax, counts_lists[ind], kpl.HistType.STEP, nbins=num_bins, label=labels[ind]
        )
    ax.legend()

    # Calculate the normalized separation
    mean_std = (1 / 2) * np.sqrt(np.var(ref_counts_list) + np.var(sig_counts_list))
    mean_diff = np.mean(ref_counts_list) - np.mean(sig_counts_list)
    norm_sep = mean_diff / mean_std
    norm_sep = round(norm_sep, 3)
    norm_sep_str = f"Normalized separation:\n{norm_sep} / sqrt(shot)\n{norm_sep / np.sqrt(readout_s)} / sqrt(s)"
    print(norm_sep_str)
    kpl.anchored_text(ax, norm_sep_str, "center right")

    ### Save

    timestamp = tb.get_time_stamp()
    raw_data = {
        "timestamp": timestamp,
        "caller_fn_name": caller_fn_name,
        "nv_sig": nv_sig,
        "num_reps": num_reps,
        "sig_img_array_list": sig_img_array_list,
        "ref_img_array_list": ref_img_array_list,
        "img_array-units": "counts",
    }

    sig_img_array_list = np.array(sig_img_array_list)
    ref_img_array_list = np.array(ref_img_array_list)
    keys_to_compress = ["sig_img_array_list", "ref_img_array_list"]
    tb.save_raw_data(raw_data, file_path, keys_to_compress=keys_to_compress)
    tb.save_figure(fig, file_path)


def single_nv_ionization(nv_sig, num_reps=1):
    caller_fn_name = "single_nv_ionization"
    with common.labrad_connect() as cxn:
        return main_with_cxn(cxn, nv_sig, caller_fn_name, num_reps)


def single_nv_polarization(nv_sig, num_reps=1):
    caller_fn_name = "single_nv_polarization"
    with common.labrad_connect() as cxn:
        return main_with_cxn(cxn, nv_sig, caller_fn_name, num_reps)


def main_with_cxn(
    cxn,
    nv_sig,
    caller_fn_name,
    num_reps=1,
    save_dict=None,
    separate_images=False,
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

    if separate_images:
        sig_img_array_list = []
        ref_img_array_list = []

    camera.arm()

    seq_args_string = tb.encode_seq_args(seq_args)
    pulse_gen.stream_load(seq_file, seq_args_string, num_reps)
    pulse_gen.stream_start()

    try:
        for ind in range(num_reps):
            img_str = camera.read()
            sub_img_array = widefield_utils.img_str_to_array(img_str)
            if ind == 0:
                sig_img_array = np.copy(sub_img_array)
            else:
                sig_img_array += sub_img_array
            if separate_images:
                sig_img_array_list.append(sub_img_array)

            img_str = camera.read()
            sub_img_array = widefield_utils.img_str_to_array(img_str)
            if ind == 0:
                ref_img_array = np.copy(sub_img_array)
            else:
                ref_img_array += sub_img_array
            if separate_images:
                ref_img_array_list.append(sub_img_array)

    except Exception as exc:
        print(exc)
        num_reps = ind
        print(num_reps)

    finally:
        camera.disarm()

    ### Process and plot

    diff_img_array = sig_img_array - ref_img_array
    sig_img_array = sig_img_array / num_reps
    ref_img_array = ref_img_array / num_reps
    diff_img_array = diff_img_array / num_reps

    kpl.init_kplotlib()

    img_arrays = [sig_img_array, ref_img_array, diff_img_array]
    title_suffixes = ["sig", "ref", "diff"]
    figs = []
    for ind in range(3):
        img_array = img_arrays[ind]
        title_suffix = title_suffixes[ind]
        fig, ax = plt.subplots()
        title = f"{caller_fn_name}, {readout_laser}, {readout_ms} ms, {title_suffix}"
        widefield_utils.imshow(ax, img_array, title=title)
        figs.append(fig)

    ### Get counts

    if False:
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

    ### Clean up and return

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
        title_suffix = title_suffixes[ind]
        name = f"{nv_name}-{title_suffix}"
        fig_file_path = tb.get_file_path(__file__, timestamp, name)
        tb.save_figure(fig, fig_file_path)

    if separate_images:
        return sig_img_array_list, ref_img_array_list, file_path, num_reps
    else:
        return sig_img_array, ref_img_array


if __name__ == "__main__":
    kpl.init_kplotlib()

    # file_name = "2023_11_07-17_52_33-johnson-nv2_2023_11_07"
    file_name = "2023_11_09-22_57_44-johnson-nv0_2023_11_09"

    data = tb.get_raw_data(file_name)
    img_array = np.array(data["img_array"])
    # pixel_coords = optimize_pixel_coords(
    #     img_array,
    #     data["nv_sig"],
    #     set_scanning_drift=False,
    #     set_pixel_drift=False,
    #     pixel_drift_adjust=False,
    # )
    pixel_coords = [299.867, 253.762]
    print(
        widefield_utils.counts_from_img_array(
            img_array, pixel_coords, drift_adjust=False
        )
    )
    bg_coords = [pixel_coords[0] + 15, pixel_coords[1] + 15]
    print(
        widefield_utils.counts_from_img_array(img_array, bg_coords, drift_adjust=False)
    )

    fig, ax = plt.subplots()
    im = widefield_utils.imshow(ax, img_array)

    plt.show(block=True)

    # # call the function: Charge State Histogram
    # num_reps = 10
    # charge_state_histogram(num_reps)
    # plt.show()
