# -*- coding: utf-8 -*-
"""
Illuminate an area, collecting onto the camera. Interleave a signal and control sequence
and plot the difference

Created on April 9th, 2019

@author: mccambria
"""


import sys
import matplotlib.pyplot as plt
import numpy as np
from utils import tool_belt as tb
from utils import data_manager as dm
from utils import common
from utils import widefield
from utils.constants import LaserKey
from utils import kplotlib as kpl
from utils import positioning as pos
from utils import data_manager as dm
from majorroutines.widefield import optimize
from scipy import ndimage
import os
import time

repr_nv_ind = 0


def create_histogram(nv_sig, sig_counts_list, ref_counts_list):
    readout = nv_sig[LaserKey.CHARGE_READOUT]["duration"]
    readout_ms = int(readout / 1e6)
    readout_s = readout / 1e9

    ### Histograms

    num_bins = 50
    num_reps = len(ref_counts_list)

    labels = ["sig", "ref"]
    counts_lists = [sig_counts_list, ref_counts_list]
    fig, ax = plt.subplots()
    ax.set_title(f"Charge prep hist, {num_bins} bins, {num_reps} reps")
    ax.set_xlabel(f"Integrated counts")
    ax.set_ylabel("Number of occurrences")
    for ind in range(2):
        kpl.histogram(ax, counts_lists[ind], num_bins, label=labels[ind])
    ax.legend()

    # Calculate the normalized separation
    mean_std = (1 / 2) * np.sqrt(np.var(ref_counts_list) + np.var(sig_counts_list))
    mean_diff = np.mean(ref_counts_list) - np.mean(sig_counts_list)
    norm_sep = mean_diff / mean_std
    norm_sep_time = norm_sep / np.sqrt(readout_s)
    norm_sep = round(norm_sep, 3)
    norm_sep_time = round(norm_sep_time, 3)
    norm_sep_str = (
        f"Normalized separation:\n{norm_sep} / sqrt(shot)\n{norm_sep_time} / sqrt(s)"
    )
    print(norm_sep_str)
    kpl.anchored_text(ax, norm_sep_str, "center right", size=kpl.Size.SMALL)

    return fig


def main(nv_list, num_reps=100, diff_polarize=False, diff_ionize=True):
    ### Setup

    kpl.init_kplotlib()

    ### Collect the data

    with common.labrad_connect() as cxn:
        ret_vals = _collect_data(cxn, nv_list, num_reps, diff_polarize, diff_ionize)
    (
        sig_img_array_list,
        ref_img_array_list,
        sig_img_array,
        ref_img_array,
        diff_img_array,
        timestamp,
    ) = ret_vals

    ### Process

    sig_counts_lists, ref_counts_lists = process_data(
        nv_list, sig_img_array_list, ref_img_array_list
    )

    ### Plot and save

    num_nvs = len(nv_list)
    for ind in range(num_nvs):
        nv_sig = nv_list[ind]
        nv_name = nv_sig["name"]
        sig_counts_list = sig_counts_lists[ind]
        ref_counts_list = ref_counts_lists[ind]
        fig = create_histogram(nv_sig, sig_counts_list, ref_counts_list)
        file_path = dm.get_file_path(__file__, timestamp, nv_name)
        dm.save_figure(fig, file_path)

    repr_nv_name = nv_list[repr_nv_ind]["name"]
    keys_to_compress = [
        "sig_counts_lists",
        "ref_counts_lists",
        "sig_img_array",
        "ref_img_array",
        "diff_img_array",
    ]
    file_path = dm.get_file_path(__file__, timestamp, repr_nv_name)
    raw_data = {
        "timestamp": timestamp,
        "nv_list": nv_list,
        "num_reps": num_reps,
        "diff_polarize": diff_polarize,
        "diff_ionize": diff_ionize,
        "sig_counts_lists": sig_counts_lists,
        "ref_counts_lists": ref_counts_lists,
        "counts-units": "photons",
        "sig_img_array": sig_img_array,
        "ref_img_array": ref_img_array,
        "diff_img_array": diff_img_array,
        "img_array-units": "ADUs",
    }
    dm.save_raw_data(raw_data, file_path, keys_to_compress)


def process_data(nv_list, sig_img_array_list, ref_img_array_list):
    # Get the actual num_reps in case something went wrong
    num_reps = len(ref_img_array_list)
    num_nvs = len(nv_list)

    # Get a nice average image for optimization
    avg_img_array = np.sum(ref_img_array_list, axis=0) / num_reps
    optimize.optimize_pixel_with_img_array(avg_img_array, nv_list[repr_nv_ind])

    sig_counts_lists = [[] for ind in range(num_nvs)]
    ref_counts_lists = [[] for ind in range(num_nvs)]

    for nv_ind in range(num_nvs):
        nv_sig = nv_list[nv_ind]
        pixel_coords = widefield.get_nv_pixel_coords(nv_sig)
        sig_counts_list = sig_counts_lists[nv_ind]
        ref_counts_list = ref_counts_lists[nv_ind]
        for rep_ind in range(num_reps):
            img_array = sig_img_array_list[rep_ind]
            sig_counts = widefield.integrate_counts_from_adus(img_array, pixel_coords)
            sig_counts_list.append(sig_counts)
            img_array = ref_img_array_list[rep_ind]
            ref_counts = widefield.integrate_counts_from_adus(img_array, pixel_coords)
            ref_counts_list.append(ref_counts)

    sig_counts_lists = np.array(sig_counts_lists)
    ref_counts_lists = np.array(ref_counts_lists)
    return sig_counts_lists, ref_counts_lists


def _collect_data(cxn, nv_list, num_reps=100, diff_polarize=False, diff_ionize=True):
    ### Some initial setup

    # First NV to represent the others
    nv_sig = nv_list[repr_nv_ind]

    tb.reset_cfm(cxn)
    laser_key = LaserKey.CHARGE_READOUT
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

    seq_args = widefield.get_base_scc_seq_args(nv_list)
    seq_args.extend([diff_polarize, diff_ionize])
    seq_file = "charge_state_histograms.py"

    # print(seq_args)
    # print(seq_file)
    # return

    ### Collect the data

    sig_img_array_list = []
    ref_img_array_list = []

    camera.arm()

    seq_args_string = tb.encode_seq_args(seq_args)
    pulse_gen.stream_load(seq_file, seq_args_string, num_reps)
    pulse_gen.stream_start()

    try:
        for ind in range(num_reps):
            img_str = camera.read()
            sub_img_array = widefield.img_str_to_array(img_str)
            sig_img_array_list.append(sub_img_array)
            img_str = camera.read()
            sub_img_array = widefield.img_str_to_array(img_str)
            ref_img_array_list.append(sub_img_array)

    except Exception as exc:
        print(exc)
        num_reps = ind
        print(num_reps)

    finally:
        camera.disarm()

    ### Process and plot

    sig_img_array = np.sum(sig_img_array_list, axis=0)
    ref_img_array = np.sum(ref_img_array_list, axis=0)
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
        title = f"{readout_laser}, {readout_ms} ms, {title_suffix}"
        kpl.imshow(ax, img_array, title=title, cbar_label="ADUs")
        figs.append(fig)

    ### Clean up and return

    tb.reset_cfm(cxn)

    timestamp = dm.get_time_stamp()
    nv_name = nv_sig["name"]
    # Save sub figs
    for ind in range(3):
        fig = figs[ind]
        title_suffix = title_suffixes[ind]
        name = f"{nv_name}-{title_suffix}"
        fig_file_path = dm.get_file_path(__file__, timestamp, name)
        dm.save_figure(fig, fig_file_path)

    return (
        sig_img_array_list,
        ref_img_array_list,
        sig_img_array,
        ref_img_array,
        diff_img_array,
        timestamp,
    )


if __name__ == "__main__":
    kpl.init_kplotlib()

    file_name = "2023_11_20-17_38_07-johnson-nv0_2023_11_09"

    data = dm.get_raw_data(file_name)
    nv_list = data["nv_list"]
    sig_img_array_list = data["sig_img_array_list"]
    ref_img_array_list = data["ref_img_array_list"]

    sig_counts_list, ref_counts_list = process_data(
        nv_list, sig_img_array_list, ref_img_array_list
    )

    nv_sig = nv_list[0]
    create_histogram(nv_sig, sig_counts_list, ref_counts_list)

    # thresh = 5050
    # print(f"Red NV0: {(sig_counts_list < thresh).sum()}")
    # print(f"Red NV-: {(sig_counts_list > thresh).sum()}")
    # print(f"Green NV0: {(ref_counts_list < thresh).sum()}")
    # print(f"Green NV-: {(ref_counts_list > thresh).sum()}")

    plt.show(block=True)
