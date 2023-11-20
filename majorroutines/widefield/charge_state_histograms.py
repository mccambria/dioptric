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


def create_histogram(sig_counts_list, ref_counts_list, num_reps, nv_sig):
    readout = nv_sig[LaserKey.IMAGING]["duration"]
    readout_ms = int(readout / 1e6)
    readout_s = readout / 1e9

    ### Histograms

    num_bins = 50

    labels = ["sig", "ref"]
    counts_lists = [sig_counts_list, ref_counts_list]
    fig, ax = plt.subplots()
    ax.set_title(f"Ionization hist, {num_bins} bins, {num_reps} reps")
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
    norm_sep_time = norm_sep / np.sqrt(readout_s)
    norm_sep = round(norm_sep, 3)
    norm_sep_time = round(norm_sep_time, 3)
    norm_sep_str = (
        f"Normalized separation:\n{norm_sep} / sqrt(shot)\n{norm_sep_time} / sqrt(s)"
    )
    print(norm_sep_str)
    kpl.anchored_text(ax, norm_sep_str, "center right", size=kpl.Size.SMALL)

    return fig


def main(nv_list, num_reps=100, diff_polarize=True, diff_ionize=True):
    ### Setup

    kpl.init_kplotlib()

    ### Collect the data

    with common.labrad_connect() as cxn:
        sig_img_array_list, ref_img_array_list, file_path = _collect_data(
            cxn, nv_list, num_reps, diff_polarize, diff_ionize
        )

    # Get the actual num_reps in case something went wrong
    num_reps = len(sig_img_array_list)

    ### Get the counts

    # Optimize pixel coords
    nv_sig = nv_list[0]
    pixel_coords = nv_sig["pixel_coords"]
    sig_img_array = np.sum(sig_img_array_list, axis=0) / num_reps
    pixel_coords = optimize.optimize_pixel_with_img_array(
        sig_img_array,
        pixel_coords,
        pixel_drift_adjust=True,
        set_pixel_drift=False,
        set_scanning_drift=False,
    )

    sig_counts_list = widefield.process_img_arrays(sig_img_array_list, nv_list)
    ref_counts_list = widefield.process_img_arrays(ref_img_array_list, nv_list)

    ### Make the histograms

    fig, sig_counts_list, ref_counts_list = create_histogram(
        sig_img_array_list, ref_img_array_list, num_reps, nv_sig
    )

    ### Save

    # Mask image arrays for compression
    for ind in range(num_reps):
        img_array = sig_img_array_list[ind]
        widefield.mask_img_array(img_array, nv_list)
        img_array = ref_img_array_list[ind]
        widefield.mask_img_array(img_array, nv_list)

    timestamp = dm.get_time_stamp()
    raw_data = {
        "timestamp": timestamp,
        "nv_list": nv_list,
        "num_reps": num_reps,
        "diff_polarize": diff_polarize,
        "diff_ionize": diff_ionize,
        "sig_counts_list": sig_counts_list,
        "ref_counts_list": ref_counts_list,
        "counts-units": "photons",
        "sig_img_array_list": sig_img_array_list,
        "ref_img_array_list": ref_img_array_list,
        "img_array-units": "counts",
    }

    sig_counts_list = np.array(sig_counts_list)
    ref_counts_list = np.array(ref_counts_list)
    dm.save_figure(fig, file_path)
    keys_to_compress = ["sig_img_array_list", "ref_img_array_list"]
    dm.save_raw_data(raw_data, file_path, keys_to_compress=keys_to_compress)


def _collect_data(
    cxn,
    nv_list,
    caller_fn_name,
    num_reps=1,
    diff_polarize=True,
    diff_ionize=True,
):
    ### Some initial setup

    # First NV to represent the others
    nv_sig = nv_list[0]

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
            if ind == 0:
                sig_img_array = np.copy(sub_img_array)
            else:
                sig_img_array += sub_img_array
            sig_img_array_list.append(sub_img_array)

            img_str = camera.read()
            sub_img_array = widefield.img_str_to_array(img_str)
            if ind == 0:
                ref_img_array = np.copy(sub_img_array)
            else:
                ref_img_array += sub_img_array
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
        widefield.imshow(ax, img_array, title=title)
        figs.append(fig)

    ### Clean up and return

    tb.reset_cfm(cxn)

    timestamp = dm.get_time_stamp()
    nv_name = nv_sig["name"]
    file_path = dm.get_file_path(__file__, timestamp, nv_name)
    for ind in range(3):
        fig = figs[ind]
        title_suffix = title_suffixes[ind]
        name = f"{nv_name}-{title_suffix}"
        fig_file_path = dm.get_file_path(__file__, timestamp, name)
        dm.save_figure(fig, fig_file_path)

    return sig_img_array_list, ref_img_array_list, file_path


if __name__ == "__main__":
    kpl.init_kplotlib()

    file_name = "2023_11_17-12_11_12-johnson-nv0_2023_11_09"
    # file_name = "2023_11_16-23_54_48-johnson-nv0_2023_11_09"

    data = dm.get_raw_data(file_name)
    nv_sig = data["nv_sig"]

    # nv_sig2 = nv_sig.copy()
    # nv_sig2["pixel_coords"] = [300, 300]
    # nv_list = [nv_sig, nv_sig2]

    # img_array = data["img_array"]
    # fig, ax = plt.subplots()
    # # widefield.mask_img_array(img_array, nv_list, drift_adjust=False)
    # # data["img_array"] = img_array
    # img_array = widefield.adus_to_photons(img_array)
    # widefield.imshow(ax, img_array)

    # timestamp = dm.get_time_stamp()
    # nv_name = nv_sig["name"]
    # file_path = dm.get_file_path(__file__, timestamp, nv_name)
    # keys_to_compress = ["img_array"]
    # dm.save_raw_data(data, file_path, keys_to_compress=keys_to_compress)

    # plt.show(block=True)

    sig_counts_list = np.array(data["sig_counts_list"])
    sig_counts_list = widefield.adus_to_photons(sig_counts_list)
    ref_counts_list = np.array(data["ref_counts_list"])
    ref_counts_list = widefield.adus_to_photons(ref_counts_list)
    num_reps = data["num_reps"]

    create_histogram(sig_counts_list, ref_counts_list, num_reps, nv_sig)

    thresh = 5050
    print(f"Red NV0: {(sig_counts_list < thresh).sum()}")
    print(f"Red NV-: {(sig_counts_list > thresh).sum()}")
    print(f"Green NV0: {(ref_counts_list < thresh).sum()}")
    print(f"Green NV-: {(ref_counts_list > thresh).sum()}")

    plt.show(block=True)
