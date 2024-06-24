# -*- coding: utf-8 -*-
"""
Illuminate an area, collecting onto the camera. Interleave a signal and control sequence
and plot the difference

Created on Fall 2023

@author: mccambria
"""

import os
import sys
import time
import traceback

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit
from scipy.special import factorial
from skimage.metrics import structural_similarity as ssim

from majorroutines.widefield import base_routine, optimize
from utils import common, widefield
from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import positioning as pos
from utils import tool_belt as tb
from utils.constants import LaserKey, NVSig
from utils.tool_belt import determine_threshold

# region Process and plotting functions


def process_and_plot(raw_data):
    ### Setup

    nv_list = raw_data["nv_list"]
    counts = np.array(raw_data["counts"])[0]
    num_reps = raw_data["num_reps"]
    num_runs = raw_data["num_runs"]

    states = widefield.threshold_counts(nv_list, counts, dynamic_thresh=False)
    num_nvn = np.sum(states, axis=0)
    num_nvn = num_nvn[:, 0, :]  # Just one step
    avg_num_nvn = np.mean(num_nvn, axis=0)  # Average over runs
    avg_num_nvn_ste = np.std(num_nvn, axis=0) / np.sqrt(num_runs)

    reps_vals = np.array(range(num_reps)) + 1

    fig, ax = plt.subplots()
    kpl.plot_points(ax, reps_vals, avg_num_nvn, yerr=avg_num_nvn_ste)
    ax.set_xlabel("Number of attempts")
    ax.set_ylabel("Number in NV-")

    return fig


# endregion


def main(
    nv_list,
    num_reps,
    num_runs,
):
    ### Some initial setup
    seq_file = "charge_state_conditional_init.py"
    num_steps = 1

    charge_prep_fn = base_routine.charge_prep_no_verification

    pulse_gen = tb.get_server_pulse_gen()

    ### Collect the data

    def run_fn(shuffled_step_inds):
        pol_coords_list = widefield.get_coords_list(nv_list, LaserKey.CHARGE_POL)
        seq_args = [pol_coords_list]
        seq_args_string = tb.encode_seq_args(seq_args)
        pulse_gen.stream_load(seq_file, seq_args_string, num_reps)

    raw_data = base_routine.main(
        nv_list,
        num_steps,
        num_reps,
        num_runs,
        run_fn=run_fn,
        save_images=False,
        save_images_avg_reps=False,
        charge_prep_fn=charge_prep_fn,
        num_exps=1,
        # uwave_ind_list=[0, 1],  # MCC
    )

    ### Processing

    try:
        fig = process_and_plot(raw_data)
    except Exception:
        print(traceback.format_exc())
        fig = None

    ### Save and clean up

    timestamp = dm.get_time_stamp()
    repr_nv_sig = widefield.get_repr_nv_sig(nv_list)
    repr_nv_name = repr_nv_sig.name
    raw_data |= {
        "timestamp": timestamp,
        "img_array-units": "photons",
    }
    file_path = dm.get_file_path(__file__, timestamp, repr_nv_name)
    dm.save_raw_data(raw_data, file_path)
    if fig is not None:
        dm.save_figure(fig, file_path)

    tb.reset_cfm()

    return raw_data


if __name__ == "__main__":
    kpl.init_kplotlib()

    # data = dm.get_raw_data(file_id=1567068167346)
    # process_and_plot(data)

    base_pixel_drift = [7, 52]
    buffer = 30

    # Background image
    data = dm.get_raw_data(file_id=1567907418932, load_npz=True)
    pixel_drifts = np.array(data["pixel_drifts"])
    pixel_drift = np.mean(pixel_drifts, axis=0)
    offset = [
        pixel_drift[1] - base_pixel_drift[1] - 1,
        pixel_drift[0] - base_pixel_drift[0] + 1,
    ]
    ref_img_arrays = []
    for key in ["sig_img_array", "ref_img_array"]:
        img_array = np.array(data[key])
        widefield.replace_dead_pixel(img_array)
        img_array = widefield.crop_img_array(img_array, offset, buffer)
        ref_img_arrays.append(img_array)
    bg_img_array = ref_img_arrays[0]
    mask_img_array = ref_img_arrays[1] - ref_img_arrays[0]
    del data
    # fig, ax = plt.subplots()
    # kpl.imshow(ax, bg_img_array, no_cbar=True)
    # ax.axis("off")
    # kpl.show(block=True)

    # Single shot image from experiment
    data = dm.get_raw_data(file_id=1567068167346, use_cache=False, load_npz=True)
    # process_and_plot(data)
    img_arrays = np.array(data["img_arrays"])
    # bg_img_array = np.mean(img_arrays, axis=(0, 1, 2, 3))
    # bg_img_array = np.quantile(img_arrays, 0.1, axis=(0, 1, 2, 3))
    run_ind = 30
    rep_ind = 0
    img_array = img_arrays[0, run_ind, 0, rep_ind]
    # img_array = np.mean(img_arrays, axis=(0, 1, 2, 3))
    widefield.replace_dead_pixel(img_array)
    states = np.array(data["states"])
    print(states[0, :, run_ind, 0, rep_ind])
    pixel_drift = np.array(data["pixel_drifts"])[run_ind]
    offset = [
        pixel_drift[1] - base_pixel_drift[1],
        pixel_drift[0] - base_pixel_drift[0],
    ]
    cropped_img_array = widefield.crop_img_array(img_array, offset, buffer)
    # proc_img_array = bg_img_array - cropped_img_array
    # proc_img_array = cropped_img_array - bg_img_array
    proc_img_array = (cropped_img_array - bg_img_array) * mask_img_array
    # score, proc_img_array = ssim(
    #     cropped_img_array - bg_img_array,
    #     mask_img_array,
    #     data_range=mask_img_array.max() - mask_img_array.min(),
    #     full=True,
    # )
    proc_img_array = widefield.downsample_img_array(proc_img_array, 5)
    # proc_img_array = gaussian_filter(proc_img_array, 3)
    fig, ax = plt.subplots()
    # kpl.imshow(ax, proc_img_array, no_cbar=True)
    kpl.imshow(ax, proc_img_array, clim=[0, None], no_cbar=True)
    ax.axis("off")
    del img_arrays
    del data

    kpl.show(block=True)
