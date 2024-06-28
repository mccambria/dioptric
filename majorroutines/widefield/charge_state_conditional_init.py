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

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from scipy.ndimage import gaussian_filter, uniform_filter
from scipy.optimize import curve_fit
from scipy.special import factorial

from majorroutines.widefield import base_routine, optimize
from utils import common, widefield
from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import positioning as pos
from utils import tool_belt as tb
from utils.constants import LaserKey, NVSig
from utils.tool_belt import determine_threshold

# region Process and plotting functions


def process_and_plot(raw_data, mean_val=None):
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

    reps_vals = np.array(range(num_reps))

    fig, ax = plt.subplots()
    kpl.plot_points(ax, reps_vals, avg_num_nvn, yerr=avg_num_nvn_ste)
    ax.set_xlabel("Number of attempts")
    ax.set_ylabel("Mean number NV$^{-}$")
    ax.set_xlim((-0.5, 10.5))
    ax.set_xticks(np.array(range(11)))
    ax.set_yticks(np.array(range(9)) + 1)

    if mean_val is not None:
        ax.axhline(mean_val, color=kpl.KplColors.DARK_GRAY)

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
    # charge_prep_fn = None

    pulse_gen = tb.get_server_pulse_gen()

    ### Collect the data

    def run_fn(shuffled_step_inds):
        ion_coords_list = widefield.get_coords_list(nv_list, LaserKey.ION)
        pol_coords_list = widefield.get_coords_list(nv_list, LaserKey.CHARGE_POL)
        seq_args = [ion_coords_list, pol_coords_list]
        seq_args_string = tb.encode_seq_args(seq_args)
        pulse_gen.stream_load(seq_file, seq_args_string, num_reps)

    raw_data = base_routine.main(
        nv_list,
        num_steps,
        num_reps,
        num_runs,
        run_fn=run_fn,
        save_images=True,
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

    ### Just get a mean val
    data = dm.get_raw_data(file_id=1573560918521)
    nv_list = data["nv_list"]
    states = np.array(data["states"])[0]
    num_runs = data["num_runs"]
    num_reps = data["num_reps"]
    mean_val = np.sum(states) / (num_runs * num_reps)

    ### Main plot
    # data = dm.get_raw_data(file_id=1567957794147)
    # data = dm.get_raw_data(file_id=1570547331729)
    data = dm.get_raw_data(file_id=1573541903486)  # init ionized data
    process_and_plot(data, mean_val=mean_val)
    kpl.show(block=True)

    ### Inset images

    base_pixel_drift = [4, 35]
    buffer = 42

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

    # Single shot image from experiment
    data = dm.get_raw_data(file_id=1573541903486, use_cache=False, load_npz=True)
    num_runs = data["num_runs"]
    num_reps = data["num_reps"]

    # Identify good candidate images
    # states = np.array(data["states"])[0]
    # num_in_nvm = np.sum(states, axis=0)
    # for run_ind in range(num_runs):
    #     for rep_ind in range(9):
    #         if (
    #             num_in_nvm[run_ind, 0, 0] == 0
    #             and num_in_nvm[run_ind, 0, 1] == 7
    #             and num_in_nvm[run_ind, 0, rep_ind] == 9
    #             and num_in_nvm[run_ind, 0, rep_ind + 1] == 9
    #             and not (
    #                 states[:, run_ind, 0, rep_ind] == states[:, run_ind, 0, rep_ind + 1]
    #             ).all()
    #         ):
    #             print(run_ind, rep_ind)
    # sys.exit()

    nv_list = data["nv_list"]
    num_nvs = len(nv_list)
    img_arrays = np.array(data["img_arrays"])
    # bg_img_array = np.sum(img_arrays[0, :, 0, 0], axis=0) / num_runs
    # run_ind = 24
    # rep_indd = 5
    # rep_indd = 7
    # run_ind = 42
    # rep_indd = 3
    run_ind = 74
    rep_indd = 8
    for rep_ind in [0, 1, rep_indd, rep_indd + 1]:
        img_array = img_arrays[0, run_ind, 0, rep_ind]
        # img_array = np.mean(img_arrays, axis=(0, 1, 2, 3))
        widefield.replace_dead_pixel(img_array)
        states = np.array(data["states"])
        states = states[0, :, run_ind, 0, rep_ind]
        print(states)
        pixel_drift = np.array(data["pixel_drifts"])[run_ind]
        offset = [
            pixel_drift[1] - base_pixel_drift[1],
            pixel_drift[0] - base_pixel_drift[0],
        ]
        proc_img_array = widefield.crop_img_array(img_array, offset, buffer)
        proc_img_array = proc_img_array - bg_img_array
        drift = [
            pixel_drift[0] - buffer - offset[1],
            pixel_drift[1] - buffer - offset[0],
        ]

        # Downsampling / smoothing
        # downsample_factor = 6
        # proc_img_array = widefield.downsample_img_array(
        #     proc_img_array, downsample_factor
        # )
        # proc_img_array = np.repeat(proc_img_array, downsample_factor, axis=0)
        # proc_img_array = np.repeat(proc_img_array, downsample_factor, axis=1)
        proc_img_array = gaussian_filter(proc_img_array, 4)
        # proc_img_array = uniform_filter(proc_img_array, 5)

        fig, ax = plt.subplots()
        kpl.imshow(
            ax,
            proc_img_array,
            # clim=[0, 5],
            # clim=[0, 15],
            clim=[0.0, 0.2],
            no_cbar=True,
            # cmap=mpl.colormaps["gist_gray"],
        )
        ax.axis("off")
        nvm_inds = [ind for ind in range(num_nvs) if states[ind]]
        nv0_inds = [ind for ind in range(num_nvs) if not states[ind]]
        widefield.draw_circles_on_nvs(
            ax, nv_list, drift, linestyle="solid", no_legend=True, include_inds=nvm_inds
        )
        widefield.draw_circles_on_nvs(
            ax,
            nv_list,
            drift,
            linestyle="dashed",
            no_legend=True,
            include_inds=nv0_inds,
        )
    del img_arrays
    del data

    kpl.show(block=True)
