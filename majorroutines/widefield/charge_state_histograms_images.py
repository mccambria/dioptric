# -*- coding: utf-8 -*-
"""
Illuminate an area, collecting onto the camera. Interleave a signal and control sequence
and plot the difference.

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
from scipy.optimize import curve_fit
from scipy.special import factorial

from analysis import bimodal_histogram
from analysis.bimodal_histogram import (
    ProbDist,
    determine_threshold,
    fit_bimodal_histogram,
)
from majorroutines.widefield import base_routine
from utils import common, widefield
from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import positioning as pos
from utils import tool_belt as tb
from utils.constants import NVSig, VirtualLaserKey
from utils.positioning import get_scan_1d as calculate_aom_voltage_range

# region Process and plotting functions


def process_and_extract(raw_data, prob_dist: ProbDist = ProbDist.COMPOUND_POISSON):
    ### Setup
    nv_list = raw_data["nv_list"]
    num_nvs = len(nv_list)
    counts = np.array(raw_data["counts"])
    sig_counts_lists = [counts[0, nv_ind].flatten() for nv_ind in range(num_nvs)]
    ref_counts_lists = [counts[1, nv_ind].flatten() for nv_ind in range(num_nvs)]
    num_reps = raw_data["num_reps"]
    num_runs = raw_data["num_runs"]
    num_shots = num_reps * num_runs

    ### Histograms and thresholding (without plotting)
    threshold_list = []
    readout_fidelity_list = []
    prep_fidelity_list = []

    for ind in range(num_nvs):
        sig_counts_list = sig_counts_lists[ind]
        ref_counts_list = ref_counts_lists[ind]

        # Only use ref counts for threshold determination
        popt = fit_bimodal_histogram(ref_counts_list, prob_dist, no_print=True)
        threshold, readout_fidelity = determine_threshold(
            popt, prob_dist, dark_mode_weight=0.5, no_print=True, ret_fidelity=True
        )
        threshold_list.append(threshold)
        readout_fidelity_list.append(readout_fidelity)
        if popt is not None:
            prep_fidelity = 1 - popt[0]
        else:
            prep_fidelity = np.nan
        prep_fidelity_list.append(prep_fidelity)

    # Prep fidelity
    print(f"readout_fidelity_list: {readout_fidelity_list}")
    print(f"prep_fidelity_list: {prep_fidelity_list}")

    # Report out the results
    threshold_list = np.array(threshold_list)
    readout_fidelity_list = np.array(readout_fidelity_list)
    prep_fidelity_list = np.array(prep_fidelity_list)

    # Scatter readout vs prep fidelity
    fig, ax = plt.subplots()
    kpl.plot_points(ax, readout_fidelity_list, prep_fidelity_list)
    ax.set_xlabel("Readout fidelity")
    ax.set_ylabel("NV- preparation fidelity")

    # Plot prep fidelity vs distance from center
    coords_key = "laser_INTE_520_aod"
    distances = []
    for nv in nv_list:
        coords = pos.get_nv_coords(nv, coords_key, drift_adjust=False)
        dist = np.sqrt((110 - coords[0]) ** 2 + (110 - coords[1]) ** 2)
        distances.append(dist)
    fig, ax = plt.subplots()
    kpl.plot_points(ax, distances, prep_fidelity_list)
    ax.set_xlabel("Distance from center frequencies (MHz)")
    ax.set_ylabel("NV- preparation fidelity")

    # Report averages
    avg_readout_fidelity = np.nanmean(readout_fidelity_list)
    std_readout_fidelity = np.nanstd(readout_fidelity_list)
    avg_prep_fidelity = np.nanmean(prep_fidelity_list)
    std_prep_fidelity = np.nanstd(prep_fidelity_list)
    str_readout_fidelity = tb.round_for_print(
        avg_readout_fidelity, std_readout_fidelity
    )
    str_prep_fidelity = tb.round_for_print(avg_prep_fidelity, std_prep_fidelity)
    print(f"Average readout fidelity: {str_readout_fidelity}")
    print(f"Average NV- preparation fidelity: {str_prep_fidelity}")

    ### Image extraction (without plotting histograms)
    if "img_arrays" not in raw_data:
        return

    laser_key = VirtualLaserKey.WIDEFIELD_CHARGE_READOUT
    laser_dict = tb.get_virtual_laser_dict(laser_key)
    readout_laser = laser_dict["physical_name"]
    readout = laser_dict["duration"]
    readout_ms = readout / 10**6

    img_arrays = raw_data["img_arrays"]
    mean_img_arrays = np.mean(img_arrays, axis=(1, 2, 3))
    sig_img_array = mean_img_arrays[0]
    ref_img_array = mean_img_arrays[1]
    diff_img_array = sig_img_array - ref_img_array
    img_arrays_to_save = [sig_img_array, ref_img_array, diff_img_array]

    return img_arrays_to_save, readout_fidelity_list, prep_fidelity_list


def main(
    nv_list,
    num_steps,
    num_reps,
    num_runs,
    aom_voltage_center,
    aom_voltage_range,
    vary_pol_laser=False,
    verify_charge_states=False,
    diff_polarize=False,
    diff_ionize=True,
    ion_include_inds=None,
):
    ### Initial setup
    seq_file = "charge_state_histograms_dynamical.py"
    aom_voltages = calculate_aom_voltage_range(
        aom_voltage_center, aom_voltage_range, num_steps
    )
    aom_voltages = aom_voltages.tolist()
    print(f"aom_voltages:{aom_voltages}")
    charge_prep_fn = base_routine.charge_prep_loop if verify_charge_states else None
    pulse_gen = tb.get_server_pulse_gen()

    def configure_laser_powers(step_ind):
        if vary_pol_laser:
            pol_laser_power = aom_voltages[step_ind]
            readout_laser_power = None
        else:
            pol_laser_power = None
            readout_laser_power = aom_voltages[step_ind]

        print(
            f"Inside configure_laser_powers - pol_laser_power: {pol_laser_power}, readout_laser_power: {readout_laser_power}"
        )

        return pol_laser_power, readout_laser_power

    def step_fn(step_ind):
        """Runs the experiment for a given step index."""
        print(f"Step Index: {step_ind}")
        pol_coords_list = widefield.get_coords_list(nv_list, VirtualLaserKey.CHARGE_POL)
        ion_coords_list = widefield.get_coords_list(
            nv_list, VirtualLaserKey.ION, include_inds=ion_include_inds
        )
        pol_laser_power, readout_laser_power = configure_laser_powers(step_ind)
        seq_args = [
            pol_coords_list,
            ion_coords_list,
            diff_polarize,
            diff_ionize,
            verify_charge_states,
            pol_laser_power,
            readout_laser_power,
        ]
        seq_args_string = tb.encode_seq_args(seq_args)
        print(f"Step Index: {step_ind}, Sequence Arguments: {seq_args}")
        pulse_gen.stream_load(seq_file, seq_args_string, num_reps)
        print(f"Successfully loaded and executed step {step_ind}")

    raw_data = base_routine.main(
        nv_list,
        num_steps,
        num_reps,
        num_runs,
        step_fn=step_fn,
        save_images=True,
        save_images_avg_reps=False,
        charge_prep_fn=charge_prep_fn,
        stream_load_in_run_fn=False,
    )

    ### Processing (no changes to this section)
    timestamp = dm.get_time_stamp()
    repr_nv_sig = widefield.get_repr_nv_sig(nv_list)
    repr_nv_name = repr_nv_sig.name

    try:
        imgs, img_figs = process_and_extract(raw_data)

        # Save the images
        title_suffixes = ["sig", "ref", "diff"]
        num_figs = len(img_figs)
        for ind in range(num_figs):
            fig = img_figs[ind]
            title = title_suffixes[ind]
            file_path = dm.get_file_path(__file__, timestamp, f"{repr_nv_name}-{title}")
            dm.save_figure(fig, file_path)
        sig_img_array, ref_img_array, diff_img_array = imgs
        keys_to_compress = ["sig_img_array", "ref_img_array", "diff_img_array"]

    except Exception:
        print(traceback.format_exc())
        sig_img_array = None
        ref_img_array = None
        diff_img_array = None
        keys_to_compress = None

    try:
        del raw_data["img_arrays"]
    except Exception:
        pass

    ### Save raw data

    file_path = dm.get_file_path(__file__, timestamp, repr_nv_name)
    raw_data |= {
        "timestamp": timestamp,
        "diff_polarize": diff_polarize,
        "diff_ionize": diff_ionize,
        "sig_img_array": sig_img_array,
        "ref_img_array": ref_img_array,
        "diff_img_array": diff_img_array,
        "img_array-units": "photons",
    }
    dm.save_raw_data(raw_data, file_path, keys_to_compress)

    tb.reset_cfm()

    return raw_data


if __name__ == "__main__":
    kpl.init_kplotlib()
    data = dm.get_raw_data(file_id=1688554695897, load_npz=False)
    process_and_extract(data)
    kpl.show(block=True)
