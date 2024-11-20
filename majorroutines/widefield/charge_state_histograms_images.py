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
    print(raw_data.keys())
    nv_list = raw_data["nv_list"]
    num_nvs = len(nv_list)
    counts = np.array(raw_data["counts"])
    num_steps = raw_data["num_steps"]
    num_reps = raw_data["num_reps"]
    num_runs = raw_data["num_runs"]
    num_shots = num_reps * num_runs

    # Correct the calculation of aom_voltages to ensure it's a list or array
    aom_voltage_center = 1.0
    aom_voltage_range = 0.1

    aom_voltages = calculate_aom_voltage_range(
        aom_voltage_center, aom_voltage_range, num_steps
    )
    aom_voltages = aom_voltages * 0.39
    # Assuming counts shape is (2, num_nvs, num_runs, num_steps, num_reps)
    counts = counts.reshape(2, num_nvs, num_runs, num_steps, -1)

    ### Histograms and thresholding per step
    readout_fidelity_per_step = []
    prep_fidelity_per_step = []

    for step_ind, voltage in zip(range(num_steps), aom_voltages):
        print(f"Step Index: {step_ind}, Voltage: {voltage}")
        readout_fidelity_list = []
        prep_fidelity_list = []

        for ind in range(num_nvs):
            sig_counts_list = counts[0, ind, :, step_ind].flatten()
            ref_counts_list = counts[1, ind, :, step_ind].flatten()

            # Only use ref counts for threshold determination
            popt = fit_bimodal_histogram(ref_counts_list, prob_dist, no_print=True)
            threshold, readout_fidelity = determine_threshold(
                popt, prob_dist, dark_mode_weight=0.5, no_print=True, ret_fidelity=True
            )
            if popt is not None:
                prep_fidelity = 1 - popt[0]
            else:
                prep_fidelity = np.nan

            readout_fidelity_list.append(readout_fidelity)
            prep_fidelity_list.append(prep_fidelity)

        readout_fidelity_per_step.append(np.nanmean(readout_fidelity_list))
        prep_fidelity_per_step.append(np.nanmean(prep_fidelity_list))

        # Plot fidelity vs prep fidelity for each step
        fig, ax = plt.subplots()
        ax.scatter(
            readout_fidelity_list, prep_fidelity_list, label=f"Voltage {voltage:.3f} V"
        )
        ax.set_xlabel("Readout Fidelity")
        ax.set_ylabel("Preparation Fidelity")
        ax.set_title(
            f"Fidelity vs Preparation Fidelity - Step {step_ind} (Voltage: {voltage:.3f} V)"
        )
        ax.legend()
        plt.show()

    # Convert lists to arrays for easier plotting
    readout_fidelity_per_step = np.array(readout_fidelity_per_step)
    prep_fidelity_per_step = np.array(prep_fidelity_per_step)

    # Plot average fidelity vs AOM voltage
    fig, ax = plt.subplots()
    ax.plot(
        aom_voltages,
        readout_fidelity_per_step,
        marker="o",
        label="Average Readout Fidelity",
    )
    ax.plot(
        aom_voltages,
        prep_fidelity_per_step,
        marker="s",
        label="Average Preparation Fidelity",
    )
    ax.set_xlabel("AOM Voltage (V)")
    ax.set_ylabel("Average Fidelity")
    ax.set_title("Average Fidelity vs AOM Voltage")
    ax.legend()
    plt.show()

    return readout_fidelity_per_step, prep_fidelity_per_step


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

        # print(
        #     f"Inside configure_laser_powers - pol_laser_power: {pol_laser_power}, readout_laser_power: {readout_laser_power}"
        # )

        return pol_laser_power, readout_laser_power

    def step_fn(step_ind):
        """Runs the experiment for a given step index."""
        # print(f"Step Index: {step_ind}")
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
        # print(f"Step Index: {step_ind}, Sequence Arguments: {seq_args}")
        pulse_gen.stream_load(seq_file, seq_args_string, num_reps)
        # print(f"Successfully loaded and executed step {step_ind}")

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
    # data = dm.get_raw_data(file_id=1688554695897, load_npz=False)
    data = dm.get_raw_data(file_id=1704871497302, load_npz=False)
    process_and_extract(data)
    kpl.show(block=True)
