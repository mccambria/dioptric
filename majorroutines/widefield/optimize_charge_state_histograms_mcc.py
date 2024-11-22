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

# region Process and plotting functions


def process_and_plot(raw_data):
    nv_list = raw_data["nv_list"]
    num_nvs = len(nv_list)
    min_step_val = raw_data["min_step_val"]
    max_step_val = raw_data["max_step_val"]
    num_steps = raw_data["num_steps"]
    step_vals = np.linspace(min_step_val, max_step_val, num_steps)
    optimize_pol_or_readout = raw_data["optimize_pol_or_readout"]
    optimize_duration_or_amp = raw_data["optimize_duration_or_amp"]

    counts = np.array(raw_data["counts"])
    # [nv_ind, run_ind, steq_ind, rep_ind]
    condensed_counts = [
        [counts[nv_ind, :, step_ind, :].flatten() for step_ind in range(num_steps)]
        for nv_ind in range(num_nvs)
    ]
    condensed_counts = np.array(condensed_counts)

    prob_dist = ProbDist.COMPOUND_POISSON

    readout_fidelity_arr = np.empty((num_nvs))
    prep_fidelity_arr = np.empty((num_nvs))
    for nv_ind in range(num_nvs):
        for step_ind in range(num_steps):
            popt = fit_bimodal_histogram(condensed_counts[nv_ind, step_ind], prob_dist)
            threshold, readout_fidelity = determine_threshold(
                popt, prob_dist, dark_mode_weight=0.5, ret_fidelity=True
            )
            readout_fidelity_arr[nv_ind, step_ind] = readout_fidelity
            if popt is not None:
                prep_fidelity = 1 - popt[0]
            else:
                prep_fidelity = np.nan
            prep_fidelity_arr[nv_ind, step_ind] = prep_fidelity

    ### Plotting

    if optimize_pol_or_readout:
        if optimize_duration_or_amp:
            x_label = "Polarization duration"
        else:
            x_label = "Polarization amplitude"
    else:
        if optimize_duration_or_amp:
            x_label = "Readout duration"
        else:
            x_label = "Readout amplitude"

    print("Plotting results for first 10 NVs")
    for nv_ind in range(10):
        fig, ax = plt.subplots()
        kpl.plot_points(ax, step_vals, readout_fidelity_arr[nv_ind, :])
        ax.set_xlabel(x_label)
        ax.set_ylabel("Readout fidelity")
        ax.set_title(f"NV{nv_ind}")
        fig, ax = plt.subplots()
        kpl.plot_points(ax, step_vals, prep_fidelity_arr[nv_ind, :])
        ax.set_xlabel(x_label)
        ax.set_ylabel("Charge polarization fidelity")
        ax.set_title(f"NV{nv_ind}")


# endregion


def optimize_pol_duration(
    nv_list, num_steps, num_reps, num_runs, min_duration, max_duration
):
    return _main(
        nv_list, num_steps, num_reps, num_runs, min_duration, max_duration, True, True
    )


def optimize_pol_amp(nv_list, num_steps, num_reps, num_runs, min_amp, max_amp):
    return _main(nv_list, num_steps, num_reps, num_runs, min_amp, max_amp, True, False)


def optimize_readout_duration(
    nv_list, num_steps, num_reps, num_runs, min_duration, max_duration
):
    return _main(
        nv_list, num_steps, num_reps, num_runs, min_duration, max_duration, False, True
    )


def optimize_readout_amp(nv_list, num_steps, num_reps, num_runs, min_amp, max_amp):
    return _main(nv_list, num_steps, num_reps, num_runs, min_amp, max_amp, False, False)


def _main(
    nv_list,
    num_steps,
    num_reps,
    num_runs,
    min_step_val,
    max_step_val,
    optimize_pol_or_readout,
    optimize_duration_or_amp,
):
    ### Initial setup
    seq_file = "optimize_charge_state_histograms.py"
    step_vals = np.linspace(min_step_val, max_step_val, num_steps)

    pulse_gen = tb.get_server_pulse_gen()

    ### Collect the data

    def run_fn(shuffled_step_inds):
        shuffled_step_vals = step_vals[shuffled_step_inds]
        pol_coords_list, pol_duration_list, pol_amp_list = (
            widefield.get_pulse_parameter_lists(nv_list, VirtualLaserKey.CHARGE_POL)
        )
        ion_coords_list = widefield.get_coords_list(nv_list, VirtualLaserKey.ION)
        seq_args = [
            pol_coords_list,
            pol_duration_list,
            pol_amp_list,
            ion_coords_list,
            shuffled_step_vals,
            optimize_pol_or_readout,
            optimize_duration_or_amp,
        ]
        seq_args_string = tb.encode_seq_args(seq_args)
        pulse_gen.stream_load(seq_file, seq_args_string, num_reps)

    raw_data = base_routine.main(nv_list, num_steps, num_reps, num_runs, run_fn=run_fn)

    ### Processing

    timestamp = dm.get_time_stamp()
    repr_nv_sig = widefield.get_repr_nv_sig(nv_list)
    repr_nv_name = repr_nv_sig.name

    raw_data |= {
        "timestamp": timestamp,
        "min_step_val": min_step_val,
        "max_step_val": max_step_val,
        "optimize_pol_or_readout": optimize_pol_or_readout,
        "optimize_duration_or_amp": optimize_duration_or_amp,
    }

    try:
        process_and_plot(raw_data)

    except Exception:
        print(traceback.format_exc())

    try:
        del raw_data["img_arrays"]
    except Exception:
        pass

    ### Save and clean up

    file_path = dm.get_file_path(__file__, timestamp, repr_nv_name)
    dm.save_raw_data(raw_data, file_path)

    tb.reset_cfm()

    return raw_data


if __name__ == "__main__":
    kpl.init_kplotlib()
    data = dm.get_raw_data(file_id=1701595753308, load_npz=False)
    # data = dm.get_raw_data(file_id=1691569540529, load_npz=False)
    process_and_plot(data, do_plot_histograms=True)
    kpl.show(block=True)
