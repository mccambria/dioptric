# -*- coding: utf-8 -*-
"""
Illuminate an area, collecting onto the camera. Interleave a signal and control sequence
and plot the difference

Created on Fall 2024

@author: Saroj Chand
"""

import os
import sys
import time
import traceback

import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed

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


def find_optimal_value_geom_mean(
    step_vals, prep_fidelity, readout_fidelity, goodness_of_fit, weights=(1, 1, 1)
):
    """
    Finds the optimal step value using a weighted geometric mean of fidelities and goodness of fit.

    Parameters:
    ----------
    step_vals : np.ndarray
        Array of step values.
    prep_fidelity : np.ndarray
        Array of preparation fidelities.
    readout_fidelity : np.ndarray
        Array of readout fidelities.
    goodness_of_fit : np.ndarray
        Array of goodness of fit (chi-squared values).
    weights : tuple(float, float, float)
        Weights for readout fidelity, prep fidelity, and goodness of fit, respectively.

    Returns:
    -------
    optimal_step_val : float
        The step value corresponding to the optimal combined score.
    """
    w1, w2, w3 = weights

    # Normalize metrics
    norm_prep_fidelity = (prep_fidelity - np.nanmin(prep_fidelity)) / (
        np.nanmax(prep_fidelity) - np.nanmin(prep_fidelity)
    )
    norm_readout_fidelity = (readout_fidelity - np.nanmin(readout_fidelity)) / (
        np.nanmax(readout_fidelity) - np.nanmin(readout_fidelity)
    )
    norm_goodness = (goodness_of_fit - np.nanmin(goodness_of_fit)) / (
        np.nanmax(goodness_of_fit) - np.nanmin(goodness_of_fit)
    )
    inverted_goodness = 1 - norm_goodness  # Minimize goodness of fit

    # Compute weighted geometric mean
    combined_score = (
        (norm_readout_fidelity**w1) * (norm_prep_fidelity**w2) * (inverted_goodness**w3)
    ) ** (1 / (w1 + w2 + w3))

    # Find the step value corresponding to the maximum combined score
    max_index = np.nanargmax(combined_score)
    optimal_step_val = step_vals[max_index]

    return optimal_step_val


def process_and_plot(raw_data):
    nv_list = raw_data["nv_list"]
    num_nvs = len(nv_list)
    min_step_val = raw_data["min_step_val"]
    max_step_val = raw_data["max_step_val"]
    num_steps = raw_data["num_steps"]
    optimize_pol_or_readout = raw_data["optimize_pol_or_readout"]
    optimize_type = raw_data["optimize_type"]

    if optimize_type == "both":
        # Dual-parameter optimization
        duration_vals = np.linspace(min_step_val[0], max_step_val[0], num_steps)
        amp_vals = np.linspace(min_step_val[1], max_step_val[1], num_steps)
        duration_vals = duration_vals.astype(int)
        step_vals = np.array(np.meshgrid(duration_vals, amp_vals)).T.reshape(-1, 2)
    else:
        # Single-parameter optimization
        step_vals = np.linspace(min_step_val, max_step_val, num_steps)
        if optimize_type == "duration":
            step_vals = step_vals.astype(int)

    counts = np.array(raw_data["counts"])
    ref_exp_ind = 1
    condensed_counts = [
        [
            counts[ref_exp_ind, nv_ind, :, step_ind, :].flatten()
            for step_ind in range(num_steps)
        ]
        for nv_ind in range(num_nvs)
    ]
    condensed_counts = np.array(condensed_counts)

    prob_dist = ProbDist.COMPOUND_POISSON

    # Function to process a single NV and step
    def process_nv_step(nv_ind, step_ind):
        counts_data = condensed_counts[nv_ind, step_ind]
        popt, norm_res = fit_bimodal_histogram(counts_data, prob_dist)

        if popt is None:
            return np.nan, np.nan, np.nan, np.nan
        # Threshold, prep and readout fidelity
        threshold, readout_fidelity = determine_threshold(
            popt, prob_dist, dark_mode_weight=0.5, ret_fidelity=True
        )
        prep_fidelity = 1 - popt[0]  # Population weight of dark state

        return readout_fidelity, prep_fidelity, norm_res

    # Parallel processing --> n_jobs=-1: using all available cores.
    results = Parallel(n_jobs=-1)(
        delayed(process_nv_step)(nv_ind, step_ind)
        for nv_ind in range(num_nvs)
        for step_ind in range(num_steps)
    )

    # Reshape results into arrays
    results = np.array(results).reshape(num_nvs, num_steps, 3)
    readout_fidelity_arr = results[:, :, 0]
    prep_fidelity_arr = results[:, :, 1]
    goodness_of_fit_arr = results[:, :, 2]

    ### Plotting
    if optimize_type == "both":
        # Dual-parameter heatmap
        duration_vals = np.linspace(min_step_val[0], max_step_val[0], num_steps)
        amp_vals = np.linspace(min_step_val[1], max_step_val[1], num_steps)
        avg_readout_fidelity = np.nanmean(readout_fidelity_arr, axis=0).reshape(
            len(duration_vals), len(amp_vals)
        )

        plt.figure(figsize=(8, 6))
        plt.title("Average Readout Fidelity Heatmap")
        plt.xlabel("Amplitude")
        plt.ylabel("Duration")
        plt.imshow(
            avg_readout_fidelity,
            extent=[
                amp_vals[0],
                amp_vals[-1],
                duration_vals[0],
                duration_vals[-1],
            ],
            aspect="auto",
            origin="lower",
            cmap="viridis",
        )
        plt.colorbar(label="Avg. Readout Fidelity")
        plt.show()
    else:
        # Single-parameter optimization
        if optimize_pol_or_readout:
            if optimize_type == "duration":
                x_label = "Polarization Duration"
            else:
                x_label = "Polarization Amplitude"
        else:
            if optimize_type == "duration":
                x_label = "Readout Duration"
            else:
                x_label = "Readout Amplitude"

        # Calculate averages
        avg_readout_fidelity = np.nanmean(readout_fidelity_arr, axis=0)
        avg_prep_fidelity = np.nanmean(prep_fidelity_arr, axis=0)
        avg_goodness_of_fit = np.nanmean(goodness_of_fit_arr, axis=0)

        # Calculate the optimal step value
        optimal_step_val = find_optimal_value_geom_mean(
            step_vals,
            avg_readout_fidelity,
            avg_prep_fidelity,
            avg_goodness_of_fit,
        )

        # Plot average readout and prep fidelity
        fig, ax1 = plt.subplots(figsize=(7, 5))
        ax1.plot(
            step_vals,
            avg_readout_fidelity,
            label="Avg. Readout Fidelity",
            color="blue",
        )
        ax1.plot(
            step_vals,
            avg_prep_fidelity,
            label="Avg. Prep Fidelity",
            color="orange",
        )
        ax1.set_xlabel(x_label)
        ax1.set_ylabel("Fidelity")
        ax1.tick_params(axis="y")

        # Plot average goodness of fit (chi-squared)
        ax2 = ax1.twinx()
        ax2.plot(
            step_vals,
            avg_goodness_of_fit,
            color="green",
            linestyle="--",
            label="Avg. Goodness of Fit (chi-squared)",
        )
        ax2.set_ylabel("Goodness of Fit (chi-squared)", color="green")

        ax2.tick_params(axis="y", labelcolor="green")
        ax2.axvline(
            optimal_step_val,
            color="red",
            linestyle="--",
            label=f"Optimal Step Val: {optimal_step_val:.2f}",
        )

        # Combine legends
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax2.legend(
            lines_1 + lines_2, labels_1 + labels_2, loc="upper right", fontsize=9
        )

        # Title and layout
        ax1.set_title("Average Metrics Across All NVs", fontsize=12)
        fig.tight_layout()
        plt.show()


# endregion


def optimize_pol_duration(
    nv_list, num_steps, num_reps, num_runs, min_duration, max_duration
):
    return _main(
        nv_list,
        num_steps,
        num_reps,
        num_runs,
        min_duration,
        max_duration,
        True,
        "duration",
    )


def optimize_pol_amp(nv_list, num_steps, num_reps, num_runs, min_amp, max_amp):
    return _main(nv_list, num_steps, num_reps, num_runs, min_amp, max_amp, True, "amp")


def optimize_readout_duration(
    nv_list, num_steps, num_reps, num_runs, min_duration, max_duration
):
    return _main(
        nv_list,
        num_steps,
        num_reps,
        num_runs,
        min_duration,
        max_duration,
        False,
        "duration",
    )


def optimize_readout_amp(nv_list, num_steps, num_reps, num_runs, min_amp, max_amp):
    return _main(nv_list, num_steps, num_reps, num_runs, min_amp, max_amp, False, "amp")


def optimize_readout_amp_and_duration(
    nv_list, num_steps, num_reps, num_runs, min_duration, max_duration, min_amp, max_amp
):
    return _main(
        nv_list,
        num_steps,
        num_reps,
        num_runs,
        (min_duration, min_amp),
        (max_duration, max_amp),
        False,
        "both",
    )


def _main(
    nv_list,
    num_steps,
    num_reps,
    num_runs,
    min_step_val,
    max_step_val,
    optimize_pol_or_readout,
    optimize_type,
):
    """
    Main optimization logic to handle different types of parameter sweeps.

    Parameters
    ----------
    nv_list : list
        List of NVs to optimize.
    num_steps : int
        Number of steps in the sweep.
    num_reps : int
        Number of repetitions for each step.
    num_runs : int
        Number of runs in the experiment.
    min_step_val : int, float, or tuple
        Minimum value(s) for the parameter(s) being swept.
    max_step_val : int, float, or tuple
        Maximum value(s) for the parameter(s) being swept.
    optimize_pol_or_readout : bool
        True for polarization optimization, False for readout optimization.
    optimize_type : str
        Optimization type: "duration", "amp", or "both".
    """

    seq_file = "optimize_amp_duration_charge_state_histograms.py"

    if optimize_type == "both":
        # Create 2D grid for duration and amplitude
        duration_vals = np.linspace(min_step_val[0], max_step_val[0], num_steps).astype(
            int
        )
        amp_vals = np.linspace(min_step_val[1], max_step_val[1], num_steps)
        step_vals = np.array(np.meshgrid(duration_vals, amp_vals)).T.reshape(-1, 2)
    else:
        step_vals = np.linspace(min_step_val, max_step_val, num_steps)
        if optimize_type == "duration":
            step_vals = step_vals.astype(int)

    pulse_gen = tb.get_server_pulse_gen()

    def run_fn(shuffled_step_inds):
        shuffled_step_vals = step_vals[shuffled_step_inds]
        pol_coords_list, pol_duration_list, pol_amp_list = (
            widefield.get_pulse_parameter_lists(nv_list, VirtualLaserKey.CHARGE_POL)
        )
        ion_coords_list = widefield.get_coords_list(nv_list, VirtualLaserKey.ION)

        if optimize_type == "both":
            durations = [int(step[0]) for step in shuffled_step_vals]
            amps = [float(step[1]) for step in shuffled_step_vals]
            seq_args = [
                pol_coords_list,
                pol_duration_list,
                pol_amp_list,
                ion_coords_list,
                durations,
                amps,
                optimize_pol_or_readout,
                optimize_type,
            ]
        else:
            seq_args = [
                pol_coords_list,
                pol_duration_list,
                pol_amp_list,
                ion_coords_list,
                shuffled_step_vals.tolist(),
                optimize_pol_or_readout,
                optimize_type,
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
        "optimize_type": optimize_type,
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
    # raw_data = dm.get_raw_data(file_id=1714802805037, load_npz=False)
    # raw_data = dm.get_raw_data(file_id=1709868774004, load_npz=False) #yellow ampl var
    raw_data = dm.get_raw_data(file_id=1710843759806, load_npz=False)  # yellow amp var
    # raw_data = dm.get_raw_data(file_id=1711618252292, load_npz=False) #green ampl var
    process_and_plot(raw_data)
    kpl.show(block=True)