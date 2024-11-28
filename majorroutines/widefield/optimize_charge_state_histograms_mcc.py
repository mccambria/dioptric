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
from joblib import Parallel, delayed
from scipy import ndimage
from scipy.interpolate import interp1d

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
def find_intersection(x, y1, y2):
    """
    Finds the intersection point(s) of two curves y1 and y2 over x.
    Parameters
    ----------
    x : np.ndarray
        Array of x-values.
    y1 : np.ndarray
        First curve (e.g., fidelity).
    y2 : np.ndarray
        Second curve (e.g., goodness of fit).
    Returns
    -------
    float
        x-value of the intersection.
    """
    interp_fidelity = interp1d(
        x, y1, kind="linear", bounds_error=False, fill_value="extrapolate"
    )
    interp_r_squared = interp1d(
        x, y2, kind="linear", bounds_error=False, fill_value="extrapolate"
    )
    # Calculate the difference between the two curves
    diff = np.abs(interp_fidelity(x) - interp_r_squared(x))
    min_index = np.argmin(diff)
    return x[min_index]


def find_optimal_combined_value(
    step_vals, readout_fidelity, goodness_of_fit, weight=0.5
):
    """
    Finds the optimal step value that maximizes a weighted combination
    of readout fidelity and goodness of fit (R²).

    Parameters
    ----------
    step_vals : np.ndarray
        Array of step values.
    readout_fidelity : np.ndarray
        Array of readout fidelity values corresponding to step values.
    goodness_of_fit : np.ndarray
        Array of goodness of fit (R²) values corresponding to step values.
    weight : float
        Weight given to readout fidelity in the combined score (default is 0.5).

    Returns
    -------
    float
        The optimal step value that maximizes the combined score.
    float
        The maximum combined score.
    """
    # Normalize both metrics to ensure equal weighting
    norm_fidelity = (readout_fidelity - np.nanmin(readout_fidelity)) / (
        np.nanmax(readout_fidelity) - np.nanmin(readout_fidelity)
    )
    norm_goodness = (goodness_of_fit - np.nanmin(goodness_of_fit)) / (
        np.nanmax(goodness_of_fit) - np.nanmin(goodness_of_fit)
    )

    # Combined score: weighted sum of normalized metrics
    combined_score = weight * norm_fidelity + (1 - weight) * norm_goodness

    # Find the step value corresponding to the maximum combined score
    max_index = np.nanargmax(combined_score)
    optimal_step_val = step_vals[max_index]
    max_combined_score = combined_score[max_index]

    return optimal_step_val, max_combined_score

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
        popt, r_squared = fit_bimodal_histogram(counts_data, prob_dist)

        if popt is None:
            return np.nan, np.nan, np.nan, np.nan

        # Threshold, prep and readout fidelity
        threshold, readout_fidelity = determine_threshold(
            popt, prob_dist, dark_mode_weight=0.5, ret_fidelity=True
        )
        prep_fidelity = 1 - popt[0]  # Population weight of dark state

        # Calculate inter-class and intra-class variance for separation metric
        w_dark, mu_dark, mu_bright = popt[0], popt[1], popt[2]
        sigma_dark = np.sqrt(mu_dark)
        sigma_bright = np.sqrt(mu_bright)
        inter_class_variance = w_dark * (1 - w_dark) * (mu_bright - mu_dark) ** 2
        intra_class_variance = w_dark * sigma_dark**2 + (1 - w_dark) * sigma_bright**2
        separation_metric = (
            inter_class_variance / intra_class_variance
            if intra_class_variance > 0
            else np.nan
        )

        return readout_fidelity, prep_fidelity, separation_metric, r_squared

    # Parallel processingv -->  n_jobs :  Defaults to using all available cores (-1).
    results = Parallel(n_jobs=-1)(
        delayed(process_nv_step)(nv_ind, step_ind)
        for nv_ind in range(num_nvs)
        for step_ind in range(num_steps)
    )

    # Reshape results into arrays
    results = np.array(results).reshape(num_nvs, num_steps, 4)
    readout_fidelity_arr = results[:, :, 0]
    prep_fidelity_arr = results[:, :, 1]
    separation_metric_arr = results[:, :, 2]
    goodness_of_fit_arr = results[:, :, 3]

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

    # Optimal values
    optimal_values = []  # To store results

    for nv_ind in range(num_nvs):
        try:
            # Calculate the optimal step value
            optimal_step_val, max_combined_score = find_optimal_combined_value(
                step_vals,
                # readout_fidelity_arr[nv_ind],
                prep_fidelity_arr[nv_ind],
                goodness_of_fit_arr[nv_ind],
                weight=0.5,  # Adjust this to change weighting
            )
            optimal_values.append((nv_ind, optimal_step_val, max_combined_score))
        except Exception as e:
            print(f"Failed to process NV{nv_ind}: {e}")
            optimal_values.append((nv_ind, np.nan, np.nan))
            continue

        # Plotting
        fig, ax1 = plt.subplots(figsize=(6, 5))

        # Plot readout fidelity
        ax1.plot(
            step_vals,
            # readout_fidelity_arr[nv_ind],
            prep_fidelity_arr[nv_ind],
            # label="Readout Fidelity",
            label="Prep Fidelity",
            color="blue",
        )
        ax1.set_xlabel(x_label)
        ax1.set_ylabel("Readout Fidelity")
        ax1.set_ylabel("Prep Fidelity")
        ax1.tick_params(axis="y", labelcolor="blue")
        ax1.legend(loc="upper left", fontsize=9)

        # Plot goodness of fit (R²)
        ax2 = ax1.twinx()
        ax2.plot(
            step_vals,
            goodness_of_fit_arr[nv_ind],
            color="green",
            label="Goodness of Fit (R²)",
            alpha=0.7,
        )
        ax2.set_ylabel("Goodness of Fit (R²)", color="green")
        ax2.tick_params(axis="y", labelcolor="green")

        # Highlight optimal step value
        ax1.axvline(
            optimal_step_val,
            color="red",
            linestyle="--",
            label=f"Optimal Step Val: {optimal_step_val:.2f}",
        )
        ax2.axvline(
            optimal_step_val,
            color="red",
            linestyle="--",
            label=f"Optimal Step Val: {optimal_step_val:.2f}",
        )

        # Add legends for both y-axes
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax2.legend(
            lines_1 + lines_2, labels_1 + labels_2, loc="upper right", fontsize=9
        )

        # Title and layout
        ax1.set_title(f"NV{nv_ind} - Optimal Step Val: {optimal_step_val:.2f}")
        fig.tight_layout()
        plt.show()

    # Save results to a file
    with open("optimal_combined_values.txt", "w") as f:
        f.write("NV Index, Optimal Step Value, Max Combined Score\n")
        for nv_index, opt_step, max_score in optimal_values:
            f.write(f"{nv_index}, {opt_step:.6f}, {max_score:.6f}\n")
    print("Optimal combined values saved to 'optimal_combined_values.txt'.")


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
        shuffled_step_vals = step_vals[shuffled_step_inds].tolist()
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
        # print(seq_args)
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
    # raw_data = dm.get_raw_data(file_id=1705172140093, load_npz=False)
    # raw_data = dm.get_raw_data(file_id=1709868774004, load_npz=False)
    # raw_data = dm.get_raw_data(file_id=1710843759806, load_npz=False)
    raw_data = dm.get_raw_data(file_id=1711618252292, load_npz=False)
    process_and_plot(raw_data)
    kpl.show(block=True)
