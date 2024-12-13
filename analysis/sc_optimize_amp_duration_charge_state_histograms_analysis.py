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


import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from utils import data_manager as dm, widefield
from analysis.bimodal_histogram import fit_bimodal_histogram, determine_threshold
from utils.constants import ProbDist


def process_and_save(raw_data, file_path):

    # Extract data
    nv_list = raw_data["nv_list"]
    num_nvs = len(nv_list)
    num_steps = raw_data["num_steps"]
    counts = np.array(raw_data["counts"])
    step_vals = np.array(raw_data["step_vals"])
    optimize_pol_or_readout = raw_data["optimize_pol_or_readout"]

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

    def process_nv_step(nv_ind, step_ind):
        """
        Process a single NV and step index to extract metrics.
        """
        counts_data = condensed_counts[nv_ind, step_ind]
        if counts_data.size == 0:
            return np.nan, np.nan, np.nan
        try:
            popt, _, chi_squared = fit_bimodal_histogram(counts_data, prob_dist)
            if popt is None:
                return np.nan, np.nan, np.nan
            threshold, readout_fidelity = determine_threshold(
                popt, prob_dist, dark_mode_weight=0.5, ret_fidelity=True
            )
            prep_fidelity = 1 - popt[0]
        except Exception:
            return np.nan, np.nan, np.nan
        return readout_fidelity, prep_fidelity, chi_squared

    # Parallel processing
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

    # Extract amplitude and duration values
    amp_vals = np.unique(step_vals[:, 0])
    duration_vals = np.unique(step_vals[:, 1])

    # Save processed data
    processed_data = {
        "nv_list": nv_list,
        "amp_vals": amp_vals,
        "duration_vals": duration_vals,
        "readout_fidelity_arr": readout_fidelity_arr,
        "prep_fidelity_arr": prep_fidelity_arr,
        "goodness_of_fit_arr": goodness_of_fit_arr,
        "step_vals": step_vals,
        "optimize_pol_or_readout": optimize_pol_or_readout,
    }
    dm.save_raw_data(processed_data, file_path)
    print(f"Processed data saved to: {file_path}")

    return processed_data


def analyze_and_visualize(processed_data):
    """
    Analyze and visualize processed data.

    Parameters
    ----------
    processed_data : dict
        Dictionary containing processed data.
    """
    amp_vals = processed_data["amp_vals"]
    duration_vals = processed_data["duration_vals"]
    readout_fidelity_arr = processed_data["readout_fidelity_arr"]
    prep_fidelity_arr = processed_data["prep_fidelity_arr"]
    goodness_of_fit_arr = processed_data["goodness_of_fit_arr"]

    # Calculate average metrics
    avg_readout_fidelity = np.nanmean(readout_fidelity_arr, axis=0).reshape(
        len(duration_vals), len(amp_vals)
    )
    avg_prep_fidelity = np.nanmean(prep_fidelity_arr, axis=0).reshape(
        len(duration_vals), len(amp_vals)
    )
    avg_goodness_of_fit = np.nanmean(goodness_of_fit_arr, axis=0).reshape(
        len(duration_vals), len(amp_vals)
    )

    # Generate heatmaps for metrics
    metrics = {
        "Readout Fidelity": avg_readout_fidelity,
        "Preparation Fidelity": avg_prep_fidelity,
        "Goodness of Fit": avg_goodness_of_fit,
    }

    for title, data in metrics.items():
        plt.figure(figsize=(8, 6))
        plt.imshow(
            data.T,
            extent=[duration_vals[0], duration_vals[-1], amp_vals[0], amp_vals[-1]],
            aspect="auto",
            origin="lower",
            cmap="viridis",
        )
        plt.colorbar(label=title)
        plt.xlabel("Duration (ns)")
        plt.ylabel("Amplitude (relative)")
        plt.title(f"{title} Heatmap")
        plt.grid(alpha=0.3)
        plt.show()

    # Combined heatmap using geometric mean of metrics
    combined_score = (
        avg_readout_fidelity * avg_prep_fidelity * (1 / (1 + avg_goodness_of_fit))
    ) ** (1 / 3)
    plt.figure(figsize=(8, 6))
    plt.imshow(
        combined_score.T,
        extent=[duration_vals[0], duration_vals[-1], amp_vals[0], amp_vals[-1]],
        aspect="auto",
        origin="lower",
        cmap="plasma",
    )
    plt.colorbar(label="Combined Score")
    plt.xlabel("Duration (ns)")
    plt.ylabel("Amplitude (relative)")
    plt.title("Combined Metrics Heatmap")
    plt.grid(alpha=0.3)
    plt.show()


if __name__ == "__main__":
    # Initialize
    file_id = 1721471390254  # Example file ID
    raw_data = dm.get_raw_data(file_id=file_id, load_npz=False)

    # Save processed data
    timestamp = dm.get_time_stamp()
    file_path = dm.get_file_path(__file__, timestamp, f"processed_data_{file_id}")
    processed_data = process_and_save(raw_data, file_path)

    # Analyze and visualize
    analyze_and_visualize(processed_data)
