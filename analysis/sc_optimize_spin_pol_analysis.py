# -*- coding: utf-8 -*-
"""
Illuminate an area, collecting onto the camera. Interleave a signal and control sequence
and plot the difference
Created on Fall 2024
@author: saroj chand
"""

import os
import sys
import time
import traceback
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from matplotlib import font_manager as fm
from matplotlib import rcParams

from analysis.bimodal_histogram import (
    ProbDist,
    determine_threshold,
    fit_bimodal_histogram,
)
from utils import common, widefield
from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import positioning as pos
from utils import tool_belt as tb
from utils.constants import NVSig, VirtualLaserKey
from utils.tool_belt import curve_fit


def fit_fn(tau, delay, slope, decay):
    """
    Fit function modeling the preparation fidelity as a function of polarization duration.
    """
    tau = np.array(tau) - delay
    return slope * tau * np.exp(-tau / decay)


def process_and_plot(raw_data):
    # print(raw_data.keys())
    nv_list = raw_data["nv_list"]
    num_nvs = len(nv_list)
    step_vals = np.array(raw_data["taus"])
    optimize_pol_or_readout = False
    optimize_duration_or_amp = False
    counts = np.array(raw_data["counts"])
    sig_counts = counts[0]
    ref_counts = counts[1]
    # a, b, c = 3.7e5, 6.97, 8e-14 # old para
    a, b, c = 161266.751, 6.617, -19.492  # new para

    # get yellow amplitude
    yellow_spin_amp = raw_data["opx_config"]["waveforms"]["yellow_spin_pol"]["sample"]
    # print(yellow_spin_amp``)
    green_aod_cw_charge_pol_amp = raw_data["opx_config"]["waveforms"][
        "green_aod_cw-charge_pol"
    ]["sample"]

    ### Plotting
    if optimize_pol_or_readout:
        if optimize_duration_or_amp:
            # step_vals *= 1e-3
            step_vals = step_vals
            x_label = "Polarization duration (ns)"
        else:
            step_vals *= green_aod_cw_charge_pol_amp
            x_label = "Polarization amplitude"
    else:
        if optimize_duration_or_amp:
            step_vals *= 1e-6
            x_label = "Readout duration (ms)"
        else:
            step_vals *= yellow_spin_amp
            # x_label = "Readout amplitude"
            step_vals = a * (step_vals**b) + c
            x_label = "Readout amplitude (uW)"
        # Apply thresholds

    # Calculate metrics
    avg_snr, avg_snr_ste = widefield.calc_snr(sig_counts, ref_counts)
    # Extract single step values
    # step_ind = 0
    # avg_snr = avg_snr[:, step_ind]
    # avg_snr_ste = avg_snr_ste[:, step_ind]

    # Get NV coordinates and Compute distances

    # Prepare DataFrame for analysis
    # Prepare DataFrame for analysis

    # indices_to_remove = [ind for ind in range(len(snr)) if snr[ind] < 0.05]
    # indices_to_remove = []
    # print(indices_to_remove)
    # selected_indices = [ind for ind in range(num_nvs) if ind not in indices_to_remove]
    median_snr = np.median(avg_snr, axis=0)
    median_snr_ste = np.median(avg_snr_ste, axis=0)
    plt.figure(figsize=(6, 5))
    plt.errorbar(
        step_vals,
        median_snr,
        median_snr_ste,
        fmt="o",
        ecolor="gray",
        capsize=3,
    )
    plt.title("SNR vs. Distance", fontsize=15)
    plt.xlabel("Power (uW)", fontsize=15)
    plt.ylabel("SNR", fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.grid(True)
    plt.legend(fontsize=11)
    plt.show()


# endregion

if __name__ == "__main__":
    kpl.init_kplotlib()
    file_id = 1832391968450  # spin durations 75NVs
    raw_data = dm.get_raw_data(file_id=file_id, load_npz=False)
    file_name = dm.get_file_name(file_id=file_id)
    print(f"{file_name}_{file_id}")
    process_and_plot(raw_data)
    plt.show(block=True)
