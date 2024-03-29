# -*- coding: utf-8 -*-
"""
Check for crosstalk in SCC from green and red lasers

Created on March 29th, 2024

@author: sbchand
"""

import os
import sys
import time
from random import shuffle

import matplotlib.pyplot as plt
import numpy as np

from majorroutines.pulsed_resonance import fit_resonance, voigt, voigt_split
from majorroutines.widefield import base_routine, optimize
from utils import common
from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import positioning as pos
from utils import tool_belt as tb
from utils import widefield as widefield
from utils.constants import NVSig, NVSpinState
from utils.positioning import get_scan_1d as calculate_freqs


def create_raw_data_figure(nv_sig, axis_ind, relative_aod_freqs, avg_snr, avg_snr_ste):
    fig, ax = plt.subplots()
    kpl.plot_points(relative_aod_freqs, avg_snr, yerr=avg_snr_ste)

    ax.set_xlabel("AOD frequency deviation from target NV (MHz)")
    ax.set_ylabel("SNR")
    axis_ind_labels = ["x", "y", "z"]
    ax.set_title(f"{axis_ind_labels[axis_ind]}, {nv_sig.name}")


def main(
    nv_sig: NVSig,
    num_steps,
    num_reps,
    num_runs,
    aod_freq_range,
    laser_name,
    axis_ind,  # 0: x, 1: y, 2: z
    uwave_ind=0,
):
    ### Some initial setup

    pulse_gen = tb.get_server_pulse_gen()
    laser_coords = pos.get_nv_coords(nv_sig, coords_key=laser_name, drift_adjust=False)
    aod_freq_center = laser_coords[axis_ind]
    crosstalk_coords = laser_coords.copy()
    aod_freqs = calculate_freqs(aod_freq_center, aod_freq_range, num_steps)
    relative_aod_freqs = aod_freqs - aod_freq_center

    seq_file = "crosstalk_check.py"

    ### Collect the data

    nv_list = [nv_sig]

    def step_fn(step_ind):
        # Base seq args
        seq_args = widefield.get_base_scc_seq_args(nv_list)
        seq_args.append(uwave_ind)
        seq_args.append(laser_name)

        # Add on the coordinates for the crosstalk pulse
        aod_freq = aod_freqs[step_ind]
        crosstalk_coords[axis_ind] = aod_freq
        crosstalk_coords_adj = pos.adjust_coords_for_drift(
            crosstalk_coords, coords_key=laser_name
        )
        seq_args.append(crosstalk_coords_adj)

        # Pass it over to the OPX
        seq_args_string = tb.encode_seq_args(seq_args)
        pulse_gen.stream_load(seq_file, seq_args_string, num_reps)

    counts, raw_data = base_routine.main(
        nv_list,
        num_steps,
        num_reps,
        num_runs,
        step_fn,
        uwave_ind=uwave_ind,
        stream_load_in_run_fn=False,
    )

    ### Process and plot

    sig_counts = counts[0]
    ref_counts = counts[1]

    # avg_sig_counts, avg_sig_counts_ste = widefield.average_counts(sig_counts)
    # avg_ref_counts, avg_ref_counts_ste = widefield.average_counts(ref_counts)
    avg_snr, avg_snr_ste = widefield.calc_snr(sig_counts, ref_counts)

    raw_fig = create_raw_data_figure(
        nv_sig, axis_ind, relative_aod_freqs, avg_snr, avg_snr_ste
    )

    ### Clean up and return

    tb.reset_cfm()
    kpl.show()

    timestamp = dm.get_time_stamp()
    raw_data |= {
        "timestamp": timestamp,
        "aod_freq_range": aod_freq_range,
        "laser_name": laser_name,
        "axis_ind": axis_ind,
    }

    repr_nv_sig = widefield.get_repr_nv_sig(nv_list)
    repr_nv_name = repr_nv_sig.name
    file_path = dm.get_file_path(__file__, timestamp, repr_nv_name)
    if "img_arrays" in raw_data:
        keys_to_compress = ["img_arrays"]
    else:
        keys_to_compress = None
    dm.save_raw_data(raw_data, file_path, keys_to_compress)
    dm.save_figure(raw_fig, file_path)


if __name__ == "__main__":
    kpl.init_kplotlib()

    data = dm.get_raw_data(file_id=1470392816628, no_npz=True)

    nv_list = data["nv_list"]
    num_nvs = len(nv_list)
    num_steps = data["num_steps"]
    num_runs = data["num_runs"]
    num_reps = data["num_reps"]
    freqs = data["freqs"]
    counts = np.array(data["counts"])
    sig_counts = counts[0]
    ref_counts = counts[1]

    kpl.show(block=True)
