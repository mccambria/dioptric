# -*- coding: utf-8 -*-
"""
Check for crosstalk in SCC from green and red lasers

Created on March 29th, 2024

@author: sbchand
"""

import matplotlib.pyplot as plt
import numpy as np

from majorroutines.widefield import base_routine
from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import positioning as pos
from utils import tool_belt as tb
from utils import widefield as widefield
from utils.constants import NVSig
from utils.positioning import get_scan_1d as calculate_freqs


def create_raw_data_figure(
    nv_sig, laser_name, axis_ind, relative_aod_freqs, avg_snr, avg_snr_ste
):
    fig, ax = plt.subplots()
    kpl.plot_points(ax, relative_aod_freqs, avg_snr, yerr=avg_snr_ste)

    ax.set_xlabel("AOD frequency deviation from target NV (MHz)")
    ax.set_ylabel("SNR")
    axis_ind_labels = ["x", "y", "z"]
    ax.set_title(f"{laser_name}, {axis_ind_labels[axis_ind]} axis, {nv_sig.name}")

    return fig


def main(
    nv_sig: NVSig,
    num_steps,
    num_reps,
    num_runs,
    aod_freq_range,
    laser_name,
    axis_ind,  # 0: x, 1: y, 2: z
    uwave_ind_list=[0, 1],
):
    ### Some initial setup

    pulse_gen = tb.get_server_pulse_gen()
    laser_coords = pos.get_nv_coords(nv_sig, coords_key=laser_name, drift_adjust=False)
    aod_freq_center = laser_coords[axis_ind]
    aod_freqs = calculate_freqs(aod_freq_center, aod_freq_range, num_steps)
    crosstalk_coords_list = [laser_coords.copy() for ind in range(num_steps)]
    for ind in range(num_steps):
        crosstalk_coords_list[ind][axis_ind] = aod_freqs[ind]

    seq_file = "crosstalk_check.py"

    ### Collect the data

    nv_list = [nv_sig]

    def run_fn(step_ind_list):
        # Base seq args
        seq_args = []
        seq_args.append(widefield.get_base_scc_seq_args(nv_list, uwave_ind_list))
        seq_args.append(laser_name)

        # Add on the coordinates for the crosstalk pulse
        crosstalk_coords_list_shuffle = []
        for ind in step_ind_list:
            coords = crosstalk_coords_list[ind]
            adj_coords = pos.adjust_coords_for_drift(coords, coords_key=laser_name)
            crosstalk_coords_list_shuffle.append(adj_coords)
        seq_args.append(crosstalk_coords_list_shuffle)

        # Pass it over to the OPX
        seq_args_string = tb.encode_seq_args(seq_args)
        pulse_gen.stream_load(seq_file, seq_args_string, num_reps)

    raw_data = base_routine.main(
        nv_list,
        num_steps,
        num_reps,
        num_runs,
        run_fn=run_fn,
        uwave_ind_list=uwave_ind_list,
    )

    ### Process and plot

    try:
        # experiment, nv, run, step, rep
        counts = raw_data["states"]
        sig_counts = counts[0]
        ref_counts = counts[1]

        # avg_sig_counts, avg_sig_counts_ste = widefield.average_counts(sig_counts)
        # avg_ref_counts, avg_ref_counts_ste = widefield.average_counts(ref_counts)
        avg_snr, avg_snr_ste = widefield.calc_snr(sig_counts, ref_counts)

        avg_snr = avg_snr[0]
        avg_snr_ste = avg_snr_ste[0]

        relative_aod_freqs = aod_freqs - aod_freq_center
        raw_fig = create_raw_data_figure(
            nv_sig, laser_name, axis_ind, relative_aod_freqs, avg_snr, avg_snr_ste
        )
    except Exception:
        raw_fig = None

    ### Clean up and return

    tb.reset_cfm()
    kpl.show()

    timestamp = dm.get_time_stamp()
    raw_data |= {
        "nv_sig": nv_sig,
        "timestamp": timestamp,
        "aod_freq_range": aod_freq_range,
        "laser_name": laser_name,
        "axis_ind": axis_ind,
    }

    repr_nv_name = nv_sig.name
    file_path = dm.get_file_path(__file__, timestamp, repr_nv_name)
    if "img_arrays" in raw_data:
        keys_to_compress = ["img_arrays"]
    else:
        keys_to_compress = None
    dm.save_raw_data(raw_data, file_path, keys_to_compress)
    if raw_fig is not None:
        dm.save_figure(raw_fig, file_path)


if __name__ == "__main__":
    kpl.init_kplotlib()

    data = dm.get_raw_data(file_id=1488892009219, load_npz=True)

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
