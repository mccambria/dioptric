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

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from scipy import ndimage
from scipy.optimize import curve_fit
from scipy.special import factorial

from majorroutines.widefield import base_routine, optimize
from utils import common, widefield
from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import positioning as pos
from utils import tool_belt as tb
from utils.constants import LaserKey, NVSig


def detect_cosmic_rays(nv_list, num_reps, num_runs, dark_time):
    main(
        nv_list,
        num_reps,
        num_runs,
        "detect_cosmic_rays",
        base_routine.charge_prep_loop,
        process_detect_cosmic_rays,
        dark_time=dark_time,
    )


def process_detect_cosmic_rays(data):
    nv_list = data["nv_list"]
    num_nvs = len(nv_list)
    counts = np.array(data["counts"])
    sig_counts = counts[0]
    states, _ = widefield.threshold_counts(nv_list, sig_counts)
    img_array = np.array([states[nv_ind].flatten() for nv_ind in range(num_nvs)])
    fig, ax = plt.subplots()
    kpl.imshow(ax, img_array, aspect="auto")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    return fig


def check_readout_fidelity(nv_list, num_reps, num_runs):
    main(
        nv_list,
        num_reps,
        num_runs,
        "check_readout_fidelity",
        # base_routine.charge_prep_loop_first_rep,
        # base_routine.charge_prep_no_verification,
        base_routine.charge_prep_no_prep,
        process_check_readout_fidelity,
    )


def charge_quantum_jump(nv_list, num_reps):
    num_runs = 1
    main(
        nv_list,
        num_reps,
        num_runs,
        "charge_quantum_jump",
        base_routine.charge_prep_loop_first_rep,
        process_detect_cosmic_rays,
    )


def process_check_readout_fidelity(data):
    nv_list = data["nv_list"]
    num_nvs = len(nv_list)
    counts = np.array(data["counts"])
    num_runs = counts.shape[2]
    num_reps = counts.shape[4]
    sig_counts = counts[0]
    states, _ = widefield.threshold_counts(nv_list, sig_counts)
    states = np.array(data["charge_states"])[0]

    figsize = kpl.figsize
    figsize[1] *= 1.5
    fig, axes_pack = plt.subplots(2, 1, sharex=True, figsize=figsize)
    labels = {0: "NV0", 1: "NV-"}
    for init_state in [0, 1]:
        ax = axes_pack[init_state]
        for nv_ind in range(num_nvs):
            shots_list = []
            for run_ind in range(num_runs):
                for rep_ind in range(num_reps):
                    # prev_state = (
                    #     1 if rep_ind == 0 else states[nv_ind, run_ind, 0, rep_ind - 1]
                    # )
                    if rep_ind == 0:
                        continue
                    prev_state = states[nv_ind, run_ind, 0, rep_ind - 1]
                    current_state = states[nv_ind, run_ind, 0, rep_ind]
                    if prev_state == init_state:
                        shots_list.append(current_state == prev_state)
            prob = np.mean(shots_list)
            err = np.std(shots_list, ddof=1) / np.sqrt(len(shots_list))
            nv_num = widefield.get_nv_num(nv_list[nv_ind])
            kpl.plot_points(ax, nv_num, prob, yerr=err)
        label = labels[init_state]
        ax.set_ylabel(f"P({label}|previous shot {label})")

    ax.set_xlabel("NV index")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    return fig


def main(
    nv_list,
    num_reps,
    num_runs,
    caller_fn_name,
    charge_prep_fn,
    data_processing_fn,
    dark_time=0,
):
    ### Some initial setup
    seq_file = "charge_monitor.py"

    tb.reset_cfm()
    pulse_gen = tb.get_server_pulse_gen()

    num_steps = 1
    num_exps_per_rep = 1

    ### Collect the data

    def run_fn(shuffled_step_inds):
        pol_coords_list = widefield.get_coords_list(nv_list, LaserKey.CHARGE_POL)
        seq_args = [pol_coords_list, dark_time]
        seq_args_string = tb.encode_seq_args(seq_args)
        pulse_gen.stream_load(seq_file, seq_args_string, num_reps)

    counts, raw_data = base_routine.main(
        nv_list,
        num_steps,
        num_reps,
        num_runs,
        run_fn=run_fn,
        num_exps_per_rep=num_exps_per_rep,
        charge_prep_fn=charge_prep_fn,
    )

    repr_nv_sig = widefield.get_repr_nv_sig(nv_list)
    repr_nv_name = repr_nv_sig.name
    timestamp = dm.get_time_stamp()
    raw_data |= {
        "timestamp": timestamp,
        "caller_fn_name": caller_fn_name,
    }

    try:
        fig = data_processing_fn(raw_data)
    except Exception:
        fig = None

    ### Save and clean up

    file_path = dm.get_file_path(__file__, timestamp, repr_nv_name)
    dm.save_raw_data(raw_data, file_path)
    if fig is not None:
        dm.save_figure(fig, file_path)
    tb.reset_cfm()


if __name__ == "__main__":
    kpl.init_kplotlib()

    data = dm.get_raw_data(file_id=1508535628026)

    process_check_readout_fidelity(data)
    # process_detect_cosmic_rays(data)

    kpl.show(block=True)
