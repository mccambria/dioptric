# -*- coding: utf-8 -*-
"""
Widefield Rabi experiment

Created on November 29th, 2023

@author: mccambria
"""


import sys
import time
import matplotlib.pyplot as plt
import numpy as np
from utils import tool_belt as tb
from utils import data_manager as dm
from utils import widefield as widefield
from utils import kplotlib as kpl
from utils import data_manager as dm
from scipy.optimize import curve_fit
from majorroutines.widefield import base_routine
from utils.constants import NVSpinState


def create_raw_data_figure(
    nv_list, taus, counts, counts_ste, init_state, readout_state
):
    fig, ax = plt.subplots()
    taus_ms = np.array(taus) / 1e6
    widefield.plot_raw_data(ax, nv_list, taus_ms, counts, counts_ste)
    ax.set_xlabel("Relaxation time (ms)")
    ax.set_ylabel("Counts")
    state_str_dict = {
        NVSpinState.ZERO: "0",
        NVSpinState.LOW: "-1",
        NVSpinState.HIGH: "+1",
    }
    init_state_str = state_str_dict[init_state]
    readout_state_str = state_str_dict[readout_state]
    ax.set_title(f"Init state: {init_state_str}; readout state: {readout_state_str}")
    return fig


def sq_relaxation(nv_list, num_steps, num_reps, num_runs, min_tau, max_tau):
    init_state_1 = NVSpinState.ZERO
    readout_state_1 = NVSpinState.ZERO
    init_state_2 = NVSpinState.ZERO
    readout_state_2 = NVSpinState.HIGH
    base_args = [nv_list, num_steps, num_reps, num_runs, min_tau, max_tau]
    return main(
        *base_args, init_state_1, readout_state_1, init_state_2, readout_state_2
    )


def dq_relaxation(nv_list, num_steps, num_reps, num_runs, min_tau, max_tau):
    init_state_1 = NVSpinState.HIGH
    readout_state_1 = NVSpinState.HIGH
    init_state_2 = NVSpinState.HIGH
    readout_state_2 = NVSpinState.LOW
    base_args = [nv_list, num_steps, num_reps, num_runs, min_tau, max_tau]
    return main(
        *base_args, init_state_1, readout_state_1, init_state_2, readout_state_2
    )


def main(
    nv_list,
    num_steps,
    num_reps,
    num_runs,
    min_tau,
    max_tau,
    init_state_0=NVSpinState.ZERO,
    readout_state_0=NVSpinState.ZERO,
    init_state_1=NVSpinState.ZERO,
    readout_state_1=NVSpinState.HIGH,
):
    ### Some initial setup

    pulse_gen = tb.get_server_pulse_gen()
    seq_file = "relaxation_interleave.py"
    taus = np.linspace(min_tau, max_tau, num_steps)

    ### Collect the data

    def step_fn(tau_ind):
        tau = taus[tau_ind]
        seq_args = widefield.get_base_scc_seq_args(nv_list)
        seq_args.append(
            tau, init_state_0, readout_state_0, init_state_1, readout_state_1
        )
        seq_args_string = tb.encode_seq_args(seq_args)
        pulse_gen.stream_load(seq_file, seq_args_string, num_reps)

    counts, raw_data = base_routine.main(
        nv_list,
        num_steps,
        num_reps,
        num_runs,
        step_fn,
        uwave_ind=[0, 1],
        num_images_per_rep=2,
    )
    counts_0 = counts[0]
    counts_1 = counts[1]

    ### Process and plot

    avg_counts_0, avg_counts_0_ste = widefield.process_counts(counts_0)
    avg_counts_1, avg_counts_1_ste = widefield.process_counts(counts_1)

    raw_fig_0 = create_raw_data_figure(
        nv_list, taus, avg_counts_0, avg_counts_0_ste, init_state_0, readout_state_0
    )
    raw_fig_1 = create_raw_data_figure(
        nv_list, taus, avg_counts_1, avg_counts_1_ste, init_state_1, readout_state_1
    )

    ### Clean up and return

    tb.reset_cfm()
    kpl.show()

    timestamp = dm.get_time_stamp()
    raw_data |= {
        "timestamp": timestamp,
        "taus": taus,
        "tau-units": "ns",
        "min_tau": max_tau,
        "max_tau": max_tau,
        "init_state_0": init_state_0,
        "readout_state_0": readout_state_0,
        "init_state_1": init_state_1,
        "readout_state_1": readout_state_1,
    }

    repr_nv_sig = widefield.get_repr_nv_sig(nv_list)
    repr_nv_name = repr_nv_sig["name"]
    file_path = dm.get_file_path(__file__, timestamp, repr_nv_name)
    if "img_arrays" in raw_data:
        keys_to_compress = ["img_arrays"]
    else:
        keys_to_compress = None
    dm.save_raw_data(raw_data, file_path, keys_to_compress)
    dm.save_figure(raw_fig_0, file_path + "-0")
    dm.save_figure(raw_fig_1, file_path + "-1")


if __name__ == "__main__":
    kpl.init_kplotlib()

    # file_name = ""
    # data = dm.get_raw_data(file_name)
    data = dm.get_raw_data(file_id=1382892086081)

    nv_list = data["nv_list"]
    num_nvs = len(nv_list)
    img_arrays = data["img_arrays"]
    num_steps = data["num_steps"]
    num_runs = data["num_runs"]
    avg_img_arrays = np.average(img_arrays, axis=1)
    taus = data["taus"]
    counts = np.array(data["counts"])

    avg_counts, avg_counts_ste = widefield.process_counts(counts)
    raw_fig = create_raw_data_figure(nv_list, taus, avg_counts, avg_counts_ste)
    fit_fig = create_fit_figure(nv_list, taus, avg_counts, avg_counts_ste)

    plt.show(block=True)
