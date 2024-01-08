# -*- coding: utf-8 -*-
"""
Widefield Rabi experiment

Created on November 29th, 2023

@author: mccambria
"""


import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from majorroutines.widefield import base_routine
from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import tool_belt as tb
from utils import widefield as widefield


def create_raw_data_figure(nv_list, taus, counts, counts_ste):
    fig, ax = plt.subplots()
    total_evolution_times = 2 * np.array(taus) / 1e3
    widefield.plot_raw_data(ax, nv_list, total_evolution_times, counts, counts_ste)
    ax.set_xlabel("Total evolution time (us)")
    ax.set_ylabel("Counts")
    return fig


def create_fit_figure(nv_list, taus, counts, counts_ste):
    pass


def main(nv_list, num_steps, num_reps, num_runs, min_tau, max_tau):
    ### Some initial setup

    pulse_gen = tb.get_server_pulse_gen()
    seq_file = "correlation_test.py"
    taus = np.linspace(min_tau, max_tau, num_steps)

    ### Collect the data

    def step_fn(tau_ind):
        tau = taus[tau_ind]
        seq_args = widefield.get_base_scc_seq_args(nv_list)
        seq_args.append(tau)
        seq_args_string = tb.encode_seq_args(seq_args)
        pulse_gen.stream_load(seq_file, seq_args_string, num_reps)

    counts, raw_data = base_routine.main(
        nv_list, num_steps, num_reps, num_runs, step_fn
    )

    ### Process and plot

    avg_counts, avg_counts_ste = widefield.process_counts(counts)

    raw_fig = create_raw_data_figure(nv_list, taus, avg_counts, avg_counts_ste)
    try:
        fit_fig = create_fit_figure(nv_list, taus, avg_counts, avg_counts_ste)
    except Exception as exc:
        print(exc)
        fit_fig = None

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
    }

    repr_nv_sig = widefield.get_repr_nv_sig(nv_list)
    repr_nv_name = repr_nv_sig["name"]
    file_path = dm.get_file_path(__file__, timestamp, repr_nv_name)
    if "img_arrays" in raw_data:
        keys_to_compress = ["img_arrays"]
    else:
        keys_to_compress = None
    dm.save_raw_data(raw_data, file_path, keys_to_compress)
    dm.save_figure(raw_fig, file_path)
    if fit_fig is not None:
        file_path = dm.get_file_path(__file__, timestamp, repr_nv_name + "-fit")
        dm.save_figure(fit_fig, file_path)


if __name__ == "__main__":
    kpl.init_kplotlib()

    # file_name = ""
    # data = dm.get_raw_data(file_name)
    data = dm.get_raw_data(file_id=1396164244162, no_npz=True)

    nv_list = data["nv_list"]
    num_nvs = len(nv_list)
    num_steps = data["num_steps"]
    num_runs = data["num_runs"]
    taus = data["taus"]
    counts = data["counts"]

    avg_counts, avg_counts_ste = widefield.process_counts(counts)
    raw_fig = create_raw_data_figure(nv_list, taus, avg_counts, avg_counts_ste)
    fit_fig = create_fit_figure(nv_list, taus, avg_counts, avg_counts_ste)

    plt.show(block=True)
