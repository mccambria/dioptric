# -*- coding: utf-8 -*-
"""
Created on November 29th, 2023

@author: mccambria

Updated on April 17th, 2023

@author: sbchand
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
    widefield.plot_raw_data(ax, nv_list, taus, counts, counts_ste)
    ax.set_xlabel("Delay (ns)")
    ax.set_ylabel("Counts")
    return fig


def main(nv_list, num_steps, num_reps, num_runs, taus, i_or_q=True):
    ### Some initial setup

    pulse_gen = tb.get_server_pulse_gen()
    seq_file = "calibrate_iq_delay.py"
    # taus = np.linspace(min_tau, max_tau, num_steps)
    uwave_ind_list = [1]

    def run_fn(shuffled_step_inds):
        shuffled_taus = [taus[ind] for ind in shuffled_step_inds]
        seq_args = [
            widefield.get_base_scc_seq_args(nv_list, uwave_ind_list),
            shuffled_taus,
        ]
        # print(seq_args)
        seq_args_string = tb.encode_seq_args(seq_args)
        pulse_gen.stream_load(seq_file, seq_args_string, num_reps)

    ### Collect the data
    raw_data = base_routine.main(
        nv_list,
        num_steps,
        num_reps,
        num_runs,
        run_fn=run_fn,
        uwave_ind_list=uwave_ind_list,
        load_iq=True,
    )

    timestamp = dm.get_time_stamp()
    raw_data |= {
        "timestamp": timestamp,
        "taus": taus,
        "tau-units": "ns",
    }
    ### save the raw data
    repr_nv_sig = widefield.get_repr_nv_sig(nv_list)
    repr_nv_name = repr_nv_sig["name"]
    file_path = dm.get_file_path(__file__, timestamp, repr_nv_name)
    dm.save_raw_data(raw_data, file_path)
    ### Clean up and return
    tb.reset_cfm()

    ### Process and plot
    counts = raw_data["counts"]
    sig_counts, ref_counts = counts[0], counts[1]
    norm_counts, norm_counts_ste = widefield.process_counts(
        nv_list, sig_counts, ref_counts, threshold=True
    )
    raw_fig = create_raw_data_figure(nv_list, taus, norm_counts, norm_counts_ste)
    dm.save_figure(raw_fig, file_path)
    kpl.show()


if __name__ == "__main__":
    kpl.init_kplotlib()

    # file_name = ""
    # data = dm.get_raw_data(file_name)
    data = dm.get_raw_data(file_id=1386933040254)

    nv_list = data["nv_list"]
    num_nvs = len(nv_list)
    num_steps = data["num_steps"]
    num_runs = data["num_runs"]
    taus = data["taus"]
    counts = np.array(data["counts"])

    avg_counts, avg_counts_ste = widefield.process_counts(counts)
    raw_fig = create_raw_data_figure(nv_list, taus, avg_counts, avg_counts_ste)

    plt.show(block=True)
