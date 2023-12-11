# -*- coding: utf-8 -*-
"""
Optimize SCC parameters

Created on December 6th, 2023

@author: mccambria
"""


import copy
from random import shuffle
import sys
import matplotlib.pyplot as plt
import numpy as np
from majorroutines.widefield import optimize
from utils import tool_belt as tb
from utils import data_manager as dm
from utils import common
from utils import widefield as widefield
from utils import kplotlib as kpl
from utils import positioning as pos
from utils import data_manager as dm
from utils.constants import LaserKey, NVSpinState
import os
import time
from utils.positioning import get_scan_1d as calculate_freqs
from majorroutines.pulsed_resonance import fit_resonance, voigt_split, voigt
from majorroutines.widefield import base_routine


def process_and_plot(nv_list, taus, sig_counts, ref_counts):
    avg_sig_counts, avg_sig_counts_ste = widefield.process_counts(sig_counts)
    avg_ref_counts, avg_ref_counts_ste = widefield.process_counts(ref_counts)
    avg_snr, avg_snr_ste = widefield.calc_snr(sig_counts, ref_counts)
    # avg_snr_ste = None

    kpl.init_kplotlib()

    sig_fig, sig_ax = plt.subplots()
    widefield.plot_raw_data(sig_ax, nv_list, taus, avg_sig_counts, avg_sig_counts_ste)
    sig_ax.set_xlabel("Ionization pulse duration (ns)")
    sig_ax.set_ylabel("Signal counts")

    ref_fig, ref_ax = plt.subplots()
    widefield.plot_raw_data(ref_ax, nv_list, taus, avg_ref_counts, avg_ref_counts_ste)
    ref_ax.set_xlabel("Ionization pulse duration (ns)")
    ref_ax.set_ylabel("Reference counts")

    snr_fig, snr_ax = plt.subplots()
    widefield.plot_raw_data(snr_ax, nv_list, taus, avg_snr, avg_snr_ste)
    snr_ax.set_xlabel("Ionization pulse duration (ns)")
    snr_ax.set_ylabel("SNR")

    return sig_fig, ref_fig, snr_fig


def main(
    nv_list, uwave_list, uwave_ind, num_steps, num_reps, num_runs, min_tau, max_tau
):
    ### Some initial setup

    seq_file = "resonance_ref.py"
    taus = np.linspace(min_tau, max_tau, num_steps)
    pulse_gen = tb.get_server_pulse_gen()

    uwave_dict = uwave_list[uwave_ind]
    uwave_duration = tb.get_pi_pulse_dur(uwave_dict["rabi_period"])

    ### Collect the data

    def step_fn(tau_ind):
        tau = taus[tau_ind]
        seq_args = widefield.get_base_scc_seq_args(nv_list, pol_duration=tau)
        seq_args.extend([uwave_ind, uwave_duration])
        seq_args_string = tb.encode_seq_args(seq_args)
        pulse_gen.stream_load(seq_file, seq_args_string, num_reps)

    sig_counts, ref_counts, raw_data = base_routine.main(
        nv_list,
        uwave_list,
        uwave_ind,
        num_steps,
        num_reps,
        num_runs,
        step_fn,
        reference=True,
    )

    ### Process and plot

    sig_fig, ref_fig, snr_fig = process_and_plot(nv_list, taus, sig_counts, ref_counts)

    ### Clean up and return

    tb.reset_cfm()

    kpl.show()

    timestamp = dm.get_time_stamp()
    raw_data |= {
        "timestamp": timestamp,
        "taus": taus,
        "tau-units": "ns",
        "min_tau": min_tau,
        "max_tau": max_tau,
    }

    repr_nv_sig = widefield.get_repr_nv_sig(nv_list)
    repr_nv_name = repr_nv_sig["name"]
    file_path = dm.get_file_path(__file__, timestamp, repr_nv_name)
    dm.save_raw_data(raw_data, file_path)
    file_path = dm.get_file_path(__file__, timestamp, repr_nv_name + "-sig")
    dm.save_figure(sig_fig, file_path)
    file_path = dm.get_file_path(__file__, timestamp, repr_nv_name + "-ref")
    dm.save_figure(ref_fig, file_path)
    file_path = dm.get_file_path(__file__, timestamp, repr_nv_name + "-snr")
    dm.save_figure(snr_fig, file_path)


if __name__ == "__main__":
    kpl.init_kplotlib()

    # file_name = "2023_11_27-19_31_32-johnson-nv0_2023_11_25"
    # data = dm.get_raw_data(file_name)
    data = dm.get_raw_data(file_id=1381739434842)  # 0.19
    # data = dm.get_raw_data(file_id=1381902242339)  # 0.14
    # data = dm.get_raw_data(file_id=)  # 0.17

    nv_list = data["nv_list"]
    taus = data["taus"]
    sig_counts = data["sig_counts"]
    ref_counts = data["ref_counts"]

    process_and_plot(nv_list, taus, sig_counts, ref_counts)

    plt.show(block=True)
