# -*- coding: utf-8 -*-
"""
Confocal rabi experiment using base routine.
Sweeps microwave frequency, reads signal and reference counts via APD tagger.

Created on Augu 2, 2025
@author: Saroj Chand
"""

import os
import time
from random import shuffle

import labrad
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

import majorroutines.confocal as confocal_base_routine
import majorroutines.targeting as targeting
import utils.confocal_utils as confocal_utils
import utils.data_manager as dm
import utils.kplotlib as kpl
import utils.positioning as positioning
import utils.tool_belt as tb


def main(nv_sig, num_steps, num_reps, num_runs, min_tau, max_tau, uwave_ind_list):
    ### Some initial setup

    pulse_streamer = tb.get_server_pulse_streamer()
    seq_file = "rabi_seq.py"
    taus = np.linspace(min_tau, max_tau, num_steps)

    ### Collect the data
    def run_fn(shuffled_step_inds):
        shuffled_taus = [taus[ind] for ind in shuffled_step_inds]
        seq_args = confocal_utils.get_base_seq_args(
            nv_sig, uwave_ind_list, shuffled_taus
        )
        seq_args_string = tb.encode_seq_args(seq_args)
        pulse_streamer.stream_load(seq_file, seq_args_string, num_reps)

    raw_data = confocal_base_routine.main(
        nv_sig,
        num_steps,
        num_reps,
        num_runs,
        run_fn=run_fn,
        uwave_ind_list=uwave_ind_list,
    )

    ### save the rawa data
    timestamp = dm.get_time_stamp()
    raw_data |= {
        "timestamp": timestamp,
        "taus": taus,
        "tau-units": "ns",
        "min_tau": max_tau,
        "max_tau": max_tau,
    }
    # save the raw data
    nv_name = nv_sig.name
    file_path = dm.get_file_path(__file__, timestamp, nv_name)
    dm.save_raw_data(raw_data, file_path)

    ### Process and plot

    try:
        raw_fig = None
        fit_fig = None
        counts = raw_data["counts"]
        readout = nv_sig["spin_readout_dur"]
        sig_counts = counts[0]
        ref_counts = counts[1]
        norm_counts, norm_counts_ste, sig_counts_avg_kcps, ref_counts_avg_kcps = (
            confocal_utils.process_counts(nv_sig, sig_counts, ref_counts, readout)
        )

        raw_fig = create_raw_data_figure(nv_sig, taus, norm_counts, norm_counts_ste)
        fit_fig = create_fit_figure(nv_sig, taus, norm_counts, norm_counts_ste)
    except Exception:
        print(traceback.format_exc())
        raw_fig = None
        fit_fig = None

    if raw_fig is not None:
        dm.save_figure(raw_fig, file_path)
    if fit_fig is not None:
        file_path = dm.get_file_path(__file__, timestamp, nv_name + "-fit")
        dm.save_figure(fit_fig, file_path)
    ### Clean up and return

    tb.reset_cfm()
    kpl.show()


if __name__ == "__main__":
    path = "pc_rabi/branch_master/rabi/2023_01"
    file = "2023_01_27-09_42_22-siena-nv4_2023_01_16"
    data = tb.get_raw_data(file, path)
    kpl.init_kplotlib()

    norm_avg_sig = data["norm_avg_sig"]
    uwave_time_range = data["uwave_time_range"]
    num_steps = data["num_steps"]
    uwave_freq = data["uwave_freq"]
    norm_avg_sig_ste = None

    # fit_func = tool_belt.cosexp_1_at_0

    sig_counts = data["sig_counts"]
    ref_counts = data["ref_counts"]
    num_reps = data["num_reps"]
    nv_sig = data["nv_sig"]
    readout = nv_sig["spin_readout_dur"]
    ret_vals = tb.process_counts(sig_counts, ref_counts, num_reps, readout)
    (
        sig_counts_avg_kcps,
        ref_counts_avg_kcps,
        norm_avg_sig,
        norm_avg_sig_ste,
    ) = ret_vals

# %%
