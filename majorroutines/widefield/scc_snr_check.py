# -*- coding: utf-8 -*-
"""
Lighweight check of the SCC SNR

Created on December 6th, 2023

@author: mccambria
"""

import time

import numpy as np

from majorroutines.widefield import base_routine
from utils import common
from utils import tool_belt as tb
from utils import widefield as widefield


def main(nv_list, num_reps, num_runs):
    ### Some initial setup

    uwave_ind = 0
    seq_file = "resonance_ref.py"
    pulse_gen = tb.get_server_pulse_gen()
    seq_args = widefield.get_base_scc_seq_args(nv_list)
    seq_args.append(uwave_ind)
    seq_args_string = tb.encode_seq_args(seq_args)

    def step_fn(step_ind):
        pulse_gen.stream_load(seq_file, seq_args_string, num_reps)

    ### Collect the data

    counts, _ = base_routine.main(
        nv_list, 1, num_reps, num_runs, step_fn=step_fn, save_images=False
    )
    sig_counts = counts[0]
    ref_counts = counts[1]

    ### Report the results and return

    avg_sig_counts, avg_sig_counts_ste = widefield.average_counts(sig_counts)
    avg_ref_counts, avg_ref_counts_ste = widefield.average_counts(ref_counts)
    avg_snr, avg_snr_ste = widefield.calc_snr(sig_counts, ref_counts)

    # There's only one point, so only consider that
    avg_sig_counts = avg_sig_counts[:, 0]
    avg_sig_counts_ste = avg_sig_counts_ste[:, 0]
    avg_ref_counts = avg_ref_counts[:, 0]
    avg_ref_counts_ste = avg_ref_counts_ste[:, 0]
    avg_snr = avg_snr[:, 0]
    avg_snr_ste = avg_snr_ste[:, 0]

    # Print
    for ind in range(len(nv_list)):
        nv_sig = nv_list[ind]
        nv_num = widefield.get_nv_num(nv_sig)
        nv_ref_counts = tb.round_for_print(avg_ref_counts[ind], avg_ref_counts_ste[ind])
        nv_sig_counts = tb.round_for_print(avg_sig_counts[ind], avg_sig_counts_ste[ind])
        nv_snr = tb.round_for_print(avg_snr[ind], avg_snr_ste[ind])
        print(f"NV {nv_num}: a0={nv_ref_counts}, a1={nv_sig_counts}, SNR={nv_snr}")

    tb.reset_cfm()
