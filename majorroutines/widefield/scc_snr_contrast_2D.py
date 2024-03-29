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


def main(nv_list, snr_coords_list, num_reps, num_runs):
    ### Some initial setup

    uwave_ind = 0
    uwave_dict = tb.get_uwave_dict(uwave_ind)
    uwave_freq = uwave_dict["frequency"]

    seq_file = "resonance_ref.py"
    pulse_gen = tb.get_server_pulse_gen()

    def run_fn(step_inds):
        seq_args = widefield.get_base_scc_seq_args(nv_list)
        seq_args.append(uwave_ind)
        shuffled_freqs = [uwave_freq]  # Just one frequency, not used by sequence anyway
        seq_args.append(shuffled_freqs)
        seq_args_string = tb.encode_seq_args(seq_args)
        pulse_gen.stream_load(seq_file, seq_args_string, num_reps)

    ### Collect the data

    counts, _ = base_routine.main(
        nv_list,
        snr_coords_list,
        1,
        num_reps,
        num_runs,
        run_fn=run_fn,
        uwave_ind=uwave_ind,
        save_images=False,
    )
    sig_counts = counts[0]
    ref_counts = counts[1]

    ### Report the results and return

    scc_snr_contrast = np.zeros((len(x_range), len(y_range)))
    scc_snr_contrast_ste = np.zeros((len(x_range), len(y_range)))

    for i in enumerate(x_range):
        for j in enumerate(y_range):
            avg_snr, avg_snr_ste = widefield.calc_snr(sig_counts, ref_counts)
            avg_sig_counts, avg_sig_counts_ste = widefield.average_counts(sig_counts)
            avg_ref_counts, avg_ref_counts_ste = widefield.average_counts(ref_counts)
            scc_snr_contrast[i, j] = avg_snr
            scc_snr_contrast_ste[i, j] = avg_snr_ste

    return scc_snr_contrast, scc_snr_contrast_ste

    for ind in range(len(nv_list)):
        nv_sig = nv_list[ind]
        nv_num = widefield.get_nv_num(nv_sig)
        nv_ref_counts = tb.round_for_print(avg_ref_counts[ind], avg_ref_counts_ste[ind])
        nv_sig_counts = tb.round_for_print(avg_sig_counts[ind], avg_sig_counts_ste[ind])
        nv_snr = tb.round_for_print(avg_snr[ind], avg_snr_ste[ind])
        print(f"NV {nv_num}: a0={nv_ref_counts}, a1={nv_sig_counts}, SNR={nv_snr}")
    tb.reset_cfm()
