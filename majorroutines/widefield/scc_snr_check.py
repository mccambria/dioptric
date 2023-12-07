# -*- coding: utf-8 -*-
"""
Lighweight check of the SCC SNR 

Created on December 6th, 2023

@author: mccambria
"""


from utils import tool_belt as tb
import numpy as np
from utils import common
from utils import widefield as widefield
from majorroutines.widefield import base_routine


def main(nv_list, uwave_list, uwave_ind, num_reps):
    with common.labrad_connect() as cxn:
        main_with_cxn(cxn, nv_list, uwave_list, uwave_ind, num_reps)


def main_with_cxn(cxn, nv_list, uwave_list, uwave_ind, num_reps):
    ### Some initial setup

    seq_file = "resonance_ref.py"
    sig_gen = tb.get_server_sig_gen(cxn, uwave_ind)
    sig_gen_name = sig_gen.name
    uwave_duration = tb.get_pi_pulse_dur(uwave_list[uwave_ind]["rabi_period"])
    pulse_gen = tb.get_server_pulse_gen(cxn)
    seq_args = widefield.get_base_scc_seq_args(nv_list)
    seq_args.extend([sig_gen_name, uwave_duration])
    seq_args_string = tb.encode_seq_args(seq_args)
    pulse_gen.stream_load(seq_file, seq_args_string, num_reps)

    ### Collect the data

    ret_vals = base_routine.main(
        cxn,
        nv_list,
        uwave_list,
        1,
        num_reps,
        1,
        step_fn=None,
        reference=True,
    )
    sig_counts, ref_counts = ret_vals

    ### Report the results and return

    avg_sig_counts, avg_sig_counts_ste = widefield.process_counts(sig_counts)
    avg_sig_counts = avg_sig_counts[:, 0]
    avg_sig_counts_ste = avg_sig_counts_ste[:, 0]
    avg_ref_counts, avg_ref_counts_ste = widefield.process_counts(ref_counts)
    avg_ref_counts = avg_ref_counts[:, 0]
    avg_ref_counts_ste = avg_ref_counts_ste[:, 0]
    avg_snr = (avg_sig_counts - avg_ref_counts) / np.sqrt(
        avg_sig_counts + avg_ref_counts
    )
    sig_coeff = ((avg_sig_counts + 3 * avg_ref_counts) ** 2) / (
        4 * (avg_sig_counts + avg_ref_counts) ** 3
    )
    ref_coeff = ((3 * avg_sig_counts + avg_ref_counts) ** 2) / (
        4 * (avg_sig_counts + avg_ref_counts) ** 3
    )
    avg_snr_ste = np.sqrt(
        sig_coeff * avg_sig_counts_ste**2 + ref_coeff * avg_ref_counts_ste**2
    )

    # Print
    for ind in range(len(nv_list)):
        nv_sig = nv_list[ind]
        nv_num = widefield.get_nv_num(nv_sig)
        nv_ref_counts = tb.round_for_print(avg_ref_counts[ind], avg_ref_counts_ste[ind])
        nv_sig_counts = tb.round_for_print(avg_sig_counts[ind], avg_sig_counts_ste[ind])
        nv_snr = tb.round_for_print(avg_snr[ind], avg_snr_ste[ind])
        print(f"NV {nv_num}: a0={nv_ref_counts}, a1={nv_sig_counts}, SNR={nv_snr}")

    tb.reset_cfm(cxn)
