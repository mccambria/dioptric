# -*- coding: utf-8 -*-
"""
Lighweight check of the SCC SNR

Created on December 6th, 2023

@author: sbchand
"""

import sys
import time
import traceback

import numpy as np
from matplotlib import pyplot as plt

from majorroutines.widefield import base_routine
from utils import common
from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import positioning as pos
from utils import tool_belt as tb
from utils import widefield as widefield
from utils.constants import NVSig
from utils.positioning import get_scan_1d as calculate_powers


def process_and_plot(data):
    nv_list = data["nv_list"]
    num_nvs = len(nv_list)
    counts = np.array(data["counts"])
    # powers = data["powers"]
    powers = calculate_powers(0, 6, 16)
    sig_counts = counts[0]
    ref_counts = counts[1]

    norm_counts, norm_counts_ste = widefield.process_counts(
        nv_list, sig_counts, ref_counts, threshold=False
    )

    # There's only one point, so only consider that

    for nv_idx, nv in enumerate(nv_list):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.errorbar(
            powers,
            norm_counts[nv_idx],
            yerr=abs(norm_counts_ste[nv_idx]),
            fmt="o",
            label=f"NV {nv_idx + 1}",
        )

        ax.set_xlabel("Microwave power (dBm)")
        ax.set_ylabel("Normalized NV- population")
        ax.set_title("Power Rabi")
        ax.legend()
        ax.grid(True)
        plt.show(block=True)


def main(nv_list, num_steps, num_reps, num_runs, power_range, uwave_ind_list=[0, 1]):
    ### Some initial setup

    powers = calculate_powers(0, power_range, num_steps)
    seq_file = "power_rabi_scc_snr.py"
    pulse_gen = tb.get_server_pulse_gen()

    def run_fn(step_inds):
        seq_args = [widefield.get_base_scc_seq_args(nv_list, uwave_ind_list), step_inds]
        # print(seq_args)
        seq_args_string = tb.encode_seq_args(seq_args)
        pulse_gen.stream_load(seq_file, seq_args_string, num_reps)

    def step_fn(step_ind):
        power = powers[step_ind]
        for uwave_ind in uwave_ind_list:
            uwave_dict = tb.get_virtual_sig_gen_dict(uwave_ind)
            sig_gen = tb.get_server_sig_gen(uwave_ind)
            uwave_power = uwave_dict["uwave_power"]
            sig_gen = tb.get_server_sig_gen(uwave_ind)
            sig_gen.set_amp(round(uwave_power + power, 3))

    ### Collect the data

    data = base_routine.main(
        nv_list,
        num_steps,
        num_reps,
        num_runs,
        run_fn,
        step_fn,
        uwave_ind_list=uwave_ind_list,
        load_iq=True,  # freq modulation
    )

    ### Report results and cleanup
    timestamp = dm.get_time_stamp()
    repr_nv_name = widefield.get_repr_nv_sig(nv_list).name
    file_path = dm.get_file_path(__file__, timestamp, repr_nv_name)
    dm.save_raw_data(data, file_path)

    # process and plot
    try:
        figs = process_and_plot(data)
    except Exception:
        print(traceback.format_exc())
        figs = None

    if figs is not None:
        num_figs = len(figs)
        for ind in range(num_figs):
            file_path = dm.get_file_path(__file__, timestamp, repr_nv_name + f"-{ind}")
            dm.save_figure(figs[ind], file_path)

    tb.reset_cfm()


if __name__ == "__main__":
    kpl.init_kplotlib()

    data = dm.get_raw_data(file_id=1774297182266)
    avg_snr = process_and_plot(data)
    kpl.show(block=True)
