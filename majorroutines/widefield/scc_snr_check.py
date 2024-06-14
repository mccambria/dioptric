# -*- coding: utf-8 -*-
"""
Lighweight check of the SCC SNR

Created on December 6th, 2023

@author: mccambria
"""

import time
import traceback

import numpy as np
from matplotlib import pyplot as plt

from majorroutines.widefield import base_routine
from utils import common
from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import tool_belt as tb
from utils import widefield as widefield


def process_and_plot(data):
    threshold = True
    nv_list = data["nv_list"]
    counts = np.array(data["counts"])
    sig_counts = counts[0]
    ref_counts = counts[1]

    if threshold:
        sig_counts, ref_counts = widefield.threshold_counts(
            nv_list, sig_counts, ref_counts
        )

    ### Report the results and return

    avg_sig_counts, avg_sig_counts_ste, _ = widefield.process_counts(
        nv_list, sig_counts, threshold=False
    )
    avg_ref_counts, avg_ref_counts_ste, _ = widefield.process_counts(
        nv_list, ref_counts, threshold=False
    )
    avg_snr, avg_snr_ste = widefield.calc_snr(sig_counts, ref_counts)
    avg_contrast, avg_contrast_ste = widefield.calc_contrast(sig_counts, ref_counts)

    # There's only one point, so only consider that
    avg_sig_counts = avg_sig_counts[:, 0]
    avg_sig_counts_ste = avg_sig_counts_ste[:, 0]
    avg_ref_counts = avg_ref_counts[:, 0]
    avg_ref_counts_ste = avg_ref_counts_ste[:, 0]
    avg_snr = avg_snr[:, 0]
    avg_snr_ste = avg_snr_ste[:, 0]
    avg_contrast = avg_contrast[:, 0]
    avg_contrast_ste = avg_contrast_ste[:, 0]

    # fig, ax = plt.subplots()
    # kpl.histogram(ax, sig_counts[6].flatten())

    # Print
    for ind in range(len(nv_list)):
        nv_sig = nv_list[ind]
        nv_num = widefield.get_nv_num(nv_sig)
        nv_ref_counts = tb.round_for_print(avg_ref_counts[ind], avg_ref_counts_ste[ind])
        nv_sig_counts = tb.round_for_print(avg_sig_counts[ind], avg_sig_counts_ste[ind])
        nv_snr = tb.round_for_print(avg_snr[ind], avg_snr_ste[ind])
        print(f"NV {nv_num}: a0={nv_ref_counts}, a1={nv_sig_counts}, SNR={nv_snr}")
    print(f"Mean SNR: {np.mean(avg_snr)}")

    ### Plot

    figsize = kpl.figsize
    figsize[1] *= 1.5
    counts_fig, axes_pack = plt.subplots(2, 1, sharex=True, figsize=figsize)
    snr_fig, ax = plt.subplots()
    # fid_fig, fid_ax = plt.subplots()

    for ind in range(len(nv_list)):
        nv_sig = nv_list[ind]
        nv_num = widefield.get_nv_num(nv_sig)
        kpl.plot_bars(
            axes_pack[0], nv_num, avg_ref_counts[ind], yerr=avg_ref_counts_ste[ind]
        )
        kpl.plot_bars(
            axes_pack[1], nv_num, avg_sig_counts[ind], yerr=avg_sig_counts_ste[ind]
        )
        kpl.plot_bars(ax, nv_num, avg_snr[ind], yerr=avg_snr_ste[ind])
        # kpl.plot_bars(ax, nv_num, avg_contrast[ind], yerr=avg_contrast_ste[ind])

    axes_pack[0].set_xlabel("NV index")
    ax.set_xlabel("NV index")
    axes_pack[0].set_ylabel("NV- | prep in ms=0")
    axes_pack[1].set_ylabel("NV- | prep in ms=1")
    ax.set_ylabel("SNR")
    # ax.set_ylabel("Contrast")
    # ax.set_ylim([0, 0.2])

    # axes_pack[0].set_ylabel("NV$^{-}$ population after prep in ms=0")
    # axes_pack[1].set_ylabel("NV$^{-}$ population after prep in ms=1")
    # ax.set_ylabel("Spin experiment SNR")
    # fid_ax.set_ylabel("Spin experiment fidelity")

    return counts_fig, snr_fig
    # return counts_fig, snr_fig, fid_fig


def main(nv_list, num_reps, num_runs, scc_include_inds=None, uwave_ind_list=[0, 1]):
    ### Some initial setup

    # uwave_ind_list = [0]
    # uwave_ind_list = [1]
    # uwave_ind_list = [0, 1]
    # uwave_ind_list = []

    seq_file = "scc_snr_check.py"
    pulse_gen = tb.get_server_pulse_gen()

    def run_fn(step_inds):
        seq_args = [
            widefield.get_base_scc_seq_args(nv_list, uwave_ind_list, scc_include_inds)
        ]
        seq_args_string = tb.encode_seq_args(seq_args)
        pulse_gen.stream_load(seq_file, seq_args_string, num_reps)

    ### Collect the data

    data = base_routine.main(
        nv_list,
        1,
        num_reps,
        num_runs,
        run_fn=run_fn,
        uwave_ind_list=uwave_ind_list,
        save_images=False,
        charge_prep_fn=None,
    )

    ### Report results and cleanup

    try:
        figs = process_and_plot(data)
    except Exception:
        print(traceback.format_exc())
        figs = None

    timestamp = dm.get_time_stamp()

    repr_nv_name = widefield.get_repr_nv_sig(nv_list).name
    file_path = dm.get_file_path(__file__, timestamp, repr_nv_name)
    dm.save_raw_data(data, file_path)
    if figs is not None:
        num_figs = len(figs)
        for ind in range(num_figs):
            file_path = dm.get_file_path(__file__, timestamp, repr_nv_name + f"-{ind}")
            dm.save_figure(figs[ind], file_path)

    tb.reset_cfm()


if __name__ == "__main__":
    kpl.init_kplotlib()

    # data = dm.get_raw_data(file_id=1548854318015)  # 6/2 benchmark
    data = dm.get_raw_data(file_id=1560609724329)
    figs = process_and_plot(data)
    kpl.show(block=True)
