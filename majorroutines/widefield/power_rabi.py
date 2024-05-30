# -*- coding: utf-8 -*-
"""
Pulsed electron spin resonance on multiple NVs with spin-to-charge
conversion readout imaged onto a camera

Created on November 19th, 2023

@author: mccambria
"""

import os
import sys
import time
from random import shuffle

import matplotlib.pyplot as plt
import numpy as np

from majorroutines.pulsed_resonance import fit_resonance, voigt, voigt_split
from majorroutines.widefield import base_routine, optimize
from utils import common
from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import positioning as pos
from utils import tool_belt as tb
from utils import widefield as widefield
from utils.constants import NVSig, NVSpinState
from utils.positioning import get_scan_1d as calculate_powers


def create_raw_data_figure():
    layout = kpl.calc_mosaic_layout(num_nvs, num_rows=2)
    # layout = kpl.calc_mosaic_layout(num_nvs)
    fig, axes_pack = plt.subplot_mosaic(
        layout, figsize=[6.5, 6.0], sharex=True, sharey=True
    )
    axes_pack_flat = list(axes_pack.values())

    widefield.plot_fit(axes_pack_flat, nv_list, taus, norm_counts, norm_counts_ste)
    ax = axes_pack[layout[-1, 0]]
    ax.set_xlabel(" ")
    fig.text(0.55, 0.01, "Pulse duration (ns)", ha="center")
    ax.set_ylabel(" ")
    label = "Change in fraction in NV$^{-}$"
    fig.text(0.01, 0.55, label, va="center", rotation="vertical")
    # ax.set_ylim([0.966, 1.24])
    # ax.set_yticks([1.0, 1.2])
    return fig


def main(
    nv_list: list[NVSig],
    num_steps,
    num_reps,
    num_runs,
    power_center,
    power_range,
    uwave_ind=0,
):
    ### Some initial setup

    pulse_gen = tb.get_server_pulse_gen()
    sig_gen = tb.get_server_sig_gen()
    powers = calculate_powers(power_center, power_range, num_steps)

    seq_file = "resonance_ref.py"

    ### Collect the data

    def run_fn(step_inds):
        seq_args = [widefield.get_base_scc_seq_args(nv_list, uwave_ind), step_inds]
        seq_args_string = tb.encode_seq_args(seq_args)
        pulse_gen.stream_load(seq_file, seq_args_string, num_reps)

    def step_fn(step_ind):
        power = powers[step_ind]
        sig_gen.set_power(power)

    raw_data = base_routine.main(
        nv_list,
        num_steps,
        num_reps,
        num_runs,
        run_fn,
        step_fn,
        uwave_ind_list=uwave_ind,
    )

    ### Process and plot

    try:
        counts = raw_data["states"]
        sig_counts = counts[0]
        ref_counts = counts[1]

        avg_counts, avg_counts_ste, norms = widefield.process_counts(
            nv_list, sig_counts, ref_counts, threshold=False
        )
        raw_fig = create_raw_data_figure(nv_list, powers, avg_counts, avg_counts_ste)
    except Exception as exc:
        print(exc)
        raw_fig = None
        fit_fig = None

    ### Clean up and return

    tb.reset_cfm()
    kpl.show()

    timestamp = dm.get_time_stamp()
    raw_data |= {
        "timestamp": timestamp,
        "powers": powers,
        "power-units": "GHz",
        "power_range": power_range,
        "power_center": power_center,
    }

    repr_nv_sig = widefield.get_repr_nv_sig(nv_list)
    repr_nv_name = repr_nv_sig.name
    file_path = dm.get_file_path(__file__, timestamp, repr_nv_name)
    dm.save_raw_data(raw_data, file_path)
    if raw_fig is not None:
        dm.save_figure(raw_fig, file_path)
    if fit_fig is not None:
        file_path = dm.get_file_path(__file__, timestamp, repr_nv_name + "-fit")
        dm.save_figure(fit_fig, file_path)


if __name__ == "__main__":
    kpl.init_kplotlib()

    # data = dm.get_raw_data(file_id=1538544646977, load_npz=True)
    # data = dm.get_raw_data(file_id=1541455417524)
    # data = dm.get_raw_data(file_id=1519797150132)
    data = dm.get_raw_data(file_id=1541604395737)

    nv_list = data["nv_list"]
    num_nvs = len(nv_list)
    num_steps = data["num_steps"]
    num_runs = data["num_runs"]
    num_reps = data["num_reps"]
    powers = data["freqs"]

    counts = np.array(data["counts"])
    # counts = np.array(data["states"])
    ref_counts = counts[1]
    counts = counts[:, :, :, :, 0:1:]
    # counts = counts[:, :, :, :, 1:2:]
    # counts = counts[:, :, :, :, 2:3:]
    # counts = counts[:, :, :, :, 4:5:]
    # counts = counts[:, :, :, :, 9:10:]
    # counts = np.array(data["counts"])
    sig_counts = counts[0]
    # ref_counts = counts[1]

    avg_counts, avg_counts_ste, norms = widefield.process_counts(
        nv_list, sig_counts, ref_counts, threshold=False
    )

    raw_fig = create_raw_data_figure(nv_list, powers, avg_counts, avg_counts_ste)
    fit_fig = create_fit_figure(nv_list, powers, avg_counts, avg_counts_ste, norms)

    # img_arrays = np.array(data["mean_img_arrays"])[0]
    # proc_img_arrays = widefield.downsample_img_arrays(img_arrays, 3)

    # bottom = np.percentile(proc_img_arrays, 30, axis=0)
    # proc_img_arrays -= bottom

    norms_newaxis = norms[:, np.newaxis]
    avg_counts = avg_counts - norms_newaxis
    # widefield.animate(
    #     freqs,
    #     nv_list,
    #     avg_counts,
    #     avg_counts_ste,
    #     proc_img_arrays,
    #     cmin=0.01,
    #     cmax=0.04,
    # )

    kpl.show(block=True)
