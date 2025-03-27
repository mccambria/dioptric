# -*- coding: utf-8 -*-
"""
Widefield Rabi experiment

Created on November 29th, 2023

@author: mccambria
@author: sbchand
"""

import sys
import time
import traceback

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from majorroutines.widefield import base_routine
from utils import common
from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import tool_belt as tb
from utils import widefield as widefield
from utils.constants import NVSig


def create_fit_figure(nv_list, phis, counts, counts_ste, norms):
    num_nvs = len(nv_list)
    phi_step = phis[1] - phis[0]
    num_steps = len(phis)

    norms_ms0_newaxis = norms[0][:, np.newaxis]
    norms_ms1_newaxis = norms[1][:, np.newaxis]
    contrast = norms_ms1_newaxis - norms_ms0_newaxis
    norm_counts = (counts - norms_ms0_newaxis) / contrast
    norm_counts_ste = counts_ste / contrast

    def cos_func(phi, amp, phase_offset):
        return 0.5 * amp * np.cos(phi - phase_offset) + 0.5

    fit_fns = []
    popts = []

    for nv_ind in range(num_nvs):
        nv_counts = norm_counts[nv_ind]
        nv_counts_ste = norm_counts_ste[nv_ind]

        guess_params = [1.0, 0.0]

        try:
            popt, _ = curve_fit(
                cos_func,
                phis,
                nv_counts,
                p0=guess_params,
                sigma=nv_counts_ste,
                absolute_sigma=True,
            )
        except Exception:
            popt = None

        fit_fns.append(cos_func if popt is not None else None)
        popts.append(popt)

        if popt is not None:
            residuals = cos_func(phis, *popt) - nv_counts
            chi_sq = np.sum((residuals / nv_counts_ste) ** 2)
            red_chi_sq = chi_sq / (len(nv_counts) - len(popt))
            print(f"Red chi sq: {round(red_chi_sq, 3)}")

    layout = kpl.calc_mosaic_layout(num_nvs, num_rows=2)
    fig, axes_pack = plt.subplot_mosaic(
        layout, figsize=[6.5, 5.0], sharex=True, sharey=True
    )
    axes_pack_flat = list(axes_pack.values())

    widefield.plot_fit(
        axes_pack_flat,
        nv_list,
        phis,
        norm_counts,
        norm_counts_ste,
        fit_fns,
        popts,
        xlim=[0, 2 * np.pi],
        no_legend=True,
    )

    ax = axes_pack[layout[-1][0]]
    kpl.set_shared_ax_xlabel(ax, "Phase (radians)")
    kpl.set_shared_ax_ylabel(ax, "Norm. NV$^{-}$ population")
    ax.set_xticks([0, np.pi, 2 * np.pi])
    ax.set_xticklabels(["0", "π", "2π"])
    ax.set_yticks([0, 1])

    return fig


def main(nv_list, num_steps, num_reps, num_runs, min_phi, max_phi, uwave_ind_list):
    ### Some initial setup

    pulse_gen = tb.get_server_pulse_gen()
    seq_file = "spin_echo_phase_scan.py"
    phi_list = np.linspace(0, 2 * np.pi, num_steps)

    ### Collect the data

    def run_fn(shuffled_step_inds):
        shuffled_taus = [phi_list[ind] for ind in shuffled_step_inds]
        seq_args = [
            widefield.get_base_scc_seq_args(nv_list, uwave_ind_list),
            shuffled_taus,
        ]
        # print(seq_args)
        seq_args_string = tb.encode_seq_args(seq_args)
        pulse_gen.stream_load(seq_file, seq_args_string, num_reps)

    raw_data = base_routine.main(
        nv_list,
        num_steps,
        num_reps,
        num_runs,
        run_fn=run_fn,
        uwave_ind_list=uwave_ind_list,
        save_images=False,
    )

    ### Process and plot

    try:
        raw_fig = None
        fit_fig = None
        # counts = raw_data["counts"]
        # sig_counts = counts[0]
        # ref_counts = counts[1]
        # avg_counts, avg_counts_ste, norms = widefield.process_counts(
        #     nv_list, sig_counts, ref_counts, threshold=True
        # )

        # raw_fig = create_raw_data_figure(nv_list, taus, avg_counts, avg_counts_ste)
        # fit_fig = create_fit_figure(nv_list, taus, avg_counts, avg_counts_ste, norms)
    except Exception:
        print(traceback.format_exc())
        raw_fig = None
        fit_fig = None

    ### Clean up and return

    tb.reset_cfm()
    kpl.show()

    timestamp = dm.get_time_stamp()
    raw_data |= {
        "timestamp": timestamp,
        "phis": phi_list,
        "phi-units": "radian",
        "min_tau": min_phi,
        "max_tau": max_phi,
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
