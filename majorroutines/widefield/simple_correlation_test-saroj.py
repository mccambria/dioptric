# -*- coding: utf-8 -*-
"""
Optimize SCC parameters

Created on December 6th, 2023

@author: mccambria
"""

import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
from matplotlib.ticker import MaxNLocator
from scipy.optimize import curve_fit

from majorroutines.widefield import base_routine
from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import tool_belt as tb
from utils import widefield as widefield
from utils.constants import NVSig


def process_and_plot(
    data, ax=None, sig_or_ref=True, no_cbar=False, cbar_max=None, no_labels=False
):
    ### Unpack

    nv_list = data["nv_list"]
    counts = np.array(data["counts"])
    num_nvs = len(nv_list)

    passed_cbar_max = cbar_max

    # Break down the counts array
    # experiment, nv, run, step, rep
    sig_counts = np.array(counts[0])
    ref_counts = np.array(counts[1])

    sig_counts, ref_counts = widefield.threshold_counts(
        nv_list, sig_counts, ref_counts, None
    )

    ### Calculate the correlations
    flattened_sig_counts = [sig_counts[ind].flatten() for ind in range(num_nvs)]
    flattened_ref_counts = [ref_counts[ind].flatten() for ind in range(num_nvs)]

    sig_corr_coeffs = tb.nan_corr_coef(flattened_sig_counts)
    ref_corr_coeffs = tb.nan_corr_coef(flattened_ref_counts)

    spin_flips = np.array([-1 if nv.spin_flip else +1 for nv in nv_list])
    ideal_sig_corr_coeffs = np.outer(spin_flips, spin_flips)
    ideal_sig_corr_coeffs = ideal_sig_corr_coeffs.astype(float)

    # Replace diagonals (Cii=1) with nan so they don't show
    vals = [sig_corr_coeffs, ref_corr_coeffs, ideal_sig_corr_coeffs]
    for val in vals:
        np.fill_diagonal(val, np.nan)

    ### Plot

    # Make the colorbar symmetric about 0
    sig_max = np.nanmax(np.abs(sig_corr_coeffs))
    ref_max = np.nanmax(np.abs(ref_corr_coeffs))

    figs = []
    titles = ["Signal", "Reference", "Ideal signal"]
    cbar_maxes = [sig_max, sig_max, 1]
    for ind in range(len(vals)):
        if ax is None:
            fig, ax = plt.subplots()
            figs.append(fig)
        else:
            if sig_or_ref and ind != 0:
                continue
            if not sig_or_ref and ind != 1:
                continue
        if passed_cbar_max is not None:
            cbar_max = passed_cbar_max
        else:
            cbar_max = cbar_maxes[ind]
        kpl.imshow(
            ax,
            vals[ind],
            # title=titles[ind],
            cbar_label="Correlation coefficient",
            cmap="RdBu_r",
            vmin=-cbar_max,
            vmax=cbar_max,
            nan_color=kpl.KplColors.GRAY,
            no_cbar=no_cbar,
        )
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        if not no_labels:
            ax.set_xlabel("NV index")
            ax.set_ylabel("NV index")

    return figs


def main(nv_list, num_reps, num_runs):
    ### Some initial setup
    uwave_ind_list = [0, 1]
    seq_file = "simple_correlation_test.py"
    num_steps = 1

    pulse_gen = tb.get_server_pulse_gen()

    ### Collect the data

    def run_fn(shuffled_step_inds):
        seq_args = [widefield.get_base_scc_seq_args(nv_list, uwave_ind_list)]
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
    )

    ### Process and plot

    # process_and_print(nv_list, counts)
    try:
        figs = process_and_plot(raw_data)
    except Exception:
        figs = None

    ### Clean up and save data

    tb.reset_cfm()

    kpl.show()

    timestamp = dm.get_time_stamp()
    raw_data |= {
        "timestamp": timestamp,
    }

    repr_nv_sig = widefield.get_repr_nv_sig(nv_list)
    repr_nv_name = repr_nv_sig.name
    file_path = dm.get_file_path(__file__, timestamp, repr_nv_name)
    dm.save_raw_data(raw_data, file_path)

    if figs is not None:
        for ind in range(len(figs)):
            fig = figs[ind]
            file_path = dm.get_file_path(__file__, timestamp, f"{repr_nv_name}-{ind}")
            dm.save_figure(fig, file_path)


if __name__ == "__main__":
    kpl.init_kplotlib()

    data = dm.get_raw_data(file_id=1540048047866)  # Block

    process_and_plot(data)

    plt.show(block=True)
