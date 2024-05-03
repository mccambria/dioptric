# -*- coding: utf-8 -*-
"""
Optimize SCC parameters

Created on December 6th, 2023

@author: mccambria
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from majorroutines.widefield import base_routine
from majorroutines.widefield.scc_snr_check import process_and_print
from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import tool_belt as tb
from utils import widefield as widefield
from utils.constants import NVSig


def process_and_plot(data):
    nv_list = data["nv_list"]
    counts = np.array(data["counts"])
    # counts = data["states"]
    num_nvs = len(nv_list)
    # exclude_inds = (6, 9, 13)
    exclude_inds = ()
    nv_list = [nv_list[ind] for ind in range(num_nvs) if ind not in exclude_inds]
    num_nvs = len(nv_list)

    # Break down the counts array
    # experiment, nv, run, step, rep
    counts = np.delete(counts, exclude_inds, axis=1)
    sig_counts = np.array(counts[0])
    ref_counts = np.array(counts[1])

    sig_counts, ref_counts = widefield.threshold_counts(nv_list, sig_counts, ref_counts)

    # Calculate the correlations
    flattened_sig_counts = [sig_counts[ind].flatten() for ind in range(num_nvs)]
    flattened_ref_counts = [ref_counts[ind].flatten() for ind in range(num_nvs)]
    sig_corr_coeffs = np.corrcoef(flattened_sig_counts)
    ref_corr_coeffs = np.corrcoef(flattened_ref_counts)
    spin_flips = np.array([-1 if nv.spin_flip else +1 for nv in nv_list])
    ideal_sig_corr_coeffs = np.outer(spin_flips, spin_flips)
    ideal_sig_corr_coeffs = ideal_sig_corr_coeffs.astype(float)

    # Replace diagonals (Cii=1) with nan so they don't show
    np.fill_diagonal(sig_corr_coeffs, np.nan)
    np.fill_diagonal(ref_corr_coeffs, np.nan)
    np.fill_diagonal(ideal_sig_corr_coeffs, np.nan)

    # Make the colorbar symmetric about 0
    sig_max = np.nanmax(np.abs(sig_corr_coeffs))
    ref_max = np.nanmax(np.abs(ref_corr_coeffs))

    # test
    figs = []
    titles = ["Signal", "Reference", "Ideal signal"]
    vals = [sig_corr_coeffs, ref_corr_coeffs, ideal_sig_corr_coeffs]
    cbar_maxes = [sig_max, sig_max, 1]
    for ind in range(len(vals)):
        fig, ax = plt.subplots()
        cbar_max = cbar_maxes[ind]
        # cbar_max = 0.032
        kpl.imshow(
            ax,
            vals[ind],
            title=titles[ind],
            cbar_label="Correlation coefficient",
            cmap="RdBu_r",
            vmin=-cbar_max,
            vmax=cbar_max,
            nan_color=kpl.KplColors.GRAY,
        )
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        figs.append(fig)

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
        sig_fig, ref_fig = process_and_plot(raw_data)
    except Exception:
        sig_fig = None
        ref_fig = None

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
    if "img_arrays" in raw_data:
        keys_to_compress = ["img_arrays"]
    else:
        keys_to_compress = None
    dm.save_raw_data(raw_data, file_path, keys_to_compress)
    if sig_fig is not None:
        file_path = dm.get_file_path(__file__, timestamp, repr_nv_name + "-sig")
        dm.save_figure(sig_fig, file_path)
    if ref_fig is not None:
        file_path = dm.get_file_path(__file__, timestamp, repr_nv_name + "-ref")
        dm.save_figure(ref_fig, file_path)


if __name__ == "__main__":
    kpl.init_kplotlib()

    data = dm.get_raw_data(file_id=1519615059736)

    nv_list = data["nv_list"]
    counts = np.array(data["counts"])
    # counts = np.array(data["states"])
    process_and_print(nv_list, counts, threshold=True)
    process_and_plot(data)

    plt.show(block=True)
