# -*- coding: utf-8 -*-
"""
Optimize SCC parameters

Created on December 6th, 2023

@author: mccambria
"""

import traceback

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from majorroutines.widefield import base_routine
from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import tool_belt as tb
from utils import widefield as widefield
from utils.constants import NVSig, VirtualLaserKey


def process_and_plot(nv_list, step_vals, sig_counts, ref_counts):
    """
    Process and plot the results of the SCC optimization experiment.

    Parameters
    ----------
    nv_list : list
        List of NVs involved in the experiment.
    step_vals : ndarray
        Array of duration and amplitude step values.
    sig_counts : ndarray
        Signal counts collected during the experiment.
    ref_counts : ndarray
        Reference counts collected during the experiment.

    Returns
    -------
    figs : list
        List of matplotlib figures generated during plotting.
    """
    num_nvs = len(nv_list)

    # Threshold and normalize the counts
    sig_counts, ref_counts = widefield.threshold_counts(
        nv_list, sig_counts, ref_counts, dynamic_thresh=True
    )

    # Calculate average signal, reference, and SNR
    avg_sig_counts, avg_sig_counts_ste, _ = widefield.average_counts(sig_counts)
    avg_ref_counts, avg_ref_counts_ste, _ = widefield.average_counts(ref_counts)
    avg_snr, avg_snr_ste = widefield.calc_snr(sig_counts, ref_counts)

    duration_vals = step_vals[:, 0].reshape(-1, len(np.unique(step_vals[:, 1])))
    amp_vals = step_vals[:, 1].reshape(-1, len(np.unique(step_vals[:, 1])))

    # Create heatmaps for SNR
    snr_fig, snr_ax = plt.subplots()
    snr_map = avg_snr.mean(axis=0).reshape(duration_vals.shape)
    c = snr_ax.pcolormesh(
        duration_vals, amp_vals, snr_map, shading="auto", cmap="viridis"
    )
    snr_ax.set_xlabel("SCC Pulse Duration (ns)")
    snr_ax.set_ylabel("SCC Amplitude (relative)")
    snr_ax.set_title("Average SNR Heatmap")
    snr_fig.colorbar(c, ax=snr_ax)

    # Individual NV plots for SNR
    nv_figs = []
    for nv_ind in range(num_nvs):
        fig, ax = plt.subplots()
        kpl.plot_points(
            ax,
            step_vals[:, 0],
            avg_snr[nv_ind],
            yerr=avg_snr_ste[nv_ind],
            label=f"NV {nv_ind + 1}",
        )
        ax.set_xlabel("SCC Pulse Duration (ns)")
        ax.set_ylabel("SNR")
        ax.legend()
        fig.suptitle(f"NV {nv_ind + 1} SNR vs. SCC Duration")
        nv_figs.append(fig)

    # Aggregate plots
    figs = [snr_fig] + nv_figs

    return figs


def optimize_scc_amp_and_duration(
    nv_list,
    num_amp_steps,
    num_dur_steps,
    num_reps,
    num_runs,
    min_amp,
    max_amp,
    min_duration,
    max_duration,
):
    """
    Main function to optimize SCC parameters over amplitude and duration.

    Parameters
    ----------
    nv_list : list
        List of NVs to optimize.
    num_amp_steps : int
        Number of steps for amplitude.
    num_dur_steps : int
        Number of steps for duration.
    num_reps : int
        Number of repetitions for each step.
    num_runs : int
        Number of experimental runs.
    min_step_val : tuple
        Minimum values for duration and amplitude.
    max_step_val : tuple
        Maximum values for duration and amplitude.

    Returns
    -------
    None
    """
    ### Initial setup
    seq_file = "optimize_scc_amp_duration_seq.py"

    # Generate grid of parameter values
    duration_vals = np.linspace(min_duration, max_duration, num_dur_steps).astype(int)
    amp_vals = np.linspace(min_amp, max_amp, num_amp_steps)
    step_vals = np.array(np.meshgrid(duration_vals, amp_vals)).T.reshape(-1, 2)
    num_steps = num_amp_steps * num_dur_steps

    uwave_ind_list = [0, 1]
    pulse_gen = tb.get_server_pulse_gen()

    ### Define run function
    def run_fn(shuffled_step_inds):
        shuffled_step_vals = step_vals[shuffled_step_inds]
        seq_args = [
            widefield.get_base_scc_seq_args(nv_list, uwave_ind_list),
            shuffled_step_vals,
        ]

        seq_args_string = tb.encode_seq_args(seq_args)
        pulse_gen.stream_load(seq_file, seq_args_string, num_reps)

    ### Run the experiment
    raw_data = base_routine.main(nv_list, num_steps, num_reps, num_runs, run_fn=run_fn)

    ### Process and plot results
    counts = raw_data["counts"]
    sig_counts = counts[0]
    ref_counts = counts[1]

    try:
        figs = process_and_plot(nv_list, step_vals, sig_counts, ref_counts)
    except Exception as e:
        print("Error in process_and_plot:", e)
        print(traceback.format_exc())
        figs = None

    ### Save results
    timestamp = dm.get_time_stamp()
    raw_data |= {
        "timestamp": timestamp,
        "step_vals": step_vals,
        "step-units": ["ns", "relative"],
        "min_amp": min_amp,
        "max_amp": max_amp,
        "min_duration": min_duration,
        "max_duration": max_duration,
    }

    repr_nv_sig = widefield.get_repr_nv_sig(nv_list)
    repr_nv_name = repr_nv_sig.name
    file_path = dm.get_file_path(__file__, timestamp, repr_nv_name)
    if "img_arrays" in raw_data:
        keys_to_compress = ["img_arrays"]
    else:
        keys_to_compress = None
    dm.save_raw_data(raw_data, file_path, keys_to_compress)

    if figs is not None:
        for ind, fig in enumerate(figs):
            file_path = dm.get_file_path(__file__, timestamp, f"{repr_nv_name}-{ind}")
            dm.save_figure(fig, file_path)


if __name__ == "__main__":
    kpl.init_kplotlib()

    data = dm.get_raw_data(file_id=1564881159891)

    nv_list = data["nv_list"]
    taus = data["taus"]
    counts = np.array(data["counts"])
    sig_counts = counts[0]
    ref_counts = counts[1]

    # sig_counts, ref_counts = widefield.threshold_counts(nv_list, sig_counts, ref_counts)

    process_and_plot(nv_list, taus, sig_counts, ref_counts, False)

    plt.show(block=True)
