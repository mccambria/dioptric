# -*- coding: utf-8 -*-
"""
Confocal base routine for single-position pulse sequence experiments.
Handles APD readout, optional NIR toggling, and drift correction.

Created on August 2, 2025
@author: Saroj Chand
"""

import time
import numpy as np
from utils import tool_belt as tb
from utils import data_manager as dm


def main(
    pulse_streamer,
    seq_file,
    seq_args_fn,
    scan_coords,
    num_steps,
    num_reps,
    num_runs,
    apd_read_fn,
    tagger,
    apd_ch,
    apd_time,
    use_reference=False,
    norm_style="none",
    run_nir_fn=None,
):
    """
    Generic base routine for confocal APD-based single-position experiments (e.g. ESR, Rabi, Ramsey).

    Parameters:
        pulse_streamer: object with .stream_load(seq_file, seq_args_str, num_reps)
        seq_file: pulse sequence file to load
        seq_args_fn: function to generate sequence arguments for each step
        scan_coords: center coordinates [x, y, z] for scan
        num_steps: number of parameter scan steps (e.g. frequency, tau)
        num_reps: number of repetitions per step
        num_runs: number of full runs to average
        apd_read_fn: function to read APD counts using tagger
        tagger: PicoQuant tagger or compatible hardware
        apd_ch: APD channel index
        apd_time: APD collection window (s)
        use_reference: True if each point contains signal and reference gates
        norm_style: normalization style for signal/reference ("none", "contrast", etc.)
        run_nir_fn: optional function to toggle NIR (input: on/off bool)

    Returns:
        raw_data: dict with counts and metadata
    """
    tb.reset_cfm()

    drift = tb.get_drift()
    x_center, y_center, z_center = np.array(scan_coords) + np.array(drift)
    tb.get_xy_server().write_xy(x_center, y_center)
    tb.get_z_server().write_z(z_center)

    counts_arr = np.zeros((num_runs, num_steps))
    ref_arr = np.zeros_like(counts_arr) if use_reference else None

    for run_ind in range(num_runs):
        for step_ind in range(num_steps):
            if run_nir_fn:
                run_nir_fn(False)

            seq_args = seq_args_fn(step_ind)
            pulse_streamer.stream_load(seq_file, seq_args, num_reps)

            if use_reference:
                sig_counts, ref_counts = apd_read_fn(tagger, apd_ch, apd_time, gates=2)
                counts_arr[run_ind, step_ind] = sig_counts
                ref_arr[run_ind, step_ind] = ref_counts
            else:
                counts = apd_read_fn(tagger, apd_ch, apd_time)
                counts_arr[run_ind, step_ind] = counts

    if run_nir_fn:
        run_nir_fn(False)

    tb.reset_cfm()

    timestamp = dm.get_time_stamp()
    raw_data = {
        "timestamp": timestamp,
        "scan_coords": scan_coords,
        "num_steps": num_steps,
        "num_runs": num_runs,
        "num_reps": num_reps,
        "counts": counts_arr.tolist(),
        "ref_counts": ref_arr.tolist() if use_reference else None,
        "drift": drift,
        "apd_time": apd_time,
        "apd_ch": apd_ch,
        "nir_toggle": run_nir_fn is not None,
        "use_reference": use_reference,
        "norm_style": norm_style,
    }

    return raw_data
