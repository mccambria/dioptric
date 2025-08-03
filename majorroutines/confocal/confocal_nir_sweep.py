# -*- coding: utf-8 -*-
"""
Uses the generic confocal_base_routine to run a Rabi experiment
Created on August 2th, 2026

@author: schand
"""
import numpy as np
from utils import tool_belt as tb, data_manager as dm
from confocal_base_routine import main as confocal_base_main
from sequences import rabi_sequence  # Your pulse streamer rabi sequence loader


def main(
    pulse_streamer,
    tagger,
    seq_file,
    scan_coords,
    nir_values,
    num_reps,
    num_runs,
    apd_ch,
    apd_time,
    set_nir_fn,
    use_reference=True,
    norm_style="contrast",
):
    """
    Confocal NIR sweep experiment.
    """

    def seq_args_fn(step_idx):
        set_nir_fn(nir_values[step_idx])
        return []

    def apd_read_fn(tagger, apd_ch, apd_time):
        if use_reference:
            counts = tb.read_apd_2gates(tagger, apd_ch, apd_time)
        else:
            counts = tb.read_apd_counts(tagger, apd_ch, apd_time)
        return counts

    raw_data = confocal_base_main(
        pulse_streamer=pulse_streamer,
        seq_file=seq_file,
        seq_args_fn=seq_args_fn,
        scan_coords=scan_coords,
        num_steps=len(nir_values),
        num_reps=num_reps,
        num_runs=num_runs,
        apd_read_fn=apd_read_fn,
        tagger=tagger,
        apd_ch=apd_ch,
        apd_time=apd_time,
    )

    if use_reference:
        raw_counts = np.array(raw_data["counts"])
        norm, ste = tb.process_counts_array(
            raw_counts, gate_mode="2gate", norm_style=norm_style
        )
        raw_data["norm"] = norm.tolist()
        raw_data["norm_ste"] = ste.tolist()
        raw_data["norm_style"] = norm_style

    raw_data["nir_values"] = nir_values.tolist()
    raw_data["use_reference"] = use_reference
    dm.save_raw_data(raw_data, dm.get_file_path(__file__, raw_data["timestamp"]))
    return raw_data


if __name__ == "__main__":
    main()
