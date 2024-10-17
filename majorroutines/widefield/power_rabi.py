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

def get_lower_left_ax(axes_pack):
    """Helper function to find the lower-left axis from axes_pack."""
    if isinstance(axes_pack, dict):
        # Assuming the axes_pack dictionary has keys indicating positions (like a mosaic)
        # Let's extract the keys and find the one in the lower left
        lower_left_key = min(axes_pack.keys())  # Assuming keys represent positions and lower-left is smallest
        return axes_pack[lower_left_key]
    else:
        # If it's a list or something else, return the last axis
        return axes_pack[-1]

def create_raw_data_figure(data):
    nv_list = data["nv_list"]
    num_nvs = len(nv_list)
    powers = data["powers"]
    counts = np.array(data["states"])
    sig_counts, ref_counts = counts[0], counts[1]

    avg_counts, avg_counts_ste, norms = widefield.process_counts(
        nv_list, sig_counts, ref_counts, threshold=False
    )
    norm_counts = avg_counts - norms[0][:, np.newaxis]
    norm_counts_ste = avg_counts_ste

    fig, axes_pack, layout = kpl.subplot_mosaic(num_nvs, num_rows=2)

    widefield.plot_fit(axes_pack, nv_list, powers, norm_counts, norm_counts_ste)

    # kpl.set_shared_ax_xlabel(fig, axes_pack, layout, "Microwave power (dBm)")
    # kpl.set_shared_ax_ylabel(fig, axes_pack, layout, "Normalized NV- population")

    # Find the lower-left axis dynamically using the helper function
    lower_left_ax = get_lower_left_ax(axes_pack)
    kpl.set_shared_ax_xlabel(lower_left_ax, "Microwave power (dBm)")
    kpl.set_shared_ax_ylabel(lower_left_ax, "Normalized NV- population")

    return fig


def main(
    nv_list: list[NVSig],
    num_steps,
    num_reps,
    num_runs,
    power_range,
    uwave_ind_list=[0, 1],
):
    ### Some initial setup

    pulse_gen = tb.get_server_pulse_gen()
    powers = calculate_powers(0, power_range, num_steps)
    # powers = np.linspace(0, power_range, num_steps) + 1

    seq_file = "resonance_ref.py"

    ### Collect the data

    def run_fn(step_inds):
        seq_args = [widefield.get_base_scc_seq_args(nv_list, uwave_ind_list), step_inds]
        seq_args_string = tb.encode_seq_args(seq_args)
        pulse_gen.stream_load(seq_file, seq_args_string, num_reps)

    def step_fn(step_ind):
        power = powers[step_ind]
        for ind in uwave_ind_list:
            uwave_dict = tb.get_uwave_dict(ind)
            uwave_power = uwave_dict["uwave_power"]
            sig_gen = tb.get_server_sig_gen(ind=ind)
            sig_gen.set_amp(round(uwave_power + power, 3))

    data = base_routine.main(
        nv_list,
        num_steps,
        num_reps,
        num_runs,
        run_fn,
        step_fn,
        uwave_ind_list=uwave_ind_list,
    )

    ### Process and plot

    data["powers"] = powers
    try:
        raw_fig = create_raw_data_figure(data)
    except Exception as exc:
        print(exc)
        raw_fig = None

    ### Clean up and return

    tb.reset_cfm()
    kpl.show()

    timestamp = dm.get_time_stamp()
    data |= {
        "timestamp": timestamp,
        "power-units": "GHz",
        "power_range": power_range,
    }

    repr_nv_sig = widefield.get_repr_nv_sig(nv_list)
    repr_nv_name = repr_nv_sig.name
    file_path = dm.get_file_path(__file__, timestamp, repr_nv_name)
    dm.save_raw_data(data, file_path)
    if raw_fig is not None:
        dm.save_figure(raw_fig, file_path)


if __name__ == "__main__":
    kpl.init_kplotlib()

    data = dm.get_raw_data(file_id=1661020621314)

    raw_fig = create_raw_data_figure(data)

    kpl.show(block=True)
