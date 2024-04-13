# -*- coding: utf-8 -*-
"""
Illuminate an area, collecting onto the camera. Interleave a signal and control sequence
and plot the difference

Created on Fall 2023

@author: mccambria
"""

import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from scipy.optimize import curve_fit
from scipy.special import factorial

from majorroutines.widefield import base_routine, optimize
from utils import common, widefield
from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import positioning as pos
from utils import tool_belt as tb
from utils.constants import LaserKey, NVSig


def detect_cosmic_rays(nv_list, num_reps, num_runs, dark_time):
    charge_prep_fn = base_routine.charge_prep_loop
    data = main(nv_list, num_reps, num_runs, dark_time, charge_prep_fn)
    process_detect_cosmic_rays(data)


def process_detect_cosmic_rays(data):
    counts = data["counts"]
    sig_counts = counts[0]
    states, raw_data = widefield.threshold_counts(nv_list, sig_counts)


def check_readout_fidelity(nv_list, num_reps, num_runs):
    dark_time = 0
    charge_prep_fn = base_routine.charge_prep_loop_first_rep
    data = main(nv_list, num_reps, num_runs, dark_time, charge_prep_fn)
    process_check_readout_fidelity(data)


def process_check_readout_fidelity(data):
    counts = data["counts"]
    sig_counts = counts[0]
    states, raw_data = widefield.threshold_counts(nv_list, sig_counts)


def main(nv_list, num_reps, num_runs, dark_time, charge_prep_fn):
    ### Some initial setup
    seq_file = "charge_monitor.py"

    tb.reset_cfm()
    pulse_gen = tb.get_server_pulse_gen()

    num_steps = 1
    num_exps_per_rep = 1

    ### Collect the data

    def run_fn(shuffled_step_inds):
        pol_coords_list = widefield.get_coords_list(nv_list, LaserKey.CHARGE_POL)
        ion_coords_list = widefield.get_coords_list(nv_list, LaserKey.ION)
        seq_args = [pol_coords_list, ion_coords_list, dark_time]
        seq_args_string = tb.encode_seq_args(seq_args)
        pulse_gen.stream_load(seq_file, seq_args_string, num_reps)

    counts, raw_data = base_routine.main(
        nv_list,
        num_steps,
        num_reps,
        num_runs,
        run_fn=run_fn,
        num_exps_per_rep=num_exps_per_rep,
        charge_prep_fn=charge_prep_fn,
    )

    ### Save and clean up

    repr_nv_sig = widefield.get_repr_nv_sig(nv_list)
    repr_nv_name = repr_nv_sig.name
    timestamp = dm.get_time_stamp()
    file_path = dm.get_file_path(__file__, timestamp, repr_nv_name)
    raw_data |= {
        "timestamp": timestamp,
    }
    dm.save_raw_data(raw_data, file_path)

    tb.reset_cfm()

    return data


if __name__ == "__main__":
    kpl.init_kplotlib()

    # data = dm.get_raw_data(file_id=1496976806208, load_npz=True)
    data = dm.get_raw_data(file_id=1499208769470, load_npz=True)

    nv_list = data["nv_list"]
    nv_list = [NVSig(**nv) for nv in nv_list]
    num_nvs = len(nv_list)
    sig_counts_lists = data["sig_counts_lists"]
    ref_counts_lists = data["ref_counts_lists"]
    num_shots = len(sig_counts_lists[0])

    kpl.show(block=True)
