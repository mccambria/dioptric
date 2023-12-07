# -*- coding: utf-8 -*-
"""
Optimize SCC parameters

Created on December 6th, 2023

@author: mccambria
"""


import copy
from random import shuffle
import sys
import matplotlib.pyplot as plt
import numpy as np
from majorroutines.widefield import optimize
from utils import tool_belt as tb
from utils import data_manager as dm
from utils import common
from utils import widefield as widefield
from utils import kplotlib as kpl
from utils import positioning as pos
from utils import data_manager as dm
from utils.constants import LaserKey, NVSpinState
import os
import time
from utils.positioning import get_scan_1d as calculate_freqs
from majorroutines.pulsed_resonance import fit_resonance, voigt_split, voigt


def main_with_cxn(cxn, nv_list, uwave_nv, state, num_steps, num_reps, num_runs):
    ### Some initial setup

    tb.reset_cfm(cxn)

    # First NV to represent the others
    repr_nv_ind = 0
    repr_nv_sig = nv_list[repr_nv_ind]
    pos.set_xyz_on_nv(cxn, repr_nv_sig)
    num_nvs = len(nv_list)
    nv_list_mod = copy.deepcopy(nv_list)

    camera = tb.get_server_camera(cxn)
    pulse_gen = tb.get_server_pulse_gen(cxn)
    sig_gen = tb.get_server_sig_gen(cxn, state)
    sig_gen_name = sig_gen.name

    uwave_dict = uwave_nv[state]
    uwave_duration = tb.get_pi_pulse_dur(uwave_dict["rabi_period"])
    uwave_power = uwave_dict["uwave_power"]
    freq = uwave_dict["frequency"]
    sig_gen.set_amp(uwave_power)
    sig_gen.set_freq(freq)

    seq_file = "resonance_ref.py"

    ### Data tracking

    sig_counts = np.empty((num_nvs, num_runs, num_steps, num_reps))
    ref_counts = np.empty((num_nvs, num_runs, num_steps, num_reps))
    step_ind_master_list = [[] for ind in range(num_runs)]
    step_ind_list = list(range(0, num_steps))

    ### Collect the data

    for run_ind in range(num_runs):
        shuffle(step_ind_list)

        camera.arm()
        sig_gen.uwave_on()

        for step_ind in step_ind_list:
            pixel_coords_list = [widefield.get_nv_pixel_coords(nv) for nv in nv_list]
            step_ind_master_list[run_ind].append(step_ind)

            for nv in nv_list_mod:
                nv[LaserKey.IONIZATION]["duration"] = tau
            seq_args = widefield.get_base_scc_seq_args(nv_list_mod)
            seq_args.extend([sig_gen_name, uwave_duration])
            seq_args_string = tb.encode_seq_args(seq_args)
            pulse_gen.stream_load(seq_file, seq_args_string, num_reps)

            step_fn(step_ind)

            # Try 5 times then give up
            num_attempts = 5
            attempt_ind = 0
            while True:
                try:
                    pulse_gen.stream_start()
                    for rep_ind in range(num_reps):
                        for sig_ref_ind in range(2):
                            img_str = camera.read()
                            img_array = widefield.img_str_to_array(img_str)
                            for nv_ind in range(num_nvs):
                                pixel_coords = pixel_coords_list[nv_ind]
                                counts_val = widefield.integrate_counts_from_adus(
                                    img_array, pixel_coords
                                )
                                counts = sig_counts if sig_ref_ind == 0 else ref_counts
                                counts[nv_ind, run_ind, step_ind, rep_ind] = counts_val
                    break
                except Exception as exc:
                    print(exc)
                    camera.arm()
                    attempt_ind += 1
                    if attempt_ind == num_attempts:
                        raise RuntimeError("Maxed out number of attempts")
            if attempt_ind > 0:
                print(f"{attempt_ind} crashes occurred")

        camera.disarm()
        sig_gen.uwave_off()
        optimize.optimize_pixel_with_cxn(cxn, repr_nv_sig)


if __name__ == "__main__":
    pass
