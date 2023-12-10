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


def main(
    nv_list,
    uwave_list,
    uwave_ind,
    num_steps,
    num_reps,
    num_runs,
    step_fn=None,
    reference=False,
):
    ### Some initial setup

    tb.reset_cfm()

    repr_nv_sig = widefield.get_repr_nv_sig(nv_list)
    pos.set_xyz_on_nv(repr_nv_sig)
    num_nvs = len(nv_list)

    camera = tb.get_server_camera()
    pulse_gen = tb.get_server_pulse_gen()

    # Sig gen setup
    if type(uwave_ind) == int:
        uwave_ind_list = [uwave_ind]
    else:
        uwave_ind_list = uwave_ind
    for ind in uwave_ind_list:
        uwave_dict = uwave_list[ind]
        sig_gen = tb.get_server_sig_gen(ind=ind)
        uwave_power = uwave_dict["uwave_power"]
        freq = uwave_dict["frequency"]
        sig_gen.set_amp(uwave_power)
        sig_gen.set_freq(freq)

    ### Data tracking

    sig_counts = np.empty((num_nvs, num_runs, num_steps, num_reps))
    if reference:
        ref_counts = np.empty((num_nvs, num_runs, num_steps, num_reps))
        num_sig_ref = 2
    else:
        num_sig_ref = 1
    print(num_sig_ref)
    step_ind_master_list = [[] for ind in range(num_runs)]
    step_ind_list = list(range(0, num_steps))

    ### Collect the data

    for run_ind in range(num_runs):
        shuffle(step_ind_list)

        camera.arm()
        for ind in uwave_ind_list:
            sig_gen = tb.get_server_sig_gen(ind=ind)
            sig_gen.uwave_on()

        for step_ind in step_ind_list:
            pixel_coords_list = [widefield.get_nv_pixel_coords(nv) for nv in nv_list]
            step_ind_master_list[run_ind].append(step_ind)

            if step_fn is not None:
                step_fn(step_ind)

            # Try 5 times then give up
            num_attempts = 5
            attempt_ind = 0
            while True:
                try:
                    pulse_gen.stream_start()
                    for rep_ind in range(num_reps):
                        for sig_ref_ind in range(num_sig_ref):
                            img_str = camera.read()
                            img_array = widefield.img_str_to_array(img_str)
                            start = time.time()
                            img_array_photons = widefield.adus_to_photons(img_array)
                            stop = time.time()
                            print(f"loop time: {stop-start}")

                            def get_counts(pixel_coords):
                                widefield.integrate_counts(
                                    img_array_photons, pixel_coords
                                )

                            counts_list = [get_counts(el) for el in pixel_coords_list]
                            if sig_ref_ind == 0:
                                sig_counts[:, run_ind, step_ind, rep_ind] = counts_list
                            else:
                                ref_counts[:, run_ind, step_ind, rep_ind] = counts_list
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
        for ind in uwave_ind_list:
            sig_gen = tb.get_server_sig_gen(ind=ind)
            sig_gen.uwave_off()
        optimize.optimize_pixel(repr_nv_sig)

    ### Return

    raw_data = {
        "step_ind_master_list": step_ind_master_list,
        "nv_list": nv_list,
        "uwave_list": uwave_list,
        "uwave_ind": uwave_ind,
        "num_reps": num_reps,
        "num_steps": num_steps,
        "num_runs": num_runs,
        "counts-units": "photons",
    }

    if reference:
        raw_data |= {"sig_counts": sig_counts, "ref_counts": ref_counts}
        return sig_counts, ref_counts, raw_data
    else:
        counts = sig_counts
        raw_data |= {"counts": counts}
        return counts, raw_data


if __name__ == "__main__":
    pass
