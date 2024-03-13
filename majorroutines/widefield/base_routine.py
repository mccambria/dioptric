# -*- coding: utf-8 -*-
"""
Optimize SCC parameters

Created on December 6th, 2023

@author: mccambria
"""

import time
from random import shuffle

import numpy as np

from majorroutines.widefield import optimize
from utils import positioning as pos
from utils import tool_belt as tb
from utils import widefield


def main(
    nv_list,
    num_steps,
    num_reps,
    num_runs,
    step_fn=None,
    uwave_ind=0,
    uwave_freq=None,
    num_exps_per_rep=1,
    load_iq=False,
    save_images=True,
):
    ### Some initial setup

    tb.reset_cfm()

    repr_nv_sig = widefield.get_repr_nv_sig(nv_list)
    pos.set_xyz_on_nv(repr_nv_sig)
    num_nvs = len(nv_list)

    pulse_gen = tb.get_server_pulse_gen()
    camera = tb.get_server_camera()

    # Sig gen setup
    if isinstance(uwave_ind, int):
        uwave_ind_list = [uwave_ind]
    else:
        uwave_ind_list = uwave_ind
    for ind in uwave_ind_list:
        uwave_dict = tb.get_uwave_dict(ind)
        uwave_power = uwave_dict["uwave_power"]
        if uwave_freq is None:
            freq = uwave_dict["frequency"]
        elif isinstance(uwave_ind, int):
            freq = uwave_freq
        else:
            freq = uwave_freq[ind]
        sig_gen = tb.get_server_sig_gen(ind=ind)
        if load_iq:  # MCC
            uwave_power += 0.4
        sig_gen.set_amp(uwave_power)
        sig_gen.set_freq(freq)

    ### Data tracking

    counts = np.empty((num_exps_per_rep, num_nvs, num_runs, num_steps, num_reps))
    ref_counts = np.empty((num_nvs, num_runs, num_reps))
    if save_images:
        shape = widefield.get_img_array_shape()
        img_arrays = np.empty((num_exps_per_rep, num_runs, num_steps, *shape))
    step_ind_master_list = [[] for ind in range(num_runs)]
    step_ind_list = list(range(0, num_steps))

    ### Collect the data

    # Runs loops
    for run_ind in range(num_runs):
        print(f"Run index: {run_ind}")
        shuffle(step_ind_list)

        pixel_coords_list = [widefield.get_nv_pixel_coords(nv) for nv in nv_list]

        for ind in uwave_ind_list:
            sig_gen = tb.get_server_sig_gen(ind=ind)
            sig_gen.uwave_on()
            if load_iq:
                sig_gen.load_iq()

        camera.arm()

        # Steps loops
        for step_ind in step_ind_list:
            step_ind_master_list[run_ind].append(step_ind)

            if step_fn is not None:
                step_fn(step_ind)

            if save_images:
                img_array_list = [[] for exp_ind in range(num_exps_per_rep)]

            # Reps loops
            def rep_fn(rep_ind):
                for exp_ind in range(num_exps_per_rep):
                    img_str = camera.read()
                    img_array = widefield.img_str_to_array(img_str)
                    if save_images:
                        img_array_list[exp_ind].append(img_array)
                    img_array_photons = widefield.adus_to_photons(img_array)

                    def get_counts(pixel_coords):
                        return widefield.integrate_counts(
                            img_array_photons, pixel_coords
                        )

                    counts_list = [get_counts(el) for el in pixel_coords_list]
                    counts[exp_ind, :, run_ind, step_ind, rep_ind] = counts_list

            widefield.rep_loop(num_reps, rep_fn)

            if save_images:
                for exp_ind in range(num_exps_per_rep):
                    img_arrays[exp_ind, run_ind, step_ind, :, :] = np.mean(
                        img_array_list[exp_ind], axis=0
                    )

        ### Move on to the next run

        # Get a reference
        seq_args = widefield.get_base_scc_seq_args(nv_list)
        seq_args_string = tb.encode_seq_args(seq_args)
        pulse_gen.stream_load("scc_reference.py", seq_args_string, num_reps)

        def rep_fn(rep_ind):
            img_str = camera.read()
            img_array = widefield.img_str_to_array(img_str)
            img_array_photons = widefield.adus_to_photons(img_array)

            def get_counts(pixel_coords):
                return widefield.integrate_counts(img_array_photons, pixel_coords)

            counts_list = [get_counts(el) for el in pixel_coords_list]
            ref_counts[:, run_ind, rep_ind] = counts_list

        widefield.rep_loop(num_reps, rep_fn)

        # Turn stuff off
        camera.disarm()
        for ind in uwave_ind_list:
            sig_gen = tb.get_server_sig_gen(ind=ind)
            sig_gen.uwave_off()

        optimize.optimize_pixel(repr_nv_sig)

    ### Return

    if num_exps_per_rep == 1:
        counts = counts[0]
        if save_images:
            img_arrays = img_arrays[0]

    raw_data = {
        "nv_list": nv_list,
        "num_reps": num_reps,
        "num_steps": num_steps,
        "num_runs": num_runs,
        "uwave_ind": uwave_ind,
        "step_ind_master_list": step_ind_master_list,
        "counts-units": "photons",
        "counts": counts,
        "ref_counts": ref_counts,
    }
    if save_images:
        raw_data |= {
            "img_arrays-units": "ADUs",
            "img_arrays": img_arrays,
        }
    return counts, ref_counts, raw_data


if __name__ == "__main__":
    pass
