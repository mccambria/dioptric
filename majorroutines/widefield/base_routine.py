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
from utils import tool_belt as tb, widefield, positioning as pos


def main(
    nv_list,
    num_steps,
    num_reps,
    num_runs,
    step_fn=None,
    uwave_ind=0,
    num_exps_per_rep=1,
    load_iq=False,
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
        uwave_dict = tb.get_uwave_dict(ind)
        uwave_power = uwave_dict["uwave_power"]
        freq = uwave_dict["frequency"]
        sig_gen = tb.get_server_sig_gen(ind=ind)
        sig_gen.set_amp(uwave_power)
        sig_gen.set_freq(freq)
        if load_iq:
            sig_gen.load_iq()

    ### Data tracking

    counts = np.empty((num_exps_per_rep, num_nvs, num_runs, num_steps, num_reps))
    step_ind_master_list = [[] for ind in range(num_runs)]
    step_ind_list = list(range(0, num_steps))

    ### Collect the data

    for run_ind in range(num_runs):
        shuffle(step_ind_list)

        pixel_coords_list = [widefield.get_nv_pixel_coords(nv) for nv in nv_list]

        for ind in uwave_ind_list:
            sig_gen = tb.get_server_sig_gen(ind=ind)
            sig_gen.uwave_on()

        camera.arm()

        for step_ind in step_ind_list:
            step_ind_master_list[run_ind].append(step_ind)

            if step_fn is not None:
                step_fn(step_ind)

            # Try 5 times then give up
            num_attempts = 5
            attempt_ind = 0
            start_rep = 0  # Use this variable to pick up where we left off
            while True:
                try:
                    pulse_gen.stream_start()
                    for rep_ind in range(start_rep, num_reps):
                        for exp_ind in range(num_exps_per_rep):
                            img_str = camera.read()
                            img_array = widefield.img_str_to_array(img_str)
                            img_array_photons = widefield.adus_to_photons(img_array)

                            def get_counts(pixel_coords):
                                return widefield.integrate_counts(
                                    img_array_photons, pixel_coords
                                )

                            counts_list = [get_counts(el) for el in pixel_coords_list]
                            counts[exp_ind, :, run_ind, step_ind, rep_ind] = counts_list
                    break
                except Exception as exc:
                    pulse_gen.halt()

                    # Check if we can just continue
                    nuvu_237 = "NuvuException: 237"
                    if "NuvuException: 237" in str(exc):
                        print(nuvu_237)
                    else:
                        raise exc
                    attempt_ind += 1
                    if attempt_ind == num_attempts:
                        raise RuntimeError("Maxed out number of attempts")

                    start_rep = rep_ind
                    camera.arm()
            if attempt_ind > 0:
                print(f"{attempt_ind} crashes occurred")

        pulse_gen.halt()
        camera.disarm()
        for ind in uwave_ind_list:
            sig_gen = tb.get_server_sig_gen(ind=ind)
            sig_gen.uwave_off()
        optimize.optimize_pixel(repr_nv_sig)

    ### Return

    if num_exps_per_rep == 1:
        counts = counts[0]

    raw_data = {
        "nv_list": nv_list,
        "num_reps": num_reps,
        "num_steps": num_steps,
        "num_runs": num_runs,
        "uwave_ind": uwave_ind,
        "step_ind_master_list": step_ind_master_list,
        "counts-units": "photons",
        "counts": counts,
    }
    return counts, raw_data


if __name__ == "__main__":
    pass
