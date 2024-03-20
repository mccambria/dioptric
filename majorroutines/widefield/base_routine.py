# -*- coding: utf-8 -*-
"""
Base routine for widefield experiments with many spatially resolved NV centers.

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
    num_exps_per_rep=2,
    load_iq=False,
    save_images=False,
):
    """Base routine for widefield experiments with many spatially resolved NV centers.

    The routine is broken down into a sequence of identical "runs". In between runs
    we update the sample drift so that all our targeting remains accurate. Each run
    consists of a sequence of "steps" where each step represents a specific value of
    the dependent variable (e.g. a specific frequency in an ESR experiment). In turn
    each step consists of some number of identical "repetitions", where in each
    repetition we may perform one or several distinct experiments (e.g. a signal and a
    reference experiment).

    Parameters
    ----------
    nv_list : list(nv_sig)
        List of NV sig objects to interrogate
    num_steps : int
        Number of steps
    num_reps : int
        Number of repetitions, or "reps"
    num_runs : int
        Number of runs
    step_fn : function, optional
        Function to run when moving to a new step. Most likely we want to load a new
        sequence onto the pulse streamer here, but we may also want to just update a
        microwave frequency or something similar. If None, do nothing. By default None
    uwave_ind : int, optional
        Index of microwave signal chain to use, by default 0
    uwave_freq : float, optional
        Microwave frequency to set in GHz, by default retrieved from config
    num_exps_per_rep : int, optional
        Number of experiments to perform in a single rep, by default 2
    load_iq : bool, optional
        Whether to load IQ modulation for the microwave signal chain, by default False
    save_images : bool, optional
        Whether to return the images from the experiments to the caller routine.
        To save space, reps are averaged over. Results in large files, roughly 1 gb
        per hour of runtime after compression. By default False

    Returns
    -------
    ndarray
        Array of photon counts. Indexing is experiment, nv, run, step, rep
    dict
        Raw data object to save. Populated with basic data, such as the counts array,
        nv_list, and images if save_images is True. Add whatever routine-specific
        data we should also save onto this dict before writing it to the cloud
    """
    ### Some initial setup

    tb.reset_cfm()

    repr_nv_sig = widefield.get_repr_nv_sig(nv_list)
    pos.set_xyz_on_nv(repr_nv_sig)
    num_nvs = len(nv_list)

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

        # Turn stuff off
        camera.disarm()
        for ind in uwave_ind_list:
            sig_gen = tb.get_server_sig_gen(ind=ind)
            sig_gen.uwave_off()

        # Update coordinates
        optimize.optimize_pixel(repr_nv_sig)

    ### Return

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
    if save_images:
        raw_data |= {
            "img_arrays-units": "ADUs",
            "img_arrays": img_arrays,
        }
    return counts, raw_data


if __name__ == "__main__":
    pass
