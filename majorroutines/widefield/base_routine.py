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

try:
    camera = tb.get_server_camera()
    pulse_gen = tb.get_server_pulse_gen()
except Exception:
    pass


def charge_prep_loop(pixel_coords_list, threshold_list, initial_counts_list=None):
    num_nvs = len(pixel_coords_list)

    counts_list = initial_counts_list
    num_attempts = 10
    attempt_ind = 0
    while True:
        if counts_list is not None:
            charge_pol_target_list = [
                counts_list[ind] < threshold_list[ind] for ind in range(num_nvs)
            ]
        else:
            charge_pol_target_list = [True for ind in range(num_nvs)]

        out_of_attempts = attempt_ind == num_attempts
        charge_pol_complete = True not in charge_pol_target_list or out_of_attempts
        pulse_gen.insert_input_stream(
            "_cache_charge_pol_incomplete", not charge_pol_complete
        )
        if charge_pol_complete:
            break

        for val in charge_pol_target_list:
            pulse_gen.insert_input_stream("_cache_charge_pol_target", val)

        attempt_ind += 1
        _, counts_list = read_image_and_get_counts(pixel_coords_list)

    # pulse_gen.resume()
    return attempt_ind


def read_image_and_get_counts(pixel_coords_list):
    img_str = camera.read()
    img_array = widefield.img_str_to_array(img_str)
    img_array_photons = widefield.adus_to_photons(img_array)

    def get_counts(pixel_coords):
        return widefield.integrate_counts(img_array_photons, pixel_coords)

    counts_list = [get_counts(el) for el in pixel_coords_list]

    return img_array, counts_list


def main(
    nv_list,
    num_steps,
    num_reps,
    num_runs,
    run_fn=None,
    step_fn=None,
    uwave_ind=0,
    uwave_freq=None,
    num_exps_per_rep=2,
    load_iq=False,
    save_images=False,
    stream_load_in_run_fn=True,
) -> tuple[np.ndarray, dict]:
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
    num_runs : int
        Number of runs
    num_steps : int
        Number of steps
    num_reps : int
        Number of repetitions, or "reps"
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

    do_charge_prep_loop = True

    repr_nv_sig = widefield.get_repr_nv_sig(nv_list)
    pos.set_xyz_on_nv(repr_nv_sig)
    num_nvs = len(nv_list)

    threshold_list = [nv.threshold for nv in nv_list]

    # Sig gen setup - all but turning on the output
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
    # MCC
    mean_vals = np.empty((num_exps_per_rep, num_runs, num_steps, num_reps))
    median_vals = np.empty((num_exps_per_rep, num_runs, num_steps, num_reps))
    if save_images:
        shape = widefield.get_img_array_shape()
        img_arrays = np.empty((num_exps_per_rep, num_runs, num_steps, *shape))
    step_ind_master_list = [None for ind in range(num_runs)]
    step_ind_list = list(range(0, num_steps))

    ### Collect the data

    # Runs loop
    for run_ind in range(num_runs):
        num_attempts = 5
        attempt_ind = 0

        while True:
            try:
                print(f"\nRun index: {run_ind}")

                pixel_coords_list = [
                    widefield.get_nv_pixel_coords(nv) for nv in nv_list
                ]
                counts_list = None
                charge_prep_readouts_list = []

                for ind in uwave_ind_list:
                    sig_gen = tb.get_server_sig_gen(ind=ind)
                    sig_gen.uwave_on()
                    if load_iq:
                        sig_gen.load_iq()

                shuffle(step_ind_list)
                if run_fn is not None:
                    run_fn(step_ind_list)

                camera.arm()
                if stream_load_in_run_fn:
                    pulse_gen.stream_start()

                # Steps loop
                for step_ind in step_ind_list:
                    if step_fn is not None:
                        step_fn(step_ind)

                    # If the sequence wasn't loaded in the run_fn, it must be loaded in
                    # the step_fn - this will be slower due to frequent compiling
                    if not stream_load_in_run_fn:
                        pulse_gen.stream_start()

                    if save_images:
                        img_array_list = [[] for exp_ind in range(num_exps_per_rep)]

                    # Reps loop
                    for rep_ind in range(num_reps):
                        for exp_ind in range(num_exps_per_rep):
                            if do_charge_prep_loop:
                                charge_prep_readouts = charge_prep_loop(
                                    pixel_coords_list,
                                    threshold_list,
                                    initial_counts_list=counts_list,
                                )
                                charge_prep_readouts_list.append(charge_prep_readouts)
                            img_array, counts_list = read_image_and_get_counts(
                                pixel_coords_list
                            )
                            counts[exp_ind, :, run_ind, step_ind, rep_ind] = counts_list
                            mean_vals[exp_ind, run_ind, step_ind, rep_ind] = np.mean(
                                img_array
                            )
                            median_vals[
                                exp_ind, run_ind, step_ind, rep_ind
                            ] = np.median(img_array)
                            if save_images:
                                img_array_list[exp_ind].append(img_array)

                    if save_images:
                        for exp_ind in range(num_exps_per_rep):
                            img_arrays[exp_ind, run_ind, step_ind, :, :] = np.mean(
                                img_array_list[exp_ind], axis=0
                            )

                    pulse_gen.resume()

                ### Move on to the next run

                if len(charge_prep_readouts_list) > 0:
                    print(np.mean(charge_prep_readouts_list))

                # Turn stuff off
                pulse_gen.halt()
                camera.disarm()
                for ind in uwave_ind_list:
                    sig_gen = tb.get_server_sig_gen(ind=ind)
                    sig_gen.uwave_off()

                # Record step order
                step_ind_master_list[run_ind] = step_ind_list.copy()

                # Update coordinates
                optimize.optimize_pixel_and_z(repr_nv_sig)

                break

            except Exception as exc:
                pulse_gen.halt()
                # Camera disarmed automatically

                nuvu_237 = "NuvuException: 237"
                nuvu_214 = "NuvuException: 214"
                if "NuvuException: 237" in str(exc):
                    print(nuvu_237)
                elif "NuvuException: 214" in str(exc):
                    print(nuvu_214)
                else:
                    raise exc

                attempt_ind += 1
                if attempt_ind == num_attempts:
                    raise RuntimeError("Maxed out number of attempts")

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
        "mean_vals": mean_vals,
        "median_vals": median_vals,
    }
    if save_images:
        raw_data |= {
            "img_arrays-units": "ADUs",
            "img_arrays": img_arrays,
        }
    return counts, raw_data


if __name__ == "__main__":
    pass
