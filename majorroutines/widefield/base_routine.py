# -*- coding: utf-8 -*-
"""
Base routine for widefield experiments with many spatially resolved NV centers.

Created on December 6th, 2023

@author: mccambria
"""

import time
import traceback
from random import shuffle

import numpy as np

from majorroutines.widefield import optimize
from utils import common, widefield
from utils import positioning as pos
from utils import tool_belt as tb
from utils.constants import ChargeStateEstimationMode


def charge_prep_no_verification(rep_ind, nv_list, initial_states_list=None):
    charge_prep_base(nv_list, initial_states_list)


# def charge_prep_first_rep_only(rep_ind, nv_list, initial_states_list=None):
#     if rep_ind == 0:
#         charge_prep_loop(nv_list, initial_states_list)


def charge_prep_loop(rep_ind, nv_list, initial_states_list=None):
    charge_prep_base(
        nv_list,
        initial_states_list,
        targeted_polarization=True,
        verify_charge_states=True,
    )


def charge_prep_base(
    nv_list,
    initial_states_list=None,
    targeted_polarization=True,
    verify_charge_states=False,
):
    # Initial setup
    pulse_gen = tb.get_server_pulse_gen()
    num_nvs = len(nv_list)
    states_list = initial_states_list

    # Inner function for determining which NVs to target
    def assemble_charge_pol_target_list(states_list):
        if states_list is not None:
            charge_pol_target_list = [el is None or el == 0 for el in states_list]
        else:
            charge_pol_target_list = [True for ind in range(num_nvs)]
        # MCC
        # charge_pol_target_list = [False for ind in range(num_nvs)]
        # charge_pol_target_list[2] = True
        return charge_pol_target_list

    if verify_charge_states:
        max_num_attempts = 10
        # max_num_attempts = 1
        out_of_attempts = False
        attempt_ind = 0

        # Loop until we have a reason to stop
        while True:
            out_of_attempts = attempt_ind == max_num_attempts
            charge_pol_target_list = assemble_charge_pol_target_list(states_list)

            # Reasons to stop
            no_more_targets = True not in charge_pol_target_list
            charge_pol_complete = no_more_targets or out_of_attempts
            pulse_gen.insert_input_stream(
                "_cache_charge_pol_incomplete", not charge_pol_complete
            )
            if charge_pol_complete:
                break

            pulse_gen.insert_input_stream("_cache_target_list", charge_pol_target_list)

            _, _, states_list = read_and_process_image(nv_list)

            # Move on to next attempt
            attempt_ind += 1
    elif targeted_polarization:
        charge_pol_target_list = assemble_charge_pol_target_list(states_list)
        pulse_gen.insert_input_stream("_cache_target_list", charge_pol_target_list)
    else:
        pass


# def read_image_and_get_counts(nv_list):
#     img_str = camera.read()
#     img_array_adus, baseline = widefield.img_str_to_array(img_str)
#     # baseline = 300
#     img_array = widefield.adus_to_photons(img_array_adus, baseline=baseline)

#     def get_counts(pixel_coords):
#         return widefield.integrate_counts(img_array, pixel_coords)

#     counts_list = [get_counts(el) for el in nv_list]

#     return img_array, counts_list


def read_and_process_image(nv_list):
    camera = tb.get_server_camera()
    img_str = camera.read()
    img_array_adus, baseline = widefield.img_str_to_array(img_str)
    # baseline = 300
    img_array = widefield.adus_to_photons(img_array_adus, baseline=baseline)

    def get_counts(nv_sig):
        pixel_coords = widefield.get_nv_pixel_coords(nv_sig)
        return widefield.integrate_counts(img_array, pixel_coords)

    counts_list = [get_counts(nv) for nv in nv_list]

    config = common.get_config_dict()
    charge_state_estimation_mode = config["charge_state_estimation_mode"]
    if charge_state_estimation_mode == ChargeStateEstimationMode.THRESHOLDING:
        num_nvs = len(nv_list)
        states_list = []
        for nv_ind in range(num_nvs):
            states_list.append(
                widefield.threshold(nv_list[nv_ind], counts_list[nv_ind])
            )
    elif charge_state_estimation_mode == ChargeStateEstimationMode.MLE:
        # start = time.time()
        states_list = widefield.charge_state_mle(nv_list, img_array)
        # stop = time.time()
        # print(stop - start)

    return img_array, counts_list, states_list


def main(
    nv_list,
    num_steps,
    num_reps,
    num_runs,
    run_fn=None,
    step_fn=None,
    uwave_ind_list=[0, 1],
    uwave_freq=None,
    num_exps_per_rep=2,
    load_iq=False,
    save_all_images=False,
    save_mean_images=False,
    stream_load_in_run_fn=True,
    charge_prep_fn=None,
) -> dict:
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

    repr_nv_sig = widefield.get_repr_nv_sig(nv_list)
    pos.set_xyz_on_nv(repr_nv_sig)
    num_nvs = len(nv_list)
    num_shots = num_reps * num_runs
    camera = tb.get_server_camera()
    pulse_gen = tb.get_server_pulse_gen()

    # Sig gen setup - all but turning on the output
    if isinstance(uwave_ind_list, int):
        uwave_ind_list = [uwave_ind_list]
    for ind in uwave_ind_list:
        uwave_dict = tb.get_uwave_dict(ind)
        uwave_power = uwave_dict["uwave_power"]
        if uwave_freq is None:
            freq = uwave_dict["frequency"]
        elif isinstance(uwave_freq, float):
            freq = uwave_freq
        else:
            freq = uwave_freq[ind]
        sig_gen = tb.get_server_sig_gen(ind=ind)
        # if load_iq:  # MCC
        #     uwave_power += 0.4
        sig_gen.set_amp(uwave_power)
        sig_gen.set_freq(freq)

    ### Data tracking

    counts = np.empty((num_exps_per_rep, num_nvs, num_runs, num_steps, num_reps))
    states = np.empty((num_exps_per_rep, num_nvs, num_runs, num_steps, num_reps))
    if save_all_images:
        shape = widefield.get_img_array_shape()
        img_arrays = np.empty((num_exps_per_rep, num_runs, num_steps, num_reps, *shape))
    if save_mean_images:
        shape = widefield.get_img_array_shape()
        mean_img_arrays = np.zeros((num_exps_per_rep, num_steps, *shape))
    step_ind_master_list = [None for ind in range(num_runs)]
    step_ind_list = list(range(0, num_steps))

    ### Collect the data

    try:
        # Runs loop
        for run_ind in range(num_runs):
            num_attempts = 5
            attempt_ind = 0

            while True:
                try:
                    print(f"\nRun index: {run_ind}")

                    states_list = None

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

                        # Reps loop
                        # start = time.time()
                        for rep_ind in range(num_reps):
                            for exp_ind in range(num_exps_per_rep):
                                if charge_prep_fn is not None:
                                    charge_prep_fn(
                                        rep_ind,
                                        nv_list,
                                        initial_states_list=states_list,
                                    )
                                (
                                    img_array,
                                    counts_list,
                                    states_list,
                                ) = read_and_process_image(nv_list)
                                counts[exp_ind, :, run_ind, step_ind, rep_ind] = (
                                    counts_list
                                )
                                states[exp_ind, :, run_ind, step_ind, rep_ind] = (
                                    states_list
                                )

                                if save_all_images:
                                    img_arrays[
                                        exp_ind, run_ind, step_ind, rep_ind, :, :
                                    ] = img_array
                                if save_mean_images:
                                    mean_img_arrays[exp_ind, step_ind, :, :] += (
                                        img_array
                                    )
                        # stop = time.time()
                        # print((stop - start) / (num_reps * num_exps_per_rep))
                        # print()

                        pulse_gen.resume()

                    ### Move on to the next run

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
                        print(traceback.format_exc())
                        raise exc

                    attempt_ind += 1
                    if attempt_ind == num_attempts:
                        raise RuntimeError("Maxed out number of attempts")

    except Exception:
        pass

    ### Return

    raw_data = {
        "nv_list": nv_list,
        "num_steps": num_steps,
        "num_reps": num_reps,
        "num_runs": num_runs,
        "uwave_ind": uwave_ind_list,
        "uwave_freq": uwave_freq,
        "num_exps_per_rep": num_exps_per_rep,
        "load_iq": load_iq,
        "step_ind_master_list": step_ind_master_list,
        "counts-units": "photons",
        "counts": counts,
        "states": states,
    }
    if save_all_images:
        raw_data |= {
            "img_arrays-units": "photons",
            "img_arrays": img_arrays,
        }
    if save_mean_images:
        mean_img_arrays /= num_shots
        raw_data |= {
            "mean_img_arrays-units": "photons",
            "mean_img_arrays": mean_img_arrays,
        }
    return raw_data


if __name__ == "__main__":
    pass
