# -*- coding: utf-8 -*-
"""
Base routine for widefield experiments with many spatially resolved NV centers.

Created on December 6th, 2023

@author: mccambria
"""

import logging
import time
import traceback
from random import shuffle

import numpy as np

from majorroutines import targeting
from utils import common, widefield
from utils import positioning as pos
from utils import tool_belt as tb
from utils.constants import ChargeStateEstimationMode, CoordsKey


def charge_prep_no_verification_skip_first_rep(
    rep_ind, nv_list, initial_states_list=None
):
    if rep_ind > 0:
        charge_prep_base(nv_list, initial_states_list)


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
        # charge_pol_target_list = [not (el) for el in charge_pol_target_list]
        # charge_pol_target_list = [False for ind in range(num_nvs)]
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
        # print(charge_pol_target_list)
        pulse_gen.insert_input_stream("_cache_target_list", charge_pol_target_list)
    else:
        pass


def read_and_process_image(nv_list):
    camera = tb.get_server_camera()
    img_str = camera.read()
    img_array_adus, baseline = widefield.img_str_to_array(img_str)
    # baseline = 300
    img_array = widefield.adus_to_photons(img_array_adus, baseline=baseline)

    def get_counts(nv_sig):
        pixel_coords = pos.get_nv_coords(nv_sig, CoordsKey.PIXEL)
        return widefield.integrate_counts(img_array, pixel_coords)

    counts_list = [get_counts(nv) for nv in nv_list]

    config = common.get_config_dict()
    charge_state_estimation_mode = config["charge_state_estimation_mode"]
    if charge_state_estimation_mode == ChargeStateEstimationMode.THRESHOLDING:
        num_nvs = len(nv_list)
        states_list = []
        for nv_ind in range(num_nvs):
            states_list.append(
                tb.threshold(counts_list[nv_ind], nv_list[nv_ind].threshold)
            )
    elif charge_state_estimation_mode == ChargeStateEstimationMode.MLE:
        # start = time.time()
        states_list = widefield.charge_state_mle(nv_list, img_array)
        # stop = time.time()
        # print(stop - start)

    counts_list = np.array(counts_list)
    states_list = np.array(states_list)

    return img_array, counts_list, states_list


def dtype_clip(arr, dtype):
    min_val = np.finfo(dtype).min
    max_val = np.finfo(dtype).max
    arr = np.where(arr >= min_val, arr, min_val)
    arr = np.where(arr <= max_val, arr, max_val)
    return arr


def main(
    nv_sig,
    num_steps,
    num_reps,
    num_runs,
    run_fn=None,  # called once per run, passes shuffled step indices; must do stream_load
    step_fn=None,  # called per step (optional)
    uwave_ind_list=[0],
    uwave_freq_list=None,
    num_exps=2,  # signal + reference
    apd_indices=[0],
    load_iq=False,
    stream_load_in_run_fn=True,  # kept for API compatibility
    charge_prep_fn=None,  # optional pre-run prep
) -> dict:
    """
    Unified confocal base routine using Pulse Streamer + Swabian Time Tagger.

    Handles:
    - NV positioning (SAMPLE space)
    - Microwave generator setup (power/frequency) for one or more sources
    - Run structure: num_runs × (shuffle steps → load seq → loop steps → tagger counts)
    - 2 experiments per repetition by default (signal, reference)

    Returns a dict with counts and metadata:
        counts shape: (num_exps, num_runs, num_steps, num_reps)
    """
    # ---------- NV positioning ----------
    tb.reset_cfm()
    pos.set_xyz_on_nv(nv_sig)

    pulsegen = tb.get_server_pulse_streamer()  # Pulse Streamer server
    tagger = tb.get_server_time_tagger()  # Swabian Time Tagger

    # ---------- Microwave setup ----------
    if isinstance(uwave_ind_list, int):
        uwave_ind_list = [uwave_ind_list]

    for uwave_ind in uwave_ind_list:
        uwave_dict = tb.get_virtual_sig_gen_dict(uwave_ind)
        uwave_power = uwave_dict["uwave_power"]
        freq = (
            uwave_freq_list[uwave_ind] if uwave_freq_list else uwave_dict["frequency"]
        )
        sig_gen = tb.get_server_sig_gen(uwave_ind)

        if load_iq:
            sig_gen.load_iq()
        sig_gen.set_amp(uwave_power)
        sig_gen.set_freq(freq)
        print(f"MW[{uwave_ind}]  freq: {freq} GHz,  power: {uwave_power} dBm")

    # ---------- Containers ----------
    counts = np.zeros((num_exps, num_runs, num_steps, num_reps), dtype=np.int32)
    step_ind_master_list = [None] * num_runs
    crash_counter = [None] * num_runs
    step_ind_list = list(range(num_steps))

    try:
        for run_ind in range(num_runs):
            num_attempts = 15
            attempt = 0

            while True:
                try:
                    print(f"\n[Run {run_ind + 1}/{num_runs}]")

                    # (Optional) custom preparation
                    if charge_prep_fn:
                        charge_prep_fn(nv_sig)

                    # MW ON for all sources
                    for uwave_ind in uwave_ind_list:
                        tb.get_server_sig_gen(uwave_ind).uwave_on()

                    # Randomize step order; let run_fn do the stream_load for this run
                    shuffle(step_ind_list)
                    if run_fn:
                        run_fn(step_ind_list)
                    step_ind_master_list[run_ind] = step_ind_list.copy()

                    # Tagger stream (gated by APD gate, using Pulse Streamer clock)
                    tagger.start_tag_stream(
                        apd_indices=apd_indices, apd_gate=True, clock=True
                    )

                    # Step loop
                    for step_ind in step_ind_list:
                        if step_fn:
                            step_fn(step_ind)

                        tagger.clear_buffer()
                        pulsegen.stream_start(
                            num_reps
                        )  # play sequence for this step 'num_reps' times

                        # Read & collapse counts -> (num_exps, num_reps)
                        new_counts = tagger.read_counter_complete()
                        # new_counts[0] → first (and only) acquisition block
                        new_counts = new_counts[0]
                        # Sum over APDs: becomes shape (num_gates_per_rep * num_reps,)
                        new_counts = new_counts.sum(axis=0)

                        # De-interleave into experiments (exp 0, exp 1, ...)
                        # Layout: [exp0_rep0, exp1_rep0, exp0_rep1, exp1_rep1, ...]
                        for exp_ind in range(num_exps):
                            counts[exp_ind, run_ind, step_ind, :] = new_counts[
                                exp_ind::num_exps
                            ]

                    # MW OFF
                    for uwave_ind in uwave_ind_list:
                        tb.get_server_sig_gen(uwave_ind).uwave_off()

                    tagger.stop_tag_stream()

                    # Drift compensation between runs
                    targeting.compensate_for_drift(nv_sig, no_crash=True)

                    crash_counter[run_ind] = attempt
                    break  # completed this run

                except Exception:
                    # Ensure outputs are forced to final state before retry
                    pulsegen.force_final()
                    attempt += 1
                    if attempt >= num_attempts:
                        raise RuntimeError("Too many failures during run")

    except Exception:
        print(traceback.format_exc())

    return {
        "nv_sig": nv_sig,
        "num_steps": num_steps,
        "num_reps": num_reps,
        "num_runs": num_runs,
        "num_exps_per_rep": num_exps,
        "load_iq": load_iq,
        "uwave_ind_list": uwave_ind_list,
        "uwave_freq_list": uwave_freq_list,
        "counts": counts,
        "counts-units": "photons",
        "step_ind_master_list": step_ind_master_list,
        "crash_counter": crash_counter,
    }


if __name__ == "__main__":
    pass

# def main(
#     nv_sig,
#     num_steps,
#     num_reps,
#     num_runs,
#     run_fn=None,
#     step_fn=None,
#     uwave_ind_list=[0],
#     uwave_freq_list=None,
#     num_exps=2,
#     apd_indices=[0],
#     load_iq=False,
#     stream_load_in_run_fn=True,
#     charge_prep_fn=None,
# ) -> dict:
#     """
#     Unified confocal base routine using Pulse Streamer + Swabian Time Tagger.
#     """

#     tb.reset_cfm()
#     pos.set_xyz_on_nv(nv_sig)

#     pulsegen = tb.get_server_pulse_gen()  # pulse_gen_SWAB_82
#     tagger = tb.get_server_time_tagger()  # tagger_SWAB_20

#     # === Microwave signal generator setup ===
#     if isinstance(uwave_ind_list, int):
#         uwave_ind_list = [uwave_ind_list]

#     for uwave_ind in uwave_ind_list:
#         uwave_dict = tb.get_virtual_sig_gen_dict(uwave_ind)
#         uwave_power = uwave_dict["uwave_power"]
#         freq = (
#             uwave_freq_list[uwave_ind] if uwave_freq_list else uwave_dict["frequency"]
#         )
#         sig_gen = tb.get_server_sig_gen(uwave_ind)

#         if load_iq:
#             sig_gen.load_iq()
#         sig_gen.set_amp(uwave_power)
#         sig_gen.set_freq(freq)
#         print(f"MW [{uwave_ind}] - freq: {freq} GHz, power: {uwave_power} dBm")

#     # === Data containers ===
#     counts = np.zeros((num_exps, num_runs, num_steps, num_reps), dtype=np.int32)
#     step_ind_master_list = [None for _ in range(num_runs)]
#     crash_counter = [None] * num_runs
#     step_ind_list = list(range(num_steps))

#     try:
#         for run_ind in range(num_runs):
#             num_attempts = 15
#             attempt = 0

#             while True:
#                 try:
#                     print(f"\n[Run {run_ind + 1}/{num_runs}]")

#                     # Turn MW on
#                     for uwave_ind in uwave_ind_list:
#                         tb.get_server_sig_gen(uwave_ind).uwave_on()

#                     # Shuffle and run_fn logic
#                     shuffle(step_ind_list)
#                     if run_fn:
#                         run_fn(step_ind_list)

#                     step_ind_master_list[run_ind] = step_ind_list.copy()

#                     # Load pulse sequence
#                     pulsegen.stream_start()

#                     # Start tagger stream
#                     tagger.start_tag_stream(
#                         apd_indices=apd_indices, apd_gate=True, clock=True
#                     )

#                     for step_ind in step_ind_list:
#                         if step_fn:
#                             step_fn(step_ind)

#                         tagger.clear_buffer()
#                         pulsegen.stream_start(num_reps)

#                         new_counts = tagger.read_counter_complete()
#                         new_counts = new_counts[0]  # Just one sample
#                         new_counts = new_counts.sum(axis=0)  # Sum over APDs
#                         for exp_ind in range(num_exps):
#                             counts[exp_ind, run_ind, step_ind, :] = new_counts[
#                                 exp_ind::num_exps
#                             ]

#                         # if stream_load_in_run_fn:
#                         #     tagger.clear_buffer()
#                         #     pulsegen.stream_start()
#                         # else:
#                         #     tagger.clear_buffer()
#                         #     pulsegen.stream_start()

#                     # Turn off MW
#                     for uwave_ind in uwave_ind_list:
#                         tb.get_server_sig_gen(uwave_ind).uwave_off()

#                     tagger.stop_tag_stream()
#                     targeting.compensate_for_drift(nv_sig, no_crash=True)
#                     crash_counter[run_ind] = attempt
#                     break

#                 except Exception:
#                     pulsegen.force_final()
#                     attempt += 1
#                     if attempt == num_attempts:
#                         raise RuntimeError("Too many failures during run")

#     except Exception:
#         print(traceback.format_exc())

#     return {
#         "nv_sig": nv_sig,
#         "num_steps": num_steps,
#         "num_reps": num_reps,
#         "num_runs": num_runs,
#         "num_exps_per_rep": num_exps,
#         "load_iq": load_iq,
#         "uwave_ind_list": uwave_ind_list,
#         "uwave_freq_list": uwave_freq_list,
#         "counts": counts,
#         "counts-units": "photons",
#         "step_ind_master_list": step_ind_master_list,
#         "crash_counter": crash_counter,
#     }


if __name__ == "__main__":
    pass
