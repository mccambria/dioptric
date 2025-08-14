# -*- coding: utf-8 -*-
"""
Base routine for widefield experiments with many spatially resolved NV centers.

Created on December 6th, 2023

@author: mccambria
"""

import logging
import time
import traceback
from math import isclose
from random import shuffle

import numpy as np

from majorroutines import targeting
from utils import common, widefield
from utils import positioning as pos
from utils import tool_belt as tb
from utils.constants import ChargeStateEstimationMode, CoordsKey


def start_tagger_stream(apd_indices, apd_gate=True, clock=True):
    tagger = tb.get_server_counter()
    # LabRAD requires positional args; cast to plain types
    apd_indices = list(int(i) for i in apd_indices)
    tagger.start_tag_stream(apd_indices, bool(apd_gate), bool(clock))


def dtype_clip(arr, dtype):
    min_val = np.finfo(dtype).min
    max_val = np.finfo(dtype).max
    arr = np.where(arr >= min_val, arr, min_val)
    arr = np.where(arr <= max_val, arr, max_val)
    return arr


def _as_pos_int(name, val):
    # Accepts Python ints, numpy ints, floats that are whole numbers.
    try:
        iv = int(val)
    except Exception:
        raise TypeError(f"{name} must be an integer, got {type(val).__name__}: {val!r}")
    if iv < 0:
        raise ValueError(f"{name} must be >= 0, got {iv}")
    # If original was float, make sure it was a whole number (e.g. 2.0 is ok, 2.3 is not)
    if isinstance(val, float) and not isclose(val, iv, rel_tol=0.0, abs_tol=1e-9):
        raise TypeError(f"{name} must be an integer, got non-integer float: {val!r}")
    return iv


def main(
    nv_sig,
    num_steps,
    num_reps,
    num_runs,
    run_fn=None,
    step_fn=None,
    uwave_ind_list=[0],
    uwave_freq_list=None,
    num_exps=2,
    apd_indices=[0],
    load_iq=False,
    stream_load_in_run_fn=True,
    charge_prep_fn=None,
) -> dict:
    # ---------- sanitize/validate integers ----------
    num_steps = _as_pos_int("num_steps", num_steps)
    num_reps = _as_pos_int("num_reps", num_reps)
    num_runs = _as_pos_int("num_runs", num_runs)
    num_exps = _as_pos_int("num_exps", num_exps)

    # (Optional) also sanitize list-y ints:
    uwave_ind_list = [
        _as_pos_int("uwave_ind", i)
        for i in (
            uwave_ind_list
            if isinstance(uwave_ind_list, (list, tuple))
            else [uwave_ind_list]
        )
    ]
    apd_indices = [
        _as_pos_int("apd_index", i)
        for i in (
            apd_indices if isinstance(apd_indices, (list, tuple)) else [apd_indices]
        )
    ]

    # ---------- NV positioning ----------
    tb.reset_cfm()
    pos.set_xyz_on_nv(nv_sig, CoordsKey.SAMPLE)

    pulsegen = tb.get_server_pulse_streamer()
    tagger = tb.get_server_counter()

    # ---------- Microwave setup ----------
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
                    if charge_prep_fn:
                        charge_prep_fn(nv_sig)

                    for uwave_ind in uwave_ind_list:
                        tb.get_server_sig_gen(uwave_ind).uwave_on()

                    shuffle(step_ind_list)
                    if run_fn:
                        run_fn(step_ind_list)
                    step_ind_master_list[run_ind] = step_ind_list.copy()

                    # tagger.start_tag_stream(
                    #     apd_indices=apd_indices, apd_gate=True, clock=True
                    # )
                    tagger.start_tag_stream(apd_indices, True, True)

                    for step_ind in step_ind_list:
                        if step_fn:
                            step_fn(step_ind)

                        tagger.clear_buffer()
                        pulsegen.stream_start(
                            num_reps
                        )  # play the preloaded sequence 'num_reps' times

                        new_counts = tagger.read_counter_complete(1)[
                            0
                        ]  # shape: (num_apds, num_gates_total)
                        new_counts = new_counts.sum(
                            axis=0
                        )  # sum APDs -> (num_gates_total,)

                        # sanity check the number of gates
                        expected = num_exps * num_reps
                        if new_counts.size != expected:
                            raise RuntimeError(
                                f"Got {new_counts.size} gated counts; expected {expected} "
                                f"(num_exps={num_exps} × num_reps={num_reps}). "
                                f"Make sure your sequence has exactly {num_exps} APD gates per repetition."
                            )

                        # de-interleave by experiment
                        # layout: [exp0_rep0, exp1_rep0, exp0_rep1, exp1_rep1, ...]
                        for exp_ind in range(num_exps):
                            counts[exp_ind, run_ind, step_ind, :] = new_counts[
                                exp_ind::num_exps
                            ]

                    for uwave_ind in uwave_ind_list:
                        tb.get_server_sig_gen(uwave_ind).uwave_off()

                    tagger.stop_tag_stream()
                    targeting.compensate_for_drift(nv_sig, no_crash=True)
                    crash_counter[run_ind] = attempt
                    break

                except Exception:
                    # pulsegen.force_final()
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
