# -*- coding: utf-8 -*-
import traceback
from math import isclose
from random import shuffle

import numpy as np

from majorroutines import targeting
from utils import positioning as pos
from utils import tool_belt as tb
from utils.constants import CoordsKey


def _as_pos_int(name, val):
    try:
        iv = int(val)
    except Exception:
        raise TypeError(f"{name} must be an integer, got {type(val).__name__}: {val!r}")
    if iv < 0:
        raise ValueError(f"{name} must be >= 0, got {iv}")
    if isinstance(val, float) and not isclose(val, iv, rel_tol=0.0, abs_tol=1e-9):
        raise TypeError(f"{name} must be an integer, got non-integer float: {val!r}")
    return iv


def _to_int_list(name, x):
    if isinstance(x, (list, tuple)):
        return [_as_pos_int(name, v) for v in x]
    return [_as_pos_int(name, x)]


def _extract_gate_counts(read_counter_complete_ret):
    """
    Accepts whatever your counter returns and produces 1D gate counts:
      (num_gates_total,)
    Typical case: ret = [array(num_apds, num_gates_total)]
    """
    raw = read_counter_complete_ret
    if isinstance(raw, (list, tuple)) and len(raw) == 1:
        raw = raw[0]
    arr = np.array(raw)

    if arr.ndim == 2:
        # (num_apds, num_gates) -> sum APDs
        return arr.sum(axis=0)
    if arr.ndim == 1:
        return arr
    raise RuntimeError(f"Unexpected counter return shape: {arr.shape}")


def main(
    nv_sig,
    num_steps,
    num_reps,
    num_runs,
    run_fn=None,
    step_fn=None,
    uwave_ind_list=(0,),
    uwave_freq_list=None,
    num_exps=2,
    apd_indices=(0,),
    load_iq=False,
    stream_load_in_run_fn=True,  # kept for compatibility; not enforced here
    charge_prep_fn=None,
) -> dict:
    # ---------- validate ints ----------
    num_steps = _as_pos_int("num_steps", num_steps)
    num_reps = _as_pos_int("num_reps", num_reps)
    num_runs = _as_pos_int("num_runs", num_runs)
    num_exps = _as_pos_int("num_exps", num_exps)

    uwave_ind_list = _to_int_list("uwave_ind", uwave_ind_list)
    apd_indices = _to_int_list("apd_index", apd_indices)

    # ---------- NV positioning ----------
    tb.reset_cfm()
    pos.set_xyz_on_nv(nv_sig)

    pulsegen = tb.get_server_pulse_streamer()
    counter = tb.get_server_counter()

    # ---------- Microwave setup ----------
    for uwave_ind in uwave_ind_list:
        vsg = tb.get_virtual_sig_gen_dict(uwave_ind)
        uwave_power = vsg["uwave_power"]
        freq = uwave_freq_list[uwave_ind] if uwave_freq_list else vsg["frequency"]

        sig_gen = tb.get_server_sig_gen(uwave_ind)
        if load_iq:
            sig_gen.load_iq()
        sig_gen.set_amp(uwave_power)
        sig_gen.set_freq(freq)
        print(f"MW[{uwave_ind}]  freq: {freq} GHz,  power: {uwave_power} dBm")

    # ---------- containers ----------
    counts = np.zeros((num_exps, num_runs, num_steps, num_reps), dtype=np.int32)
    step_ind_master_list = [None] * num_runs
    crash_counter = [None] * num_runs
    step_ind_list = list(range(num_steps))

    tb.init_safe_stop()

    try:
        for run_ind in range(num_runs):
            print(f"\n[Run {run_ind + 1}/{num_runs}]")

            if tb.safe_stop():
                break

            if charge_prep_fn:
                charge_prep_fn(nv_sig)

            # MW on
            for uwave_ind in uwave_ind_list:
                tb.get_server_sig_gen(uwave_ind).uwave_on()

            # randomize step order
            shuffle(step_ind_list)
            step_ind_master_list[run_ind] = step_ind_list.copy()

            # optional per-run hook (e.g. preload something)
            if run_fn:
                run_fn(step_ind_list)

            # start counter stream
            counter.start_tag_stream(apd_indices, True, True)

            for step_ind in step_ind_list:
                if tb.safe_stop():
                    break

                if step_fn:
                    step_fn(step_ind)

                # clear and run
                counter.clear_buffer()
                pulsegen.stream_start(num_reps)

                gate_counts = _extract_gate_counts(counter.read_counter_complete(1))
                expected = num_exps * num_reps
                if gate_counts.size != expected:
                    raise RuntimeError(
                        f"Got {gate_counts.size} gated counts; expected {expected} "
                        f"(num_exps={num_exps} × num_reps={num_reps}). "
                        f"Sequence must produce exactly {num_exps} APD gates per repetition."
                    )

                # De-interleave: [exp0_rep0, exp1_rep0, exp0_rep1, exp1_rep1, ...]
                for exp_ind in range(num_exps):
                    counts[exp_ind, run_ind, step_ind, :] = gate_counts[exp_ind::num_exps]

            # # MW off
            for uwave_ind in uwave_ind_list:
                tb.get_server_sig_gen(uwave_ind).uwave_off()

            counter.stop_tag_stream()

            #TODO drift compensate (best-effort)
            # try:
            #     targeting.compensate_for_drift(nv_sig, no_crash=True)
            # except Exception:
            #     pass

            crash_counter[run_ind] = 0

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
