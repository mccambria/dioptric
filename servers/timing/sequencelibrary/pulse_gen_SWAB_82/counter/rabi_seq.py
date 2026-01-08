# -*- coding: utf-8 -*-
"""
Rabi sequence (Pulse Streamer): two APD gates per period (signal, then reference).
Constant period per step by padding with (pad_budget_ns - tau_ns).

Author: Saroj Chand
"""

# from typing import Callable, List, Tuple

# from pulsestreamer import OutputState, Sequence

# from servers.timing.sequencelibrary.pulse_gen_SWAB_82 import seq_utils
# from servers.timing.sequencelibrary.pulse_gen_SWAB_82.counter import base_sequence
# from utils import tool_belt as tb
# from utils.constants import VirtualLaserKey

# MWMacro = Callable[[seq_utils.PSBuilder, int, int], int]


# def _pick_align_index(uwave_inds: List[int], readout_laser: str) -> int:
#     """Use the chain with the largest MW delay for base_args alignment."""
#     best = uwave_inds[0]
#     _, best_udel, _, _, _ = seq_utils._delays(readout_laser, best)
#     for ind in uwave_inds[1:]:
#         _, udel, _, _, _ = seq_utils._delays(readout_laser, ind)
#         if udel > best_udel:
#             best, best_udel = ind, udel
#     return best


# def uwave_rabi(uwave_inds: List[int], readout_laser: str) -> MWMacro:
#     """Gate all MW chains in uwave_inds simultaneously for τ."""
#     # collect per-chain MW delays and get a shared t0
#     udel = {}
#     ldel_list = []
#     short = 10
#     for ind in uwave_inds:
#         ldel_i, udel_i, _t0, _uwbuf, short = seq_utils._delays(readout_laser, ind)
#         ldel_list.append(ldel_i)
#         udel[ind] = udel_i
#     # laser delay for readout path (same laser for both experiments)
#     # ldel = ldel_list[0]
#     ldel = 0
#     t0 = max([ldel] + list(udel.values())) + short

#     def _fn(b: seq_utils.PSBuilder, t: int, tau_ns: int) -> int:
#         for ind in uwave_inds:
#             seq_utils.macro_mw_pulse(
#                 b,
#                 uwave_ind=ind,
#                 start_ns=t,
#                 dur_ns=int(tau_ns),
#                 uwave_delay_ns=udel[ind],
#                 t0_ns=t0,
#             )
#         return t + int(tau_ns)

#     return _fn


# def uwave_ref() -> MWMacro:
#     def _fn(b: seq_utils.PSBuilder, t: int, _tau_ns: int) -> int:
#         return t

#     return _fn


# def get_seq(_server, _config, args) -> Tuple[Sequence, OutputState, List[int]]:
#     """
#     args:
#       0: base_args = [pol_ns, readout_ns, uwave_ind_align, readout_laser, readout_power, pad_budget_ns]
#       1: step_tau_ns (int)
#       2: num_reps_ignored (int)
#       3: uwave_inds (optional list[int])  # if omitted, falls back to [uwave_ind_align]
#     """
#     base_args, step_tau_ns, _num_reps = args[:3]
#     uwave_inds = args[3] if len(args) >= 4 else None

#     pol_ns, readout_ns, uwave_ind_align, ro_laser, ro_power, pad_budget_ns = base_args

#     # If caller didn't pass a list, use the align chain as the single source.
#     # if uwave_inds is None:
#     #     uwave_inds = uwave_ind_align
#     # else:
#     #     uwave_inds = [int(i) for i in uwave_inds]

#     # Ensure base_args uses the chain with the largest delay for alignment
#     align_ind = _pick_align_index(uwave_inds, ro_laser)
#     base_args = [
#         int(pol_ns),
#         int(readout_ns),
#         int(align_ind),
#         str(ro_laser),
#         (None if ro_power is None else float(ro_power)),
#         int(pad_budget_ns),
#     ]

#     # Two experiments total: (signal with all chains) + (reference)
#     uwave_macros = [uwave_rabi(uwave_inds, ro_laser), uwave_ref()]

#     return base_sequence.macro(
#         base_args=base_args,
#         uwave_macros=uwave_macros,
#         step_val_ns=int(step_tau_ns),
#         num_reps_ignored=int(_num_reps),
#         include_reference=False,
#     )

# Optional local preview
# if __name__ == "__main__":
#     tau_ns = 200
#     pol_ns = 1000
#     readout_ns = 300
#     pad_budget_ns = 220  # constant-evolution padding budget per period
#     ro_laser = tb.get_physical_laser_name(VirtualLaserKey.WIDEFIELD_CHARGE_READOUT)
#     ro_power = None  # None if digital; float volts if analog
#     # use both chains:
#     uwave_inds = [0, 1]
#     align_ind = _pick_align_index(uwave_inds, ro_laser)

#     base_args = [pol_ns, readout_ns, align_ind, ro_laser, ro_power, pad_budget_ns]
#     args = [base_args, tau_ns, 1, uwave_inds]

#     seq, _final, (period_ns,) = get_seq(None, None, args)
#     print(
#         f"[RABI PREVIEW] period = {period_ns} ns  (tau = {tau_ns} ns, MW={uwave_inds})"
#     )
#     # built-in plotter
#     import matplotlib.pyplot as plt

#     seq.plot()
#     plt.show()


# rabi_seq.py
import numpy as np
from pulsestreamer import OutputState, Sequence

from utils import common
from utils import tool_belt as tb
from utils.constants import Digital, ModMode, VirtualLaserKey


LOW = Digital.LOW
HIGH = Digital.HIGH


def _as_int64(name, v):
    try:
        iv = int(v)
    except Exception:
        raise TypeError(f"{name} must be int-like, got {type(v).__name__}: {v!r}")
    if iv < 0:
        raise ValueError(f"{name} must be >= 0, got {iv}")
    return np.int64(iv)


def _vkey_from_arg(x):
    """
    Accepts:
      - VirtualLaserKey member
      - "SPIN_READOUT"
      - "VirtualLaserKey.SPIN_READOUT"
    Returns VirtualLaserKey
    """
    if isinstance(x, VirtualLaserKey):
        return x
    if isinstance(x, str):
        name = x.split(".")[-1]
        return VirtualLaserKey[name]
    raise TypeError(f"Bad virtual laser key: {x!r}")


def _train_len(train):
    return np.int64(sum(int(d) for d, _ in train))


def get_seq(pulse_streamer, config, args):
    """
    Args formats supported:
      - [base_args, tau]
      - [base_args, tau, num_reps]   (num_reps ignored; base routine handles repetitions)
    where:
      base_args = [pol_ns, readout_ns, uwave_ind_list, readout_vkey, readout_power, max_tau]
    """
    # -------- parse args --------
    if len(args) >= 2 and isinstance(args[0], (list, tuple)):
        base_args = args[0]
        tau = args[1]
    else:
        raise ValueError(
            "Expected args as [base_args, tau] or [base_args, tau, num_reps]. "
            f"Got: {args!r}"
        )

    if len(base_args) != 6:
        raise ValueError(
            "base_args must be [pol_ns, readout_ns, uwave_ind_list, readout_vkey, "
            "readout_power, max_tau]. Got: "
            f"{base_args!r}"
        )

    pol_ns, readout_ns, uwave_ind_list, readout_vkey_arg, readout_power, max_tau = base_args

    pol_ns = _as_int64("pol_ns", pol_ns)
    readout_ns = _as_int64("readout_ns", readout_ns)
    tau = _as_int64("tau", tau)
    max_tau = _as_int64("max_tau", max_tau)

    if readout_ns > pol_ns:
        raise ValueError(f"readout_ns ({readout_ns}) must be <= pol_ns ({pol_ns}).")
    if tau > max_tau:
        raise ValueError(f"tau ({tau}) must be <= max_tau ({max_tau}).")

    pad_ns = max_tau - tau
    pre_post = pol_ns - readout_ns

    # normalize uwave_ind_list (Rabi: enforce exactly 1 channel for correct timing)
    if isinstance(uwave_ind_list, (int, np.integer)):
        uwave_ind_list = [int(uwave_ind_list)]
    else:
        uwave_ind_list = [int(x) for x in uwave_ind_list]

    if len(uwave_ind_list) != 1:
        raise ValueError("This rabi_seq expects exactly one uwave channel (uwave_ind_list length == 1).")

    uwave_ind = uwave_ind_list[0]

    # laser key
    readout_vkey = _vkey_from_arg(readout_vkey_arg)
    laser_name = tb.get_physical_laser_name(readout_vkey)

    # -------- wiring & delays --------
    pulser_wiring = config["Wiring"]["PulseGen"]
    do_apd_gate = pulser_wiring["do_apd_gate"]
    do_sample_clock = pulser_wiring["do_sample_clock"]

    # MW channel wiring
    vsg = tb.get_virtual_sig_gen_dict(uwave_ind)
    sig_gen_name = vsg["physical_name"]
    do_sig_gen_dm = pulser_wiring[f"do_{sig_gen_name}_dm"]

    laser_delay = _as_int64(
        "laser_delay", config["Optics"]["PhysicalLasers"][laser_name]["delay"]
    )
    uwave_delay = _as_int64(
        "uwave_delay", config["Microwaves"]["PhysicalSigGens"][sig_gen_name]["delay"]
    )
    common_delay = np.int64(max(laser_delay, uwave_delay))

    uwave_buffer = _as_int64("uwave_buffer", config["CommonDurations"]["uwave_buffer"])

    # -------- trains (common timebase) --------
    # Two APD gates per repetition: signal then reference
    apd_train = [
        (common_delay, LOW),
        (pre_post, LOW),
        (uwave_buffer, LOW),
        (max_tau, LOW),
        (uwave_buffer, LOW),
        (readout_ns, HIGH),
        (pre_post, LOW),
        (uwave_buffer, LOW),
        (max_tau, LOW),
        (uwave_buffer, LOW),
        (readout_ns, HIGH),
        (pre_post, LOW),
    ]

    # Laser starts early by (common_delay - laser_delay)
    laser_train = [
        (common_delay - laser_delay, HIGH),
        (pre_post, HIGH),
        (uwave_buffer, LOW),
        (max_tau, LOW),
        (uwave_buffer, LOW),
        (readout_ns, HIGH),
        (pre_post, HIGH),
        (uwave_buffer, LOW),
        (max_tau, LOW),
        (uwave_buffer, LOW),
        (readout_ns, HIGH),
        (pre_post + laser_delay, HIGH),
    ]

    # MW starts early by (common_delay - uwave_delay)
    # Only ON in the signal evolution window: tau HIGH then pad LOW
    mw_train = [
        (common_delay - uwave_delay, LOW),
        (pre_post, LOW),
        (uwave_buffer, LOW),
        (tau, HIGH),
        (pad_ns, LOW),
        (uwave_buffer, LOW),
        (readout_ns, LOW),
        (pre_post, LOW),
        (uwave_buffer, LOW),
        (max_tau, LOW),  # reference window: always OFF
        (uwave_buffer, LOW),
        (readout_ns, LOW),
        (pre_post + uwave_delay, LOW),
    ]

    # -------- assemble --------
    seq = Sequence()
    seq.setDigital(do_apd_gate, apd_train)
    seq.setDigital(do_sig_gen_dm, mw_train)

    # Laser: support optional power override for analog modulation
    mod_mode = config["Optics"]["PhysicalLasers"][laser_name]["mod_mode"]
    if mod_mode is ModMode.DIGITAL:
        tb.process_laser_seq(seq, readout_vkey, laser_train)
    elif mod_mode is ModMode.ANALOG:
        if readout_power is None:
            # fall back to config virtual laser "laser_power"
            vld = tb.get_virtual_laser_dict(readout_vkey)
            readout_power = vld.get("laser_power", None)
        if readout_power is None:
            raise ValueError(
                f"{laser_name} is ANALOG modulated but readout_power is None and config "
                f"VirtualLasers[{readout_vkey}] has no 'laser_power'."
            )
        ao_chan = pulser_wiring[f"ao_{laser_name}_am"]
        processed = [(dur, 0.0 if val is LOW else float(readout_power)) for dur, val in laser_train]
        seq.setAnalog(ao_chan, processed)
    else:
        raise ValueError(f"Unknown mod_mode for {laser_name}: {mod_mode}")

    period = _train_len(apd_train)

    # Sample clock: 100 ns HIGH near end with 100 ns buffers (if long enough)
    if period >= 300:
        clk_train = [(period - 200, LOW), (100, HIGH), (100, LOW)]
    else:
        clk_train = [(period, LOW)]
    seq.setDigital(do_sample_clock, clk_train)

    final = OutputState([], 0.0, 0.0)
    return seq, final, [period]


if __name__ == "__main__":
    cfg = common.get_config_dict()
    # tb.set_delays_to_zero(cfg)  # uncomment for cleaner timing debug plots

    pol_ns = 5000
    readout_ns = 300
    uwave_ind_list = [0]
    readout_vkey = VirtualLaserKey.SPIN_READOUT.name  # e.g. "SPIN_READOUT"
    readout_power = None  # set float if ANALOG and you want override
    max_tau = 1000
    tau = 100

    base_args = [pol_ns, readout_ns, uwave_ind_list, readout_vkey, readout_power, max_tau]
    seq, final, ret = get_seq(None, cfg, [base_args, tau])

    print("Period (ns):", int(ret[0]))
    seq.plot()
