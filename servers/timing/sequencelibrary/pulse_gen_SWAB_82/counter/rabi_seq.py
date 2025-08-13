# -*- coding: utf-8 -*-
"""
Rabi sequence (Pulse Streamer): two APD gates per period (signal, then reference).
Constant period per step by padding with (pad_budget_ns - tau_ns).

Author: Saroj Chand
"""

from typing import Callable, List, Tuple

from pulsestreamer import OutputState, Sequence

from servers.timing.sequencelibrary.pulse_gen_SWAB_82 import seq_utils
from servers.timing.sequencelibrary.pulse_gen_SWAB_82.counter import base_sequence
from utils import tool_belt as tb
from utils.constants import VirtualLaserKey

MWMacro = Callable[[seq_utils.PSBuilder, int, int], int]


def _pick_align_index(uwave_inds: List[int], readout_laser: str) -> int:
    """Use the chain with the largest MW delay for base_args alignment."""
    best = uwave_inds[0]
    _, best_udel, _, _, _ = seq_utils._delays(readout_laser, best)
    for ind in uwave_inds[1:]:
        _, udel, _, _, _ = seq_utils._delays(readout_laser, ind)
        if udel > best_udel:
            best, best_udel = ind, udel
    return best


def uwave_rabi(uwave_inds: List[int], readout_laser: str) -> MWMacro:
    """Gate all MW chains in uwave_inds simultaneously for τ."""
    # collect per-chain MW delays and get a shared t0
    udel = {}
    ldel_list = []
    short = 10
    for ind in uwave_inds:
        ldel_i, udel_i, _t0, _uwbuf, short = seq_utils._delays(readout_laser, ind)
        ldel_list.append(ldel_i)
        udel[ind] = udel_i
    # laser delay for readout path (same laser for both experiments)
    # ldel = ldel_list[0]
    ldel = 0
    t0 = max([ldel] + list(udel.values())) + short

    def _fn(b: seq_utils.PSBuilder, t: int, tau_ns: int) -> int:
        for ind in uwave_inds:
            seq_utils.macro_mw_pulse(
                b,
                uwave_ind=ind,
                start_ns=t,
                dur_ns=int(tau_ns),
                uwave_delay_ns=udel[ind],
                t0_ns=t0,
            )
        return t + int(tau_ns)

    return _fn


def uwave_ref() -> MWMacro:
    def _fn(b: seq_utils.PSBuilder, t: int, _tau_ns: int) -> int:
        return t

    return _fn


def get_seq(_server, _config, args) -> Tuple[Sequence, OutputState, List[int]]:
    """
    args:
      0: base_args = [pol_ns, readout_ns, uwave_ind_align, readout_laser, readout_power, pad_budget_ns]
      1: step_tau_ns (int)
      2: num_reps_ignored (int)
      3: uwave_inds (optional list[int])  # if omitted, falls back to [uwave_ind_align]
    """
    base_args, step_tau_ns, _num_reps = args[:3]
    uwave_inds = args[3] if len(args) >= 4 else None

    pol_ns, readout_ns, uwave_ind_align, ro_laser, ro_power, pad_budget_ns = base_args

    # If caller didn't pass a list, use the align chain as the single source.
    if uwave_inds is None:
        uwave_inds = [int(uwave_ind_align)]
    else:
        uwave_inds = [int(i) for i in uwave_inds]

    # Ensure base_args uses the chain with the largest delay for alignment
    align_ind = _pick_align_index(uwave_inds, ro_laser)
    base_args = [
        int(pol_ns),
        int(readout_ns),
        int(align_ind),
        str(ro_laser),
        (None if ro_power is None else float(ro_power)),
        int(pad_budget_ns),
    ]

    # Two experiments total: (signal with all chains) + (reference)
    uwave_macros = [uwave_rabi(uwave_inds, ro_laser), uwave_ref()]

    return base_sequence.macro(
        base_args=base_args,
        uwave_macros=uwave_macros,
        step_val_ns=int(step_tau_ns),
        num_reps_ignored=int(_num_reps),
        include_reference=False,
    )


# Optional local preview
if __name__ == "__main__":
    tau_ns = 200
    pol_ns = 1000
    readout_ns = 300
    pad_budget_ns = 220  # constant-evolution padding budget per period
    ro_laser = tb.get_physical_laser_name(VirtualLaserKey.WIDEFIELD_CHARGE_READOUT)
    ro_power = None  # None if digital; float volts if analog
    # use both chains:
    uwave_inds = [0, 1]
    align_ind = _pick_align_index(uwave_inds, ro_laser)

    base_args = [pol_ns, readout_ns, align_ind, ro_laser, ro_power, pad_budget_ns]
    args = [base_args, tau_ns, 1, uwave_inds]

    seq, _final, (period_ns,) = get_seq(None, None, args)
    print(
        f"[RABI PREVIEW] period = {period_ns} ns  (tau = {tau_ns} ns, MW={uwave_inds})"
    )
    # built-in plotter
    import matplotlib.pyplot as plt

    seq.plot()
    plt.show()
