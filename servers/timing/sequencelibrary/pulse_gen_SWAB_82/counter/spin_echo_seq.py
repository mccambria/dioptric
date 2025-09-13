# -*- coding: utf-8 -*-
"""
Spin-echo sequence (Pulse Streamer): π/2 — τ/2 — π — τ/2
Two experiments per period by default: signal (MW on) + reference (no MW).
Period is kept constant by padding with pad_budget_ns (typically max_tau - tau).

Author: Saroj Chand
"""

from typing import Callable, List, Tuple

from pulsestreamer import OutputState, Sequence

from servers.timing.sequencelibrary.pulse_gen_SWAB_82 import seq_utils
from servers.timing.sequencelibrary.pulse_gen_SWAB_82.counter import base_sequence
from utils import tool_belt as tb
from utils.constants import VirtualLaserKey

# MW macro signature: (builder, t_ns, step_val_ns) -> new_t_ns
MWMacro = Callable[[seq_utils.PSBuilder, int, int], int]


def _common_t0(
    readout_laser: str, uwave_inds: List[int]
) -> tuple[int, dict[int, int], int]:
    """
    Compute a single alignment epoch t0 for all chains based on the readout path.
    Returns (laser_delay, {chain: uwave_delay}, t0).
    """
    # Use first chain to read laser delay; they all share the same readout laser
    ldel, _, _t0, _buf, short = seq_utils._delays(readout_laser, uwave_inds[0])
    udel = {}
    for ind in uwave_inds:
        _, udel[ind], _, _, _ = seq_utils._delays(readout_laser, ind)
    t0 = max([ldel] + list(udel.values())) + short
    return ldel, udel, t0


def uwave_spin_echo(uwave_inds: List[int], readout_laser: str) -> MWMacro:
    """
    Build the MW macro for Hahn echo (π/2 — τ/2 — π — τ/2) applied
    simultaneously on all provided MW chains.
    """
    _ldel, udel, t0 = _common_t0(readout_laser, uwave_inds)

    def _fn(b: seq_utils.PSBuilder, t: int, tau_ns: int) -> int:
        # split τ into two halves (handle odd τ)
        half_a = int(tau_ns) // 2
        half_b = int(tau_ns) - half_a

        # π/2
        for ind in uwave_inds:
            seq_utils.macro_pi_on_2(
                b, uwave_ind=ind, start_ns=t, uwave_delay_ns=udel[ind], t0_ns=t0
            )
        t += half_a

        # π
        for ind in uwave_inds:
            seq_utils.macro_pi(
                b, uwave_ind=ind, start_ns=t, uwave_delay_ns=udel[ind], t0_ns=t0
            )
        t += half_b

        # π/2
        for ind in uwave_inds:
            seq_utils.macro_pi_on_2(
                b, uwave_ind=ind, start_ns=t, uwave_delay_ns=udel[ind], t0_ns=t0
            )
        # t += half_a

        return t

    return _fn


def uwave_ref() -> MWMacro:
    """Reference experiment: no MW."""

    def _fn(b: seq_utils.PSBuilder, t: int, _tau_ns: int) -> int:
        return t

    return _fn


def get_seq(_server, _config, args) -> Tuple[Sequence, OutputState, List[int]]:
    """
    Args:
      0: base_args = [pol_ns, readout_ns, uwave_ind_align, readout_laser, readout_power, pad_budget_ns]
      1: step_tau_ns (int)                   # total echo gap τ
      2: num_reps_ignored (int)              # PS repeats are set by stream_start()
      3: uwave_inds (optional list[int])     # if omitted, uses [uwave_ind_align]

    Returns:
      (Sequence, OutputState, [period_ns])
    """
    base_args, step_tau_ns, _num_reps = args[:3]
    uwave_inds = args[3] if len(args) >= 4 else None

    pol_ns, readout_ns, uwave_ind_align, ro_laser, ro_power, pad_budget_ns = base_args

    # Default to single chain if none provided
    if uwave_inds is None:
        uwave_inds = [int(uwave_ind_align)]
    else:
        uwave_inds = [int(i) for i in uwave_inds]

    # Use the largest-delay chain for alignment in base_args
    def _pick_align_index(indices: List[int], readout_laser: str) -> int:
        best = indices[0]
        _, best_udel, *_ = seq_utils._delays(readout_laser, best)
        for ind in indices[1:]:
            _, udel, *_ = seq_utils._delays(readout_laser, ind)
            if udel > best_udel:
                best, best_udel = ind, udel
        return best

    align_ind = _pick_align_index(uwave_inds, ro_laser)
    base_args = [
        int(pol_ns),
        int(readout_ns),
        int(align_ind),
        str(ro_laser),
        (None if ro_power is None else float(ro_power)),
        int(pad_budget_ns),
    ]

    # Two experiments: signal (spin-echo), reference
    uwave_macros = [uwave_spin_echo(uwave_inds, ro_laser), uwave_ref()]

    return base_sequence.macro(
        base_args=base_args,
        uwave_macros=uwave_macros,
        step_val_ns=int(step_tau_ns),
        num_reps_ignored=int(_num_reps),
        include_reference=False,  # we passed ref explicitly
    )


# -------- Optional local preview --------
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    tau_ns = 500
    pol_ns = 1000
    readout_ns = 300
    pad_budget_ns = 800  # usually max_tau - tau to keep constant-period
    ro_laser = tb.get_physical_laser_name(VirtualLaserKey.WIDEFIELD_CHARGE_READOUT)
    ro_power = None  # None if digital; float volts if analog
    uwave_inds = [0, 1]  # run both chains simultaneously

    # choose the alignment chain
    def _pick_align(indices, laser):
        best = indices[0]
        _, best_udel, *_ = seq_utils._delays(laser, best)
        for ind in indices[1:]:
            _, udel, *_ = seq_utils._delays(laser, ind)
            if udel > best_udel:
                best, best_udel = ind, udel
        return best

    align_ind = _pick_align(uwave_inds, ro_laser)
    base_args = [pol_ns, readout_ns, align_ind, ro_laser, ro_power, pad_budget_ns]
    args = [base_args, tau_ns, 1, uwave_inds]

    seq, _final, (period_ns,) = get_seq(None, None, args)
    print(
        f"[SPIN-ECHO PREVIEW] period = {period_ns} ns  (τ = {tau_ns} ns, MW={uwave_inds})"
    )
    seq.plot()
    plt.show()
