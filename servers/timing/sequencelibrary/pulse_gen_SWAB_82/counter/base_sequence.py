# -*- coding: utf-8 -*-
"""
Base spin period for Pulse Streamer (confocal).
Builds ONE period that can contain multiple experiments (signal, reference, …).
"""

from typing import Callable, List, Tuple

from pulsestreamer import OutputState, Sequence

from servers.timing.sequencelibrary.pulse_gen_SWAB_82 import seq_utils
from utils.constants import VirtualLaserKey

# MW macro signature stays the same: (builder, t_ns, step_ns) -> new_t_ns
MWMacro = Callable[[seq_utils.PSBuilder, int, int], int]


def macro(
    base_args: list,
    uwave_macros: List[MWMacro],
    step_val_ns: int,
    *,
    num_reps_ignored: int = 1,  # (PS repeats in stream_start)
    include_reference: bool = False,  # (unused; pass uwave_ref in uwave_macros if needed)
) -> Tuple[Sequence, OutputState, List[int]]:
    """
    ONE period:
      for each experiment in uwave_macros:
         polarize → (optional pad) → MW macro(step_val_ns) → readout
    Returns: (Sequence, OutputState, [period_ns])
    """
    (
        pol_ns,
        readout_ns,
        uwave_ind,
        readout_laser,  # physical name, e.g. "laser_OPTO_589"
        readout_power,  # float volts if analog; None if digital
        pad_budget_ns,  # e.g. max_tau - tau for constant-evolution budget
    ) = base_args

    b = seq_utils.PSBuilder()
    # Delays + alignment origin (shared by laser/MW so arrivals line up at t0)
    ldel, udel, t0, uwbuf, short = seq_utils._delays(readout_laser, int(uwave_ind))

    t = 0
    for idx, mw_macro in enumerate(uwave_macros):
        # Polarize (aligned to same t0 so timing is consistent)
        t = seq_utils.macro_polarize(
            b,
            vkey=VirtualLaserKey.CHARGE_POL,
            start=t,
            duration=int(pol_ns),
            readout_laser_for_align=readout_laser,
            uwave_ind=int(uwave_ind),
        )

        # Buffer + evolution-budget pad only before the first experiment
        t += uwbuf + (int(pad_budget_ns) if idx == 0 else 0)

        # User-provided MW block for this experiment (your MW macro should
        # already call seq_utils.macro_mw_pulse / macro_pi / macro_pi_on_2
        # with the udel,t0 it captured when you created the macro)
        t = mw_macro(b, t, int(step_val_ns)) + uwbuf

        # Readout (aligned) + APD gate
        t = seq_utils.macro_readout(
            b,
            readout_laser=readout_laser,
            start=t,
            readout_ns=int(readout_ns),
            readout_power=readout_power,
            laser_delay=ldel,
            t0=t0,
            gate_apd=True,
        )
        # Keep the laser high long enough (OPX-like tail), then a short gap
        t += max(int(readout_ns), int(pol_ns)) - int(readout_ns) + short + uwbuf

    period = t + 500  # small trailer
    b.daq_tick_near_end(period)
    return b.emit(tail_pad=0)
