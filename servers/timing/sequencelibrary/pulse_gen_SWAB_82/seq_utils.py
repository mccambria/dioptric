# -*- coding: utf-8 -*-
"""
Pulse Streamer sequence utilities (single-NV / confocal).

This module gives you OPX-like "macros" for the Swabian Pulse Streamer:
- A lightweight builder to accumulate trains per channel then emit a Sequence
- High-level macros: DAQ clock, APD gate windows, laser pulses, microwave gates,
  polarization, SCC, charge readout, simple readout, charge-init + readout, Rabi base
- Utilities that were previously in tool_belt but used by sequences (encode/decode etc.)

Conventions:
- Durations/times are in nanoseconds (ns).
- We read wiring from config["Wiring"]["PulseGen"].
- Digital channels use utils.constants.Digital.{LOW,HIGH}.
- Analog laser "power" is passed as a float (volts), mapped to ao_<laser>_am.

Author: Saroj Chand
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cache
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from pulsestreamer import OutputState, Sequence

from utils import common
from utils.constants import Digital, ModMode, VirtualLaserKey

LOW, HIGH = 0, 1

# ---------------------------------------------------------------------
# Cached config getters
# ---------------------------------------------------------------------


@cache
def CFG() -> dict:
    return common.get_config_dict()


@cache
def W() -> dict:
    return CFG()["Wiring"]["PulseGen"]


@cache
def vd_laser(vkey: VirtualLaserKey) -> dict:
    return CFG()["Optics"]["VirtualLasers"][vkey]


@cache
def phys_laser_name(vkey: VirtualLaserKey) -> str:
    return vd_laser(vkey)["physical_name"]


@cache
def phys_laser_dict(physical_name: str) -> dict:
    return CFG()["Optics"]["PhysicalLasers"][physical_name]


@cache
def vsg(uwave_ind: int) -> dict:
    return CFG()["Microwaves"]["VirtualSigGens"][uwave_ind]


@cache
def sig_gen_name(uwave_ind: int) -> str:
    return vsg(uwave_ind)["physical_name"]


@cache
def common_ns(key: str) -> int:
    return int(CFG()["CommonDurations"][key])


# ---------------------------------------------------------------------
# Builder: collect trains, auto-pad, emit Sequence
# ---------------------------------------------------------------------
@dataclass
class _Train:
    segs: List[Tuple[int, Any]] = field(default_factory=list)
    is_digital: bool = True

    def length(self) -> int:
        return int(sum(d for d, _ in self.segs))

    def pad_to(self, t_ns: int):
        cur = self.length()
        if t_ns > cur:
            # pad with baseline, not the last value!
            baseline = Digital.LOW if self.is_digital else 0.0
            self.segs.append((t_ns - cur, baseline))

    def add(self, start_ns: int, dur_ns: int, val: Any):
        self.pad_to(start_ns)
        if dur_ns > 0:
            self.segs.append((int(dur_ns), val))


class PSBuilder:
    """Accumulates per-channel trains, then emits a padded Sequence."""

    def __init__(self):
        self._tr: Dict[Tuple[str, int], _Train] = {}

    def _get(self, kind: str, chan: int) -> _Train:
        key = (kind, chan)
        if key not in self._tr:
            self._tr[key] = _Train(is_digital=(kind == "do"))
        return self._tr[key]

    # Channel resolvers
    def _do(self, name: str) -> int:
        return W()[f"do_{name}"]

    def _ao(self, name: str) -> int:
        return W()[f"ao_{name}"]

    # Primitives
    def digital(self, do_name: str, start: int, dur: int, high: bool = True):
        tr = self._get("do", self._do(do_name))
        tr.add(start, dur, HIGH if high else LOW)

    def analog(self, ao_name: str, start: int, dur: int, value: float):
        tr = self._get("ao", self._ao(ao_name))
        tr.add(start, dur, float(value))

    # Convenience
    def apd_gate(self, start: int, dur: int):
        self.digital("apd_gate", start, dur, True)

    def cam_trig(self, start: int, dur: int):
        self.digital("camera_trigger", start, dur, True)

    def daq_tick_near_end(self, period: int, hi: int = 100, lo: int = 100):
        ch = self._do("sample_clock")
        tr = self._get("do", ch)
        pre = max(0, period - (hi + lo) - tr.length())
        if pre:
            tr.add(tr.length(), pre, LOW)
        tr.add(tr.length(), hi, HIGH)
        tr.add(tr.length(), lo, LOW)

    # Aligned pulses (respect per-path delays)
    def laser_aligned(
        self,
        laser_name: str,
        start: int,
        dur: int,
        power: Optional[float],
        laser_delay: int,
        t0: int,
    ):
        # align epoch such that arrivals line up at t0
        start_adj = start + (t0 - laser_delay)
        mm = phys_laser_dict(laser_name)["mod_mode"]
        if mm is ModMode.DIGITAL:
            self.digital(f"{laser_name}_dm", start_adj, dur, True)
        else:
            self.analog(
                f"{laser_name}_am",
                start_adj,
                dur,
                0.0 if power is None else float(power),
            )

    def mw_gate_aligned(
        self, sig_name: str, start: int, dur: int, uwave_delay: int, t0: int
    ):
        start_adj = start + (t0 - uwave_delay)
        self.digital(f"{sig_name}_gate", start_adj, dur, True)

    def emit(self, tail_pad: int = 0) -> Tuple[Sequence, OutputState, List[int]]:
        period = max((tr.length() for tr in self._tr.values()), default=0) + int(
            tail_pad
        )
        for tr in self._tr.values():
            tr.pad_to(period)
        seq = Sequence()
        for (kind, chan), tr in self._tr.items():
            if kind == "do":
                seq.setDigital(chan, tr.segs)
            else:
                seq.setAnalog(chan, tr.segs)
        final = OutputState([W()["do_sample_clock"]], 0.0, 0.0)
        return seq, final, [np.int64(period)]


# ---------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------
@cache
def _pi_durations_ns(uwave_ind: int) -> tuple[int, int]:
    """
    Return (pi_ns, pi_over_2_ns) for this microwave chain.
    Falls back to rabi_period/2 and rabi_period/4 if explicit values
    aren't present in the config.
    """
    d = vsg(uwave_ind)
    # Prefer explicit calibration if present
    pi_ns = int(d.get("pi_pulse")) if "pi_pulse" in d else None
    pi2_ns = int(d.get("pi_on_2_pulse")) if "pi_on_2_pulse" in d else None

    if pi_ns is None or pi2_ns is None:
        rabi = int(d.get("rabi_period", 0))
        if rabi <= 0 and (pi_ns is None or pi2_ns is None):
            raise KeyError(
                f"VirtualSigGens[{uwave_ind}] needs either pi_pulse/pi_on_2_pulse "
                f"or rabi_period in config."
            )
        if pi_ns is None:
            pi_ns = max(1, rabi // 2)
        if pi2_ns is None:
            pi2_ns = max(1, rabi // 4)
    return pi_ns, pi2_ns


def _delays(readout_laser: str, uwave_ind: int) -> Tuple[int, int, int, int, int]:
    """laser_delay, uwave_delay, t0(common), uwave_buf, short_buf"""
    laser_delay = int(phys_laser_dict(readout_laser)["delay"])
    uwave_delay = int(
        CFG()["Microwaves"]["PhysicalSigGens"][sig_gen_name(uwave_ind)]["delay"]
    )
    short_buf = 10
    uwave_buf = int(CFG()["CommonDurations"]["uwave_buffer"])
    t0 = max(laser_delay, uwave_delay) + short_buf
    return laser_delay, uwave_delay, t0, uwave_buf, short_buf


# ---------------------------------------------------------------------
# Mid-level macros (single-NV; no AOD stepping here)
# ---------------------------------------------------------------------


def macro_polarize(
    b: PSBuilder,
    vkey: VirtualLaserKey,
    start: int,
    duration: Optional[int] = None,
    power: Optional[float] = None,
    readout_laser_for_align: Optional[str] = None,
    uwave_ind: Optional[int] = None,
) -> int:
    laser = phys_laser_name(vkey)
    dur = int(vd_laser(vkey)["duration"] if duration is None else duration)
    # align using same t0 as readout path (pass both when available)
    if readout_laser_for_align is not None and uwave_ind is not None:
        ldel, udel, t0, *_ = _delays(readout_laser_for_align, uwave_ind)
        b.laser_aligned(laser, start, dur, power, ldel, t0)
    else:
        # fallback: no alignment offset
        mm = phys_laser_dict(laser)["mod_mode"]
        if mm is ModMode.DIGITAL:
            b.digital(f"{laser}_dm", start, dur, True)
        else:
            b.analog(f"{laser}_am", start, dur, 0.0 if power is None else float(power))
    return start + dur


def macro_readout(
    b: PSBuilder,
    readout_laser: str,
    start: int,
    readout_ns: int,
    readout_power: Optional[float],
    laser_delay: int,
    t0: int,
    gate_apd: bool = True,
) -> int:
    b.laser_aligned(readout_laser, start, readout_ns, readout_power, laser_delay, t0)
    if gate_apd:
        b.apd_gate(start + t0, readout_ns)  # open gate when arrivals occur
    return start + readout_ns


# --- one primitive, two wrappers --------------------------------------


def macro_mw_pulse(
    b: PSBuilder,
    uwave_ind: int,
    start_ns: int,
    dur_ns: int,
    uwave_delay_ns: int,
    t0_ns: int,
) -> int:
    """
    Low-level primitive: gate the microwave source for dur_ns with alignment.
    Returns the time immediately after the pulse (start_ns + dur_ns).
    """
    # NOTE: use the *actual* getter name you have; in your snippet it was `sig_gen_name`,
    # elsewhere it's `get_sig_gen_name`. Keep it consistent:
    name = sig_gen_name(uwave_ind)
    b.mw_gate_aligned(name, start_ns, int(dur_ns), int(uwave_delay_ns), int(t0_ns))
    return start_ns + int(dur_ns)


def macro_pi(
    b: PSBuilder,
    uwave_ind: int,
    start_ns: int,
    uwave_delay_ns: int,
    t0_ns: int,
    dur_override_ns: int | None = None,
) -> int:
    """
    π pulse wrapper. If dur_override_ns is None, use config; otherwise use override.
    """
    pi_ns, _ = _pi_durations_ns(uwave_ind)
    dur = int(pi_ns if dur_override_ns is None else dur_override_ns)
    return macro_mw_pulse(b, uwave_ind, start_ns, dur, uwave_delay_ns, t0_ns)


def macro_pi_on_2(
    b: PSBuilder,
    uwave_ind: int,
    start_ns: int,
    uwave_delay_ns: int,
    t0_ns: int,
    dur_override_ns: int | None = None,
) -> int:
    """
    π/2 pulse wrapper. If dur_override_ns is None, use config; otherwise use override.
    """
    _, pi2_ns = _pi_durations_ns(uwave_ind)
    dur = int(pi2_ns if dur_override_ns is None else dur_override_ns)
    return macro_mw_pulse(b, uwave_ind, start_ns, dur, uwave_delay_ns, t0_ns)


# ---------------------------------------------------------------------
# Back-compat helpers (if older seq files import these)
# ---------------------------------------------------------------------


def process_laser_seq(seq, config, laser_name, laser_power, train):
    if config is None:
        config = CFG()
    mm = config["Optics"][laser_name]["mod_mode"]
    if mm is ModMode.DIGITAL:
        seq.setDigital(W()[f"do_{laser_name}_dm"], train)
    else:
        processed = []
        for dur, lvl in train:
            v = (
                0.0
                if lvl is LOW
                else (0.0 if laser_power is None else float(laser_power))
            )
            processed.append((int(dur), v))
        seq.setAnalog(W()[f"ao_{laser_name}_am"], processed)


def set_delays_to_zero(cfg: dict):
    for k, v in list(cfg.items()):
        if k == "delay":
            cfg[k] = 0
        elif isinstance(v, dict):
            set_delays_to_zero(v)


def set_delays_to_sixteen(cfg: dict):
    for k, v in list(cfg.items()):
        if isinstance(v, dict):
            set_delays_to_sixteen(v)
        elif isinstance(k, str) and k.endswith("delay"):
            cfg[k] = 16


def seq_train_length_check(train: Iterable[Tuple[int, Any]]) -> int:
    total = int(sum(d for d, _ in train))
    print(total)
    return total


# ---------------------------------------------------------------------
# Quick self-test / preview: python seq_utils.py
# ---------------------------------------------------------------------

if __name__ == "__main__":
    # Minimal sanity test & preview (requires matplotlib installed for ps.plot()).
    args = dict(
        tau=200,
        pol=1000,
        ro=300,
        max_tau=1000,
        uw=0,
        ro_laser="laser_INTE_520",
        ro_pow=None,
    )
    seq, _, [per] = build_spin_echo_sequence(
        args["tau"],
        args["pol"],
        args["ro"],
        args["max_tau"],
        args["uw"],
        args["ro_laser"],
        args["ro_pow"],
    )
    try:
        seq.plot()
    except Exception as e:
        print("Preview unavailable:", e)
    print("Period (ns):", per)
