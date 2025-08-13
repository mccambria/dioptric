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
from typing import Any, Dict, Iterable, List, Tuple

from pulsestreamer import OutputState, Sequence

from utils import common
from utils import tool_belt as tb
from utils.constants import Digital, ModMode, VirtualLaserKey

# ---------------------------------------------------------------------
# Config getters used by sequences
# ---------------------------------------------------------------------


@cache
def get_cfg() -> dict:
    return common.get_config_dict()


@cache
def W() -> dict:
    """Pulse Streamer wiring (digital + analog channel numbers)."""
    return get_cfg()["Wiring"]["PulseGen"]


@cache
def get_virtual_laser_dict(vkey: VirtualLaserKey) -> dict:
    return get_cfg()["Optics"]["VirtualLasers"][vkey]


@cache
def get_physical_laser_name(vkey: VirtualLaserKey) -> str:
    return get_virtual_laser_dict(vkey)["physical_name"]


@cache
def get_physical_laser_dict(physical_laser_name: str) -> dict:
    return get_cfg()["Optics"]["PhysicalLasers"][physical_laser_name]


@cache
def get_virtual_sig_gen_dict(uwave_ind: int) -> dict:
    return get_cfg()["Microwaves"]["VirtualSigGens"][uwave_ind]


@cache
def get_sig_gen_name(uwave_ind: int) -> str:
    return get_virtual_sig_gen_dict(uwave_ind)["physical_name"]


@cache
def get_common_duration_ns(key: str) -> int:
    return get_cfg()["CommonDurations"][key]


# ---------------------------------------------------------------------
# Builder: accumulate per-channel trains then emit a Sequence
# ---------------------------------------------------------------------


@dataclass
class _Train:
    """Internal channel train."""

    # list of (dur_ns, value) where value is Digital or float depending on type
    segs: List[Tuple[int, Any]] = field(default_factory=list)
    is_digital: bool = True

    def length(self) -> int:
        return int(sum(d for d, _ in self.segs))

    def last_val(self) -> Any:
        if not self.segs:
            return Digital.LOW if self.is_digital else 0.0
        return self.segs[-1][1]

    def pad_to(self, t_ns: int):
        cur = self.length()
        if t_ns > cur:
            self.segs.append((t_ns - cur, self.last_val()))

    def add(self, start_ns: int, dur_ns: int, val: Any):
        self.pad_to(start_ns)
        if dur_ns > 0:
            self.segs.append((int(dur_ns), val))


class PSBuilder:
    """
    Helper that collects trains for named channels, then writes them to a
    pulsestreamer.Sequence with consistent period/padding.
    """

    def __init__(self):
        self._trains: Dict[Tuple[str, int], _Train] = {}  # key=(kind, chan)

    # ---- resolve channels
    def _get_train(self, kind: str, chan: int) -> _Train:
        key = (kind, chan)
        if key not in self._trains:
            self._trains[key] = _Train(is_digital=(kind == "do"))
        return self._trains[key]

    def _do(self, name: str) -> int:
        return W()[f"do_{name}"]

    def _ao(self, name: str) -> int:
        return W()[f"ao_{name}"]

    # ---- write primitives
    def digital_window(self, do_name: str, start: int, dur: int, high: bool = True):
        chan = self._do(do_name)
        tr = self._get_train("do", chan)
        tr.add(start, dur, Digital.HIGH if high else Digital.LOW)

    def analog_window(self, ao_name: str, start: int, dur: int, value: float):
        chan = self._ao(ao_name)
        tr = self._get_train("ao", chan)
        tr.add(start, dur, float(value))

    # ---- canned patterns
    def add_apd_gate(self, start: int, dur: int):
        self.digital_window("apd_gate", start, dur, True)

    def add_camera_trigger(self, start: int, dur: int):
        self.digital_window("camera_trigger", start, dur, True)

    def add_daq_clock_once(self, period: int, high_ns: int = 100, low_ns: int = 100):
        """
        Emit one rising edge per period near the end (matches your legacy pattern):
        (period - (high+low)) LOW, then high_ns HIGH, then low_ns LOW.
        """
        chan = self._do("sample_clock")
        tr = self._get_train("do", chan)
        total = tr.length()
        # align this channel independently to 'period'
        if total < period - (high_ns + low_ns):
            tr.add(total, period - (high_ns + low_ns) - total, Digital.LOW)
        tr.add(tr.length(), high_ns, Digital.HIGH)
        tr.add(tr.length(), low_ns, Digital.LOW)

    def add_mw_gate(self, sig_gen_name: str, start: int, dur: int):
        """Gate for a microwave source (digital)."""
        self.digital_window(f"{sig_gen_name}_gate", start, dur, True)

    def add_laser_window(
        self, laser_name: str, start: int, dur: int, power: float | None
    ):
        """
        Laser modulation by config: DIGITAL -> do_<laser>_dm, ANALOG -> ao_<laser>_am
        """
        mod_mode = get_physical_laser_dict(laser_name)["mod_mode"]
        if mod_mode is ModMode.DIGITAL:
            self.digital_window(f"{laser_name}_dm", start, dur, True)
        else:
            if power is None:
                power = 0.0  # safe fallback
            self.analog_window(f"{laser_name}_am", start, dur, power)

    # ---- finalize
    def emit(self, tail_pad: int = 0) -> Tuple[Sequence, OutputState, List[int]]:
        # compute unified period
        period = max((tr.length() for tr in self._trains.values()), default=0) + int(
            tail_pad
        )
        for tr in self._trains.values():
            tr.pad_to(period)

        seq = Sequence()
        for (kind, chan), tr in self._trains.items():
            if kind == "do":
                seq.setDigital(chan, tr.segs)
            else:
                seq.setAnalog(chan, tr.segs)

        # nothing to force, leave outputs safe: sample clock low, etc.
        final_dos = [W()["do_sample_clock"]]
        final = OutputState(final_dos, 0.0, 0.0)
        return seq, final, [period]


# ---------------------------------------------------------------------
# Mid-level helpers (mirroring OPX-style macros)
# ---------------------------------------------------------------------


def macro_simple_readout(
    delay: int,
    readout_ns: int,
    readout_laser: str,
    readout_power: float | None,
    tail_pad: int = 300,
) -> Tuple[Sequence, OutputState, List[int]]:
    """
    DAQ clock + APD gate + readout laser only.
    """
    b = PSBuilder()
    # APD gate during readout
    b.add_apd_gate(start=delay, dur=readout_ns)
    # Laser window
    b.add_laser_window(
        laser_name=readout_laser, start=delay, dur=readout_ns, power=readout_power
    )
    # Compute period and clock
    period = delay + readout_ns + tail_pad
    b.add_daq_clock_once(period)
    return b.emit(tail_pad=0)  # already included in 'period'


def macro_charge_init_simple_readout(
    init_ns: int,
    readout_ns: int,
    init_laser: str,
    init_power: float | None,
    readout_laser: str,
    readout_power: float | None,
    uwave_buffer: int = 0,
    tail_pad: int = 300,
) -> Tuple[Sequence, OutputState, List[int]]:
    """
    Charge initialization window → (optional buffer) → readout gate+laser.
    """
    b = PSBuilder()
    t = 0
    # init
    b.add_laser_window(init_laser, start=t, dur=init_ns, power=init_power)
    t += init_ns + uwave_buffer
    # APD + readout
    b.add_apd_gate(start=t, dur=readout_ns)
    b.add_laser_window(readout_laser, start=t, dur=readout_ns, power=readout_power)
    period = t + readout_ns + tail_pad
    b.add_daq_clock_once(period)
    return b.emit(tail_pad=0)


def macro_polarize(
    builder: PSBuilder,
    pol_coords_unused: List[List[float]] | None = None,
    duration_ns: int | None = None,
    amp_unused: float | None = None,
    duration_override: int | None = None,
    vkey: VirtualLaserKey = VirtualLaserKey.CHARGE_POL,
    start_ns: int = 0,
) -> int:
    """
    Simple polarization macro for single-NV confocal (no AOD stepping here).
    Returns end time.
    """
    laser = get_physical_laser_name(vkey)
    dur = (
        duration_override
        if duration_override is not None
        else (
            get_virtual_laser_dict(vkey)["duration"]
            if duration_ns is None
            else duration_ns
        )
    )
    builder.add_laser_window(laser, start=start_ns, dur=dur, power=None)
    return start_ns + dur


def macro_scc(
    builder: PSBuilder,
    duration_ns: int | None = None,
    duration_override: int | None = None,
    vkey: VirtualLaserKey = VirtualLaserKey.SCC,
    start_ns: int = 0,
    amp: float | None = None,
) -> int:
    """
    SCC pulse macro for single-NV.
    """
    laser = get_physical_laser_name(vkey)
    dur = (
        duration_override
        if duration_override is not None
        else (
            get_virtual_laser_dict(vkey)["duration"]
            if duration_ns is None
            else duration_ns
        )
    )
    builder.add_laser_window(laser, start=start_ns, dur=dur, power=amp)
    return start_ns + dur


def macro_charge_state_readout(
    builder: PSBuilder,
    duration_ns: int | None = None,
    amp: float | None = None,
    start_ns: int = 0,
) -> int:
    """
    Yellow charge readout (parallel in WF; here just a window) + APD gate.
    """
    vkey = VirtualLaserKey.WIDEFIELD_CHARGE_READOUT
    laser = get_physical_laser_name(vkey)
    dur = (
        get_virtual_laser_dict(vkey)["duration"] if duration_ns is None else duration_ns
    )
    builder.add_laser_window(laser, start=start_ns, dur=dur, power=amp)
    builder.add_apd_gate(start=start_ns, dur=dur)
    return start_ns + dur


def macro_pi_pulse(builder: PSBuilder, uwave_ind: int, start_ns: int, dur_ns: int):
    """
    Gate the microwave source for a duration (digital).
    You can also add IQ analogs similarly if you wire them to AO channels.
    """
    sig = get_sig_gen_name(uwave_ind)
    builder.add_mw_gate(sig_gen_name=sig, start=start_ns, dur=dur_ns)


# ---------------------------------------------------------------------
# Example: a "Rabi-like" base macro using the builder (signal + reference)
# ---------------------------------------------------------------------


def build_rabi_like_sequence(
    tau_ns: int,
    pol_ns: int,
    readout_ns: int,
    max_tau_ns: int,
    uwave_ind: int,
    readout_laser: str,
    readout_power: float | None,
) -> Tuple[Sequence, OutputState, List[int]]:
    """
    Structure: [pol] --buf--> [tau uwave] --buf--> [readout (signal)] --buf-->
               [tau ref path] --buf--> [readout (reference)]
    Buffers are pulled from CommonDurations where appropriate.
    """
    b = PSBuilder()
    uwave_buf = get_common_duration_ns("uwave_buffer")
    short_buf = 10  # to mirror OPX min waits
    # --- Signal path
    t = 0
    # polarization
    t = macro_polarize(b, duration_ns=pol_ns, start_ns=t)
    t += uwave_buf
    # wait so that total evolution between pol->readout is max_tau_ns
    if max_tau_ns < tau_ns:
        raise ValueError("max_tau_ns must be >= tau_ns")
    t += max_tau_ns - tau_ns
    # microwave window (tau)
    macro_pi_pulse(b, uwave_ind=uwave_ind, start_ns=t, dur_ns=tau_ns)
    t += tau_ns + uwave_buf
    # signal readout
    b.add_apd_gate(start=t, dur=readout_ns)
    b.add_laser_window(readout_laser, start=t, dur=readout_ns, power=readout_power)
    t += (
        max(readout_ns, pol_ns) + short_buf + uwave_buf
    )  # keep laser on long enough, mimic OPX

    # --- Reference path (no uwave)
    # fixed delay to mirror the time budget
    t += max_tau_ns + uwave_buf
    # reference readout
    b.add_apd_gate(start=t, dur=readout_ns)
    b.add_laser_window(readout_laser, start=t, dur=readout_ns, power=readout_power)
    t += readout_ns + short_buf

    period = t + 500  # small trailer
    b.add_daq_clock_once(period)
    return b.emit(tail_pad=0)


# ---------------------------------------------------------------------
# Legacy helpers brought from tool_belt so sequences can import only this file
# ---------------------------------------------------------------------


def process_laser_seq(pulse_streamer, seq, config, laser_name, laser_power, train):
    """
    Kept for backwards-compatibility with older sequence files.
    """
    if config is None:
        config = get_cfg()
    mod_mode = config["Optics"][laser_name]["mod_mode"]
    if mod_mode is ModMode.DIGITAL:
        seq.setDigital(W()[f"do_{laser_name}_dm"], train)
    else:
        processed = []
        for dur, level in train:
            val = (
                0.0
                if level is Digital.LOW
                else (0.0 if laser_power is None else float(laser_power))
            )
            processed.append((int(dur), val))
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
