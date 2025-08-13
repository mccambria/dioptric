# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 17:39:27 2019

@author: mccambria
Updated by @Saroj Chand on August 6, 2025
"""

import numpy
import numpy as np
from pulsestreamer import OutputState, Sequence

import utils.tool_belt as tb
import utils.tool_belt as tool_belt
from servers.timing.sequencelibrary.pulse_gen_SWAB_82 import 
from utils.tool_belt import Digital, States


def macro(
    taus,
    pol_time,
    readout_time,
    max_tau,
    state,
    laser_name,
    laser_power,
    include_reference=True,
):
    """
    Create a full pulse streamer sequence for multiple taus, one per step.
    """
    wiring = config["Wiring"]["PulseGen"]
    apd_gate = wiring["do_apd_gate"]
    sig_gen_name = config["Servers"][f"sig_gen_{States(state).name}"]
    sig_gate = wiring[f"do_{sig_gen_name}_gate"]
    sample_clock = wiring["do_sample_clock"]

    laser_delay = config["Optics"][laser_name]["delay"]
    uwave_delay = config["Microwaves"][sig_gen_name]["delay"]
    uwave_buffer = config["CommonDurations"]["uwave_buffer"]
    short_buffer = 10
    final_readout_buffer = 500
    common_delay = max(laser_delay, uwave_delay) + short_buffer
    readout_pol_max = max(readout_time, pol_time) + short_buffer

    master_seq = Sequence()

    for tau in taus:
        # === APD Gate ===
        apd_train = [
            (common_delay, Digital.LOW),
            (pol_time, Digital.LOW),
            (uwave_buffer, Digital.LOW),
            (max_tau, Digital.LOW),
            (uwave_buffer, Digital.LOW),
            (readout_time, Digital.HIGH),
            (readout_pol_max - readout_time, Digital.LOW),
            (uwave_buffer, Digital.LOW),
            (max_tau, Digital.LOW),
            (uwave_buffer, Digital.LOW),
            (readout_time, Digital.HIGH),
            (final_readout_buffer + short_buffer, Digital.LOW),
        ]
        seq = Sequence()
        seq.setDigital(apd_gate, apd_train)

        # === Laser ===
        laser_train = [
            (common_delay - laser_delay, Digital.LOW),
            (pol_time, Digital.HIGH),
            (uwave_buffer, Digital.LOW),
            (max_tau, Digital.LOW),
            (uwave_buffer, Digital.LOW),
            (readout_pol_max, Digital.HIGH),
            (uwave_buffer, Digital.LOW),
            (max_tau, Digital.LOW),
            (uwave_buffer, Digital.LOW),
            (readout_time + final_readout_buffer, Digital.HIGH),
            (short_buffer, Digital.LOW),
            (laser_delay, Digital.LOW),
        ]
        tb.process_laser_seq(seq, config, laser_name, laser_power, laser_train)

        # === MW pulse ===
        microwave_pi_pulse(
            seq,
            sig_gate,
            tau,
            max_tau,
            pol_time,
            readout_pol_max,
            uwave_delay,
            uwave_buffer,
        )

        master_seq.append(seq)

        if include_reference:
            master_seq.append(
                build_reference_sequence(
                    config,
                    max_tau,
                    pol_time,
                    readout_time,
                    state,
                    laser_name,
                    laser_power,
                    sig_gate,
                    apd_gate,
                    sample_clock,
                )
            )

    final = OutputState([sample_clock], 0.0, 0.0)
    return master_seq, final, None


def get_seq(pulse_streamer, config, args):
    # The first 9 args are ns durations and we need them as int64s
    durations = []
    for ind in range(4):
        durations.append(numpy.int64(args[ind]))

    # Unpack the durations
    tau, polarization_time, readout, max_tau = durations

    # Signify which signal generator to use
    state = args[4]
    state = States(state)
    sig_gen_name = config["Servers"][f"sig_gen_{state.name}"]

    # Laser specs
    laser_name = args[5]
    laser_power = args[6]

    # Get what we need out of the wiring dictionary
    pulser_wiring = config["Wiring"]["PulseGen"]

    pulser_do_apd_gate = pulser_wiring["do_apd_gate"]
    sig_gen_gate_chan_name = "do_{}_gate".format(sig_gen_name)
    pulser_do_sig_gen_gate = pulser_wiring[sig_gen_gate_chan_name]

    # Get the other durations we need
    # print(laser_name)
    laser_delay = config["Optics"][laser_name]["delay"]
    uwave_delay = config["Microwaves"][sig_gen_name]["delay"]
    short_buffer = 10  # Helps avoid weird things that happen for ~0 ns pulses
    common_delay = max(laser_delay, uwave_delay) + short_buffer
    uwave_buffer = config["CommonDurations"]["uwave_buffer"]
    # uwave_buffer = 1000
    # Keep the laser on for only as long as we need
    readout_pol_max = max(readout, polarization_time) + short_buffer
    final_readout_buffer = 500

    # %% Define the sequence

    seq = Sequence()

    # APD gating - first high is for signal, second high is for reference
    train = [
        (common_delay, Digital.LOW),
        (polarization_time, Digital.LOW),
        (uwave_buffer, Digital.LOW),
        (max_tau, Digital.LOW),
        (uwave_buffer, Digital.LOW),
        (readout, Digital.HIGH),
        (readout_pol_max - readout, Digital.LOW),
        (uwave_buffer, Digital.LOW),
        (max_tau, Digital.LOW),
        (uwave_buffer, Digital.LOW),
        (readout, Digital.HIGH),
        (final_readout_buffer + short_buffer, Digital.LOW),
    ]
    seq.setDigital(pulser_do_apd_gate, train)
    period = 0
    for el in train:
        period += el[0]
    print(period)

    # Laser for polarization and readout
    train = [
        (common_delay - laser_delay, Digital.LOW),
        (polarization_time, Digital.HIGH),
        (uwave_buffer, Digital.LOW),
        (max_tau, Digital.LOW),
        (uwave_buffer, Digital.LOW),
        (readout_pol_max, Digital.HIGH),
        (uwave_buffer, Digital.LOW),
        (max_tau, Digital.LOW),
        (uwave_buffer, Digital.LOW),
        (readout + final_readout_buffer, Digital.HIGH),
        (short_buffer, Digital.LOW),
        (laser_delay, Digital.LOW),
    ]
    tool_belt.process_laser_seq(
        pulse_streamer, seq, config, laser_name, laser_power, train
    )
    total_dur = 0
    for el in train:
        total_dur += el[0]
    print(total_dur)

    # Pulse the microwave for tau
    train = [
        (common_delay - uwave_delay, Digital.LOW),
        (polarization_time, Digital.LOW),
        (uwave_buffer, Digital.LOW),
        (max_tau - tau, Digital.LOW),
        (tau, Digital.HIGH),
        (uwave_buffer, Digital.LOW),
        (readout_pol_max, Digital.LOW),
        (uwave_buffer, Digital.LOW),
        (max_tau, Digital.LOW),
        (uwave_buffer, Digital.LOW),
        (readout + final_readout_buffer, Digital.LOW),
        (short_buffer, Digital.LOW),
        (uwave_delay, Digital.LOW),
    ]
    seq.setDigital(pulser_do_sig_gen_gate, train)
    # print(train)
    total_dur = 0
    for el in train:
        total_dur += el[0]
    print(total_dur)

    final_digital = [pulser_wiring["do_sample_clock"]]
    final = OutputState(final_digital, 0.0, 0.0)
    return seq, final, [period]


if __name__ == "__main__":
    config = tool_belt.get_config_dict()
    tool_belt.set_delays_to_zero(config)
    args = [100, 1000.0, 300, 300, 3, "laserglow_532", None]
    # args = [1000, 10000.0, 300, 2000, 3, 'integrated_520', None]
    seq = get_seq(None, config, args)[0]
    seq.plot()
