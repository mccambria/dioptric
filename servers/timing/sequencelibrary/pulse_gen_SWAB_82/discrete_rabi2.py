# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 17:39:27 2019

@author: mccambria
"""

from json import tool
from pulsestreamer import Sequence
from pulsestreamer import OutputState
import numpy
import utils.tool_belt as tool_belt
from utils.tool_belt import States

LOW = 0
HIGH = 1


def get_seq(pulse_streamer, config, args):

    ### Parse wiring and args

    # The first 9 args are ns durations and we need them as int64s
    durations = []
    for ind in range(5):
        durations.append(numpy.int64(args[ind]))

    # Unpack the durations
    (
        polarization_time,
        iq_delay,
        readout,
        uwave_pi_pulse,
        uwave_pi_on_2_pulse,
    ) = durations

    uwave_buffer = config["CommonDurations"]["uwave_buffer"]

    num_pi_pulses = int(args[5])
    max_num_pi_pulses = int(args[6])

    # Get the APD indices
    apd_index = args[7]

    # Signify which signal generator to use
    state = args[8]

    # Laser specs
    laser_name = args[9]
    laser_power = args[10]

    # Get what we need out of the wiring dictionary
    pulser_wiring = config["Wiring"]["PulseStreamer"]
    key = "do_apd_{}_gate".format(apd_index)
    pulser_do_apd_gate = pulser_wiring[key]
    state = States(state)
    sig_gen_name = config["Microwaves"]["sig_gen_{}".format(state.name)]
    sig_gen_gate_chan_name = "do_{}_gate".format(sig_gen_name)
    pulser_do_sig_gen_gate = pulser_wiring[sig_gen_gate_chan_name]
    pulser_do_arb_wave_trigger = pulser_wiring["do_arb_wave_trigger"]

    # Delays
    laser_delay = config["Optics"][laser_name]["delay"]
    uwave_delay = config["Microwaves"][sig_gen_name]["delay"]
    common_delay = max(laser_delay, uwave_delay, iq_delay) + 100

    ### Couple calculated values

    composite_pulse_time = 5 * uwave_pi_pulse

    tau = composite_pulse_time * num_pi_pulses
    max_tau = composite_pulse_time * max_num_pi_pulses
    max_tau_remainder = max_tau - tau

    # Period is independent of a particular tau and long enough for the longest tau
    period = (
        common_delay
        + polarization_time
        + uwave_buffer
        + tau
        + uwave_buffer
        + polarization_time
        + max_tau_remainder
        + uwave_buffer
        + tau
        + uwave_buffer
        + readout
        + max_tau_remainder
    )

    ### Define the sequence

    seq = Sequence()

    # APD gating
    train = [
        (common_delay, LOW),
        (polarization_time, LOW),
        (uwave_buffer, LOW),
        (tau, LOW),
        (uwave_buffer, LOW),
        (readout, HIGH),
        (polarization_time - readout, LOW),
        (max_tau_remainder, LOW),
        (uwave_buffer, LOW),
        (tau, LOW),
        (uwave_buffer, LOW),
        (readout, HIGH),
        (max_tau_remainder, LOW),
    ]
    seq.setDigital(pulser_do_apd_gate, train)

    # Laser
    train = [
        (common_delay - laser_delay, HIGH),
        (polarization_time, HIGH),
        (uwave_buffer, LOW),
        (tau, LOW),
        (uwave_buffer, LOW),
        (polarization_time, HIGH),
        (max_tau_remainder, LOW),
        (uwave_buffer, LOW),
        (tau, LOW),
        (uwave_buffer, LOW),
        (readout, HIGH),
        (max_tau_remainder, HIGH),
        (laser_delay, HIGH),
    ]
    tool_belt.process_laser_seq(
        pulse_streamer, seq, config, laser_name, laser_power, train
    )

    # Microwave train, first run is signal, second run is ref
    train = [
        (common_delay - uwave_delay, LOW),
        (polarization_time, LOW),
        (uwave_buffer, LOW),
        (tau, HIGH),
        (uwave_buffer, LOW),
        (polarization_time, LOW),
        (max_tau_remainder, LOW),
        (uwave_buffer, LOW),
        (tau, LOW),
        (uwave_buffer, LOW),
        (readout, LOW),
        (max_tau_remainder, LOW),
        (laser_delay, LOW),
    ]
    seq.setDigital(pulser_do_sig_gen_gate, train)

    # Switch the phase with the AWG
    composite_pulse = [(10, HIGH), (uwave_pi_pulse - 10, LOW)] * 5
    train = [
        (common_delay - iq_delay, LOW),
        (polarization_time, LOW),
        (uwave_buffer, LOW),
    ]
    for _ in range(num_pi_pulses):
        train.extend(composite_pulse)
    train.extend(
        [
            (uwave_buffer, LOW),
            (polarization_time, LOW),
            (max_tau_remainder, LOW),
            (uwave_buffer, LOW),
            (tau, LOW),
            (uwave_buffer, LOW),
            (readout, LOW),
            (max_tau_remainder, LOW),
            (iq_delay, LOW),
        ]
    )
    seq.setDigital(pulser_do_arb_wave_trigger, train)

    final_digital = [pulser_wiring["do_sample_clock"]]
    final = OutputState(final_digital, 0.0, 0.0)
    return seq, final, [period]


if __name__ == "__main__":
    # fmt: off
    seq_args =[100000.0, 0, 6000.0, 162, 81, 2, 6, 0, 1, 'laserglow_532', None]
    # fmt: on
    config = tool_belt.get_config_dict()
    # print(config)
    tool_belt.set_delays_to_zero(config)
    # seq_args = [0, 1000.0, 350, 23, 12, 100000, 1, 2, 'integrated_520', None]
    seq_args = [5000, 0, 1000, 2000, 1000, 0, 3, 0,  3, 'integrated_520', None]
    seq, final, ret_vals = get_seq(None, config, seq_args)
    seq.plot()
