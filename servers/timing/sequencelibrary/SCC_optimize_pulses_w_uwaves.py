#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 20:40:44 2020

@author: agardill
"""

from pulsestreamer import Sequence
from pulsestreamer import OutputState
import utils.tool_belt as tool_belt
from utils.tool_belt import States
import numpy

LOW = 0
HIGH = 1


def get_seq(pulse_streamer, config, args):

    # Unpack the args
    (
        readout_time,
        reion_time,
        ion_time,
        pi_pulse,
        shelf_time,
        uwave_tau_max,
        green_laser_name,
        yellow_laser_name,
        red_laser_name,
        sig_gen_name,
        apd_indices,
        readout_power,
        shelf_power,
    ) = args

    # Convert all times to int64
    readout_time = numpy.int64(readout_time)
    reion_time = numpy.int64(reion_time)
    ion_time = numpy.int64(ion_time)
    pi_pulse = numpy.int64(pi_pulse)
    shelf_time = numpy.int64(shelf_time)
    uwave_tau_max = numpy.int64(uwave_tau_max)

    # Get the wait time between pulses
    wait_time = config["CommonDurations"]["uwave_buffer"]

    # delays
    green_delay_time = config["Optics"][green_laser_name]["delay"]
    yellow_delay_time = config["Optics"][yellow_laser_name]["delay"]
    red_delay_time = config["Optics"][red_laser_name]["delay"]
    rf_delay_time = config["Microwaves"][sig_gen_name]["delay"]
    # green_delay_time = 0
    # yellow_delay_time = 0
    # red_delay_time = 0
    # rf_delay_time = 0

    common_delay = (
        max(green_delay_time, yellow_delay_time, red_delay_time, rf_delay_time)
        + 100
    )

    # For rabi experiment, we want to have sequence take same amount of time
    # over each tau, so have some waittime after the readout to accoutn for this
    # +++ Artifact from rabi experiments, in determine SCC durations, this is 0
    post_wait_time = uwave_tau_max - pi_pulse
    # Test period
    period = (
        common_delay
        + (
            reion_time
            + ion_time
            + shelf_time
            + pi_pulse
            + readout_time
            + post_wait_time
            + 4 * wait_time
        )
        * 2
    )

    # Get what we need out of the wiring dictionary
    pulser_wiring = config["Wiring"]["PulseStreamer"]
    pulser_do_apd_gate = pulser_wiring["do_apd_{}_gate".format(apd_indices)]
    pulser_do_clock = pulser_wiring["do_sample_clock"]
    analog_key = "ao_{}_am".format(yellow_laser_name)
    digital_key = "do_{}_dm".format(yellow_laser_name)
    analog_yellow = analog_key in pulser_wiring
    if analog_yellow:
        pulser_ao_589_aom = pulser_wiring[analog_key]
    else:
        pulser_do_589_dm = pulser_wiring[digital_key]
    sig_gen_gate_chan_name = "do_{}_gate".format(sig_gen_name)
    pulser_do_sig_gen_gate = pulser_wiring[sig_gen_gate_chan_name]

    # Make sure the ao_aom voltage to the 589 aom is within 0 and 1 V
    if readout_power is not None:
        tool_belt.aom_ao_589_pwr_err(readout_power)
    if shelf_power is not None:
        tool_belt.aom_ao_589_pwr_err(shelf_power)

    seq = Sequence()

    # collect photons for certain timewindow tR in APD
    train = [
        (
            common_delay
            + reion_time
            + pi_pulse
            + shelf_time
            + ion_time
            + 3 * wait_time,
            LOW,
        ),
        (readout_time, HIGH),
        (
            post_wait_time
            + reion_time
            + pi_pulse
            + shelf_time
            + ion_time
            + 4 * wait_time,
            LOW,
        ),
        (readout_time, HIGH),
        (post_wait_time + wait_time, LOW),
    ]
    seq.setDigital(pulser_do_apd_gate, train)

    # reionization pulse (green)
    # MCC
    delay = common_delay - green_delay_time
    train = [(delay, LOW), (reion_time, HIGH),
              (4*wait_time + post_wait_time + pi_pulse + shelf_time + ion_time + readout_time, LOW),
              (reion_time, HIGH),
              (4*wait_time + post_wait_time + pi_pulse + shelf_time + ion_time + readout_time + green_delay_time, LOW)]
    # train = [(delay, LOW),
    #           (reion_time, HIGH),
    #           (wait_time + pi_pulse + wait_time, LOW),
    #           (shelf_time, HIGH),
    #           (ion_time, LOW),
    #           (wait_time, LOW),
    #           (readout_time, LOW),
    #           (post_wait_time + wait_time, LOW),
    #           (reion_time, HIGH),
    #           (wait_time + pi_pulse + wait_time, LOW),
    #           (shelf_time, HIGH),
    #           (ion_time, LOW),
    #           (wait_time, LOW),
    #           (readout_time, LOW),
    #           (post_wait_time + wait_time + green_delay_time, LOW)]
    tool_belt.process_laser_seq(pulse_streamer, seq, config,
                                green_laser_name, None, train)

    # ionization pulse (red)
    delay = common_delay - red_delay_time
    train = [
        (delay + 2 * wait_time + reion_time + pi_pulse + shelf_time, LOW),
        (ion_time, HIGH),
        (
            4 * wait_time
            + post_wait_time
            + readout_time
            + reion_time
            + pi_pulse
            + shelf_time,
            LOW,
        ),
        (ion_time, HIGH),
        (2 * wait_time + post_wait_time + readout_time + red_delay_time, LOW),
    ]
    tool_belt.process_laser_seq(
        pulse_streamer, seq, config, red_laser_name, None, train
    )

    # uwave pulses
    delay = common_delay - rf_delay_time
    train = [
        (delay + reion_time + wait_time, LOW),
        (pi_pulse, HIGH),
        (
            7 * wait_time
            + 2 * shelf_time
            + pi_pulse
            + reion_time
            + 2 * post_wait_time
            + 2 * readout_time
            + 2 * ion_time
            + rf_delay_time,
            LOW,
        ),
    ]
    seq.setDigital(pulser_do_sig_gen_gate, train)

    # readout with 589
    # Dummy values for digital modulation
    # shelf_power = 0.0  # MCC
    if not analog_yellow:
        shelf_power = HIGH
        readout_power = HIGH
    delay = common_delay - yellow_delay_time
    train = [
        (delay + reion_time + 2 * wait_time + pi_pulse, LOW),
        (shelf_time, shelf_power),
        (ion_time, LOW),
        (wait_time, LOW),
        (readout_time, readout_power),
        (post_wait_time + 3 * wait_time + reion_time + pi_pulse, LOW),
        (shelf_time, shelf_power),
        (ion_time, LOW),
        (wait_time, LOW),
        (readout_time, readout_power),
        (post_wait_time + wait_time + yellow_delay_time, LOW),
    ]
    if analog_yellow:
        seq.setAnalog(pulser_ao_589_aom, train)
    else:
        seq.setDigital(pulser_do_589_dm, train)
    # tool_belt.process_laser_seq(pulse_streamer, seq, config,
    #                             yellow_laser_name, readout_power, train)

    final_digital = [pulser_do_clock]
    final = OutputState(final_digital, 0.0, 0.0)

    return seq, final, [str(period)]


if __name__ == "__main__":
    config = tool_belt.get_config_dict()
    config["Optics"]["cobolt_638"]["feedthrough"] = "False"
    seq_args = [
        1500.0,
        1000.0,
        100,
        68,
        0,
        68,
        "laserglow_532",
        "laserglow_589",
        "cobolt_638",
        "signal_generator_sg394",
        1,
        0.68,
        1.0,
    ]
    seq = get_seq(None, config, seq_args)[0]
    seq.plot()
