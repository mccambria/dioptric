#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 9 20:40:44 2022

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
        uwave_tau_max,
        green_laser_name,
        yellow_laser_name,
        red_laser_name,
        state,
        reion_power,
        ion_power,
        readout_power,
    ) = args

    # Convert all times to int64
    readout_time = numpy.int64(readout_time)
    reion_time = numpy.int64(reion_time)
    ion_time = numpy.int64(ion_time)
    pi_pulse = numpy.int64(pi_pulse)
    uwave_tau_max = numpy.int64(uwave_tau_max)
    
    state = States(state)


    sig_gen_name = config['Servers']['sig_gen_{}'.format(state.name)]
    
    # Get the wait time between pulses
    uwave_buffer =  config["CommonDurations"]["uwave_buffer"]
    scc_ion_readout_buffer = config['CommonDurations']['scc_ion_readout_buffer']
    sig_ref_buffer = uwave_buffer

    # delays
    green_delay_time = config["Optics"][green_laser_name]["delay"]
    yellow_delay_time = config["Optics"][yellow_laser_name]["delay"]
    red_delay_time = config["Optics"][red_laser_name]["delay"]
    rf_delay_time = config["Microwaves"][sig_gen_name]["delay"]

    common_delay = (
        max(green_delay_time, yellow_delay_time, red_delay_time, rf_delay_time)
        + 100
    )

    # For rabi experiment, we want to have sequence take same amount of time
    # over each tau, so have some waittime after the readout to accoutn for this
    # +++ Artifact from rabi experiments, in determine SCC durations, this is 0
    post_wait_time = uwave_tau_max - pi_pulse


    # Get what we need out of the wiring dictionary
    pulser_wiring = config["Wiring"]["PulseGen"]
    pulser_do_apd_gate = pulser_wiring['do_apd_gate']
    pulser_do_clock = pulser_wiring["do_sample_clock"]
    sig_gen_gate_chan_name = "do_{}_gate".format(sig_gen_name)
    pulser_do_sig_gen_gate = pulser_wiring[sig_gen_gate_chan_name]

# %%
    seq = Sequence()

    # APD readout
    train = [
        (common_delay, LOW),
        # Signal
        (reion_time, LOW),
        (uwave_buffer, LOW),
        (pi_pulse, LOW),
        (uwave_buffer, LOW),
        (ion_time, LOW),
        (scc_ion_readout_buffer, LOW),
        (readout_time, HIGH),
        (post_wait_time, LOW),
        (sig_ref_buffer, LOW),
        # Reference
        (reion_time, LOW),
        (uwave_buffer, LOW),
        (pi_pulse, LOW),
        (uwave_buffer, LOW),
        (ion_time, LOW),
        (scc_ion_readout_buffer, LOW),
        (readout_time, HIGH),
        (post_wait_time, LOW),
        (sig_ref_buffer, LOW),
    ]
    seq.setDigital(pulser_do_apd_gate, train)
    period = 0
    for el in train:
        period += el[0]
    print(period)

    # Reionization pulse (green)
    delay = common_delay - green_delay_time
    train = [
        (delay, LOW),
        # Signal
        (reion_time, HIGH),
        (uwave_buffer, LOW),
        (pi_pulse, LOW),
        (uwave_buffer, LOW),
        (ion_time, LOW),
        (scc_ion_readout_buffer, LOW),
        (readout_time, LOW),
        (post_wait_time, LOW),
        (sig_ref_buffer, LOW),
        # Reference
        (reion_time, HIGH),
        (uwave_buffer, LOW),
        (pi_pulse, LOW),
        (uwave_buffer, LOW),
        (ion_time, LOW),
        (scc_ion_readout_buffer, LOW),
        (readout_time, LOW),
        (post_wait_time, LOW),
        (sig_ref_buffer, LOW),
        (green_delay_time, LOW)
    ]
    power_list = reion_power
    tool_belt.process_laser_seq(
        pulse_streamer, seq, config, green_laser_name,power_list , train
    )
    period = 0
    for el in train:
        period += el[0]
    print(period)

    # Ionization pulse (red)
    delay = common_delay - red_delay_time
    train = [
        (delay, LOW),
        # Signal
        (reion_time, LOW),
        (uwave_buffer, LOW),
        (pi_pulse, LOW),
        (uwave_buffer, LOW),
        (ion_time, HIGH),
        (scc_ion_readout_buffer, LOW),
        (readout_time, LOW),
        (post_wait_time, LOW),
        (sig_ref_buffer, LOW),
        # Reference
        (reion_time, LOW),
        (uwave_buffer, LOW),
        (pi_pulse, LOW),
        (uwave_buffer, LOW),
        (ion_time, HIGH),
        (scc_ion_readout_buffer, LOW),
        (readout_time, LOW),
        (post_wait_time, LOW),
        (sig_ref_buffer, LOW),
        (red_delay_time, LOW)
    ]
    power_list = [ion_power, ion_power]
    tool_belt.process_laser_seq(
        pulse_streamer, seq, config, red_laser_name, power_list, train
    )
    period = 0
    for el in train:
        period += el[0]
    print(period)

    # uwave pulses
    delay = common_delay - rf_delay_time
    train = [
        (delay, LOW),
        # Signal
        (reion_time, LOW),
        (uwave_buffer, LOW),
        (pi_pulse, HIGH),
        (uwave_buffer, LOW),
        (ion_time, LOW),
        (scc_ion_readout_buffer, LOW),
        (readout_time, LOW),
        (post_wait_time, LOW),
        (sig_ref_buffer, LOW),
        # Reference
        (reion_time, LOW),
        (uwave_buffer, LOW),
        (pi_pulse, LOW),
        (uwave_buffer, LOW),
        (ion_time, LOW),
        (scc_ion_readout_buffer, LOW),
        (readout_time, LOW),
        (post_wait_time, LOW),
        (sig_ref_buffer, LOW),
        (rf_delay_time, LOW)
    ]
    seq.setDigital(pulser_do_sig_gen_gate, train)
    period = 0
    for el in train:
        period += el[0]
    print(period)

    # Readout with yellow
    delay = common_delay - yellow_delay_time
    train = [
        (delay, LOW),
        # Signal
        (reion_time, LOW),
        (uwave_buffer, LOW),
        (pi_pulse, LOW),
        (uwave_buffer, LOW),
        (ion_time, LOW),
        (scc_ion_readout_buffer, LOW),
        (readout_time, HIGH),
        (post_wait_time, LOW),
        (sig_ref_buffer, LOW),
        # Reference
        (reion_time, LOW),
        (uwave_buffer, LOW),
        (pi_pulse, LOW),
        (uwave_buffer, LOW),
        (ion_time, LOW),
        (scc_ion_readout_buffer, LOW),
        (readout_time, HIGH),
        (post_wait_time, LOW),
        (sig_ref_buffer, LOW),
        (yellow_delay_time, LOW)
    ]
    # power_list= [shelf_power, readout_power, shelf_power, readout_power]
    # power_list= readout_power
    tool_belt.process_laser_seq(pulse_streamer, seq, config,
                                yellow_laser_name, 
                                [readout_power], train)
    period = 0
    for el in train:
        period += el[0]
    print(period)

    final_digital = [pulser_do_clock]
    final = OutputState(final_digital, 0.0, 0.0)

    return seq, final, [period]


if __name__ == "__main__":
    config = tool_belt.get_config_dict()
    tool_belt.set_delays_to_zero(config)
    # seq_args = [10000.0, 1000.0, 100, 50, 0, 50, 
    #             'integrated_520', 'laserglow_589', 'cobolt_638', 
    #             'signal_generator_sg394', 1, None, None, 1.0, 0.5]
    seq_args = [2000.0, 1000.0, 200, 41, 41, 'integrated_520', 'laser_LGLO_589', 'cobolt_638',
                1, None, None, None]
    seq = get_seq(None, config, seq_args)[0]
    seq.plot()