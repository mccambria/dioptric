# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 17:39:27 2019

@author: mccambria
"""

from pulsestreamer import Sequence
from pulsestreamer import OutputState
import numpy
import utils.tool_belt as tool_belt
from utils.tool_belt import States

LOW = 0
HIGH = 1


def get_seq(pulse_streamer, config, args):

    # %% Parse wiring and args

    # The first 9 args are ns durations and we need them as int64s
    durations = []
    for ind in range(4):
        durations.append(numpy.int64(args[ind]))
        
    # Unpack the durations
    tau, polarization_time, readout, max_tau = durations

    # Get the APD indices
    apd_index = args[4]

    # Signify which signal generator to use
    state = args[5]
    state = States(state)
    sig_gen_name = config['Servers']['sig_gen_{}'.format(state.name)]
    
    # Laser specs
    laser_name = args[6]
    laser_power = args[7]

    # Get what we need out of the wiring dictionary
    pulser_wiring = config['Wiring']['PulseGen']
    key = 'do_apd_{}_gate'.format(apd_index)
    pulser_do_apd_gate = pulser_wiring[key]
    sig_gen_gate_chan_name = 'do_{}_gate'.format(sig_gen_name)
    pulser_do_sig_gen_gate = pulser_wiring[sig_gen_gate_chan_name]
    pulser_do_arb_wave_trigger = pulser_wiring['do_arb_wave_trigger']

    # Get the other durations we need
    laser_delay =  config['Optics'][laser_name]['delay']
    uwave_delay = config['Microwaves'][sig_gen_name]['delay']
    iq_trigger_delay = config['Microwaves']['iq_delay']
    short_buffer = 10  # Helps avoid weird things that happen for ~0 ns pulses 
    iq_trigger_time = 10
    common_delay = max(laser_delay, uwave_delay, iq_trigger_delay) + short_buffer
    uwave_buffer = config['CommonDurations']['uwave_buffer']
    half_uwave_buffer = int(uwave_buffer/2)
    
    # Keep the laser on for only as long as we need
    readout_pol_min = max(readout, polarization_time) + short_buffer
    final_readout_buffer = 500

    # %% Define the sequence

    seq = Sequence()

    # APD gating - first high is for signal, second high is for reference
    train = [(common_delay, LOW),
             (polarization_time, LOW),
             (uwave_buffer, LOW),
             (max_tau, LOW),
             (uwave_buffer, LOW),
             (readout, HIGH), 
             (readout_pol_min - readout, LOW),
             (uwave_buffer, LOW),
             (max_tau, LOW),
             (uwave_buffer, LOW),
             (readout, HIGH),
             (final_readout_buffer + short_buffer, LOW)]
    seq.setDigital(pulser_do_apd_gate, train)
    period = 0
    for el in train:
        period += el[0]
    print(period)

    # Laser for polarization and readout
    train = [(common_delay - laser_delay, LOW),
             (polarization_time, HIGH),
             (uwave_buffer, LOW),
             (max_tau, LOW),
             (uwave_buffer, LOW),
             (readout_pol_min, HIGH), 
             (uwave_buffer, LOW),
             (max_tau, LOW),
             (uwave_buffer, LOW),
             (readout + final_readout_buffer, HIGH),
             (short_buffer, LOW),
             (laser_delay, LOW)]
    tool_belt.process_laser_seq(pulse_streamer, seq, config,
                                laser_name, laser_power, train)
    total_dur = 0
    for el in train:
        total_dur += el[0]
    print(total_dur)

    # Pulse the microwave for tau
    train = [(common_delay - uwave_delay, LOW),
             (polarization_time, LOW),
             (uwave_buffer, LOW),
             (max_tau-tau, LOW),
             (tau, HIGH),
             (uwave_buffer, LOW),
             (readout_pol_min, LOW), 
             (uwave_buffer, LOW), 
             (max_tau, LOW),
             (uwave_buffer, LOW),
             (readout + final_readout_buffer, LOW),
             (short_buffer, LOW),
             (uwave_delay, LOW)]
    seq.setDigital(pulser_do_sig_gen_gate, train)
    total_dur = 0
    for el in train:
        total_dur += el[0]
    print(total_dur)
    
    # IQ modulation
    train = [(common_delay - iq_trigger_delay, LOW),
             (iq_trigger_time, HIGH),
             (polarization_time - iq_trigger_time, LOW),
             (half_uwave_buffer, LOW),
             (iq_trigger_time, HIGH),
             (uwave_buffer - half_uwave_buffer - iq_trigger_time, LOW),
             (max_tau, LOW),
             (uwave_buffer, LOW),
             (iq_trigger_time, HIGH),
             (readout_pol_min - iq_trigger_time, LOW), 
             (uwave_buffer, LOW), 
             (max_tau, LOW),
             (uwave_buffer, LOW),
             (readout + final_readout_buffer, LOW),
             (short_buffer, LOW),
             (uwave_delay, LOW)]
    seq.setDigital(pulser_do_arb_wave_trigger, train)
    total_dur = 0
    for el in train:
        total_dur += el[0]
    print(total_dur)

    final_digital = [pulser_wiring['do_sample_clock']]
    final = OutputState(final_digital, 0.0, 0.0)
    return seq, final, [period]


if __name__ == '__main__':
    config = tool_belt.get_config_dict()
    tool_belt.set_delays_to_zero(config)
    args = [100, 10000.0, 300, 300, 1, 3, 'integrated_520', None]
    seq = get_seq(None, config, args)[0]
    seq.plot()
