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
    for ind in range(5):
        durations.append(numpy.int64(args[ind]))

    # Unpack the durations
    polarization_time, iq_delay_time, gate_time, uwave_pi_pulse, uwave_pi_on_2_pulse = durations
        
    uwave_to_readout_time = config['CommonDurations']['uwave_buffer']
    signal_wait_time = uwave_to_readout_time
    reference_time = signal_wait_time  # not sure what this is
    background_wait_time = signal_wait_time  # not sure what this is
    reference_wait_time = 2 * signal_wait_time  # not sure what this is
        
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
    pulser_wiring = config['Wiring']['PulseStreamer']
    key = 'do_apd_{}_gate'.format(apd_index)
    pulser_do_apd_gate = pulser_wiring[key]
    state = States(state)
    sig_gen_name = config['Microwaves']['sig_gen_{}'.format(state.name)]
    sig_gen_gate_chan_name = 'do_{}_gate'.format(sig_gen_name)
    pulser_do_sig_gen_gate = pulser_wiring[sig_gen_gate_chan_name]
    pulser_do_arb_wave_trigger = pulser_wiring['do_arb_wave_trigger']
    
    # Delays
    aom_delay_time = config['Optics'][laser_name]['delay']
    uwave_delay_time = config['Microwaves'][sig_gen_name]['delay']

    # %% Couple calculated values
    
    composite_pulse_time = 5 * uwave_pi_pulse 
    
    tau = composite_pulse_time * num_pi_pulses
    max_tau = composite_pulse_time * max_num_pi_pulses

    prep_time = polarization_time + signal_wait_time + \
        tau + uwave_to_readout_time
    end_rest_time = max_tau - tau

    # The period is independent of the particular tau, but it must be long
    # enough to accomodate the longest tau
    period = aom_delay_time + polarization_time + reference_wait_time + \
        reference_wait_time + polarization_time + reference_wait_time + \
        reference_time + max_tau

    # %% Define the sequence

    seq = Sequence()

    # APD gating - first high is for signal, second high is for reference
    pre_duration = aom_delay_time + prep_time
    post_duration = reference_time - gate_time + \
        background_wait_time + end_rest_time
    mid_duration = polarization_time + reference_wait_time - gate_time
    train = [(pre_duration, LOW),
             (gate_time, HIGH),
             (mid_duration, LOW),
             (gate_time, HIGH),
             (post_duration, LOW)]
    seq.setDigital(pulser_do_apd_gate, train)

    # Laser
    train = [(polarization_time, HIGH),
             (signal_wait_time + tau + uwave_to_readout_time, LOW),
             (polarization_time, HIGH),
             (reference_wait_time, LOW),
             (reference_time, HIGH),
             (background_wait_time + end_rest_time + aom_delay_time, LOW)]
    tool_belt.process_laser_seq(pulse_streamer, seq, config, 
                                laser_name, laser_power, train)

    # Microwave train
    pre_duration = aom_delay_time + polarization_time + signal_wait_time - uwave_delay_time
    post_duration = uwave_to_readout_time + polarization_time + \
        reference_wait_time + reference_time + \
        background_wait_time + end_rest_time + uwave_delay_time
    train = [(pre_duration, LOW), (tau, HIGH), (post_duration, LOW)]
    seq.setDigital(pulser_do_sig_gen_gate, train)
    
    # Switch the phase with the AWG
    composite_pulse = [(10, HIGH), (uwave_pi_pulse-10, LOW)] * 5
    pre_duration = aom_delay_time + polarization_time + signal_wait_time - iq_delay_time
    post_duration = uwave_to_readout_time + polarization_time + \
        reference_wait_time + reference_time + \
        background_wait_time + end_rest_time + iq_delay_time
    train = [(pre_duration, LOW)]
    for i in range(num_pi_pulses):
        train.extend(composite_pulse)
    train.extend([(post_duration, LOW)])
    seq.setDigital(pulser_do_arb_wave_trigger, train)

    final_digital = [pulser_wiring['do_sample_clock']]
    final = OutputState(final_digital, 0.0, 0.0)
    return seq, final, [period]


if __name__ == '__main__':
#    args = [0, 3000, 1000, 1000, 2000, 1000, 1000, 300, 150, 0, 3]
    # seq_args = [12000, 1000, 1000, 2000, 1000, 1060, 1000, 555, 350, 92, 46, 8, 8, 0, 3]
    # args = [500, 1000, 1000, 2000, 1000, 0, 0, 0, 350, 92, 46, 0, 4, 0, 3]
    # seq_args = [1200, 1000, 1000, 10000, 1000, 0, 0, 0, 350, 78, 39, 1, 1, 0, 3]
    config = tool_belt.get_config_dict()
    tool_belt.set_delays_to_zero(config)
    # seq_args = [0, 1000.0, 350, 23, 12, 100000, 1, 2, 'integrated_520', None]
    seq_args = [5000, 0, 1000, 2000, 1000, 1, 3, 0,  3, 'integrated_520', None]
    seq, final, ret_vals = get_seq(None, config, seq_args)
    seq.plot()
