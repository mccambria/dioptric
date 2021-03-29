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


def get_seq(pulser_wiring, args):

    # %% Parse wiring and args

    # The first 9 args are ns durations and we need them as int64s
    durations = []
    for ind in range(11):
        durations.append(numpy.int64(args[ind]))

    # Unpack the durations
    polarization_time, reference_time, signal_wait_time, \
        reference_wait_time, background_wait_time, \
        aom_delay_time, uwave_delay_time, iq_delay_time, \
        gate_time, uwave_pi_pulse, uwave_pi_on_2_pulse = durations
        
    num_pi_pulses = int(args[11])
    max_num_pi_pulses = int(args[12])

    # Get the APD indices
    apd_index = args[13]

    # Signify which signal generator to use
    state_value = args[14]

    # Get what we need out of the wiring dictionary
    key = 'do_apd_{}_gate'.format(apd_index)
    pulser_do_apd_gate = pulser_wiring[key]
    sig_gen_name = tool_belt.get_signal_generator_name(States(state_value))
    sig_gen_gate_chan_name = 'do_{}_gate'.format(sig_gen_name)
    pulser_do_sig_gen_gate = pulser_wiring[sig_gen_gate_chan_name]
    pulser_do_aom = pulser_wiring['do_532_aom']
    pulser_do_arb_wave_trigger = pulser_wiring['do_arb_wave_trigger']

    # %% Couple calculated values
    
    gap_time = 0
    half_gap_time = gap_time//2
    buffer = 0
    
    composite_pulse_time = buffer + uwave_pi_on_2_pulse + gap_time + uwave_pi_pulse + gap_time + uwave_pi_on_2_pulse + buffer
    
    tau = composite_pulse_time * num_pi_pulses
    max_tau = composite_pulse_time * max_num_pi_pulses

    prep_time = polarization_time + signal_wait_time + \
        tau + signal_wait_time
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
#    mid_duration = period - (pre_duration + (2 * gate_time) + post_duration)
    mid_duration = polarization_time + reference_wait_time - gate_time
    train = [(pre_duration, LOW),
             (gate_time, HIGH),
             (mid_duration, LOW),
             (gate_time, HIGH),
             (post_duration, LOW)]
    seq.setDigital(pulser_do_apd_gate, train)

    # # Ungate (high) the APD channel for the background
    # gateBackgroundTrain = [( AOMDelay + preparationTime + polarizationTime + referenceWaitTime + referenceTime + backgroundWaitTime, low),
    #                       (gateTime, high), (endRestTime - gateTime, low)]
    # pulserSequence.setDigital(pulserDODaqGate0, gateBackgroundTrain)

    # Pulse the laser with the AOM for polarization and readout
    train = [(polarization_time, HIGH),
             (signal_wait_time + tau + signal_wait_time, LOW),
             (polarization_time, HIGH),
             (reference_wait_time, LOW),
             (reference_time, HIGH),
             (background_wait_time + end_rest_time + aom_delay_time, LOW)]
    seq.setDigital(pulser_do_aom, train)

    # Microwave train
    composite_pulse = [(buffer, LOW), (uwave_pi_on_2_pulse, HIGH), (gap_time, LOW), (uwave_pi_pulse, HIGH), 
                       (gap_time, LOW), (uwave_pi_on_2_pulse, HIGH), (buffer, LOW)]
    pre_duration = aom_delay_time + polarization_time + signal_wait_time - uwave_delay_time
    post_duration = signal_wait_time + polarization_time + \
        reference_wait_time + reference_time + \
        background_wait_time + end_rest_time + uwave_delay_time
    train = [(pre_duration, LOW)]
    for i in range(num_pi_pulses):
        train.extend(composite_pulse)
    train.extend([(post_duration-100, LOW)])
    seq.setDigital(pulser_do_sig_gen_gate, train)
    
    # Switch the phase with the AWG
    composite_pulse = [(10, HIGH), (buffer + uwave_pi_on_2_pulse+half_gap_time-10, LOW), (10, HIGH), (uwave_pi_pulse+gap_time-10, LOW), (10, HIGH), (uwave_pi_on_2_pulse+half_gap_time-10 + buffer, LOW)]
    pre_duration = aom_delay_time + polarization_time + signal_wait_time - iq_delay_time - uwave_delay_time
    post_duration = signal_wait_time + polarization_time + \
        reference_wait_time + reference_time + \
        background_wait_time + end_rest_time + iq_delay_time + uwave_delay_time
    train = [(pre_duration, LOW)]
    for i in range(num_pi_pulses):
        train.extend(composite_pulse)
    train.extend([(post_duration, LOW)])
    seq.setDigital(pulser_do_arb_wave_trigger, train)

    final_digital = [pulser_wiring['do_532_aom'],
                     pulser_wiring['do_sample_clock']]
    final = OutputState(final_digital, 0.0, 0.0)
    return seq, final, [period]


if __name__ == '__main__':
    wiring = {'ao_589_aom': 1, 'ao_638_laser': 0, 'do_532_aom': 3,
              'do_638_laser': 7, 'do_apd_0_gate': 5, 'do_arb_wave_trigger': 6,
              'do_sample_clock': 0, 'do_signal_generator_tsg4104a_gate': 1,
              'do_signal_generator_sg394_gate': 4}
#    args = [0, 3000, 1000, 1000, 2000, 1000, 1000, 300, 150, 0, 3]
    # args = [12000, 1000, 1000, 2000, 1000, 1060, 1000, 555, 350, 92, 46, 1, 8, 0, 3]
    # args = [500, 1000, 1000, 2000, 1000, 0, 0, 0, 350, 92, 46, 0, 4, 0, 3]
    seq_args = [12000, 1000, 1000, 2000, 1000, 0, 0, 0, 350, 78, 39, 1, 1, 0, 3]
    seq = get_seq(wiring, seq_args)[0]
    seq.plot()
