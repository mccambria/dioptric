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
    for ind in range(10):
        durations.append(numpy.int64(args[ind]))

    # Unpack the durations
    tau, polarization_time, reference_time, signal_wait_time, \
        reference_wait_time, background_wait_time, \
        aom_delay_time, uwave_delay_time, \
        gate_time, max_tau = durations

    # Get the APD indices
    apd_index = args[10]

    # Signify which signal generator to use
    sig_gen_name = args[11]
    
    # Laser specs
    laser_name = args[12]
    laser_power = args[13]

    # Get what we need out of the wiring dictionary
    key = 'do_apd_{}_gate'.format(apd_index)
    pulser_do_apd_gate = pulser_wiring[key]
    sig_gen_gate_chan_name = 'do_{}_gate'.format(sig_gen_name)
    pulser_do_sig_gen_gate = pulser_wiring[sig_gen_gate_chan_name]

    # %% Couple calculated values

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
    if laser_power == -1:
        laser_high = HIGH
        laser_low = LOW
    else:
        laser_high = laser_power
        laser_low = 0.0
    train = [(polarization_time, laser_high),
             (signal_wait_time + tau + signal_wait_time, laser_low),
             (polarization_time, laser_high),
             (reference_wait_time, laser_low),
             (reference_time, laser_high),
             (background_wait_time + end_rest_time + aom_delay_time, laser_low)]
    if laser_power == -1:
        pulser_laser_mod = pulser_wiring['do_{}_dm'.format(laser_name)]
        seq.setDigital(pulser_laser_mod, train)
    else:
        pulser_laser_mod = pulser_wiring['ao_{}_am'.format(laser_name)]
        seq.setAnalog(pulser_laser_mod, train)

    # Pulse the microwave for tau
    pre_duration = aom_delay_time + polarization_time + signal_wait_time - uwave_delay_time
    post_duration = signal_wait_time + polarization_time + \
        reference_wait_time + reference_time + \
        background_wait_time + end_rest_time + uwave_delay_time
    train = [(pre_duration, LOW), (tau, HIGH), (post_duration, LOW)]
    seq.setDigital(pulser_do_sig_gen_gate, train)

    final_digital = [pulser_wiring['do_sample_clock']]
    final = OutputState(final_digital, 0.0, 0.0)
    return seq, final, [period]


if __name__ == '__main__':
    wiring = {'ao_589_aom': 1, 'ao_638_laser': 0, 'do_laser_515_dm': 3,
              'do_638_laser': 7, 'do_apd_0_gate': 5, 'do_arb_wave_trigger': 6,
              'do_sample_clock': 0, 'do_signal_generator_tsg4104a_gate': 1,
              'do_signal_generator_sg394_gate': 4}
    args = [100, 1000.0, 1000, 1000, 2000, 1000, 0, 0, 350, 400,
            0, 'signal_generator_sg394', 'laser_515', -1]
    seq = get_seq(wiring, args)[0]
    seq.plot()
