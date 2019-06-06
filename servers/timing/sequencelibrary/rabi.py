# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 17:39:27 2019

@author: mccambria
"""

from pulsestreamer import Sequence
import numpy

LOW = 0
HIGH = 1


def get_seq(pulser_wiring, args):

    # %% Parse wiring and args

    # The first 9 args are ns durations and we need them as int64s
    durations = []
    for ind in range(9):
        durations.append(numpy.int64(args[ind]))

    # Unpack the durations
    tau, polarization_time, reference_time, signal_wait_time, \
        reference_wait_time, background_wait_time, aom_delay_time, \
        gate_time, max_tau = durations

    # Get the APD indices
    sig_apd_index, ref_apd_index = args[9:11]

    #Signify which signal generator to use
    do_uwave_gate = args[11]

    # Get what we need out of the wiring dictionary
    key = 'do_apd_gate_{}'.format(sig_apd_index)
    pulser_do_sig_apd_gate = pulser_wiring[key]
    key = 'do_apd_gate_{}'.format(ref_apd_index)
    pulser_do_ref_apd_gate = pulser_wiring[key]
    if do_uwave_gate == 0:
        pulser_do_uwave = pulser_wiring['do_uwave_gate_0']
    if do_uwave_gate == 1:
        pulser_do_uwave = pulser_wiring['do_uwave_gate_1']
    pulser_do_aom = pulser_wiring['do_aom']

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

    # Signal APD gate
    pre_duration = aom_delay_time + prep_time
    post_duration = polarization_time - gate_time + reference_wait_time + \
        reference_time + background_wait_time + end_rest_time
    train = [(pre_duration, LOW), (gate_time, HIGH), (post_duration, LOW)]
    seq.setDigital(pulser_do_sig_apd_gate, train)

    # Reference APD gate
    pre_duration = aom_delay_time + prep_time + \
        polarization_time + reference_wait_time
    post_duration = reference_time - gate_time + \
        background_wait_time + end_rest_time
    train = [(pre_duration, LOW), (gate_time, HIGH), (post_duration, LOW)]
    seq.setDigital(pulser_do_ref_apd_gate, train)

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

    # Pulse the microwave for tau
    pre_duration = aom_delay_time + polarization_time + signal_wait_time
    post_duration = signal_wait_time + polarization_time + \
        reference_wait_time + reference_time + \
        background_wait_time + end_rest_time
    train = [(pre_duration, LOW), (tau, HIGH), (post_duration, LOW)]
    seq.setDigital(pulser_do_uwave, train)

    return seq, [period]


if __name__ == '__main__':
    wiring = {'do_daq_clock': 0,
              'do_apd_gate_0': 5,
              'do_apd_gate_1': 6,
              'do_aom': 3,
              'do_uwave_gate_0': 4,
              'do_uwave_gate_1': 1}
    args = [120, 3000, 1000, 1000, 2000, 1000, 750, 450, 400, 0, 1, 0]
    seq = get_seq(wiring, args)[0]
    seq.plot()
