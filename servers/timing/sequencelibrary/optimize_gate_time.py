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
    for ind in range(10):
        durations.append(numpy.int64(args[ind]))

    # Unpack the durations
    polarization_time, signal_time, reference_time, sig_to_ref_wait_time, \
        pre_uwave_exp_wait_time, post_uwave_exp_wait_time, aom_delay_time, \
        rf_delay_time, gate_time, pi_pulse = durations

    # Get the APD indices
    sig_apd_index, ref_apd_index = args[10:12]
    
    #Signify which signal generator to use
    do_uwave_gate = args[12]

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

    prep_time = aom_delay_time + rf_delay_time + polarization_time + \
        pre_uwave_exp_wait_time + pi_pulse + post_uwave_exp_wait_time

    # The period is independent of the particular tau, but it must be long
    # enough to accomodate the longest tau
    period = aom_delay_time + rf_delay_time + polarization_time + \
        pre_uwave_exp_wait_time + pi_pulse + post_uwave_exp_wait_time + \
        signal_time + sig_to_ref_wait_time + reference_time

    # %% Define the sequence

    seq = Sequence()

    # Signal APD gate
    pre_duration = prep_time
    post_duration = signal_time - gate_time + sig_to_ref_wait_time + reference_time
    train = [(pre_duration, LOW), (gate_time, HIGH), (post_duration, LOW)]
    seq.setDigital(pulser_do_sig_apd_gate, train)

    # Reference APD gate
    pre_duration = prep_time + signal_time + sig_to_ref_wait_time
    post_duration = reference_time - gate_time
    train = [(pre_duration, LOW), (gate_time, HIGH), (post_duration, LOW)]
    seq.setDigital(pulser_do_ref_apd_gate, train)

    # Pulse the laser with the AOM for polarization and readout
    train = [(rf_delay_time + polarization_time, HIGH),
             (pre_uwave_exp_wait_time + pi_pulse + post_uwave_exp_wait_time, LOW),
             (signal_time, HIGH),
             (sig_to_ref_wait_time, LOW),
             (reference_time + aom_delay_time, HIGH)]
    seq.setDigital(pulser_do_aom, train)

    # Pulse the microwave for tau
    pre_duration = aom_delay_time + polarization_time + pre_uwave_exp_wait_time
    post_duration = post_uwave_exp_wait_time + signal_time + \
        sig_to_ref_wait_time + reference_time + rf_delay_time
    train = [(pre_duration, LOW), (pi_pulse, HIGH), (post_duration, LOW)]
    seq.setDigital(pulser_do_uwave, train)

    return seq, [period]


if __name__ == '__main__':
    wiring = {'do_daq_clock': 0,
              'do_apd_gate_0': 1,
              'do_apd_gate_1': 2,
              'do_aom': 3,
              'do_uwave_gate_0': 4,
              'do_uwave_gate_1': 5}
    args = [3000, 3000, 3000, 2000 , 1000, 1000, 0, 0, 300, 100, 0, 1, 0]
    seq, ret_vals = get_seq(wiring, args)
    seq.plot()   
