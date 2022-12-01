# -*- coding: utf-8 -*-
"""
Created on Sat May  4 08:34:08 2019

2/24/2020 Setting the start of the readout_time at the beginning of the sequence.

@author: Aedan
"""

from pulsestreamer import Sequence
from pulsestreamer import OutputState
import numpy
#import utils.tool_belt as tool_belt
#from utils.tool_belt import States

LOW = 0
HIGH = 1

def get_seq(pulser_wiring, args):

    # %% Parse wiring and args

    # The first 11 args are ns durations and we need them as int64s
    durations = []
    for ind in range(6):
        durations.append(numpy.int64(args[ind]))

    # Unpack the durations
    tau_frst, polarization_time, inter_exp_wait_time, aom_delay_time,  \
            gate_time, tau_scnd = durations

    # Get the APD indices
    apd_index = args[6]


    pulser_do_apd_gate = pulser_wiring['do_apd_{}_gate'.format(apd_index)]

    pulser_do_aom = pulser_wiring['do_532_aom']

    # %% Calclate total period. This is fixed for each tau index

    # The period is independent of the particular tau, but it must be long
    # enough to accomodate the longest tau
    period = aom_delay_time + polarization_time +  tau_frst + \
        gate_time + inter_exp_wait_time + polarization_time + tau_scnd + \
        gate_time + inter_exp_wait_time

    # %% Define the sequence

    seq = Sequence()

    # APD gating
    pre_duration = aom_delay_time +  tau_frst

    train = [(pre_duration, LOW),
             (gate_time, HIGH),
             (polarization_time + inter_exp_wait_time  + tau_scnd, LOW),
             (gate_time, HIGH),
             (polarization_time + inter_exp_wait_time, LOW)]
    seq.setDigital(pulser_do_apd_gate, train)

    # Pulse the laser with the AOM for polarization and readout
    train = [(polarization_time, HIGH),
             (tau_frst + gate_time + inter_exp_wait_time, LOW),
             (polarization_time, HIGH),
             (tau_scnd + gate_time + inter_exp_wait_time, LOW)]
    seq.setDigital(pulser_do_aom, train)

    final_digital = [pulser_wiring['do_532_aom'],
                     pulser_wiring['do_sample_clock']]
    final = OutputState(final_digital, 0.0, 0.0)
    return seq, final, [period]

if __name__ == '__main__':
    wiring = {'do_sample_clock': 0,
              'do_apd_0_gate': 4,
              'do_532_aom': 1,
              'do_signal_generator_tsg4104a_gate': 2,
              'do_signal_generator_bnc835_gate': 3}

    seq_args = [10**5/2, 10000, 0, 0, 5000, 10**5/2, 0]

    seq, final, ret_vals = get_seq(wiring, seq_args)
    seq.plot()
