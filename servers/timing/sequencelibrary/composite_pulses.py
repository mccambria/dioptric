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


def get_seq_basic(pulser_wiring, args):

    # %% Parse wiring and args

    pulser_do_arb_wave_trigger = pulser_wiring['do_arb_wave_trigger']
    pulser_do_signal_generator_sg394_gate = pulser_wiring['do_signal_generator_sg394_gate']

    # Convert the 32 bit ints into 64 bit ints
    period = 4000

    seq = Sequence()
    
    train = [(period/2, LOW), (period/2, HIGH)]
    seq.setDigital(pulser_do_arb_wave_trigger, train)
    
    train = [(period, HIGH)]
    seq.setDigital(pulser_do_signal_generator_sg394_gate, train)

    final_digital = []
    final = OutputState(final_digital, 0.0, 0.0)
    return seq, final, [period]


def get_seq(pulser_wiring, args):

    # %% Parse wiring and args
    
    durations = []
    for ind in range(5):
        durations.append(numpy.int64(args[ind]))
    # Unpack the durations
    buffer, uwave_dur, gap, switch_delay, iq_delay = durations
    half_gap = numpy.int64(gap / 2)
    gap_remainder = gap - half_gap - 10
    
    sig_gen = args[5]

    pulser_do_arb_wave_trigger = pulser_wiring['do_arb_wave_trigger']
    if sig_gen == 'sg394':
        pulser_do_sig_gen_gate = pulser_wiring['do_signal_generator_sg394_gate']
    elif sig_gen == 'tsg4104a':
        pulser_do_sig_gen_gate = pulser_wiring['do_signal_generator_tsg4104a_gate']

    # %% Write the sequence
    
    seq = Sequence()
    
    train = [(buffer - switch_delay, LOW), (uwave_dur, HIGH), (gap, LOW), (uwave_dur, HIGH), (switch_delay, LOW)]
    seq.setDigital(pulser_do_sig_gen_gate, train)
    
    train = [(buffer - iq_delay, LOW), (uwave_dur + half_gap, LOW), (10, HIGH), (gap_remainder + uwave_dur, LOW), (iq_delay, LOW)]
    seq.setDigital(pulser_do_arb_wave_trigger, train)

    # %% Turn everything off at the end

    final_digital = []
    final = OutputState(final_digital, 0.0, 0.0)
    return seq, final, []


if __name__ == '__main__':
    wiring = {'ao_589_aom': 1, 'ao_638_laser': 0, 'do_532_aom': 3,
              'do_638_laser': 7, 'do_apd_0_gate': 5, 'do_arb_wave_trigger': 2,
              'do_sample_clock': 0, 'do_signal_generator_bnc835_gate': 1,
              'do_signal_generator_sg394_gate': 4}
    # buffer, uwave_dur, gap, switch_delay, iq_delay, sig_gen
    seq_args = [100, 400, 32, 50, 0, 'sg394']
    seq = get_seq(wiring, seq_args)[0]
    seq.plot()
