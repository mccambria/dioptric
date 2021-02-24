# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 21:24:36 2019

@author: mccambria
"""

from pulsestreamer import Sequence
from pulsestreamer import OutputState
import numpy

LOW = 0
HIGH = 1


def get_seq(pulser_wiring, args):

    # Unpack the args
    delay, readout, apd_index = args

    # Get what we need out of the wiring dictionary
    pulser_do_daq_clock = pulser_wiring['do_sample_clock']
    pulser_do_daq_gate = pulser_wiring['do_apd_{}_gate'.format(apd_index)]
    pulser_do_aom = pulser_wiring['do_532_aom']

    # Convert the 32 bit ints into 64 bit ints
    delay = numpy.int64(delay)
    readout = numpy.int64(readout)
    period = numpy.int64(delay + readout + 100)

    seq = Sequence()

    # Clock signal (one at the beginning of the sequence and one at the end)
    train = [(delay, LOW), (100, HIGH), (readout-100, LOW), (100, HIGH)]
    seq.setDigital(pulser_do_daq_clock, train)
    # Gate
    train = [(delay, LOW), (100 + readout, HIGH)]
    seq.setDigital(pulser_do_daq_gate, train)
    
    # Leave the laser on all the time
    train = [(period, HIGH)]
    seq.setDigital(pulser_do_aom, train)

    final_digital = []
    final = OutputState(final_digital, 0.0, 0.0)
    return seq, final, [period]


if __name__ == '__main__':
    wiring = {'do_sample_clock': 0,
              'do_apd_0_gate': 1,
              'do_532_aom': 2}
    args = [250, 500, 0]
    seq, ret_vals, _ = get_seq(wiring, args)
    seq.plot()
