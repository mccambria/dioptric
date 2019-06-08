# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 21:24:36 2019

@author: mccambria
"""

from pulsestreamer import Sequence
import numpy

LOW = 0
HIGH = 1


def get_seq(pulser_wiring, args):

    # Unpack the args
    delay, readout, apd_index = args

    # Get what we need out of the wiring dictionary
    pulser_do_daq_clock = pulser_wiring['do_daq_clock']
    pulser_do_daq_gate = pulser_wiring['do_apd_gate_{}'.format(apd_index)]
    pulser_do_aom = pulser_wiring['do_aom']

    # Convert the 32 bit ints into 64 bit ints
    delay = numpy.int64(delay)
    readout = numpy.int64(readout)
    period = numpy.int64(delay + readout + 300)

    seq = Sequence()

    # The clock signal will be high for 100 ns with buffers of 100 ns on
    # either side. During the buffers, everything should be low. The buffers
    # account for any timing jitters/delays and ensure that everything we
    # expect to be on one side of the clock signal is indeed on that side.
    train = [(delay + readout + 100, LOW), (100, HIGH), (100, LOW)]
    seq.setDigital(pulser_do_daq_clock, train)

    train = [(delay, LOW), (readout, HIGH), (300, LOW)]
    seq.setDigital(pulser_do_daq_gate, train)

    train = [(period, HIGH)]
    seq.setDigital(pulser_do_aom, train)

    return seq, [period]


if __name__ == '__main__':
    wiring = {'do_daq_clock': 0,
              'do_apd_gate_0': 1,
              'do_aom': 2}
    args = [250, 500, 0]
    seq, ret_vals = get_seq(wiring, args)
    seq.plot()
