# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 21:24:36 2019

@author: Matt
"""

from pulsestreamer import Sequence
import numpy

LOW = 0
HIGH = 1


def get_seq(wiring, args):

    # Get what we need out of the wiring dictionary
    pulser_do_daq_clock = wiring['pulser_do_daq_clock']
    pulser_do_daq_gate = wiring['pulser_do_daq_gate']
    pulser_do_aom = wiring['pulser_do_aom']

    # Unpack the args
    period, readout = args

    # Convert the 32 bit ints into 64 bit ints
    period = numpy.int64(period)
    readout = numpy.int64(readout)

    seq = Sequence()

    train = [(100, HIGH), (period - 100, LOW)]
    seq.setDigital(pulser_do_daq_clock, train)

    train = [(period - readout, LOW), (readout, HIGH)]
    seq.setDigital(pulser_do_daq_gate, train)

    train = [(period, HIGH)]
    seq.setDigital(pulser_do_aom, train)

    return seq


if __name__ == '__main__':
    wiring = {'pulser_do_daq_clock': 0,
              'pulser_do_daq_gate': 1,
              'pulser_do_aom': 2}
    args = [11 * 10**6, 10 * 10**6]
    seq = get_seq(wiring, args)
    seq.plot()
