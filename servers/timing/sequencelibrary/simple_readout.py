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
    period = numpy.int64(delay+readout+100)

    seq = Sequence()

    train = [(delay+readout, HIGH), (100, LOW)]
    seq.setDigital(pulser_do_daq_clock, train)

    train = [(delay, LOW), (readout, HIGH), (100, LOW)]
    seq.setDigital(pulser_do_daq_gate, train)

    train = [(period, HIGH)]
    seq.setDigital(pulser_do_aom, train)

    return seq, [period]


if __name__ == '__main__':
    wiring = {'pulser_do_daq_clock': 0,
              'pulser_do_daq_gate': 1,
              'pulser_do_aom': 2}
    args = [11 * 10**6, 10 * 10**6]
    seq = get_seq(wiring, args)
    seq.plot()
