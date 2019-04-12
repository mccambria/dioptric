# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 16:19:44 2019

@author: Matt
"""

from pulsestreamer import Sequence
import numpy

LOW = 0
HIGH = 1


def get_seq(pulser_wiring, args):

    # Unpack the args
    period, readout, apd_index = args

    period = numpy.int64(period)
    readout = numpy.int64(readout)
    delay_and_readout = delay + readout

    # Get what we need out of the wiring dictionary
    pulser_do_daq_clock = pulser_wiring['do_daq_clock']
    pulser_do_apd_gate = pulser_wiring['do_apd_gate_{}'.format(apd_index)]
    pulser_do_uwave = pulser_wiring['do_uwave']
    pulser_do_aom = pulser_wiring['do_aom']

    seq = Sequence()

    # After delay, ungate the APD channel for readout.
    # The delay is to allow the signal generator to switch frequencies.
    train = [(delay, LOW), (readout, HIGH),
             (delay, LOW), (readout, HIGH)]
    seq.setDigital(pulser_do_apd_gate, train)

    # Collect two samples
    train = [(period, LOW), (100, HIGH)]
    seq.setDigital(pulser_do_daq_clock, train)

    # Uwave should be on for the first measurement and off for the second
    train = [(delay_and_readout, HIGH), (delay_and_readout, LOW)]
    seq.setDigital(pulser_do_uwave, train)

    # The AOM and uwave should always be on
    train = [(period, HIGH)]
    seq.setDigital(pulser_do_aom, train)

    return seq


if __name__ == '__main__':
    wiring = {'do_daq_clock': 0,
              'do_apd_gate_0': 1,
              'do_aom': 2,
              'do_uwave': 3}
    args = [11 * 10**6, 10 * 10**6, 0]
    seq = get_seq(wiring, args)
    seq.plot()
