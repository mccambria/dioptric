# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 21:24:36 2019

@author: Matt
"""

from pulsestreamer import Sequence

LOW = 0
HIGH = 1

def get_seq(wiring, args):

    # Get what we need out of the wiring dictionary
    pulser_do_daq_clock = wiring['pulser_do_daq_clock']
    pulser_do_daq_gate = wiring['pulser_do_daq_gate']
    pulser_do_aom = wiring['pulser_do_aom']

    # Unpack the args
    period, readout = args

    seq = Sequence()

    train = [(100, HIGH), (period - 100, LOW)]
    seq.setDigital(pulser_do_daq_clock, train)

    train = [(period - readout, LOW), (readout, HIGH)]
    seq.setDigital(pulser_do_daq_gate, train)

    train = [(period, HIGH)]
    seq.setDigital(pulser_do_aom, train)

    return seq
