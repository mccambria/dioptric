# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 21:24:36 2019

@author: mccambria
"""

from pulsestreamer import Sequence
from pulsestreamer import OutputState
import utils.tool_belt as tool_belt
import numpy

LOW = 0
HIGH = 1


def get_seq(pulser_wiring, args):

    # Unpack the args
    delay, readout_time, laser_name, laser_power, apd_index = args

    # Get what we need out of the wiring dictionary
    pulser_do_daq_clock = pulser_wiring['do_sample_clock']
    pulser_do_daq_gate = pulser_wiring['do_apd_{}_gate'.format(apd_index)]

    # Convert the 32 bit ints into 64 bit ints
    delay = numpy.int64(delay)
    readout_time = numpy.int64(readout_time)

    period = numpy.int64(delay + readout_time + 300)

    tool_belt.check_laser_power(laser_name, laser_power)

    # Define the sequence
    seq = Sequence()

    # The clock signal will be high for 100 ns with buffers of 100 ns on
    # either side. During the buffers, everything should be low. The buffers
    # account for any timing jitters/delays and ensure that everything we
    # expect to be on one side of the clock signal is indeed on that side.
    train = [(delay + readout_time + 100, LOW), (100, HIGH), (100, LOW)]
    seq.setDigital(pulser_do_daq_clock, train)

    train = [(delay, LOW), (readout_time, HIGH), (300, LOW)]
    seq.setDigital(pulser_do_daq_gate, train)

    if laser_power == -1:
        train = [(period, HIGH)]
        pulser_laser_mod = pulser_wiring['do_{}_dm'.format(laser_name)]
        seq.setDigital(pulser_laser_mod, train)
    else:
        train = [(period, laser_power)]
        pulser_laser_mod = pulser_wiring['ao_{}_am'.format(laser_name)]
        seq.setAnalog(pulser_laser_mod, train)

    final_digital = []
    final = OutputState(final_digital, 0.0, 0.0)

    return seq, final, [period]


if __name__ == '__main__':
    wiring = {'do_sample_clock': 0,
              'do_apd_0_gate': 1,
              'do_638_laser': 3,
              'do_laser_532_dm': 2,
              'ao_515_laser': 0,
              'ao_589_aom': 1}
#    args = [500000, 10000000, 0.3, 0.685, 0, '515a']
    args = [500000, 10000000.0, 'laser_532', -1, 0]
#    seq_args_string = tool_belt.encode_seq_args(args)
    seq, ret_vals, period = get_seq(wiring, args)
    seq.plot()
