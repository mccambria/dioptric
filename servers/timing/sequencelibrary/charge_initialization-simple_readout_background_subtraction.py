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


def get_seq(pulse_streamer, config, args):

    # Unpack the args
    init_time, readout_time, apd_index, init_laser_name, init_laser_power, readout_laser_name, readout_laser_power = args

    # Get what we need out of the wiring dictionary
    pulser_wiring = config['Wiring']['PulseStreamer']
    pulser_do_daq_clock = pulser_wiring['do_sample_clock']
    pulser_do_daq_gate = pulser_wiring['do_apd_{}_gate'.format(apd_index)]
    
    # And the config dictionary
    init_delay = config["Optics"][init_laser_name]["delay"]
    readout_delay = config["Optics"][readout_laser_name]["delay"]
    # init_delay = 0
    # readout_delay = 0

    # Convert the 32 bit ints into 64 bit ints
    readout_delay = numpy.int64(readout_delay)
    init_delay = numpy.int64(init_delay)
    common_delay = max(readout_delay, init_delay) + 100
    readout_time = numpy.int64(readout_time)
    init_time = numpy.int64(init_time)
    init_readout_buffer = 4e3
    # init_readout_buffer = 500
    readout_init_buffer = 500
    
    # period = numpy.int64(common_delay + init_time + readout_time + 300)

#    tool_belt.check_laser_power(laser_name, laser_power)

    # chop_factor = 1
    # chop_factor = 1000  # For yellow 1e8 readouts
    chop_factor = int(1e4)  # For green 1e7 readouts
    # chop_factor = int(1e5)  # For green 1e8 readouts
    readout_time /= chop_factor

    # Define the sequence
    seq = Sequence()

    # The clock signal will be high for 100 ns with buffers of 100 ns on
    # either side. During the buffers, everything should be low. The buffers
    # account for any timing jitters/delays and ensure that everything we
    # expect to be on one side of the clock signal is indeed on that side.
    train = [(init_time + init_readout_buffer + readout_time + readout_init_buffer, LOW)]
    train *= chop_factor * 2
    train.extend([(100, LOW), (100, HIGH), (100, LOW)])
    train[0] = (train[0][0] + common_delay, train[0][1])
    period = int(sum([el[0] for el in train]))
    # train = [(period-200, LOW), (100, HIGH), (100, LOW)]
    seq.setDigital(pulser_do_daq_clock, train)

    # ADP gating
    train = [(init_time + init_readout_buffer, LOW), (readout_time, HIGH), (readout_init_buffer, LOW)]
    train *= chop_factor * 2
    train[0] = (train[0][0] + common_delay, train[0][1])
    train.append((300, LOW))
    seq.setDigital(pulser_do_daq_gate, train)

    # Init laser
    train = [(init_time, HIGH), (init_readout_buffer + readout_time + readout_init_buffer, LOW)]
    train *= chop_factor * 2
    train[0] = (train[0][0] + common_delay - init_delay, train[0][1])
    train.append((300 + init_delay, LOW))
    tool_belt.process_laser_seq(pulse_streamer, seq, config, 
                                init_laser_name, init_laser_power, train)

    # Readout laser
    # train = [(init_time + init_readout_buffer, LOW), (readout_time, HIGH), (readout_init_buffer, LOW)]
    # train *= chop_factor
    train = [(init_time + init_readout_buffer, LOW), (readout_time, HIGH), (readout_init_buffer, LOW),
             (init_time + init_readout_buffer, LOW), (readout_time, LOW), (readout_init_buffer, LOW)]
    train *= chop_factor
    train[0] = (train[0][0] + common_delay - readout_delay, train[0][1])
    train.append((300 + readout_delay, LOW))
    tool_belt.process_laser_seq(pulse_streamer, seq, config, 
                                readout_laser_name, readout_laser_power, train)

    final_digital = []
    final = OutputState(final_digital, 0.0, 0.0)

    return seq, final, [period]


if __name__ == '__main__':
    config = tool_belt.get_config_dict()
    args = [1000.0, 10000.0, 1, 'laserglow_532', None, 'laserglow_589', 1.0]
#    seq_args_string = tool_belt.encode_seq_args(args)
    seq, ret_vals, period = get_seq(None, config, args)
    seq.plot()
