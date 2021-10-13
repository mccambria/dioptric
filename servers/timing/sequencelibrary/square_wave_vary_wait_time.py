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
    laser_on_time, wait_time_1, wait_time_2, laser_name, laser_power = args

    laser_on_time = numpy.int64(laser_on_time)
    wait_time_1 = numpy.int64(wait_time_1)
    wait_time_2 = numpy.int64(wait_time_2)
    
    period = laser_on_time * 2 + wait_time_1 + wait_time_2

    # Define the sequence
    seq = Sequence()

    train = [(laser_on_time, HIGH), (wait_time_1, LOW), (laser_on_time, HIGH), (wait_time_2, LOW)]
    tool_belt.process_laser_seq(pulse_streamer, seq, config, 
                                laser_name, laser_power, train)

    final_digital = []
    final = OutputState(final_digital, 0.0, 0.0)

    return seq, final, [period]


if __name__ == '__main__':
    config = tool_belt.get_config_dict()
    args = [1000.0, 100, 500, 'cobolt_515', None]
#    seq_args_string = tool_belt.encode_seq_args(args)
    seq, ret_vals, period = get_seq(None, config, args)
    seq.plot()
