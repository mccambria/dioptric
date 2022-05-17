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
    wait_time_1,pulse_1 ,\
        wait_time_2,pulse_2 ,\
            laser_name, laser_power = args

    wait_time_1 = numpy.int64(wait_time_1)
    pulse_1 = numpy.int64(pulse_1)
    wait_time_2 = numpy.int64(wait_time_2)
    pulse_2 = numpy.int64(pulse_2)

    period= wait_time_1 + pulse_1 + wait_time_2 + pulse_2
    
    # Define the sequence
    seq = Sequence()

    train = [(pulse_1, HIGH), (wait_time_1, LOW),
              (pulse_2, HIGH), (wait_time_2, LOW)]
             
    
    # train = [(75, HIGH), (5000, LOW)]
    tool_belt.process_laser_seq(pulse_streamer, seq, config, 
                                laser_name, laser_power, train)
    
    final_digital = []
    final = OutputState(final_digital, 0.0, 0.0)

    return seq, final, [period]


if __name__ == '__main__':
    config = tool_belt.get_config_dict()
    args = [1e4,1e4, 0, 0, 'integrated_520', None]
#    seq_args_string = tool_belt.encode_seq_args(args)
    seq, ret_vals, period = get_seq(None, config, args)
    seq.plot()
