# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 21:24:36 2019

This file simply turns the laser on for a set duration, then turns it off. 
It allows any one of the three lasers (red, yellow, green) to be used.

@author: agardill
"""

from pulsestreamer import Sequence
from pulsestreamer import OutputState
import utils.tool_belt as tool_belt
import numpy

LOW = 0
HIGH = 1




def get_seq(pulse_streamer, config, args):

    # Unpack the args
    delay, duration, laser_key, laser_power  = args
      


    # Convert the 32 bit ints into 64 bit ints
    duration = numpy.int64(duration)
    delay = numpy.int64(delay)
    
        
        
    # Define the sequence
    seq = Sequence()
    
    period = numpy.int64(delay + duration)
        
    final_digital = []
    
    train = [(delay, LOW), (duration, HIGH)]
    tool_belt.process_laser_seq(pulse_streamer, seq, config,
                            laser_key, laser_power, train)

    final = OutputState(final_digital, 0.0, 0.0)
    
    return seq, final, [period]


if __name__ == '__main__':
    config = tool_belt.get_config_dict()
    args = [100, 500, 'cobolt_515', -1]
    seq = get_seq(None, config, args)[0]
    seq.plot()
