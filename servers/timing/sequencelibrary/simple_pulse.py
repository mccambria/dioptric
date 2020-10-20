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


def get_seq(pulser_wiring, args):

    # Unpack the args
    delay, duration, aom_ao_589_pwr, color_ind = args

    # Get what we need out of the wiring dictionary
    pulser_do_532_aom = pulser_wiring['do_532_aom']
    pulser_ao_589_aom = pulser_wiring['ao_589_aom']
    pulser_do_638_aom = pulser_wiring['638_DM_laser_delay']

    # Convert the 32 bit ints into 64 bit ints
    duration = numpy.int64(duration)
    delay = numpy.int64(delay)
    
    # Make sure the aom_ao_589_pwer is within range of +1 and 0
    tool_belt.aom_ao_589_pwr_err(aom_ao_589_pwr)
        
    # make sure only the color passed is either 532 or 589
#    tool_belt.color_ind_err(color_ind)
        
    # Define the sequence
    seq = Sequence()
    
    period = numpy.int64(delay + duration)
        
    final_digital = []
    
    if color_ind == 638:
        train = [(delay, LOW), (duration, HIGH)]
        seq.setDigital(pulser_do_638_aom, train)
        
    elif color_ind == 589:
        
        train = [(delay, LOW), (duration, aom_ao_589_pwr)]
        seq.setAnalog(pulser_ao_589_aom, train)
    
    elif color_ind == 532:
        
        train = [(delay, LOW), (duration, HIGH)]
        seq.setDigital(pulser_do_532_aom, train)

    final = OutputState(final_digital, 0.0, 0.0)
    
    return seq, final, [period]


if __name__ == '__main__':
    wiring = {'do_sample_clock': 0,
              'do_apd_0_gate': 1,
              'do_532_aom': 2,
              '638_DM_laser_delay': 3,
              'ao_589_aom': 1}
    args = [100, 500, 1.0, 532]
    seq, ret_vals, _ = get_seq(wiring, args)
    seq.plot()
