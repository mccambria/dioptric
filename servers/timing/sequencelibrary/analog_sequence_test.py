# -*- coding: utf-8 -*-
"""
Created on Wed Jun 5 14:49:23 2019

@author: gardill
"""

from pulsestreamer import Sequence
from pulsestreamer import OutputState
import utils.tool_belt as tool_belt
import numpy

def get_seq(pulser_wiring, args):

    # %% Parse wiring and args

    # The first 9 args are ns durations and we need them as int64s
    durations = []
    for ind in range(2):
        durations.append(numpy.int64(args[ind]))
    
    # Unpack the durations
    aom_on_time, aom_off_time = durations
    
    analog_laser_power = args[2]
    color_ind = args[3]
    
    period = aom_on_time + aom_off_time
    
    # Analog
#    power = args[2]
#    MID = args[3]
#    HIGH = args[4]
    
    # Digital
    LOW = 0
    HIGH = 1

#    pulser_ao = 1
    
    seq = Sequence()
    
    if color_ind == 532:
        pulser_do_laser = pulser_wiring['do_532_aom']
        train = [(aom_on_time, HIGH), (aom_off_time, LOW)]
        seq.setDigital(pulser_do_laser, train)
        
    if color_ind == '515a':
        pulser_ao_515_aom = pulser_wiring['ao_515_laser']
        train = [(aom_on_time, analog_laser_power), (aom_off_time, LOW)]
        seq.setAnalog(pulser_ao_515_aom, train)
    if color_ind == 589:
        pulser_ao_515_aom = pulser_wiring['ao_589_aom']
        train = [(aom_on_time, analog_laser_power), (aom_off_time, LOW)]
        seq.setAnalog(pulser_ao_515_aom, train)
        
    if color_ind == 638:
        pulser_do_laser = pulser_wiring['do_638_laser']
        train = [(aom_on_time, HIGH), (aom_off_time, LOW)]
        seq.setDigital(pulser_do_laser, train)
        

#    train = [(aom_on_time, power), (aom_off_time + aom_on_time, 0.0), (aom_off_time, power)]
#    seq.setAnalog(pulser_ao, train)
#    
#    train = [(aom_on_time, HIGH), (aom_off_time, LOW), (aom_on_time, HIGH), (aom_off_time, LOW)]
#    seq.setDigital(pulser_do_638_laser, train)
    
#    train = [(aom_on_time, HIGH), (aom_off_time, LOW)]
#    seq.setDigital(pulser_do_laser, train)

    final = OutputState([], 0.0, 0.0)
    
    return seq, final, [period]

    # %% Define the sequence


if __name__ == '__main__':
    wiring = {'pulser_ao': 0,
              'do_638_laser': 1,
              'do_sample_clock': 2,
              'do_532_aom': 3,
              'ao_515_laser': 1}
#    args = [100, 100, 0.3, '515a']
    args = [1000, 1000000, 0.67, '515a']
    seq, _, _ = get_seq(wiring, args)
    seq.plot()   
