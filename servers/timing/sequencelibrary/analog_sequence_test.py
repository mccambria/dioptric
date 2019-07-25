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
    
    period = aom_on_time + aom_off_time + aom_on_time + aom_off_time
    
    # Analog
    power = args[2]
    half_power = power / 2
#    MID = args[3]
#    HIGH = args[4]
    
    # Digital
    LOW = 0
    HIGH = 1

    pulser_ao = 0
    
    pulser_do_638_laser = pulser_wiring['do_638_laser']
        
    seq = Sequence()

    train = [(aom_on_time, 1.0), (aom_off_time + aom_on_time, 0.9), (aom_off_time, 1.0)]
    seq.setAnalog(pulser_ao, train)
    
    train = [(aom_on_time, HIGH), (aom_off_time, LOW), (aom_on_time, HIGH), (aom_off_time, LOW)]
    seq.setDigital(pulser_do_638_laser, train)
    
    final_digital = [pulser_wiring['do_sample_clock'],
                     pulser_wiring['do_532_aom']]
    final = OutputState(final_digital, 1.0, 0.0)
    
    return seq, final, [period]

    # %% Define the sequence


if __name__ == '__main__':
    wiring = {'pulser_ao': 0,
              'do_638_laser': 1,
              'do_sample_clock': 2,
              'do_532_aom': 3}
    args = [100, 100, 1.0]
    seq, _, _ = get_seq(wiring, args)
    seq.plot()   
