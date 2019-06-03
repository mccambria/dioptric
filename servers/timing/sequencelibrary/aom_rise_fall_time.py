# -*- coding: utf-8 -*-
"""
Created on Mon Jun 3 11:49:23 2019

@author: gardill
"""

from pulsestreamer import Sequence
import numpy

LOW = 0
HIGH = 1


def get_seq(args):

    # %% Parse wiring and args

    # The first 9 args are ns durations and we need them as int64s
    durations = []
    for ind in range(2):
        durations.append(numpy.int64(args[ind]))

    # Unpack the durations
    aom_on_time, aom_off_time = durations
    
    driver_bool = args[2]

    pulser_do_aom_driver = 3
    pulser_do_switch = 4

    if driver_bool == True:
        
        seq = Sequence()

        train = [(aom_on_time, HIGH), (aom_on_time, LOW)]
        seq.setDigital(pulser_do_aom_driver, train)

    else:
        
        seq = Sequence()

        train = [(aom_on_time, HIGH), (aom_off_time, LOW)]
        seq.setDigital(pulser_do_switch, train)
        
        train = [(aom_on_time + aom_off_time, HIGH)]
        seq.setDigital(pulser_do_aom_driver, train)

    return seq

    # %% Define the sequence




if __name__ == '__main__':
#    wiring = {'do_daq_clock': 0,
#              'do_apd_gate_0': 1,
#              'do_apd_gate_1': 2,
#              'do_aom': 3,
#              'do_uwave_gate_0': 4,
#              'do_uwave_gate_1': 5}
    args = [100, 100, 0]
    seq = get_seq(args)
    seq.plot()   
