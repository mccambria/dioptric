# -*- coding: utf-8 -*-
"""
Created on Mon Jun 3 11:49:23 2019

@author: gardill
"""

from pulsestreamer import Sequence
import numpy

def get_seq(pulser_wiring, args):

    # %% Parse wiring and args

    # Get the aom name
    aom_name = args[0]
    
    
    durations = []
    for ind in (1,2):
        durations.append(numpy.int64(args[ind]))
 
    # Unpack the durations
    aom_on_time, aom_off_time = durations
    
    # Define the period
    period = aom_on_time + aom_off_time
    
    #%% Based on the aom to use, set up the sequence
    
    seq = Sequence()
    
    if aom_name == '532_aom':
        LOW = 0
        HIGH = 1
        
        pulser_do_aom_driver = pulser_wiring['do_aom']
       
        train = [(aom_on_time, HIGH), (aom_off_time, LOW)]
        seq.setDigital(pulser_do_aom_driver, train)

    if aom_name == '589_aom':
        
        LOW = args[3]
        HIGH = args[4]
        
        pulser_ao_aom_driver = pulser_wiring['ao_589_aom']
        
        train = [(aom_on_time, HIGH), (aom_off_time, LOW)]
        seq.setAnalog(pulser_ao_aom_driver, train)
        
    if aom_name == '638_aom':
        
        LOW = args[3]
        HIGH = args[4]
        
        pulser_ao_aom_driver = pulser_wiring['ao_638_aom']
        
        train = [(aom_on_time, HIGH), (aom_off_time, LOW)]
        seq.setAnalog(pulser_ao_aom_driver, train)

    return seq, [period]

    # %% Define the sequence




if __name__ == '__main__':
    wiring = {'do_daq_clock': 0,
              'do_apd_gate_0': 1,
              'do_apd_gate_1': 2,
              'do_aom': 3,
              'do_uwave_gate_0': 4,
              'do_uwave_gate_1': 5}
    args = [100, 100, 1]
    seq = get_seq(wiring, args)
    seq.plot()   
