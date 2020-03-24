# -*- coding: utf-8 -*-
"""
Created on Sat Mar  24 08:34:08 2020

For SCC, it's useful to observe the photon counts under constant illumination. 

This file is the sequence to count the photons while yellow/red light is
illuminating (after being reionized with green) or while green light it
iluminating (after being ionized with red)

@author: Aedan
"""

from pulsestreamer import Sequence
from pulsestreamer import OutputState
import numpy
#import utils.tool_belt as tool_belt
#from utils.tool_belt import States

LOW = 0
HIGH = 1

def get_seq(pulser_wiring, args):

    # %% Parse wiring and args

    # The first 3 args are ns durations and we need them as int64s
    durations = []
    for ind in range(5):
        durations.append(numpy.int64(args[ind]))

    # Unpack the durations
    readout, init_pulse_duration, illum_pulse_duration, wait_time, \
                laser_delay_time = durations
                
    aom_ao_589_pwr = args[5]
    ao_638_pwr = args[6]

    # Get the APD index
    apd_index = args[7]
    
    init_color_ind = args[8]
    illum_color_ind = args[9]

    # Get what we need out of the wiring dictionary
    pulser_do_apd_gate = pulser_wiring['do_apd_{}_gate'.format(apd_index)]
#    pulser_do_clock = pulser_wiring['do_sample_clock']
    pulser_do_532_aom = pulser_wiring['do_532_aom']
    pulser_ao_589_aom = pulser_wiring['ao_589_aom']
    pulser_ao_638_aom = pulser_wiring['ao_638_laser']
    
    # Allow any combination of first pusle and second pulse
    
    if init_color_ind == 532:
        init_high = HIGH
    elif init_color_ind == 589:
        init_pulse_channel = pulser_ao_589_aom
        init_high = aom_ao_589_pwr
    elif init_color_ind == 638:
        init_pulse_channel = pulser_ao_638_aom
        init_high = ao_638_pwr
        
    if illum_color_ind == 532:
        illum_high = HIGH
    elif illum_color_ind == 589:
        illum_pulse_channel = pulser_ao_589_aom
        illum_high = aom_ao_589_pwr
    elif illum_color_ind == 638:
        illum_pulse_channel = pulser_ao_638_aom
        illum_high = ao_638_pwr

    # %% Calclate total period. This is fixed for each tau index

    # The period is independent of the particular tau, but it must be long
    # enough to accomodate the longest tau
    period = laser_delay_time + init_pulse_duration + \
                                        illum_pulse_duration + wait_time

    # %% Define the sequence

    seq = Sequence()
    
    # APD 

    train = [(laser_delay_time + init_pulse_duration + wait_time, LOW),
             (readout, HIGH), (illum_pulse_duration - readout, LOW),]
    seq.setDigital(pulser_do_apd_gate, train)

    # initial pulse sequence.
    train = [(init_pulse_duration, init_high),
             (wait_time + illum_pulse_duration + laser_delay_time, LOW)]
    if init_color_ind == 532:
        seq.setDigital(pulser_do_532_aom, train)
    else:
        seq.setAnalog(init_pulse_channel, train) 

    # illumination pulse sequence.
    train = [(init_pulse_duration + wait_time, LOW),
             (illum_pulse_duration, illum_high), ( laser_delay_time, LOW)]
    
    if illum_color_ind == 532:
        seq.setDigital(pulser_do_532_aom, train)
    else:
        seq.setAnalog(illum_pulse_channel, train) 
        
    final_digital = []
    final = OutputState(final_digital, 0.0, 0.0)
    return seq, final, [period]

if __name__ == '__main__':
    wiring = {'ao_638_laser': 0,
              'ao_589_aom':1,
              'do_sample_clock': 0,
              'do_apd_0_gate': 4,
              'do_532_aom': 1
              }
    seq_args = [100, 100, 200, 100, 0, 1.0, 1.0, 0, 532, 589 ]

    seq, final, ret_vals = get_seq(wiring, seq_args)
    seq.plot()
