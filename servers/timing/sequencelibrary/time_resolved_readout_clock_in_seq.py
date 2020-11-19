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

    # The first 2 args are ns durations and we need them as int64s
    durations = []
    for ind in range(2):
        durations.append(numpy.int64(args[ind]))

    # Unpack the durations
    illum_pulse_duration, illum_pulse_delay = durations
                
    aom_ao_589_pwr = args[2]

    # Get the APD index
    apd_index = args[3]
    
    illum_color_ind = args[4]

    # Get what we need out of the wiring dictionary
    pulser_do_apd_gate = pulser_wiring['do_apd_{}_gate'.format(apd_index)]
    pulser_do_clock = pulser_wiring['do_sample_clock']
    pulser_do_532_aom = pulser_wiring['do_532_aom']
    pulser_ao_589_aom = pulser_wiring['ao_589_aom']
    pulser_do_638_aom = pulser_wiring['do_638_laser']
    
    clock_pulse = 500
    
    # Define params for the laser color
            
    if illum_color_ind == 532:
        illum_pulse_channel = pulser_do_532_aom
        illum_high = HIGH
    elif illum_color_ind == 589:
        illum_pulse_channel = pulser_ao_589_aom
        illum_high = aom_ao_589_pwr
    elif illum_color_ind == 638:
        illum_pulse_channel = pulser_do_638_aom
        illum_high = HIGH

    # %% Calclate total period.
    
#    We're going to readout the entire sequence, including the clock pulse
    period = illum_pulse_delay + illum_pulse_duration + 100 + clock_pulse
    
    # %% Define the sequence

    seq = Sequence()
    
    # APD 

    train = [(period, HIGH)]
    seq.setDigital(pulser_do_apd_gate, train)
    
    # clock
    
    
    train = [(illum_pulse_delay + illum_pulse_duration +100, LOW),(clock_pulse, HIGH)] 
    seq.setDigital(pulser_do_clock, train)
    

   
    # illumination pulse sequence.
    train = [(illum_pulse_duration, illum_high), 
             (100 + clock_pulse +  illum_pulse_delay, LOW)]
    if illum_color_ind == 589:
        seq.setAnalog(pulser_ao_589_aom, train)
    else:
        seq.setDigital(illum_pulse_channel, train)
        
    # If we're using red/yellow, we'll want to start in NV- each time. The
    # quickest way to implement this is to stick on a green pulse at the end
#    if init_color_ind == 638 and illum_color_ind == 589:
#        train = [(period + 1000, LOW), (3000, 1), (100, LOW)]
#        seq.setDigital(pulser_do_532_aom, train)
        
    final_digital = []
    final = OutputState(final_digital, 0.0, 0.0)
    return seq, final, [period]

if __name__ == '__main__':
    wiring = {'ao_638_laser': 0,
              'ao_589_aom':1,
              'do_sample_clock': 0,
              'do_apd_0_gate': 4,
              'do_532_aom': 1,
              'do_638_laser': 7
              }

    seq_args = [500, 0, 1, 0, 589,]

    seq, final, ret_vals = get_seq(wiring, seq_args)
    seq.plot()
