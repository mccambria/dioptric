#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 20:40:44 2020

A sequence to do G/T/Y and R/T/Y, where T isG, R, or Y. This starts the 
NV in either NV- or NV0, then applies a pusle for some duration and then 
checks the final charge state.

Right now, use only with green analog modulation!

@author: agardill
"""

from pulsestreamer import Sequence
from pulsestreamer import OutputState
import numpy

LOW = 0
HIGH = 1

def get_seq(pulser_wiring, args):

    # Unpack the args
    readout_time, green_prep_time, red_prep_time, test_time, \
            wait_time, test_color, laser_515_delay, aom_589_delay, laser_638_delay, \
            apd_indices, readout_power_589, prep_power_515, test_power = args

    readout_time = numpy.int64(readout_time)
    red_prep_time = numpy.int64(red_prep_time)
    green_prep_time = numpy.int64(green_prep_time)
    test_time = numpy.int64(test_time)
    wait_time = numpy.int64(wait_time)
    
    total_laser_delay = laser_515_delay + aom_589_delay + laser_638_delay
    # Test period
    period =  total_laser_delay + (green_prep_time + test_time + readout_time + \
                           3 * wait_time) +  (red_prep_time + test_time + readout_time + \
                           3 * wait_time)
    
    # Get what we need out of the wiring dictionary
    pulser_do_apd_gate = pulser_wiring['do_apd_{}_gate'.format(apd_indices)]
    pulser_do_clock = pulser_wiring['do_sample_clock']
    pulser_ao_515_aom = pulser_wiring['ao_515_laser']
#    pulser_do_515_aom = pulser_wiring['do_532_aom'] ##
    pulser_ao_589_aom = pulser_wiring['ao_589_aom']
    pulser_do_638_aom = pulser_wiring['do_638_laser']

    
    seq = Sequence()


    #collect photons for certain timewindow tR in APD
    train = [(total_laser_delay + green_prep_time + test_time + 2*wait_time, LOW), 
             (readout_time, HIGH),
             (red_prep_time + test_time  + 3*wait_time, LOW), 
             (readout_time, HIGH), (wait_time, LOW)]
    seq.setDigital(pulser_do_apd_gate, train)

    # laser pulses
    green_train = [(laser_638_delay + aom_589_delay, LOW),
         (green_prep_time, prep_power_515), (wait_time, LOW)]
#    green_train = [(laser_638_delay + aom_589_delay, LOW),
#         (green_prep_time, HIGH), (wait_time, LOW)]
    red_train = [(laser_515_delay + aom_589_delay + wait_time + green_prep_time, LOW)]
    yellow_train = [(laser_638_delay + laser_515_delay+ wait_time + green_prep_time, LOW)]
    
    test_pulse_off = [(test_time, LOW)] 
    if test_color == '515a':
        test_pulse_on = [(test_time, test_power)] 
        green_train.extend(test_pulse_on) 
        red_train.extend(test_pulse_off) 
        yellow_train.extend(test_pulse_off)
#    if test_color == 532:
#        test_pulse_on = [(test_time, HIGH)] 
#        green_train.extend(test_pulse_on) 
#        red_train.extend(test_pulse_off) 
#        yellow_train.extend(test_pulse_off)
    elif test_color == 638:
        test_pulse_on = [(test_time, HIGH)] 
        green_train.extend(test_pulse_off)
        red_train.extend(test_pulse_on)
        yellow_train.extend(test_pulse_off)
    elif test_color == 589:
        test_pulse_on = [(test_time, test_power)] 
        green_train.extend(test_pulse_off)
        red_train.extend(test_pulse_off) 
        yellow_train.extend(test_pulse_on)
    
    green_train.extend([(wait_time + readout_time + wait_time + red_prep_time + wait_time, LOW)])
    red_train.extend([(wait_time + readout_time + wait_time, LOW), (red_prep_time, HIGH), (wait_time, LOW)])
    yellow_train.extend([(wait_time, LOW), (readout_time, readout_power_589), (wait_time + red_prep_time + wait_time, LOW)])
    
    if test_color == '515a':
        test_pulse_on = [(test_time, test_power)] 
        green_train.extend(test_pulse_on) 
        red_train.extend(test_pulse_off) 
        yellow_train.extend(test_pulse_off)
#    if test_color == 532:
#        test_pulse_on = [(test_time, HIGH)] 
#        green_train.extend(test_pulse_on) 
#        red_train.extend(test_pulse_off) 
#        yellow_train.extend(test_pulse_off)
    elif test_color == 638:
        test_pulse_on = [(test_time, HIGH)] 
        green_train.extend(test_pulse_off)
        red_train.extend(test_pulse_on)
        yellow_train.extend(test_pulse_off)
    elif test_color == 589:
        test_pulse_on = [(test_time, test_power)] 
        green_train.extend(test_pulse_off)
        red_train.extend(test_pulse_off) 
        yellow_train.extend(test_pulse_on)
        
    green_train.extend([(wait_time + readout_time + wait_time, LOW)])
    red_train.extend([(wait_time + readout_time + wait_time, LOW)])
    yellow_train.extend([(wait_time, LOW), (readout_time, readout_power_589), (wait_time, LOW)])
    
        
    seq.setAnalog(pulser_ao_515_aom, green_train)
#    seq.setDigital(pulser_do_515_aom, green_train)
    seq.setDigital(pulser_do_638_aom, red_train)
    seq.setAnalog(pulser_ao_589_aom, yellow_train) 
    

    
    final_digital = [pulser_do_clock]
    final = OutputState(final_digital, 0.0, 0.0)

    return seq, final, [period]


if __name__ == '__main__':
    wiring = {'do_apd_0_gate': 1,
              'ao_515_laser': 1,
              'sig_gen_gate_chan_name': 3,
               'do_sample_clock':4,
              'do_515_laser': 5,
               'ao_589_aom': 0,
               'ao_638_laser': 1,
               'do_638_laser': 7             }

    args = [1000, 500, 750, 500, 1000, '515a', 500, 0, 0, 0, 1,  1, 0.5]
            
#    args = [1500, 1000, 1000, 300, 1000, 532, 0, 0, 0, 0, 1, 1, 1]
    seq, final, _ = get_seq(wiring, args)
    seq.plot()