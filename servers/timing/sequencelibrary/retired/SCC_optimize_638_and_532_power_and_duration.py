#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 20:40:44 2020
4/8/20 includes initial red ionization pulse

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
    readout_time, init_ion_pulse_time, initial_pulse_time, test_pulse_time, \
            wait_time, laser_515_delay, aom_589_delay, laser_638_delay, \
            apd_indices, aom_ao_589_pwr, color_ind = args

    readout_time = numpy.int64(readout_time)
    init_ion_pulse_time = numpy.int64(init_ion_pulse_time)
    initial_pulse_time = numpy.int64(initial_pulse_time)
    test_pulse_time = numpy.int64(test_pulse_time)
    
    total_laser_delay = laser_515_delay + aom_589_delay + laser_638_delay
    # Test period
    period =  total_laser_delay + (init_ion_pulse_time + initial_pulse_time + test_pulse_time + \
                           readout_time + 3 * wait_time)*2
    
    # Get what we need out of the wiring dictionary
    pulser_do_apd_gate = pulser_wiring['do_apd_{}_gate'.format(apd_indices)]
    pulser_do_clock = pulser_wiring['do_sample_clock']
    pulser_do_532_aom = pulser_wiring['do_532_aom']
    pulser_ao_589_aom = pulser_wiring['ao_589_aom']
    pulser_do_638_aom = pulser_wiring['do_638_laser']
    
#    if color_ind == 532:
#        init_laser_delay = laser_515_delay + aom_589_delay
#        test_laser_delay = aom_589_delay + laser_638_delay
#        init_channel = pulser_do_638_aom
#        test_channel = pulser_do_532_aom
#    elif color_ind == 638:
#        init_laser_delay = laser_638_delay + aom_589_delay
#        test_laser_delay = aom_589_delay + laser_515_delay
#        init_channel = pulser_do_532_aom
#        test_channel = pulser_do_638_aom

    # Make sure the ao_aom voltage to the 589 aom is within 0 and 1 V
    tool_belt.aom_ao_589_pwr_err(aom_ao_589_pwr)
    
    seq = Sequence()


    #collect photons for certain timewindow tR in APD
    train = [(total_laser_delay + init_ion_pulse_time + initial_pulse_time + test_pulse_time +  3*wait_time, LOW), 
             (readout_time, HIGH),
             (init_ion_pulse_time + initial_pulse_time + test_pulse_time + 4*wait_time, LOW), 
             (readout_time, HIGH), (wait_time, LOW)]
    seq.setDigital(pulser_do_apd_gate, train)
    
    # if test pulse = 532 (green pulse w/ and w/out)
    if color_ind == 532:
        # red pulse is initial
        train = [(laser_515_delay + aom_589_delay, LOW), (init_ion_pulse_time, HIGH),
                 (wait_time, LOW), (initial_pulse_time, HIGH), 
                 (3*wait_time + test_pulse_time + readout_time, LOW), 
                 (init_ion_pulse_time, HIGH), (wait_time, LOW), 
                 (initial_pulse_time, HIGH), 
                 (3*wait_time + test_pulse_time + readout_time + laser_638_delay, LOW)]
        seq.setDigital(pulser_do_638_aom, train)
        
        # green pulse is test
        train = [(aom_589_delay + laser_638_delay + init_ion_pulse_time + initial_pulse_time + 2*wait_time, LOW), 
                 (test_pulse_time, HIGH), 
                 (6*wait_time + init_ion_pulse_time + initial_pulse_time + 2*readout_time + test_pulse_time + laser_515_delay, LOW)]       
        seq.setDigital(pulser_do_532_aom, train)
 
    # if test pulse = 638 (red pulse w/ and w/out)
    elif color_ind == 638:
        # green pulse is initial
        train = [(laser_638_delay + aom_589_delay + init_ion_pulse_time + wait_time, LOW),
                 (initial_pulse_time, HIGH), 
                 (4*wait_time + test_pulse_time + readout_time + init_ion_pulse_time, LOW), 
                 (initial_pulse_time, HIGH), 
                 (3*wait_time + test_pulse_time + readout_time + laser_515_delay, LOW)]
        seq.setDigital(pulser_do_532_aom, train)
    
        # red pulse is test
        train = [(aom_589_delay + laser_515_delay, LOW), (init_ion_pulse_time, HIGH),
                 (initial_pulse_time + 2*wait_time, LOW), 
                 (test_pulse_time, HIGH), 
                 (2*wait_time + readout_time, LOW), (init_ion_pulse_time, HIGH),
                 (4*wait_time + initial_pulse_time + readout_time + test_pulse_time + laser_638_delay, LOW)]       
        seq.setDigital(pulser_do_638_aom, train)
    
    # readout with 589
    train = [(laser_515_delay + laser_638_delay + init_ion_pulse_time + initial_pulse_time + test_pulse_time +  3*wait_time, LOW), 
             (readout_time, aom_ao_589_pwr),
             (init_ion_pulse_time + initial_pulse_time + test_pulse_time + 4*wait_time, LOW), 
             (readout_time, aom_ao_589_pwr), (wait_time + aom_589_delay, LOW)]
    seq.setAnalog(pulser_ao_589_aom, train) 
    

    
    final_digital = [pulser_do_clock]
    final = OutputState(final_digital, 0.0, 0.0)

    return seq, final, [period]


if __name__ == '__main__':
    wiring = {'do_apd_0_gate': 1,
              'do_532_aom': 2,
              'sig_gen_gate_chan_name': 3,
               'do_sample_clock':4,
               'ao_589_aom': 0,
               'ao_638_laser': 1,
               'do_638_laser': 7             }

    args = [1000,500, 100, 200, 100, 0, 0, 0, 0, 0.7,  532]
    seq, final, _ = get_seq(wiring, args)
    seq.plot()