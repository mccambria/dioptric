#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 20:40:44 2020

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
    readout_time, initial_pulse_time, test_pulse_time, \
            wait_time, laser_515_delay, aom_589_delay, laser_638_delay, \
            apd_indices, aom_ao_589_pwr, color_ind = args

    readout_time = numpy.int64(readout_time)
    initial_pulse_time = numpy.int64(initial_pulse_time)
    test_pulse_time = numpy.int64(test_pulse_time)
    clock_time = 100
    
    total_laser_delay = laser_515_delay + aom_589_delay + laser_638_delay
    # Test period
    period =  total_laser_delay + (initial_pulse_time + test_pulse_time + \
                           readout_time + 3 * wait_time)*2
    
    # Get what we need out of the wiring dictionary
    pulser_do_apd_gate = pulser_wiring['do_apd_{}_gate'.format(apd_indices)]
    pulser_do_clock = pulser_wiring['do_sample_clock']
    pulser_do_532_aom = pulser_wiring['do_532_aom']
    pulser_ao_589_aom = pulser_wiring['ao_589_aom']
    pulser_do_638_aom = pulser_wiring['do_638_laser']
    
    if color_ind == 532:
        init_laser_delay = laser_515_delay + aom_589_delay
        test_laser_delay = aom_589_delay + laser_638_delay
        init_channel = pulser_do_638_aom
        test_channel = pulser_do_532_aom
    elif color_ind == 638:
        init_laser_delay = laser_638_delay + aom_589_delay
        test_laser_delay = aom_589_delay + laser_515_delay
        init_channel = pulser_do_532_aom
        test_channel = pulser_do_638_aom

    # Make sure the ao_aom voltage to the 589 aom is within 0 and 1 V
    tool_belt.aom_ao_589_pwr_err(aom_ao_589_pwr)
    
    seq = Sequence()


    #collect photons for certain timewindow tR in APD
    train = [(total_laser_delay + initial_pulse_time + test_pulse_time +  2*wait_time, LOW), (readout_time, HIGH),
             (initial_pulse_time + test_pulse_time + 3*wait_time, LOW), (readout_time, HIGH), (wait_time, LOW)]
    seq.setDigital(pulser_do_apd_gate, train)
    
    #clock pulse
    train = [(total_laser_delay + initial_pulse_time + test_pulse_time + readout_time + 3*wait_time - clock_time, LOW), (clock_time, HIGH), 
             ( initial_pulse_time + test_pulse_time + readout_time + 3*wait_time - clock_time, LOW), (clock_time, HIGH), (100, LOW)]
    seq.setDigital(pulser_do_clock, train)

    # inital pulse
    train = [ (init_laser_delay, LOW) , (initial_pulse_time, HIGH), (3*wait_time + test_pulse_time + readout_time, LOW), 
              (initial_pulse_time, HIGH), (3*wait_time + test_pulse_time + readout_time + clock_time, LOW)]
    if color_ind == 532:
        train.extend([(laser_638_delay, LOW)])
    if color_ind == 638:
        train.extend([(laser_515_delay, LOW)])       
    seq.setDigital(init_channel, train)
 
    # test pulse
    train = [(test_laser_delay + initial_pulse_time+ wait_time, LOW), (test_pulse_time, HIGH), \
             (5*wait_time + initial_pulse_time + 2*readout_time + test_pulse_time, LOW)]
    if color_ind == 532:
        train.extend([(laser_515_delay, LOW)])
    if color_ind == 638:
        train.extend([(laser_638_delay, LOW)])
    seq.setDigital(test_channel, train)
    
    # readout with 589
    train = [(laser_515_delay + laser_638_delay + initial_pulse_time + test_pulse_time +  2*wait_time, LOW), (readout_time, aom_ao_589_pwr),
             (initial_pulse_time + test_pulse_time + 3*wait_time, LOW), (readout_time, aom_ao_589_pwr), (wait_time + aom_589_delay, LOW)]
    seq.setAnalog(pulser_ao_589_aom, train) 
    

    
    final_digital = []
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

    args = [1000, 100, 200, 100, 0, 0, 0, 0, 0.7,  638]
    seq, final, _ = get_seq(wiring, args)
    seq.plot()