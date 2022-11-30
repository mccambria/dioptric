#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 20:40:44 2020

Similar to SCC_optimize_pulses_wout_uwaves, however now we have arbitrary 
control over the pusle colors, for the initial pulse, the test pulse, and the 
readout pulse.

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
    readout_time, init_pulse_time, test_pulse_time, \
            wait_time, laser_515_delay, aom_589_delay, laser_638_delay, \
            init_color, test_color, read_color, \
            apd_indices, aom_ao_589_pwr = args

    readout_time = numpy.int64(readout_time)
    init_pulse_time = numpy.int64(init_pulse_time)
    test_pulse_time = numpy.int64(test_pulse_time)
    
    total_laser_delay = laser_515_delay + aom_589_delay + laser_638_delay
    # Test period
    period =  total_laser_delay + (init_pulse_time + test_pulse_time +  \
                           readout_time + 3 * wait_time)*2
    
    # Get what we need out of the wiring dictionary
    pulser_do_apd_gate = pulser_wiring['do_apd_{}_gate'.format(apd_indices)]
    pulser_do_clock = pulser_wiring['do_sample_clock']
    pulser_do_532_aom = pulser_wiring['do_532_aom']
    pulser_ao_589_aom = pulser_wiring['ao_589_aom']
    pulser_do_638_aom = pulser_wiring['do_638_laser']


    # Make sure the ao_aom voltage to the 589 aom is within 0 and 1 V
    tool_belt.aom_ao_589_pwr_err(aom_ao_589_pwr)
    
    seq = Sequence()


    #collect photons for certain timewindow tR in APD
    train = [(total_laser_delay + init_pulse_time + test_pulse_time +  3*wait_time, LOW), 
             (readout_time, HIGH),
             (init_pulse_time + test_pulse_time + 4*wait_time, LOW), 
             (readout_time, HIGH), (wait_time, LOW)]
    seq.setDigital(pulser_do_apd_gate, train)

    
    # start each laser sequence
    train_532 = [(total_laser_delay - laser_515_delay , LOW)]
    train_589 = [(total_laser_delay - aom_589_delay, LOW)]
    train_638 = [(total_laser_delay - laser_638_delay, LOW)]
    
    wait_train = [(wait_time, LOW)]
 
    # 1st time: add the initialization pulse segment
    init_train_on = [(init_pulse_time, HIGH)]
    init_train_off = [(init_pulse_time, LOW)]
    if init_color == 532:
        train_532.extend(init_train_on)
        train_589.extend(init_train_off)
        train_638.extend(init_train_off)
    if init_color == 589:
        init_train_on = [(init_pulse_time, aom_ao_589_pwr)]
        train_532.extend(init_train_off)
        train_589.extend(init_train_on)
        train_638.extend(init_train_off)
    if init_color == 638:
        train_532.extend(init_train_off)
        train_589.extend(init_train_off)
        train_638.extend(init_train_on)
        
    train_532.extend(wait_train)
    train_589.extend(wait_train)
    train_638.extend(wait_train)
    
    # 1st time: SIGNAL add the pulse pulse segment
    pulse_train_on = [(test_pulse_time, HIGH)]
    pulse_train_off = [(test_pulse_time, LOW)]
    if test_color == 532:
        train_532.extend(pulse_train_on)
        train_589.extend(pulse_train_off)
        train_638.extend(pulse_train_off)
    if test_color == 589:
        pulse_train_on = [(test_pulse_time, aom_ao_589_pwr)]
        train_532.extend(pulse_train_off)
        train_589.extend(pulse_train_on)
        train_638.extend(pulse_train_off)
    if test_color == 638:
        train_532.extend(pulse_train_off)
        train_589.extend(pulse_train_off)
        train_638.extend(pulse_train_on)
        
    train_532.extend(wait_train)
    train_589.extend(wait_train)
    train_638.extend(wait_train)
    
    # 1st time: add the readout pulse segment
    read_train_on = [(readout_time, HIGH)]
    read_train_off = [(readout_time, LOW)]
    if read_color == 532:
        train_532.extend(read_train_on)
        train_589.extend(read_train_off)
        train_638.extend(read_train_off)
    if read_color == 589:
        read_train_on = [(readout_time, aom_ao_589_pwr)]
        train_532.extend(read_train_off)
        train_589.extend(read_train_on)
        train_638.extend(read_train_off)
    if read_color == 638:
        train_532.extend(pulse_train_off)
        train_589.extend(pulse_train_off)
        train_638.extend(read_train_off)
        
    # 2nd time: add the initialization pulse segment
    init_train_on = [(init_pulse_time, HIGH)]
    init_train_off = [(init_pulse_time, LOW)]
    if init_color == 532:
        train_532.extend(init_train_on)
        train_589.extend(init_train_off)
        train_638.extend(init_train_off)
    if init_color == 589:
        init_train_on = [(init_pulse_time, aom_ao_589_pwr)]
        train_532.extend(init_train_off)
        train_589.extend(init_train_on)
        train_638.extend(init_train_off)
    if init_color == 638:
        train_532.extend(init_train_off)
        train_589.extend(init_train_off)
        train_638.extend(init_train_on)
        
    train_532.extend(wait_train)
    train_589.extend(wait_train)
    train_638.extend(wait_train)
    
    # 2nd time: REFERENCE add the pulse pulse segment, just wait during pulse time
    pulse_train_off = [(test_pulse_time, LOW)]
    train_532.extend(pulse_train_off)
    train_589.extend(pulse_train_off)
    train_638.extend(pulse_train_off)
        
    train_532.extend(wait_train)
    train_589.extend(wait_train)
    train_638.extend(wait_train)
    
    # 2nd time: add the readout pulse segment
    read_train_on = [(readout_time, HIGH)]
    read_train_off = [(readout_time, LOW)]
    if read_color == 532:
        train_532.extend(read_train_on)
        train_589.extend(read_train_off)
        train_638.extend(read_train_off)
    if read_color == 589:
        read_train_on = [(readout_time, aom_ao_589_pwr)]
        train_532.extend(read_train_off)
        train_589.extend(read_train_on)
        train_638.extend(read_train_off)
    if read_color == 638:
        train_532.extend(pulse_train_off)
        train_589.extend(pulse_train_off)
        train_638.extend(read_train_off)
        
    train_532.extend([(100, LOW)])
    train_589.extend([(100, LOW)])
    train_638.extend([(100, LOW)])

    seq.setDigital(pulser_do_532_aom, train_532)
    seq.setAnalog(pulser_ao_589_aom, train_589)
    seq.setDigital(pulser_do_638_aom, train_638)    
    
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

    args = [1000,500, 200, 100, 0, 0, 0, 638, 532, 638, 0, 0.7]
    seq, final, _ = get_seq(wiring, args)
    seq.plot()