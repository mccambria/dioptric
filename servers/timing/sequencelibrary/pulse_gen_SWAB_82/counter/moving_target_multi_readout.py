# -*- coding: utf-8 -*-
"""
Created on Sat Mar  24 08:34:08 2020

Thsi file is for use with the 'moving_target_mlti_readout' routine.

This sequence pulses the laser r times, where r is the number of readout NVs,
then pulses the laser on some other coordinate, and then pulses a readout
laser r times.

@author: Aedan
"""

from pulsestreamer import Sequence
from pulsestreamer import OutputState
import numpy

LOW = 0
HIGH = 1

def get_seq(pulser_wiring, args):

    # %% Parse wiring and args

    # The first 2 args are ns durations and we need them as int64s
    durations = []
    for ind in range(7):
        durations.append(numpy.int64(args[ind]))

    # Unpack the durations
    initialization_time, pulse_time, readout_time, \
        delay_532, delay_589, delay_638, \
        galvo_time = durations
                
    aom_ao_589_pwr = args[7]

    # Get the APD index
    apd_index = args[8]
    
    init_color = args[9]
    pulse_color = args[10]
    read_color = args[11]
    
    num_readout_coords = args[12]

    # Get what we need out of the wiring dictionary
    pulser_do_apd_gate = pulser_wiring['do_apd_{}_gate'.format(apd_index)]
    pulser_do_clock = pulser_wiring['do_sample_clock']
    pulser_do_532_aom = pulser_wiring['do_532_aom']
    pulser_ao_589_aom = pulser_wiring['ao_589_aom']
    pulser_do_638_aom = pulser_wiring['do_638_laser']
    
    total_laser_delay = delay_532 + delay_589 + delay_638

    # %% Calclate total period.
    
#    We're going to readout the entire sequence, including the clock pulse
    period = total_laser_delay + num_readout_coords*(initialization_time + readout_time) \
     + pulse_time + (num_readout_coords-1)*2*(galvo_time + 100) + 3 * (galvo_time+100)    
    
    # %% Define the sequence

    seq = Sequence()
    
    # APD  (for now,only open gate on readout pulses)
    pre_readout_time = num_readout_coords*(initialization_time + galvo_time + 100) + pulse_time + galvo_time + 100
    train = [(total_laser_delay + pre_readout_time, LOW)]
    for r in range(num_readout_coords):
        train.extend([(readout_time, HIGH), (100 + galvo_time, LOW)])
    seq.setDigital(pulser_do_apd_gate, train)
    
    # clock 
    # I needed to add 200 ns between the readout and the clock pulse, otherwise 
    # the tagger misses some of the gate open/close clicks
    train = [(total_laser_delay + 200, LOW)] 
    for r in range(num_readout_coords):
        train.extend([(initialization_time, LOW), (100, HIGH), (galvo_time, LOW)])
    train.extend([(pulse_time, LOW), (100, HIGH), (galvo_time, LOW)])
    for r in range(num_readout_coords):
        train.extend([(readout_time, LOW), (100, HIGH), (galvo_time, LOW)])
    seq.setDigital(pulser_do_clock, train)

    # start each laser sequence
    train_532 = [(total_laser_delay - delay_532 , LOW)]
    train_589 = [(total_laser_delay - delay_589, LOW)]
    train_638 = [(total_laser_delay - delay_638, LOW)]
   
    galvo_delay_train = [(100 + galvo_time, LOW)]
    
    # add the initialization pulse segment
    init_train_on = [(initialization_time, HIGH)]
    init_train_off = [(initialization_time, LOW)]
    for r in range(num_readout_coords):
        if init_color == 532:
            train_532.extend(init_train_on)
            train_589.extend(init_train_off)
            train_638.extend(init_train_off)
        if init_color == 589:
            init_train_on = [(initialization_time, aom_ao_589_pwr)]
            train_532.extend(init_train_off)
            train_589.extend(init_train_on)
            train_638.extend(init_train_off)
        if init_color == 638:
            train_532.extend(init_train_off)
            train_589.extend(init_train_off)
            train_638.extend(init_train_on)
        
        train_532.extend(galvo_delay_train)
        train_589.extend(galvo_delay_train)
        train_638.extend(galvo_delay_train)
    
    # add the pulse pulse segment
    pulse_train_on = [(pulse_time, HIGH)]
    pulse_train_off = [(pulse_time, LOW)]
    if pulse_color == 532:
        train_532.extend(pulse_train_on)
        train_589.extend(pulse_train_off)
        train_638.extend(pulse_train_off)
    if pulse_color == 589:
        pulse_train_on = [(pulse_time, aom_ao_589_pwr)]
        train_532.extend(pulse_train_off)
        train_589.extend(pulse_train_on)
        train_638.extend(pulse_train_off)
    if pulse_color == 638:
        train_532.extend(pulse_train_off)
        train_589.extend(pulse_train_off)
        train_638.extend(pulse_train_on)
        
    train_532.extend(galvo_delay_train)
    train_589.extend(galvo_delay_train)
    train_638.extend(galvo_delay_train)
    
    # add the readout pulse segment
    read_train_on = [(readout_time, HIGH)]
    read_train_off = [(readout_time, LOW)]
    for r in range(num_readout_coords):
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
            train_638.extend(read_train_on)
            
        train_532.extend(galvo_delay_train)
        train_589.extend(galvo_delay_train)
        train_638.extend(galvo_delay_train)
        
        

    seq.setDigital(pulser_do_532_aom, train_532)
    seq.setAnalog(pulser_ao_589_aom, train_589)
    seq.setDigital(pulser_do_638_aom, train_638)    
        
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

    seq_args = [1000, 1500, 3000, 0, 0, 0, 500, 0.5, 0, 532, 532, 638, 3]
#    seq_args = [100000, 5000000, 100000, 0, 0, 0, 2000000, 0.3, 0, 532, 532, 589, 1]

    seq, final, ret_vals = get_seq(wiring, seq_args)
    seq.plot()
