# -*- coding: utf-8 -*-
"""
Created on Sat Mar  24 08:34:08 2020

Thsi file is for use with the 'moving_target_siv_init' routine.

This sequence has two pulses, seperated by wait times that allow time for
the galvo to move. We also have two clock pulses instructing the galvo to move 
and to collect counts. The first is for a remote pulse and the second is during
a readout pulse on an NV.

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
    for ind in range(6):
        durations.append(numpy.int64(args[ind]))

    # Unpack the durations
    pulse_time, readout_time, \
        delay_532, delay_589, delay_638, \
        galvo_time = durations
                
    aom_ao_589_pwr = args[6]
    green_pulse_pwr = args[7]
    green_readout_pwr = args[8]

    # Get the APD index
    apd_index = args[9]
    
    pulse_color = args[10]
    read_color = args[11]

    # Get what we need out of the wiring dictionary
    pulser_do_apd_gate = pulser_wiring['do_apd_{}_gate'.format(apd_index)]
    pulser_do_clock = pulser_wiring['do_sample_clock']
    pulser_do_532_aom = pulser_wiring['do_532_aom']
    pulser_ao_589_aom = pulser_wiring['ao_589_aom']
    pulser_do_638_aom = pulser_wiring['do_638_laser']
    pulser_ao_515_aom = pulser_wiring['ao_515_laser']
    
    total_laser_delay = delay_532 + delay_589 + delay_638

    # %% Calclate total period.
    
#    We're going to readout the entire sequence, including the clock pulse
    period = total_laser_delay  + pulse_time + readout_time \
        + 2 * galvo_time + 2* 100
    
    # %% Define the sequence

    seq = Sequence()
    
    # APD 
    train = [(total_laser_delay, LOW),(pulse_time, HIGH),
             (100 + galvo_time, LOW), (readout_time, HIGH),
             (100 + galvo_time, LOW), 
             (100, LOW)]
    seq.setDigital(pulser_do_apd_gate, train)
    
    # clock 
    # I needed to add 100 ns between the redout and the clock pulse, otherwise 
    # the tagger misses some of the gate open/close clicks
    train = [(total_laser_delay +  pulse_time + 100, LOW), (100, HIGH), 
             (galvo_time + readout_time, LOW), (100, HIGH),
             (100, LOW)] 
    seq.setDigital(pulser_do_clock, train)
    
    
    # start each laser sequence
    train_515 = [(total_laser_delay - delay_532 , LOW)]
    train_532 = [(total_laser_delay - delay_532 , LOW)]
    train_589 = [(total_laser_delay - delay_589, LOW)]
    train_638 = [(total_laser_delay - delay_638, LOW)]
   
    galvo_delay_train = [(100 + galvo_time, LOW)]
    
    # add the pulse pulse segment
    pulse_train_on = [(pulse_time, HIGH)]
    pulse_train_off = [(pulse_time, LOW)]
    if pulse_color == '515a':
        pulse_train_on = [(pulse_time, green_pulse_pwr)]
        train_515.extend(pulse_train_on)
        train_532.extend(pulse_train_off)
        train_589.extend(pulse_train_off)
        train_638.extend(pulse_train_off)
    if pulse_color == 532:
        train_515.extend(pulse_train_off)
        train_532.extend(pulse_train_on)
        train_589.extend(pulse_train_off)
        train_638.extend(pulse_train_off)
    if pulse_color == 589:
        pulse_train_on = [(pulse_time, aom_ao_589_pwr)]
        train_515.extend(pulse_train_off)
        train_532.extend(pulse_train_off)
        train_589.extend(pulse_train_on)
        train_638.extend(pulse_train_off)
    if pulse_color == 638:
        train_515.extend(pulse_train_off)
        train_532.extend(pulse_train_off)
        train_589.extend(pulse_train_off)
        train_638.extend(pulse_train_on)
        
    train_515.extend(galvo_delay_train)
    train_532.extend(galvo_delay_train)
    train_589.extend(galvo_delay_train)
    train_638.extend(galvo_delay_train)
    
    # add the readout pulse segment
    read_train_on = [(readout_time, HIGH)]
    read_train_off = [(readout_time, LOW)]
    if read_color == '515a':
        read_train_on = [(readout_time, green_readout_pwr)]
        train_515.extend(read_train_on)
        train_532.extend(read_train_off)
        train_589.extend(read_train_off)
        train_638.extend(read_train_off)
    if read_color == 532:
        train_515.extend(read_train_off)
        train_532.extend(read_train_on)
        train_589.extend(read_train_off)
        train_638.extend(read_train_off)
    if read_color == 589:
        read_train_on = [(readout_time, aom_ao_589_pwr)]
        train_515.extend(read_train_off)
        train_532.extend(read_train_off)
        train_589.extend(read_train_on)
        train_638.extend(read_train_off)
    if read_color == 638:
        train_515.extend(read_train_off)
        train_532.extend(read_train_off)
        train_589.extend(read_train_off)
        train_638.extend(read_train_on)
        
    train_515.extend([(100, LOW)])
    train_532.extend([(100, LOW)])
    train_589.extend([(100, LOW)])
    train_638.extend([(100, LOW)])
        
        

    seq.setAnalog(pulser_ao_515_aom, train_515)
    seq.setDigital(pulser_do_532_aom, train_532)
    seq.setAnalog(pulser_ao_589_aom, train_589)
    seq.setDigital(pulser_do_638_aom, train_638)    
        
    final_digital = []
    final = OutputState(final_digital, 0.0, 0.0)
    return seq, final, [period]

if __name__ == '__main__':
    wiring = {'ao_515_laser': 0,
              'ao_589_aom':1,
              'do_sample_clock': 0,
              'do_apd_0_gate': 4,
              'do_532_aom': 3,
              'do_638_laser': 7
              }

#    seq_args = [1000, 1500, 0, 0, 0, 500, 0.5, 1.0, 0.5, 0, 532, 589]
    seq_args = [5000000, 30000000, 140, 1080, 90, 2000000, 0.3, 0.65, 0.65, 0, '515a', 589]

    seq, final, ret_vals = get_seq(wiring, seq_args)
    seq.plot()
