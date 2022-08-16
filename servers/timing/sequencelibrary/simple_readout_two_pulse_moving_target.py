# -*- coding: utf-8 -*-
"""
Created on Sat Mar  24 08:34:08 2020

Thsi file is for use with the 'moving_target_siv_init' routine.

This sequence has two pulses, seperated by wait times that allow time for
the galvo to move. We also have two clock pulses instructing the galvo to move 
and to collect counts. The first is for a remote pulse and the second is during
a readout pulse on an NV.

@author: Carter Fox
"""

from pulsestreamer import Sequence
from pulsestreamer import OutputState
import numpy
import utils.tool_belt as tool_belt
LOW = 0
HIGH = 1

def get_seq(pulse_streamer, config, args):

    # %% Parse wiring and args
    # The first 2 args are ns durations and we need them as int64s
    
    init_pulse_time, readout_time, init_laser_key, readout_laser_key,init_laser_power, read_laser_power, readout_on_pulse_ind, apd_index  = args
      #NEED REST OF FILE TO WORK WITH THESE ARGs
    
    pulser_wiring = config['Wiring']['PulseStreamer']
    
    positioning = config['Positioning']

    pulse_time = init_pulse_time
    readout_time = readout_time 
    delay_532 = config['Optics'][init_laser_key]['delay']
    delay_589 = config['Optics'][readout_laser_key]['delay']
    delay_638 = config['Optics'][init_laser_key]['delay']
    if 'xy_small_response_delay' in positioning:
        galvo_time = positioning['xy_small_response_delay']
    else:
        galvo_time = positioning['xy_delay']
                
    aom_ao_589_pwr = read_laser_power
    # green_pulse_pwr = init_laser_power
    # green_readout_pwr = init_laser_power

    # Get the APD index
    # apd_index = args[9]
    
    # pulse_color = args[10]
    # read_color = args[11]

    # Get what we need out of the wiring dictionary
    pulser_do_apd_gate = pulser_wiring['do_apd_{}_gate'.format(apd_index)]
    pulser_do_clock = pulser_wiring['do_sample_clock']
    pulser_do_532_aom = pulser_wiring['do_laserglow_532_dm']
    pulser_ao_589_aom = pulser_wiring['ao_laserglow_589_am']
    pulser_do_638 = pulser_wiring['do_cobolt_638_dm']
    # pulser_ao_515_aom = pulser_wiring['do_cobolt_515_dm']
    
    if '638' in init_laser_key:
        total_laser_delay = delay_589 + delay_638
    elif '532' in init_laser_key:
        total_laser_delay = delay_532 + delay_589
        
    
    # %% Calclate total period.
    
#    We're going to readout the entire sequence, including the clock pulse
    period = total_laser_delay  + pulse_time + readout_time \
        + 2 * galvo_time + 2* 100
    
    # %% Define the sequence

    seq = Sequence()
    
    # APD 
    train = [(total_laser_delay, LOW),(pulse_time, LOW),
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
    # train_515 = [(total_laser_delay - delay_532 , LOW)]
    train_532 = [(total_laser_delay - delay_532 , LOW)]
    train_589 = [(total_laser_delay - delay_589, LOW)]
    train_638 = [(total_laser_delay - delay_638, LOW)]
   
    galvo_delay_train = [(100 + galvo_time, LOW)]
    
    # add the pulse pulse segment
    pulse_train_on = [(pulse_time, HIGH)]
    pulse_train_off = [(pulse_time, LOW)]
    # if '515a' in init_laser_key:
    #     pulse_train_on = [(pulse_time, green_pulse_pwr)]
    #     train_515.extend(pulse_train_on)
    #     train_532.extend(pulse_train_off)
    #     train_589.extend(pulse_train_off)
    #     train_638.extend(pulse_train_off)
    if '532' in init_laser_key:
        # train_515.extend(pulse_train_off)
        train_532.extend(pulse_train_on)
        train_589.extend(pulse_train_off)
        train_638.extend(pulse_train_off)
    if '589' in init_laser_key:
        pulse_train_on = [(pulse_time, aom_ao_589_pwr)]
        # train_515.extend(pulse_train_off)
        train_532.extend(pulse_train_off)
        train_589.extend(pulse_train_on)
        train_638.extend(pulse_train_off)
    if '638'in init_laser_key:
        # train_515.extend(pulse_train_off)
        train_532.extend(pulse_train_off)
        train_589.extend(pulse_train_off)
        train_638.extend(pulse_train_on)
        
    # train_515.extend(galvo_delay_train)
    train_532.extend(galvo_delay_train)
    train_589.extend(galvo_delay_train)
    train_638.extend(galvo_delay_train)
    
    # add the readout pulse segment
    read_train_on = [(readout_time, HIGH)]
    read_train_off = [(readout_time, LOW)]
    # if '515a' in readout_laser_key:
    #     read_train_on = [(readout_time, green_readout_pwr)]
    #     train_515.extend(read_train_on)
    #     train_532.extend(read_train_off)
    #     train_589.extend(read_train_off)
    #     train_638.extend(read_train_off)
    if '532' in readout_laser_key:
        # train_515.extend(read_train_off)
        train_532.extend(read_train_on)
        train_589.extend(read_train_off)
        train_638.extend(read_train_off)
    if '589' in readout_laser_key:
        read_train_on = [(readout_time, aom_ao_589_pwr)]
        # train_515.extend(read_train_off)
        train_532.extend(read_train_off)
        train_589.extend(read_train_on)
        train_638.extend(read_train_off)
    if '638' in readout_laser_key:
        # train_515.extend(read_train_off)
        train_532.extend(read_train_off)
        train_589.extend(read_train_off)
        train_638.extend(read_train_on)
        
    # train_515.extend([(100, LOW)])
    train_532.extend([(100, LOW)])
    train_589.extend([(100, LOW)])
    train_638.extend([(100, LOW)])
        
        

    # seq.setAnalog(pulser_ao_515_aom, train_515)
    seq.setDigital(pulser_do_532_aom, train_532)
    seq.setAnalog(pulser_ao_589_aom, train_589)
    seq.setDigital(pulser_do_638, train_638)    
        
    final_digital = []
    final = OutputState(final_digital, 0.0, 0.0)
    return seq, final, [period] 

if __name__ == '__main__':
    # wiring = {'ao_laserglow_589_am': 0,
    #           'do_apd_0_gate':1,
    #           'do_apd_1_gate':1,
    #           'do_arb_wave_trigger':2,
    #           'do_cobolt_515_dm':5,
    #           'do_cobolt_638_dm':5,
    #           'do_integrated_520_dm':5,
    #           'do_laserglow_532_dm':3,
    #           'do_laserglow_589_dm':6,
    #           'do_sample_clock': 0,
    #           'do_signal_generator_sg394_gate': 7,
    #           'do_signal_generator_tsg4104a_gate': 4
    #           }
# init_pulse_time, readout_time, init_laser_key, readout_laser_key,\
  # init_laser_power, read_laser_power, readout_on_pulse_ind, apd_index  = args
    config = tool_belt.get_config_dict()
    # tool_belt.set_delays_to_zero(config)
    seq_args = [50e6, 100e6, "cobolt_638", "laserglow_589", "nd_0", 1.0, 2, 0]
    # seq_args = [5000000, 30000000, 140, 1080, 90, 2000000, 0.3, 0.65, 0.65, 0, '515a', 589]

    seq, final, ret_vals = get_seq(None, config, seq_args)
    seq.plot()
