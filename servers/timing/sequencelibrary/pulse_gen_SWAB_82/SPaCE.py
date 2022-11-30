# -*- coding: utf-8 -*-
"""
Created on Sat Mar  24 08:34:08 2020

Thsi file is for use with the isolate_nv_charge_dynamics_moving_target' routine.

This sequence has three pulses, seperated by wait times that allow time for
the galvo to move. We also have two clock pulses instructing the galvo to move, 
followed by a clock pulse at the end of the sequence to signifiy the counts to read.

@author: Aedan
"""

from pulsestreamer import Sequence
from pulsestreamer import OutputState
import utils.tool_belt as tool_belt
import numpy

LOW = 0
HIGH = 1

def get_seq(pulse_streamer, config, args):

    # %% Parse wiring and args

    # The first 2 args are ns durations and we need them as int64s
    durations = []
    for ind in range(3):
        durations.append(numpy.int64(args[ind]))

    # Unpack the durations
    initialization_time, pulse_time, readout_time = durations
                
    init_power = args[3]
    test_power = args[4]
    read_power = args[5]

    # Get the APD index
    apd_index = args[6]

    init_color = args[7]
    pulse_color = args[8]
    read_color = args[9]
    
   # compare objective peizo delay (make this more general)
    galvo_move_time = config['Positioning']['xy_large_response_delay']
    galvo_move_time = numpy.int64(galvo_move_time*2)
    
    # Get what we need out of the wiring dictionary
    pulser_wiring = config['Wiring']['PulseStreamer']
    
    pulser_do_apd_gate = pulser_wiring['do_apd_{}_gate'.format(apd_index)]
    pulser_do_clock = pulser_wiring['do_sample_clock']
    # pulser_do_532_aom = pulser_wiring['do_laserglow_532_dm']
    # pulser_ao_589_aom = pulser_wiring['ao_laserglow_589_am']
    # pulser_do_638_aom = pulser_wiring['do_cobolt_638_dm']
    
    green_laser_delay = config['Optics']['integrated_520']['delay']
    yellow_laser_delay = config['Optics']['laserglow_589']['delay']
    red_laser_delay = config['Optics']['cobolt_638']['delay']
    
    
    total_laser_delay = green_laser_delay + yellow_laser_delay + red_laser_delay

    # %% Calclate total period.
    
#    We're going to readout the entire sequence, including the clock pulse
    period = total_laser_delay + initialization_time + pulse_time + readout_time \
        + 2 * galvo_move_time + 3* 100
    
    # %% Define the sequence

    seq = Sequence()
    
    # APD 
#    train = [(period - readout_time - 100, LOW), (readout_time, HIGH), (100, LOW)]
#    train = [(readout_time, HIGH), (100, LOW)]
    readout_50th = int(readout_time/50)*0 #time resolved readout. Now not doing anything with *0
    train = [(total_laser_delay, LOW), (initialization_time, LOW),
             (100 + galvo_move_time, LOW), (pulse_time, LOW ),
             (100 + galvo_move_time - readout_50th, LOW), (readout_time + 2*readout_50th, HIGH),
             (100, LOW)]
    seq.setDigital(pulser_do_apd_gate, train)
    
    # clock 
    # I needed to add 100 ns between the redout and the clock pulse, otherwise 
    # the tagger misses some of the gate open/close clicks
    train = [(total_laser_delay + initialization_time+ 100, LOW),(100, HIGH),
             (galvo_move_time + pulse_time, LOW), (100, HIGH), 
             (galvo_move_time + readout_time, LOW), (100, HIGH),
             (100, LOW)] 
#    train = [(period + 100, LOW), (100, HIGH), (100, LOW)]
    seq.setDigital(pulser_do_clock, train)
    
#    train = [(period, HIGH)]
#    seq.setDigital(pulser_do_532_aom, train)
    
    # start each laser sequence
    green_powers = []
    yellow_powers = []
    red_powers = []
    
    train_532 = [(total_laser_delay - green_laser_delay , LOW)]
    train_589 = [(total_laser_delay - yellow_laser_delay, LOW)]
    train_638 = [(total_laser_delay - red_laser_delay, LOW)]
   
    galvo_delay_train = [(100 + galvo_move_time, LOW)]
    
    # add the initialization pulse segment
    init_train_on = [(initialization_time, HIGH)]
    init_train_off = [(initialization_time, LOW)]
    if init_color == 520 :
        green_powers.append(init_power)
        train_532.extend(init_train_on)
        train_589.extend(init_train_off)
        train_638.extend(init_train_off)
    if init_color == 589:
        yellow_powers.append(init_power)
        # init_train_on = [(initialization_time, aom_ao_589_pwr)]
        train_532.extend(init_train_off)
        train_589.extend(init_train_on)
        train_638.extend(init_train_off)
    if init_color == 638:
        red_powers.append(init_power)
        train_532.extend(init_train_off)
        train_589.extend(init_train_off)
        train_638.extend(init_train_on)
        
    train_532.extend(galvo_delay_train)
    train_589.extend(galvo_delay_train)
    train_638.extend(galvo_delay_train)
    
    # add the pulse pulse segment
    pulse_train_on = [(pulse_time, HIGH)]
    pulse_train_off = [(pulse_time, LOW)]
    if pulse_color == 520:
        green_powers.append(test_power)
        train_532.extend(pulse_train_on)
        train_589.extend(pulse_train_off)
        train_638.extend(pulse_train_off)
    if pulse_color == 589:
        yellow_powers.append(test_power)
        # pulse_train_on = [(pulse_time, aom_ao_589_pwr)]
        train_532.extend(pulse_train_off)
        train_589.extend(pulse_train_on)
        train_638.extend(pulse_train_off)
    if pulse_color == 638:
        red_powers.append(test_power)
        train_532.extend(pulse_train_off)
        train_589.extend(pulse_train_off)
        train_638.extend(pulse_train_on)
        
    train_532.extend(galvo_delay_train)
    train_589.extend(galvo_delay_train)
    train_638.extend(galvo_delay_train)
    
    # add the readout pulse segment
    read_train_on = [(readout_time, HIGH)]
    read_train_off = [(readout_time, LOW)]
    if read_color == 520:
        green_powers.append(read_power)
        train_532.extend(read_train_on)
        train_589.extend(read_train_off)
        train_638.extend(read_train_off)
    if read_color == 589:
        yellow_powers.append(read_power)
        # read_train_on = [(readout_time, aom_ao_589_pwr)]
        train_532.extend(read_train_off)
        train_589.extend(read_train_on)
        train_638.extend(read_train_off)
    if read_color == 638:
        red_powers.append(read_power)
        train_532.extend(read_train_off)
        train_589.extend(read_train_off)
        train_638.extend(read_train_on)
        
    train_532.extend([(100, LOW)])
    train_589.extend([(100, LOW)])
    train_638.extend([(100, LOW)])
    
        #######################
    # train_532.extend([(1E7 * 21 * 2, HIGH),(100, LOW)])
        #######################

    # seq.setDigital(pulser_do_532_aom, train_532)
    
    # fix this so it isn't hard coded in
    # print(red_powers)
    tool_belt.process_laser_seq(pulse_streamer, seq, config,
                            'integrated_520', green_powers, train_532)
    tool_belt.process_laser_seq(pulse_streamer, seq, config,
                            'laserglow_589', yellow_powers, train_589)
    tool_belt.process_laser_seq(pulse_streamer, seq, config,
                            'cobolt_638', red_powers, train_638)
        
    final_digital = []
    final = OutputState(final_digital, 0.0, 0.0)
    return seq, final, [period]

if __name__ == '__main__':
    config = tool_belt.get_config_dict()

    # seq_args = [1000.0, 100000.0, 100000.0, 0.15, 0, 638, 532, 589]
    seq_args = [10000.0, 3500000.0, 5000000.0, None, 0.56, 0.15, 1, 520, 638, 589]
    
    seq = get_seq(None, config, seq_args)[0]
    seq.plot()
