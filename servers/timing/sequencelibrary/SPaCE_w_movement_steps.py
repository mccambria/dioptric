# -*- coding: utf-8 -*-
"""
Created on Sat Mar  24 08:34:08 2020

Thsi file is for use with the' SPaCE' routine.

This sequence has three pulses, seperated by wait times that allow time for
the galvo to move. We also have two clock pulses instructing the galvo to move, 
followed by a clock pulse at the end of the sequence to signifiy the counts to read.

8/30/2021 Modified so that galvo and objective piezo will step incrementally
to desired position, instead of one sudden jump.

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
    
    incr_movement_delay = args[3]
    settling_time = args[4]
                
    aom_ao_589_pwr = args[5]

    # Get the APD index
    apd_index = args[6]

    init_color = args[7]
    pulse_color = args[8]
    read_color = args[9]
    
    movement_incr = args[10]
    
   # compare objective peizo delay (make this more general)
    # galvo_move_time = config['Positioning']['xy_large_response_delay']
    incr_movement_delay = numpy.int64(incr_movement_delay)
    
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
        + 2 * ((incr_movement_delay + 100) * movement_incr + settling_time) + 100
    
    # %% Define the sequence

    movement_delay_train = [((incr_movement_delay + 100) * movement_incr + settling_time, LOW)]

    seq = Sequence()
    
    # APD 
#    train = [(period - readout_time - 100, LOW), (readout_time, HIGH), (100, LOW)]
#    train = [(readout_time, HIGH), (100, LOW)]
    train = [(total_laser_delay, LOW), (initialization_time, HIGH)]
    train.extend(movement_delay_train)
    train.extend([(pulse_time, HIGH)])
    train.extend(movement_delay_train)
    train.extend([(readout_time, HIGH),(100, LOW)])
    seq.setDigital(pulser_do_apd_gate, train)
    
    # clock 
    # I needed to add 100 ns between the redout and the clock pulse, otherwise 
    # the tagger misses some of the gate open/close clicks
    train = [(total_laser_delay + initialization_time+ 100, LOW)]
    for i in range(movement_incr):
        train.extend([(100, HIGH), (incr_movement_delay, LOW)])
    train.extend([(settling_time + pulse_time, LOW)])
    for i in range(movement_incr):
        train.extend([(100, HIGH), (incr_movement_delay, LOW)])
    train.extend([(settling_time + readout_time, LOW), (100, HIGH), (100, LOW)]) 
    seq.setDigital(pulser_do_clock, train)
    
#    train = [(period, HIGH)]
#    seq.setDigital(pulser_do_532_aom, train)
    
    # start each laser sequence
    train_532 = [(total_laser_delay - green_laser_delay , LOW)]
    train_589 = [(total_laser_delay - yellow_laser_delay, LOW)]
    train_638 = [(total_laser_delay - red_laser_delay, LOW)]
   
    
    # add the initialization pulse segment
    init_train_on = [(initialization_time, HIGH)]
    init_train_off = [(initialization_time, LOW)]
    if init_color == 520 :
        train_532.extend(init_train_on)
        train_589.extend(init_train_off)
        train_638.extend(init_train_off)
    if init_color == 589:
        # init_train_on = [(initialization_time, aom_ao_589_pwr)]
        train_532.extend(init_train_off)
        train_589.extend(init_train_on)
        train_638.extend(init_train_off)
    if init_color == 638:
        train_532.extend(init_train_off)
        train_589.extend(init_train_off)
        train_638.extend(init_train_on)
        
    train_532.extend(movement_delay_train)
    train_589.extend(movement_delay_train)
    train_638.extend(movement_delay_train)
    
    # add the pulse pulse segment
    pulse_train_on = [(pulse_time, HIGH)]
    pulse_train_off = [(pulse_time, LOW)]
    if pulse_color == 520:
        train_532.extend(pulse_train_on)
        train_589.extend(pulse_train_off)
        train_638.extend(pulse_train_off)
    if pulse_color == 589:
        # pulse_train_on = [(pulse_time, aom_ao_589_pwr)]
        train_532.extend(pulse_train_off)
        train_589.extend(pulse_train_on)
        train_638.extend(pulse_train_off)
    if pulse_color == 638:
        train_532.extend(pulse_train_off)
        train_589.extend(pulse_train_off)
        train_638.extend(pulse_train_on)
        
    train_532.extend(movement_delay_train)
    train_589.extend(movement_delay_train)
    train_638.extend(movement_delay_train)
    
    # add the readout pulse segment
    read_train_on = [(readout_time, HIGH)]
    read_train_off = [(readout_time, LOW)]
    if read_color == 520:
        train_532.extend(read_train_on)
        train_589.extend(read_train_off)
        train_638.extend(read_train_off)
    if read_color == 589:
        # read_train_on = [(readout_time, aom_ao_589_pwr)]
        train_532.extend(read_train_off)
        train_589.extend(read_train_on)
        train_638.extend(read_train_off)
    if read_color == 638:
        train_532.extend(pulse_train_off)
        train_589.extend(pulse_train_off)
        train_638.extend(read_train_on)
        
    # Add some extra time at the end before next sequence
    train_532.extend([(500, LOW)])
    train_589.extend([(500, LOW)])
    train_638.extend([(500, LOW)])
    
        #######################
    # train_532.extend([(1E7 * 21 * 2, HIGH),(100, LOW)])
        #######################

    # seq.setDigital(pulser_do_532_aom, train_532)
    
    # fix this so it isn't hard coded in
    tool_belt.process_laser_seq(pulse_streamer, seq, config,
                            'integrated_520', None, train_532)
    tool_belt.process_laser_seq(pulse_streamer, seq, config,
                            'laserglow_589', aom_ao_589_pwr, train_589)
    tool_belt.process_laser_seq(pulse_streamer, seq, config,
                            'cobolt_638', [aom_ao_589_pwr, aom_ao_589_pwr], train_638)
        
    final_digital = []
    final = OutputState(final_digital, 0.0, 0.0)
    return seq, final, [period]

if __name__ == '__main__':
    config = tool_belt.get_config_dict()

    # seq_args = [1000.0, 100000.0, 100000.0, 1e5, 1e6, 0.15, 0, 638, 532, 589, 2]
    seq_args = [100000.0, 100000.0, 2500.0, 80000, 2000000, 0.66, 0, 638, 520, 638, 25]
    seq = get_seq(None, config, seq_args)[0]
    seq.plot()
