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

    init_time, pulse_time, readout_time, \
        init_laser_key, pulse_laser_key, readout_laser_key,\
        init_laser_power, pulse_laser_power, read_laser_power, \
        incr_movement_delay, settling_time, movement_incr, apd_index  = args
      
    init_time = numpy.int64(init_time)
    pulse_time = numpy.int64(pulse_time)
    readout_time = numpy.int64(readout_time)
    
    settling_time = numpy.int64(settling_time)
    incr_movement_delay = numpy.int64(incr_movement_delay)
    
    # Get what we need out of the wiring dictionary
    pulser_wiring = config['Wiring']['PulseStreamer']
    
    pulser_do_apd_gate = pulser_wiring['do_apd_{}_gate'.format(apd_index)]
    pulser_do_clock = pulser_wiring['do_sample_clock']
    
    init_aom_delay_time = config['Optics'][init_laser_key]['delay']
    pulse_aom_delay_time = config['Optics'][pulse_laser_key]['delay']
    read_aom_delay_time = config['Optics'][readout_laser_key]['delay']
    
    
    if init_laser_key == pulse_laser_key and init_laser_key == readout_laser_key:
        total_delay = init_aom_delay_time
    elif init_laser_key == readout_laser_key and init_laser_key != pulse_laser_key:
        total_delay = init_aom_delay_time + pulse_aom_delay_time
    elif init_laser_key != readout_laser_key and init_laser_key == pulse_laser_key:
        total_delay = init_aom_delay_time + read_aom_delay_time
    elif init_laser_key != readout_laser_key and readout_laser_key == pulse_laser_key:
        total_delay = init_aom_delay_time + read_aom_delay_time
    else:
        total_delay = init_aom_delay_time + pulse_aom_delay_time + read_aom_delay_time
    
    
    # %% Calclate total period.
    
#    We're going to readout the entire sequence, including the clock pulse
    period = total_delay + init_time + pulse_time + readout_time \
        + 2 * ((incr_movement_delay + 100) * movement_incr + settling_time) + 100
    
    # %% Define the sequence

    movement_delay_train = [((incr_movement_delay + 100) * movement_incr + settling_time, LOW)]

    seq = Sequence()
    
    # APD 
#    train = [(period - readout_time - 100, LOW), (readout_time, HIGH), (100, LOW)]
#    train = [(readout_time, HIGH), (100, LOW)]
    train = [(total_delay, LOW), (init_time, HIGH)]
    train.extend(movement_delay_train)
    train.extend([(pulse_time, HIGH)])
    train.extend(movement_delay_train)
    train.extend([(readout_time, HIGH),(100, LOW)])
    seq.setDigital(pulser_do_apd_gate, train)
    
    # clock 
    # I needed to add 100 ns between the redout and the clock pulse, otherwise 
    # the tagger misses some of the gate open/close clicks
    train = [(total_delay + init_time+ 100, LOW)]
    for i in range(movement_incr):
        train.extend([(100, HIGH), (incr_movement_delay, LOW)])
    train.extend([(settling_time + pulse_time, LOW)])
    for i in range(movement_incr):
        train.extend([(100, HIGH), (incr_movement_delay, LOW)])
    train.extend([(settling_time + readout_time, LOW), (100, HIGH), (100, LOW)]) 
    seq.setDigital(pulser_do_clock, train)
    
        
    if init_laser_key == pulse_laser_key and init_laser_key == readout_laser_key:
        laser_powers = [init_laser_power, pulse_laser_power, read_laser_power]
        
        train = [(init_time, HIGH)]
        train.extend(movement_delay_train)
        train.extend([(pulse_time, HIGH)])
        train.extend(movement_delay_train)
        train.extend([(readout_time, HIGH)])
        train.extend([(500, LOW)])
        
        # print(train)
        tool_belt.process_laser_seq(pulse_streamer, seq, config,
                                readout_laser_key, laser_powers, train)
    
    elif init_laser_key == readout_laser_key and init_laser_key != pulse_laser_key:
    
        train = [(total_delay - read_aom_delay_time, LOW), 
                 (init_time, HIGH)]
        train.extend(movement_delay_train)
        train.extend([(pulse_time, LOW)])
        train.extend(movement_delay_train)
        train.extend([(readout_time, HIGH)])
        train.extend([(500, LOW)])
        
        # print(train)
        tool_belt.process_laser_seq(pulse_streamer, seq, config,
                                readout_laser_key, [init_laser_power, read_laser_power], train)
        
        train = [(total_delay - pulse_aom_delay_time, LOW),
                 (init_time, LOW)]
        train.extend(movement_delay_train)
        train.extend([(pulse_time, HIGH)])
        train.extend(movement_delay_train)
        train.extend([(readout_time, LOW)])
        train.extend([(500, LOW)])
        
        # print(train)
        tool_belt.process_laser_seq(pulse_streamer, seq, config,
                                pulse_laser_key, pulse_laser_power, train)
    
    elif init_laser_key != readout_laser_key and init_laser_key == pulse_laser_key:
        train = [(total_delay - init_aom_delay_time, LOW),
                 (init_time, HIGH)]
        train.extend(movement_delay_train)
        train.extend([(pulse_time, HIGH)])
        # train.extend([(pulse_time, LOW)])
        train.extend(movement_delay_train)
        train.extend([(readout_time, LOW)])
        train.extend([(500, LOW)])
        
        tool_belt.process_laser_seq(pulse_streamer, seq, config,
                                # init_laser_key, [init_laser_power], train)
                                init_laser_key, [init_laser_power, pulse_laser_power], train)
        
        train = [(total_delay - read_aom_delay_time, LOW),
                 (init_time, LOW)]
        train.extend(movement_delay_train)
        train.extend([(pulse_time, LOW)])
        train.extend(movement_delay_train)
        train.extend([(readout_time, HIGH)])
        train.extend([(500, LOW)])
        
        tool_belt.process_laser_seq(pulse_streamer, seq, config,
                                readout_laser_key, read_laser_power, train)
        
    elif init_laser_key != readout_laser_key and readout_laser_key == pulse_laser_key:
        
        train = [(total_delay - pulse_aom_delay_time, LOW),
                 (init_time, LOW)]
        train.extend(movement_delay_train)
        train.extend([(pulse_time, HIGH)])
        train.extend(movement_delay_train)
        train.extend([(readout_time, HIGH)])
        train.extend([(500, LOW)])
        
        tool_belt.process_laser_seq(pulse_streamer, seq, config,
                                pulse_laser_key, [pulse_laser_power, read_laser_power], train)
        
        train = [(total_delay - init_aom_delay_time, LOW),
                 (init_time, HIGH)]
        train.extend(movement_delay_train)
        train.extend([(pulse_time, LOW)])
        train.extend(movement_delay_train)
        train.extend([(readout_time, LOW)])
        train.extend([(500, LOW)])
        
        tool_belt.process_laser_seq(pulse_streamer, seq, config,
                                init_laser_key, init_laser_power, train)
        
 
    else:
                
        train = [(total_delay - init_aom_delay_time, LOW),
                 (init_time, HIGH)]
        train.extend(movement_delay_train)
        train.extend([(pulse_time, LOW)])
        train.extend(movement_delay_train)
        train.extend([(readout_time, LOW)])
        train.extend([(500, LOW)])
        
        # print(train)
        tool_belt.process_laser_seq(pulse_streamer, seq, config,
                                init_laser_key, init_laser_power, train)
        
                
        train = [(total_delay - pulse_aom_delay_time, LOW),
                 (init_time, LOW)]
        train.extend(movement_delay_train)
        train.extend([(pulse_time, HIGH)])
        train.extend(movement_delay_train)
        train.extend([(readout_time, LOW)])
        train.extend([(500, LOW)])
        
        tool_belt.process_laser_seq(pulse_streamer, seq, config,
                                pulse_laser_key, pulse_laser_power, train)
                
        train = [(total_delay - read_aom_delay_time, LOW),
                 (init_time, LOW)]
        train.extend(movement_delay_train)
        train.extend([(pulse_time, LOW)])
        train.extend(movement_delay_train)
        train.extend([(readout_time, HIGH)])
        train.extend([(500, LOW)])
        
        tool_belt.process_laser_seq(pulse_streamer, seq, config,
                                readout_laser_key, read_laser_power, train)
        
    final_digital = []
    final = OutputState(final_digital, 0.0, 0.0)
    return seq, final, [period]

if __name__ == '__main__':
    config = tool_belt.get_config_dict()

    # seq_args = [1000.0, 100000.0, 100000.0, 1e5, 1e6, 0.15, 0, 638, 532, 589, 2]
    # seq_args = [100000.0, 100000.0, 2500.0, 
    #             'cobolt_638', 'cobolt_638','cobolt_638',
    #             0.66, 0.5, 1.0, 
    #             80000, 2000000,  3, 0]
    seq_args = [10000.0, 100000, 10000.0, 'integrated_520', 'integrated_520', 'cobolt_638', None, None, 0.565, 48780, 2000000, 41, 1]
    seq = get_seq(None, config, seq_args)[0]
    seq.plot()
