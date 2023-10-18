# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 21:24:36 2019

11/11/2021: added readout_on_pulse_ind, so that you can choose whether to 
readout on first of second pulse.
@author: mccambria
"""

from pulsestreamer import Sequence
from pulsestreamer import OutputState
import utils.tool_belt as tool_belt
import numpy

LOW = 0
HIGH = 1


def get_seq(pulse_streamer, config, args):

    # Unpack the args
    prep_time, test_time, readout_time, \
        prep_laser_key, test_laser_key, readout_laser_key,\
      prep_laser_power, test_laser_power, read_laser_power, apd_index  = args

    # Get what we need out of the wiring dictionary
    pulser_wiring = config['Wiring']['PulseStreamer']
    pulser_do_daq_clock = pulser_wiring['do_sample_clock']
    pulser_do_daq_gate = pulser_wiring['do_apd_{}_gate'.format(apd_index)]
    
    galvo_move_time = config['Positioning']['xy_small_response_delay']
    prep_aom_delay_time = config['Optics'][prep_laser_key]['delay']
    read_aom_delay_time = config['Optics'][readout_laser_key]['delay']
    if test_laser_key == None:
        test_aom_delay_time = 0
    else:
        test_aom_delay_time = config['Optics'][test_laser_key]['delay']
    
    # Convert the 32 bit ints into 64 bit ints
    prep_time = numpy.int64(prep_time)
    test_time = numpy.int64(test_time)
    readout_time = numpy.int64(readout_time)
    
    intra_pulse_delay = config['CommonDurations']['cw_meas_buffer']
    
    if prep_laser_key == test_laser_key and prep_laser_key == readout_laser_key:
        total_delay = prep_aom_delay_time
    elif prep_laser_key == readout_laser_key and prep_laser_key != test_laser_key:
        total_delay = prep_aom_delay_time + test_aom_delay_time
    elif prep_laser_key != readout_laser_key and prep_laser_key == test_laser_key:
        total_delay = prep_aom_delay_time + read_aom_delay_time
    elif prep_laser_key != readout_laser_key and readout_laser_key == test_laser_key:
        total_delay = prep_aom_delay_time + read_aom_delay_time
    else:
        total_delay = prep_aom_delay_time + test_aom_delay_time + read_aom_delay_time


    period = galvo_move_time + total_delay + prep_time + test_time + readout_time +\
                                        intra_pulse_delay*2 + 300
        
    #%% Define the sequence
    seq = Sequence()

    # Clock
    train = [(galvo_move_time + total_delay + prep_time + intra_pulse_delay + test_time+ \
              intra_pulse_delay + readout_time + 100, LOW), (100, HIGH), (100, LOW)]
    seq.setDigital(pulser_do_daq_clock, train)

    # APD gate
    train = [(galvo_move_time + total_delay + prep_time + intra_pulse_delay + \
              test_time + intra_pulse_delay, LOW), 
             (readout_time, HIGH), (300, LOW)]
    seq.setDigital(pulser_do_daq_gate, train)
    
    # Handle the case where we don't want to pulse a laser for the test, just wait that time
    if test_laser_key == None:
        if prep_laser_key == readout_laser_key:
            laser_powers = [prep_laser_power, read_laser_power]
            
            train = [(galvo_move_time, LOW),
                     (prep_time, HIGH),
                     (intra_pulse_delay, LOW),
                     (test_time, LOW),
                     (intra_pulse_delay, LOW), 
                     (readout_time, HIGH), 
                     (100, LOW)]
            tool_belt.process_laser_seq(pulse_streamer, seq, config,
                                    readout_laser_key, laser_powers, train)
            
        elif prep_laser_key != readout_laser_key:
            train = [(total_delay -prep_aom_delay_time + galvo_move_time, LOW),
                     (prep_time, HIGH),
                     (intra_pulse_delay, LOW),
                     (test_time, LOW),
                     (intra_pulse_delay, LOW), 
                     (readout_time, LOW), 
                     (100, LOW)]
            tool_belt.process_laser_seq(pulse_streamer, seq, config,
                                        prep_laser_key, [prep_laser_power], train)
        
            train = [(total_delay -read_aom_delay_time + galvo_move_time, LOW),
                     (prep_time, LOW),
                     (intra_pulse_delay, LOW),
                     (test_time, LOW),
                     (intra_pulse_delay, LOW), 
                     (readout_time, HIGH), 
                     (100, LOW)]
            tool_belt.process_laser_seq(pulse_streamer, seq, config,
                                    readout_laser_key, [read_laser_power], train)
          
    # Otherwise, we have the normal possibilities of combining three lasers
    else:
        if prep_laser_key == test_laser_key and prep_laser_key == readout_laser_key:
            laser_powers = [prep_laser_power, test_laser_power, read_laser_power]
            
            train = [(galvo_move_time, LOW),
                     (prep_time, HIGH),
                     (intra_pulse_delay, LOW),
                      (test_time, HIGH), 
                     (intra_pulse_delay, LOW), 
                     (readout_time, HIGH), 
                     (100, LOW)]
            tool_belt.process_laser_seq(pulse_streamer, seq, config,
                                    readout_laser_key, laser_powers, train)
        
        elif prep_laser_key == readout_laser_key and prep_laser_key != test_laser_key:
            
            train = [(total_delay -prep_aom_delay_time + galvo_move_time, LOW),
                     (prep_time, HIGH),
                     (intra_pulse_delay, LOW),
                     (test_time, LOW),
                     (intra_pulse_delay, LOW), 
                     (readout_time, HIGH), 
                     (100, LOW)]
            tool_belt.process_laser_seq(pulse_streamer, seq, config,
                                    readout_laser_key, [prep_laser_power, read_laser_power], train)
            
            train = [(total_delay -prep_aom_delay_time + galvo_move_time, LOW),
                     (prep_time, LOW),
                     (intra_pulse_delay, LOW),
                     (test_time, HIGH),
                     (intra_pulse_delay, LOW), 
                     (readout_time, LOW), 
                     (100, LOW)]
            tool_belt.process_laser_seq(pulse_streamer, seq, config,
                                    test_laser_key, [test_laser_power], train)
            
            
        elif prep_laser_key != readout_laser_key and prep_laser_key == test_laser_key:
            train = [(total_delay -test_aom_delay_time + galvo_move_time, LOW),
                     (prep_time, HIGH),
                     (intra_pulse_delay, LOW),
                     (test_time, HIGH),
                     (intra_pulse_delay, LOW), 
                     (readout_time, LOW), 
                     (100, LOW)]
            tool_belt.process_laser_seq(pulse_streamer, seq, config,
                                    prep_laser_key, [prep_laser_power, test_laser_power], train)
            
            train = [(total_delay -read_aom_delay_time + galvo_move_time, LOW),
                     (prep_time, LOW),
                     (intra_pulse_delay, LOW),
                     (test_time, LOW),
                     (intra_pulse_delay, LOW), 
                     (readout_time, HIGH), 
                     (100, LOW)]
            tool_belt.process_laser_seq(pulse_streamer, seq, config,
                                    readout_laser_key, [read_laser_power], train)
            
                    
        elif prep_laser_key != readout_laser_key and readout_laser_key == test_laser_key:
            train = [(total_delay -test_aom_delay_time + galvo_move_time, LOW),
                     (prep_time, LOW),
                     (intra_pulse_delay, LOW),
                     (test_time, HIGH),
                     (intra_pulse_delay, LOW), 
                     (readout_time, HIGH), 
                     (100, LOW)]
            tool_belt.process_laser_seq(pulse_streamer, seq, config,
                                    readout_laser_key, [test_laser_power, read_laser_power], train)
            
            train = [(total_delay -prep_aom_delay_time + galvo_move_time, LOW),
                     (prep_time, HIGH),
                     (intra_pulse_delay, LOW),
                     (test_time, LOW),
                     (intra_pulse_delay, LOW), 
                     (readout_time, LOW), 
                     (100, LOW)]
            tool_belt.process_laser_seq(pulse_streamer, seq, config,
                                    prep_laser_key, [prep_laser_power], train)
        
        else:
            train = [(total_delay -prep_aom_delay_time + galvo_move_time, LOW),
                     (prep_time, HIGH),
                     (intra_pulse_delay, LOW),
                     (test_time, LOW),
                     (intra_pulse_delay, LOW), 
                     (readout_time, LOW), 
                     (100, LOW)]
            tool_belt.process_laser_seq(pulse_streamer, seq, config,
                                    prep_laser_key, [prep_laser_power], train)
            
            train = [(total_delay -test_aom_delay_time + galvo_move_time, LOW),
                     (prep_time, LOW),
                     (intra_pulse_delay, LOW),
                     (test_time, HIGH),
                     (intra_pulse_delay, LOW), 
                     (readout_time, LOW), 
                     (100, LOW)]
            tool_belt.process_laser_seq(pulse_streamer, seq, config,
                                    test_laser_key, [test_laser_power], train)
            
            train = [(total_delay -read_aom_delay_time + galvo_move_time, LOW),
                     (prep_time, LOW),
                     (intra_pulse_delay, LOW),
                     (test_time, LOW),
                     (intra_pulse_delay, LOW), 
                     (readout_time, HIGH), 
                     (100, LOW)]
            tool_belt.process_laser_seq(pulse_streamer, seq, config,
                                    readout_laser_key, [read_laser_power], train)

        
    final_digital = []
    final = OutputState(final_digital, 0.0, 0.0)

    return seq, final, [period]


if __name__ == '__main__':
    config = tool_belt.get_config_dict()
    # args = [200000.0, 1000000.0, 50000.0, 'cobolt_638', 'integrated_520', 'cobolt_638', 0.67, None, 0.55, 1]
    args = [200000.0, 1000000.0, 50000.0, 'cobolt_638', None, 'cobolt_638', 0.67, 0.67, 0.55, 1]
    seq = get_seq(None, config, args)[0]
    seq.plot()
