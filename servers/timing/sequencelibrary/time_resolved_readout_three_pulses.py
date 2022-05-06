# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 21:24:36 2019

A similar sequence file to simple_readout_three_pulses, but this file can vary the 
readout duration of the apd independently from the readout laser duration.

I intend to use this with varying the analog modulated voltage to one of the 
lasers between pulses, so I will need to manually write the sequences, rather 
than use the tool_belt.process_laser_seq

Assumes that readout is on pulse two

@author: mccambria
"""

from pulsestreamer import Sequence
from pulsestreamer import OutputState
import utils.tool_belt as tool_belt
import numpy



def get_seq(pulse_streamer, config, args):

    LOW = 0
    HIGH = 1

    # Unpack the args
    # assumes readout_time >= readout_laser_time
    init_pulse_time, test_pulse_time, readout_time, readout_laser_time, \
        init_laser_key, test_laser_key, readout_laser_key,\
      init_laser_power, test_laser_power, read_laser_power, \
          apd_index  = args

    # Get what we need out of the wiring dictionary
    pulser_wiring = config['Wiring']['PulseStreamer']
    pulser_do_daq_clock = pulser_wiring['do_sample_clock']
    pulser_do_daq_gate = pulser_wiring['do_apd_{}_gate'.format(apd_index)]

    init_pulse_aom_delay_time = config['Optics'][init_laser_key]['delay']
    test_pulse_aom_delay_time = config['Optics'][test_laser_key]['delay']
    read_pulse_aom_delay_time = config['Optics'][readout_laser_key]['delay']
        
    
    init_wavelength = config['Optics'][init_laser_key]['wavelength']
    test_wavelength = config['Optics'][test_laser_key]['wavelength']
    read_wavelength = config['Optics'][readout_laser_key]['wavelength']

    # Convert the 32 bit ints into 64 bit ints
    init_pulse_time = numpy.int64(init_pulse_time)
    test_pulse_time = numpy.int64(test_pulse_time)
    readout_laser_time = numpy.int64(readout_laser_time)
    readout_time = numpy.int64(readout_time)

    # intra_pulse_delay = config['CommonDurations']['cw_meas_buffer']
    intra_pulse_delay = config['CommonDurations']['scc_ion_readout_buffer']

    if init_laser_key == readout_laser_key and init_laser_key == readout_laser_key:
        total_delay = init_pulse_aom_delay_time
    elif init_laser_key == readout_laser_key and init_laser_key != test_laser_key:
        total_delay = init_pulse_aom_delay_time + test_pulse_aom_delay_time
    elif init_laser_key == test_laser_key and init_laser_key != readout_laser_key:
        total_delay = init_pulse_aom_delay_time + read_pulse_aom_delay_time
    else:
        total_delay = init_pulse_aom_delay_time + test_pulse_aom_delay_time + read_pulse_aom_delay_time

    period = total_delay + init_pulse_time + test_pulse_time + readout_time +\
                                        2*intra_pulse_delay + 300
                                        
    # calc the time that the apd will be reading before and after the laser turns on. 
    dead_time = int((readout_time - readout_laser_time)/2)

    #%% Define the sequence
    seq = Sequence()

    # Clock
    train = [(total_delay, LOW),
             (init_pulse_time, LOW),
             (intra_pulse_delay, LOW),
             (test_pulse_time, LOW),
             (intra_pulse_delay,LOW),
             (readout_time, LOW),
             (100, LOW), 
             (100, HIGH), 
             (100, LOW)]
    seq.setDigital(pulser_do_daq_clock, train)

    # APD gate
    train = [(total_delay, LOW),
             (init_pulse_time, LOW),
             (intra_pulse_delay, LOW),
             (test_pulse_time, LOW),
             (intra_pulse_delay,LOW),
             (readout_time, HIGH),
             (300, LOW)]
    seq.setDigital(pulser_do_daq_gate, train)


    green_laser_power_list = []
    yellow_laser_power_list = []
    red_laser_power_list = []
    
    train_green = [(total_delay, LOW)]
    train_yellow = [(total_delay, LOW)]        
    train_red = [(total_delay, LOW)]
    
    init_analog = False
    if init_laser_power != None:
        HIGH = 1#init_laser_power
        init_analog = True
    else:
        HIGH = 1
    if init_wavelength < 550:
        train_green.extend([(init_pulse_time, HIGH)])
        if init_analog:
            green_laser_power_list.append(init_laser_power)
        train_yellow.extend([(init_pulse_time, LOW)])
        train_red.extend([(init_pulse_time, LOW)])
    if init_wavelength < 600 and init_wavelength > 550:
        train_green.extend([(init_pulse_time, LOW)])
        train_yellow.extend([(init_pulse_time, HIGH)])
        if init_analog:
            yellow_laser_power_list.append(init_laser_power)
        train_red.extend([(init_pulse_time, LOW)])
    if init_wavelength > 600:
        train_green.extend([(init_pulse_time, LOW)])
        train_yellow.extend([(init_pulse_time, LOW)])
        train_red.extend([(init_pulse_time, HIGH)])
        if init_analog:
            red_laser_power_list.append(init_laser_power)
    
    train_green.extend([(intra_pulse_delay, LOW)])
    train_yellow.extend([(intra_pulse_delay, LOW)])
    train_red.extend([(intra_pulse_delay, LOW)])
    
    test_analog = False
    if test_laser_power != None:
        HIGH = 1#test_laser_power
        test_analog = True
    else:
        HIGH = 1
    if test_wavelength < 550:
        train_green.extend([(test_pulse_time, HIGH)])
        if test_analog:
            green_laser_power_list.append(test_laser_power)
        train_yellow.extend([(test_pulse_time, LOW)])
        train_red.extend([(test_pulse_time, LOW)])
    if test_wavelength < 600 and test_wavelength > 550:
        train_green.extend([(test_pulse_time, LOW)])
        train_yellow.extend([(test_pulse_time, HIGH)])
        if test_analog:
            yellow_laser_power_list.append(test_laser_power)
        train_red.extend([(test_pulse_time, LOW)])
    if test_wavelength > 600:
        train_green.extend([(test_pulse_time, LOW)])
        train_yellow.extend([(test_pulse_time, LOW)])
        train_red.extend([(test_pulse_time, HIGH)])
        if test_analog:
            red_laser_power_list.append(test_laser_power)
        
    train_green.extend([(intra_pulse_delay, LOW)])
    train_yellow.extend([(intra_pulse_delay, LOW)])
    train_red.extend([(intra_pulse_delay, LOW)])
    
    read_analog = False
    if read_laser_power != None:
        HIGH = 1#read_laser_power
        read_analog = True
    else:
        HIGH = 1
    if read_wavelength < 550:
        train_green.extend([(dead_time, LOW),
                            (readout_laser_time, HIGH),
                            (dead_time, LOW)])
        if read_analog:
            green_laser_power_list.append(read_laser_power)
        train_yellow.extend([(readout_time, LOW)])
        train_red.extend([(readout_time, LOW)])
    if read_wavelength < 600 and read_wavelength > 550:
        train_green.extend([(readout_time, LOW)])
        train_yellow.extend([(dead_time, LOW),
                            (readout_laser_time, HIGH),
                           (dead_time, LOW)])
        if read_analog:
            yellow_laser_power_list.append(read_laser_power)
        train_red.extend([(readout_time, LOW)])
    if read_wavelength > 600:
        train_green.extend([(readout_time, LOW)])
        train_yellow.extend([(readout_time, LOW)])
        train_red.extend([(dead_time, LOW),
                            (readout_laser_time, HIGH),
                            (dead_time, LOW)])
        if read_analog:
            red_laser_power_list.append(read_laser_power)
        
    train_green.extend([(100, LOW)])
    train_yellow.extend([(100, LOW)])
    train_red.extend([(100, LOW)])
    
    green_power = None
    yellow_power = None
    red_power = None
    
    if len(green_laser_power_list)!=0:
        green_power = green_laser_power_list
    if len(yellow_laser_power_list)!=0:
        yellow_power = yellow_laser_power_list
    if len(red_laser_power_list)!=0:
        red_power = red_laser_power_list
    
    tool_belt.process_laser_seq(pulse_streamer, seq, config,
                            'integrated_520', green_power, train_green)
    tool_belt.process_laser_seq(pulse_streamer, seq, config,
                            'laserglow_589', yellow_power, train_yellow)
    tool_belt.process_laser_seq(pulse_streamer, seq, config,
                            'cobolt_638', red_power, train_red)
    
    
    
    final_digital = []
    final = OutputState(final_digital, 0.0, 0.0)

    return seq, final, [period]


if __name__ == '__main__':
    config = tool_belt.get_config_dict()

    # args = [100000,100000, 200000, 100000,
    #         'integrated_520', 'cobolt_638', 'cobolt_638',
    #         None,0.8, 0.1,   0]
    args = [200000.0, 1000000.0, 75000.0, 'cobolt_638', 'integrated_520', 'cobolt_638', 0.69, None, 0.61, 1]
    seq = get_seq(None, config, args)[0]
    seq.plot()
