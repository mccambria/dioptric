# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 21:24:36 2019

A similar sequence file to simple_readout_two_pulses, but this file can vary the 
readout duration of the apd independently from the readout laser duration.

Assumes that readout is on pulse two

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
    # assumes readout_time >= readout_laser_time
    init_pulse_time, readout_time, readout_laser_time, init_laser_key, readout_laser_key,\
      init_laser_power, read_laser_power, readout_on_pulse_ind, apd_index  = args

    # Get what we need out of the wiring dictionary
    pulser_wiring = config['Wiring']['PulseStreamer']
    pulser_do_daq_clock = pulser_wiring['do_sample_clock']
    pulser_do_daq_gate = pulser_wiring['do_apd_{}_gate'.format(apd_index)]

    init_pulse_aom_delay_time = config['Optics'][init_laser_key]['delay']
    read_pulse_aom_delay_time = config['Optics'][readout_laser_key]['delay']

    # Convert the 32 bit ints into 64 bit ints
    init_pulse_time = numpy.int64(init_pulse_time)
    readout_time = numpy.int64(readout_time)

    # intra_pulse_delay = config['CommonDurations']['cw_meas_buffer']
    intra_pulse_delay = config['CommonDurations']['scc_ion_readout_buffer']

    if init_laser_key == readout_laser_key:
        total_delay = init_pulse_aom_delay_time
    else:
        total_delay = init_pulse_aom_delay_time + read_pulse_aom_delay_time

    period = total_delay + init_pulse_time + readout_time +\
                                        intra_pulse_delay + 300
                                        
    # calc the time that the apd will be reading before and after the laser turns on. 
    dead_time = int((readout_time - readout_laser_time)/2)

    #%% Define the sequence
    seq = Sequence()

    # Clock
    train = [(total_delay + init_pulse_time + intra_pulse_delay + readout_time + 100, LOW), (100, HIGH), (100, LOW)]
    seq.setDigital(pulser_do_daq_clock, train)

    # APD gate
    train = [(total_delay + init_pulse_time + intra_pulse_delay, LOW), (readout_time, HIGH), (300, LOW)]
    seq.setDigital(pulser_do_daq_gate, train)

    if init_laser_key == readout_laser_key:
        laser_key = readout_laser_key
        laser_powers = [init_laser_power, read_laser_power]

        train = [(init_pulse_time, HIGH),
                 (intra_pulse_delay + dead_time, LOW),
                 (readout_laser_time, HIGH), (dead_time + 100 ,LOW )]
        tool_belt.process_laser_seq(pulse_streamer, seq, config,
                                laser_key, laser_powers, train)

    else:
        train_init_laser = [(read_pulse_aom_delay_time, LOW),
                            (init_pulse_time, HIGH),
                            (100  + intra_pulse_delay + dead_time + readout_laser_time + dead_time,LOW )]
        tool_belt.process_laser_seq(pulse_streamer, seq, config,
                                init_laser_key, init_laser_power, train_init_laser)


        train_read_laser = [(init_pulse_aom_delay_time + init_pulse_time + intra_pulse_delay + dead_time, LOW),
                            (readout_laser_time, HIGH), 
                            (dead_time + 100 ,LOW )]
        tool_belt.process_laser_seq(pulse_streamer, seq, config,
                                readout_laser_key, read_laser_power, train_read_laser)

    final_digital = []
    final = OutputState(final_digital, 0.0, 0.0)

    return seq, final, [period]


if __name__ == '__main__':
    config = tool_belt.get_config_dict()      
    args = [100000, 200000, 100000, 'cobolt_638', 'cobolt_638', 0.6, 0.2, 2, 0]
    # args = [1000.0, 100000000, 'cobolt_515', 'laserglow_589', None, 0.15, 0]
    seq = get_seq(None, config, args)[0]
    seq.plot()
