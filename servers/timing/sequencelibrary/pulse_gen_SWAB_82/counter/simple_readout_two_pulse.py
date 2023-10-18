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
    init_pulse_time, readout_time, init_laser_key, readout_laser_key,\
      init_laser_power, read_laser_power, readout_on_pulse_ind  = args

    # Get what we need out of the wiring dictionary
    pulser_wiring = config['Wiring']['PulseGen']
    pulser_do_daq_clock = pulser_wiring['do_sample_clock']
    pulser_do_daq_gate = pulser_wiring['do_apd_gate']

    positioning = config['Positioning']
    if 'xy_small_response_delay' in positioning:
        galvo_move_time = positioning['xy_small_response_delay']
    else:
        galvo_move_time = positioning['xy_delay']
    init_pulse_aom_delay_time = config['Optics'][init_laser_key]['delay']
    read_pulse_aom_delay_time = config['Optics'][readout_laser_key]['delay']
    # init_pulse_aom_delay_time = 0
    # read_pulse_aom_delay_time = 0
    # galvo_move_time = 0

    # Convert the 32 bit ints into 64 bit ints
    init_pulse_time = numpy.int64(init_pulse_time)
    readout_time = numpy.int64(readout_time)

    # intra_pulse_delay = config['CommonDurations']['cw_meas_buffer']
    intra_pulse_delay = config['CommonDurations']['scc_ion_readout_buffer']

    if init_laser_key == readout_laser_key:
        total_delay = init_pulse_aom_delay_time
    else:
        total_delay = init_pulse_aom_delay_time + read_pulse_aom_delay_time

    period = galvo_move_time + total_delay + init_pulse_time + readout_time +\
                                        intra_pulse_delay + 300

    #%% Define the sequence
    seq = Sequence()

    # Clock
    train = [(galvo_move_time + total_delay + init_pulse_time + intra_pulse_delay + readout_time + 100, LOW), (100, HIGH), (100, LOW)]
    seq.setDigital(pulser_do_daq_clock, train)

    # APD gate
    if readout_on_pulse_ind == 1: # readout on first pulse
        train = [(galvo_move_time + total_delay, LOW), (init_pulse_time, HIGH),
                 (intra_pulse_delay + readout_time + 300, LOW)]
    elif readout_on_pulse_ind == 2: # readout on second pulse
        train = [(galvo_move_time + total_delay + init_pulse_time + intra_pulse_delay, LOW), (readout_time, HIGH), (300, LOW)]
    seq.setDigital(pulser_do_daq_gate, train)

    if init_laser_key == readout_laser_key:
        laser_key = readout_laser_key
        laser_power = read_laser_power

        train = [(galvo_move_time, LOW), (init_pulse_time, HIGH),
                 (intra_pulse_delay, LOW),
                 (readout_time, HIGH), (100 ,LOW )]
        tool_belt.process_laser_seq(pulse_streamer, seq, config,
                                laser_key, laser_power, train)

    else:
        train_init_laser = [(galvo_move_time + read_pulse_aom_delay_time, LOW),
                            (init_pulse_time, HIGH),
                            (100  + intra_pulse_delay + readout_time,LOW )]
        tool_belt.process_laser_seq(pulse_streamer, seq, config,
                                init_laser_key, [init_laser_power], train_init_laser)


        train_read_laser = [(galvo_move_time + init_pulse_aom_delay_time + init_pulse_time + intra_pulse_delay, LOW),
                            (readout_time, HIGH), 
                            (100 ,LOW )]
        tool_belt.process_laser_seq(pulse_streamer, seq, config,
                                readout_laser_key, read_laser_power, train_read_laser)
        print(train_read_laser)
    final_digital = []
    final = OutputState(final_digital, 0.0, 0.0)

    return seq, final, [period]


if __name__ == '__main__':
    config = tool_belt.get_config_dict()
    tool_belt.set_delays_to_zero(config)
    # args = [1000000.0, 50000000, 'cobolt_638', 'laserglow_589', None, 0.2, 2, 1]
    args = [1000, 50000, "cobolt_638", "laser_LGLO_589", None, None, 2]
    seq = get_seq(None, config, args)[0]
    seq.plot()
