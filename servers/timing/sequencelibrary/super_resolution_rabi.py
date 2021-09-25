#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 13:40:44 2021

@author: agardill
"""

from pulsestreamer import Sequence
from pulsestreamer import OutputState
import utils.tool_belt as tool_belt
from utils.tool_belt import States
import numpy

LOW = 0
HIGH = 1

def get_seq(pulse_streamer, config, args):

    # Unpack the args
    readout_time, init_time, depletion_time, ion_time, pi_pulse, shelf_time,\
            uwave_tau_max, init_color, depletion_color, \
            green_laser_name, yellow_laser_name, red_laser_name, sig_gen_name, \
            apd_indices, readout_power, shelf_power = args

    # Convert all times to int64
    readout_time = numpy.int64(readout_time)
    init_time = numpy.int64(init_time)
    depletion_time = numpy.int64(depletion_time)
    ion_time = numpy.int64(ion_time)
    pi_pulse = numpy.int64(pi_pulse)
    shelf_time = numpy.int64(shelf_time)
    uwave_tau_max = numpy.int64(uwave_tau_max)
    
    # Get the wait time between pulses
    wait_time =config['CommonDurations']['uwave_buffer']
    galvo_move_time = config['Positioning']['xy_small_response_delay']
    galvo_move_time = numpy.int64(galvo_move_time)
    
        
    # delays
    green_delay_time = config['Optics'][green_laser_name]['delay']
    yellow_delay_time = config['Optics'][yellow_laser_name]['delay']
    red_delay_time =config['Optics'][red_laser_name]['delay']
    rf_delay_time = config['Microwaves'][sig_gen_name]['delay']
    
    # TESTING
    # wait_time =100
    # galvo_move_time = 500
    # galvo_move_time = numpy.int64(galvo_move_time)
    # green_delay_time = 0
    # yellow_delay_time = 0
    # red_delay_time =0
    # rf_delay_time = 0
    
    
    total_delay = green_delay_time + yellow_delay_time + red_delay_time + rf_delay_time
    
    # For rabi experiment, we want to have sequence take same amount of time 
    # over each tau, so have some waittime after the readout to accoutn for this
    post_wait_time = uwave_tau_max - pi_pulse
    # Test period
    period =  total_delay + (init_time + depletion_time + ion_time + shelf_time + pi_pulse + \
                           readout_time + post_wait_time + 3 * wait_time + 2*galvo_move_time)*2
    
    # Get what we need out of the wiring dictionary
    pulser_wiring = config['Wiring']['PulseStreamer']
    pulser_do_apd_gate = pulser_wiring['do_apd_{}_gate'.format(apd_indices)]
    pulser_do_clock = pulser_wiring['do_sample_clock']
    pulser_ao_589_aom = pulser_wiring['ao_{}_am'.format(yellow_laser_name)]
    sig_gen_gate_chan_name = 'do_{}_gate'.format(sig_gen_name)
    pulser_do_sig_gen_gate = pulser_wiring[sig_gen_gate_chan_name]
    
    # Make sure the ao_aom voltage to the 589 aom is within 0 and 1 V
    tool_belt.aom_ao_589_pwr_err(readout_power)
    tool_belt.aom_ao_589_pwr_err(shelf_power)
    
    seq = Sequence()

    #collect photons for certain timewindow tR in APD
    train = [(total_delay + init_time + galvo_move_time, LOW),
             (depletion_time, HIGH),
             (galvo_move_time + pi_pulse + wait_time + shelf_time + ion_time + wait_time, LOW), 
             (readout_time, HIGH),
             (post_wait_time + wait_time + init_time + galvo_move_time, LOW),
             (depletion_time, HIGH),
             (galvo_move_time + pi_pulse + wait_time + shelf_time + ion_time + wait_time, LOW),
             (readout_time, HIGH), 
             (post_wait_time + wait_time, LOW)]
    seq.setDigital(pulser_do_apd_gate, train)
    
    # clock 
    # I needed to add 100 ns between the redout and the clock pulse, otherwise 
    # the tagger misses some of the gate open/close clicks
    train = [(total_delay + init_time + 100, LOW),(100, HIGH),
             (galvo_move_time - 100 + depletion_time, LOW), (100, HIGH), 
             (galvo_move_time + pi_pulse - 100 + wait_time + shelf_time + ion_time + wait_time + readout_time, LOW), (100, HIGH),
             (post_wait_time + wait_time + init_time - 100, LOW),(100, HIGH),
             (galvo_move_time - 100 + depletion_time, LOW), (100, HIGH), 
             (galvo_move_time + pi_pulse - 100 + wait_time + shelf_time + ion_time + wait_time + readout_time, LOW), (100, HIGH), (100, LOW)
             ] 
    seq.setDigital(pulser_do_clock, train)
    
    
    # uwave pulses
    delay = total_delay - rf_delay_time
    train = [(delay  + init_time + depletion_time + galvo_move_time + galvo_move_time, LOW), (pi_pulse, HIGH), 
             (wait_time, LOW)]
    seq.setDigital(pulser_do_sig_gen_gate, train)
    
    # Green laser
    green_delay = total_delay - green_delay_time
    green_train = [ (green_delay, LOW)]
    
    # Red laser
    red_delay = total_delay - red_delay_time
    red_train = [(red_delay, LOW)]
    
    init_train_on = [(init_time, HIGH)]
    init_train_off = [(init_time, LOW)]
    if init_color < 589 :
        green_train.extend(init_train_on)
        red_train.extend(init_train_off)
    if init_color > 589:
        green_train.extend(init_train_off)
        red_train.extend(init_train_on)
    
    green_train.extend([(galvo_move_time, LOW)])
    red_train.extend([(galvo_move_time, LOW)])
    
    deplete_train_on = [(depletion_time, HIGH)]
    deplete_train_off = [(depletion_time, LOW)]
    if depletion_color < 589 :
        green_train.extend(deplete_train_on)
        red_train.extend(deplete_train_off)
    if depletion_color > 589:
        green_train.extend(deplete_train_off)
        red_train.extend(deplete_train_on)
        
    
    green_train.extend([(galvo_move_time + 3*wait_time + post_wait_time + \
                         pi_pulse + shelf_time + ion_time + readout_time, LOW)])
    red_train.extend([(galvo_move_time + pi_pulse + wait_time + shelf_time, LOW), 
             (ion_time, HIGH), 
             (2*wait_time + post_wait_time + readout_time, LOW)])
    
    
    if init_color < 589 :
        green_train.extend(init_train_on)
        red_train.extend(init_train_off)
    if init_color > 589:
        green_train.extend(init_train_off)
        red_train.extend(init_train_on)
    
    green_train.extend([(galvo_move_time, LOW)])
    red_train.extend([(galvo_move_time, LOW)])
    
    if depletion_color < 589 :
        green_train.extend(deplete_train_on)
        red_train.extend(deplete_train_off)
    if depletion_color > 589:
        green_train.extend(deplete_train_off)
        red_train.extend(deplete_train_on)
        
    green_train.extend([(galvo_move_time + 3*wait_time + post_wait_time + \
                         pi_pulse + shelf_time + ion_time + readout_time + \
                             green_delay_time, LOW)])
    red_train.extend([(galvo_move_time + pi_pulse + wait_time + shelf_time, LOW), 
             (ion_time, HIGH), 
             (2*wait_time + post_wait_time + readout_time + red_delay_time, LOW)])
    
    
    tool_belt.process_laser_seq(pulse_streamer, seq, config,
                            green_laser_name, None, green_train)
 
    tool_belt.process_laser_seq(pulse_streamer, seq, config,
                            red_laser_name, None, red_train)
    
    # Yellow laser
    delay = total_delay - yellow_delay_time
    train = [(delay + init_time + depletion_time + 2*galvo_move_time + pi_pulse + wait_time, LOW), 
             (shelf_time + ion_time, shelf_power), 
             (wait_time, LOW), 
             (readout_time, readout_power),
             (post_wait_time + wait_time + init_time + depletion_time + 2*galvo_move_time + pi_pulse + wait_time, LOW),
             (shelf_time + ion_time, shelf_power),
             (wait_time, LOW), 
             (readout_time, readout_power), 
             (post_wait_time + wait_time + yellow_delay_time, LOW)]
    seq.setAnalog(pulser_ao_589_aom, train) 
    

#    train = [(period + 100, LOW), (100, HIGH), (100, LOW)]
    
    final_digital = []
    final = OutputState(final_digital, 0.0, 0.0)

    return seq, final, [period]


if __name__ == '__main__':
    config = tool_belt.get_config_dict()
    
            
    args = [1000,  100, 400, 500, 100, 100, 200, 515, 638, 'cobolt_515','laserglow_589', 'cobolt_638', 'signal_generator_bnc835',
            0, 0.8, 0.8]
    # args = [500000.0, 100000.0, 1500.0, 84, 200, 84, 'cobolt_515', 'laserglow_589', 'cobolt_638', 'signal_generator_bnc835', 0, 0.2, 0.6]
    seq = get_seq(None, config, args)[0]
    seq.plot()