#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 20:40:44 2020

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
    readout_time, reion_time, ion_time, pi_pulse, shelf_time,\
            uwave_tau_max, green_laser_name, yellow_laser_name, red_laser_name, sig_gen_name, \
            apd_indices, readout_power, shelf_power = args

    # Convert all times to int64
    readout_time = numpy.int64(readout_time)
    reion_time = numpy.int64(reion_time)
    ion_time = numpy.int64(ion_time)
    pi_pulse = numpy.int64(pi_pulse)
    shelf_time = numpy.int64(shelf_time)
    uwave_tau_max = numpy.int64(uwave_tau_max)
    
    # Get the wait time between pulses
    wait_time = config['CommonDurations']['uwave_buffer']
        
    # delays
    green_delay_time = config['Optics'][green_laser_name]['delay']
    yellow_delay_time = config['Optics'][yellow_laser_name]['delay']
    red_delay_time = config['Optics'][red_laser_name]['delay']
    rf_delay_time = config['Microwaves'][sig_gen_name]['delay']
    
    
    total_delay = green_delay_time + yellow_delay_time + red_delay_time + rf_delay_time
    
    # For rabi experiment, we want to have sequence take same amount of time 
    # over each tau, so have some waittime after the readout to accoutn for this
    # +++ Artifact from rabi experiments, in determine SCC durations, this is 0
    post_wait_time = uwave_tau_max - pi_pulse
    # Test period
    period =  total_delay + (reion_time + ion_time + shelf_time + pi_pulse + \
                           readout_time + post_wait_time + 4 * wait_time)*2
    
    # Get what we need out of the wiring dictionary
    pulser_wiring = config['Wiring']['PulseStreamer']
    pulser_do_apd_gate = pulser_wiring['do_apd_{}_gate'.format(apd_indices)]
    pulser_do_clock = pulser_wiring['do_sample_clock']
    analog_key = 'ao_{}_am'.format(yellow_laser_name)
    digital_key = 'do_{}_dm'.format(yellow_laser_name)
    analog_yellow = (analog_key in pulser_wiring)
    if analog_yellow:
        pulser_ao_589_aom = pulser_wiring[analog_key]
    else:
        pulser_do_589_dm = pulser_wiring[digital_key]
    sig_gen_gate_chan_name = 'do_{}_gate'.format(sig_gen_name)
    pulser_do_sig_gen_gate = pulser_wiring[sig_gen_gate_chan_name]
    
    # Make sure the ao_aom voltage to the 589 aom is within 0 and 1 V
    if readout_power is not None:
        tool_belt.aom_ao_589_pwr_err(readout_power)
    if shelf_power is not None:
        tool_belt.aom_ao_589_pwr_err(shelf_power)
    
    seq = Sequence()

    #collect photons for certain timewindow tR in APD
    train = [(total_delay + reion_time  + pi_pulse + shelf_time + ion_time + 3*wait_time, LOW), 
             (readout_time, HIGH), 
             (post_wait_time  + reion_time + pi_pulse + shelf_time + ion_time + 4*wait_time, LOW), 
             (readout_time, HIGH), (post_wait_time + wait_time, LOW)]
    seq.setDigital(pulser_do_apd_gate, train)
    
    # reionization pulse (green)
    delay = total_delay - green_delay_time
    train = [ (delay, LOW), (reion_time, HIGH), 
             (4*wait_time + post_wait_time + pi_pulse + shelf_time + ion_time + readout_time, LOW), 
             (reion_time, HIGH), 
             (4*wait_time + post_wait_time + pi_pulse + shelf_time + ion_time + readout_time + green_delay_time, LOW)]  
    # seq.setDigital(pulser_do_532_aom, train)
    tool_belt.process_laser_seq(pulse_streamer, seq, config,
                            green_laser_name, None, train)
 
    # ionization pulse (red)
    delay = total_delay - red_delay_time
    train = [(delay + 2*wait_time + reion_time + pi_pulse + shelf_time, LOW), 
             (ion_time, HIGH), 
             (4*wait_time + post_wait_time + readout_time + reion_time + pi_pulse + shelf_time, LOW), 
             (ion_time, HIGH), 
             (2*wait_time + post_wait_time + readout_time + red_delay_time, LOW)]
    # seq.setDigital(pulser_do_638_aom, train)
    tool_belt.process_laser_seq(pulse_streamer, seq, config,
                            red_laser_name, None, train)
    
    # uwave pulses
    delay = total_delay - rf_delay_time
    train = [(delay  + reion_time + wait_time, LOW), (pi_pulse, HIGH), 
             (7*wait_time + 2*shelf_time + pi_pulse + reion_time \
              + 2*post_wait_time + 2*readout_time + 2*ion_time + rf_delay_time, LOW)]
    seq.setDigital(pulser_do_sig_gen_gate, train)
    
    # readout with 589
    # Dummy values for digital modulation
    if not analog_yellow is None:
        shelf_power = HIGH 
        readout_power = HIGH
    delay = total_delay - yellow_delay_time
    train = [(delay + reion_time + 2*wait_time + pi_pulse, LOW), 
             (shelf_time + ion_time,shelf_power), 
             (wait_time, LOW), 
             (readout_time, readout_power),
             (post_wait_time + 3*wait_time + reion_time + pi_pulse, LOW),
             (shelf_time + ion_time, shelf_power),
             (wait_time, LOW), 
             (readout_time, readout_power), 
             (post_wait_time + wait_time + yellow_delay_time, LOW)]
    if analog_yellow:
        seq.setAnalog(pulser_ao_589_aom, train) 
    else:
        seq.setDigital(pulser_do_589_dm, train) 
    

    
    final_digital = [pulser_do_clock]
    final = OutputState(final_digital, 0.0, 0.0)

    return seq, final, [period]


if __name__ == '__main__':
    config = tool_belt.get_config_dict()
    
            
    # args = [1000,  100, 500, 100, 100, 200, 'cobolt_515','laserglow_589', 'cobolt_638', 'signal_generator_bnc835',
    #         0, 0.8, 0.8]
    args = [500000.0, 100000.0, 1500.0, 84, 200, 84, 'cobolt_515', 'laserglow_589', 'cobolt_638', 'signal_generator_bnc835', 0, 0.2, 0.6]
    seq = get_seq(None, config, args)[0]
    seq.plot()