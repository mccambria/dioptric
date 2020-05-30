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

def get_seq(pulser_wiring, args):

    # Unpack the args
    readout_time, init_ion_time, reion_time, ion_time, pi_pulse, shelf_time,\
            wait_time, uwave_tau_max, laser_515_delay, aom_589_delay, laser_638_delay, rf_delay, \
            apd_indices, aom_ao_589_pwr, shelf_power, state_value = args

    readout_time = numpy.int64(readout_time)
    init_ion_time = numpy.int64(init_ion_time)
    reion_time = numpy.int64(reion_time)
    ion_time = numpy.int64(ion_time)
    pi_pulse = numpy.int64(pi_pulse)
    shelf_time = numpy.int64(shelf_time)
    uwave_tau_max = numpy.int64(uwave_tau_max)
    
    total_delay = laser_515_delay + aom_589_delay + laser_638_delay + rf_delay
    
    # For rabi experiment, we want to have sequence take same amount of time 
    # over each tau, so have some waittime after the readout to accoutn for this
    post_wait_time = uwave_tau_max - pi_pulse
    # Test period
    period =  total_delay + (init_ion_time + reion_time + ion_time + shelf_time + pi_pulse + \
                           readout_time + post_wait_time + 5 * wait_time)*2
    
    # Get what we need out of the wiring dictionary
    pulser_do_apd_gate = pulser_wiring['do_apd_{}_gate'.format(apd_indices)]
    pulser_do_clock = pulser_wiring['do_sample_clock']
    pulser_do_532_aom = pulser_wiring['do_532_aom']
    pulser_ao_589_aom = pulser_wiring['ao_589_aom']
    pulser_do_638_aom = pulser_wiring['do_638_laser']
    sig_gen_name = tool_belt.get_signal_generator_name(States(state_value))
    sig_gen_gate_chan_name = 'do_{}_gate'.format(sig_gen_name)
    pulser_do_sig_gen_gate = pulser_wiring[sig_gen_gate_chan_name]
    
    # Make sure the ao_aom voltage to the 589 aom is within 0 and 1 V
    tool_belt.aom_ao_589_pwr_err(aom_ao_589_pwr)
    tool_belt.aom_ao_589_pwr_err(shelf_power)
    
    seq = Sequence()

    #collect photons for certain timewindow tR in APD
    train = [(total_delay + init_ion_time + reion_time  + pi_pulse + shelf_time + ion_time + 4*wait_time, LOW), 
             (readout_time, HIGH), 
             (post_wait_time + init_ion_time + reion_time + pi_pulse + shelf_time + ion_time + 5*wait_time, LOW), 
             (readout_time, HIGH), (post_wait_time + wait_time, LOW)]
    seq.setDigital(pulser_do_apd_gate, train)
    
    # reionization pulse (green)
    delay = total_delay - laser_515_delay
    train = [ (delay + init_ion_time + wait_time, LOW), (reion_time, HIGH), 
             (5*wait_time + post_wait_time + pi_pulse + shelf_time + ion_time + readout_time + init_ion_time, LOW), 
             (reion_time, HIGH), 
             (4*wait_time + post_wait_time + pi_pulse + shelf_time + ion_time + readout_time + laser_515_delay, LOW)]  
    seq.setDigital(pulser_do_532_aom, train)
 
    # ionization pulse (red)
    delay = total_delay - laser_638_delay
    train = [(delay, LOW), (init_ion_time, HIGH), 
             (3*wait_time + reion_time + pi_pulse + shelf_time, LOW), 
             (ion_time, HIGH), 
             (2*wait_time + post_wait_time + readout_time, LOW), (init_ion_time, HIGH),
             (3*wait_time + reion_time + pi_pulse + shelf_time, LOW), 
             (ion_time, HIGH), 
             (2*wait_time + post_wait_time + readout_time + laser_638_delay, LOW)]
    seq.setDigital(pulser_do_638_aom, train)
    
    # uwave pulses
    delay = total_delay - rf_delay
    train = [(delay + init_ion_time + reion_time + 2*wait_time, LOW), (pi_pulse, HIGH), 
             (8*wait_time + 2*shelf_time + init_ion_time + pi_pulse + reion_time \
              + 2*post_wait_time + 2*readout_time + 2*ion_time + rf_delay, LOW)]
    seq.setDigital(pulser_do_sig_gen_gate, train)
    
    # readout with 589
    delay = total_delay - aom_589_delay
    train = [(delay + init_ion_time + reion_time + 3*wait_time + pi_pulse, LOW), 
             (shelf_time + ion_time, shelf_power), 
             (wait_time, LOW), 
             (readout_time, aom_ao_589_pwr),
             (post_wait_time + 4*wait_time + init_ion_time + reion_time + pi_pulse, LOW),
             (shelf_time + ion_time, shelf_power),
             (wait_time, LOW), 
             (readout_time, aom_ao_589_pwr), 
             (post_wait_time + wait_time + aom_589_delay, LOW)]
    seq.setAnalog(pulser_ao_589_aom, train) 
    

    
    final_digital = [pulser_do_clock]
    final = OutputState(final_digital, 0.0, 0.0)

    return seq, final, [period]


if __name__ == '__main__':
    wiring = {'do_apd_0_gate': 1,
              'do_532_aom': 2,
              'do_signal_generator_bnc835_gate': 3,
              'do_signal_generator_tsg4104a_gate': 4,
               'do_sample_clock':5,
               'do_638_laser': 6,
               'ao_589_aom': 0,
               'ao_638_laser': 1,

}

    args = [1000, 500, 100, 100, 100, 100, 100, 200, 0, 0, 0, 0, 0, 0.8, 0.8, 1]
#    args = [10000000, 25000, 100000, 500, 100.0, 200, 1000, 200, 140, 1080, 90, 40, 0, 0.25, 0.35, 1]
    seq, final, _ = get_seq(wiring, args)
    seq.plot()