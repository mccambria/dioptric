#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 20:40:44 2020

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
    readout_time, init_ion_time, reion_time, ion_time, shelf_time, wait_time,\
            pi_pulse_low, pi_pulse_high, tau_shrt, tau_long, \
            laser_515_delay, aom_589_delay, laser_638_delay, rf_delay, \
            apd_indices, \
            init_state_value, read_state_value,\
            aom_ao_589_pwr, shelf_power = args

    # Convert times to int64 type
    readout_time = numpy.int64(readout_time)
    init_ion_time = numpy.int64(init_ion_time)
    reion_time = numpy.int64(reion_time)
    ion_time = numpy.int64(ion_time)
    shelf_time = numpy.int64(shelf_time)
    wait_time = numpy.int64(wait_time)
    pi_pulse_low = numpy.int64(pi_pulse_low)
    pi_pulse_high = numpy.int64(pi_pulse_high)
    tau_shrt = numpy.int64(tau_shrt)
    tau_long = numpy.int64(tau_long)

    # Conditionals
    # Default the pulses to 0
    init_pi_low = 0
    init_pi_high = 0
    read_pi_low = 0
    read_pi_high = 0

    if init_state_value == States.LOW.value:
        init_pi_low = pi_pulse_low
    elif init_state_value == States.HIGH.value:
        init_pi_high = pi_pulse_high

    if read_state_value == States.LOW.value:
        read_pi_low = pi_pulse_low
    elif read_state_value == States.HIGH.value:
        read_pi_high = pi_pulse_high
        
    # Define total delay time    
    total_delay = laser_515_delay + aom_589_delay + laser_638_delay + rf_delay

    # Test period
    period =  total_delay + (init_ion_time + reion_time + ion_time + \
                     shelf_time + readout_time + 5 * wait_time + \
                     init_pi_low + init_pi_high + read_pi_low + read_pi_high)*4\
                     + (tau_shrt + tau_long)
    
    # Define the wirings of the channels
    pulser_do_apd_gate = pulser_wiring['do_apd_{}_gate'.format(apd_indices)]
    pulser_do_clock = pulser_wiring['do_sample_clock']
    pulser_do_532_aom = pulser_wiring['do_532_aom']
    pulser_ao_589_aom = pulser_wiring['ao_589_aom']
    pulser_do_638_aom = pulser_wiring['do_638_laser']
    low_sig_gen_name = tool_belt.get_signal_generator_name(States.LOW)
    low_sig_gen_gate_chan_name = 'do_{}_gate'.format(low_sig_gen_name)
    pulser_do_sig_gen_low_gate = pulser_wiring[low_sig_gen_gate_chan_name]
    high_sig_gen_name = tool_belt.get_signal_generator_name(States.HIGH)
    high_sig_gen_gate_chan_name = 'do_{}_gate'.format(high_sig_gen_name)
    pulser_do_sig_gen_high_gate = pulser_wiring[high_sig_gen_gate_chan_name]
    
    # Make sure the ao_aom voltage to the 589 aom is within 0 and 1 V
    tool_belt.aom_ao_589_pwr_err(aom_ao_589_pwr)
    tool_belt.aom_ao_589_pwr_err(shelf_power)
 
    # Define some uwave experiment shortcuts
    base_uwave_experiment_dur = init_pi_high + init_pi_low + \
                    read_pi_high + read_pi_low
    uwave_experiment_shrt = base_uwave_experiment_dur + tau_shrt
    uwave_experiment_long = base_uwave_experiment_dur + tau_long
    
    # %% 
    seq = Sequence()

    #collect photons for certain timewindow tR in APD
    pre_duration = init_ion_time + reion_time  + shelf_time + ion_time + 4*wait_time
    train = [(total_delay + pre_duration + uwave_experiment_shrt, LOW), 
             (readout_time, HIGH), 
             (wait_time + pre_duration , LOW),
             (readout_time, HIGH), 
             (wait_time + pre_duration + uwave_experiment_long, LOW),
             (readout_time, HIGH), 
             (wait_time + pre_duration , LOW),
             (readout_time, HIGH), 
             (wait_time, LOW)]
    seq.setDigital(pulser_do_apd_gate, train)
    
    # reionization pulse (green)
    delay = total_delay - laser_515_delay
    pre_duration = init_ion_time + wait_time
    mid_exp_duration = 4*wait_time + shelf_time + ion_time + readout_time
    train = [(delay + pre_duration, LOW), 
             (reion_time, HIGH), 
             (mid_exp_duration + uwave_experiment_shrt + pre_duration, LOW), 
             (reion_time, HIGH), 
             (mid_exp_duration + pre_duration, LOW),
             (reion_time, HIGH), 
             (mid_exp_duration + uwave_experiment_long + pre_duration, LOW), 
             (reion_time, HIGH), 
             (mid_exp_duration + laser_515_delay, LOW)]  
    seq.setDigital(pulser_do_532_aom, train)
 
    # ionization pulse (red)
    delay = total_delay - laser_638_delay
    mid_exp_duration = 3*wait_time + reion_time + shelf_time
    scc_readout = 2*wait_time + readout_time
    train = [(delay, LOW), 
             (init_ion_time, HIGH), 
             (mid_exp_duration + uwave_experiment_shrt, LOW), 
             (ion_time, HIGH), 
             (scc_readout, LOW), 
             (init_ion_time, HIGH),
             (mid_exp_duration, LOW), 
             (ion_time, HIGH), 
             (scc_readout, LOW),
             (init_ion_time, HIGH), 
             (mid_exp_duration + uwave_experiment_long, LOW), 
             (ion_time, HIGH), 
             (scc_readout, LOW), 
             (init_ion_time, HIGH),
             (mid_exp_duration, LOW), 
             (ion_time, HIGH), 
             (scc_readout + laser_638_delay, LOW)]
    seq.setDigital(pulser_do_638_aom, train)
    
    # uwave pulses
    delay = total_delay - rf_delay
    initialization = init_ion_time + reion_time + 2*wait_time
    scc_readout = shelf_time + ion_time + 3*wait_time + readout_time
    pre_duration = initialization
    mid_duration = 2*(scc_readout + initialization)
    post_duration = 2* scc_readout + initialization
    
    train = [(delay + pre_duration, LOW)]
    train.extend([(init_pi_high, HIGH), (tau_shrt + init_pi_low, LOW), (read_pi_high, HIGH)])
    train.extend([(read_pi_low + mid_duration, LOW)])
    train.extend([(init_pi_high, HIGH), (tau_long + init_pi_low, LOW), (read_pi_high, HIGH)])
    train.extend([(read_pi_low + post_duration + rf_delay, LOW)])
    seq.setDigital(pulser_do_sig_gen_high_gate, train)

    train = [(delay + pre_duration, LOW)]
    train.extend([(init_pi_low, HIGH), (tau_shrt + init_pi_high, LOW), (read_pi_low, HIGH)])
    train.extend([(read_pi_high + mid_duration, LOW)])
    train.extend([(init_pi_low, HIGH), (tau_long + init_pi_high, LOW), (read_pi_low, HIGH)])
    train.extend([(read_pi_high + post_duration + rf_delay, LOW)])
    seq.setDigital(pulser_do_sig_gen_low_gate, train)
    
    # readout with 589
    delay = total_delay - aom_589_delay
    pre_duration = init_ion_time + reion_time + 3*wait_time
    train = [(delay + pre_duration + uwave_experiment_shrt, LOW), 
             (shelf_time + ion_time, shelf_power), 
             (wait_time, LOW), 
             (readout_time, aom_ao_589_pwr),
             (wait_time + pre_duration, LOW),
             (shelf_time + ion_time, shelf_power),
             (wait_time, LOW), 
             (readout_time, aom_ao_589_pwr), 
             (wait_time + pre_duration + uwave_experiment_long, LOW), 
             (shelf_time + ion_time, shelf_power), 
             (wait_time, LOW), 
             (readout_time, aom_ao_589_pwr),
             (wait_time + pre_duration, LOW),
             (shelf_time + ion_time, shelf_power),
             (wait_time, LOW), 
             (readout_time, aom_ao_589_pwr), 
             (wait_time + aom_589_delay, LOW)]
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

            
    args = [1000, 500, 100, 100, 100, 100, 
            100, 100, 0, 100,
            0, 0, 0, 0, 
            0, 
            3, 3, 
            0.8, 0.8]
    seq, final, _ = get_seq(wiring, args)
    seq.plot()