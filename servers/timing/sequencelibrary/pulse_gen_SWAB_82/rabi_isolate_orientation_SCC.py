# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 17:39:27 2020

@author: agardill
"""

from pulsestreamer import Sequence
from pulsestreamer import OutputState
import numpy
import utils.tool_belt as tool_belt
from utils.tool_belt import States

LOW = 0
HIGH = 1


def get_seq(pulser_wiring, args):

    # %% Parse wiring and args

    # Unpack the args
    tau, readout_time, yellow_pol_time, shelf_time, init_ion_time, reion_time, ion_time, target_pi_pulse,\
            wait_time, num_ionizations, laser_515_delay, aom_589_delay, laser_638_delay, rf_delay, \
            apd_indices, aom_ao_589_pwr, yellow_pol_pwr, shelf_pwr, \
            shelf_state_value, test_state_value = args
            
    #convert the time values
    num_ionizations = int(num_ionizations)
    tau = numpy.int64(tau)
    shelf_time = numpy.int64(shelf_time)
    readout_time = numpy.int64(readout_time)
    init_ion_time = numpy.int64(init_ion_time)
    reion_time = numpy.int64(reion_time)
    ion_time = numpy.int64(ion_time)
    target_pi_pulse = numpy.int64(target_pi_pulse)
    yellow_pol_time = numpy.int64(yellow_pol_time)
    
    # Get what we need out of the wiring dictionary
    # Get what we need out of the wiring dictionary
    pulser_do_apd_gate = pulser_wiring['do_apd_{}_gate'.format(apd_indices)]
    pulser_do_clock = pulser_wiring['do_sample_clock']
    pulser_do_532_aom = pulser_wiring['do_532_aom']
    pulser_ao_589_aom = pulser_wiring['ao_589_aom']
    pulser_do_638_aom = pulser_wiring['do_638_laser']
    shelf_sig_gen_name = tool_belt.get_signal_generator_name(States(shelf_state_value))
    shelf_sig_gen_gate_name = 'do_{}_gate'.format(shelf_sig_gen_name)
    pulser_do_shlef_sig_gen_gate = pulser_wiring[shelf_sig_gen_gate_name]
    test_sig_gen_name = tool_belt.get_signal_generator_name(States(test_state_value))
    test_sig_gen_gate_name = 'do_{}_gate'.format(test_sig_gen_name)
    pulser_do_test_sig_gen_gate = pulser_wiring[test_sig_gen_gate_name]
#    
    # Make sure the ao_aom voltage to the 589 aom is within 0 and 1 V
    tool_belt.aom_ao_589_pwr_err(aom_ao_589_pwr)
    tool_belt.aom_ao_589_pwr_err(yellow_pol_pwr)
    tool_belt.aom_ao_589_pwr_err(shelf_pwr)

    # %% Couple calculated values
    total_delay = laser_515_delay + aom_589_delay + laser_638_delay + rf_delay

    # ionization repeated time
    total_ion_rep_time = num_ionizations * (yellow_pol_time + target_pi_pulse + shelf_time + ion_time + 3*wait_time)
    
    # The period is independent of the particular tau, but it must be long
    # enough to accomodate the longest tau
    period = total_delay + (init_ion_time + reion_time + total_ion_rep_time + \
                           yellow_pol_time + tau + shelf_time + ion_time + \
                           readout_time + wait_time*6)*2 

    # %% Define the sequence

    seq = Sequence()

    #collect photons for certain timewindow tR in APD
    train = [(total_delay + init_ion_time + reion_time + total_ion_rep_time + yellow_pol_time + tau + shelf_time + ion_time + 5*wait_time, LOW), 
             (readout_time, HIGH), 
             (6*wait_time + total_ion_rep_time +  init_ion_time + reion_time + yellow_pol_time + tau + shelf_time + ion_time, LOW),
             (readout_time, HIGH), (wait_time, LOW)]
    seq.setDigital(pulser_do_apd_gate, train)
    
    # reionization pulse (green)
    delay = total_delay - laser_515_delay
    train = [ (delay + init_ion_time + wait_time, LOW), (reion_time, HIGH), 
             (6*wait_time + total_ion_rep_time + yellow_pol_time + tau + shelf_time + ion_time + readout_time + init_ion_time, LOW),
             (reion_time, HIGH), 
             (5*wait_time + total_ion_rep_time + yellow_pol_time + tau + shelf_time + ion_time + readout_time + laser_515_delay, LOW)]  
    seq.setDigital(pulser_do_532_aom, train)
 
    # ionization pulse (red)
    delay = total_delay - laser_638_delay
    train = [(delay, LOW), (init_ion_time, HIGH), 
             (2*wait_time + reion_time, LOW)]
    for i in range(num_ionizations):
        train.extend([(yellow_pol_time + 2* wait_time + target_pi_pulse + shelf_time, LOW),
                           (ion_time, HIGH), (wait_time, LOW)])
    train.extend([(yellow_pol_time + tau + 2*wait_time + shelf_time, LOW), (ion_time, HIGH), 
                  (readout_time + 2*wait_time, LOW), (init_ion_time, HIGH), 
             (2*wait_time + reion_time, LOW)])
    for i in range(num_ionizations):
        train.extend([(yellow_pol_time + 2* wait_time + target_pi_pulse + shelf_time, LOW),
                           (ion_time, HIGH), (wait_time, LOW)])
    train.extend([(yellow_pol_time + tau + 2*wait_time + shelf_time, LOW), (ion_time, HIGH), 
                  (readout_time + 2*wait_time + laser_638_delay, LOW)])
    seq.setDigital(pulser_do_638_aom, train)
    
    # target uwave pulse (repeated)
    delay = total_delay - rf_delay
    train = [(delay + init_ion_time + reion_time + 2*wait_time, LOW)]
    for i in range(num_ionizations):
        train.extend([(yellow_pol_time + wait_time, LOW), (target_pi_pulse, HIGH),
                      (shelf_time + ion_time + 2*wait_time, LOW)]) 
    train.extend([(yellow_pol_time + tau + shelf_time + ion_time + readout_time + init_ion_time + reion_time + 6*wait_time, LOW)])
    for i in range(num_ionizations):
        train.extend([(yellow_pol_time + wait_time, LOW), (target_pi_pulse, HIGH),
                      (shelf_time + ion_time + 2*wait_time, LOW)]) 
    train.extend([(yellow_pol_time + tau + shelf_time + ion_time + readout_time + 4*wait_time + rf_delay, LOW)]) 
    seq.setDigital(pulser_do_shlef_sig_gen_gate, train)

    # test uwave pulse (toggled on/off after reps)
    delay = total_delay - rf_delay
    train = [(delay + init_ion_time + reion_time + 2*wait_time + total_ion_rep_time, LOW)]
    train.extend([(yellow_pol_time + wait_time, LOW), (tau, HIGH), 
                  (shelf_time + ion_time + readout_time + init_ion_time + reion_time + 5*wait_time, LOW)])
    train.extend([(total_ion_rep_time + tau + shelf_time + ion_time +  readout_time + 4*wait_time + rf_delay, LOW)]) 
    seq.setDigital(pulser_do_test_sig_gen_gate, train)
    
    # readout with 589
    delay = total_delay - aom_589_delay
    train = [(delay + init_ion_time + reion_time + 2*wait_time, LOW)]
    for i in range(num_ionizations):
        train.extend([(yellow_pol_time, yellow_pol_pwr), 
                      (2*wait_time + target_pi_pulse, LOW), (shelf_time + ion_time, shelf_pwr),
                      (wait_time, LOW)])     
    train.extend([(yellow_pol_time, yellow_pol_pwr), (tau + 2*wait_time, LOW), 
                  (shelf_time + ion_time, shelf_pwr), 
                  (wait_time, LOW), (readout_time, aom_ao_589_pwr), 
                  (3*wait_time + init_ion_time + reion_time, LOW)]) 
    for i in range(num_ionizations):
        train.extend([(yellow_pol_time, yellow_pol_pwr), 
                      (2*wait_time + target_pi_pulse, LOW), (shelf_time + ion_time, shelf_pwr),
                      (wait_time, LOW)])    
    train.extend([(yellow_pol_time, yellow_pol_pwr), (tau + 2*wait_time, LOW), 
                  (shelf_time + ion_time, shelf_pwr), 
                  (wait_time, LOW), (readout_time, aom_ao_589_pwr),
                  (aom_589_delay, LOW)])
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
               'ao_638_laser': 1,}
#tau, readout_time, yellow_pol_time, shelf_time, init_ion_time, reion_time, ion_time, target_pi_pulse,\
#            wait_time, num_ionizations, laser_515_delay, aom_589_delay, laser_638_delay, rf_delay, \
#            apd_indices, aom_ao_589_pwr, yellow_pol_pwr, shelf_pwr, target_state_value, \
#            test_state_value
    args = [100, 1000, 200, 100, 200, 200, 100, 100, 100, 2, 0, 0, 0, 0, 0, 1.0, 1.0, 1.0, 1, 3]
#    args = [125, 300, 2000, 50, 200000, 200000, 450, 94, 1000, 13, 0, 0, 0, 0, 0, 0.8, 0.3, 0.3, 1, 3]
    seq = get_seq(wiring, args)[0]
    seq.plot()
