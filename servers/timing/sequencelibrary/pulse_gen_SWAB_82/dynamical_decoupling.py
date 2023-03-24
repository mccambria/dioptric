# -*- coding: utf-8 -*-
"""
Created on Thur Aug 4, 2022

This is a sequence for dynamical decoupling sequence for routines that have
the following family of microwave sequences:

pi/2 - (T - pi - T)*N - pi/2

For example, CPMG pulses use this with N>1 repetitions. And XY4 and XY8 use this
with N = mod 4 (or 8).
    
The phase of the microwave pulses are controlled in the actual routine, so this can be used
for CPMG, XY4, or XY8.

Note that the variable pi_pulse_reps is the number of pi_pulses to perform. 
For example, if we want to perform XY4-2, we would want the number of pi pulses
to be 4 * 2. There are 4 pusles in the XY4 pulse sequence, 
and we want to repeate those 2 times.

10/26/2022 IK added pi pulse to the readout pulse, so that normalized population starts at 1

@author: Aedan
"""

from pulsestreamer import Sequence
from pulsestreamer import OutputState
import numpy
import utils.tool_belt as tool_belt
from utils.tool_belt import States

LOW = 0
HIGH = 1


def get_seq(pulse_streamer, config, args):

    # %% Parse wiring and args

    # The first 11 args are ns durations and we need them as int64s
    durations = []
    for ind in range(7):
        durations.append(numpy.int64(args[ind]))

    # Unpack the durations
    tau_shrt, polarization_time, gate_time, pi_pulse, pi_on_2_pulse, tau_long, dd_wait_time = durations
    
    

    # Get the APD indices
    pi_pulse_reps, state, laser_name, laser_power = args[7:12]
    state = States(state)
        
    # time of illumination during which signal readout occurs
    signal_time = polarization_time
    # time of illumination during which reference readout occurs
    reference_time = polarization_time
    
    pre_uwave_exp_wait_time = config['CommonDurations']['uwave_buffer']
    post_uwave_exp_wait_time = config['CommonDurations']['uwave_buffer']
    
    # time between signal and reference without illumination
    sig_to_ref_wait_time_base = pre_uwave_exp_wait_time + post_uwave_exp_wait_time
    sig_to_ref_wait_time_shrt = sig_to_ref_wait_time_base 
    sig_to_ref_wait_time_long = sig_to_ref_wait_time_base 
    
    laser_delay_time = config['Optics'][laser_name]['delay']
    sig_gen_name = config['Servers']['sig_gen_{}'.format(state.name)]
    rf_delay_time = config['Microwaves'][sig_gen_name]['delay']
    iq_delay_time = config['Microwaves']['iq_delay']
    
    back_buffer = 200
    delay_buffer = max(laser_delay_time,rf_delay_time, iq_delay_time, 100)
    iq_trigger_time = numpy.int64(min(pi_on_2_pulse, 10))
    final_pi_pulse_wait = dd_wait_time
    
    # we half the tau to put an IQ pulse in between. To create shorthand,
    # we define them here. But we need to make sure that the two halves still
    # add up to tau
    half_tau_shrt_st =int(tau_shrt/2)
    half_tau_shrt_en = int(tau_shrt -  half_tau_shrt_st)
    half_tau_long_st =int(tau_long/2)
    half_tau_long_en = int(tau_long -  half_tau_long_st)
    
    pulser_wiring = config['Wiring']['PulseGen']
    pulser_do_apd_gate = pulser_wiring['do_apd_gate']
    sig_gen_gate_chan_name = 'do_{}_gate'.format(sig_gen_name)
    pulser_do_sig_gen_gate = pulser_wiring[sig_gen_gate_chan_name]
    pulser_do_arb_wave_trigger = pulser_wiring['do_arb_wave_trigger']

    # %% Write the microwave sequence to be used. 
    # Also add up the total time of the uwave experiment and use for other channels

    uwave_experiment_train_shrt = [(pi_on_2_pulse, HIGH)]
    if pi_pulse_reps == 0:
        rep_train = [(tau_shrt, LOW), (tau_shrt, LOW)]
    else:  
        rep_train = [(tau_shrt, LOW), (pi_pulse, HIGH), (tau_shrt, LOW)]*pi_pulse_reps
    uwave_experiment_train_shrt.extend(rep_train)
    uwave_experiment_train_shrt.extend([(pi_on_2_pulse, HIGH)])
    uwave_experiment_train_shrt.extend([(final_pi_pulse_wait, LOW)]) # adding a wait between pi/2 and pi
    uwave_experiment_train_shrt.extend([(pi_pulse, HIGH)]) # adding a pi pulse to readout
    
    uwave_experiment_dur_shrt = 0
    for el in uwave_experiment_train_shrt:
        uwave_experiment_dur_shrt += el[0]
    
    uwave_experiment_train_long = [(pi_on_2_pulse, HIGH)]
    if pi_pulse_reps == 0:
        rep_train = [(tau_long, LOW), (tau_long, LOW)]
    else:  
        rep_train = [(tau_long, LOW), (pi_pulse, HIGH), (tau_long, LOW)]*pi_pulse_reps
    uwave_experiment_train_long.extend(rep_train)
    uwave_experiment_train_long.extend([(pi_on_2_pulse, HIGH)])
    uwave_experiment_train_long.extend([(final_pi_pulse_wait, LOW)]) # adding a wait between pi/2 and pi
    uwave_experiment_train_long.extend([(pi_pulse, HIGH)]) # adding a pi pulse to readout
        
    uwave_experiment_dur_long = 0
    for el in uwave_experiment_train_long:
        uwave_experiment_dur_long += el[0]

    # %% Write the IQ trigger pulse sequence
    # The first IQ pulse will occur right after optical pulse polarization
    # follwoing IQ pulses will occur half tau before the pi pulse
    uwave_iq_train_shrt = [(iq_trigger_time, HIGH), 
                           (pre_uwave_exp_wait_time + pi_on_2_pulse-iq_trigger_time, LOW),
                           (half_tau_shrt_st, LOW)]
    if pi_pulse_reps == 0:
        uwave_iq_train_shrt.extend([(iq_trigger_time, HIGH), 
                                    (half_tau_shrt_en - iq_trigger_time + half_tau_shrt_st, LOW),
                                    # (half_tau_shrt_en - iq_trigger_time + pi_on_2_pulse, LOW)])
                                    (half_tau_shrt_en + pi_on_2_pulse + final_pi_pulse_wait + pi_pulse, LOW)])
    else:    
        rep_train = [(iq_trigger_time, HIGH),
                 (half_tau_shrt_en - iq_trigger_time + pi_pulse + tau_shrt + half_tau_shrt_st, LOW)]*(pi_pulse_reps-1)
        uwave_iq_train_shrt.extend(rep_train)
        uwave_iq_train_shrt.extend([(iq_trigger_time, HIGH), 
                                    (half_tau_shrt_en - iq_trigger_time + pi_pulse + half_tau_shrt_st, LOW),
                                    (iq_trigger_time, HIGH), 
                                    # (half_tau_shrt_en - iq_trigger_time + pi_on_2_pulse, LOW)])
                                    (half_tau_shrt_en - iq_trigger_time + pi_on_2_pulse + final_pi_pulse_wait + pi_pulse, LOW)])
    
    
    uwave_iq_train_long = [(iq_trigger_time, HIGH), 
                           (pre_uwave_exp_wait_time + pi_on_2_pulse-iq_trigger_time, LOW),
                           (half_tau_long_st, LOW)]
    if pi_pulse_reps == 0:
        uwave_iq_train_long.extend([(iq_trigger_time, HIGH), 
                                    (half_tau_long_en - iq_trigger_time + half_tau_long_st, LOW),
                                    # (half_tau_shrt_en - iq_trigger_time + pi_on_2_pulse, LOW)
                                    (half_tau_long_en + pi_on_2_pulse + final_pi_pulse_wait + pi_pulse, LOW)])
    else:
        rep_train = [(iq_trigger_time, HIGH),
                 (half_tau_long_en - iq_trigger_time + pi_pulse + tau_long + half_tau_long_st, LOW)]*(pi_pulse_reps-1)
        uwave_iq_train_long.extend(rep_train)
        uwave_iq_train_long.extend([(iq_trigger_time, HIGH), 
                                    (half_tau_long_en - iq_trigger_time + pi_pulse + half_tau_long_st, LOW),
                                    (iq_trigger_time, HIGH), 
                                    # (half_tau_long_en - iq_trigger_time + pi_on_2_pulse, LOW)])
                                    (half_tau_long_en - iq_trigger_time + pi_on_2_pulse + final_pi_pulse_wait + pi_pulse, LOW)])
    

    # %% Define the sequence

    seq = Sequence()

    # APD gating
    train = [(delay_buffer, LOW),
             (polarization_time,LOW),
             (pre_uwave_exp_wait_time, LOW),
             (uwave_experiment_dur_shrt, LOW),
             (post_uwave_exp_wait_time, LOW),
             (gate_time, HIGH),
             (signal_time - gate_time, LOW),
             (sig_to_ref_wait_time_shrt, LOW),
             (gate_time, HIGH),
             (reference_time - gate_time, LOW),
             (pre_uwave_exp_wait_time, LOW),
             (uwave_experiment_dur_long, LOW),
             (post_uwave_exp_wait_time, LOW),
             (gate_time, HIGH),
             (signal_time - gate_time, LOW),
             (sig_to_ref_wait_time_long, LOW),
             (gate_time, HIGH),
             (reference_time - gate_time, LOW),
             (back_buffer, LOW)]             
    seq.setDigital(pulser_do_apd_gate, train)
    period = 0
    for el in train:
        period += el[0]
    print(period)

    # Laser
    train = [(delay_buffer - laser_delay_time, HIGH),
             (polarization_time, HIGH),
             (pre_uwave_exp_wait_time, LOW),
             (uwave_experiment_dur_shrt, LOW),
             (post_uwave_exp_wait_time, LOW),
             (signal_time, HIGH),
             (sig_to_ref_wait_time_shrt, LOW),
             (reference_time, HIGH),
             (pre_uwave_exp_wait_time, LOW),
             (uwave_experiment_dur_long, LOW),
             (post_uwave_exp_wait_time, LOW),
             (signal_time, HIGH),
             (sig_to_ref_wait_time_long, LOW),
             (reference_time, HIGH),
             (back_buffer + laser_delay_time, LOW)]   
    tool_belt.process_laser_seq(pulse_streamer, seq, config,
                                laser_name, laser_power, train)
    period = 0
    for el in train:
        period += el[0]
    print(period)

    # Microwaves
    train = [(delay_buffer - rf_delay_time, LOW),
             (polarization_time,LOW),
             (pre_uwave_exp_wait_time, LOW)]
    train.extend(uwave_experiment_train_shrt)
    train.extend([
             (post_uwave_exp_wait_time, LOW),
             (signal_time, LOW),
             (sig_to_ref_wait_time_shrt, LOW),
             (reference_time, LOW),
             (pre_uwave_exp_wait_time, LOW)])
    train.extend(uwave_experiment_train_long)
    train.extend([
             (post_uwave_exp_wait_time, LOW),
             (signal_time, LOW),
             (sig_to_ref_wait_time_long, LOW),
             (reference_time, LOW),
             (back_buffer + rf_delay_time, LOW)])
    seq.setDigital(pulser_do_sig_gen_gate, train)
    
    period = 0
    for el in train:
        period += el[0]
    print(period)

    # IQ modulation triggers
    train = [(delay_buffer - iq_delay_time, LOW),
             (iq_trigger_time, HIGH),
             (polarization_time-iq_trigger_time, LOW)]
              # (pre_uwave_exp_wait_time, LOW)]
    train.extend(uwave_iq_train_shrt)
    train.extend([
              (post_uwave_exp_wait_time, LOW),
              (signal_time, LOW),
              (sig_to_ref_wait_time_shrt, LOW),
              (reference_time, LOW),])
              # (pre_uwave_exp_wait_time, LOW)])
    train.extend(uwave_iq_train_long)
    train.extend([
              (post_uwave_exp_wait_time, LOW),
              (signal_time, LOW),
              (sig_to_ref_wait_time_long, LOW),
              (reference_time, LOW),
              (back_buffer + iq_delay_time, LOW)])
    seq.setDigital(pulser_do_arb_wave_trigger, train)
    # print(train)
    period = 0
    for el in train:
        period += el[0]
    print(period)
    
    final_digital = [pulser_wiring['do_sample_clock']]
    final = OutputState(final_digital, 0.0, 0.0)
    return seq, final, [period]

if __name__ == '__main__':
    config = tool_belt.get_config_dict()
    tool_belt.set_delays_to_zero(config)   
    # tau_shrt, polarization_time, gate_time, pi_pulse, pi_on_2_pulse, tau_long
    #pi_pulse_reps,  state, laser_name, laser_power
    seq_args = [1000, 10000.0, 300, 88, 44, 1000, 1, 3, 'integrated_520', None]
    seq, final, ret_vals = get_seq(None, config, seq_args)
    seq.plot()