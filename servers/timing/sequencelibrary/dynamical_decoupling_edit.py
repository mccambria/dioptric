# -*- coding: utf-8 -*-
"""
Created on Thur Aug 4, 2022

This is a sequence for dynamical decoupling sequence for routines that have
the following family of microwave sequences:

oi/2 - (T - pi - T)*N - pi/2

For example, CPMG pulses use this with N>1 repetitions. And XY4 and XY8 use this
with N = mod 8.
    
The phase of the microwave pulses are controlled in the actual routine, so this can be used
for CPMG, XY4, or XY8.

Note that the variable pi_pulse_reps is the number of pi_pulses to perform. 
For example, if we want to perform XY4-2, we would want the number of pi pulses
to be 8 * 2. There are 8 pusles in the XY4 pulse sequence, 
and we want to repeate those 2 times.

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
    for ind in range(6):
        durations.append(numpy.int64(args[ind]))

    # Unpack the durations
    tau_shrt, polarization_time, gate_time, pi_pulse, pi_on_2_pulse, tau_long = durations
    
    

    # Get the APD indices
    pi_pulse_reps, apd_index, state, laser_name, laser_power = args[6:12]
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
    sig_gen_name = config['Microwaves']['sig_gen_{}'.format(state.name)]
    rf_delay_time = config['Microwaves'][sig_gen_name]['delay']
    iq_delay_time = config['Microwaves']['iq_delay']
    
    back_buffer = 200
    delay_buffer = max(laser_delay_time,rf_delay_time, iq_delay_time, 100)
    # shift the iq triggers to half tau before the pi pulses
    iq_forward_shift_shrt = numpy.int64(tau_shrt/2 )
    iq_forward_shift_long = numpy.int64(tau_long/2)
    iq_trigger_time = numpy.int64(min(pi_on_2_pulse, 10))
    
    # we often half the tau to put a pulse in between.To create shorthand,
    # we define them here. But we need to make sure that the two halves still
    # add up to tau
    half_tau_shrt_st =int(tau_shrt/2)
    half_tau_shrt_en = int(tau_shrt -  half_tau_shrt_st)
    half_tau_long_st =int(tau_long/2)
    half_tau_long_en = int(tau_long -  half_tau_long_st)
    
    pulser_wiring = config['Wiring']['PulseStreamer']
    pulser_do_apd_gate = pulser_wiring['do_apd_{}_gate'.format(apd_index)]
    sig_gen_gate_chan_name = 'do_{}_gate'.format(sig_gen_name)
    pulser_do_sig_gen_gate = pulser_wiring[sig_gen_gate_chan_name]
    pulser_do_arb_wave_trigger = pulser_wiring['do_arb_wave_trigger']

    # %% Write the microwave sequence to be used.

    # In t1, the sequence is just a pi pulse, wait for a relaxation time, then
    # then a second pi pulse

    # I define both the time of this experiment, which is useful for the AOM
    # and gate sequences to dictate the time for them to be LOW
    # And I define the actual uwave experiement to be plugged into the rf
    # sequence. I hope that this formatting works.

    # With future protocols--ramsey, spin echo, etc--it will be easy to use
    # this format of sequence building and just change this section of the file

    # uwave_experiment_dur_shrt = (pi_on_2_pulse + \
    #                  (tau_shrt + pi_pulse + tau_shrt)*8*num_dd_reps + \
    #                          pi_on_2_pulse)
    # uwave_experiment_dur_long = (pi_on_2_pulse + \
    #                  (tau_long + pi_pulse + tau_long)*8*num_dd_reps + \
    #                          pi_on_2_pulse)   
    
    uwave_experiment_train_shrt = [(pi_on_2_pulse, HIGH)]
    rep_train = [(tau_shrt, LOW), (pi_pulse, HIGH), (tau_shrt, LOW)]*pi_pulse_reps
    uwave_experiment_train_shrt.extend(rep_train)
    uwave_experiment_train_shrt.extend([(pi_on_2_pulse, HIGH)])
    
    uwave_experiment_dur_shrt = 0
    for el in uwave_experiment_train_shrt:
        uwave_experiment_dur_shrt += el[0]
    
    
    uwave_experiment_train_long = [(pi_on_2_pulse, HIGH)]
    rep_train = [(tau_long, LOW), (pi_pulse, HIGH), (tau_long, LOW)]*pi_pulse_reps
    uwave_experiment_train_long.extend(rep_train)
    uwave_experiment_train_long.extend([(pi_on_2_pulse, HIGH)])
    
    uwave_experiment_dur_long = 0
    for el in uwave_experiment_train_long:
        uwave_experiment_dur_long += el[0]
        
    # The first IQ pulse will occur right after optical pulse polarization
    uwave_iq_train_shrt = [(iq_trigger_time, HIGH), 
                            (pre_uwave_exp_wait_time + pi_on_2_pulse-iq_trigger_time, LOW),
                            (half_tau_shrt_st, LOW)]
    rep_train = [(iq_trigger_time, HIGH),
              (half_tau_shrt_en - iq_trigger_time + pi_pulse + tau_shrt + half_tau_shrt_st, LOW)]*(pi_pulse_reps-1)
    uwave_iq_train_shrt.extend(rep_train)
    uwave_iq_train_shrt.extend([(iq_trigger_time, HIGH), 
                                (half_tau_shrt_en - iq_trigger_time + pi_pulse + half_tau_shrt_st, LOW),
                                (iq_trigger_time, HIGH), 
                                (half_tau_shrt_en - iq_trigger_time + pi_on_2_pulse, LOW)])
    # print(uwave_iq_train_shrt)
    
    # first IQ pulse will occur right at start of pi/2 pulse
    # uwave_iq_train_shrt = [(iq_trigger_time, HIGH), 
    #                         (pi_on_2_pulse-iq_trigger_time, LOW)]
    # rep_train = [(tau_shrt, LOW),
    #               (iq_trigger_time, HIGH),
    #               (pi_pulse - iq_trigger_time + tau_shrt, LOW)]*(pi_pulse_reps)
    # uwave_iq_train_shrt.extend(rep_train)
    # uwave_iq_train_shrt.extend([(iq_trigger_time, HIGH), 
    #                             (pi_on_2_pulse - iq_trigger_time, LOW)])
    
    
    uwave_iq_train_long = [(iq_trigger_time, HIGH), 
                            (pre_uwave_exp_wait_time + pi_on_2_pulse-iq_trigger_time, LOW),
                            (half_tau_long_st, LOW)]
    rep_train = [(iq_trigger_time, HIGH),
              (half_tau_long_en - iq_trigger_time + pi_pulse + tau_long + half_tau_long_st, LOW)]*(pi_pulse_reps-1)
    uwave_iq_train_long.extend(rep_train)
    uwave_iq_train_long.extend([(iq_trigger_time, HIGH), 
                                (half_tau_long_en - iq_trigger_time + pi_pulse + half_tau_long_st, LOW),
                                (iq_trigger_time, HIGH), 
                                (half_tau_long_en - iq_trigger_time + pi_on_2_pulse, LOW)])
    
    
    # uwave_iq_train_long = [(iq_trigger_time, HIGH), 
    #                         (pi_on_2_pulse-iq_trigger_time, LOW)]
    # rep_train = [(tau_long, LOW),
    #               (iq_trigger_time, HIGH),
    #               (pi_pulse - iq_trigger_time + tau_long, LOW)]*(pi_pulse_reps)
    # uwave_iq_train_long.extend(rep_train)
    # uwave_iq_train_long.extend([(iq_trigger_time, HIGH), 
    #                             (pi_on_2_pulse - iq_trigger_time, LOW)])
    

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
    # train.extend([(uwave_experiment_dur_shrt, HIGH)])
    train.extend([
             (post_uwave_exp_wait_time, LOW),
             (signal_time, LOW),
             (sig_to_ref_wait_time_shrt, LOW),
             (reference_time, LOW),
             (pre_uwave_exp_wait_time, LOW)])
    train.extend(uwave_experiment_train_long)
    # train.extend([(uwave_experiment_dur_long, HIGH)])
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
              (polarization_time,LOW),]
              # (pre_uwave_exp_wait_time-20, LOW)]
    train.extend(uwave_iq_train_shrt)
    train.extend([
              (post_uwave_exp_wait_time, LOW),
              (signal_time, LOW),
              (sig_to_ref_wait_time_shrt, LOW),
              (reference_time, LOW),])
              # (pre_uwave_exp_wait_time-20, LOW)])
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
    #pi_pulse_reps, apd_index, state, laser_name, laser_power
    seq_args = [62, 1000.0, 350, 45, 23, 3062, 8, 1, 3, 'integrated_520', None]
    seq, final, ret_vals = get_seq(None, config, seq_args)
    seq.plot()
