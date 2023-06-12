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
    for ind in range(10):
        durations.append(numpy.int64(args[ind]))

    # Unpack the durations
    tau_shrt, polarization_time, ion_time, gate_time, pi_pulse_low, pi_on_2_pulse_low,\
        pi_pulse_high, pi_on_2_pulse_high, tau_long, comp_wait_time = durations
    
    

    # Get the APD indices
    pi_pulse_reps, state_activ, state_proxy,  \
        green_laser_name, green_laser_power, \
            red_laser_name, red_laser_power, \
                yellow_laser_name, yellow_laser_power = args[10:20]
    state_activ = States(state_activ)
    state_proxy = States(state_proxy)
    
        
    green_laser_delay_time = config['Optics'][green_laser_name]['delay']
    red_laser_delay_time = config['Optics'][red_laser_name]['delay']
    yellow_laser_delay_time = config['Optics'][yellow_laser_name]['delay']
    sig_gen_low_name = config['Servers']['sig_gen_LOW']
    uwave_delay_low = config['Microwaves'][sig_gen_low_name]['delay']
    sig_gen_high_name = config['Servers']['sig_gen_HIGH']
    uwave_delay_high = config['Microwaves'][sig_gen_high_name]['delay']
    iq_delay_time = config['Microwaves']['iq_delay']
    
    uwave_buffer = config['CommonDurations']['uwave_buffer']
    post_readout_buffer = uwave_buffer
    scc_ion_readout_buffer = config['CommonDurations']['scc_ion_readout_buffer']
    # print(scc_ion_readout_buffer)
    back_buffer = 200
    echo_buffer = comp_wait_time
    coh_buffer = comp_wait_time
    delay_buffer = max(green_laser_delay_time,red_laser_delay_time, yellow_laser_delay_time
                       ,uwave_delay_low, uwave_delay_high, iq_delay_time, 100)
    iq_trigger_time = numpy.int64(min(pi_on_2_pulse_low,pi_on_2_pulse_high , 10))
    
    # we half the tau to put an IQ pulse in between.But we need to make sure 
    # that the two halves still
    # add up to tau. To create shorthand, we define them here. 
    half_tau_shrt_st =numpy.int64(tau_shrt/2)
    half_tau_shrt_en = numpy.int64(tau_shrt -  half_tau_shrt_st)
    half_tau_long_st =numpy.int64(tau_long/2)
    half_tau_long_en = numpy.int64(tau_long -  half_tau_long_st)
    # tau_norm = numpy.int64(100)
    # half_tau_norm_st =numpy.int64(tau_norm/2)
    # half_tau_norm_en = numpy.int64(tau_norm -  half_tau_norm_st)
    
    pulser_wiring = config['Wiring']['PulseGen']
    pulser_do_apd_gate = pulser_wiring['do_apd_gate']
    sig_gen_gate_chan_name_low = 'do_{}_gate'.format(sig_gen_low_name)
    pulser_do_sig_gen_gate_low = pulser_wiring[sig_gen_gate_chan_name_low]
    sig_gen_gate_chan_name_high = 'do_{}_gate'.format(sig_gen_high_name)
    pulser_do_sig_gen_gate_high = pulser_wiring[sig_gen_gate_chan_name_high]
    pulser_do_arb_wave_trigger = pulser_wiring['do_arb_wave_trigger']

    ###


    coh_pulse_activ_low = 0
    coh_pulse_proxy_low = 0
    coh_pulse_activ_high = 0
    coh_pulse_proxy_high = 0
    
    echo_pulse_activ_low = 0
    echo_pulse_proxy_low = 0
    echo_pulse_activ_high = 0
    echo_pulse_proxy_high = 0
    
    if state_activ.value == States.LOW.value:
        coh_pulse_activ_low = pi_on_2_pulse_low
        coh_pulse_proxy_high = pi_pulse_high
        echo_pulse_activ_low = pi_pulse_low
        echo_pulse_proxy_high = pi_pulse_high
    elif state_activ.value == States.HIGH.value:
        coh_pulse_activ_high = pi_on_2_pulse_high
        coh_pulse_proxy_low = pi_pulse_low
        echo_pulse_activ_high = pi_pulse_high
        echo_pulse_proxy_low = pi_pulse_low
          
    uwave_coh_pulse_dur = coh_pulse_activ_low + coh_pulse_proxy_low + coh_buffer + \
        coh_pulse_activ_high + coh_pulse_proxy_high
    uwave_echo_pulse_dur = echo_pulse_proxy_low + echo_pulse_proxy_high + echo_buffer +\
                echo_pulse_activ_low + echo_pulse_activ_high + echo_buffer+ \
                echo_pulse_proxy_low + echo_pulse_proxy_high 
    # final_pi_pulse = echo_pulse_activ_low + echo_pulse_activ_high
    total_uwave_dur_low = echo_pulse_activ_low*2 + (echo_pulse_activ_low*3)*pi_pulse_reps
    total_uwave_dur_high = echo_pulse_activ_high*2 + (echo_pulse_activ_high*3)*pi_pulse_reps
    
    ### Write the microwave sequence to be used. 
    # Also add up the total time of the uwave experiment and use for other channels

    ###
    
    if pi_pulse_reps==0:
        rep_train_low_shrt = [(tau_shrt, LOW), 
                     (tau_shrt, LOW)]
    else:
        rep_train_low_shrt = [(tau_shrt, LOW), 
                     (echo_pulse_proxy_low, HIGH),
                     (echo_pulse_proxy_high, LOW),
                     (echo_buffer, LOW),
                     (echo_pulse_activ_low, HIGH), 
                     (echo_pulse_activ_high, LOW), 
                     (echo_buffer, LOW),
                     (echo_pulse_proxy_low, HIGH),
                     (echo_pulse_proxy_high, LOW),
                     (tau_shrt, LOW)]*pi_pulse_reps
    
    uwave_experiment_train_low_shrt = [(coh_pulse_activ_low, HIGH),
                                        (coh_pulse_activ_high, LOW),
                                        (coh_buffer, LOW),
                                        (coh_pulse_proxy_low, HIGH),
                                        (coh_pulse_proxy_high, LOW)]
    uwave_experiment_train_low_shrt.extend(rep_train_low_shrt)
    uwave_experiment_train_low_shrt.extend([(coh_pulse_proxy_low, HIGH),
                                            (coh_pulse_proxy_high, LOW),
                                            (coh_buffer, LOW),
                                            (coh_pulse_activ_low, HIGH),
                                            (coh_pulse_activ_high, LOW),
                                            ])
    # uwave_experiment_train_low_shrt.extend([(20, LOW)]) # adding a wait between pi/2 and pi
    # uwave_experiment_train_low_shrt.extend([(echo_pulse_activ_low, HIGH),
    #                                         (echo_pulse_activ_high, LOW)]) # adding a pi pulse to readout
    uwave_experiment_dur_shrt = 0
    for el in uwave_experiment_train_low_shrt:
        uwave_experiment_dur_shrt += el[0]
        
    ###
    if pi_pulse_reps==0:
        rep_train_low_long = [(tau_long, LOW), 
                     (tau_long, LOW)]
    else:
        rep_train_low_long = [(tau_long, LOW), 
                     (echo_pulse_proxy_low, HIGH),
                     (echo_pulse_proxy_high, LOW),
                     (echo_buffer, LOW),
                     (echo_pulse_activ_low, HIGH), 
                     (echo_pulse_activ_high, LOW), 
                     (echo_buffer, LOW),
                     (echo_pulse_proxy_low, HIGH),
                     (echo_pulse_proxy_high, LOW),
                     (tau_long, LOW)]*pi_pulse_reps
    uwave_experiment_train_low_long = [(coh_pulse_activ_low, HIGH),
                                        (coh_pulse_activ_high, LOW),
                                        (coh_buffer, LOW),
                                        (coh_pulse_proxy_low, HIGH),
                                        (coh_pulse_proxy_high, LOW)]
    uwave_experiment_train_low_long.extend(rep_train_low_long)
    uwave_experiment_train_low_long.extend([(coh_pulse_proxy_low, HIGH),
                                            (coh_pulse_proxy_high, LOW),
                                            (coh_buffer, LOW),
                                            (coh_pulse_activ_low, HIGH),
                                            (coh_pulse_activ_high, LOW),
                                            ])
    # uwave_experiment_train_low_long.extend([(20, LOW)]) # adding a wait between pi/2 and pi
    # uwave_experiment_train_low_long.extend([(echo_pulse_activ_low, HIGH),
    #                                         (echo_pulse_activ_high, LOW)]) # adding a pi pulse to readout
    uwave_experiment_dur_long = 0
    for el in uwave_experiment_train_low_long:
        uwave_experiment_dur_long += el[0]
        
    ###
    uwave_experiment_train_low_norm = [(total_uwave_dur_low*8, HIGH),
                                       (total_uwave_dur_high*8, LOW)
                                       ]
    
    # if pi_pulse_reps==0:
    #     rep_train_low_norm = [(tau_norm, LOW), 
    #                  (tau_norm, LOW)]
    # else:
    #     rep_train_low_norm = [(tau_norm, LOW), 
    #                  (echo_pulse_proxy_low, HIGH),
    #                  (echo_pulse_proxy_high, LOW),
    #                  (echo_buffer, LOW),
    #                  (echo_pulse_activ_low, HIGH), 
    #                  (echo_pulse_activ_high, LOW), 
    #                  (echo_buffer, LOW),
    #                  (echo_pulse_proxy_low, HIGH),
    #                  (echo_pulse_proxy_high, LOW),
    #                  (tau_norm, LOW)]*pi_pulse_reps
    # uwave_experiment_train_low_norm = [(coh_pulse_activ_low, HIGH),
    #                                     (coh_pulse_activ_high, LOW),
    #                                     (coh_buffer, LOW),
    #                                     (coh_pulse_proxy_low, HIGH),
    #                                     (coh_pulse_proxy_high, LOW)]
    # uwave_experiment_train_low_norm.extend(rep_train_low_norm)
    # uwave_experiment_train_low_norm.extend([(coh_pulse_proxy_low, HIGH),
    #                                         (coh_pulse_proxy_high, LOW),
    #                                         (coh_buffer, LOW),
    #                                         (coh_pulse_activ_low, HIGH),
    #                                         (coh_pulse_activ_high, LOW),
    #                                         ])
    # # uwave_experiment_train_low_norm.extend([(20, LOW)]) # adding a wait between pi/2 and pi
    # # uwave_experiment_train_low_norm.extend([(echo_pulse_activ_low, HIGH),
    # #                                         (echo_pulse_activ_high, LOW)]) # adding a pi pulse to readout
    uwave_experiment_dur_norm = 0
    for el in uwave_experiment_train_low_norm:
        uwave_experiment_dur_norm += el[0]
    
    
    ###
    if pi_pulse_reps==0:
        rep_train_high_shrt = [(tau_shrt, LOW), 
                     (tau_shrt, LOW)]
    else:
        rep_train_high_shrt = [(tau_shrt, LOW), 
                     (echo_pulse_proxy_low, LOW),
                     (echo_pulse_proxy_high, HIGH),
                     (echo_buffer, LOW),
                     (echo_pulse_activ_low, LOW), 
                     (echo_pulse_activ_high, HIGH), 
                     (echo_buffer, LOW),
                     (echo_pulse_proxy_low, LOW),
                     (echo_pulse_proxy_high, HIGH),
                     (tau_shrt, LOW)]*pi_pulse_reps
    
    uwave_experiment_train_high_shrt = [(coh_pulse_activ_low, LOW),
                                (coh_pulse_activ_high, HIGH),
                                (coh_buffer, LOW),
                                (coh_pulse_proxy_low, LOW),
                                (coh_pulse_proxy_high, HIGH)]
    uwave_experiment_train_high_shrt.extend(rep_train_high_shrt)
    uwave_experiment_train_high_shrt.extend([(coh_pulse_proxy_low, LOW),
                                            (coh_pulse_proxy_high, HIGH),
                                            (coh_buffer, LOW),
                                            (coh_pulse_activ_low, LOW),
                                            (coh_pulse_activ_high, HIGH),
                                            ])
    # uwave_experiment_train_high_shrt.extend([(20, LOW)]) # adding a wait between pi/2 and pi
    # uwave_experiment_train_high_shrt.extend([(echo_pulse_activ_high, HIGH),
    #                                         (echo_pulse_activ_low, LOW)]) # adding a pi pulse to readout
        
    ###
    if pi_pulse_reps==0:
        rep_train_high_long = [(tau_long, LOW), 
                     (tau_long, LOW)]
    else:
        rep_train_high_long = [(tau_long, LOW), 
                     (echo_pulse_proxy_low, LOW),
                     (echo_pulse_proxy_high, HIGH),
                     (echo_buffer, LOW),
                     (echo_pulse_activ_low, LOW), 
                     (echo_pulse_activ_high, HIGH), 
                     (echo_buffer, LOW),
                     (echo_pulse_proxy_low, LOW),
                     (echo_pulse_proxy_high, HIGH),
                     (tau_long, LOW)]*pi_pulse_reps
    uwave_experiment_train_high_long = [(coh_pulse_activ_low, LOW),
                                (coh_pulse_activ_high, HIGH),
                                (coh_buffer, LOW),
                                (coh_pulse_proxy_low, LOW),
                                (coh_pulse_proxy_high, HIGH)]
    uwave_experiment_train_high_long.extend(rep_train_high_long)
    uwave_experiment_train_high_long.extend([(coh_pulse_proxy_low, LOW),
                                            (coh_pulse_proxy_high, HIGH),
                                            (coh_buffer, LOW),
                                            (coh_pulse_activ_low, LOW),
                                            (coh_pulse_activ_high, HIGH),
                                            ])
    # uwave_experiment_train_high_long.extend([(20, LOW)]) # adding a wait between pi/2 and pi
    # uwave_experiment_train_high_long.extend([(echo_pulse_activ_high, HIGH),
    #                                         (echo_pulse_activ_low, LOW)]) # adding a pi pulse to readout
    
    ###
    uwave_experiment_train_high_norm = [(total_uwave_dur_low*8, LOW),
                                       (total_uwave_dur_high*8, HIGH)
                                       ]
    # if pi_pulse_reps==0:
    #     rep_train_high_norm = [(tau_norm, LOW), 
    #                  (tau_norm, LOW)]
    # else:
    #     rep_train_high_norm = [(tau_norm, LOW), 
    #                  (echo_pulse_proxy_low, LOW),
    #                  (echo_pulse_proxy_high, HIGH),
    #                  (echo_buffer, LOW),
    #                  (echo_pulse_activ_low, LOW), 
    #                  (echo_pulse_activ_high, HIGH), 
    #                  (echo_buffer, LOW),
    #                  (echo_pulse_proxy_low, LOW),
    #                  (echo_pulse_proxy_high, HIGH),
    #                  (tau_norm, LOW)]*pi_pulse_reps
    # uwave_experiment_train_high_norm = [(coh_pulse_activ_low, LOW),
    #                             (coh_pulse_activ_high, HIGH),
    #                             (coh_buffer, LOW),
    #                             (coh_pulse_proxy_low, LOW),
    #                             (coh_pulse_proxy_high, HIGH)]
    # uwave_experiment_train_high_norm.extend(rep_train_high_norm)
    # uwave_experiment_train_high_norm.extend([(coh_pulse_proxy_low, LOW),
    #                                         (coh_pulse_proxy_high, HIGH),
    #                                         (coh_buffer, LOW),
    #                                         (coh_pulse_activ_low, LOW),
    #                                         (coh_pulse_activ_high, HIGH),
    #                                         ])
    # # uwave_experiment_train_high_norm.extend([(20, LOW)]) # adding a wait between pi/2 and pi
    # # uwave_experiment_train_high_norm.extend([(echo_pulse_activ_high, HIGH),
    # #                                         (echo_pulse_activ_low, LOW)]) # adding a pi pulse to readout

    ### Write the IQ trigger pulse sequence
    # The first IQ pulse will occur right after optical pulse polarization
    # follwoing IQ pulses will occur half tau before the pi pulse
    uwave_iq_train_shrt = [(iq_trigger_time, HIGH), 
                           (uwave_buffer + uwave_coh_pulse_dur-iq_trigger_time, LOW),
                           (half_tau_shrt_st, LOW)]
    if pi_pulse_reps == 0:
        uwave_iq_train_shrt.extend([(iq_trigger_time, HIGH), 
                                    (half_tau_shrt_en - iq_trigger_time + half_tau_shrt_st, LOW),
                                    (half_tau_shrt_en + uwave_coh_pulse_dur, LOW)])
    else:
        rep_train = [(iq_trigger_time, HIGH),
                 (half_tau_shrt_en - iq_trigger_time + uwave_echo_pulse_dur +\
                  tau_shrt + half_tau_shrt_st, LOW)]*(pi_pulse_reps-1)
        uwave_iq_train_shrt.extend(rep_train)
        uwave_iq_train_shrt.extend([(iq_trigger_time, HIGH), 
                                    (half_tau_shrt_en - iq_trigger_time + uwave_echo_pulse_dur + half_tau_shrt_st, LOW),
                                    (iq_trigger_time, HIGH), 
                                    (half_tau_shrt_en - iq_trigger_time+ uwave_coh_pulse_dur, LOW)])
                                    # (half_tau_shrt_en - iq_trigger_time+ uwave_coh_pulse_dur + 20 + final_pi_pulse, LOW)])
    
    
    uwave_iq_train_long = [(iq_trigger_time, HIGH), 
                           (uwave_buffer + uwave_coh_pulse_dur-iq_trigger_time, LOW),
                           (half_tau_long_st, LOW)]
    if pi_pulse_reps == 0:
        uwave_iq_train_long.extend([(iq_trigger_time, HIGH), 
                                    (half_tau_long_en - iq_trigger_time + half_tau_long_st, LOW),
                                    (half_tau_long_en + uwave_coh_pulse_dur, LOW)])
    else:
        rep_train = [(iq_trigger_time, HIGH),
                 (half_tau_long_en - iq_trigger_time + uwave_echo_pulse_dur +\
                  tau_long + half_tau_long_st, LOW)]*(pi_pulse_reps-1)
        uwave_iq_train_long.extend(rep_train)
        uwave_iq_train_long.extend([(iq_trigger_time, HIGH), 
                                    (half_tau_long_en - iq_trigger_time + uwave_echo_pulse_dur + half_tau_long_st, LOW),
                                    (iq_trigger_time, HIGH), 
                                    (half_tau_long_en - iq_trigger_time + uwave_coh_pulse_dur, LOW)])
                                    # (half_tau_long_en - iq_trigger_time + uwave_coh_pulse_dur + 20+ final_pi_pulse, LOW)])
        

    
    # uwave_iq_train_norm = [(iq_trigger_time, HIGH), 
    #                        (uwave_buffer + uwave_coh_pulse_dur-iq_trigger_time, LOW),
    #                        (half_tau_norm_st, LOW)]
    # if pi_pulse_reps == 0:
    #     uwave_iq_train_norm.extend([(iq_trigger_time, HIGH), 
    #                                 (half_tau_norm_en - iq_trigger_time + half_tau_norm_st, LOW),
    #                                 (half_tau_norm_en + uwave_coh_pulse_dur, LOW)])
    # else:
    #     rep_train = [(iq_trigger_time, HIGH),
    #              (half_tau_norm_en - iq_trigger_time + uwave_echo_pulse_dur +\
    #               tau_norm + half_tau_norm_st, LOW)]*(pi_pulse_reps-1)
    #     uwave_iq_train_norm.extend(rep_train)
    #     uwave_iq_train_norm.extend([(iq_trigger_time, HIGH), 
    #                                 (half_tau_norm_en - iq_trigger_time + uwave_echo_pulse_dur + half_tau_norm_st, LOW),
    #                                 (iq_trigger_time, HIGH), 
    #                                 (half_tau_norm_en - iq_trigger_time + uwave_coh_pulse_dur, LOW)])
    #                                 # (half_tau_norm_en - iq_trigger_time + uwave_coh_pulse_dur + 20+ final_pi_pulse, LOW)])
                                    
    ### Define the sequence

    seq = Sequence()

    # APD gating
    train = [(delay_buffer, LOW),
             (polarization_time,LOW),
             (uwave_buffer, LOW),
             (uwave_experiment_dur_shrt, LOW),
             (uwave_buffer, LOW),
             (ion_time, LOW),
             (scc_ion_readout_buffer, LOW),
             (gate_time, HIGH),
             (post_readout_buffer, LOW),
             (polarization_time, LOW),
             (uwave_buffer, LOW),
             # (uwave_experiment_dur_norm, LOW),
             (uwave_buffer, LOW),
             (ion_time, LOW),
             (scc_ion_readout_buffer, LOW),
             (gate_time, HIGH),
             (post_readout_buffer, LOW),
             (polarization_time, LOW),
             (uwave_buffer, LOW),
             (uwave_experiment_dur_long, LOW),
             (uwave_buffer, LOW),
             (ion_time, LOW),
             (scc_ion_readout_buffer, LOW),
             (gate_time, HIGH),
             (post_readout_buffer, LOW),
             (polarization_time , LOW),
             (uwave_buffer, LOW),
            #  (uwave_experiment_dur_norm, LOW),
             (uwave_buffer, LOW),
             (ion_time, LOW),
             (scc_ion_readout_buffer, LOW),
             (gate_time, HIGH),
             (back_buffer, LOW)]             
    seq.setDigital(pulser_do_apd_gate, train)
    period = 0
    for el in train:
        period += el[0]
    print(period)

    # Green Laser
    train = [(delay_buffer - green_laser_delay_time, HIGH),
             (polarization_time, HIGH),
             (uwave_buffer, LOW),
             (uwave_experiment_dur_shrt, LOW),
             (uwave_buffer, LOW),
             (ion_time, LOW),
             (scc_ion_readout_buffer, LOW),
             (gate_time, LOW),
             (post_readout_buffer, LOW),
             (polarization_time, HIGH),
             (uwave_buffer, LOW),
            #  (uwave_experiment_dur_norm, LOW),
             (uwave_buffer, LOW),
             (ion_time, LOW),
             (scc_ion_readout_buffer, LOW),
             (gate_time, LOW),
             (post_readout_buffer, LOW),
             (polarization_time, HIGH),
             (uwave_buffer, LOW),
             (uwave_experiment_dur_long, LOW),
             (uwave_buffer, LOW),
             (ion_time, LOW),
             (scc_ion_readout_buffer, LOW),
             (gate_time, LOW),
             (post_readout_buffer, LOW),
             (polarization_time, HIGH),
             (uwave_buffer, LOW),
           #   (uwave_experiment_dur_norm, LOW),
             (uwave_buffer, LOW),
             (ion_time, LOW),
             (scc_ion_readout_buffer, LOW),
             (gate_time, LOW),
             (back_buffer + green_laser_delay_time, LOW)]   
    tool_belt.process_laser_seq(pulse_streamer, seq, config,
                                green_laser_name, green_laser_power, train)
    period = 0
    for el in train:
        period += el[0]
    print(period)

    # Red Laser
    train = [(delay_buffer - red_laser_delay_time, LOW),
             (polarization_time, LOW),
             (uwave_buffer, LOW),
             (uwave_experiment_dur_shrt, LOW),
             (uwave_buffer, LOW),
             (ion_time, HIGH),
             (scc_ion_readout_buffer, LOW),
             (gate_time, LOW),
             (post_readout_buffer, LOW),
             (polarization_time, LOW),
             (uwave_buffer, LOW),
            #  (uwave_experiment_dur_norm, LOW),
             (uwave_buffer, LOW),
             (ion_time, HIGH),
             (scc_ion_readout_buffer, LOW),
             (gate_time, LOW),
             (post_readout_buffer, LOW),
             (polarization_time, LOW),
             (uwave_buffer, LOW),
             (uwave_experiment_dur_long, LOW),
             (uwave_buffer, LOW),
             (ion_time, HIGH),
             (scc_ion_readout_buffer, LOW),
             (gate_time, LOW),
             (post_readout_buffer, LOW),
             (polarization_time, LOW),
             (uwave_buffer, LOW),
            #  (uwave_experiment_dur_norm, LOW),
             (uwave_buffer, LOW),
             (ion_time, HIGH),
             (scc_ion_readout_buffer, LOW),
             (gate_time, LOW),
             (back_buffer + red_laser_delay_time, LOW)]   
    tool_belt.process_laser_seq(pulse_streamer, seq, config,
                                red_laser_name, red_laser_power, train)
    period = 0
    for el in train:
        period += el[0]
    print(period)
    
    # Yellow Laser
    train = [(delay_buffer - yellow_laser_delay_time, LOW),
             (polarization_time, LOW),
             (uwave_buffer, LOW),
             (uwave_experiment_dur_shrt, LOW),
             (uwave_buffer, LOW),
             (ion_time, LOW),
             (scc_ion_readout_buffer, LOW),
             (gate_time, HIGH),
             (post_readout_buffer, LOW),
             (polarization_time, LOW),
             (uwave_buffer, LOW),
            #  (uwave_experiment_dur_norm, LOW),
             (uwave_buffer, LOW),
             (ion_time, LOW),
             (scc_ion_readout_buffer, LOW),
             (gate_time, HIGH),
             (post_readout_buffer, LOW),
             (polarization_time, LOW),
             (uwave_buffer, LOW),
             (uwave_experiment_dur_long, LOW),
             (uwave_buffer, LOW),
             (ion_time, LOW),
             (scc_ion_readout_buffer, LOW),
             (gate_time, HIGH),
             (post_readout_buffer, LOW),
             (polarization_time, LOW),
             (uwave_buffer, LOW),
            #  (uwave_experiment_dur_norm, LOW),
             (uwave_buffer, LOW),
             (ion_time, LOW),
             (scc_ion_readout_buffer, LOW),
             (gate_time, HIGH),
             (back_buffer + yellow_laser_delay_time, LOW)]   
    tool_belt.process_laser_seq(pulse_streamer, seq, config,
                                yellow_laser_name, [yellow_laser_power], train)
    period = 0
    for el in train:
        period += el[0]
    print(period)
    
    # Microwaves LOW
    train = [(delay_buffer - uwave_delay_low, LOW),
             (polarization_time,LOW),
             (uwave_buffer, LOW)]
    train.extend(uwave_experiment_train_low_shrt)
    train.extend([
             (uwave_buffer, LOW),
             (ion_time, LOW),
             (scc_ion_readout_buffer, LOW),
             (gate_time, LOW),
             (post_readout_buffer, LOW),
             (polarization_time, LOW),
             (uwave_buffer, LOW)])
   # train.extend(uwave_experiment_train_low_norm)
    train.extend([
             (uwave_buffer, LOW),
             (ion_time, LOW),
             (scc_ion_readout_buffer, LOW),
             (gate_time, LOW),
             (post_readout_buffer, LOW),
             (polarization_time, LOW),
             (uwave_buffer, LOW)])
    train.extend(uwave_experiment_train_low_long)
    train.extend([
             (uwave_buffer, LOW),
             (ion_time, LOW),
             (scc_ion_readout_buffer, LOW),
             (gate_time, LOW),
             (post_readout_buffer, LOW),
             (polarization_time, LOW),
             (uwave_buffer, LOW)])
    #train.extend(uwave_experiment_train_low_norm)
    train.extend([
             (uwave_buffer, LOW),
             (ion_time, LOW),
             (scc_ion_readout_buffer, LOW),
             (gate_time, LOW),
             (back_buffer + uwave_delay_low, LOW)])
    seq.setDigital(pulser_do_sig_gen_gate_low, train)
    
    period = 0
    for el in train:
        period += el[0]
    print(period)

    # Microwaves HIGH
    train = [(delay_buffer - uwave_delay_high, LOW),
             (polarization_time,LOW),
             (uwave_buffer, LOW)]
    train.extend(uwave_experiment_train_high_shrt)
    train.extend([
             (uwave_buffer, LOW),
             (ion_time, LOW),
             (scc_ion_readout_buffer, LOW),
             (gate_time, LOW),
             (post_readout_buffer, LOW),
             (polarization_time, LOW),
             (uwave_buffer, LOW)])
  #  train.extend(uwave_experiment_train_high_norm)
    train.extend([
             (uwave_buffer, LOW),
             (ion_time, LOW),
             (scc_ion_readout_buffer, LOW),
             (gate_time, LOW),
             (post_readout_buffer, LOW),
             (polarization_time, LOW),
             (uwave_buffer, LOW)])
    train.extend(uwave_experiment_train_high_long)
    train.extend([
             (uwave_buffer, LOW),
             (ion_time, LOW),
             (scc_ion_readout_buffer, LOW),
             (gate_time, LOW),
             (post_readout_buffer, LOW),
             (polarization_time, LOW),
             (uwave_buffer, LOW)])
   # train.extend(uwave_experiment_train_high_norm)
    train.extend([
             (uwave_buffer, LOW),
             (ion_time, LOW),
             (scc_ion_readout_buffer, LOW),
             (gate_time, LOW),
             (back_buffer + uwave_delay_high, LOW)])
    seq.setDigital(pulser_do_sig_gen_gate_high, train)
    
    period = 0
    for el in train:
        period += el[0]
    print(period)
    
    # IQ modulation triggers
    train = [(delay_buffer - iq_delay_time, LOW),
             (iq_trigger_time, HIGH),
             (polarization_time-iq_trigger_time, LOW)]
              # (uwave_buffer, LOW)]
    train.extend(uwave_iq_train_shrt)
    train.extend([
              (uwave_buffer, LOW),
              (ion_time, LOW),
              (scc_ion_readout_buffer, LOW),
              (gate_time, LOW),
              (post_readout_buffer, LOW),
              (polarization_time, LOW),
               (uwave_buffer, LOW)])
   # train.extend(uwave_experiment_dur_norm)
    train.extend([
              (uwave_buffer, LOW),
              (ion_time, LOW),
              (scc_ion_readout_buffer, LOW),
              (gate_time, LOW),
              (post_readout_buffer, LOW),
              (polarization_time, LOW),])
              # (uwave_buffer, LOW)])
    train.extend(uwave_iq_train_long)
    train.extend([
              (uwave_buffer, LOW),
              (ion_time, LOW),
              (scc_ion_readout_buffer, LOW),
              (gate_time, LOW),
              (post_readout_buffer, LOW),
              (polarization_time, LOW),
               (uwave_buffer, LOW)])
   # train.extend(uwave_experiment_dur_norm)
    train.extend([
              (uwave_buffer, LOW),
              (ion_time, LOW),
              (scc_ion_readout_buffer, LOW),
              (gate_time, LOW),
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
    # tau_shrt, polarization_time, ion_time, gate_time, pi_pulse_low, pi_on_2_pulse_low,\
    #     pi_pulse_high, pi_on_2_pulse_high, tau_long = durations
    # pi_pulse_reps, state_activ, state_proxy,  \
    #     green_laser_name, green_laser_power, \
    #         red_laser_name, red_laser_power, \
    #             yellow_laser_name, yellow_laser_power = args[9:19]
    seq_args =[100, 1000.0, 250, 1000.0, 69.71, 34.41, 56.38, 28.75, 1000, 
               132.0, 2, 3, 1, 'integrated_520', None, 'cobolt_638', None, 'laser_LGLO_589', None]
    seq, final, ret_vals = get_seq(None, config, seq_args)
    seq.plot()