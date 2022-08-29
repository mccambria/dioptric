# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 15:17:35 2022

based off this paper: https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.105.077601

The signal, if no errors, should be half the contrast, which we will define as 0

@author: kolkowitz
"""


import utils.tool_belt as tool_belt
import majorroutines.optimize as optimize
from scipy.optimize import minimize_scalar
from numpy import pi
import numpy
import time
import matplotlib.pyplot as plt
from random import shuffle
import labrad
from utils.tool_belt import States
from scipy.optimize import curve_fit


def measure(cxn, 
            nv_sig,
            uwave_pi_pulse,
            num_uwave_pulses,
            iq_phases,
            pulse_1_dur,
            pulse_2_dur,
            pulse_3_dur,
            apd_indices,
            state=States.HIGH,):
    
    # print(iq_phases)
    num_reps = int(5e6)
    
    tool_belt.reset_cfm(cxn)
    seq_file = 'test_iq_pulse_errors.py'
    #  Sequence setup

    laser_key = "spin_laser"
    laser_name = nv_sig[laser_key]
    tool_belt.set_filter(cxn, nv_sig, laser_key)
    laser_power = tool_belt.set_laser_power(cxn, nv_sig, laser_key)
    polarization_time = nv_sig["spin_pol_dur"]
    gate_time = nv_sig["spin_readout_dur"]


    
    optimize.main_with_cxn(cxn, nv_sig, apd_indices)
    # Turn on the microwaves for determining microwave delay
    sig_gen_cxn = tool_belt.get_signal_generator_cxn(cxn, state)
    sig_gen_cxn.set_freq(nv_sig["resonance_{}".format(state.name)])
    sig_gen_cxn.set_amp(nv_sig["uwave_power_{}".format(state.name)])
    sig_gen_cxn.load_iq()
    sig_gen_cxn.uwave_on()
    cxn.arbitrary_waveform_generator.load_arb_phases(iq_phases)

    cxn.apd_tagger.start_tag_stream(apd_indices)
        
        
    seq_args = [gate_time, uwave_pi_pulse, 
            pulse_1_dur, pulse_2_dur, pulse_3_dur, 
            polarization_time, num_uwave_pulses, state.value, apd_indices[0], laser_name, laser_power]
    # print(seq_args)
    # return
    cxn.apd_tagger.clear_buffer()
    seq_args_string = tool_belt.encode_seq_args(seq_args)
    cxn.pulse_streamer.stream_immediate(
        seq_file, num_reps, seq_args_string
    )

    new_counts = cxn.apd_tagger.read_counter_separate_gates(1)
    sample_counts = new_counts[0]
    if len(sample_counts) != 3 * num_reps:
        print("Error!")
    # first are the counts after polarization into ms = 0
    ref_0_counts = sample_counts[0::3] 
    # second are the counts after a pi_x into +/-1
    ref_H_counts = sample_counts[1::3]
    # third are the counts after the uwave sequence
    sig_counts = sample_counts[2::3]

    cxn.apd_tagger.stop_tag_stream()
    
    tool_belt.reset_cfm(cxn)
    
    # print(ref_0_counts)
    # print(ref_H_counts)
    # print(sig_counts)
    # analysis
    ref_0_avg = numpy.average(ref_0_counts)
    ref_H_avg = numpy.average(ref_H_counts)
    sig_avg = numpy.average(sig_counts)
    # print(ref_0_avg)
    # print(ref_H_avg)
    # print(sig_avg)
    
    ref_0_ste = numpy.std(
        ref_0_counts, ddof=1
        ) / numpy.sqrt(num_reps)
    ref_H_ste = numpy.std(
        ref_H_counts, ddof=1
        ) / numpy.sqrt(num_reps)
    sig_ste = numpy.std(
        sig_counts, ddof=1
        ) / numpy.sqrt(num_reps)
    
    contrast = ref_0_avg-ref_H_avg
    contrast_ste = numpy.sqrt(ref_0_ste**2 + ref_H_ste**2)
    signal_m_H = sig_avg-ref_H_avg
    signal_m_H_ste = numpy.sqrt(sig_ste**2 + ref_H_ste**2)
    half_contrast = 0.5
    
    signal_perc = signal_m_H / contrast
    # print(signal_perc)
    signal_perc_ste = signal_perc*numpy.sqrt((contrast_ste/contrast)**2 + \
                                             (signal_m_H_ste/signal_m_H)**2)
        
    # half_contrast_counts = contrast/2 + ref_H_avg
    # print(half_contrast_counts)
    # half_contrast_ste = numpy.sqrt(ref_H_ste**2 + (contrast_ste/2)**2)
    
    pulse_error = signal_perc - half_contrast
    pulse_error_ste = signal_perc_ste
    
        
    return pulse_error, pulse_error_ste

def test_1_pulse(cxn, 
                 nv_sig,
                 apd_indices,
                 state=States.HIGH,):
    '''
    This pulse sequence consists of pi/2 pulses with the same phase:
        1: pi/2_x
        2: pi/2_y
    '''

    num_uwave_pulses = 1
    
    rabi_period = nv_sig["rabi_{}".format(state.name)]

    # Get pulse frequencies
    uwave_pi_pulse = tool_belt.get_pi_pulse_dur(rabi_period)
    uwave_pi_on_2_pulse = tool_belt.get_pi_on_2_pulse_dur(rabi_period)
    
    pulse_1_dur = uwave_pi_on_2_pulse
    pulse_2_dur = 0
    pulse_3_dur = 0
    
    phases_x = [0, 0]
    phases_y = [0, pi/2]
    
    ##### 1
    iq_phases = phases_x
    pulse_error, pulse_error_ste = measure(cxn, 
                nv_sig,
                uwave_pi_pulse,
                num_uwave_pulses,
                iq_phases,
                pulse_1_dur,
                pulse_2_dur,
                pulse_3_dur,
                apd_indices,
                state=States.HIGH)
    pe_1_1 = pulse_error
    pe_1_1_err = pulse_error_ste
    
    print(r"pi/2_x rotation angle error, -2 phi' = {:.4f} +/- {:.4f}".format(pulse_error, pulse_error_ste))
    
    #### 2
    iq_phases = phases_y
    pulse_error, pulse_error_ste = measure(cxn, 
                nv_sig,
                uwave_pi_pulse,
                num_uwave_pulses,
                iq_phases,
                pulse_1_dur,
                pulse_2_dur,
                pulse_3_dur,
                apd_indices,
                state=States.HIGH)
    pe_1_2 = pulse_error
    pe_1_2_err = pulse_error_ste
    
    print(r"pi/2_y rotation angle error, -2 chi' = {:.4f} +/- {:.4f}".format(pulse_error, pulse_error_ste))
    return pe_1_1, pe_1_1_err, pe_1_2, pe_1_2_err

def test_2_pulse(cxn, 
                 nv_sig,
                 apd_indices,
                 state=States.HIGH,
                 ):
    '''
        1: pi_y - pi/2_x
        2: pi_x - pi/2_y
        3: pi/2_x - pi_y
        4: pi/2_y - pi_x
        5: pi/2_x - pi/2_y
        6: pi/2_y - pi/2_x
    '''
    
    num_uwave_pulses = 2
    
    rabi_period = nv_sig["rabi_{}".format(state.name)]

    # Get pulse frequencies
    uwave_pi_pulse = tool_belt.get_pi_pulse_dur(rabi_period)
    uwave_pi_on_2_pulse = tool_belt.get_pi_on_2_pulse_dur(rabi_period)
    
    phases_yx = [0, pi/2, 0]
    phases_xy = [0, 0, pi/2]
    
    ### 1
    pulse_1_dur = uwave_pi_pulse
    pulse_2_dur = uwave_pi_on_2_pulse
    pulse_3_dur = 0
    
    iq_phases = phases_yx
    pulse_error, pulse_error_ste = measure(cxn, 
                nv_sig,
                uwave_pi_pulse,
                num_uwave_pulses,
                iq_phases,
                pulse_1_dur,
                pulse_2_dur,
                pulse_3_dur,
                apd_indices,
                state)
    pe_2_1 = pulse_error
    pe_2_1_err = pulse_error_ste
    print(r"2 (phi' + phi) = {:.4f} +/- {:.4f}".format(pulse_error, pulse_error_ste))
            
    ### 2
    pulse_1_dur = uwave_pi_pulse
    pulse_2_dur = uwave_pi_on_2_pulse
    pulse_3_dur = 0
    
    iq_phases = phases_xy
    pulse_error, pulse_error_ste = measure(cxn, 
                nv_sig,
                uwave_pi_pulse,
                num_uwave_pulses,
                iq_phases,
                pulse_1_dur,
                pulse_2_dur,
                pulse_3_dur,
                apd_indices,
                state)
    pe_2_2 = pulse_error
    pe_2_2_err = pulse_error_ste
    print(r"2 (chi' + chi) = {:.4f} +/- {:.4f}".format(pulse_error, pulse_error_ste))
    ### 3
    pulse_1_dur = uwave_pi_on_2_pulse
    pulse_2_dur = uwave_pi_pulse
    pulse_3_dur = 0
    
    iq_phases = phases_xy
    pulse_error, pulse_error_ste = measure(cxn, 
                nv_sig,
                uwave_pi_pulse,
                num_uwave_pulses,
                iq_phases,
                pulse_1_dur,
                pulse_2_dur,
                pulse_3_dur,
                apd_indices,
                state)
    pe_2_3 = pulse_error
    pe_2_3_err = pulse_error_ste
    print(r"-2 v_z + 2 phi' = {:.4f} +/- {:.4f}".format(pulse_error, pulse_error_ste))
    ### 4
    pulse_1_dur = uwave_pi_on_2_pulse
    pulse_2_dur = uwave_pi_pulse
    pulse_3_dur = 0
    
    iq_phases = phases_yx
    pulse_error, pulse_error_ste = measure(cxn, 
                nv_sig,
                uwave_pi_pulse,
                num_uwave_pulses,
                iq_phases,
                pulse_1_dur,
                pulse_2_dur,
                pulse_3_dur,
                apd_indices,
                state)
    pe_2_4 = pulse_error
    pe_2_4_err = pulse_error_ste
    print(r"2 e_z + 2 chi' = {:.4f} +/- {:.4f}".format(pulse_error, pulse_error_ste))
    ### 5
    pulse_1_dur = uwave_pi_on_2_pulse
    pulse_2_dur = uwave_pi_on_2_pulse
    pulse_3_dur = 0
    
    iq_phases = phases_xy
    pulse_error, pulse_error_ste = measure(cxn, 
                nv_sig,
                uwave_pi_pulse,
                num_uwave_pulses,
                iq_phases,
                pulse_1_dur,
                pulse_2_dur,
                pulse_3_dur,
                apd_indices,
                state)
    pe_2_5 = pulse_error
    pe_2_5_err = pulse_error_ste
    print(r"-e_y' - e_z' - v_x' - v_z' = {:.4f} +/- {:.4f}".format(pulse_error, pulse_error_ste))
    ### 6
    pulse_1_dur = uwave_pi_on_2_pulse
    pulse_2_dur = uwave_pi_on_2_pulse
    pulse_3_dur = 0
    
    iq_phases = phases_yx
    pulse_error, pulse_error_ste = measure(cxn, 
                nv_sig,
                uwave_pi_pulse,
                num_uwave_pulses,
                iq_phases,
                pulse_1_dur,
                pulse_2_dur,
                pulse_3_dur,
                apd_indices,
                state)
    pe_2_6 = pulse_error
    pe_2_6_err = pulse_error_ste
    print(r"-e_y' + e_z' - v_x' + v_z' = {:.4f} +/- {:.4f}".format(pulse_error, pulse_error_ste))
    
    ret_vals = pe_2_1, pe_2_1_err, \
                pe_2_2, pe_2_2_err, \
                pe_2_3, pe_2_3_err, \
                pe_2_4, pe_2_4_err, \
                pe_2_5, pe_2_5_err, \
                pe_2_6, pe_2_6_err,
    return ret_vals

def test_3_pulse(cxn, 
                 nv_sig,
                 apd_indices,
                 state=States.HIGH,
                 ):
    '''
        pi/2_y - pi_x - pi/2_x
        pi/2_x - pi_x - pi/2_y
        pi/2_y - pi_y - pi/2_x
        pi/2_x - pi_y - pi/2_y
    '''
    num_uwave_pulses = 3
    
    rabi_period = nv_sig["rabi_{}".format(state.name)]

    # Get pulse frequencies
    uwave_pi_pulse = tool_belt.get_pi_pulse_dur(rabi_period)
    uwave_pi_on_2_pulse = tool_belt.get_pi_on_2_pulse_dur(rabi_period)
    
    
    ### 1
    pulse_1_dur = uwave_pi_on_2_pulse
    pulse_2_dur = uwave_pi_pulse
    pulse_3_dur = uwave_pi_on_2_pulse
    
    iq_phases = [0, pi/2, 0, 0]
    pulse_error, pulse_error_ste = measure(cxn, 
                nv_sig,
                uwave_pi_pulse,
                num_uwave_pulses,
                iq_phases,
                pulse_1_dur,
                pulse_2_dur,
                pulse_3_dur,
                apd_indices,
                state)
    pe_3_1 = pulse_error
    pe_3_1_err = pulse_error_ste
    print(r"-e_y' + e_z' + v_x' - v_z' + 2e_y  = {:.4f} +/- {:.4f}".format(pulse_error, pulse_error_ste))
    ### 2
    pulse_1_dur = uwave_pi_on_2_pulse
    pulse_2_dur = uwave_pi_pulse
    pulse_3_dur = uwave_pi_on_2_pulse
    
    iq_phases = [0, 0, 0, pi/2]
    pulse_error, pulse_error_ste = measure(cxn, 
                nv_sig,
                uwave_pi_pulse,
                num_uwave_pulses,
                iq_phases,
                pulse_1_dur,
                pulse_2_dur,
                pulse_3_dur,
                apd_indices,
                state)
    pe_3_2 = pulse_error
    pe_3_2_err = pulse_error_ste
    print(r"-e_y' - e_z' + v_x' + v_z' + 2e_y  = {:.4f} +/- {:.4f}".format(pulse_error, pulse_error_ste))
    ### 3
    pulse_1_dur = uwave_pi_on_2_pulse
    pulse_2_dur = uwave_pi_pulse
    pulse_3_dur = uwave_pi_on_2_pulse
    
    iq_phases = [0,  pi/2, pi/2, 0]
    pulse_error, pulse_error_ste = measure(cxn, 
                nv_sig,
                uwave_pi_pulse,
                num_uwave_pulses,
                iq_phases,
                pulse_1_dur,
                pulse_2_dur,
                pulse_3_dur,
                apd_indices,
                state)
    pe_3_3 = pulse_error
    pe_3_3_err = pulse_error_ste
    print(r"e_y' - e_z' - v_x' + v_z' + 2v_x  = {:.4f} +/- {:.4f}".format(pulse_error, pulse_error_ste))
    ### 4
    pulse_1_dur = uwave_pi_on_2_pulse
    pulse_2_dur = uwave_pi_pulse
    pulse_3_dur = uwave_pi_on_2_pulse
    
    iq_phases = [0,  0, pi/2, pi/2]
    pulse_error, pulse_error_ste = measure(cxn, 
                nv_sig,
                uwave_pi_pulse,
                num_uwave_pulses,
                iq_phases,
                pulse_1_dur,
                pulse_2_dur,
                pulse_3_dur,
                apd_indices,
                state)
    pe_3_4 = pulse_error
    pe_3_4_err = pulse_error_ste
    print(r"e_y' + e_z' - v_x' - v_z' + 2v_x  = {:.4f} +/- {:.4f}".format(pulse_error, pulse_error_ste))
    
    ret_vals =  pe_3_1, pe_3_1_err, \
                pe_3_2, pe_3_2_err, \
                pe_3_3, pe_3_3_err, \
                pe_3_4, pe_3_4_err
    return ret_vals

def full_test(cxn, 
              nv_sig,
             apd_indices,
             state=States.HIGH,):
    
    pe1, pe1e, pe2, pe2e = test_1_pulse(cxn, 
                    nv_sig,
                    apd_indices,
                    state)
    
    ret_vals = test_2_pulse(cxn, 
                    nv_sig,
                    apd_indices,
                    state)
    pe3, pe3e, pe4, pe4e, pe5, pe5e, pe6, pe6e, pe7, pe7e, pe8, pe8e = ret_vals
    
    ret_vals=test_3_pulse(cxn, 
                    nv_sig,
                    apd_indices,
                    state)
    
    pe9, pe9e, pe10, pe10e, pe11, pe11e, pe12, pe12e = ret_vals
    
    print([pe1, pe2, pe3, pe4, pe5, pe6, pe7, pe8, pe9, pe10,
           pe11, pe12])
    
    return

# %%
if __name__ == "__main__":
    sample_name = "rubin"
    green_power = 8000
    nd_green = "nd_0.4"
    green_laser = 'integrated_520'
    yellow_laser = "laserglow_589"
    red_laser = "cobolt_638"
    
    apd_indices = [1]
    
    nv_sig = { 
            "coords":[-0.853, -0.593, 6.16],
        "name": "{}-nv1".format(sample_name,),
        "disable_opt":False,
        "ramp_voltages": False,
        "expected_count_rate":12,
        "correction_collar": 0.12,
        
        
          "spin_laser":green_laser,
          "spin_laser_power": green_power,
         "spin_laser_filter": nd_green,
          "spin_readout_dur": 350,
          "spin_pol_dur": 1000.0,
        
          "imaging_laser":green_laser,
        "imaging_laser_power": green_power,
         "imaging_laser_filter": nd_green,
          "imaging_readout_dur": 1e7,
          
         "charge_readout_laser": yellow_laser,
          "charge_readout_laser_filter": "nd_0",
        

        
        "collection_filter": "715_sp+630_lp", # NV band only
        "magnet_angle": 156,
        "resonance_LOW":2.6053,
        "rabi_LOW":96.2,     
        "uwave_power_LOW": 15,   
        "resonance_HIGH":3.1345,
        "rabi_HIGH":88.9,
        "uwave_power_HIGH": 10,
    }  
    
    with labrad.connect() as cxn:
        full_test(cxn, 
                      nv_sig,
                      apd_indices,
                      state=States.HIGH)
        # test_1_pulse(cxn, 
        #                 nv_sig,
        #                 apd_indices,
        #                 States.HIGH)
        