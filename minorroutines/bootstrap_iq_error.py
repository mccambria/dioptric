# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 15:17:35 2022

based off this paper: https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.105.077601

The signal, if no errors, should be half the contrast, which we will define as 0

@author: kolkowitz
"""


import utils.tool_belt as tool_belt
import utils.kplotlib as kpl
from utils.kplotlib import KplColors
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
import copy


def lin_line(x,a):
    return x*a

def measurement(cxn, 
            nv_sig,
            uwave_pi_pulse,
            num_uwave_pulses,
            iq_phases,
            pulse_1_dur,
            pulse_2_dur,
            pulse_3_dur,
            num_runs,
            num_reps,
            state=States.HIGH,
            do_plot = False,
            title = None,
            do_dq = False,
            inter_pulse_time = 175):
    '''
    The basic building block to perform these measurements. Can apply 1, 2, or 3
    MW pulses, and returns counts from [ms=0, ms=+/-1, counts after mw pulses]
    
    nv_sig: dictionary
        the dictionary of the nv_sig
    uwave_pi_pulse: int
        integer value of the pi pulse to prepare NV into either +1 or -1 state
    iq_phases: list
        list of phases for the IQ modulation. First value is the phase of the 
        pi pulse used to measure counts from +/-1. In radians
    pulse_1_dur: int
        length of time for first MW pulse, either pi/2 or pi pulse
    pulse_2_dur: int
        if applicable, length of time for second MW pulse, either pi/2 or pi pulse
    pulse_3_dur: int
        if applicable, length of time for third MW pulse, either pi/2 or pi pulse
    num_runs: int
        number of runs to sum over. Will optimize before every run
    state: state value
        the state (and thus signal generator) to run MW through (needs IQ mod capabilities)
    do_plot: True/False
        If True, will plot the population calculated from measurement, for each run
    title: string
        if do_plot, provide a title for the plot
    num_reps: int
        number of times the measurement is repeated for each run. These values are summed together
    inter_puls_time: int
        duration of time between MW pulses. see lab notebook 11/18/2022
        
        
    RETURNS
    ref_0_sum: int
        NV prepared in ms = 0, sum of all counts collected
    ref_H_sum: int
        NV prepared in ms = +/-1, sum of all counts collected
    sig_sum: int
        NV after MW pulses applied, sum of all counts collected
    ref_0_ste: float
        NV prepared in ms = 0, sum of stndard error measured for each run
    ref_H_ste: float
        NV prepared in ms = +/-1, sum of stndard error measured for each run
    sig_ste: float
        NV after MW pulses applied, sum of stndard error measured for each run
    '''
    
    tool_belt.reset_cfm(cxn)
    kpl.init_kplotlib()
    counter_server = tool_belt.get_server_counter(cxn)
    pulsegen_server = tool_belt.get_server_pulse_gen(cxn)
    arbwavegen_server = tool_belt.get_server_arb_wave_gen(cxn)
    seq_file = 'test_iq_pulse_errors.py'
    
    #  Sequence setup
    laser_key = "spin_laser"
    laser_name = nv_sig[laser_key]
    tool_belt.set_filter(cxn, nv_sig, laser_key)
    laser_power = tool_belt.set_laser_power(cxn, nv_sig, laser_key)
    polarization_time = nv_sig["spin_pol_dur"]
    gate_time = nv_sig["spin_readout_dur"]

    ref_0_list = []
    ref_H_list = []
    sig_list = []
    
    ref_0_ste_list = []
    ref_H_ste_list = []
    sig_ste_list = []
    
    for n in range(num_runs):
        print(n)
        optimize.main_with_cxn(cxn, nv_sig)
        # Turn on the microwaves for determining microwave delay
        if do_dq:
            sig_gen_low_cxn = tool_belt.get_server_sig_gen(cxn, States.LOW)
            sig_gen_low_cxn.set_freq(nv_sig["resonance_{}".format(States.LOW.name)])
            sig_gen_low_cxn.set_amp(nv_sig["uwave_power_{}".format(States.LOW.name)])
            sig_gen_low_cxn.uwave_on()
            sig_gen_high_cxn = tool_belt.get_server_sig_gen(cxn, States.HIGH)
            sig_gen_high_cxn.set_freq(nv_sig["resonance_{}".format(States.HIGH.name)])
            sig_gen_high_cxn.set_amp( nv_sig["uwave_power_{}".format(States.HIGH.name)])
            sig_gen_high_cxn.load_iq()
            sig_gen_high_cxn.uwave_on()
        else:
            sig_gen_cxn = tool_belt.get_server_sig_gen(cxn, state)
            sig_gen_cxn.set_freq(nv_sig["resonance_{}".format(state.name)])
            sig_gen_cxn.set_amp(nv_sig["uwave_power_{}".format(state.name)])
            sig_gen_cxn.load_iq()
            sig_gen_cxn.uwave_on()
        arbwavegen_server.load_arb_phases(iq_phases)
    
        counter_server.start_tag_stream()
            
        if  do_dq:
            seq_args = []
        else:
            seq_args = [gate_time, uwave_pi_pulse, 
                    pulse_1_dur, pulse_2_dur, pulse_3_dur, 
                    polarization_time, inter_pulse_time, num_uwave_pulses, state.value,  laser_name, laser_power]
        # print(seq_args)
        # return
        counter_server.clear_buffer()
        seq_args_string = tool_belt.encode_seq_args(seq_args)
        pulsegen_server.stream_immediate(
            seq_file, num_reps, seq_args_string
        )
    
        new_counts = counter_server.read_counter_separate_gates(1)
        sample_counts = new_counts[0]
        if len(sample_counts) != 3 * num_reps:
            print("Error!")
        # first are the counts after polarization into ms = 0
        ref_0_counts = sample_counts[0::3] 
        # second are the counts after a pi_x into +/-1
        ref_H_counts = sample_counts[1::3]
        # third are the counts after the uwave sequence
        sig_counts = sample_counts[2::3]
    
        counter_server.stop_tag_stream()
        
        tool_belt.reset_cfm(cxn)
        
        
        # analysis
        
        # ref_0_sum = sum(ref_0_counts)
        # ref_H_sum = sum(ref_H_counts)
        # sig_sum =sum(sig_counts)
        
        ref_0_avg = numpy.average(ref_0_counts)
        ref_H_avg = numpy.average(ref_H_counts)
        sig_avg = numpy.average(sig_counts)
        
        ref_0_ste = numpy.std(ref_0_counts)/numpy.sqrt(num_reps)
        ref_H_ste = numpy.std(ref_H_counts)/numpy.sqrt(num_reps)
        sig_ste = numpy.std(sig_counts)/numpy.sqrt(num_reps) 
        
        
        
        ref_0_list.append(ref_0_avg)
        ref_H_list.append(ref_H_avg)
        sig_list.append(sig_avg)
        
        ref_0_ste_list.append(ref_0_ste)
        ref_H_ste_list.append(ref_H_ste)
        sig_ste_list.append(sig_ste)
        
    
    ref_0_avg_avg = numpy.average(ref_0_list) 
    ref_H_avg_avg = numpy.average(ref_H_list)
    sig_avg_avg = numpy.average(sig_list)
    
    ref_0_ste_avg = numpy.average(ref_0_ste_list) / numpy.sqrt(num_runs)
    ref_H_ste_avg = numpy.average(ref_H_ste_list) / numpy.sqrt(num_runs)
    sig_ste_avg = numpy.average(sig_ste_list) / numpy.sqrt(num_runs)
    
    # print(ref_0_avg_avg, ref_0_ste_avg)
    # ref_0_ste = numpy.std(
    #     ref_0_list, ddof=1
    #     ) / numpy.sqrt(num_runs)
    # ref_H_ste = numpy.std(
    #     ref_H_list, ddof=1
    #     ) / numpy.sqrt(num_runs)
    # sig_ste = numpy.std(
    #     sig_list, ddof=1
    #     ) / numpy.sqrt(num_runs)
    
    
    
    N = (numpy.array(sig_list)-numpy.array(ref_H_list))
    D = (numpy.array(ref_0_list)-numpy.array(ref_H_list))
    N_unc = numpy.sqrt( numpy.array(sig_ste_list)**2 + numpy.array(ref_H_ste_list)**2)
    D_unc = numpy.sqrt( numpy.array(ref_0_ste_list)**2 + numpy.array(ref_H_ste_list)**2)
    
    pop = N/D
    pop_unc = pop * numpy.sqrt( (N_unc/N)**2 + (D_unc/D)**2)
    
    if do_plot:
        fig, ax = plt.subplots()
        kpl.plot_points(ax, range(num_runs), pop, yerr=pop_unc, color = KplColors.RED)
        # ax.error_bar(range(num_runs), pop,y_err=pop_unc, "ro")
        ax.set_xlabel(r"Num repitition")
        ax.set_ylabel("Population")
        if title:
            ax.set_title(title)


    timestamp = tool_belt.get_time_stamp()

    raw_data = {'timestamp': timestamp,
                'nv_sig': nv_sig,
                'num_uwave_pulses': num_uwave_pulses,
                'iq_phases': iq_phases,
                'pulse_durations': [pulse_1_dur, pulse_2_dur, pulse_3_dur],
                'state': state.name,
                'num_reps': num_reps,
                'num_runs': num_runs,
                'inter_pulse_time':inter_pulse_time,
                'population': pop.tolist(),
                'ref_0_avg_avg': ref_0_avg_avg,
                'ref_H_avg_avg': ref_H_avg_avg,
                'sig_avg_avg': sig_avg_avg,
                'ref_0_ste_avg': ref_0_ste_avg,
                'ref_H_ste_avg': ref_H_ste_avg,
                'sig_ste_avg': sig_ste_avg,
                'ref_0_list': ref_0_list,
                'ref_H_list': ref_H_list,
                'sig_list': sig_list,
                'ref_0_ste_list': ref_0_ste_list,
                'ref_H_ste_list': ref_H_ste_list,
                'sig_ste_list': sig_ste_list,}

    nv_name = nv_sig["name"]
    file_path = tool_belt.get_file_path(__file__, timestamp, nv_name)    
    if do_plot:
        tool_belt.save_figure(fig, file_path)
    tool_belt.save_raw_data(raw_data, file_path)
    # print(ref_0_sum, ref_H_sum, sig_sum)
        
    return ref_0_avg_avg, ref_H_avg_avg, sig_avg_avg, ref_0_ste_avg, ref_H_ste_avg, sig_ste_avg

def measure_pulse_error(cxn, 
            nv_sig,
            uwave_pi_pulse,
            num_uwave_pulses,
            iq_phases,
            pulse_1_dur,
            pulse_2_dur,
            pulse_3_dur,
            num_runs,
            num_reps,
            state=States.HIGH,
            do_plot = False,
            Title = None,):
    '''
    Outer shell function to run a measurement, and then to calculate the signal 
    from the measured values.
    
    From the measurement, we get the counts in ms=0, counts in ms=+/-1, and the
    signal. The signal we want is the expectation value, which ranges from -1 <--> +1,
    where we are measuring the population. We need to normalize the population, and then
    double the value.
    
    nv_sig: dictionary
        the dictionary of the nv_sig
    uwave_pi_pulse: int
        integer value of the pi pulse to prepare NV into either +1 or -1 state
    iq_phases: list
        list of phases for the IQ modulation. First value is the phase of the 
        pi pulse used to measure counts from +/-1. In radians
    pulse_1_dur: int
        length of time for first MW pulse, either pi/2 or pi pulse
    pulse_2_dur: int
        if applicable, length of time for second MW pulse, either pi/2 or pi pulse
    pulse_3_dur: int
        if applicable, length of time for third MW pulse, either pi/2 or pi pulse
    state: state value
        the state (and thus signal generator) to run MW through (needs IQ mod capabilities)
    do_plot: True/False
        If True, will plot the population calculated from measurement, for each run
    Title: string
        if do_plot, provide a title for the plot
    '''
    
    ret_vals = measurement(cxn, 
                nv_sig,
                uwave_pi_pulse,
                num_uwave_pulses,
                iq_phases,
                pulse_1_dur,
                pulse_2_dur,
                pulse_3_dur,
                num_runs,
                num_reps,
                state = States.HIGH,
                do_plot = do_plot,
                title = Title)
    
    ref_0_avg, ref_H_avg, sig_avg, ref_0_ste, ref_H_ste, sig_ste = ret_vals
    # print(ref_0_avg, ref_H_avg, sig_avg)
    # print(ref_0_ste, ref_H_ste, sig_ste)
    
    
    contrast = ref_0_avg-ref_H_avg
    contrast_ste = numpy.sqrt(ref_0_ste**2 + ref_H_ste**2)
    signal_m_H = sig_avg-ref_H_avg
    signal_m_H_ste = numpy.sqrt(sig_ste**2 + ref_H_ste**2)
    half_contrast = 0.5
    
    signal_perc = signal_m_H / contrast
    signal_perc_ste = signal_perc*numpy.sqrt((contrast_ste/contrast)**2 + \
                                             (signal_m_H_ste/signal_m_H)**2)
    
    #Subtract 0.5, so that the expectation value is centered at 0, and
    # multiply by 2 so that we report the expectation value from -1 to +1.
    pulse_error = (signal_perc - half_contrast) *2
    pulse_error_ste = signal_perc_ste * 2
    
    return pulse_error, pulse_error_ste

def solve_errors(meas_list):
    '''
    Given a list of the measured signals from the bootstrapping method,
    calculate the pulse errors.
    
    This follows the order of measurements stated in 
    https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.105.077601
    
    Like in the paper, we set the error of the pi2_x error along the y axis as 0
    
    '''
    A1 = meas_list[0]
    B1 = meas_list[1]
    
    A2 = meas_list[2]
    B2= meas_list[3]
    
    A3 = meas_list[4]
    B3 = meas_list[5]
    
    S = meas_list[6:11]
    
    
    phi_p = -A1/2
    chi_p = -B1/2
    
    phi = A2/2 - phi_p
    chi = B2/2 - chi_p
    
    v_z = - (A3 - 2*phi_p)/2
    e_z = (B3 - 2*chi_p)/2
    
    M = numpy.array([
            [-1,-1,-1,0,0],
            [1,-1,1,0,0],
            [1,1,-1,2,0],
            [-1,1,1,2,0],
            [-1,-1,1,0,2],
            # [1,-1,-1,0,2] #exclude last equation
          ])
    X = numpy.linalg.inv(M).dot(S)
    
    e_y_p = 0 # we are setting this value to 0
    e_z_p = X[0]
    v_x_p = X[1]
    v_z_p = X[2]
    e_y = X[3]
    v_x = X[4]
    
    return [phi_p, chi_p, phi, chi, v_z, e_z,  e_y_p, e_z_p, v_x_p,v_z_p,  e_y, v_x]

def calc_pulse_error_ste(ste_list):
    '''
    Given a list of the ste's of the measured signals, calculate the ste of the
    pulse errors
    
    '''
    A1 = ste_list[0]
    B1 = ste_list[1]
    
    A2 = ste_list[2]
    B2= ste_list[3]
    
    A3 = ste_list[4]
    B3 = ste_list[5]
    
    S = ste_list[6:12]
    
    
    phi_p_ste = A1/2
    chi_p_ste = B1/2
    
    phi_ste = numpy.sqrt((A2/2)**2 + (phi_p_ste)**2 )
    chi_ste = numpy.sqrt((B2/2)**2 + (chi_p_ste)**2 )
    
    v_z_ste = numpy.sqrt((A3/2)**2 + (phi_p_ste)**2 )
    e_z_ste = numpy.sqrt((B3/2)**2 + (chi_p_ste)**2 )
    
    # I don't know a goo way to propegate the uncertainty through a system
    # of linear equations so instead, I'm going to try to give a best guess for the uncertainty
    
    # first, I take the average value of the uncert from measurements 7 and 8, and then 9-12
    # because the the respective two measurements are identical for extracting uncertainty:
    # ei: the 7th and 8th measurements result in: DS78 = Sqrt(Dey'^2 + Dez'^2 + Dvx'^2 + Dvz'^2)
    # so the ste from those measurements "should" be the sam,e so we'll just average them.
    S78_ste = numpy.average([S[0], S[1]])
    S910_ste = numpy.average([S[2], S[3]])
    S112_ste = numpy.average([S[4], S[5]])
    
    
    e_y_p_ste = 0 # we are setting this value to 0
    
    #then, the equations for uncert: DS78 = sqrt(Dey'^2 + Dez'^2 + Dvx'^2 + Dvz'^2)
    # we set Dey' == 0, so then, and we will weight the other uncert equally,
    # so i.e.: Dez' = Sqrt(DS7**2 / 3)
    e_z_p_ste = numpy.sqrt(S78_ste**2 / 3)
    v_x_p_ste = numpy.sqrt(S78_ste**2 / 3)
    v_z_p_ste = numpy.sqrt(S78_ste**2 / 3)
    # Similarly, for the final two variables, they are in the equations like:
    # DS910 = sqrt(Dey'^2 + Dez'^2 + Dvx'^2 + Dvz'^2 + (2Dvx)^2)
    # DS910 = sqrt(DS78^2 + (2Dvx)^2)
    # and to get Dvx,
    # Dvx = Sqrt((DS910^2 - DS78^2) / 2 )
    #   note that I take the absolute difference to avoid imaginary values
    e_y_ste = numpy.sqrt(abs(S910_ste**2 - S78_ste**2) / 2)
    v_x_ste = numpy.sqrt(abs(S112_ste**2 - S78_ste**2) / 2)
    
    return [phi_p_ste, chi_p_ste, phi_ste, chi_ste, v_z_ste, e_z_ste,  e_y_p_ste,
                        e_z_p_ste, v_x_p_ste,v_z_p_ste,  e_y_ste, v_x_ste]
def test_1_pulse(cxn, 
                 nv_sig,
                 num_runs,
                 num_reps,
                 state=States.HIGH,
                 int_phase = 0,
                 plot = False,):
    '''
    This pulse sequence consists of pi/2 pulses:
        1: pi/2_x
        2: pi/2_y
        
    You can intentionally adjust the phase of the pi/2_y pulse by inputting a value for
    int_phase
    '''

    num_uwave_pulses = 1
    
    # Get pulse frequencies
    uwave_pi_pulse = nv_sig["pi_pulse_{}".format(state.name)]
    uwave_pi_on_2_pulse = nv_sig["pi_on_2_pulse_{}".format(state.name)]
    
    pulse_1_dur = uwave_pi_on_2_pulse
    pulse_2_dur = 0
    pulse_3_dur = 0
    
    
    pi_x_phase = 0
    pi_2_x_phase = 0 
    pi_y_phase = pi/2+ int_phase
    pi_2_y_phase = pi/2
    
    ##### 1
    iq_phases = [0, pi_2_x_phase]
    pulse_error, pulse_error_ste = measure_pulse_error(cxn, 
                nv_sig,
                uwave_pi_pulse,
                num_uwave_pulses,
                iq_phases,
                pulse_1_dur,
                pulse_2_dur,
                pulse_3_dur,
                num_runs,
                num_reps,
                state=States.HIGH,
                do_plot = plot,
                Title = 'pi/2_x',)
    pe_1_1 = pulse_error
    pe_1_1_err = pulse_error_ste
    
    #### 2
    iq_phases = [0, pi_2_y_phase]
    pulse_error, pulse_error_ste = measure_pulse_error(cxn, 
                nv_sig,
                uwave_pi_pulse,
                num_uwave_pulses,
                iq_phases,
                pulse_1_dur,
                pulse_2_dur,
                pulse_3_dur,
                num_runs,
                num_reps,
                state=States.HIGH,
                do_plot = plot,
                Title = 'pi/2_y',)
    pe_1_2 = pulse_error
    pe_1_2_err = pulse_error_ste
    
    
    print(r"pi/2_x rotation angle error, -2 phi' = {:.4f} +/- {:.4f}".format(pe_1_1, pe_1_1_err))
    
    print(r"pi/2_y rotation angle error, -2 chi' = {:.4f} +/- {:.4f}".format(pe_1_2, pe_1_2_err))
    return pe_1_1, pe_1_1_err, pe_1_2, pe_1_2_err

def test_2_pulse(cxn, 
                 nv_sig,
                 num_runs,
                 num_reps,
                 state=States.HIGH,
                 int_phase = 0,
                 plot = False,
                 ):
    '''
    These are the measurements that conatin two MW pulses
        1: pi_y - pi/2_x
        2: pi_x - pi/2_y
        3: pi/2_x - pi_y
        4: pi/2_y - pi_x
        5: pi/2_x - pi/2_y
        6: pi/2_y - pi/2_x
        
    You can intentionally adjust the phase of the pi/2_y pulse by inputting a value for
    int_phase
    '''
    
    num_uwave_pulses = 2

    # Get pulse frequencies
    uwave_pi_pulse = nv_sig["pi_pulse_{}".format(state.name)]
    uwave_pi_on_2_pulse = nv_sig["pi_on_2_pulse_{}".format(state.name)]
    
    pi_x_phase = 0
    pi_2_x_phase = 0 
    pi_y_phase = pi/2 + int_phase
    pi_2_y_phase = pi/2
    
    ### 1
    pulse_1_dur = uwave_pi_pulse
    pulse_2_dur = uwave_pi_on_2_pulse
    pulse_3_dur = 0
    
    iq_phases = [0, pi_y_phase, pi_2_x_phase]
    pulse_error, pulse_error_ste = measure_pulse_error(cxn, 
                nv_sig,
                uwave_pi_pulse,
                num_uwave_pulses,
                iq_phases,
                pulse_1_dur,
                pulse_2_dur,
                pulse_3_dur,
                num_runs,
                num_reps,
                state,
                do_plot = plot,
                Title = 'pi_y - pi/2_x',)
    pe_2_1 = pulse_error
    pe_2_1_err = pulse_error_ste
            
    ### 2
    pulse_1_dur = uwave_pi_pulse
    pulse_2_dur = uwave_pi_on_2_pulse
    pulse_3_dur = 0
    
    iq_phases = [0, pi_x_phase, pi_2_y_phase]
    pulse_error, pulse_error_ste = measure_pulse_error(cxn, 
                nv_sig,
                uwave_pi_pulse,
                num_uwave_pulses,
                iq_phases,
                pulse_1_dur,
                pulse_2_dur,
                pulse_3_dur,
                num_runs,
                num_reps,
                state,
                do_plot = plot,
                Title='pi_x - pi/2_y',)
    pe_2_2 = pulse_error
    pe_2_2_err = pulse_error_ste
    ### 3
    pulse_1_dur = uwave_pi_on_2_pulse
    pulse_2_dur = uwave_pi_pulse
    pulse_3_dur = 0
    
    iq_phases = [0,pi_2_x_phase, pi_y_phase ]
    pulse_error, pulse_error_ste = measure_pulse_error(cxn, 
                nv_sig,
                uwave_pi_pulse,
                num_uwave_pulses,
                iq_phases,
                pulse_1_dur,
                pulse_2_dur,
                pulse_3_dur,
                num_runs,
                num_reps,
                state,
                do_plot = plot,
                Title = 'pi/2_x - pi_y',)
    pe_2_3 = pulse_error
    pe_2_3_err = pulse_error_ste
    ### 4
    pulse_1_dur = uwave_pi_on_2_pulse
    pulse_2_dur = uwave_pi_pulse
    pulse_3_dur = 0
    
    iq_phases = [0,  pi_2_y_phase, pi_x_phase]
    pulse_error, pulse_error_ste = measure_pulse_error(cxn, 
                nv_sig,
                uwave_pi_pulse,
                num_uwave_pulses,
                iq_phases,
                pulse_1_dur,
                pulse_2_dur,
                pulse_3_dur,
                num_runs,
                num_reps,
                state,
                do_plot = plot,
                Title = 'pi/2_y - pi_x',)
    pe_2_4 = pulse_error
    pe_2_4_err = pulse_error_ste
    ### 5
    pulse_1_dur = uwave_pi_on_2_pulse
    pulse_2_dur = uwave_pi_on_2_pulse
    pulse_3_dur = 0
    
    iq_phases = [0,  pi_2_x_phase, pi_2_y_phase]
    pulse_error, pulse_error_ste = measure_pulse_error(cxn, 
                nv_sig,
                uwave_pi_pulse,
                num_uwave_pulses,
                iq_phases,
                pulse_1_dur,
                pulse_2_dur,
                pulse_3_dur,
                num_runs,
                num_reps,
                state,
                do_plot = plot,
                Title = 'pi/2_x - pi/2_y',)
    pe_2_5 = pulse_error
    pe_2_5_err = pulse_error_ste
    ### 6
    pulse_1_dur = uwave_pi_on_2_pulse
    pulse_2_dur = uwave_pi_on_2_pulse
    pulse_3_dur = 0
    
    iq_phases = [0,  pi_2_y_phase, pi_2_x_phase]
    pulse_error, pulse_error_ste = measure_pulse_error(cxn, 
                nv_sig,
                uwave_pi_pulse,
                num_uwave_pulses,
                iq_phases,
                pulse_1_dur,
                pulse_2_dur,
                pulse_3_dur,
                num_runs,
                num_reps,
                state,
                do_plot = plot,
                Title = 'pi/2_y - pi/2_x',)
    pe_2_6 = pulse_error
    pe_2_6_err = pulse_error_ste
    print(r"2 (phi' + phi) = {:.4f} +/- {:.4f}".format(pe_2_1, pe_2_1_err))
    print(r"2 (chi' + chi) = {:.4f} +/- {:.4f}".format(pe_2_2, pe_2_2_err))
    print(r"-2 v_z + 2 phi' = {:.4f} +/- {:.4f}".format(pe_2_3, pe_2_3_err))
    print(r"2 e_z + 2 chi' = {:.4f} +/- {:.4f}".format(pe_2_4, pe_2_4_err))
    print(r"-e_y' - e_z' - v_x' - v_z' = {:.4f} +/- {:.4f}".format(pe_2_5, pe_2_5_err))
    print(r"-e_y' + e_z' - v_x' + v_z' = {:.4f} +/- {:.4f}".format(pe_2_6, pe_2_6_err))
    
    ret_vals = pe_2_1, pe_2_1_err, \
                pe_2_2, pe_2_2_err, \
                pe_2_3, pe_2_3_err, \
                pe_2_4, pe_2_4_err, \
                pe_2_5, pe_2_5_err, \
                pe_2_6, pe_2_6_err,
    return ret_vals

def test_3_pulse(cxn, 
                 nv_sig,
                 num_runs,
                 num_reps,
                 state=States.HIGH,
                 int_phase = 0,
                 plot = False,
                 ):
    '''
    These are the measurements that conatin two MW pulses
        pi/2_y - pi_x - pi/2_x
        pi/2_x - pi_x - pi/2_y
        pi/2_y - pi_y - pi/2_x
        pi/2_x - pi_y - pi/2_y
        
    You can intentionally adjust the phase of the pi/2_y pulse by inputting a value for
    int_phase
    '''
    num_uwave_pulses = 3
    

    # Get pulse frequencies
    uwave_pi_pulse = nv_sig["pi_pulse_{}".format(state.name)]
    uwave_pi_on_2_pulse = nv_sig["pi_on_2_pulse_{}".format(state.name)]
    
    
    ### 1
    pulse_1_dur = uwave_pi_on_2_pulse
    pulse_2_dur = uwave_pi_pulse
    pulse_3_dur = uwave_pi_on_2_pulse
    
    
    pi_x_phase = 0
    pi_2_x_phase = 0 
    pi_y_phase = pi/2 + int_phase
    pi_2_y_phase = pi/2
    
    
    iq_phases = [0, pi_2_y_phase, pi_x_phase, pi_2_x_phase]
    pulse_error, pulse_error_ste = measure_pulse_error(cxn, 
                nv_sig,
                uwave_pi_pulse,
                num_uwave_pulses,
                iq_phases,
                pulse_1_dur,
                pulse_2_dur,
                pulse_3_dur,
                num_runs,
                num_reps,
                state,
                do_plot = plot,
                Title = 'pi/2_y - pi_x - pi/2_x')
    pe_3_1 = pulse_error
    pe_3_1_err = pulse_error_ste
    ### 2
    pulse_1_dur = uwave_pi_on_2_pulse
    pulse_2_dur = uwave_pi_pulse
    pulse_3_dur = uwave_pi_on_2_pulse
    
    iq_phases = [0, pi_2_x_phase, pi_x_phase, pi_2_y_phase]
    pulse_error, pulse_error_ste = measure_pulse_error(cxn, 
                nv_sig,
                uwave_pi_pulse,
                num_uwave_pulses,
                iq_phases,
                pulse_1_dur,
                pulse_2_dur,
                pulse_3_dur,
                num_runs,
                num_reps,
                state,
                do_plot = plot,
                Title = 'pi/2_x - pi_x - pi/2_y',)
    pe_3_2 = pulse_error
    pe_3_2_err = pulse_error_ste
    ### 3
    pulse_1_dur = uwave_pi_on_2_pulse
    pulse_2_dur = uwave_pi_pulse
    pulse_3_dur = uwave_pi_on_2_pulse
    
    iq_phases = [0, pi_2_y_phase, pi_y_phase, pi_2_x_phase]
    pulse_error, pulse_error_ste = measure_pulse_error(cxn, 
                nv_sig,
                uwave_pi_pulse,
                num_uwave_pulses,
                iq_phases,
                pulse_1_dur,
                pulse_2_dur,
                pulse_3_dur,
                num_runs,
                num_reps,
                state,
                do_plot = plot,
                Title = 'pi/2_y - pi_y - pi/2_x')
    pe_3_3 = pulse_error
    pe_3_3_err = pulse_error_ste
    ### 4
    pulse_1_dur = uwave_pi_on_2_pulse
    pulse_2_dur = uwave_pi_pulse
    pulse_3_dur = uwave_pi_on_2_pulse
    
    iq_phases =  [0, pi_2_x_phase, pi_y_phase, pi_2_y_phase]
    pulse_error, pulse_error_ste = measure_pulse_error(cxn, 
                nv_sig,
                uwave_pi_pulse,
                num_uwave_pulses,
                iq_phases,
                pulse_1_dur,
                pulse_2_dur,
                pulse_3_dur,
                num_runs,
                num_reps,
                state,
                do_plot = plot,
                Title = 'pi/2_x - pi_y - pi/2_y',
                )
    pe_3_4 = pulse_error
    pe_3_4_err = pulse_error_ste
    print(r"-e_y' + e_z' + v_x' - v_z' + 2e_y  = {:.4f} +/- {:.4f}".format(pe_3_1, pe_3_1_err))
    print(r"-e_y' - e_z' + v_x' + v_z' + 2e_y  = {:.4f} +/- {:.4f}".format(pe_3_2, pe_3_2_err))
    print(r"e_y' - e_z' - v_x' + v_z' + 2v_x  = {:.4f} +/- {:.4f}".format(pe_3_3, pe_3_3_err))
    print(r"e_y' + e_z' - v_x' - v_z' + 2v_x  = {:.4f} +/- {:.4f}".format(pe_3_4, pe_3_4_err))
    
    ret_vals =  pe_3_1, pe_3_1_err, \
                pe_3_2, pe_3_2_err, \
                pe_3_3, pe_3_3_err, \
                pe_3_4, pe_3_4_err
    return ret_vals

def full_test(cxn, 
              nv_sig,
             num_runs,
             num_reps,
             state=States.HIGH,
             int_phase = 0,
             plot = False,
             ret_ste = False,):
    '''
    This function runs all 12 measurements, and returns the measured signals
    '''
    pe1, pe1e, pe2, pe2e = test_1_pulse(cxn, 
                    nv_sig,
                    num_runs,
                    num_reps,
                    state,
                    int_phase,
                    plot)
    
    ret_vals = test_2_pulse(cxn, 
                    nv_sig,
                    num_runs,
                    num_reps,
                    state,
                    int_phase,
                    plot,)
    pe3, pe3e, pe4, pe4e, pe5, pe5e, pe6, pe6e, pe7, pe7e, pe8, pe8e = ret_vals
    
    ret_vals=test_3_pulse(cxn, 
                    nv_sig,
                    num_runs,
                    num_reps,
                    state,
                    int_phase,
                    plot,)
    
    pe9, pe9e, pe10, pe10e, pe11, pe11e, pe12, pe12e = ret_vals
    
    # print([pe1, pe2, pe3, pe4, pe5, pe6, pe7, pe8, pe9, pe10,
    #        pe11, pe12])
    if ret_ste == True:
        return [pe1, pe2, pe3, pe4, pe5, pe6, pe7, pe8, pe9, pe10,
                pe11, pe12], [pe1e, pe2e, pe3e, pe4e, pe5e, pe6e, pe7e, pe8e, pe9e, pe10e,
                        pe11e, pe12e]
    
    else:
        return [pe1, pe2, pe3, pe4, pe5, pe6, pe7, pe8, pe9, pe10,
               pe11, pe12]


    
def replot_imposed_phases(file):
    '''
    Replotting functionality for measurements where an intentional phase was applied
    '''
    data = tool_belt.get_raw_data(file)
    
    phases = data['phases']
    phi_p_list = data['phi_p_list']
    chi_p_list = data['chi_p_list']
    phi_list = data['phi_list']
    chi_list= data['chi_list']
    e_z_p_list = data['e_z_p_list']
    v_x_p_list = data['v_x_p_list']
    v_z_p_list = data['v_z_p_list']
    e_y_list = data['e_y_list']
    v_x_list = data['v_x_list']
    v_z_list = data['v_z_list']
    e_z_list = data['e_z_list']
    
    plot_errors_vs_changed_param(phases,
                       'Imposed phase on pi_x pulse (deg)',
                           phi_p_list,
                           chi_p_list,
                           phi_list,
                           chi_list,
                           e_z_p_list,
                           v_x_p_list,
                           v_z_p_list,
                           e_y_list,
                           v_x_list,
                           v_z_list,
                           e_z_list,
                           do_expected_phases = True
                           )
       

def plot_errors_vs_changed_param(x_vals,
                                 x_axis_label,
                       phi_p_list,
                       chi_p_list,
                       phi_list,
                       chi_list,
                       e_z_p_list,
                       v_x_p_list,
                       v_z_p_list,
                       e_y_list,
                       v_x_list,
                       v_z_list,
                       e_z_list,
                       do_expected_phases = False):

    '''
    Plotting capabilities
    '''
    if len(phi_p_list) != 0:
        fig1, ax = plt.subplots()
        ax.plot(x_vals,phi_p_list, 'ro', label = r"$\Phi'$" )
        ax.plot(x_vals,chi_p_list, 'bo', label = r"$\chi'$" )
        ax.plot(x_vals,phi_list, 'go', label = r"$\Phi$" )
        ax.plot(x_vals,chi_list, 'mo', label = r"$\chi$" )
        ax.set_xlabel(x_axis_label)
        ax.set_ylabel('Error')
        ax.legend()
    
    if len(e_z_p_list) != 0:
        fig2, ax = plt.subplots()
        ax.plot(x_vals,e_z_p_list, 'ro', label = r"$e_z'$" )
        ax.plot(x_vals,v_x_p_list, 'bo', label = r"$v_x'$" )
        ax.plot(x_vals,v_z_p_list, 'go', label = r"$v_z'$" )
        ax.plot(x_vals,e_y_list, 'mo', label = r"$e_y$" )
        ax.plot(x_vals,v_x_list, 'co', label = r"$v_x$" )
        
        if do_expected_phases:
            x_start = min(x_vals)
            x_end = max(x_vals)
            lin_x = numpy.linspace(x_start, x_end,100)
            ax.plot(lin_x, lin_line(lin_x, pi/180), 'r-', label="expected")
        
        ax.set_xlabel(x_axis_label)
        ax.set_ylabel('Error')
        ax.legend()
    
    if len(v_z_list) != 0:
        fig3, ax = plt.subplots()
        ax.plot(x_vals,v_z_list, 'ro', label = r"$v_z$" )
        ax.plot(x_vals,e_z_list, 'bo', label = r"$e_z$" )
        ax.set_xlabel(x_axis_label)
        ax.set_ylabel('Error')
        ax.legend()
        
        
def do_impose_phase(cxn, 
              nv_sig,
              num_runs,
              num_reps,
              state=States.HIGH,):
        '''
        To test that the measurements are working, we can recreate Fig 1 from
        https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.105.077601
        where an intential phase is applied to the pi/2_y pulse, and we 
        see that imtentional phase in the calculated pulse errors.
        
        '''
        phi_list = []
        chi_list = []
        phi_p_list = []
        chi_p_list = []
        
        v_z_list = []
        e_z_list = []
        
        e_y_p_list = []
        e_z_p_list = []
        v_x_p_list = []
        v_z_p_list = []
        e_y_list = []
        v_x_list = []
        
        errs_list = []
        
        phases = numpy.linspace(-30, 30, 7)
        shuffle(phases)
        for p in phases:
            phase_rad = p*pi/180
            s_list = full_test(cxn, 
                          nv_sig,
                          num_runs,
                          num_reps,
                          state=state,
                          int_phase = phase_rad,
                          plot = False)
            
            errs = solve_errors(s_list )  
            errs_list.append(errs)
            
            phi_p_list.append(errs[0])
            chi_p_list.append(errs[1])
            phi_list.append(errs[2])
            chi_list.append(errs[3])
            
            v_z_list.append(errs[4])
            e_z_list.append(errs[5])
            
            e_y_p_list.append(errs[6])
            
            e_z_p_list.append(errs[7])
            v_x_p_list.append(errs[8])
            v_z_p_list.append(errs[9])
            e_y_list.append( errs[10])
            v_x_list.append( errs[11])
        
        plot_errors_vs_changed_param(phases,
                           'Imposed phase on pi_y pulse (deg)',
                               phi_p_list,
                               chi_p_list,
                               phi_list,
                               chi_list,
                               e_z_p_list,
                               v_x_p_list,
                               v_z_p_list,
                               e_y_list,
                               v_x_list,
                               v_z_list,
                               e_z_list,
                               do_expected_phases = True
                               )
        
        
        timestamp = tool_belt.get_time_stamp()
        
        raw_data = {'timestamp': timestamp,
                    'nv_sig': nv_sig,
                    'phases': phases.tolist(),
                    'phases-units': 'degrees',
                    
                    'phi_list':phi_list,
                    'chi_list':chi_list,
                    'phi_p_list':phi_p_list,
                    'chi_p_list':chi_p_list,
                    'v_z_list':v_z_list,
                    'e_z_list':e_z_list,
                    'e_y_p_list':e_y_p_list,
                    'e_z_p_list':e_z_p_list,
                    'v_x_p_list':v_x_p_list,
                    'v_z_p_list':v_z_p_list,
                    'e_y_list':e_y_list,
                    'v_x_list':v_x_list,
                    'errs_list': errs_list}
    
        nv_name = nv_sig["name"]
        file_path = tool_belt.get_file_path(__file__, timestamp, nv_name)
        tool_belt.save_raw_data(raw_data, file_path)
        # tool_belt.save_figure(fig1, file_path)
        # tool_belt.save_figure(fig2, file_path)
        # tool_belt.save_figure(fig3, file_path)
    
def measure_pulse_errors(cxn, 
              nv_sig,
             num_runs,
             num_reps,
             state=States.HIGH):  
    '''
    Measure the pulse errors and print them out
    '''
    s_list, ste_list = full_test(cxn, 
                      nv_sig,
                      num_runs,
                      num_reps,
                      state=state,
                      ret_ste = True,)
    # print(ste_list)
    pulse_errors = solve_errors(s_list ) 
    pulse_ste = calc_pulse_error_ste(ste_list ) 
    
    phi_p, chi_p, phi, chi, v_z, e_z,  e_y_p, e_z_p, v_x_p,v_z_p,  e_y, v_x = pulse_errors
    phi_p_ste, chi_p_ste, phi_ste, chi_ste, v_z_ste, e_z_ste,  e_y_p_ste, \
            e_z_p_ste, v_x_p_ste,v_z_p_ste,  e_y_ste, v_x_ste = pulse_ste
    
    timestamp = tool_belt.get_time_stamp()
    raw_data = {'timestamp': timestamp,
                'nv_sig': nv_sig,
                'num_runs': num_runs,
                'num_reps': num_reps,
                'state': state.name,
                
                'phi':phi,
                'phi_p_ste':phi_p_ste,
                'e_y':e_y,
                'e_y_ste':e_y_ste,
                'e_z':e_z,
                'e_z_ste':e_z_ste,
                
                'chi':chi,
                'chi_ste':chi_ste,
                'v_x':v_x,
                'v_x_ste':v_x_ste,
                'v_z':v_z,
                'v_z_ste':v_z_ste,
                
                'phi_p':phi_p,
                'phi_p_ste':phi_p_ste,
                'e_y_p':e_y_p,
                'e_y_p_ste':0,
                'e_z_p':e_z_p,
                'e_z_p_ste':e_z_p_ste,
                
                'chi_p':chi_p,
                'chi_p_ste':chi_p_ste,
                'v_x_p':v_x_p,
                'v_x_p_ste':v_x_p_ste,
                'v_z_p':v_z_p,
                'v_z_p_ste':v_z_p_ste,
                
                'original signal list': pulse_errors,
                'original signal ste list': pulse_ste
                }
    nv_name = nv_sig["name"]
    file_path = tool_belt.get_file_path(__file__, timestamp, nv_name)
    tool_belt.save_raw_data(raw_data, file_path)
    
    print('')
    print('************RESULTS************')
    print("Pi_X pulse-------")
    print("Rotation angle error, Phi = {:.5f} +/- {:.5f} rad".format(phi,phi_p_ste ))
    print("Rotation axis error along y-axis, e_y = {:.5f} +/- {:.5f} rad".format(e_y, e_y_ste))
    print("Rotation axis error along z-axis, e_z = {:.5f} +/- {:.5f} rad".format(e_z, e_z_ste))
    print("Pi_Y pulse-------")
    print("Rotation angle error, Chi = {:.5f} +/- {:.5f} rad".format(chi, chi_ste))
    print("Rotation axis error along x-axis, v_x = {:.5f} +/- {:.5f} rad".format(v_x, v_x_ste))
    print("Rotation axis error along z-axis, v_z = {:.5f} +/- {:.5f} rad".format(v_z, v_z_ste))
    print('')
    print("Pi/2_X pulse-------")
    print("Rotation angle error, Phi' = {:.5f} +/- {:.5f} rad".format(phi_p, phi_p_ste))
    print("Rotation axis error along y-axis, e_y' = {:.5f} rad (intentionally set to 0)".format(e_y_p))
    print("Rotation axis error along z-axis, e_z' = {:.5f} +/- {:.5f} rad".format(e_z_p, e_z_p_ste))
    print("Pi/2_Y pulse-------")
    print("Rotation angle error, Chi' = {:.5f} +/- {:.5f} rad".format(chi_p, chi_p_ste))
    print("Rotation axis error along x-axis, v_x' = {:.5f} +/- {:.5f} rad".format(v_x_p, v_x_p_ste))
    print("Rotation axis error along z-axis, v_z' = {:.5f} +/- {:.5f} rad".format(v_z_p, v_z_p_ste))
    print('*******************************')
    
    
# %%
if __name__ == "__main__":
    sample_name = "siena"
    green_power = 8000
    nd_green = "nd_1.1"
    green_laser = 'integrated_520'
    yellow_laser = "laserglow_589"
    red_laser = "cobolt_638"
    
    
    nv_sig = { 
            "coords":[0.030, -0.302, 5.09],
        "name": "{}-nv4_2023_01_16".format(sample_name,),
        "disable_opt":False,
        "ramp_voltages": False,
        "expected_count_rate":42,
        
        
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
        "magnet_angle": 53.5,
        "resonance_LOW":2.81921,
        # "rabi_LOW":67*2,     
        "uwave_power_LOW": 15,   
        "resonance_HIGH":2.92159,
        # "rabi_HIGH":210.73,
        "uwave_power_HIGH": 10,
        
    "pi_pulse_LOW": 67,
    "pi_on_2_pulse_LOW": 33,# 37,
    "pi_pulse_HIGH": 128,
    "pi_on_2_pulse_HIGH": 59,
    }  
    
    with labrad.connect() as cxn:
        num_runs = 8
        num_reps = int(1e5)
        ### measure the phase errors, will print them out
        # measure_pulse_errors(cxn, 
        #                nv_sig,
        #               num_runs,
        #                num_reps,
        #               state=States.HIGH)
        
        # test_2_pulse(cxn, 
        #                 nv_sig,
        #                 num_runs,
        #                 num_reps,
        #                 States.HIGH,
        #                 numpy.pi/4,
        #                 plot=False)
        
        ### Run a test by intentionally adding phase to pi_y pulses and 
        ### see that in the extracted pulse errors
        do_impose_phase(cxn, 
                       nv_sig,
                         num_runs,
                         num_reps,)
        
        # file ='2023_02_02-10_22_37-siena-nv4_2023_01_16'
        # replot_imposed_phases(file)
            
    
    #     num_uwave_pulses = 2
    #     state = States.HIGH
    #     plot = False
    #     int_phase = 0
        
    #     # Get pulse frequencies
    #     uwave_pi_pulse = nv_sig["pi_pulse_{}".format(state.name)]
    #     uwave_pi_on_2_pulse = nv_sig["pi_on_2_pulse_{}".format(state.name)]
        
    #     pi_x_phase = 0
    #     pi_2_x_phase = 0
    #     pi_y_phase = pi/2 + int_phase
    #     pi_2_y_phase = pi/2
    
    #     pulse_1_dur = uwave_pi_on_2_pulse
    #     pulse_2_dur = uwave_pi_on_2_pulse
    #     pulse_3_dur = 0
    #     iq_phases = [0,  pi_2_x_phase, pi_2_y_phase]
    #     pulse_error, pulse_error_ste = measure_pulse_error(cxn, 
    #                 nv_sig,
    #                 uwave_pi_pulse,
    #                 num_uwave_pulses,
    #                 iq_phases,
    #                 pulse_1_dur,
    #                 pulse_2_dur,
    #                 pulse_3_dur,
    #                 num_runs,
    #                 num_reps,
    #                 state,
    #                 do_plot = plot,
    #                 Title = 'pi/2_x - pi/2_y',)
    #     pe_2_5 = pulse_error
    #     pe_2_5_err = pulse_error_ste
    #     ### 6
    #     pulse_1_dur = uwave_pi_on_2_pulse
    #     pulse_2_dur = uwave_pi_on_2_pulse
    #     pulse_3_dur = 0
        
    #     iq_phases = [0,  pi_2_y_phase, pi_2_x_phase]
    #     pulse_error, pulse_error_ste = measure_pulse_error(cxn, 
    #                 nv_sig,
    #                 uwave_pi_pulse,
    #                 num_uwave_pulses,
    #                 iq_phases,
    #                 pulse_1_dur,
    #                 pulse_2_dur,
    #                 pulse_3_dur,
    #                 num_runs,
    #                 num_reps,
    #                 state,
    #                 do_plot = plot,
    #                 Title = 'pi/2_y - pi/2_x',)
    #     pe_2_6 = pulse_error
    #     pe_2_6_err = pulse_error_ste
    
    
    #     ret_vals=test_3_pulse(cxn, 
    #                     nv_sig,
    #                     num_runs,
    #                     num_reps,
    #                     state,
    #                     int_phase,
    #                     plot,)
        
    #     pe9, pe9e, pe10, pe10e, pe11, pe11e, pe12, pe12e = ret_vals
        
    #     S = [pe_2_5, pe_2_6, pe9,  pe10,  pe11]
    #     M = numpy.array([
    #             [-1,-1,-1,0,0],
    #             [1,-1,1,0,0],
    #             [1,1,-1,2,0],
    #             [-1,1,1,2,0],
    #             [-1,-1,1,0,2],
    #             # [1,-1,-1,0,2] #exclude last equation
    #           ])
    #     X = numpy.linalg.inv(M).dot(S)
        
    #     e_y_p = 0 # we are setting this value to 0
    #     e_z_p = X[0]
    #     v_x_p = X[1]
    #     v_z_p = X[2]
    #     e_y = X[3]
    #     v_x = X[4]
        
    # print('')
    # print('************RESULTS************')
    # print("Pi_X pulse-------")
    # print("Rotation axis error along y-axis, e_y = {:.5f} rad".format(e_y))
    # print("Pi_Y pulse-------")
    # print("Rotation axis error along x-axis, v_x = {:.5f} rad".format(v_x))
    # print('')
    # print("Pi/2_X pulse-------")
    # print("Rotation axis error along y-axis, e_y' = {:.5f} rad (intentionally set to 0)".format(e_y_p))
    # print("Rotation axis error along z-axis, e_z' = {:.5f} rad".format(e_z_p))
    # print("Pi/2_Y pulse-------")
    # print("Rotation axis error along x-axis, v_x' = {:.5f} rad".format(v_x_p))
    # print("Rotation axis error along z-axis, v_z' = {:.5f} rad".format(v_z_p))
    # print('*******************************')
