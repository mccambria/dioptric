# -*- coding: utf-8 -*-
"""
This file contains functions to control the CFM. Just change the function call
in the main section at the bottom of this file and run the file. Shared or
frequently changed parameters are in the __main__ body and relatively static
parameters are in the function definitions.

Created on Sun Nov 25 14:00:28 2018

@author: mccambria
"""


# %% Imports

import labrad
import numpy
import utils.tool_belt as tool_belt
import majorroutines.image_sample as image_sample
import majorroutines.optimize as optimize
import majorroutines.stationary_count as stationary_count
import majorroutines.resonance as resonance
import majorroutines.rabi as rabi
import majorroutines.g2_measurement as g2_measurement
import majorroutines.t1_measurement as t1_measurement
import majorroutines.t1_init_read_control as t1_init_read_control
import majorroutines.t1_double_quantum as t1_double_quantum
#import majorroutines.t1_measurement_single as t1_measurement_single


# %% Minor Routines


def set_xyz(coords):
    with labrad.connect() as cxn:
        tool_belt.set_xyz(cxn, coords)


def set_xyz_zero():
    with labrad.connect() as cxn:
        tool_belt.set_xyz_zero(cxn)


# %% Major Routines


def do_image_sample(name, coords, nd_filter, scan_range, num_steps, apd_index):

    readout = 10 * 10**6  # In nanoseconds

    with labrad.connect() as cxn:
        # For now we only support square scans so pass scan_range twice
        image_sample.main(cxn, coords, nd_filter, scan_range, scan_range,
                          num_steps, readout, apd_index, name=name)


def do_optimize(name, coords, nd_filter, apd_index):

    with labrad.connect() as cxn:
        optimize.main(cxn, coords, nd_filter, apd_index, name,
                      expected_counts=None, set_to_opti_centers=False,
                      save_data=True, plot_data=True, )
        
def do_optimize_list(name, coords, nd_filter, apd_index):

    with labrad.connect() as cxn:
        optimize.optimize_list(cxn, coords, nd_filter, apd_index, name,
                      expected_counts=None, set_to_opti_centers=False,
                      save_data=True, plot_data=True, )


def do_stationary_count(name, coords, nd_filter, apd_index):

    # In nanoseconds
    run_time = 2 * 10**9
    readout = 100 * 10**6

    with labrad.connect() as cxn:
        stationary_count.main(cxn, coords, nd_filter, run_time, readout, apd_index,
                              name=name)


def do_g2_measurement(name, coords, nd_filter, apd_a_index, apd_b_index):

    run_time = 60 * 5
#    run_time = 2
#    run_time = 30
    diff_window = 150 * 10**3  # 100 ns in ps
    
    with labrad.connect() as cxn:
        g2_measurement.main(cxn, coords, nd_filter, run_time, diff_window,
                            apd_a_index, apd_b_index, name=name)


def do_resonance(name, coords, nd_filter, apd_index, expected_counts):

    freq_center = 2.87
    freq_range = 0.2
#    freq_range = 0.05
    num_steps = 101
    num_runs = 6
    uwave_power = -13.0  # -13.0 with a 1.5 ND is a good starting point

    with labrad.connect() as cxn:
        resonance.main(cxn, coords, nd_filter, apd_index, expected_counts, freq_center, freq_range,
                       num_steps, num_runs, uwave_power, name=name)


def do_rabi(name, coords, nd_filter, sig_apd_index, ref_apd_index, expected_counts):

    uwave_freq = 2.880
    uwave_power = 9.0  # 9.0 is the highest reasonable value, accounting for saturation 
    # ND 1.5 is a good starting point
    uwave_time_range = [0, 400]
    num_steps = 51
    
    num_reps = 10**5
#    num_reps = 100
#    num_runs = 2
    num_runs = 8

    with labrad.connect() as cxn:
        rabi.main(cxn, coords, nd_filter, sig_apd_index, ref_apd_index, 
                  expected_counts, uwave_freq, uwave_power, uwave_time_range,
                  num_steps, num_reps, num_runs, name=name)


def do_t1_measurement(name, coords, nd_filter,
                      sig_shrt_apd_index, ref_shrt_apd_index,
                      sig_long_apd_index, ref_long_apd_index, expected_counts,
                      uwave_freq, uwave_pi_pulse, relaxation_time_range, 
                      measure_spin_0):
    
#    uwave_freq = 2.851
    uwave_power = 9
#    uwave_pi_pulse = round(110.3 / 2)
#    relaxation_time_range = [0, 100 * 10**3]
#    relaxation_time_range = [0, 1000 * 10**3]
#    relaxation_time_range = [0, 500 * 10**3]
#    relaxation_time_range = [0, 100 * 10**4]
    num_steps = 101
    num_reps =  6 * 10**3
    num_runs = 20
#    measure_spin_0 = False
    
    with labrad.connect() as cxn:
         new_coords = t1_measurement.main(cxn, coords, nd_filter,
                     sig_shrt_apd_index, ref_shrt_apd_index,
                     sig_long_apd_index, ref_long_apd_index,
                     expected_counts,
                     uwave_freq, uwave_power, uwave_pi_pulse,
                     relaxation_time_range, num_steps, num_reps, num_runs, 
                     name, measure_spin_0)
         
    return new_coords

def do_t1_init_read_control(name, coords, nd_filter,
                      sig_shrt_apd_index, ref_shrt_apd_index,
                      sig_long_apd_index, ref_long_apd_index, expected_counts,
                      uwave_freq_plus, uwave_freq_minus, 
                      uwave_pi_pulse_plus, uwave_pi_pulse_minus,
                      init_state, read_state):
    
    # Set right now for 2019-04-30-NV2
    
#    uwave_freq_plus = 2.851
#    uwave_pi_pulse_plus = 104
#    uwave_freq_minus = 2.880
#    uwave_pi_pulse_minus = 126
    
    uwave_power = 9
    relaxation_time_range = [0, 0.1 * 10**6]
#    relaxation_time_range = [0, 1.5 * 10**3]
    
    num_steps = 201
#    num_steps = 5
    
    num_reps =  5 * 10**4
    
    num_runs = 25
#    num_runs = 1
    
    with labrad.connect() as cxn:
         new_coords = t1_init_read_control.main(cxn, coords, nd_filter,
                     sig_shrt_apd_index, ref_shrt_apd_index,
                     sig_long_apd_index, ref_long_apd_index,
                     expected_counts,
                     uwave_freq_plus, uwave_freq_minus, uwave_power, 
                     uwave_pi_pulse_plus, uwave_pi_pulse_minus,
                     relaxation_time_range, num_steps, num_reps, num_runs, 
                     init_state, read_state, name)
         
    return new_coords

def do_t1_double_quantum(name, coords, nd_filter,
                      sig_shrt_apd_index, ref_shrt_apd_index,
                      sig_long_apd_index, ref_long_apd_index, expected_counts,
                      uwave_freq_plus, uwave_freq_minus, 
                      uwave_pi_pulse_plus, uwave_pi_pulse_minus,
                      relaxation_time_range, num_steps,
                      init_state, read_state):
    
    # Set right now for 2019-04-30-NV2
    
#    uwave_freq_plus = 2.851
#    uwave_pi_pulse_plus = 104
#    uwave_freq_minus = 2.880
#    uwave_pi_pulse_minus = 126
    
    uwave_power = 9
#    relaxation_time_range = [0, 0.1 * 10**6]
#    relaxation_time_range = [0, 1.5 * 10**3]
    
#    num_steps = 201
#    num_steps = 5
    
    num_reps =  2 * 10**4
    
    num_runs = 20
#    num_runs = 1
    
    with labrad.connect() as cxn:
         new_coords = t1_double_quantum.main(cxn, coords, nd_filter,
                     sig_shrt_apd_index, ref_shrt_apd_index,
                     sig_long_apd_index, ref_long_apd_index,
                     expected_counts,
                     uwave_freq_plus, uwave_freq_minus, uwave_power, 
                     uwave_pi_pulse_plus, uwave_pi_pulse_minus,
                     relaxation_time_range, num_steps, num_reps, num_runs, 
                     init_state, read_state, name)
         
    return new_coords
         
#def do_t1_measurement_single(name, coords, nd_filter,
#                             sig_apd_index, ref_apd_index, expected_counts):
#    
#    uwave_freq = 2.888
#    uwave_power = 9
#    uwave_pi_pulse = round( 0 / 2)
#    relaxation_time_range = [0, 1.5 * 10**6]
#    num_steps = 101
#    num_reps = 3 * 10**3
#    num_runs = 10  
#    measure_spin_0 = True
#    
#    
#    with labrad.connect() as cxn:
#        t1_measurement_single.main(cxn, coords, nd_filter, sig_apd_index, ref_apd_index, expected_counts,
#                        uwave_freq, uwave_power, uwave_pi_pulse, relaxation_time_range,
#                        num_steps, num_reps, num_runs, 
#                        name, measure_spin_0)
    


# %% Script Code


# Functions only run when called. Since this part of the script is not in a
# function, it will run when the script is run.
# __name__ will only be __main__ if we're running the file as a program.
# The below pattern enables us to import this file as a module without
# running it as a program.
if __name__ == '__main__':

    # %% Frequently modified/shared parameters

    name = 'ayrton12'
    
    #  2019-04-30-NV2
#    nv2 = [-0.044, 0.043, 49.1] ## coordinates 5/7 18:00
#    nv2 = [-0.072, 0.039, 47.7] ## coordinates 5/8 9:00
    
    nv2_2019_04_30 = [-0.080, 0.041, 48.3] # 2019-04-30-NV2
#    nv_list = [nv2]
    
    # 2019-05-07-NV6
    nv6 = [-0.071, 0.085, 48.7] ##

#    nv_list = [nv6]
    
    # 2019-05-10
#    nv_list = [[0.257, 0.234, 48.5],
#        [0.285, 0.218, 48.8],
#        [0.347, 0.206, 48.8],
##        [0.331, 0.213, 48.5],
#        [0.446, 0.150, 48.6],
#        [0.313, 0.143, 48.9],
#        [0.189, 0.149, 48.8],
#        [0.462, 0.099, 48.7],
#        [0.370, 0.111, 48.6],
#        [0.393, 0.071, 48.7],
#        [0.261, 0.084, 48.8],
#        [0.183, 0.099, 48.7],
#        [0.283, 0.029, 48.8],
#        [0.301, 0.011, 48.7],
#        [0.458, 0.050, 48.8],
#        [0.463, 0.039, 48.8],
#        [0.268, -0.032, 48.7],
#        [0.124, -0.032, 48.8],
##        [0.211, -0.079, 48.9],
#        [0.301, -0.120, 48.7],
#        [0.289, -0.128, 48.7],
#        [0.235, -0.122, 48.9],
#        [0.189, -0.133, 48.6],
#        [0.125, -0.159, 48.7],
#        [0.292, -0.158, 48.7]]
    
    nv1 = [0.256, 0.235, 48.4] # Great nv!
    nv2 = [0.370, 0.111, 48.6]
    nv3 = [0.235, -0.122, 48.9]
    nv4 = [0.288, -0.156, 48.4] # Good nv
    
    # Decent g2    
    nv5 = [0.285, 0.218, 48.8]
    nv6 = [0.313, 0.143, 48.9]
    nv7 = [0.189, 0.149, 48.8]
    nv8 = [0.283, 0.029, 48.8]
    nv9 = [0.268, -0.032, 48.7]
    
    
#    nv_list = [[0.257, 0.234, 48.5],
#        [0.370, 0.111, 48.6],
#        [0.235, -0.122, 48.9],
#        [0.292, -0.158, 48.7]]
#    other_coords = [0.25 ,0.0,48.7]
    
    other_coords = [0.35 ,0.0, 48.7]
    
    
    nv_list = [nv1, nv2_2019_04_30, nv4]
    nv_list = [nv1]
        
    nd_filter = 1.5

    apd_a_index = 0
    apd_b_index = 1
    apd_c_index = 2
    apd_d_index = 3

    scan_range = 0.1
    num_scan_steps = 60
     
    # Based on the current nv, what kcounts/s do we expect?
    # If not know, set to None
    expected_counts = 30
    
    # arrays for the t1 measuremnt info
    
    # 2019-04-30-NV2
#    m_plus_one = [[0, 2 * 10**3], 2.852, 99.55, False]
#    m_minus_one = [[0, 2 * 10**3], 2.880, 126.85, False]
#    m_zero = [[0, 1.2 * 10**6], 2.87, 0, True]
    
    # 2019-04-30-NV2 
    zero_to_zero = [0,0]
    plus_to_plus = [1,1]
    minus_to_minus = [-1,-1]
    plus_to_zero = [1,0]
    minus_to_zero = [-1,0]
    zero_to_plus = [0,1]
    zero_to_minus = [0,-1]
    plus_to_minus = [1,-1]
    minus_to_plus = [-1,1]
     
    # 2019-05-07-NV6
#    m_plus_one = [[0, 25 * 10**3], 2.850, 56.25, False]
#    m_minus_one = [[0, 25 * 10**3], 2.881, 42.9, False]
#    m_zero = [[0, 1.5 * 10**6], 2.87, 0, True]
#
#
#    t1_array = numpy.array([m_plus_one, m_minus_one, m_zero])
#    t1_exp_array = numpy.array([zero_to_zero, plus_to_plus, minus_to_minus, 
#                                 plus_to_zero, minus_to_zero, zero_to_plus, 
#                                 zero_to_minus])
    
    t1_exp_array = numpy.array([[minus_to_minus, [0, 50*10**3],   101],
                                [plus_to_minus,  [0, 50*10**3],   101],
                                [minus_to_plus,  [0, 50*10**3],   101],
                                [plus_to_minus,  [0, 700*10**3],  151],
                                [minus_to_plus,  [0, 700*10**3],  151]])

    
#    params_array = numpy.array([[nv1, 2.851, 89, 2.880, 82, 35],
#                                [nv2_2019_04_30, 2.854, 104, 2.880, 126, 50],
#                                [nv4, 2.856, 94, 2.880, 82, 50]])
    params_array = numpy.array([[nv1, 2.851, 80.5, 2.880, 88.3, 35]])

    # %% Functions to run
    try:
        
#        for nv in nv_list:
#            coords = nv
#            set_xyz_zero()
#            do_image_sample(name, coords, nd_filter, scan_range, num_scan_steps, apd_a_index)
#            do_optimize(name, coords, nd_filter, apd_a_index)
#            do_optimize_list(name, coords, nd_filter, apd_a_index)
#            do_stationary_count(name, coords, nd_filter, apd_a_index)
#            do_g2_measurement(name, coords, nd_filter, apd_a_index, apd_b_index)
#            do_resonance(name, coords, nd_filter, apd_a_index, expected_counts)
#            do_rabi(name, coords, nd_filter, apd_a_index, apd_b_index, expected_counts)
#            do_t1_measurement(name, coords, nd_filter, apd_a_index, 
#                              apd_b_index, apd_c_index, apd_d_index, expected_counts,
#                              uwave_freq, uwave_pi_pulse, relaxation_time_range, measure_spin_0)
#            do_t1_measurement_single(name, coords, nd_filter, apd_a_index, apd_b_index, expected_counts)
#            do_t1_init_read_control(name, coords, nd_filter, apd_a_index, 
#                              apd_b_index, apd_c_index, apd_d_index, expected_counts,
#                              init_state = -1, read_state = 0)
            
        # Double Quantum t1

        for nv_ind in range(len(params_array)):
            
            coords = params_array[nv_ind, 0]
            
            uwave_freq_plus = params_array[nv_ind, 1]
            uwave_pi_pulse_plus = params_array[nv_ind, 2]
            uwave_freq_minus = params_array[nv_ind, 3]
            uwave_pi_pulse_minus = params_array[nv_ind, 4]
            expected_counts = params_array[nv_ind, 5]
            
            for exp_ind in range(len(t1_exp_array)):
                
                states = t1_exp_array[exp_ind, 0]
                init_state = states[0]
                read_state = states[1]
                relaxation_time_range = t1_exp_array[exp_ind, 1]
                num_steps = t1_exp_array[exp_ind, 2]
                
                ret_val = do_t1_double_quantum(name, coords, nd_filter, apd_a_index, 
                              apd_b_index, apd_c_index, apd_d_index, expected_counts,
                              uwave_freq_plus, uwave_freq_minus, 
                              uwave_pi_pulse_plus, uwave_pi_pulse_minus,
                              relaxation_time_range, num_steps,
                              init_state, read_state)
                
                print("new coordinates:" + str(ret_val)) 
                coords = ret_val  
          
          
        # full control t1

#        for nv_ind in range(len(params_array)):
#            
#            coords = params_array[nv_ind, 0]
#            
#            uwave_freq_plus = params_array[nv_ind, 1]
#            uwave_pi_pulse_plus = params_array[nv_ind, 2]
#            uwave_freq_minus = params_array[nv_ind, 3]
#            uwave_pi_pulse_minus = params_array[nv_ind, 4]
#            expected_counts = params_array[nv_ind, 5]
#            
#            for exp_ind in range(len(t1_exp_array)):
#                
#                init_state = t1_exp_array[exp_ind, 0]
#                read_state = t1_exp_array[exp_ind, 1]
#                
#                ret_val = do_t1_init_read_control(name, coords, nd_filter, apd_a_index, 
#                              apd_b_index, apd_c_index, apd_d_index, expected_counts,
#                              uwave_freq_plus, uwave_freq_minus, 
#                              uwave_pi_pulse_plus, uwave_pi_pulse_minus,
#                              init_state, read_state)
#                
#                print("new coordinates:" + str(ret_val)) 
#                coords = ret_val                
#            
            
        # t1 measurement
        
#        for nv in nv_list: 
#            coords = nv
#            
#            for t1_ind in [0,1]:
#                
#                relaxation_time_range = t1_array[t1_ind,0]
#                uwave_freq = t1_array[t1_ind, 1]
#                uwave_pi_pulse = t1_array[t1_ind, 2]
#                measure_spin_0 = t1_array[t1_ind, 3]
#                expected_counts = t1_array[t1_ind, 4]
#            
#                ret_val = do_t1_measurement(name, coords, nd_filter, apd_a_index, 
#                                  apd_b_index, apd_c_index, apd_d_index, expected_counts,
#                                  uwave_freq, uwave_pi_pulse, relaxation_time_range, measure_spin_0)
#                
#                print("new coordinates:" + str(ret_val)) 
#                coords = ret_val
    finally:
        # Kill safe stop
        if tool_belt.check_safe_stop_alive():
            print("\n\nRoutine complete. Press enter to exit.")
            tool_belt.poll_safe_stop()
