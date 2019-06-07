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
import majorroutines.t1_double_quantum as t1_double_quantum
import majorroutines.ramsey as ramsey


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
    run_time = 60 * 10**9
    readout = 100 * 10**6

    with labrad.connect() as cxn:
        stationary_count.main(cxn, coords, nd_filter, run_time, readout, apd_index,
                              name=name)


def do_g2_measurement(name, coords, nd_filter, apd_a_index, apd_b_index):

    # Run times are in seconds
#    run_time = 2
#    run_time = 30
#    run_time = 60 * 3
#    run_time = 60 * 5
    run_time = 60 * 10
#    run_time = 60 * 20
    
    diff_window = 150 * 10**3  # 100 ns in ps
    
    with labrad.connect() as cxn:
        g2_measurement.main(cxn, coords, nd_filter, run_time, diff_window,
                            apd_a_index, apd_b_index, name=name)


def do_resonance(name, coords, nd_filter, apd_index, expected_counts, freq_center=2.87, freq_range=0.2):

#    freq_center = 2.87
#    freq_range = 0.2
#    freq_range = 0.1
#    freq_range = 0.03
    
    num_steps = 101
    num_runs = 5
    uwave_power = -13.0  # -13.0 with a 1.5 ND is a good starting point

    with labrad.connect() as cxn:
        resonance.main(cxn, coords, nd_filter, apd_index, expected_counts, freq_center, freq_range,
                       num_steps, num_runs, uwave_power, name=name)


def do_rabi(name, coords, nd_filter, apd_indices, 
            expected_counts, uwave_freq, do_uwave_gate_number):

    uwave_power = 9.0  # 9.0 is the highest reasonable value, accounting for saturation 
    uwave_time_range = [0, 400]
    num_steps = 51
    
    num_reps = 10**5
    
    num_runs = 1
#    num_runs = 2
#    num_runs = 6

    with labrad.connect() as cxn:
        new_coords = rabi.main(cxn, coords, nd_filter, apd_indices, 
                  expected_counts, uwave_freq, uwave_power, uwave_time_range,
                  do_uwave_gate_number,
                  num_steps, num_reps, num_runs, name=name)
        
    return new_coords

def do_t1_double_quantum(name, coords, nd_filter, apd_indices,
                         expected_counts, uwave_freq_plus, uwave_freq_minus, 
                         uwave_pi_pulse_plus, uwave_pi_pulse_minus,
                         relaxation_time_range, num_steps, num_reps,
                         init_read_list):
    
    uwave_power = 9
#    num_runs = 1
    num_runs = 40
#    num_runs = 20
    
    with labrad.connect() as cxn:
         new_coords = t1_double_quantum.main(cxn, coords, nd_filter,
                     apd_indices, expected_counts,
                     uwave_freq_plus, uwave_freq_minus, uwave_power, 
                     uwave_pi_pulse_plus, uwave_pi_pulse_minus,
                     relaxation_time_range, num_steps, num_reps, num_runs, 
                     init_read_list, name)
         
    return new_coords

def do_ramsey_measurement(name, coords, nd_filter, 
                      sig_shrt_apd_index, ref_shrt_apd_index,
                      sig_long_apd_index, ref_long_apd_index, expected_counts):
    
    uwave_power = 9
    uwave_freq = 2.852
    uwave_pi_half_pulse = 32
    precession_time_range = [0, 1 * 10**3]
    
    num_steps = 21
    num_reps = 10**5
    num_runs = 3
    
    
    with labrad.connect() as cxn:
            ramsey.main(cxn, coords, nd_filter, sig_shrt_apd_index, ref_shrt_apd_index,
                        sig_long_apd_index, ref_long_apd_index, expected_counts,
                        uwave_freq, uwave_power, uwave_pi_half_pulse, precession_time_range,
                        num_steps, num_reps, num_runs, 
                        name)
         
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

    # %% General

    name = 'ayrton12'  # Sample name
    
    nd_filter = 1.5

    apd_a_index = 0
    apd_b_index = 1
    
    # %% NV coordinates
    
    center = [0.0, 0.0, 53.6]
    
    ############### Pre 5/28 ###############
    # The below coordinates are shifted by ~[]
    
    #  2019-04-30-NV2
    
#    nv2_2019_04_30 = [-0.044, 0.053, 53.4]  # 2019-04-30-NV2
#    nv2_2019_04_30 = [-0.041, 0.054, 54.3]  # 5/27
#    nv2_2019_04_30 = [-0.042, 0.053, 54.0]  # 5/28
    
#    nv2_2019_04_30 = [-0.011, 0.006, 57.0]  # 4/29 coords
#    nv2_2019_04_30 = [0.35, 0.0, 57.0]  # 5/10 coords
#    nv2_2019_04_30 = [-0.079, 0.039, 57.0]  # optimize
#    nv2_2019_04_30 = [0.0, 0.0, 57.0]  # zero
    
    # 2019-05-07-NV6
#    nv6 = [-0.071, 0.085, 48.7] ##

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

    # 2019-05-10 NVs    
#    nv1 = [0.291, 0.246, 53.7] # Great nv!
#    nv2 = [0.370, 0.111, 48.6]
#    nv3 = [0.235, -0.122, 48.9]
#    nv4 = [0.288, -0.156, 48.4] # Good nv
    
    # Decent g2    
#    nv5 = [0.318, 0.2338, 53.7]
#    nv6 = [0.313, 0.143, 48.9]
#    nv7 = [0.189, 0.149, 48.8]
#    nv8 = [0.283, 0.029, 48.8]
#    nv9 = [0.268, -0.032, 48.7]
    
#    nv_list = [[0.257, 0.234, 48.5],
#        [0.370, 0.111, 48.6],
#        [0.235, -0.122, 48.9],
#        [0.292, -0.158, 48.7]]
#    other_coords = [0.25 ,0.0,48.7]
    
#    other_coords = [0.242, 0.237, 49.9]
    
    
#    nv_list = [nv1, nv2_2019_04_30, nv4]
    
    ############### Post 5/28 ###############
    # micrometer coords 203.1, 277.1
    
#    nv2_2019_04_30 = [-0.045, 0.072, 56.5] # 5/30
#    nv2_2019_04_30 = [-0.036, 0.071, 56.6]  # 5/30 after installing new magnet mount
#    nv2_2019_04_30 = [-0.046, 0.079, 56.3]  # 5/31 after reinstalling new magnet mount
#    nv2_2019_04_30 = [-0.055, 0.076, 56.0]  # 6/1 noon
#    nv2_2019_04_30 = [-0.053, 0.078, 56.2]  # 6/1 1:15pm
#    nv2_2019_04_30 = [-0.055, 0.077, 56.1]  # 6/2
#    nv1_2019_05_10 = [0.286, 0.266, 56.5]
    
    ############### 6/3 ###############
    # micrometer coords 208.3, 277.1
    
    nv1_2019_06_03 = [0.0, 0.0, 51.6]
    
    nv_list = [[0.188, 0.236, 52.0],
               [0.158, 0.223, 52.0],
               [0.097, 0.236, 52.0],
               [0.079, 0.197, 52.0],
               [0.012, 0.231, 52.0],
               [0.024, 0.203, 52.0],
               [0.078, 0.196, 52.0],
               [0.019, 0.160, 52.0],
               [-0.033, -0.078, 52.0],
               [-0.050, -0.053, 52.0],
               [-0.238, 0.161, 52.0],
               [-0.224, 0.135, 52.0],
               [-0.237, 0.123, 52.0],
               [-0.231, 0.088, 52.0],
               [-0.226, 0.068, 52.0],
               [-0.242, 0.076, 52.0],
               [-0.236, 0.029, 52.0],
               [0.046, -0.144, 52.0],
               [0.039, -0.125, 52.0],
               [-0.023, 0.030, 52.0],
               [0.001, -0.003, 52.0],
               [0.126, -0.068, 52.0],
               [0.120, -0.094, 52.0],
               [0.110, -0.106, 52.0],
               [0.125, 0.116, 52.0],
               [-0.139, -0.127, 52.0],
               [-0.162, -0.127, 52.0],
               [-0.181, -0.111, 52.0],
               [-0.154, -0.149, 52.0],
               [-0.137, -0.057, 52.0],
               [-0.219, -0.022, 52.0]]
    
#    nv_list = [[0.189, 0.233, 51.8], # Previous list of coordinates 6/3
#               [0.159, 0.219, 51.5],
#               [0.095, 0.233, 49.8],
#               [0.078, 0.194, 51.5],
#               [0.012, 0.228, 51.6],
#               [0.024, 0.193, 51.7],
#               [0.078, 0.194, 51.6],
#               [0.019, 0.158, 51.7],
#               [-0.035, -0.080, 51.7],
#               [-0.050, -0.054, 51.7],
#               [-0.239, 0.160, 51.6],
#               [-0.226, 0.134, 50.3],
#               [-0.237, 0.120, 51.6],
#               [-0.232, 0.086, 51.5],
#               [-0.227, 0.083, 51.7],
#               [-0.252, 0.075, 51.8],
#               [-0.236, 0.027, 51.6],
#               [0.046, -0.126, 48.8],
#               [0.039, -0.125, 49.8],
#               [-0.024, 0.029, 51.8],
#               [0.001, -0.001, 51.8],
#               [0.124, -0.069, 51.9],
#               [0.119, -0.094, 51.7],
#               [0.109, -0.106, 51.8],
#               [0.165, 0.111, 54.5],
#               [-0.140, -0.127, 51.9],
#               [-0.163, -0.125, 51.8],
#               [-0.182, -0.123, 51.8],
#               [-0.154, -0.151, 51.6],
#               [-0.138, -0.056, 51.7],
#               [-0.221, -0.023, 51.6]]
    
    nv_list = [[0.249, 0.238, 53.6], # new list 6/4
               [0.234, 0.241, 53.6],
               [0.150, 0.221, 53.6],
               [0.004, 0.229, 53.6],
#               [-0.032, 0.221, 53.2],
                [0.207, 0.194, 53.6],
                [0.118, 0.192, 53.6],
                [0.070, 0.195, 53.6],
                [0.220, 0.158, 53.6],
                [0.093, 0.148, 53.6],
                [0.011, 0.160, 53.6],
                [-0.247, 0.161, 53.6],
                [0.143, 0.143, 53.6],
                [-0.245, 0.121, 53.6],
                [0.129, 0.102, 53.6],
                [0.146, 0.070, 53.6],
                [-0.099, 0.061, 53.6],
                [0.139, 0.055, 53.6],
                [-0.098, 0.037, 53.6],
                [-0.244, 0.028, 53.6],
                [0.115, 0.004, 53.6],
                [-0.033, 0.030, 53.6],
                [-0.080, 0.022, 53.6],
                [-0.043, -0.079, 53.6],
                [0.032, -0.123, 53.6],
                [-0.178, -0.075, 53.6],
                [-0.188, -0.110, 53.6],
                [-0.171, -0.124, 53.6],
                [-0.109, -0.161, 53.6],
                [-0.034, -0.136, 53.6]]
     
    
#    nv_list = [center]
#    nv_list = [[-0.083, 0.018, 51.6]]
#    nv_list = [[0.247, 0.236, 53.2]]
#    nv_list = [nv1_2019_06_03]
    
    drift = numpy.array([0.0, 0.0, 0.0])

    # %% Image_sample scan ranges
    
#    scan_range = 10.0
#    num_scan_steps = 400
    
#    scan_range = 5.0
#    num_scan_steps = 200
    
#    scan_range = 1.5
#    num_scan_steps = 450
#    num_scan_steps = 600
    
#    scan_range = 1.0
#    num_scan_steps = 300
    
#    scan_range = 0.5
#    num_scan_steps = 150
#    num_scan_steps = 200
    
#    scan_range = 0.3
#    num_scan_steps = 90
    
    scan_range = 0.2
    num_scan_steps = 60
    
#    scan_range = 0.05
#    num_scan_steps = 60
#    num_scan_steps = 30
#    num_scan_steps = 15
    
#    scan_range = 0.01
#    num_scan_steps = 60
     
    # %% Optimization parameters
    
    # Based on the current nv, what kcounts/s do we expect?
    # If not known, set to None
#    expected_counts = None
    expected_counts = 70
    
    # %% t1 measurements, preparation population and readout population.
    
    zero_to_zero = [0,0]
    plus_to_plus = [1,1]
    minus_to_minus = [-1,-1]
    plus_to_zero = [1,0]
    minus_to_zero = [-1,0]
    zero_to_plus = [0,1]
    zero_to_minus = [0,-1]
    plus_to_minus = [1,-1]
    minus_to_plus = [-1,1]
    
    # Array for the t1 measuremnt, formatted:
    # [init_read_list, relaxation_time_range, num_steps, num_reps]
    
#    t1_exp_array = numpy.array([
#                                [plus_to_minus, [0, 800*10**3], 51, 2 * 10**4],
#                                [minus_to_plus, [0, 800*10**3], 51, 2 * 10**4],
#                                [plus_to_minus, [0, 100*10**3], 101, 4 * 10**4],
#                                [minus_to_plus, [0, 100*10**3], 101, 4 * 10**4],
#  
#                                [plus_to_plus,   [0, 100*10**3],  101, 4 * 10**4],
#                                [minus_to_minus, [0, 100*10**3],  101, 4 * 10**4], 
#                                [plus_to_plus,   [0, 800*10**3],  26, 2 * 10**4],
#                                [minus_to_minus, [0, 800*10**3],  26, 2 * 10**4], 
#                                
#                                [zero_to_plus,   [0, 1000*10**3], 26, 2 * 10**4],
#                                [zero_to_minus,  [0, 1000*10**3], 26, 2 * 10**4], 
#                                [plus_to_zero,   [0, 800*10**3], 51, 2 * 10**4],
#                                [minus_to_zero,  [0, 800*10**3], 51, 2 * 10**4], 
#
#                                [zero_to_zero,   [0, 1000*10**3], 26, 2 * 10**4]])


    # For splittings < 75 MHz
    
    # ~13 hours
    t1_exp_array = numpy.array([[plus_to_minus,  [0, 100*10**3], 51, 2 * 10**4],
                                [plus_to_minus,  [0, 500*10**3], 41,  1 * 10**4],
                                [plus_to_plus,   [0, 100*10**3], 51, 2 * 10**4],
                                [plus_to_plus,   [0, 500*10**3], 41,  1 * 10**4],
                                [plus_to_zero,   [0, 500*10**3], 41, 1 * 10**4],
                                [zero_to_plus,   [0, 1500*10**3], 41, 1 * 10**4],
                                [zero_to_zero,   [0, 1500*10**3], 41, 1 * 10**4]])

    # For splittings > 75 MHz
    
    # ~18 hours
#    t1_exp_array = numpy.array([[plus_to_minus,  [0, 1500*10**3], 41, 1 * 10**4],
#                                [plus_to_plus,   [0, 1500*10**3], 41, 1 * 10**4],
#                                [plus_to_zero,   [0, 1500*10**3], 41, 1 * 10**4],
#                                [zero_to_plus,   [0, 1500*10**3], 41, 1 * 10**4],
#                                [zero_to_zero,   [0, 1500*10**3], 41, 1 * 10**4]])
    
    # ~18 hours
#    t1_exp_array = numpy.array([[plus_to_minus,  [0, 1500*10**3], 41, 1 * 10**4],
#                                [plus_to_plus,   [0, 1500*10**3], 41, 1 * 10**4],
#                                [plus_to_zero,   [0, 2000*10**3], 31, 1 * 10**4],
#                                [zero_to_plus,   [0, 2000*10**3], 31, 1 * 10**4],
#                                [zero_to_zero,   [0, 2000*10**3], 31, 1 * 10**4]])
    
    # Array for the parameters of a given NV, formatted:
    # [nv coordinates, uwave_freq_plus, uwave_pi_pulse_plus, uwave_freq_minus,
    #                            uwave_pi_pulse_minus, expected_counts]
    #   uwave_MINUS should be associated with the HP signal generator
#    params_array = numpy.array([[nv2_2019_04_30, 2.8380, 96, 2.8942, 102, 62]])

    # %% Functions to run
    
#    apd_indices = [apd_a_index]
    apd_indices = [apd_a_index, apd_b_index]
    
    # 6/6 12:45
#    nv_list = [[0.251, 0.235, 54.0],
#               [0.005, 0.226, 53.9],
#               [0.140, 0.052, 54.1],
#               [0.032, -0.126, 54.1],
#               [-0.176, -0.077, 54.1],
#               [-0.188, -0.112, 53.9]]
    
#    nv0_2019_06_06 = [0.251, 0.236, 54.3]  # 6/6 noon
#    nv0_2019_06_06 = [0.249, 0.235, 53.8]  # 6/6 2:21
#    nv0_2019_06_06 = [0.252, 0.238, 54.1]  # 6/6 3:00
    nv0_2019_06_06 = [0.251, 0.235, 54.1]  # 6/6 5:00
    
    nv1_2019_06_06 = [0.208, 0.195, 54.0]  # 6/7 am
    
    nv2_2019_06_06 = [0.131, 0.102, 54.0]  # 6/7 am
    
    nv_list = [nv1_2019_06_06, nv2_2019_06_06]
    params_array = numpy.array([[nv_list[0], 2.8430, 34, 2.8911, 48, expected_counts]])
    
    try:
        for nv in nv_list:
            coords = nv
#            set_xyz_zero()
#            do_image_sample(name, coords, nd_filter, scan_range, num_scan_steps, apd_indices)
#            do_optimize(name, coords, nd_filter, apd_indices)
#            do_optimize_list(name, coords, nd_filter, apd_indices)
#            do_stationary_count(name, coords, nd_filter, apd_indices)
            do_g2_measurement(name, coords, nd_filter, apd_indices[0], apd_indices[1])
#            do_resonance(name, coords, nd_filter, apd_indices, expected_counts)
#            do_resonance(name, coords, nd_filter, apd_indices, expected_counts, freq_center=2.843, freq_range=0.05)
#            do_resonance(name, coords, nd_filter, apd_indices, expected_counts, freq_center=2.892, freq_range=0.05)
#            ret_val = do_rabi(name, coords, nd_filter, apd_indices, expected_counts, 2.8554, 0)
#            coords = ret_val 
#            do_rabi(name, coords, nd_filter, apd_indices, expected_counts, 2.8430, 0)
#            do_rabi(name, coords, nd_filter, apd_indices, expected_counts, 2.8911, 1)
#            do_ramsey_measurement(name, coords, nd_filter, apd_indices, expected_counts)
#            do_t1_measurement(name, coords, nd_filter, apd_indices, expected_counts,
#                              uwave_freq, uwave_pi_pulse, relaxation_time_range, measure_spin_0)
#            do_t1_measurement_single(name, coords, nd_filter, apd_indices, expected_counts)
#            do_t1_init_read_control(name, coords, nd_filter, apd_indices, expected_counts,
#                              init_state = -1, read_state = 0)

        
          
#         %% FULL CONTROL T1

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
##            for exp_ind in [2,3,4,5,6,7]:
#            
#                init_read_list = t1_exp_array[exp_ind, 0]
#                relaxation_time_range = t1_exp_array[exp_ind, 1]
#                num_steps = t1_exp_array[exp_ind, 2]
#                num_reps = t1_exp_array[exp_ind, 3]
#        
#                ret_val = do_t1_double_quantum(name, coords, nd_filter,
#                              apd_indices, expected_counts,
#                              uwave_freq_plus, uwave_freq_minus, 
#                              uwave_pi_pulse_plus, uwave_pi_pulse_minus,
#                              relaxation_time_range, num_steps, num_reps,
#                              init_read_list)                
#                
#                print('new coords: \n' + '[{:.3f}, {:.3f}, {:.1f}]'.format(*ret_val)) 
#                coords = ret_val       
#                
## %%            

    finally:
        # Kill safe stop
        if tool_belt.check_safe_stop_alive():
            print("\n\nRoutine complete. Press enter to exit.")
            tool_belt.poll_safe_stop()
