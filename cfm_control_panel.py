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


def set_xyz(nv_sig):
    with labrad.connect() as cxn:
        tool_belt.set_xyz(cxn, nv_sig)


def set_xyz_zero():
    with labrad.connect() as cxn:
        tool_belt.set_xyz_zero(cxn)


# %% Major Routines


def do_image_sample(name, nv_sig, nd_filter, apd_indices):
    
    # Scan ranges
    
#    scan_range = 0.5
#    num_scan_steps = 150
#    num_scan_steps = 200
    
    scan_range = 0.2
    num_scan_steps = 60

    with labrad.connect() as cxn:
        # For now we only support square scans so pass scan_range twice
        image_sample.main(cxn, nv_sig, nd_filter, scan_range, scan_range,
                          num_scan_steps, apd_indices, name=name)


def do_optimize(name, nv_sig, nd_filter, apd_indices):

    with labrad.connect() as cxn:
        optimize.main(cxn, nv_sig, nd_filter, apd_indices, name,
                      set_to_opti_coords=False,
                      save_data=True, plot_data=True)
        
        
def do_optimize_list(name, nv_sig_list, nd_filter, apd_indices):

    with labrad.connect() as cxn:
        optimize.optimize_list(cxn, nv_sig_list, nd_filter, apd_indices,
                               name, set_to_opti_coords=False,
                               save_data=True, plot_data=False)


def do_stationary_count(name, nv_sig, nd_filter, apd_indices):

    # In nanoseconds
    run_time = 60 * 10**9
    readout = 100 * 10**6

    with labrad.connect() as cxn:
        stationary_count.main(cxn, nv_sig, nd_filter, run_time, readout, apd_indices,
                              name=name)


def do_g2_measurement(name, nv_sig, nd_filter, apd_a_index, apd_b_index):

    # Run times are in seconds
#    run_time = 2
#    run_time = 30
#    run_time = 60 * 3
#    run_time = 60 * 5
    run_time = 60 * 10
#    run_time = 60 * 20
    
    diff_window = 150 * 10**3  # 150 ns in ps
    
    with labrad.connect() as cxn:
        g2_measurement.main(cxn, nv_sig, nd_filter, run_time, diff_window,
                            apd_a_index, apd_b_index, name=name)



def do_resonance(name, nv_sig, nd_filter, apd_indices, freq_center=2.87, freq_range=0.2):
    
    num_steps = 101
    num_runs = 4
    uwave_power = -13.0  # -13.0 with a 1.5 ND is a good starting point
    
    with labrad.connect() as cxn:
        resonance.main(cxn, nv_sig, nd_filter, apd_indices, freq_center, freq_range,
                               num_steps, num_runs, uwave_power, name=name)

def do_rabi(name, nv_sig, nd_filter, apd_indices, 
            expected_counts, uwave_freq, do_uwave_gate_number):

    uwave_power = 9.0  # 9.0 is the highest reasonable value, accounting for saturation 
    uwave_time_range = [0, 400]
    num_steps = 51
    
    num_reps = 10**5
    
#    num_runs = 1
#    num_runs = 2
    num_runs = 5
#    num_runs = 6

    with labrad.connect() as cxn:
        new_coords = rabi.main(cxn, nv_sig, nd_filter, apd_indices, 
                  expected_counts, uwave_freq, uwave_power, uwave_time_range,
                  do_uwave_gate_number,
                  num_steps, num_reps, num_runs, name=name)
        
    return new_coords

def do_t1_double_quantum(name, nv_sig, nd_filter, apd_indices,
                         expected_counts, uwave_freq_plus, uwave_freq_minus, 
                         uwave_pi_pulse_plus, uwave_pi_pulse_minus,
                         relaxation_time_range, num_steps, num_reps,
                         init_read_list):
    
    uwave_power = 9
    num_runs = 80  # This'll double the expected duration listed below!!
#    num_runs = 40
#    num_runs = 20
#    num_runs = 1
    
    with labrad.connect() as cxn:
         new_coords = t1_double_quantum.main(cxn, nv_sig, nd_filter,
                     apd_indices, expected_counts,
                     uwave_freq_plus, uwave_freq_minus, uwave_power, 
                     uwave_pi_pulse_plus, uwave_pi_pulse_minus,
                     relaxation_time_range, num_steps, num_reps, num_runs, 
                     init_read_list, name)
         
    return new_coords

def do_ramsey_measurement(name, nv_sig, nd_filter, 
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
            ramsey.main(cxn, nv_sig, nd_filter, sig_shrt_apd_index, ref_shrt_apd_index,
                        sig_long_apd_index, ref_long_apd_index, expected_counts,
                        uwave_freq, uwave_power, uwave_pi_half_pulse, precession_time_range,
                        num_steps, num_reps, num_runs, 
                        name)
         
#def do_t1_measurement_single(name, nv_sig, nd_filter,
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
#        t1_measurement_single.main(cxn, nv_sig, nd_filter, sig_apd_index, ref_apd_index, expected_counts,
#                        uwave_freq, uwave_power, uwave_pi_pulse, relaxation_time_range,
#                        num_steps, num_reps, num_runs, 
#                        name, measure_spin_0)
            
def do_sample_nvs(name, nv_sig_list, nd_filter, apd_indices):
    
    # g2 parameters
    run_time = 60 * 5
    diff_window = 150 * 10**3  # 150 ns in ps
    
    # ESR parameters
    num_steps = 101
    num_runs = 5
    uwave_power = -13.0  # -13.0 with a 1.5 ND is a good starting point
    
    for nv_sig in nv_sig_list:
    
        with labrad.connect() as cxn:
            g2_zero = g2_measurement.main(cxn, nv_sig, nd_filter, run_time,
                                          diff_window, apd_indices[0],
                                          apd_indices[1], name=name)
            if g2_zero < 0.5:
                resonance.main(cxn, nv_sig, nd_filter, apd_indices, 2.87, 0.1,
                               num_steps, num_runs, uwave_power, name=name)
    


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

#    apd_indices = [0]
    apd_indices = [0, 1]
    
    # %% NV coordinates
    
    center = [0.0, 0.0, 53.6]
     
#    nv_list = [center]
#    nv_list = [[-0.083, 0.018, 51.6]]
#    nv_list = [[0.247, 0.236, 53.2]]
#    nv_list = [nv1_2019_06_03]
    
    drift = numpy.array([0.0, 0.0, 0.0])
    
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
    
    
#    nv_sig = [-0.3, 0.3, 55.7]
#    background = [-0.273, 0.331, 54.0]
#    nv = [-0.237, 0.318, 56.0]
#    zero_coords = [0.0, 0.0, 50.0]
    
    z_voltage = 50.2
    background_count_rate = 3
    
#    nv_sig_list = [
#               [-0.142, 0.501, z_voltage, 53, background_count_rate],
#               [-0.133, 0.420, z_voltage, 45, background_count_rate],
#               [-0.141, 0.269, z_voltage, 92, background_count_rate],
#               [-0.224, 0.070, z_voltage, 49, background_count_rate],
#               [-0.234, 0.123, z_voltage, 83, background_count_rate],
#               [-0.236, 0.163, z_voltage, 78, background_count_rate],
#               [-0.269, 0.184, z_voltage, 40, background_count_rate],
#               [-0.306, 0.160, z_voltage, 64, background_count_rate],
#               [-0.269, 0.184, z_voltage, 40, background_count_rate],
#               [-0.287, 0.260, z_voltage, 66, background_count_rate],
#               [-0.308, 0.270, z_voltage, 30, background_count_rate],
#               [-0.335, 0.280, z_voltage, 74, background_count_rate],
#               [-0.324, 0.325, z_voltage, 90, background_count_rate],
#               [-0.379, 0.280, z_voltage, 43, background_count_rate],
#               [-0.388, 0.294, z_voltage, 31, background_count_rate],
#               [-0.389, 0.264, z_voltage, 85, background_count_rate],
#               [-0.375, 0.183, z_voltage, 45, background_count_rate],
#               [-0.416, 0.398, z_voltage, 35, background_count_rate],
#               [-0.397, 0.383, z_voltage, 100, background_count_rate],
#               [-0.397, 0.337, z_voltage, 85, background_count_rate],
#               [-0.456, 0.152, z_voltage, 63, background_count_rate],
#               [-0.415, 0.398, z_voltage, 33, background_count_rate],
#               [-0.393, 0.484, z_voltage, 60, background_count_rate]]
#    
    nv_sig_list = [
               [-0.324, 0.325, z_voltage, 90, background_count_rate]]
    
#    nv_list = [nv_sig]
#    nv_list =    [ [-0.308, 0.270, 50, 45]]
    
#    offsetxy=nv4_2019_06_06_offset # this adds an offset to the XY galvo values for certain functions 
#                                   # (currently resonance or T1_double_quantum) - SK 6/8/19     
   
#    offsetxy = [0,0]
    
#    params_array = numpy.array([[nv4_2019_06_06_ref, 2.8501, 66, 2.8786, 62, expected_counts]])
    
    try:
        
        # Routines that expect lists
#        optimize_list(name, cxn, nv_sig_list, nd_filter, apd_indices)
#        do_sample_nvs(name, nv_sig_list, nd_filter, apd_indices)
            
        # Routines that expect single NVs
        for nv_sig in nv_sig_list:
#            coords = nv_sig[0:3]
#            set_xyz_zero()
#            do_image_sample(name, coords, nd_filter, scan_range, num_scan_steps, apd_indices)
#            do_optimize(name, nv_sig, nd_filter, apd_indices)
#            do_stationary_count(name, nv_sig, nd_filter, apd_indices)
#            do_g2_measurement(name, nv_sig, nd_filter, apd_indices[0], apd_indices[1])
#            do_resonance(name, nv_sig, nd_filter, apd_indices, expected_counts)
            do_resonance(name, nv_sig, nd_filter, apd_indices, freq_center=2.82, freq_range=0.1)
#            do_resonance(name, nv_sig, nd_filter, apd_indices, expected_counts, freq_center=2.878, freq_range=0.05)
#            ret_val = do_rabi(name, nv_sig, nd_filter, apd_indices, expected_counts, 2.8554, 0)
#            nv_sig = ret_val 
#            do_rabi(name, nv_sig, nd_filter, apd_indices, expected_counts, 2.8501, 0)
#            do_rabi(name, nv_sig, nd_filter, apd_indices, expected_counts, 2.8786, 1)
#            do_ramsey_measurement(name, nv_sig, nd_filter, apd_indices, expected_counts)
#            do_t1_measurement(name, nv_sig, nd_filter, apd_indices, expected_counts,
#                              uwave_freq, uwave_pi_pulse, relaxation_time_range, measure_spin_0)
#            do_t1_measurement_single(name, nv_sig, nd_filter, apd_indices, expected_counts)
#            do_t1_init_read_control(name, nv_sig, nd_filter, apd_indices, expected_counts,
#                              init_state = -1, read_state = 0)
        
          
#         %% FULL CONTROL T1

#        for nv_ind in range(len(params_array)):
#            
#            nv_sig = params_array[nv_ind, 0]
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
#                ret_val = do_t1_double_quantum(name, nv_sig, nd_filter,
#                              apd_indices, expected_counts,
#                              uwave_freq_plus, uwave_freq_minus, 
#                              uwave_pi_pulse_plus, uwave_pi_pulse_minus,
#                              relaxation_time_range, num_steps, num_reps,
#                              init_read_list)                
#                
#                print('new nv_sig: \n' + '[{:.3f}, {:.3f}, {:.1f}]'.format(*ret_val)) 
#                nv_sig = ret_val       
                
## %%            

    finally:
        tool_belt.reset_state()
        # Kill safe stop
        if tool_belt.check_safe_stop_alive():
            print("\n\nRoutine complete. Press enter to exit.")
            tool_belt.poll_safe_stop()
            
