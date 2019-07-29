# -*- coding: utf-8 -*-
"""This file contains functions to control the CFM. Just change the function call
in the main section at the bottom of this file and run the file. Shared or
frequently changed parameters are in the __main__ body and relatively static
parameters are in the function definitions.

Created on Sun Nov 25 14:00:28 2018

@author: mccambria
"""


# %% Imports


import labrad
import numpy
import copy
import utils.tool_belt as tool_belt
import majorroutines.image_sample as image_sample
import majorroutines.optimize as optimize
import majorroutines.stationary_count as stationary_count
import majorroutines.resonance as resonance
import majorroutines.pulsed_resonance as pulsed_resonance
import majorroutines.optimize_magnet_angle as optimize_magnet_angle
import majorroutines.rabi as rabi
import majorroutines.g2_measurement as g2_measurement
import majorroutines.t1_double_quantum as t1_double_quantum
import majorroutines.ramsey as ramsey
import majorroutines.spin_echo as spin_echo
import majorroutines.set_drift_from_reference_image as set_drift_from_reference_image
import debug.test_major_routines as test_major_routines


# %% Minor Routines


def set_xyz(nv_sig):

    with labrad.connect() as cxn:
        tool_belt.set_xyz(cxn, nv_sig)

def set_xyz_zero():

    with labrad.connect() as cxn:
        tool_belt.set_xyz_zero(cxn)


# %% Major Routines


def do_image_sample(nv_sig, apd_indices):
    
    scan_range = 0.5
    num_steps = 100

    # For now we only support square scans so pass scan_range twice
    image_sample.main(nv_sig, scan_range, scan_range, num_steps, apd_indices)

def do_optimize(nv_sig, apd_indices):

    optimize.main(nv_sig, apd_indices,
              set_to_opti_coords=False, save_data=True, plot_data=True)

def do_optimize_list(nv_sig_list, apd_indices):

    optimize.optimize_list(nv_sig_list, apd_indices)

def do_stationary_count(nv_sig, apd_indices):

    run_time = 120 * 10**9  # ns

    stationary_count.main(nv_sig, run_time, apd_indices)

def do_g2_measurement(nv_sig, apd_a_index, apd_b_index):

    run_time = 60 * 10  # s
    diff_window = 150  # ns

    g2_measurement.main(nv_sig, run_time, diff_window,
                        apd_a_index, apd_b_index)

def do_resonance(nv_sig, apd_indices, freq_center=2.87, freq_range=0.2):

    num_steps = 76
    num_runs = 2
    uwave_power = -13.0  # -13.0 with a 1.5 ND is a good starting point

    resonance.main(nv_sig, apd_indices, freq_center, freq_range,
                   num_steps, num_runs, uwave_power)

def do_pulsed_resonance(nv_sig, apd_indices,
                        freq_center=2.87, freq_range=0.2):
    
    num_steps = 51
    num_reps = 10**5
    num_runs = 2
    uwave_power = 9.0
    uwave_pulse_dur = 32

    pulsed_resonance.main(nv_sig, apd_indices, freq_center, freq_range,
                          num_steps, num_reps, num_runs,
                          uwave_power, uwave_pulse_dur)

def do_optimize_magnet_angle(nv_sig, apd_indices):

    angle_range = [0, 150]
    num_angle_steps = 6
    freq_center = 2.87
    freq_range = 0.35
    num_freq_steps = 76
    num_freq_reps = 10**5
    num_freq_runs = 7
#    num_freq_runs = 3
    uwave_power = 9.0
    uwave_pulse_dur = 120

    optimize_magnet_angle.main(nv_sig, apd_indices,
               angle_range, num_angle_steps, freq_center, freq_range,
               num_freq_steps, num_freq_reps, num_freq_runs,
               uwave_power, uwave_pulse_dur)

def do_rabi(nv_sig, apd_indices, do_uwave_gate_number):

    uwave_time_range = [0, 150]
    num_steps = 51
    num_reps = 3*10**5
    num_runs = 2

    rabi.main(nv_sig, apd_indices, uwave_time_range,
              do_uwave_gate_number, num_steps, num_reps, num_runs)

def do_t1_battery(nv_sig, apd_indices):

    num_runs = 20

    # T1 experiment parameters, formatted:
    # [[init state, read state], relaxation_time_range, num_steps, num_reps]
    # ~ 11 hours total
    t1_exp_array = numpy.array([[[1,-1], [0, 50*10**3], 51, 8*10**4], # 50 min, optimize every 2.5 min
                                [[1,-1], [0, 500*10**3], 26, 3*10**4], # 1.5 hrs, optimize every 3.5 min
                                [[1,1], [0, 50*10**3], 51, 8*10**4], # 50 min, optimize every 2.5 min
                                [[1,1], [0, 500*10**3], 26, 3*10**4], # 1.5 hrs, optimize every 3.5 min
                                [[0,1], [0, 2*10**6], 26, 2*10**4], # 3 hrs, optimize every 9 min
                                [[0,0], [0, 2*10**6], 26, 2*10**4]]) # 3 hrs, optimize every 9 min

    # Loop through the experiments
    for exp_ind in range(len(t1_exp_array)):

        init_read_states = t1_exp_array[exp_ind, 0]
        relaxation_time_range = t1_exp_array[exp_ind, 1]
        num_steps = t1_exp_array[exp_ind, 2]
        num_reps = t1_exp_array[exp_ind, 3]

        t1_double_quantum.main(nv_sig, apd_indices, relaxation_time_range,
                           num_steps, num_reps, num_runs, init_read_states)

def do_ramsey(nv_sig, apd_indices):

    detuning = 2.5  # MHz
    precession_time_range = [0, 4 * 10**3]
    num_steps = 151
    num_reps = 3 * 10**5
    num_runs = 1

    ramsey.main(nv_sig, apd_indices, detuning, precession_time_range,
                num_steps, num_reps, num_runs)

def do_spin_echo(nv_sig, apd_indices):

    precession_time_range = [0, 50 * 10**3]
    num_steps = 101
    num_reps = 3 * 10**4
    num_runs = 20

    spin_echo.main(nv_sig, apd_indices, precession_time_range,
                   num_steps, num_reps, num_runs)

def do_sample_nvs(nv_sig_list, apd_indices):

    # g2 parameters
    run_time = 60 * 5
    diff_window = 150

    # PESR parameters
    num_steps = 101
    num_reps = 10**5
    num_runs = 5
    uwave_power = 9.0
    uwave_pulse_dur = 120

    g2 = g2_measurement.main_with_cxn
    pesr = pulsed_resonance.main_with_cxn

    with labrad.connect() as cxn:
        for nv_sig in nv_sig_list:
            g2_zero = g2(cxn, nv_sig, run_time, diff_window,
                         apd_indices[0], apd_indices[1])
            if g2_zero < 0.5:
                pesr(cxn, nv_sig, apd_indices, 2.87, 0.2, num_steps,
                     num_reps, num_runs, uwave_power, uwave_pulse_dur)

def do_set_drift_from_reference_image(nv_sig, apd_indices):

    # ref_file_name = '2019-06-10_15-22-25_ayrton12'  # 60 x 60
    ref_file_name = '2019-06-27_16-37-18_johnson1' # bulk nv, first one we saw

    set_drift_from_reference_image.main(ref_file_name, nv_sig, apd_indices)

def do_test_major_routines(nv_sig, apd_indices):
    """Run this whenver you make a significant code change. It'll make sure
    you didn't break anything in the major routines.
    """

    test_major_routines.main(nv_sig, apd_indices)


# %% Run the file


if __name__ == '__main__':

    # %% Shared parameters

    apd_indices = [0]
#    apd_indices = [0, 1]
    sample_name = 'ayrton12'
    
    # Master list 7/25
    coords_list = [   [0.225, 0.142, 5.03], 
                      [0.180, 0.190, 5.02],
                      [0.016, 0.242, 5.03],
                      [-0.038, 0.231, 5.01],
                      [0.003, 0.216, 5.02], # take g(2) again
                      [0.061, 0.164, 5.03],  #  great! nv5_2019_07_25
                      [0.006, 0.187, 5.03],  # take g(2) again
                      [0.003, 0.170, 5.03],  
                      [-0.010, 0.145, 5.01],
                      [-0.080, 0.162, 5.01],
                      [-0.169, 0.161, 5.03], # great! nv10_2019_07_25
                      [-0.148, 0.111, 5.03],
                      [-0.221, 0.154, 5.03],
                      [-0.235, 0.140, 5.03],
                      [-0.229, 0.116, 5.02],
                      [-0.128, 0.049, 5.02], # possibly nv15_2019_07_25
                      [-0.191, 0.041, 5.04], # great! nv16_2019_07_25
                      [-0.101, 0.048, 5.02],
                      [0.032, 0.006, 5.03],  # great! low counts nv18_2019_07_25
                      [-0.075, 0.042, 5.02],
                      [-0.085, -0.006, 5.04],
                      [-0.012, -0.032, 5.03],
                      [0.045, -0.042, 5.01],
                      [0.026, -0.068, 5.01], # take g(2) again
                      [0.036, -0.188, 5.03],
                      [0.122, -0.219, 5.02], # great! nv25_2019_07_25
                      [-0.101, -0.082, 5.00],
                      [-0.229, -0.052, 5.03], # great! nv27_2019_07_25
                      [-0.209, -0.105, 5.05],
                      [-0.222, -0.121, 5.03], # possibly nv29_2019_07_25
                      [-0.056, -0.015, 5.02],
                      [-0.137, -0.046, 5.03],
                      [0.242, -0.018, 5.03],
                      [0.229, -0.024, 5.07]] # take g(2) again
    
    nv5_2019_07_25 = {'coords': coords_list[5],
          'name': '{}-nv{}_2019_07_25'.format(sample_name, 5),
          'expected_count_rate': 22,
          'nd_filter': 'nd_1.5', 'magnet_angle': 257.4,
          'resonance_low': 2.7890, 'rabi_low': 76.3, 'uwave_power_low': 9.0,
          'resonance_high': 2.9385, 'rabi_high': 54.5, 'uwave_power_high': 10.0}
    
    nv5_2019_07_25['resonance_low'] = 2.800
    nv5_2019_07_25['resonance_high'] = 2.9395
    nv5_2019_07_25['rabi_low'] = 63
    nv5_2019_07_25['rabi_high'] = 55.7
    
    
    nv16_2019_07_25 = {'coords': coords_list[16],
          'name': '{}-nv{}_2019_07_25'.format(sample_name, 16),
          'expected_count_rate': 19,
          'nd_filter': 'nd_1.5', 'magnet_angle': 194.1,
          'resonance_low': 2.8221, 'rabi_low': 111.6, 'uwave_power_low': 9.0,
          'resonance_high': 2.8994, 'rabi_high': 115.1, 'uwave_power_high': 10.0}
    nv25_2019_07_25 = {'coords': coords_list[25],
          'name': '{}-nv{}_2019_07_25'.format(sample_name, 25),
          'expected_count_rate': 38,
          'nd_filter': 'nd_1.5', 'magnet_angle': 222.3,
          'resonance_low': 2.8584, 'rabi_low': 423.2, 'uwave_power_low': 9.0,
          'resonance_high': 2.9034, 'rabi_high': 271.5, 'uwave_power_high': 10.0}
    nv27_2019_07_25 = {'coords': coords_list[27],
          'name': '{}-nv{}_2019_07_25'.format(sample_name, 27),
          'expected_count_rate': 20,
          'nd_filter': 'nd_1.5', 'magnet_angle': 15.4,
          'resonance_low': None, 'rabi_low': None, 'uwave_power_low': 9.0,
          'resonance_high': None, 'rabi_high': None, 'uwave_power_high': 10.0}
    nv29_2019_07_25 = {'coords': coords_list[29],
          'name': '{}-nv{}_2019_07_25'.format(sample_name, 29),
          'expected_count_rate': 39,
          'nd_filter': 'nd_1.5', 'magnet_angle': None,
          'resonance_low': None, 'rabi_low': None, 'uwave_power_low': 9.0,
          'resonance_high': None, 'rabi_high': None, 'uwave_power_high': 10.0}
    
#    Debug NV
#    nv27_2019_07_25 = {'coords': coords_list[27],
#          'name': '{}-nv{}_2019_07_25'.format(sample_name, 27),
#          'expected_count_rate': 20,
#          'nd_filter': 'nd_1.5', 'magnet_angle': None,
#          'resonance_low': 2.80, 'rabi_low': 122, 'uwave_power_low': 9.0,
#          'resonance_high': 2.90, 'rabi_high': 155, 'uwave_power_high': 10.0}
    
#    nv_sig_list = [nv5_2019_07_25, nv16_2019_07_25, nv25_2019_07_25,
#                   nv27_2019_07_25, nv29_2019_07_25]
    nv_sig_list = [nv5_2019_07_25]

    # %% Functions to run

    try:

        # Operations that don't need an NV
        # set_xyz_zero()
#         set_xyz([0.0, 0.0, 5.0])
#        drift = tool_belt.get_drift()
#        tool_belt.set_drift([float(drift[0])+0.02, float(drift[1])-0.02, 0.15])
#        tool_belt.set_drift([-0.012, 0.0, -0.02])
        # 
#        set_xyz([0.0, 0.0, z_voltage + tool_belt.get_drift()[2]])

        # Routines that expect lists of NVs
#        do_optimize_list(nv_sig_list, apd_indices)
#        do_sample_nvs(nv_sig_list, apd_indices)
#        do_g2_measurement(nv_sig_list[0], apd_indices[0], apd_indices[1])

        # Routines that expect single NVs
        for ind in range(len(nv_sig_list)):
            nv_sig = nv_sig_list[ind]
#            for z in numpy.linspace(5.5, 6.5, 6):
#                nv_sig_copy = copy.deepcopy(nv_sig)
#                coords = nv_sig_copy['coords']
#                nv_sig_copy['coords'] = [coords[0], coords[1], z]
#                do_image_sample(nv_sig_copy, apd_indices)
#            do_image_sample(nv_sig, apd_indices)
#            do_optimize(nv_sig, apd_indices)
#            do_stationary_count(nv_sig, apd_indices)
#            do_g2_measurement(nv_sig, apd_indices[0], apd_indices[1])
#            do_optimize_magnet_angle(nv_sig, apd_indices)
#            do_resonance(nv_sig, apd_indices)
#            do_pulsed_resonance(nv_sig, apd_indices)
            do_pulsed_resonance(nv_sig, apd_indices, freq_center=2.800, freq_range=0.15)
#            do_pulsed_resonance(nv_sig, apd_indices, freq_center=2.9398, freq_range=0.1)
#            do_pulsed_resonance(nv_sig, apd_indices, freq_center=2.87, freq_range=0.15)
#            do_pulsed_resonance(nv_sig, apd_indices, freq_center=2.935, freq_range=0.06)
#            do_rabi(nv_sig, apd_indices, 0) 
#            do_rabi(nv_sig, apd_indices, 1)
#            do_t1_battery(nv_sig, apd_indices)
#            do_ramsey(nv_sig, apd_indices)
#            do_spin_echo(nv_sig, apd_indices)
#            do_set_drift_from_reference_image(nv_sig, apd_indices)
#            do_test_major_routines(nv_sig, apd_indices)
#            with labrad.connect() as cxn:
#                tool_belt.set_xyz_on_nv(cxn, nv_sig)

    finally:
        # Reset our hardware - this should be done in each routine, but
        # let's double check here
        tool_belt.reset_cfm()
        # Kill safe stop
        if tool_belt.check_safe_stop_alive():
            print('\n\nRoutine complete. Press enter to exit.')
            tool_belt.poll_safe_stop()
