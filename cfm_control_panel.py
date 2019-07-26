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
import minorroutines.set_drift_from_reference_image as set_drift_from_reference_image
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
    
#    scan_range = 0.75
#    num_steps = 125

#    scan_range = 0.2
#    num_steps = 60

#    scan_range = 0.10
#    num_steps = 30

    # For now we only support square scans so pass scan_range twice
    image_sample.main(nv_sig, scan_range, scan_range, num_steps, apd_indices)

def do_optimize(nv_sig, apd_indices):

    optimize.main(nv_sig, apd_indices,
              set_to_opti_coords=False, save_data=True, plot_data=True)

def do_optimize_list(nv_sig_list, apd_indices):

    optimize.optimize_list(nv_sig_list, apd_indices)

def do_stationary_count(nv_sig, apd_indices):

    # ns
    run_time = 120 * 10**9

    stationary_count.main(nv_sig, run_time, apd_indices)

def do_g2_measurement(nv_sig, apd_a_index, apd_b_index):

    run_time = 60 * 10  # s
    diff_window = 150  # ns

    g2_measurement.main(nv_sig, run_time, diff_window,
                        apd_a_index, apd_b_index)

def do_resonance(nv_sig, apd_indices, freq_center=2.87, freq_range=0.2):

    num_steps = 101
    num_runs = 4
    uwave_power = -13.0  # -13.0 with a 1.5 ND is a good starting point

    resonance.main(nv_sig, apd_indices, freq_center, freq_range,
                   num_steps, num_runs, uwave_power)

def do_pulsed_resonance(nv_sig, apd_indices,
                        freq_center=2.87, freq_range=0.2):

#    num_steps = 51
#    num_reps = 5 * 10**4
#    num_runs = 5
#    # 9.0 dBm is the highest reasonable value, accounting for saturation
#    uwave_power = 9.0
#    uwave_pulse_dur = 80
##    uwave_pulse_dur = 250
##    uwave_pulse_dur = 0
    
    num_steps = 101
    num_reps = 10**5
    num_runs = 4
    uwave_power = 9.0
    uwave_pulse_dur = 120

    pulsed_resonance.main(nv_sig, apd_indices, freq_center, freq_range,
                          num_steps, num_reps, num_runs,
                          uwave_power, uwave_pulse_dur)

def do_optimize_magnet_angle(nv_sig, apd_indices):

    angle_range = [0, 150]
    num_angle_steps = 6
    freq_center = 2.87
    freq_range = 0.2
    num_freq_steps = 51
    num_freq_reps = 5 * 10**4
    num_freq_runs = 1
    uwave_power = 9.0
    uwave_pulse_dur = 80

    optimize_magnet_angle.main(nv_sig, apd_indices,
                       angle_range, num_angle_steps, freq_center, freq_range,
                       num_freq_steps, num_freq_reps, num_freq_runs,
                       uwave_power, uwave_pulse_dur)

def do_rabi(nv_sig, apd_indices,
            uwave_freq, do_uwave_gate_number):

    uwave_power = 9.0
#    uwave_time_range = [0, 700]
    uwave_time_range = [0, 300]
    num_steps = 51
    num_reps = 10**5
    num_runs = 2

    rabi.main(nv_sig, apd_indices, uwave_freq, uwave_power, uwave_time_range,
              do_uwave_gate_number, num_steps, num_reps, num_runs)

def do_t1_battery(nv_sig, apd_indices):

    uwave_power = 9
    num_runs = 420

    # Tektronix controls plus, Berkeley controls minus
    uwave_freq_plus = 2.8067
    uwave_pi_pulse_plus = 72
    uwave_freq_minus = 2.9364
    uwave_pi_pulse_minus = 116

    # T1 experiment parameters, formatted:
    # [[init state, read state], relaxation_time_range, num_steps, num_reps]
    t1_exp_array = numpy.array([[[1,-1], [0, 15*10**6], 11, 900],
                                [[1,1], [0, 15*10**6], 11, 900],
                                [[0,1], [0, 15*10**6], 11, 900],
                                [[0,0], [0, 15*10**6], 11, 900]])

    # Loop through the experiments
    for exp_ind in range(len(t1_exp_array)):

        init_read_states = t1_exp_array[exp_ind, 0]
        relaxation_time_range = t1_exp_array[exp_ind, 1]
        num_steps = t1_exp_array[exp_ind, 2]
        num_reps = t1_exp_array[exp_ind, 3]

        t1_double_quantum.main(nv_sig, apd_indices,
                 uwave_freq_plus, uwave_freq_minus,
                 uwave_power, uwave_power,
                 uwave_pi_pulse_plus, uwave_pi_pulse_minus,
                 relaxation_time_range, num_steps, num_reps, num_runs,
                 init_read_states)

def do_ramsey(nv_sig, apd_indices):

    uwave_power = 9
    uwave_freq = 2.8086
    detuning = 2.5  # MHz
    uwave_pi_half_pulse = 32
    precession_time_range = [0, 4 * 10**3]

    num_steps = 151
    num_reps = 3 * 10**5
    num_runs = 1

    ramsey.main(nv_sig, apd_indices,
                uwave_freq, detuning, uwave_power, uwave_pi_half_pulse,
                precession_time_range, num_steps, num_reps, num_runs)

def do_spin_echo(nv_sig, apd_indices):

    uwave_power = 9
    uwave_freq = 2.7569
    rabi_period = 116.0
#    precession_time_range = [0, 200 * 10**3]
#    precession_time_range = [0, 100 * 10**3]
    precession_time_range = [0, 50 * 10**3]

    num_steps = 101
    num_reps = 3 * 10**4
    num_runs = 20
#    num_steps = 51
#    num_reps = 10**5
#    num_runs = 1

    spin_echo.main(nv_sig, apd_indices,
                   uwave_freq, uwave_power, rabi_period,
                   precession_time_range, num_steps, num_reps, num_runs)

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

#    apd_indices = [0]
    apd_indices = [0, 1]
    
    sample_name = 'ayrton12'
    
    # 7/25/2019
    z_voltage = 5.03
    search = {'coords': [0.0, 0.0, z_voltage],
              'name': '{}_search'.format(sample_name),
              'nd_filter': 'nd_1.5', 'magnet_angle': None}
    nv0_2019_07_25 = {'coords': [0.047, -0.041, z_voltage],
                      'name': '{}_nv0_2019_07_25'.format(sample_name),
                      'expected_count_rate': None,
                      'nd_filter': 'nd_1.5', 'magnet_angle': None}
    
    coords_list = [   [0.225, 0.142, 5.03], 
                      [0.180, 0.190, 5.02],
                      [0.016, 0.242, 5.03],
                      [-0.038, 0.231, 5.01],
                      [0.003, 0.216, 5.02], # take g(2) again
                      [0.061, 0.164, 5.03],  #  great!
                      [0.006, 0.187, 5.03],  # take g(2) again
                      [0.003, 0.170, 5.03],  
                      [-0.010, 0.145, 5.01],
                      [-0.080, 0.162, 5.01],
                      [-0.169, 0.161, 5.03], # great!
                      [-0.148, 0.111, 5.03],
                      [-0.221, 0.154, 5.03],
                      [-0.235, 0.140, 5.03],
                      [-0.229, 0.116, 5.02],
                      [-0.128, 0.049, 5.02], # possibly
                      [-0.191, 0.041, 5.04], # great!
                      [-0.101, 0.048, 5.02],
                      [0.032, 0.006, 5.03],  # great! low counts
                      [-0.075, 0.042, 5.02],
                      [-0.085, -0.006, 5.04],
                      [-0.012, -0.032, 5.03],
                      [0.045, -0.042, 5.01],
                      [0.026, -0.068, 5.01], # take g(2) again
                      [0.036, -0.188, 5.03],
                      [0.122, -0.219, 5.02], # great!
                      [-0.101, -0.082, 5.00],
                      [-0.229, -0.052, 5.03], # great!
                      [-0.209, -0.105, 5.05],
                      [-0.222, -0.121, 5.03], # possibly
                      [-0.056, -0.015, 5.02],
                      [-0.137, -0.046, 5.03],
                      [0.242, -0.018, 5.03],
                      [0.229, -0.024, 5.07]] # take g(2) again
    
    coords_g2_retake = [[0.003, 0.216, 5.02],
                        [0.006, 0.187, 5.03],
                        [0.026, -0.068, 5.01],
                        [0.229, -0.024, 5.07]]
    
    pESR_list = [[0.061, 0.164, 5.03],
                 [-0.169, 0.161, 5.03],
                 [-0.128, 0.049, 5.02],
                 [-0.191, 0.041, 5.04],
                 [0.032, 0.006, 5.03],
                 [0.122, -0.219, 5.02],
                 [-0.229, -0.052, 5.03],
                 [-0.222, -0.121, 5.03]]

#    optimize_list = [[0.061, 0.164, 5.03]]
    
    nv_sig_list = []
    for ind in [1]:
        coords = pESR_list[ind]
        nv_sig = {'coords': coords,
                  'name': '{}-nv{}_2019_07_25'.format(sample_name, ind),
                  'expected_count_rate': None,
                  'nd_filter': 'nd_1.5', 'magnet_angle': None}
        nv_sig_list.append(nv_sig)
    
#    nv_sig_list = [search]
#    nv_sig_list = [nv0_2019_07_25]

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
            do_resonance(nv_sig, apd_indices)
#            do_pulsed_resonance(nv_sig, apd_indices)
#            do_pulsed_resonance(nv_sig, apd_indices, freq_center=2.76, freq_range=0.1)
#            do_pulsed_resonance(nv_sig, apd_indices, freq_center=2.98, freq_range=0.1)
#            do_pulsed_resonance(nv_sig, apd_indices, freq_center=2.87, freq_range=0.05)
#            do_pulsed_resonance(nv_sig, apd_indices, freq_center=2.935, freq_range=0.06)
#            do_rabi(nv_sig, apd_indices, 2.8067, 0)  # 143.1
#            do_rabi(nv_sig, apd_indices, 2.9364, 1)  # 231.5
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
