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

    # Scan ranges

    scan_range = 0.2
    num_scan_steps = 60

    scan_range = 0.10
    num_scan_steps = 60

    with labrad.connect() as cxn:
        # For now we only support square scans so pass scan_range twice
        image_sample.main(cxn, nv_sig, scan_range, scan_range,
                          num_scan_steps, apd_indices)


def do_optimize(nv_sig, apd_indices):

    with labrad.connect() as cxn:
        optimize.main(cxn, nv_sig, apd_indices,
                      set_to_opti_coords=False,
                      save_data=True, plot_data=True)


def do_optimize_list(nv_sig_list, apd_indices):

    with labrad.connect() as cxn:
        optimize.optimize_list(cxn, nv_sig_list, apd_indices,
                   set_to_opti_coords=False, save_data=True, plot_data=False)


def do_stationary_count(nv_sig, apd_indices):

    # In nanoseconds
    run_time = 120 * 10**9
    readout = 100 * 10**6

    with labrad.connect() as cxn:
        stationary_count.main(cxn, nv_sig, run_time, readout, apd_indices)


def do_g2_measurement(nv_sig, apd_a_index, apd_b_index):

    # Run times are in seconds
#    run_time = 2
#    run_time = 30
#    run_time = 60 * 3
#    run_time = 60 * 5
    run_time = 60 * 10
#    run_time = 60 * 20

    diff_window = 150  # ns

    with labrad.connect() as cxn:
        g2_measurement.main(cxn, nv_sig, run_time, diff_window,
                            apd_a_index, apd_b_index)


def do_resonance(nv_sig, apd_indices, freq_center=2.87, freq_range=0.2):

    num_steps = 101
    num_runs = 2
    uwave_power = -13.0  # -13.0 with a 1.5 ND is a good starting point
#    uwave_power = -11.0
#    uwave_power = -10.0

    with labrad.connect() as cxn:
        resonance.main(cxn, nv_sig, apd_indices, freq_center, freq_range,
                               num_steps, num_runs, uwave_power)

def do_pulsed_resonance(nv_sig, apd_indices,
                        freq_center=2.87, freq_range=0.2):

    num_steps = 51
#    num_steps = 101
    num_runs = 1
    uwave_power = 9.0  # 9.0 is the highest reasonable value, accounting for saturation
#    uwave_power = 5.0

    with labrad.connect() as cxn:
        pulsed_resonance.main(cxn, nv_sig, apd_indices, freq_center,
                              freq_range, num_steps, num_runs, uwave_power)

def do_optimize_magnet_angle(nv_sig, apd_indices):

    angle_range = [0, 150]
    num_angle_steps = 6
    freq_center = 2.87
    freq_range = 0.2
    num_freq_steps = 51
    num_freq_runs = 1
    uwave_power = 9.0

    optimize_magnet_angle.main(nv_sig, apd_indices,
                               angle_range, num_angle_steps,
                               freq_center, freq_range,
                               num_freq_steps, num_freq_runs, uwave_power)

def do_rabi(nv_sig, apd_indices,
            uwave_freq, do_uwave_gate_number):

    uwave_power = 9.0  # 9.0 is the highest reasonable value, accounting for saturation
#    uwave_time_range = [0, 500]
#    uwave_time_range = [0, 400]
    uwave_time_range = [0, 300]
    num_steps = 51

    num_reps = 10**5

#    num_runs = 1
    num_runs = 2
#    num_runs = 4
#    num_runs = 6

    with labrad.connect() as cxn:
        rabi.main(cxn, nv_sig, apd_indices,
                  uwave_freq, uwave_power, uwave_time_range,
                  do_uwave_gate_number,
                  num_steps, num_reps, num_runs)


def do_t1_double_quantum(nv_sig, apd_indices,
                         uwave_freq_plus, uwave_freq_minus,
                         uwave_pi_pulse_plus, uwave_pi_pulse_minus,
                         relaxation_time_range, num_steps, num_reps,
                         init_read_list):

    uwave_power = 9
    num_runs = 120  # This'll triple the expected duration listed below!!
#    num_runs = 80  # This'll double the expected duration listed below!!
#    num_runs = 40
#    num_runs = 20  # This'll halve the expected duration listed below
#    num_runs = 1  # Pick this one for the best noise to signal ratio

    with labrad.connect() as cxn:
        t1_double_quantum.main(cxn, nv_sig, apd_indices,
                     uwave_freq_plus, uwave_freq_minus,
                     uwave_power, uwave_power,
                     uwave_pi_pulse_plus, uwave_pi_pulse_minus,
                     relaxation_time_range, num_steps, num_reps, num_runs,
                     init_read_list)


def do_ramsey(name, nv_sig, nd_filter, apd_indices):

    uwave_power = 9
    uwave_freq = 2.8086
    detuning = 2.5 # MHz
    uwave_pi_half_pulse = 32
#    precession_time_range = [0, 15 * 10**3]
#    precession_time_range = [0, 2 * 10**3]
    precession_time_range = [0, 4 * 10**3]
#    precession_time_range = [0, 8 * 10**3]

    num_steps = 151
#    num_steps = 101
#    num_steps = 51
    num_reps = 3 * 10**5
#    num_reps = 10**6
#    num_runs = 6
#    num_runs = 4
#    num_runs = 2
    num_runs = 1

    with labrad.connect() as cxn:
            ramsey.main(cxn, nv_sig, nd_filter, apd_indices,
                        uwave_freq, detuning, uwave_power, uwave_pi_half_pulse,
                        precession_time_range, num_steps, num_reps, num_runs,
                        name)


def do_spin_echo(name, nv_sig, nd_filter, apd_indices):

    uwave_power = 9
    uwave_freq = 2.8589
    rabi_period = 144.4
    precession_time_range = [0, 200 * 10**3]
#    precession_time_range = [0, 2 * 10**6]
#    precession_time_range = [0, 10 * 10**6]

#    num_steps = 101
#    num_steps = 51
    num_steps = 21
#    num_steps = 11
#    num_steps = 6
#    num_reps = 10**4
    num_reps = 5 * 10**5
#    num_reps = 10**6
#    num_runs = 4
    num_runs = 2
#    num_runs = 1

    with labrad.connect() as cxn:
            spin_echo.main(cxn, nv_sig, nd_filter, apd_indices,
                        uwave_freq, uwave_power, rabi_period,
                        precession_time_range, num_steps, num_reps, num_runs,
                        name)


def do_sample_nvs(nv_sig_list, apd_indices):

    # g2 parameters
    run_time = 60 * 5
    diff_window = 150 * 10**3  # 150 ns in ps

    # ESR parameters
    num_steps = 101
    num_runs = 5
    uwave_power = -13.0  # -13.0 with a 1.5 ND is a good starting point

    for nv_sig in nv_sig_list:

        with labrad.connect() as cxn:
            g2_zero = g2_measurement.main(cxn, nv_sig, run_time,
                                          diff_window, apd_indices[0],
                                          apd_indices[1])
            if g2_zero < 0.5:
                resonance.main(cxn, nv_sig, apd_indices, 2.87, 0.1,
                               num_steps, num_runs, uwave_power)


def do_set_drift_from_reference_image(nv_sig, apd_indices):

#    ref_file_name = '2019-06-10_15-22-25_ayrton12'  # 60 x 60
    ref_file_name = '2019-06-27_16-37-18_johnson1' # bulk nv, first one we saw

    with labrad.connect() as cxn:
        set_drift_from_reference_image.main(cxn, ref_file_name,
                                            nv_sig, apd_indices)


def do_test_major_routines(nv_sig, apd_indices):
    """Run this whenver you make a significant code change. It'll make sure
    you didn't break anything in the major routines.
    """

    test_major_routines.main(nv_sig, apd_indices)


# %% Script Code


# Functions only run when called. Since this part of the script is not in a
# function, it will run when the script is run.
# __name__ will only be __main__ if we're running the file as a program.
# The below pattern enables us to import this file as a module without
# running it as a program.
if __name__ == '__main__':

    # %% Shared parameters


#    apd_indices = [0]
    apd_indices = [0, 1]

    sample_name = 'johnson1'

    nv0_2019_06_27 = {'coords': [-0.169, -0.306, 38.74], 'nd_filter': 'nd_0.5',
                      'expected_count_rate': 45, 'magnet_angle': 41.8,
                      'name': sample_name}

    nv_sig_list = [nv0_2019_06_27]

    # %% t1 measurements, preparation population and readout population.

    # Array for the t1 measuremnt, formatted:
    # [init_read_list, relaxation_time_range, num_steps, num_reps]
    t1_exp_array = numpy.array([[[1,-1], [0, 15*10**6], 11, 5000],
                                [[1,1], [0, 15*10**6], 11, 5000],
                                [[0,1], [0, 15*10**6], 11, 5000],
                                [[0,0], [0, 15*10**6], 11, 5000]])

    # Array for the parameters of a given NV, formatted:
    # [nv_sig, uwave_freq_plus, uwave_pi_pulse_plus, uwave_freq_minus, uwave_pi_pulse_minus]
    # Tektronix controls plus, Berkeley controls minus
    params_array = numpy.array([[nv0_2019_06_27, 2.8086, 65, 2.9345, 105]])

    # %% Functions to run

    try:

        # Routines that don't need an NV
#        set_xyz_zero()
#        set_xyz([0.229, 0.163, 50.0])
#        tool_belt.set_drift([0.0, 0.0, 0.0])

        # Routines that expect lists
#        optimize_list(cxn, nv_sig_list, apd_indices)
#        do_sample_nvs(nv_sig_list, apd_indices)
#        drift = tool_belt.get_drift()

        # Routines that expect single NVs
        for nv_sig in nv_sig_list:
#            do_image_sample(nv_sig, apd_indices)
#            set_xyz([0.0, 0.0, z_voltage + tool_belt.get_drift()[2]])
#            do_optimize(nv_sig, apd_indices)
#            do_stationary_count(nv_sig, apd_indices)
#            do_g2_measurement(nv_sig, apd_indices[0], apd_indices[1])
#            do_resonance(nv_sig, apd_indices, freq_range=0.2)
#            do_resonance(nv_sig, apd_indices, freq_center=2.875, freq_range=0.1)
#            do_resonance(nv_sig, apd_indices, freq_center=2.7631, freq_range=0.05)
#            do_resonance(nv_sig, apd_indices, freq_center=2.825, freq_range=0.05)
#            do_resonance(nv_sig, apd_indices, freq_center=2.878, freq_range=0.05)
#            do_pulsed_resonance(nv_sig, apd_indices)
#            do_pulsed_resonance(nv_sig, apd_indices, freq_center=2.810, freq_range=0.06)
#            do_pulsed_resonance(nv_sig, apd_indices, freq_center=2.935, freq_range=0.06)
#            do_optimize_magnet_angle(nv_sig, apd_indices)
#            do_rabi(nv_sig, apd_indices, 2.8086, 0)
#            do_rabi(nv_sig, apd_indices, 2.9345, 1)
            do_ramsey(nv_sig, apd_indices)
#            do_set_drift_from_reference_image(nv_sig, apd_indices)
            do_test_major_routines(nv_sig, apd_indices)

#         %% FULL CONTROL T1

#        for nv_ind in range(len(params_array)):
#
#            nv_sig = params_array[nv_ind, 0]
#
#            uwave_freq_plus = params_array[nv_ind, 1]
#            uwave_pi_pulse_plus = params_array[nv_ind, 2]
#            uwave_freq_minus = params_array[nv_ind, 3]
#            uwave_pi_pulse_minus = params_array[nv_ind, 4]
#
#            for exp_ind in range(len(t1_exp_array)):
##            for exp_ind in [2,3,4,5,6,7]:
#
#                init_read_list = t1_exp_array[exp_ind, 0]
#                relaxation_time_range = t1_exp_array[exp_ind, 1]
#                num_steps = t1_exp_array[exp_ind, 2]
#                num_reps = t1_exp_array[exp_ind, 3]
#
#                do_t1_double_quantum(nv_sig, apd_indices,
#                              uwave_freq_plus, uwave_freq_minus,
#                              uwave_pi_pulse_plus, uwave_pi_pulse_minus,
#                              relaxation_time_range, num_steps, num_reps,
#                              init_read_list)


    finally:
        # Reset our hardware - this should be done in each routine, but
        # let's double check here
        tool_belt.reset_cfm()
        # Kill safe stop
        if tool_belt.check_safe_stop_alive():
            print("\n\nRoutine complete. Press enter to exit.")
            tool_belt.poll_safe_stop()
