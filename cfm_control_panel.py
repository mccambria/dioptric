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
import utils.tool_belt as tool_belt
import majorroutines.image_sample as image_sample
import majorroutines.optimize as optimize
import majorroutines.stationary_count as stationary_count
import majorroutines.resonance as resonance
import majorroutines.rabi as rabi
import majorroutines.g2_measurement as g2_measurement
import majorroutines.t1_measurement as t1_measurement


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
                      set_to_opti_centers=False,
                      save_data=True, plot_data=True)


def do_stationary_count(name, coords, nd_filter, apd_index):

    # In nanoseconds
    run_time = 20 * 10**9
    readout = 100 * 10**6

    with labrad.connect() as cxn:
        stationary_count.main(cxn, coords, nd_filter, run_time, readout, apd_index,
                              name=name)


def do_g2_measurement(name, coords, nd_filter, apd_a_index, apd_b_index):

    run_time = 60 * 10
#    run_time = 30
    diff_window = 150 * 10**3  # 100 ns in ps
    
    with labrad.connect() as cxn:
        g2_measurement.main(cxn, coords, nd_filter, run_time, diff_window,
                            apd_a_index, apd_b_index, name=name)


def do_resonance(name, coords, nd_filter, apd_index):

#    freq_center = 2.87
#    freq_center = 2.843
    freq_center = 2.853
#    freq_center = 2.875
#    freq_center = 2.888
#    freq_range = 0.2
    freq_range = 0.05
    num_steps = 101
    num_runs = 5
    uwave_power = -13.0  # -13.0 with a 1.5 ND is a good starting point

    with labrad.connect() as cxn:
        resonance.main(cxn, coords, nd_filter, apd_index, freq_center, freq_range,
                       num_steps, num_runs, uwave_power, name=name)


def do_rabi(name, coords, nd_filter, sig_apd_index, ref_apd_index):

    uwave_freq = 2.853
    uwave_power = 9.0  # 9.0 is the highest reasonable value, accounting for saturation 
    # ND 1.5 is a good starting point
    uwave_time_range = [0, 500]
    num_steps = 51
    
    num_reps = 10**5
#    num_reps = 100
    num_runs = 3
#    num_runs = 8

    with labrad.connect() as cxn:
        rabi.main(cxn, coords, nd_filter, sig_apd_index, ref_apd_index,
                  uwave_freq, uwave_power, uwave_time_range,
                  num_steps, num_reps, num_runs, name=name)


def do_t1_measurement(name, coords, nd_filter,
                      sig_shrt_apd_index, ref_shrt_apd_index,
                      sig_long_apd_index, ref_long_apd_index):
    
    uwave_freq = 2.853
    uwave_power = 9
    uwave_pi_pulse = round(195.4 / 2)
#    relaxation_time_range = [0, 100 * 10**3]
#    relaxation_time_range = [0, 1000 * 10**3]
#    relaxation_time_range = [0, 500 * 10**3]
    relaxation_time_range = [0, 100 * 10**3]
    num_steps = 26
    num_reps = 3 * 10**4
    num_runs = 10
    measure_spin_0 = False
    
    with labrad.connect() as cxn:
         t1_measurement.main(cxn, coords, nd_filter,
                     sig_shrt_apd_index, ref_shrt_apd_index,
                     sig_long_apd_index, ref_long_apd_index,
                     uwave_freq, uwave_power, uwave_pi_pulse,
                     relaxation_time_range, num_steps, num_reps, num_runs, 
                     name, measure_spin_0)


# %% Script Code


# Functions only run when called. Since this part of the script is not in a
# function, it will run when the script is run.
# __name__ will only be __main__ if we're running the file as a program.
# The below pattern enables us to import this file as a module without
# running it as a program.
if __name__ == '__main__':

    # %% Frequently modified/shared parameters

    name = 'ayrton12'
    
    #  Coords are from 5/6 unless otherwise stated
    nv0 = [-0.060, 0.041, 49.6]
    nv_list = [nv0]

#    other_coords = [-0.15, 0.05, 49.6]
#    nv_list = [other_coords]
    
    nd_filter = 1.5

    apd_a_index = 0
    apd_b_index = 1
    apd_c_index = 2
    apd_d_index = 3

    scan_range = 0.2
    num_scan_steps = 120

    # %% Functions to run

    try:
        for nv in nv_list:
            coords = nv
            print(coords)
#            set_xyz_zero()
#            do_image_sample(name, coords, nd_filter, scan_range, num_scan_steps, apd_a_index)
#            do_optimize(name, coords, nd_filter, apd_a_index)
#            do_stationary_count(name, coords, nd_filter, apd_a_index)
#            do_g2_measurement(name, coords, nd_filter, nd_filter, apd_a_index, apd_b_index)
#            do_resonance(name, coords, nd_filter, apd_a_index)
#            do_rabi(name, coords, nd_filter, apd_a_index, apd_b_index)
            do_t1_measurement(name, coords, nd_filter, apd_a_index, apd_b_index, apd_c_index, apd_d_index)
    finally:
        # Kill safe stop
        if tool_belt.check_safe_stop_alive():
            print("\n\nRoutine complete. Press enter to exit.")
            tool_belt.poll_safe_stop()
