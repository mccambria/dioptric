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


def do_resonance(name, coords, nd_filter, apd_index):

    freq_center = 2.87
    freq_range = 0.2
    num_steps = 201
    num_runs = 10
    uwave_power = -13.0  # -13.0 with a 1.0 ND is a good starting point

    with labrad.connect() as cxn:
        resonance.main(cxn, coords, nd_filter, apd_index, freq_center, freq_range,
                       num_steps, num_runs, uwave_power, name=name)


def do_rabi(name, coords, nd_filter, sig_apd_index, ref_apd_index):

    uwave_freq = 2.886
    uwave_power = 9.0
    uwave_time_range = [0, 500]
    num_steps = 51
    
    num_reps = 10**5
#    num_reps = 100
    num_runs = 8
#    num_runs = 3

    with labrad.connect() as cxn:
        rabi.main(cxn, coords, nd_filter, sig_apd_index, ref_apd_index,
                  uwave_freq, uwave_power, uwave_time_range,
                  num_steps, num_reps, num_runs, name=name)


def do_g2_measurement(name, coords, nd_filter, apd_a_index, apd_b_index):

    run_time = 60 * 10
#    run_time = 30
    diff_window = 150 * 10**3  # 100 ns in ps
    
    with labrad.connect() as cxn:
        g2_measurement.main(cxn, coords, nd_filter, run_time, diff_window,
                            apd_a_index, apd_b_index, name=name)


def do_t1_measurement():
    pass


    # with labrad.connect() as cxn:
    #     t1_measurement.main(cxn, coords, sig_apd_index, ref_apd_index,
    #                         uwave_freq, uwave_power, uwave_time_range,
    #                         num_steps, num_reps, num_runs, name=name)


# %% Script Code


# Functions only run when called. Since this part of the script is not in a
# function, it will run when the script is run.
# __name__ will only be __main__ if we're running the file as a program.
# The below pattern enables us to import this file as a module without
# running it as a program.
if __name__ == '__main__':

    # %% Frequently modified/shared parameters

    name = 'ayrton12'
    
    #  Coords are from 4/30 unless otherwise stated
#    nv0 = [-0.157, -0.199, 48.7]
    nv1 = [0.000, 0.051, 49.1]
    nv2 = [-0.060, 0.040, 49.0]
    nv2 = [-0.051, 0.042, 49.4]  # 5/1 am
    nv2 = [-0.060, 0.040, 49.0]  # 5/1 pm
    nv3 = [0.006, 0.017, 49.0]
    nv4 = [-0.021, 0.019, 49.0]
    nv5 = [-0.026, -0.041, 49.0]
    nv6 = [-0.069, -0.035, 49.2]
    nv7 = [-0.080, -0.057, 49.0]
    
    nv_list = [nv2]
#    nv_list = [nv4]
#    nv_list = [nv2, nv4]

#    other_coords = [0.0, -0.1, 49.2]
#    other_coords = [-0.011, 0.006, 49.2]
#    nv_list = [other_coords]
    
    nd_filter = 1.5

    primary_apd_index = 0
    secondary_apd_index = 1

    scan_range = 0.1
    num_scan_steps = 40

    # %% Functions to run

    try:
        for nv in nv_list:
            coords = nv
            print(coords)
#            set_xyz_zero()
#            do_image_sample(name, coords, nd_filter, scan_range, num_scan_steps, primary_apd_index)
#            do_optimize(name, coords, nd_filter, primary_apd_index)
#            do_stationary_count(name, coords, nd_filter, primary_apd_index)
#            do_resonance(name, coords, nd_filter, primary_apd_index)
            do_rabi(name, coords, nd_filter, primary_apd_index, secondary_apd_index)
#            do_g2_measurement(name, coords, nd_filter, nd_filter, primary_apd_index, secondary_apd_index)
#            do_t1_measurement(name, coords, nd_filter, primary_apd_index)
    finally:
        # Kill safe stop
        if tool_belt.check_safe_stop_alive():
            print("\n\nRoutine complete. Press enter to exit.")
            tool_belt.poll_safe_stop()
