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
import time
import copy
import utils.tool_belt as tool_belt
import majorroutines.image_sample as image_sample
import majorroutines.optimize as optimize
import majorroutines.stationary_count as stationary_count
import majorroutines.resonance as resonance
import majorroutines.pulsed_resonance as pulsed_resonance
import majorroutines.optimize_magnet_angle as optimize_magnet_angle
import majorroutines.rabi as rabi
import majorroutines.discrete_rabi as discrete_rabi
import majorroutines.g2_measurement as g2_measurement
import majorroutines.t1_double_quantum as t1_double_quantum
import majorroutines.t1_dq_knill as t1_dq_knill
import majorroutines.t1_interleave as t1_interleave
import majorroutines.t1_interleave_knill as t1_interleave_knill
import majorroutines.ramsey as ramsey
import majorroutines.spin_echo as spin_echo
import majorroutines.lifetime as lifetime
import majorroutines.lifetime_v2 as lifetime_v2
# import majorroutines.set_drift_from_reference_image as set_drift_from_reference_image
import debug.test_major_routines as test_major_routines
from utils.tool_belt import States
import time


# %% Major Routines


def do_image_sample(nv_sig, apd_indices):
    
    scan_range = 0.5
    # num_steps = 90
    # num_steps = 120
    
    # scan_range = 0.15
    # num_steps = 60
    
    # scan_range = 0.75
    # num_steps = 150
    
    # scan_range = 5.0
    # scan_range = 3.0
    # scan_range = 1.5
    # scan_range = 1.0
    # scan_range = 0.75
    # scan_range = 0.3
    # scan_range = 0.2
    # scan_range = 0.15
    # scan_range = 0.1
    # scan_range = 0.075
#    scan_range = 0.025
    
    # num_steps = 300
    # num_steps = 200
    # num_steps = 150
#    num_steps = 135
    # num_steps = 120
    # num_steps = 90
    num_steps = 60
    # num_steps = 50
    # num_steps = 20

    # For now we only support square scans so pass scan_range twice
    image_sample.main(nv_sig, scan_range, scan_range, num_steps, apd_indices)


def do_optimize(nv_sig, apd_indices):

    optimize.main(nv_sig, apd_indices,
              set_to_opti_coords=False, save_data=True, plot_data=True)


def do_optimize_list(nv_sig_list, apd_indices):

    optimize.optimize_list(nv_sig_list, apd_indices)
    
    
def do_opti_z(nv_sig_list, apd_indices):
    
    optimize.opti_z(nv_sig_list, apd_indices,
              set_to_opti_coords=False, save_data=True, plot_data=True)


def do_stationary_count(nv_sig, apd_indices):

    run_time = 3 * 60 * 10**9  # ns

    stationary_count.main(nv_sig, run_time, apd_indices)


def do_g2_measurement(nv_sig, apd_a_index, apd_b_index):

    run_time =30  # s
    # diff_window = 200  # ns
    diff_window = 1000  # ns

    g2_measurement.main(nv_sig, run_time, diff_window,
                        apd_a_index, apd_b_index)


def do_resonance(nv_sig, apd_indices, freq_center=2.87, freq_range=0.2):

    num_steps = 51
    num_runs = 5
    uwave_power = -5.0

    resonance.main(nv_sig, apd_indices, freq_center, freq_range,
                   num_steps, num_runs, uwave_power, state=States.HIGH)


def do_resonance_state(nv_sig, apd_indices, state):

    freq_center = nv_sig['resonance_{}'.format(state.name)]
    uwave_power = -5.0  
    
#    freq_range = 0.200
#    num_steps = 51
#    num_runs = 2
    
    # Zoom
    freq_range = 0.05
    num_steps = 51
    num_runs = 4

    resonance.main(nv_sig, apd_indices, freq_center, freq_range,
                   num_steps, num_runs, uwave_power)


def do_pulsed_resonance(nv_sig, apd_indices,
                        freq_center=2.87, freq_range=0.2):

    num_steps = 51
    num_reps = 500
    num_runs = 10
    uwave_power = 14.5
    uwave_pulse_dur = 100

    pulsed_resonance.main(nv_sig, apd_indices, freq_center, freq_range,
                          num_steps, num_reps, num_runs,
                          uwave_power, uwave_pulse_dur)


def do_pulsed_resonance_state(nv_sig, apd_indices, state):

    # freq_range = 0.150
    # num_steps = 51
    # num_reps = 10**4
    # num_runs = 8
    
    # Zoom
    freq_range = 0.035
    # freq_range = 0.120
    num_steps = 51
    num_reps = 8000
    num_runs = 3
    
    composite = False

    pulsed_resonance.state(nv_sig, apd_indices, state, freq_range,
                          num_steps, num_reps, num_runs, composite)


def do_optimize_magnet_angle(nv_sig, apd_indices):

    angle_range = [0, 150]
    # angle_range = [25, 35]
    num_angle_steps = 6
    freq_center = 2.87
    freq_range = 0.180
    num_freq_steps = 51
    num_freq_runs = 10
    
    # Pulsed
    # uwave_power = 14.5
    # uwave_pulse_dur = 100
    # num_freq_reps = 1 * 10**4
    
    # CW
    uwave_power = -5.0
    uwave_pulse_dur = None
    num_freq_reps = None

    optimize_magnet_angle.main(nv_sig, apd_indices,
               angle_range, num_angle_steps, freq_center, freq_range,
               num_freq_steps, num_freq_reps, num_freq_runs,
               uwave_power, uwave_pulse_dur)


def do_rabi(nv_sig, apd_indices, state, uwave_time_range=[0, 200]):
 
    num_steps = 51
    num_reps = 8000
    num_runs = 10

    rabi.main(nv_sig, apd_indices, uwave_time_range,
              state, num_steps, num_reps, num_runs)


def do_discrete_rabi(nv_sig, apd_indices, state, max_num_pi_pulses=4):

    # num_reps = 2 * 10**4
    num_reps = 5000
    num_runs = 10
        
    discrete_rabi.main(nv_sig, apd_indices,
                       state, max_num_pi_pulses, num_reps, num_runs)


def do_t1_battery(nv_sig, apd_indices):

    # T1 experiment parameters, formatted:
    # [[init state, read state], relaxation_time_range, num_steps, num_reps, num_runs]
    
    num_runs = 250
    num_reps = 500
    num_steps = 12
    min_tau = 20e3
    max_tau_omega = 29e6
    max_tau_gamma = 18e6
    # max_tau_omega = 1e6
    # max_tau_gamma = max_tau_omega
    t1_exp_array = numpy.array([
            [[States.ZERO, States.HIGH], [min_tau, max_tau_omega], num_steps, num_reps, num_runs],
            [[States.ZERO, States.ZERO], [min_tau, max_tau_omega], num_steps, num_reps, num_runs],
            [[States.ZERO, States.HIGH], [min_tau, max_tau_omega//3], num_steps, num_reps, num_runs],
            [[States.ZERO, States.ZERO], [min_tau, max_tau_omega//3], num_steps, num_reps, num_runs],
            [[States.HIGH, States.LOW], [min_tau, max_tau_gamma], num_steps, num_reps, num_runs],
            [[States.HIGH, States.HIGH], [min_tau, max_tau_gamma], num_steps, num_reps, num_runs],
            [[States.HIGH, States.LOW], [min_tau, max_tau_gamma//3], num_steps, num_reps, num_runs],
            [[States.HIGH, States.HIGH], [min_tau, max_tau_gamma//3], num_steps, num_reps, num_runs],
            [[States.LOW, States.HIGH], [min_tau, max_tau_gamma], num_steps, num_reps, num_runs],
            [[States.LOW, States.LOW], [min_tau, max_tau_gamma], num_steps, num_reps, num_runs],
            [[States.LOW, States.HIGH], [min_tau, max_tau_gamma//3], num_steps, num_reps, num_runs],
            [[States.LOW, States.LOW], [min_tau, max_tau_gamma//3], num_steps, num_reps, num_runs],
            ], dtype=object)

    # Loop through the experiments
    for exp_ind in range(len(t1_exp_array)):

        init_read_states = t1_exp_array[exp_ind, 0]
        relaxation_time_range = t1_exp_array[exp_ind, 1]
        num_steps = t1_exp_array[exp_ind, 2]
        num_reps = t1_exp_array[exp_ind, 3]
        num_runs = t1_exp_array[exp_ind, 4]

        t1_double_quantum.main(nv_sig, apd_indices, relaxation_time_range,
                           num_steps, num_reps, num_runs, init_read_states)


def do_t1_dq_knill_battery(nv_sig, apd_indices):

    # T1 experiment parameters, formatted:
    # [[init state, read state], relaxation_time_range, num_steps, num_reps, num_runs]
    num_runs = 150
    num_reps = 1e3
    num_steps = 2
    min_tau = 20e3
    # max_tau_omega = 29e6
    # max_tau_gamma = 18e6
    max_tau_omega = 1e6
    max_tau_gamma = max_tau_omega
    t1_exp_array = numpy.array([
            [[States.ZERO, States.HIGH], [min_tau, max_tau_omega], num_steps, num_reps, num_runs],
            [[States.ZERO, States.ZERO], [min_tau, max_tau_omega], num_steps, num_reps, num_runs],
            [[States.ZERO, States.HIGH], [min_tau, max_tau_omega//3], num_steps, num_reps, num_runs],
            [[States.ZERO, States.ZERO], [min_tau, max_tau_omega//3], num_steps, num_reps, num_runs],
            # [[States.HIGH, States.LOW], [min_tau, max_tau_gamma], num_steps, num_reps, num_runs],
            # [[States.HIGH, States.HIGH], [min_tau, max_tau_gamma], num_steps, num_reps, num_runs],
            # [[States.HIGH, States.LOW], [min_tau, max_tau_gamma//3], num_steps, num_reps, num_runs],
            # [[States.HIGH, States.HIGH], [min_tau, max_tau_gamma//3], num_steps, num_reps, num_runs],
            # [[States.LOW, States.HIGH], [min_tau, max_tau_gamma], num_steps, num_reps, num_runs],
            # [[States.LOW, States.LOW], [min_tau, max_tau_gamma], num_steps, num_reps, num_runs],
            # [[States.LOW, States.HIGH], [min_tau, max_tau_gamma//3], num_steps, num_reps, num_runs],
            # [[States.LOW, States.LOW], [min_tau, max_tau_gamma//3], num_steps, num_reps, num_runs],
            ], dtype=object)

    # Loop through the experiments
    for exp_ind in range(len(t1_exp_array)):

        init_read_states = t1_exp_array[exp_ind, 0]
        relaxation_time_range = t1_exp_array[exp_ind, 1]
        num_steps = t1_exp_array[exp_ind, 2]
        num_reps = t1_exp_array[exp_ind, 3]
        num_runs = t1_exp_array[exp_ind, 4]

        t1_dq_knill.main(nv_sig, apd_indices, relaxation_time_range,
                         num_steps, num_reps, num_runs, init_read_states)


def do_t1_interleave_knill(nv_sig, apd_indices):
    # T1 experiment parameters, formatted:
    # [[init state, read state], relaxation_time_range, num_steps, num_reps]
    num_runs = 250
    num_reps = 50
    num_steps = 12
    min_tau = 20e3
    max_tau_omega = 1540e6
    max_tau_gamma = 800e6
    t1_exp_array = numpy.array([
            [[States.ZERO, States.HIGH], [min_tau, max_tau_omega], num_steps, num_reps],
            [[States.ZERO, States.ZERO], [min_tau, max_tau_omega], num_steps, num_reps],
            [[States.ZERO, States.HIGH], [min_tau, max_tau_omega//3], num_steps, num_reps],
            [[States.ZERO, States.ZERO], [min_tau, max_tau_omega//3], num_steps, num_reps],
            [[States.HIGH, States.LOW], [min_tau, max_tau_gamma], num_steps, num_reps],
            [[States.HIGH, States.HIGH], [min_tau, max_tau_gamma], num_steps, num_reps],
            [[States.HIGH, States.LOW], [min_tau, max_tau_gamma//3], num_steps, num_reps],
            [[States.HIGH, States.HIGH], [min_tau, max_tau_gamma//3], num_steps, num_reps],
            # [[States.LOW, States.HIGH], [min_tau, max_tau_gamma], num_steps, num_reps],
            # [[States.LOW, States.LOW], [min_tau, max_tau_gamma], num_steps, num_reps],
            # [[States.LOW, States.HIGH], [min_tau, max_tau_gamma//3], num_steps, num_reps],
            # [[States.LOW, States.LOW], [min_tau, max_tau_gamma//3], num_steps, num_reps],
            ], dtype=object)

    t1_interleave_knill.main(nv_sig, apd_indices, t1_exp_array, num_runs)
    
    
def do_lifetime(nv_sig, apd_indices, filter, voltage, reference = False):
    
#    num_reps = 100 #MM 
    num_reps = 500 #SM
    num_bins = 101 
    num_runs = 10
    readout_time_range = [0, 1.0 * 10**6] #ns
    polarization_time = 60 * 10**3 #ns
    
    lifetime_v2.main(nv_sig, apd_indices, readout_time_range,
         num_reps, num_runs, num_bins, filter, voltage, polarization_time, reference)
    
    
def do_ramsey(nv_sig, apd_indices):

    detuning = 2.5  # MHz
    precession_time_range = [0, 4 * 10**3]
    num_steps = 151
    num_reps = 3 * 10**5
    num_runs = 1

    ramsey.main(nv_sig, apd_indices, detuning, precession_time_range,
                num_steps, num_reps, num_runs)


def do_spin_echo(nv_sig, apd_indices):

    # T2* in nanodiamond NVs is just a couple us at 300 K
    # In bulk it's more like 100 us at 300 K
    max_time = 120  # us
    num_steps = max_time + 1  # 1 point per us
    precession_time_range = [0, max_time * 10**3]
    num_reps = 8000
    num_runs = 20
    
#    num_steps = 151
#    precession_time_range = [0, 10*10**3]
#    num_reps = int(10.0 * 10**4)
#    num_runs = 6
    
    state = States.HIGH

    spin_echo.main(nv_sig, apd_indices, precession_time_range,
                   num_steps, num_reps, num_runs, state)


def do_sample_nvs(nv_sig_list, apd_indices):

    # g2 parameters
    run_time = 60 * 5
    diff_window = 150

    # PESR parameters
    num_steps = 101
    num_reps = 10**5
    num_runs = 5
    uwave_power = 9.0
    uwave_pulse_dur = 100

    g2 = g2_measurement.main_with_cxn
    pesr = pulsed_resonance.main_with_cxn

    with labrad.connect() as cxn:
        for nv_sig in nv_sig_list:
            g2_zero = g2(cxn, nv_sig, run_time, diff_window,
                         apd_indices[0], apd_indices[1])
            if g2_zero < 0.5:
                pesr(cxn, nv_sig, apd_indices, 2.87, 0.1, num_steps,
                     num_reps, num_runs, uwave_power, uwave_pulse_dur)


def do_test_major_routines(nv_sig, apd_indices):
    """Run this whenver you make a significant code change. It'll make sure
    you didn't break anything in the major routines.
    """

    test_major_routines.main(nv_sig, apd_indices)


# %% Run the file


if __name__ == '__main__':

    
    # %% Shared parameters


    # apd_indices = [0]
    apd_indices = [1]
    # apd_indices = [0,1]
    
    # nd = 'nd_0'
    nd = 'nd_0.5'
    # nd = 'nd_1.0'
    # nd = 'nd_2.0'
    sample_name = 'hopper'
    
    # nv_sig = { 'coords': [0.0, 0.0, 0],
    #         'name': '{}-search'.format(sample_name),
    #         'expected_count_rate': None, 'nd_filter': nd,
    #         'pulsed_readout_dur': 350, 'magnet_angle': None,
    #         'resonance_LOW': 2.87, 'rabi_LOW': 160, 'uwave_power_LOW': 14.5,
    #         'resonance_HIGH': None, 'rabi_HIGH': None, 'uwave_power_HIGH': 13.0}
    
    # nv_sig = { 'coords': [0.0, 0.0, 35],
    nv_sig = { 'coords': [0.0, 0.0, 5.0],
            'name': '{}-search'.format(sample_name),
            'disable_opt': True, 'expected_count_rate': 1000,
            'imaging_laser': 'laserglow_532', 'imaging_laser_filter': nd, 'imaging_readout_dur': 1E7,
            'spin_laser': 'laserglow_532', 'spin_laser_filter': nd, 'spin_pol_dur': 1E5, 'spin_readout_dur': 350,
            'charge_readout_laser': 'laser_589', 'charge_readout_laser_filter': nd, 'charge_readout_dur': 350,
            'NV-_pol_laser': 'laser_589', 'NV-_pol_laser_filter': nd, 'NV-_pol_dur': 350,
            'collection_filter': None, 'magnet_angle': 148,
            'resonance_LOW': 2.8036, 'rabi_LOW': 242.9, 'uwave_power_LOW': 15.5,  # 15.5 max
            'resonance_HIGH': 2.9512, 'rabi_HIGH': 219.6, 'uwave_power_HIGH': 12.0}   # 14.5 max
    
    
    # %% Functions to run

    try:
        
        # for pos in numpy.linspace(35, -50, 11, dtype=int): 
        #     with labrad.connect() as cxn:
        #         cxn.cryo_piezos.write_xy(pos, 915)
        #     # do_pulsed_resonance_state(nv_sig, apd_indices, States.LOW)
        #     # do_pulsed_resonance_state(nv_sig, apd_indices, States.HIGH)
        #     do_image_sample(nv_sig, apd_indices)
        
        # with labrad.connect() as cxn:
        #     cxn.cryo_piezos.write_xy(-770, 72)
        
        # do_image_sample(nv_sig, apd_indices)
        # do_optimize(nv_sig, apd_indices)
        # tool_belt.set_drift([0.0, 0.0, 0.0])  # Totally reset 
        # drift = tool_belt.get_drift()
        # tool_belt.set_drift([0.0, 0.0, drift[2]])  # Keep z
        # tool_belt.set_drift([drift[0], drift[1], 0.0])  # Keep xy
        # do_stationary_count(nv_sig, apd_indices)
        do_resonance(nv_sig, apd_indices, 2.87, 0.175)
        # do_pulsed_resonance(nv_sig, apd_indices, 2.87, 0.200)
        # do_pulsed_resonance_state(nv_sig, apd_indices, States.LOW)
        # do_pulsed_resonance_state(nv_sig, apd_indices, States.HIGH)
        # do_optimize_magnet_angle(nv_sig, apd_indices)
        # do_rabi(nv_sig, apd_indices, States.LOW, uwave_time_range=[0, 400])
        # do_rabi(nv_sig, apd_indices, States.HIGH, uwave_time_range=[0, 400])
        # do_discrete_rabi(nv_sig, apd_indices, States.LOW, 4)
        # do_discrete_rabi(nv_sig, apd_indices, States.HIGH, 4)
        # do_spin_echo(nv_sig, apd_indices)
        # do_g2_measurement(nv_sig, 0, 1)  # 0, (394.6-206.0)/31 = 6.084 ns, 164.3 MHz; 1, (396.8-203.6)/33 = 5.855 ns, 170.8 MHz
        # do_t1_battery(nv_sig, apd_indices)
        # do_t1_interleave_knill(nv_sig, apd_indices)
        # for i in range(4):
        #     do_t1_dq_knill_battery(nv_sig, apd_indices)
        
        # for res in numpy.linspace(2.9435, 2.9447, 7):
        #     nv_sig['resonance_HIGH'] = res
        #     do_discrete_rabi(nv_sig, apd_indices, States.HIGH, 8)
        
        # for i in range(5):
        #     do_discrete_rabi(nv_sig, apd_indices, States.HIGH, 9)
        
        # tool_belt.init_safe_stop()
        # while True:
        #     if tool_belt.safe_stop():
        #         break
        #     do_optimize(nv_sig, apd_indices)
        #     do_image_sample(nv_sig, apd_indices)
        #     time.sleep(300) 
        
        # tool_belt.init_safe_stop()
        # for z in numpy.linspace(35, -50, 18, dtype=int):  
        #     if tool_belt.safe_stop():
        #         break
        #     nv_sig['coords'][2] = int(z)
        #     do_image_sample(nv_sig, apd_indices)
        
        # Operations that don't need an NV
        # 
        # tool_belt.set_drift([0.0, 0.0, 0.0])  # Totally reset
        # tool_belt.set_drift([0.0, 0.0, tool_belt.get_drift()[2]])  # Keep z
        
        # set_xyz([0.0, 0.0 , 0])
        # set_xyz([0.454, 0.832, -88])
        
        # Routines that expect lists of NVs
#        do_optimize_list(nv_sig_list, apd_indices)
#        do_sample_nvs(nv_sig_list, apd_indices)
#        do_g2_measurement(nv_sig_list, apd_indices[0], apd_indices[1])

        # tool_belt.init_safe_stop()
        # for z in numpy.linspace(-100, 0, 11):
        #     if tool_belt.safe_stop():
        #         break
        #     nv_sig['coords'][2] = int(z)
        #     do_image_sample(nv_sig, apd_indices)
        
        # for y in numpy.linspace(2, -2, 9):
        #     nv_sig['coords'][1] = y
        #     do_image_sample(nv_sig, apd_indices)
            
        # for x in numpy.linspace(-150, 150, 5):
        #     for y in numpy.linspace(-150, 150, 5):
        #         for z in numpy.linspace(-150, 150, 5):
        #             print(tool_belt.get_time_stamp())
        #             print([x,y,z])
        
        #             with labrad.connect() as cxn:
        #                 cxn.cryo_piezos.write_xy(int(x),int(y))
        #             nv_sig['coords'][2] = int(z)
        #             do_image_sample(nv_sig, apd_indices)
        # for z in numpy.linspace(-400, 400, 5)
        # with labrad.connect() as cxn:
        #     cxn.cryo_piezos.write_z(0)
        #     cxn.cryo_piezos.write_xy(0,0)
        # with labrad.connect() as cxn:
        #     cxn.cryo_piezos.write_xy(0,0)
        

    finally:
        # Reset our hardware - this should be done in each routine, but
        # let's double check here
        tool_belt.reset_cfm()
        # Leave green on
        # with labrad.connect() as cxn:
        #     cxn.pulse_streamer.constant([3], 0.0, 0.0)
        # Kill safe stop
        if tool_belt.check_safe_stop_alive():
            print('\n\nRoutine complete. Press enter to exit.')
            tool_belt.poll_safe_stop()
