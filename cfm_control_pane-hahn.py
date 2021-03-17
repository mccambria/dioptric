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
import majorroutines.g2_measurement as g2_measurement
import majorroutines.t1_double_quantum as t1_double_quantum
import majorroutines.t1_interleave as t1_interleave
import majorroutines.ramsey as ramsey
import majorroutines.spin_echo as spin_echo
import majorroutines.lifetime as lifetime
import majorroutines.lifetime_v2 as lifetime_v2
# import majorroutines.set_drift_from_reference_image as set_drift_from_reference_image
import debug.test_major_routines as test_major_routines
from utils.tool_belt import States


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
    num_steps = 90
    # num_steps = 120
    
    # scan_range = 0.15
    # num_steps = 60
    
    # scan_range = 0.75
    # num_steps = 150
    
    # scan_range = 5.0
    # scan_range = 3.0
#    scan_range = 1.2
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
    # num_steps = 60
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

    run_time = 60 * 2  # s
    # diff_window = 400  # ns
    diff_window = 1176  # ns

    g2_measurement.main(nv_sig, run_time, diff_window,
                        apd_a_index, apd_b_index)

def do_resonance(nv_sig, apd_indices, freq_center=2.87, freq_range=0.2):

    num_steps = 51
    num_runs = 5
    uwave_power = -5.0

    resonance.main(nv_sig, apd_indices, freq_center, freq_range,
                   num_steps, num_runs, uwave_power)

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
    num_reps = 1 * 10**4
    num_runs = 8
    uwave_power = 14.5
    uwave_pulse_dur = 100

    pulsed_resonance.main(nv_sig, apd_indices, freq_center, freq_range,
                          num_steps, num_reps, num_runs,
                          uwave_power, uwave_pulse_dur)

def do_pulsed_resonance_state(nv_sig, apd_indices, state):

#    freq_range = 0.150
#    num_steps = 51
#    num_reps = 10**5
#    num_runs = 1
    
    # Zoom
    freq_range = 0.03
    num_steps = 51
    num_reps = 10**4
    num_runs = 12

    pulsed_resonance.state(nv_sig, apd_indices, state, freq_range,
                          num_steps, num_reps, num_runs)

def do_optimize_magnet_angle(nv_sig, apd_indices):

    angle_range = [0, 150]
    # angle_range = [25, 35]
    num_angle_steps = 6
    freq_center = 2.877
    freq_range = 0.12
    num_freq_steps = 51
    num_freq_runs = 10
    uwave_power = 14.5
    # uwave_power = -5.0
    uwave_pulse_dur = 100  # Set to None for CWESR
    num_freq_reps = 1 * 10**4

    optimize_magnet_angle.main(nv_sig, apd_indices,
               angle_range, num_angle_steps, freq_center, freq_range,
               num_freq_steps, num_freq_reps, num_freq_runs,
               uwave_power, uwave_pulse_dur)

def do_rabi(nv_sig, apd_indices, state, uwave_time_range=[0, 200]):

    num_steps = 51
    num_reps = 1 * 10**4
    num_runs = 12

    rabi.main(nv_sig, apd_indices, uwave_time_range,
              state, num_steps, num_reps, num_runs)

def do_t1_battery(nv_sig, apd_indices):

    # T1 experiment parameters, formatted:
    # [[init state, read state], relaxation_time_range, num_steps, num_reps, num_runs]

    t1_exp_array = numpy.array([
        [[States.HIGH, States.LOW], [0, 70*10**6], 11, 0.5*10**3, 100],
        [[States.HIGH, States.HIGH], [0, 70*10**6], 11, 0.5*10**3, 100],
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

def do_t1_interleave(nv_sig, apd_indices):
    # T1 experiment parameters, formatted:
    # [[init state, read state], relaxation_time_range, num_steps, num_reps]
    # ~18 hrs
    num_runs = 30
    t1_exp_array = numpy.array([
        [[States.HIGH, States.LOW], [0, 50*10**3], 51, 8*10**4, num_runs],
        [[States.HIGH, States.LOW], [0, 120*10**3], 26, 8*10**4, num_runs],
    
        [[States.HIGH, States.HIGH], [0, 50*10**3], 51, 8*10**4, num_runs],
        [[States.HIGH, States.HIGH], [0, 120*10**3], 26, 8*10**4, num_runs],
    
        [[States.ZERO, States.HIGH], [0, 2.5*10**6], 26, 1*10**4, num_runs],
    
        [[States.ZERO, States.ZERO], [0, 2.5*10**6], 26, 1*10**4, num_runs],
        ])

    t1_interleave.main(nv_sig, apd_indices, t1_exp_array, num_runs)
    
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

    # T2 in nanodiamond NVs without dynamical decoupling is just a couple
    # us so don't bother looking past 10s of us
    
#    num_steps = 101
#    precession_time_range = [0, 100 * 10**3]
#    num_reps = int(3.0 * 10**4)
#    num_runs = 4
    
    num_steps = 101
    precession_time_range = [0, 150 * 10**3]
    num_reps = int(1 * 10**4)
    num_runs = 16
    
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

def find_resonance_and_rabi(nv_sig, apd_indices):
    # Given resonances and rabi periods in the nv_sig, automatically remeasures
    state_list = [States.LOW, States.HIGH]
    num_steps = 51
    num_runs = 2

    fail_bool = False

    value_list = []
    for state in state_list:

        # Run resonance and save the resonance found
        num_reps = 10**5
        freq_range = 0.04

        print('Measureing pESR on {}\n'.format(state.name))
        resonance_list = pulsed_resonance.state(nv_sig, apd_indices, state, freq_range,
                              num_steps, num_reps, num_runs)
        resonance = resonance_list[0]
        value_list.append('%.4f'%resonance)

        if resonance is None:
            print('Resonance fitting failed')
            fail_bool = True
            return

        # If the resonance has shifted more than 1 MHz in either direction, stop
        shift_res = 10/1000
        limit_high_res = (nv_sig['resonance_{}'.format(state.name)] + shift_res)
        limit_low_res =  (nv_sig['resonance_{}'.format(state.name)] - shift_res)

        if resonance > limit_high_res or resonance < limit_low_res:
            print('Resonance has shifted more than {} MHz'.format(float(shift_res)))
            fail_bool = True
            return
        else:
            nv_sig['resonance_{}'.format(state.name)] = float('%.4f'%resonance)

        # Run rabi and save the rabi period
        uwave_time_range = [0, 200]
        num_reps = 2*10**5

        print('Measureing rabi on {}\n'.format(state.name))
        rabi_per = rabi.main(nv_sig, apd_indices, uwave_time_range,
                  state, num_steps, num_reps, num_runs)
        value_list.append('%.1f'%rabi_per)

        if rabi_per is None:
            print('Rabi fitting failed')
            fail_bool = True
            return

        # If the rabi period has shifted more than 50 ns in either direction, stop
        shift_per =50
        limit_high_per = (nv_sig['rabi_{}'.format(state.name)] + shift_per)
        limit_low_per =  (nv_sig['rabi_{}'.format(state.name)] - shift_per)

        if rabi_per > limit_high_per or rabi_per < limit_low_per:
            print('Rabi period has changed more than {} ns'.format(shift_per))
            fail_bool = True
            return
        else:
            nv_sig['rabi_{}'.format(state.name)] = float('%.1f'%rabi_per)

    print(value_list)

    timestamp = tool_belt.get_time_stamp()
    raw_data = {'nv_sig': nv_sig,
                'nv_sig-units': tool_belt.get_nv_sig_units(),
                'value_list': value_list,
                'value_list-units': 'GHz, ns, GHz, ns'
                }

    file_path = 'E:/Shared drives/Kolkowitz Lab Group/nvdata/auto_pESR_rabi/' + '{}-{}'.format(timestamp, nv_sig['name'])
    tool_belt.save_raw_data(raw_data, file_path)

    return fail_bool

# def do_set_drift_from_reference_image(nv_sig, apd_indices):

#     # ref_file_name = '2019-06-10_15-22-25_ayrton12'  # 60 x 60
#     ref_file_name = '2019-06-27_16-37-18_johnson1' # bulk nv, first one we saw

#     set_drift_from_reference_image.main(ref_file_name, nv_sig, apd_indices)

def do_test_major_routines(nv_sig, apd_indices):
    """Run this whenver you make a significant code change. It'll make sure
    you didn't break anything in the major routines.
    """

    test_major_routines.main(nv_sig, apd_indices)


# %% Run the file


if __name__ == '__main__':

    # %% Shared parameters

    # apd_indices = [0]  # bright, connected
    # apd_indices = [1]  # dark, disconnected
    apd_indices = [0]
    
    # nd = 'nd_0'
    # nd = 'nd_0.5'
    nd = 'nd_1.0'
    sample_name = 'hopper'
    
    # nv_sig = { 'coords': [0.0, 0.0, 0],
    #         'name': '{}-search'.format(sample_name),
    #         'expected_count_rate': None, 'nd_filter': nd,
    #         'pulsed_readout_dur': 350, 'magnet_angle': None,
    #         'resonance_LOW': 2.87, 'rabi_LOW': 160, 'uwave_power_LOW': 14.5,
    #         'resonance_HIGH': None, 'rabi_HIGH': None, 'uwave_power_HIGH': 13.0}
    
    nv_sig = { 'coords': [0.0, 0.0, 40],
            'name': '{}-nv1_2021_03_16'.format(sample_name),
            'expected_count_rate': 1000, 'nd_filter': nd, 'single': False,
            'pulsed_readout_dur': 350, 'magnet_angle': None,
            'resonance_LOW': 2.8312, 'rabi_LOW': 217.5, 'uwave_power_LOW': 14.5,
            'resonance_HIGH': 2.9216, 'rabi_HIGH': 175.2, 'uwave_power_HIGH': 12.0}
    
    # nv_sig = { 'coords': [0.1, 0.0, 70],
    #         'name': '{}-nv2_2021_03_15'.format(sample_name),
    #         'expected_count_rate': 1000, 'nd_filter': nd, 'single': False,
    #         'pulsed_readout_dur': 350, 'magnet_angle': None,
    #         'resonance_LOW': 2.8717, 'rabi_LOW': 231.6, 'uwave_power_LOW': 14.5,
    #         'resonance_HIGH': 2.8849, 'rabi_HIGH': 207.3, 'uwave_power_HIGH': 13.0}
    
    
    
    # %% Functions to run

    try:
        
        # do_image_sample(nv_sig, apd_indices)
        # do_optimize(nv_sig, apd_indices)
        # tool_belt.set_drift([0.0, 0.0, 0.0])  # Totally reset
        # drift = tool_belt.get_drift()
        # tool_belt.set_drift([0.0, 0.0, drift[2]])  # Keep z
        # tool_belt.set_drift([drift[0], drift[1], 0.0])  # Keep xy
        # do_stationary_count(nv_sig, apd_indices)
        # do_resonance(nv_sig, apd_indices, 2.87, 0.1)
        # do_pulsed_resonance(nv_sig, apd_indices, 2.877, 0.2)
        # do_pulsed_resonance_state(nv_sig, apd_indices, States.LOW)
        # do_pulsed_resonance_state(nv_sig, apd_indices, States.HIGH)
        # do_optimize_magnet_angle(nv_sig, apd_indices)
        # do_rabi(nv_sig, apd_indices, States.LOW, uwave_time_range=[0, 400])
        # do_rabi(nv_sig, apd_indices, States.HIGH, uwave_time_range=[0, 400])
        # do_spin_echo(nv_sig, apd_indices)
        # do_g2_measurement(nv_sig, 0, 1)  # 0, (394.6-206.0)/31 = 6.084 ns, 164.3 MHz; 1, (396.8-203.6)/33 = 5.855 ns, 170.8 MHz
        do_t1_battery(nv_sig, apd_indices)
        
        # with labrad.connect() as cxn:
        #     cxn.cryo_piezos.write_xy(0,0)
        
        # tool_belt.init_safe_stop()
        # while True:
        #     if tool_belt.safe_stop():
        #         break
        #     do_optimize(nv_sig, apd_indices)
        #     do_image_sample(nv_sig, apd_indices)
        #     time.sleep(300)
        
        # tool_belt.init_safe_stop()
        # for z in numpy.tile([0, -10],10):  
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
