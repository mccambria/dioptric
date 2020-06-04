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
#import majorroutines.image_sample as image_sample
import majorroutines.optimize as optimize
import majorroutines.stationary_count as stationary_count
import majorroutines.resonance as resonance
import majorroutines.pulsed_resonance as pulsed_resonance
import majorroutines.optimize_magnet_angle as optimize_magnet_angle
import majorroutines.rabi as rabi
import majorroutines.g2_measurement as g2_measurement
import majorroutines.t1_double_quantum as t1_double_quantum
import majorroutines.t1_double_quantum_scc_readout as t1_double_quantum_scc_readout
import majorroutines.t1_interleave as t1_interleave
import minorroutines.t1_image_sample as t1_image_sample
import majorroutines.ramsey as ramsey
import majorroutines.spin_echo as spin_echo
import majorroutines.lifetime as lifetime
import majorroutines.set_drift_from_reference_image as set_drift_from_reference_image
import debug.test_major_routines as test_major_routines
import minorroutines.photon_collections_under_589 as photon_collections_under_589
import minorroutines.determine_n_thresh as determine_n_thresh
import minorroutines.determine_n_thresh_with_638 as determine_n_thresh_with_638
import minorroutines.time_resolved_readout as time_resolved_readout
from utils.tool_belt import States


# %% Minor Routines


def set_xyz(nv_sig):

    with labrad.connect() as cxn:
        tool_belt.set_xyz(cxn, nv_sig)

def set_xyz_zero():

    with labrad.connect() as cxn:
        tool_belt.set_xyz_zero(cxn)


# %% Major Routines


def do_image_sample(nv_sig, aom_ao_589_pwr, apd_indices, color_ind, save_data, plot_data):
    
#    scan_range = 5.0
#    num_steps = 150
#    scan_range = 4.0
#    num_steps = 600
#    num_steps = 120
#    num_steps = 75
#    scan_range = 1.0
#    scan_range = 0.5
#    num_steps = 200
#    scan_range = 0.2
#    num_steps = 150
    scan_range = 0.1
#    num_steps = 120
#    scan_range = 0.3
#    num_steps = 90
#    scan_range = 0.05
    num_steps = 60
#    scan_range = 0.025
#    num_steps = 10
#    num_steps = 5
    
#    scan_range = 0.5 # 250
#    scan_range = 0.25 # 125
#    scan_range = 0.1 # 50
#    num_steps = int(scan_range / 0.1 * 50)
    
    # For now we only support square scans so pass scan_range twice
    image_sample.main(nv_sig, scan_range, scan_range, num_steps, 
                              aom_ao_589_pwr, apd_indices, color_ind, save_data, plot_data)
    
#def do_image_sample_SCC(nv_sig, aom_ao_589_pwr, apd_indices):
#    
#
#    scan_range = 0.2
#    num_steps = 60
#    
#    # For now we only support square scans so pass scan_range twice
#    image_sample_SCC.main(nv_sig, scan_range, scan_range, num_steps, 
#                              aom_ao_589_pwr, apd_indices)

def do_optimize(nv_sig, apd_indices, color_ind):

#    aom_power
    
    optimize.main(nv_sig, apd_indices, color_ind,
              set_to_opti_coords=False, save_data=True, plot_data=True)
    
def do_opti_z(nv_sig, apd_indices, color_ind):

#    aom_power
    optimize.opti_z(nv_sig, apd_indices, color_ind, aom_ao_589_pwr = 1.0, 
                    set_to_opti_coords=False, save_data=True, plot_data=True)

def do_optimize_list(nv_sig_list, apd_indices, color_ind):

    optimize.optimize_list(nv_sig_list, apd_indices, color_ind)

def do_stationary_count(nv_sig, aom_ao_589_pwr, apd_indices, color_ind):

    run_time = 90 * 10**9  # ns

    stationary_count.main(nv_sig, run_time, aom_ao_589_pwr, apd_indices, color_ind)

def do_g2_measurement(nv_sig, apd_a_index, apd_b_index):

    run_time = 60 * 5  # s
    diff_window = 150  # ns

    g2_measurement.main(nv_sig, run_time, diff_window,
                        apd_a_index, apd_b_index)

def do_resonance(nv_sig, apd_indices, color_ind, freq_center=2.87, freq_range=0.2):
#    # green @ 8 mW
#    num_steps = 101
#    num_runs = 5
#    uwave_power = -7.0
    
    # green @ 4 mW
    num_steps = 51
    num_runs = 12
    uwave_power = -16.0
    
    # yellow @ 40 uW
#    num_steps = 101
#    num_runs = 7
#    uwave_power = -21.0  

    resonance.main(nv_sig, apd_indices, freq_center, freq_range,
                   num_steps, num_runs, uwave_power, color_ind)

def do_resonance_state(nv_sig, apd_indices, state):

    freq_center = nv_sig['resonance_{}'.format(state.name)]
    freq_range = 0.07
    num_steps = 51
    num_runs = 4
    uwave_power = -16.0  # -7.0 for 515 nm light at 4 mW
#    uwave_power = -13.0  # -13.0 with a 1.5 ND is a good starting point
#    uwave_power = -5.0  # After inserting mixer

    resonance.main(nv_sig, apd_indices, freq_center, freq_range,
                   num_steps, num_runs, uwave_power, 532)

def do_pulsed_resonance(nv_sig, apd_indices,
                        freq_center=2.87, freq_range=0.2):

    num_steps = 51
    num_reps = 10**5
    num_runs = 1
    uwave_power = 9.0
    uwave_pulse_dur = 125

    pulsed_resonance.main(nv_sig, apd_indices, freq_center, freq_range,
                          num_steps, num_reps, num_runs,
                          uwave_power, uwave_pulse_dur)

def do_pulsed_resonance_state(nv_sig, apd_indices, state):

    freq_range = 0.05
#    num_steps = 101
#    freq_range = 0.025
    num_steps = 51
    num_reps = 5*10**3
    num_runs = 3

    pulsed_resonance.state(nv_sig, apd_indices, state, freq_range,
                          num_steps, num_reps, num_runs)

def do_optimize_magnet_angle(nv_sig, apd_indices):

    angle_range = [0, 150]
    num_angle_steps = 6
    freq_center = 2.87
    freq_range = 0.3
    num_freq_steps = 151
    num_freq_runs = 10
    uwave_power = -14.0
    uwave_pulse_dur = None  # Set to None for CWESR
    num_freq_reps = 10**5

    optimize_magnet_angle.main(nv_sig, apd_indices,
               angle_range, num_angle_steps, freq_center, freq_range,
               num_freq_steps, num_freq_reps, num_freq_runs,
               uwave_power, uwave_pulse_dur)

def do_rabi(nv_sig, apd_indices, state, uwave_time_range=[0, 200]):

    num_steps = 51
    num_reps = 10**3
    num_runs = 3

    rabi.main(nv_sig, apd_indices, uwave_time_range,
              state, num_steps, num_reps, num_runs)

def do_t1_battery(nv_sig, apd_indices):

    # T1 experiment parameters, formatted:
    # [[init state, read state], relaxation_time_range, num_steps, num_reps, num_runs]
    t1_exp_array = numpy.array([
        [[States.HIGH, States.LOW], [0, 6*10**6], 15, 2.2*10**2, 1000],
        [[States.HIGH, States.HIGH], [0, 6*10**6], 15, 2.2*10**2, 1000],
        [[States.ZERO, States.HIGH], [0, 6*10**6], 15, 2.2*10**2, 1000],
        [[States.ZERO, States.ZERO], [0, 6*10**6], 15, 2.2*10**2, 1000]
            ])


    # Loop through the experiments
    for exp_ind in range(len(t1_exp_array)):

        init_read_states = t1_exp_array[exp_ind, 0]
        relaxation_time_range = t1_exp_array[exp_ind, 1]
        num_steps = t1_exp_array[exp_ind, 2]
        num_reps = t1_exp_array[exp_ind, 3]
        num_runs = t1_exp_array[exp_ind, 4]

        t1_double_quantum.main(nv_sig, apd_indices, relaxation_time_range,
                           num_steps, num_reps, num_runs, init_read_states)
        
def do_t1_battery_scc(nv_sig, apd_indices):

    # T1 experiment parameters, formatted:
    # [[init state, read state], relaxation_time_range, num_steps, num_reps, num_runs]
    t1_exp_array = numpy.array([
        [[States.HIGH, States.LOW], [0, 3*10**6], 11, 10**3, 12],
        [[States.HIGH, States.HIGH], [0, 3*10**6], 11, 10**3, 12],
        [[States.ZERO, States.HIGH], [0, 3*10**6], 11, 10**3, 12],
        [[States.ZERO, States.ZERO], [0, 3*10**6], 11, 10**3, 12]
            ])


    # Loop through the experiments
    for exp_ind in range(len(t1_exp_array)):

        init_read_states = t1_exp_array[exp_ind, 0]
        relaxation_time_range = t1_exp_array[exp_ind, 1]
        num_steps = t1_exp_array[exp_ind, 2]
        num_reps = t1_exp_array[exp_ind, 3]
        num_runs = t1_exp_array[exp_ind, 4]

        t1_double_quantum_scc_readout.main(nv_sig, apd_indices, relaxation_time_range,
                           num_steps, num_reps, num_runs, init_read_states)

def do_t1_interleave(nv_sig, apd_indices):
    # T1 experiment parameters, formatted:
    # [[init state, read state], relaxation_time_range, num_steps, num_reps]
    # ~18 hrs
    num_runs = 40
    t1_exp_array = numpy.array([
                        [[States.HIGH, States.LOW], [0, 10*10**3], 51, 8*10**4, num_runs],
                        [[States.HIGH, States.LOW], [0, 50*10**3], 51, 8*10**4, num_runs],
                        [[States.HIGH, States.LOW], [0, 150*10**3], 26, 8*10**4, num_runs],
                        [[States.HIGH, States.HIGH], [0, 10*10**3], 51, 8*10**4, num_runs],
                        [[States.HIGH, States.HIGH], [0, 50*10**3], 51, 8*10**4, num_runs],
                        [[States.HIGH, States.HIGH], [0, 150*10**3], 26, 8*10**4, num_runs],
                        [[States.ZERO, States.HIGH], [0, 2*10**6], 26, 1.3*10**4, num_runs],
                        [[States.ZERO, States.ZERO], [0, 2*10**6], 26, 1.3*10**4, num_runs]
                        ])

    t1_interleave.main(nv_sig, apd_indices, t1_exp_array, num_runs)
    
def do_t1_image_sample(nv_sig, apd_indices):
    
    scan_range = 0.5
    num_steps = 5
    relaxation_time_point = 1*10**6
    
    t1_image_sample.main(nv_sig, scan_range, num_steps, relaxation_time_point, apd_indices)

def do_lifetime(nv_sig, apd_indices):
    
    num_reps = 10**4
    num_runs = 40
    relaxation_time_range = [0, 1.2*10**6]
    num_steps = 41
    
#    num_reps = 10**4
#    num_runs = 1
#    relaxation_time_range = [10000, 10000]
#    num_steps = 2
    
    
    lifetime. main(nv_sig, apd_indices, relaxation_time_range,
         num_steps, num_reps, num_runs)
    
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
#    precession_time_range = [0, 100 * 10**3]
    precession_time_range = [0, 100 * 10**3]
    num_steps = 101
    num_reps = int(3.0 * 10**4)
    num_runs = 6
    state = States.LOW

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
        freq_range = 0.018

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
        uwave_time_range = [0, 400]
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

def do_set_drift_from_reference_image(nv_sig, apd_indices):

    # ref_file_name = '2019-06-10_15-22-25_ayrton12'  # 60 x 60
    ref_file_name = '2019-06-27_16-37-18_johnson1' # bulk nv, first one we saw

    set_drift_from_reference_image.main(ref_file_name, nv_sig, apd_indices)

def do_test_major_routines(nv_sig, apd_indices):
    """Run this whenver you make a significant code change. It'll make sure
    you didn't break anything in the major routines.
    """

    test_major_routines.main(nv_sig, apd_indices)

def do_photon_collections_under_589(nv_sig, apd_indices):
    #"collect photons for tR at fixed power P and return a probability distribution"
    num_runs = 1
    num_reps = 10
    readout_time = 8 * 10**6
    aom_ao_589_pwr = 0.1 #V
    photon_collections_under_589.main(nv_sig, apd_indices, aom_ao_589_pwr, readout_time, num_runs, num_reps)
    
def do_determine_n_thresh(nv_sig, aom_ao_589_pwr, readout_time, apd_indices):
    
    num_runs = 1
    num_reps = 1* 10**3
    
    determine_n_thresh.main(nv_sig, apd_indices, aom_ao_589_pwr, readout_time, num_runs, num_reps)
    
def do_determine_n_thresh_with_638(nv_sig, apd_indices):
    
    num_reps = 500
    
    determine_n_thresh_with_638.main(nv_sig, apd_indices, num_reps)
    
def do_time_resolved_readout(nv_sig, apd_indices,
                                 init_color_ind, illum_color_ind):
#    illumination_time = 500 # turns on at 250 and turns off at 750
#    num_reps = 10**5
#    num_bins = 500
    
#    illumination_time = 1000 # turns on at 250 and turns off at 750
#    num_reps = 10**5
#    num_bins = 1000
    
#    illumination_time = 10**4 
#    num_reps = 10**4
#    num_bins = 1000
    
#    illumination_time = 10**6 
#    num_reps = 10**3
#    num_bins = 500

    illumination_time = 1*10**6 
    num_reps = 10**3
#    num_reps = 10**4
    num_bins = 500
    
    # 1
#    illumination_time = 15*10**6    
#    num_reps = 2*10**2
#    num_bins = 1500
    
    # 2
#    illumination_time = 10*10**6    
#    num_reps = 10**3
#    num_bins = 1000
    
    # 3
#    illumination_time = 5*10**6    
#    num_reps = 10**3
#    num_bins = 1000
    
    
    init_pulse_duration = 10**4
    num_runs = 1
    time_resolved_readout.main(nv_sig, apd_indices, 
                   illumination_time, init_pulse_duration,
                   init_color_ind, illum_color_ind,
                   num_reps, num_runs, num_bins)

    
# %% Run the file


if __name__ == '__main__':

    # %% Shared parameters

    apd_indices = [0]
#    apd_indices = [0, 1]
    
    sample_name = 'bachman-2'

    search_1 = { 'coords':[2.512, -1.617, 4.86],
            'name': '{}-search'.format(sample_name),
            'expected_count_rate': None, 'nd_filter': 'nd_0',
            'pulsed_readout_dur': 300,
            'pulsed_SCC_readout_dur': 1*10**7, 'am_589_power': 0.25, 
            'pulsed_initial_ion_dur': 25*10**3,
            'pulsed_shelf_dur': 200, 
            'am_589_shelf_power': 0.35,
            'pulsed_ionization_dur': 500, 'cobalt_638_power': 160, 
            'pulsed_reionization_dur': 100*10**3, 'cobalt_532_power': 8, 
            'magnet_angle': 0,
            "resonance_LOW": 2.754,"rabi_LOW": 180.0, "uwave_power_LOW": 9.0,
            "resonance_HIGH": 2.9877,"rabi_HIGH": 299.2,"uwave_power_HIGH": 10.0} 

    search_2 = { 'coords':[2.518, -1.287, 4.91],
            'name': '{}-search'.format(sample_name),
            'expected_count_rate': None, 'nd_filter': 'nd_0',
            'pulsed_readout_dur': 300,
            'pulsed_SCC_readout_dur': 1*10**7, 'am_589_power': 0.25, 
            'pulsed_initial_ion_dur': 25*10**3,
            'pulsed_shelf_dur': 200, 
            'am_589_shelf_power': 0.35,
            'pulsed_ionization_dur': 500, 'cobalt_638_power': 160, 
            'pulsed_reionization_dur': 100*10**3, 'cobalt_532_power': 8, 
            'magnet_angle': 0,
            "resonance_LOW": 2.754,"rabi_LOW": 180.0, "uwave_power_LOW": 9.0,
            "resonance_HIGH": 2.9877,"rabi_HIGH": 299.2,"uwave_power_HIGH": 10.0}   

    search_3 = { 'coords':[2.082, -1.419, 4.86],
            'name': '{}-search'.format(sample_name),
            'expected_count_rate': None, 'nd_filter': 'nd_0',
            'pulsed_readout_dur': 300,
            'pulsed_SCC_readout_dur': 1*10**7, 'am_589_power': 0.25, 
            'pulsed_initial_ion_dur': 25*10**3,
            'pulsed_shelf_dur': 200, 
            'am_589_shelf_power': 0.35,
            'pulsed_ionization_dur': 500, 'cobalt_638_power': 160, 
            'pulsed_reionization_dur': 100*10**3, 'cobalt_532_power': 8, 
            'magnet_angle': 0,
            "resonance_LOW": 2.754,"rabi_LOW": 180.0, "uwave_power_LOW": 9.0,
            "resonance_HIGH": 2.9877,"rabi_HIGH": 299.2,"uwave_power_HIGH": 10.0}       
    
    ensemble_B1 = { 'coords':[ -0.439,1.4,5.04],
            'name': '{}-B1'.format(sample_name),
            'expected_count_rate': 6600, 'nd_filter': 'nd_0',
            'pulsed_readout_dur': 300,
            'pulsed_SCC_readout_dur': 1*10**7, 'am_589_power': 0.25, 
            'pulsed_initial_ion_dur': 25*10**3,
            'pulsed_shelf_dur': 200, 
            'am_589_shelf_power': 0.35,
            'pulsed_ionization_dur': 500, 'cobalt_638_power': 160, 
            'pulsed_reionization_dur': 100*10**3, 'cobalt_532_power': 8, 
            'magnet_angle': 0,
            "resonance_LOW": 2.754,"rabi_LOW": 180.0, "uwave_power_LOW": 9.0,
            "resonance_HIGH": 2.9877,"rabi_HIGH": 299.2,"uwave_power_HIGH": 10.0} 
    
    
    nv_2 = { 'coords': [0.889, -0.132, 4.89],
            'name': '{}-nv_2'.format(sample_name),
            'expected_count_rate': None, 'nd_filter': 'nd_0',
            'pulsed_readout_dur': 300,
            'pulsed_SCC_readout_dur': 1*10**7, 'am_589_power': 0.25, 
            'pulsed_initial_ion_dur': 25*10**3,
            'pulsed_shelf_dur': 200, 
            'am_589_shelf_power': 0.35,
            'pulsed_ionization_dur': 500, 'cobalt_638_power': 160, 
            'pulsed_reionization_dur': 100*10**3, 'cobalt_532_power': 8, 
            'magnet_angle': 0,
            "resonance_LOW": 2.754,"rabi_LOW": 203.8, "uwave_power_LOW": 9.0,
            "resonance_HIGH": 2.988,"rabi_HIGH": 299.2,"uwave_power_HIGH": 10.0} 

    ensemble_A6 = { 'coords': [1.519, -0.690, 4.99],
            'name': '{}-A6'.format(sample_name),
            'expected_count_rate': None, 'nd_filter': 'nd_0',
            'pulsed_readout_dur': 300,
            'pulsed_SCC_readout_dur': 1*10**7, 'am_589_power': 0.25, 
            'pulsed_initial_ion_dur': 25*10**3,
            'pulsed_shelf_dur': 200, 
            'am_589_shelf_power': 0.35,
            'pulsed_ionization_dur': 500, 'cobalt_638_power': 160, 
            'pulsed_reionization_dur': 100*10**3, 'cobalt_532_power': 8, 
            'magnet_angle': 0,
            "resonance_LOW": 2.754,"rabi_LOW": 203.8, "uwave_power_LOW": 9.0,
            "resonance_HIGH": 2.988,"rabi_HIGH": 299.2,"uwave_power_HIGH": 10.0} 

    ensemble_B6 = { 'coords': [1.714, 0.012, 5.01],
            'name': '{}-B6'.format(sample_name),
            'expected_count_rate': None, 'nd_filter': 'nd_0',
            'pulsed_readout_dur': 300,
            'pulsed_SCC_readout_dur': 1*10**7, 'am_589_power': 0.25, 
            'pulsed_initial_ion_dur': 25*10**3,
            'pulsed_shelf_dur': 200, 
            'am_589_shelf_power': 0.35,
            'pulsed_ionization_dur': 500, 'cobalt_638_power': 160, 
            'pulsed_reionization_dur': 100*10**3, 'cobalt_532_power': 8, 
            'magnet_angle': 0,
            "resonance_LOW": 2.754,"rabi_LOW": 203.8, "uwave_power_LOW": 9.0,
            "resonance_HIGH": 2.988,"rabi_HIGH": 299.2,"uwave_power_HIGH": 10.0}       
  
    nv_sig_list = [search_2]
    
    
    aom_ao_589_pwr = 0.25
#    aom_ao_589_pwr_list = numpy.linspace(0.1, 0.7, 13)
#    cobalt_638_power = 30
#    ao_638_pwr_list = numpy.linspace(0.71, 0.9, 20)
#    color_ind = 532
#    readout_time = 100*10**3

    # %% Functions to run

    try:
        
        # Operations that don't need an NV
        
#        tool_belt.set_drift([0.0, 0.0, 0.0])  # Totally reset
#        tool_belt.set_drift([0.0, 0.0, tool_belt.get_drift()[2]])  # Keep z
        
#        set_xyz([0.0,0.0,5.0])
#        set_xyz([-0.116, -0.073, 2.61])

      
#        with labrad.connect() as cxn:
#            cxn.filter_slider_ell9k.set_filter('nd_0')           
#            cxn.pulse_streamer.constant([], 0.0, 0.0)
#            cxn.objective_piezo.write(5.1)
#            input('Laser currently turned off, Press enter to stop...')
        
        # Routines that expect lists of NVs
#        do_optimize_list(nv_sig_list, a pd_indices)
#        do_sample_nvs(nv_sig_list, apd_indices)
#        do_g2_measurement(nv_sig_list, apd_indices[0], apd_indices[1])

        
        # Routines that expect single NVs
        for ind in range(len(nv_sig_list)):
            nv_sig = nv_sig_list[ind]
            with labrad.connect() as cxn:
                cxn.filter_slider_ell9k.set_filter(nv_sig['nd_filter'])
#     
#            for image_z in numpy.linspace(4.8, 5.5, 8):
#                    nv_sig_copy = copy.deepcopy(nv_sig)
#                    coords = nv_sig_copy['coords']
#                    nv_sig_copy['coords'] = [coords[0], coords[1], image_z]                
#                    do_image_sample(nv_sig_copy, aom_ao_589_pwr, apd_indices, 532, save_data=True, plot_data=True)       

#            do_photon_collections_under_589(nv_sig, apd_indices)
#            do_determine_n_thresh(nv_sig, aom_ao_589_pwr, readout_time, apd_indices)
#            do_determine_n_thresh_with_638(nv_sig, apd_indices)
#            for p in range(len(aom_ao_589_pwr_list)):
#                aom_ao_589_pwr = aom_ao_589_pwr_list[p]
#                print(aom_ao_589_pwr)
#            do_time_resolved_readout(nv_sig, apd_indices,
#                             532, 589)
            
#            do_optimize(nv_sig, apd_indices, 532)
#            do_opti_z(nv_sig, apd_indices, 532)
            do_image_sample(nv_sig, aom_ao_589_pwr, apd_indices, 532, save_data=True, plot_data=True)
#            do_stationary_count(nv_sig, aom_ao_589_pwr, apd_indices, 589)                    

#            do_image_sample(nv_sig, aom_ao_589_pwr, apd_indices, 638, save_data=True, plot_data=True)
#            with labrad.connect() as cxn:  
#                adj_coords = (numpy.array(nv_sig['coords']) + \
#                          numpy.array(tool_belt.get_drift())).tolist()
#                x_center, y_center, z_center = adj_coords
#                tool_belt.set_xyz(cxn, [x_center, y_center, z_center])
#                cxn.pulse_streamer.constant([7], 0.0, 0.0)
##                time.sleep(5)
##                cxn.pulse_streamer.constant([], 0.0, 0.0)
#                
#                cxn.pulse_streamer.constant([3], 0.0, 0.0)
#                input()
#                time.sleep(5*60)
#                cxn.pulse_streamer.constant([], 0.0, 0.0)
# 
#            do_image_sample(nv_sig, aom_ao_589_pwr, apd_indices, 589, save_data=True, plot_data=True) 
                
#            do_g2_measurement(nv_sig, apd_indices[0], apd_indices[1])
#            do_optimize_magnet_angle(nv_sig, apd_indices)
#            do_resonance(nv_sig, apd_indices, 532)
#            do_resonance(nv_sig, apd_indices, 532, freq_center=2.878, freq_range=0.1)
#            do_resonance_state(nv_sig, apd_indices, States.LOW)
#            do_resonance_state(nv_sig, apd_indices, States.HIGH)
#            do_pulsed_resonance(nv_sig, apd_indices)
#            do_pulsed_resonance(nv_sig, apd_indices, freq_center=2.9406, freq_range=0.05)
#            do_pulsed_resonance_state(nv_sig, apd_indices, States.LOW)
#            do_pulsed_resonance_state(nv_sig, apd_indices, States.HIGH)
#            do_rabi(nv_sig, apd_indices, States.LOW, [0, 200])
#            do_rabi(nv_sig, apd_indices, States.HIGH, [0, 300])
#            find_resonance_and_rabi(nv_sig, apd_indices)
#            do_t1_battery(nv_sig, apd_indices)
#            do_t1_battery_scc(nv_sig, apd_indices)
#            do_t1_interleave(nv_sig, apd_indices)
#            do_t1_image_sample(nv_sig, apd_indices)
#            do_lifetime(nv_sig, apd_indices)
#            find_resonance_and_rabi(nv_sig, apd_indices)
            
#            fail_bool = find_resonance_and_rabi(nv_sig, apd_indices)
#            if fail_bool == True:
#                print('Failed to record pESR and Rabi')
#                break
#            else:
#                do_t1_battery(nv_sig, apd_indices)
            
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
