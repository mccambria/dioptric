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
import matplotlib.pyplot as plt
import majorroutines.image_sample as image_sample
import majorroutines.image_sample_xz as image_sample_xz
import chargeroutines.image_sample_charge_state_compare as image_sample_charge_state_compare
import majorroutines.optimize as optimize
import majorroutines.stationary_count as stationary_count
import majorroutines.resonance as resonance
import majorroutines.pulsed_resonance as pulsed_resonance
import majorroutines.optimize_magnet_angle as optimize_magnet_angle
import majorroutines.rabi as rabi
import majorroutines.g2_measurement as g2_measurement
import majorroutines.ramsey as ramsey
import majorroutines.spin_echo as spin_echo
import majorroutines.lifetime_v2 as lifetime_v2
import minorroutines.time_resolved_readout as time_resolved_readout
import chargeroutines.SPaCE as SPaCE
import chargeroutines.scc_pulsed_resonance as scc_pulsed_resonance
import chargeroutines.scc_spin_echo as scc_spin_echo
import chargeroutines.super_resolution_pulsed_resonance as super_resolution_pulsed_resonance
import chargeroutines.super_resolution_ramsey as super_resolution_ramsey
import chargeroutines.super_resolution_spin_echo as super_resolution_spin_echo
import chargeroutines.g2_measurement as g2_SCC_branch

# import majorroutines.set_drift_from_reference_image as set_drift_from_reference_image
import debug.test_major_routines as test_major_routines
from utils.tool_belt import States
import time


# %% Major Routines


def do_image_sample(nv_sig, apd_indices):

    # scan_range = 0.5
    # num_steps = 90
    # num_steps = 120
    #
    # scan_range = 0.15
    # num_steps = 60
    #
    # scan_range = 0.75
    # num_steps = 150
    
    # scan_range = 2
    # num_steps = 160
    # scan_range =.5
    # num_steps = 90
    # scan_range = 0.5
    # num_steps = 120
    # scan_range = 0.05
    # num_steps = 60
    # 80 um / V
    # 
    # scan_range = 5.0
    # scan_range = 2.5
    # scan_range = 1.7
    # scan_range =4
    # scan_range = 3
    # scan_range = 0.5
    # scan_range = 0.4
    # scan_range = 0.25
    # scan_range = 0.2
    # scan_range = 0.15
    # scan_range = 0.1
    scan_range = 0.04
    # scan_range = 0.025
    # scan_range = 0.01
    
    # num_steps = 400
    # num_steps = 300
    # num_steps = 200
    # num_steps = 160
    # num_steps = 135
    # num_steps =120
    # num_steps = 90
    num_steps = 60
    # num_steps = 31
    # num_steps = 15
    
    #individual line pairs:
    # scan_range = 0.16
    # num_steps = 160
    
    #both line pair sets:
    # scan_range = 0.35
    # num_steps = 160
        

    # For now we only support square scans so pass scan_range twice
    image_sample.main(nv_sig, scan_range, scan_range, num_steps, apd_indices)


def do_image_sample_xz(nv_sig, apd_indices):

    scan_range_x = .1
# z code range 3 to 7 if centered at 5
    scan_range_z =2
    num_steps = 60

    image_sample_xz.main(
        nv_sig,
        scan_range_x,
        scan_range_z,
        num_steps,
        apd_indices,
        um_scaled=False,
    )


def do_image_charge_states(nv_sig, apd_indices):

    scan_range = 0.01

    num_steps = 31
    num_reps= 10
    
    image_sample_charge_state_compare.main(
        nv_sig, scan_range, scan_range, num_steps,num_reps, apd_indices
    )


def do_optimize(nv_sig, apd_indices):

    optimize.main(
        nv_sig,
        apd_indices,
        set_to_opti_coords=False,
        save_data=True,
        plot_data=True,
    )


def do_optimize_list(nv_sig_list, apd_indices):

    optimize.optimize_list(nv_sig_list, apd_indices)


def do_opti_z(nv_sig_list, apd_indices):

    optimize.opti_z(
        nv_sig_list,
        apd_indices,
        set_to_opti_coords=False,
        save_data=True,
        plot_data=True,
    )


def do_stationary_count(nv_sig, apd_indices):

    run_time = 1 * 60 * 10 ** 9  # ns

    stationary_count.main(nv_sig, run_time, apd_indices)


def do_g2_measurement(nv_sig, apd_a_index, apd_b_index):

    run_time = 5*60  # s
    diff_window =15  # ns

    # g2_measurement.main(
    g2_SCC_branch.main(
        nv_sig, run_time, diff_window, apd_a_index, apd_b_index
    )


def do_resonance(nv_sig, opti_nv_sig,apd_indices, freq_center=2.87, freq_range=0.2):

    num_steps = 11#101
    num_runs = 2#15
    uwave_power = -10.0

    resonance.main(
        nv_sig,
        apd_indices,
        freq_center,
        freq_range,
        num_steps,
        num_runs,
        uwave_power,
        state=States.HIGH,
        opti_nv_sig = opti_nv_sig
    )


def do_resonance_state(nv_sig, opti_nv_sig, apd_indices, state):

    freq_center = nv_sig["resonance_{}".format(state.name)]
    uwave_power = -10.0

    freq_range = 0.1
    num_steps = 51
    num_runs = 10

    # Zoom
    # freq_range = 0.060
    # num_steps = 51
    # num_runs = 10

    resonance.main(
        nv_sig,
        apd_indices,
        freq_center,
        freq_range,
        num_steps,
        num_runs,
        uwave_power,
        opti_nv_sig = opti_nv_sig
    )


def do_pulsed_resonance(nv_sig, opti_nv_sig, apd_indices, freq_center=2.87, freq_range=0.2):

    num_steps =101
    num_reps = 1e4
    num_runs = 5
    uwave_power = 14.5
    uwave_pulse_dur = int(100/2)

    pulsed_resonance.main(
        nv_sig,
        apd_indices,
        freq_center,
        freq_range,
        num_steps,
        num_reps,
        num_runs,
        uwave_power,
        uwave_pulse_dur,
        opti_nv_sig = opti_nv_sig
    )


def do_pulsed_resonance_state(nv_sig, opti_nv_sig,apd_indices, state):

    # freq_range = 0.150
    # num_steps = 51
    # num_reps = 10**4
    # num_runs = 8

    # Zoom
    freq_range = 0.05
    # freq_range = 0.120
    num_steps = 51
    num_reps = int(0.5e4)
    num_runs = 5

    composite = False

    res, _ = pulsed_resonance.state(
        nv_sig,
        apd_indices,
        state,
        freq_range,
        num_steps,
        num_reps,
        num_runs,
        composite,
        opti_nv_sig = opti_nv_sig
    )
    nv_sig["resonance_{}".format(state.name)] = res


def do_optimize_magnet_angle(nv_sig, apd_indices):

    # angle_range = [132, 147]
    #    angle_range = [315, 330]
    num_angle_steps = 6
    #    freq_center = 2.7921
    #    freq_range = 0.060
    angle_range = [0, 150]
    #    num_angle_steps = 6
    freq_center = 2.87
    freq_range = 0.3
    num_freq_steps = 101
    num_freq_runs = 10

    # Pulsed
    uwave_power = 14.5
    uwave_pulse_dur = 100/2
    num_freq_reps = int(1e4)

    # CW
    #uwave_power = -10.0
    #uwave_pulse_dur = None
    #num_freq_reps = None

    optimize_magnet_angle.main(
        nv_sig,
        apd_indices,
        angle_range,
        num_angle_steps,
        freq_center,
        freq_range,
        num_freq_steps,
        num_freq_reps,
        num_freq_runs,
        uwave_power,
        uwave_pulse_dur,
    )


def do_rabi(nv_sig, opti_nv_sig, apd_indices, state, uwave_time_range=[0, 200]):

    num_steps = 51
    num_reps = int(1e4)
    num_runs = 5

    period = rabi.main(
        nv_sig,
        apd_indices,
        uwave_time_range,
        state,
        num_steps,
        num_reps,
        num_runs,
        opti_nv_sig = opti_nv_sig
    )
    nv_sig["rabi_{}".format(state.name)] = period




def do_lifetime(nv_sig, apd_indices, filter, voltage, reference=False):

    #    num_reps = 100 #MM
    num_reps = 500  # SM
    num_bins = 101
    num_runs = 5
    readout_time_range = [0, 1.0 * 10 ** 6]  # ns
    polarization_time = 60 * 10 ** 3  # ns

    lifetime_v2.main(
        nv_sig,
        apd_indices,
        readout_time_range,
        num_reps,
        num_runs,
        num_bins,
        filter,
        voltage,
        polarization_time,
        reference,
    )

    

def do_ramsey(nv_sig, opti_nv_sig, apd_indices):

    detuning = 10  # MHz
    precession_time_range = [0, 2 * 10 ** 3]
    num_steps = 101
    num_reps = int( 10 ** 4)
    num_runs = 6

    ramsey.main(
        nv_sig,
        apd_indices,
        detuning,
        precession_time_range,
        num_steps,
        num_reps,
        num_runs,
        opti_nv_sig = opti_nv_sig
    )


def do_spin_echo(nv_sig, apd_indices):

    # T2* in nanodiamond NVs is just a couple us at 300 K
    # In bulk it's more like 100 us at 300 K
    # max_time = 40  # us
    num_steps = int(20*2 + 1)  # 1 point per us
    #    num_steps = int(max_time/2) + 1  # 2 point per us
    #    max_time = 1  # us
    #    num_steps = 51
    precession_time_range = [20e3, 40 * 10 ** 3]
    #    num_reps = 8000
    #    num_runs = 5
    num_reps = 1000
    num_runs = 40

    #    num_steps = 151
    #    precession_time_range = [0, 10*10**3]
    #    num_reps = int(10.0 * 10**4)
    #    num_runs = 6

    state = States.LOW

    angle = spin_echo.main(
        nv_sig,
        apd_indices,
        precession_time_range,
        num_steps,
        num_reps,
        num_runs,
        state,
    )
    return angle



def do_time_resolved_readout(nv_sig, apd_indices):

    # nv_sig uses the initialization key for the first pulse
    # and the imaging key for the second
    
    num_reps = 1000
    num_bins = 2001
    num_runs = 20
    # disp = 0.0001#.05

    bin_centers, binned_samples_sig = time_resolved_readout.main(
        nv_sig, 
        apd_indices, 
        num_reps, 
        num_runs, 
        num_bins
    )
    return bin_centers, binned_samples_sig
    
def do_time_resolved_readout_three_pulses(nv_sig, apd_indices):

    # nv_sig uses the initialization key for the first pulse
    # and the imaging key for the second
    
    num_reps = 1000
    num_bins = 2001
    num_runs = 20
    

    bin_centers, binned_samples_sig = time_resolved_readout.main_three_pulses(
        nv_sig, 
        apd_indices, 
        num_reps, 
        num_runs, 
        num_bins
    )   
    
    return bin_centers, binned_samples_sig
    


def do_SPaCE(nv_sig, opti_nv_sig, apd_indices,num_runs, num_steps_a, num_steps_b,
               img_range_1D, img_range_2D, offset, charge_state_threshold = None):
    # dr = 0.025 / numpy.sqrt(2)
    # img_range = [[-dr,-dr],[dr, dr]] #[[x1, y1], [x2, y2]]
    # num_steps = 101
    # num_runs = 50
    # measurement_type = "1D"

    # img_range = 0.075
    # num_steps = 71
    # num_runs = 1
    # measurement_type = "2D"

    # dz = 0
    SPaCE.main(nv_sig, opti_nv_sig, apd_indices,num_runs, num_steps_a, num_steps_b,
               charge_state_threshold, img_range_1D, img_range_2D, offset )

def do_scc_resonance(nv_sig, opti_nv_sig, apd_indices, state=States.LOW):
    freq_center = nv_sig['resonance_{}'.format(state.name)]
    uwave_power = nv_sig['uwave_power_{}'.format(state.name)]
    uwave_pulse_dur = nv_sig['rabi_{}'.format(state.name)]/2
    
    freq_range = 0.05
    num_steps = 51
    num_reps = int(10**3)
    num_runs = 30
    
    scc_pulsed_resonance.main(nv_sig, opti_nv_sig, apd_indices, freq_center, freq_range,
         num_steps, num_reps, num_runs, uwave_power, uwave_pulse_dur, state )
    
def do_scc_spin_echo(nv_sig, opti_nv_sig, apd_indices, tau_start, tau_stop, state=States.LOW):
    step_size = 1 # us
    num_steps = int((tau_stop - tau_start)/step_size + 1)
    
    precession_time_range = [tau_start *1e3, tau_stop *1e3]
    
    num_reps = int(10**3)
    num_runs = 40
    
    scc_spin_echo.main(nv_sig, opti_nv_sig, apd_indices, precession_time_range,
         num_steps, num_reps, num_runs,  
         state )
    
    
    
def do_super_resolution_resonance(nv_sig, opti_nv_sig, apd_indices, state=States.LOW):
    freq_center = nv_sig['resonance_{}'.format(state.name)]
    uwave_power = nv_sig['uwave_power_{}'.format(state.name)]
    uwave_pulse_dur = nv_sig['rabi_{}'.format(state.name)]/2
    
    freq_range = 0.05
    num_steps = 51
    num_reps = int(10**3)
    num_runs = 30
    
    super_resolution_pulsed_resonance.main(nv_sig, opti_nv_sig, apd_indices, freq_center, freq_range,
         num_steps, num_reps, num_runs, uwave_power, uwave_pulse_dur, state )
    
def do_super_resolution_ramsey(nv_sig, opti_nv_sig, apd_indices,
                                  tau_start, tau_stop, state=States.LOW):
    
    detuning = 5  # MHz
    
    # step_size = 0.05 # us
    # num_steps = int((tau_stop - tau_start)/step_size + 1)
    num_steps = 101
    precession_time_range = [tau_start *1e3, tau_stop *1e3]
    
    
    num_reps = int(10**3)
    num_runs = 30
    
    super_resolution_ramsey.main(nv_sig, opti_nv_sig, apd_indices, 
                                    precession_time_range, detuning,
         num_steps, num_reps, num_runs, state )
    
def do_super_resolution_spin_echo(nv_sig, opti_nv_sig, apd_indices,
                                  tau_start, tau_stop, state=States.LOW):
    step_size = 1 # us
    num_steps = int((tau_stop - tau_start)/step_size + 1)
    print(num_steps)
    precession_time_range = [tau_start *1e3, tau_stop *1e3]
    
    
    num_reps = int(10**3)
    num_runs = 20
    
    super_resolution_spin_echo.main(nv_sig, opti_nv_sig, apd_indices, 
                                    precession_time_range,
         num_steps, num_reps, num_runs, state )

def do_sample_nvs(nv_sig_list, apd_indices):

    # g2 parameters
    run_time = 60 * 5
    diff_window = 150

    # PESR parameters
    num_steps = 101
    num_reps = 10 ** 5
    num_runs = 3
    uwave_power = 9.0
    uwave_pulse_dur = 100

    g2 = g2_measurement.main_with_cxn
    pesr = pulsed_resonance.main_with_cxn

    with labrad.connect() as cxn:
        for nv_sig in nv_sig_list:
            g2_zero = g2(
                cxn,
                nv_sig,
                run_time,
                diff_window,
                apd_indices[0],
                apd_indices[1],
            )
            if g2_zero < 0.5:
                pesr(
                    cxn,
                    nv_sig,
                    apd_indices,
                    2.87,
                    0.1,
                    num_steps,
                    num_reps,
                    num_runs,
                    uwave_power,
                    uwave_pulse_dur,
                )


def do_test_major_routines(nv_sig, apd_indices):
    """Run this whenver you make a significant code change. It'll make sure
    you didn't break anything in the major routines.
    """

    test_major_routines.main(nv_sig, apd_indices)


# %% Run the file


if __name__ == "__main__":

    # In debug mode, don't bother sending email notifications about exceptions
    debug_mode = True

    # %% Shared parameters

    # apd_indices = [0]
    apd_indices = [1]
    # apd_indices = [0,1]

    nd_yellow = "nd_1.0"
    green_power =10
    red_power = 10
    sample_name = "sandia"
    green_laser = "integrated_520"#"cobolt_515"
    yellow_laser = "laserglow_589"
    red_laser = "cobolt_638"

    nv_sig_search = {
        "coords":[0.330, 0.395,6.836], #  
        # "coords":[-0.133, 0.491,6.836], 
        "name": "{}-search".format(sample_name),
        "disable_opt": False,
        "ramp_voltages": False,
        "expected_count_rate": None,
        "correction_collar": 0.17,
        
        
        # "imaging_laser": red_laser,
        # "imaging_laser_power": 0.565,
        # "imaging_readout_dur": 1e7,
        
        
        # "imaging_laser":green_laser,
        # "imaging_laser_power": green_power,
        # "imaging_readout_dur": 1e7,
        
        # "imaging_laser":red_laser,
        # "imaging_laser_power": red_power,
        # "imaging_readout_dur": 1e7,
        
        
        "collection_filter": "715_lp",
        "magnet_angle": None,
        "resonance_LOW": 2.8012,
        "rabi_LOW": 141.5,
        "uwave_power_LOW": 15.5,  # 15.5 max
        "resonance_HIGH": 2.9445,
        "rabi_HIGH": 191.9,
        "uwave_power_HIGH": 13,
    }  # 14.5 max

    
    
    nv_sig = {  
        # "coords":[-0.699, -0.178, 6.17],#a6  , center    
        # "coords": [-0.149, -0.169, 6.17], #region 21 center
        "coords":[-0.858, -0.349, 6.17],# a6_R10c10
        # "coords":[-0.761, -0.170, 6.17], #a_R10_c10_r10 dim spot
        "name": "{}-R21_a6_r10_c10".format(sample_name,),#_r10_c10
        "disable_opt":False,
        "ramp_voltages": True,
        "expected_count_rate":None,
        
        # "spin_laser": green_laser,
        # "spin_laser_power": green_power,
        # "spin_pol_dur": 1e5,
        # "spin_readout_laser_power": green_power,
        # "spin_readout_dur": 350,
        
        # "imaging_laser": yellow_laser,
        # "imaging_laser_power": 0.3,
        # "imaging_laser_filter": nd_yellow,
        # "imaging_readout_dur": 1e7,
        
        "imaging_laser": red_laser,
        "imaging_laser_power": 0.61, # 6 mW
        "imaging_readout_dur": 1e7,
        
        # "imaging_laser":green_laser,
        # "imaging_laser_power": None,
        # "imaging_readout_dur": 1e7,
        
        # "imaging_laser": red_laser,
        # "imaging_laser_power": 0.62,
        # "imaging_readout_dur": 1e7,
        
        
        # 'nvm_prep_laser': green_laser, 'nvm_prep_laser_power': green_power, 
        # 'nvm_prep_laser_dur': 1e4,
        # 'nv0_prep_laser': red_laser, 'nv0_prep_laser_power': red_power,
        # 'nv0_prep_laser_dur':1e4,
        
        # 'spin_shelf_laser': yellow_laser, 'spin_shelf_laser_filter': nd_yellow, 
        # 'spin_shelf_laser_power': 0.4, 'spin_shelf_dur':0,
            
        
        # "initialize_laser": green_laser, 
        # "initialize_laser_power": 0.8,
        # "initialize_laser_dur":  1e5,
        # # "test_laser": green_laser, 
        # # "test_laser_power": None,
        # # "test_laser_dur":  1e5,
        
        
        # "initialize_laser": red_laser, 
        # "initialize_laser_power": 0.69,
        # "initialize_laser_dur": 2e4,
        # "test_laser": red_laser, 
        # "test_laser_power": 0.66,
        # "test_laser_dur":  1e6,
        
        
        # "charge_readout_laser": red_laser,
        # "charge_readout_laser_power": 0.6,
        # #0.6, 7 mW
        # #0.57 2 mW
        # "charge_readout_laser_dur": 75000,
        
        
        # "charge_readout_laser": green_laser,
        # "charge_readout_laser_power": None, #5.5 mW
        # #0.6, 7 mW
        # #0.57 2 mW
        # "charge_readout_laser_dur": 50000,
        
        
        # "initialize_laser": red_laser, 
        # "initialize_laser_power": 0.69,
        # "initialize_laser_dur":  2e4,
        # "CPG_laser": green_laser, 
        # "CPG_laser_power": None,
        # "CPG_laser_dur":  1e6,
        
        "initialize_laser": green_laser, # NExt experiment
        "initialize_laser_power": None,
        "initialize_laser_dur":  1e6,
        "CPG_laser": red_laser, 
        "CPG_laser_power": 0.69,
        "CPG_laser_dur":  2e4,
        
        
        "charge_readout_laser": red_laser,
        "charge_readout_laser_power": 0.6,#0.561,
        "charge_readout_laser_dur": 75000,
        
        
        "collection_filter": "715_lp",
        # "collection_filter": "715_sp+630_lp",
        "magnet_angle": None,
        "resonance_LOW":2.87,"rabi_LOW": 150,
        "uwave_power_LOW": 15.5,  # 15.5 max
        "resonance_HIGH": 2.932,
        "rabi_HIGH": 59.6,
        "uwave_power_HIGH": 14.5,
    }  # 14.5 max
    
    
    
    
      
    
    nv_sig = nv_sig
    
    
    # %% Functions to run

    try:

        # tool_belt.init_safe_stop()
        # for dz in [0, 0.15,0.3, 0.45, 0.6, 0.75,0.9, 1.05, 1.2, 1.5, 1.7, 1.85, 2, 2.15, 2.3, 2.45]: #0.5,0.4, 0.3, 0.2, 0.1,0, -0.1,-0.2,-0.3, -0.4, -0.5
            # nv_sig_copy = copy.deepcopy(nv_sig)
            # coords = nv_sig["coords"]
            # new_coords= list(numpy.array(coords)+ numpy.array([0, 0, dz]))
            # # new_coords = numpy.array(coords) +[0, 0, dz]
            # # print(new_coords)
            # nv_sig_copy['coords'] = new_coords
            # do_image_sample(nv_sig_copy, apd_indices)
        # # 
        # 
        # tool_belt.set_drift([0.0, 0.0, tool_belt.get_drift()[2]])  # Keep z
        # tool_belt.set_drift([0.0, 0.0, 0.0])  
        # tool_belt.set_xyz(labrad.connect(), [-0.141+0.05, 0.514, 7.05])  
        # for dx in [-0.2, -0.4, -0.6, -0.8, -1]:
        #     nv_sig_copy = copy.deepcopy(nv_sig)
        #     coords = nv_sig["coords"]
        #     new_coords= list(numpy.array(coords)+ numpy.array([dx, 0, 0]))
        #     nv_sig_copy['coords'] = new_coords
            # do_image_sample(nv_sig_copy, apd_indices)
            
        do_optimize(nv_sig,apd_indices)
        # do_image_sample(nv_sig, apd_indices)
        # do_stationary_count(nv_sig, apd_indices)
        # do_image_sample(nv_sig, apd_indices)
        # do_image_sample_xz(nv_sig, apd_indices)
        # do_image_charge_states(nv_sig, apd_indices)
        
        
        # tool_belt.set_xyz(labrad.connect(), [0.445, 0.236,6.16+0.0196]) #s1
        # tool_belt.set_xyz(labrad.connect(), [0.389, 0.217,6.16+0.0196]) #s2
        # tool_belt.set_xyz(labrad.connect(), [0.426, 0.349,6.16+0.0196]) #s3
        # tool_belt.set_xyz(labrad.connect(), [0.215, -0.027, 6.13]) #s4
        
        
        # 
        # do_g2_measurement(nv_sig, 0, 1)
       
        # do_time_resolved_readout_three_pulses(nv_sig, apd_indices)
# 

        # subtracting time resolved readings
    # 
        do_time_resolved = False
        
        if do_time_resolved:
            # nv_sig['initialize_laser'] = red_laser
            # nv_sig['initialize_laser_power'] = 0.66
            nv_sig['test_laser'] = red_laser
            nv_sig['test_laser_power'] = 0.66
            
            bin_centers, binned_samples_sig =  do_time_resolved_readout_three_pulses(nv_sig, apd_indices)
            nv_sig_ref = copy.deepcopy(nv_sig)
            coords_past = nv_sig_ref['coords']
            coord_new = [coords_past[0]-0.05, coords_past[1], coords_past[2]]
            nv_sig_ref['coords'] = coord_new
            nv_sig_ref['disable_opt'] = True
            bin_centers, binned_samples_ref =  do_time_resolved_readout_three_pulses(nv_sig_ref, apd_indices)
            
            binned_samples_sub_red = binned_samples_sig - binned_samples_ref
            
            fig_r, ax = plt.subplots(1, 1, figsize=(10, 8.5))
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            
            init_color = tool_belt.get_registry_entry_no_cxn('wavelength',
                              ['Config', 'Optics', nv_sig['initialize_laser']])
            test_color = tool_belt.get_registry_entry_no_cxn('wavelength',
                              ['Config', 'Optics', nv_sig['test_laser']])
            readout_color = tool_belt.get_registry_entry_no_cxn('wavelength',
                              ['Config', 'Optics', nv_sig['charge_readout_laser']])
        
            ax.plot(bin_centers, binned_samples_sub_red, 'r-')
            ax.set_xlabel('Readout time (ns)')
            ax.set_ylabel('Counts')
            ax.set_title('{} initial pulse, {} test, {} readout, backround subtracted'.format(init_color,  
                                                                              test_color,
                                                                              readout_color))
            
            time.sleep(1)
            timestamp =tool_belt.get_time_stamp()
            file_path = tool_belt.get_file_path('time_resolved_readout', timestamp, nv_sig['name'])
            tool_belt.save_figure(fig_r, file_path)
            
            # ---
            nv_sig['test_laser'] = green_laser
            nv_sig['test_laser_power'] = None
            
            bin_centers, binned_samples_sig =  do_time_resolved_readout_three_pulses(nv_sig, apd_indices)
            nv_sig_ref = copy.deepcopy(nv_sig)
            coords_past = nv_sig_ref['coords']
            coord_new = [coords_past[0]-0.05, coords_past[1], coords_past[2]]
            nv_sig_ref['coords'] = coord_new
            nv_sig_ref['disable_opt'] = True
            bin_centers, binned_samples_ref =  do_time_resolved_readout_three_pulses(nv_sig_ref, apd_indices)
            
            binned_samples_sub_green = binned_samples_sig - binned_samples_ref
            
            fig_g, ax = plt.subplots(1, 1, figsize=(10, 8.5))
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            
            init_color = tool_belt.get_registry_entry_no_cxn('wavelength',
                              ['Config', 'Optics', nv_sig['initialize_laser']])
            test_color = tool_belt.get_registry_entry_no_cxn('wavelength',
                              ['Config', 'Optics', nv_sig['test_laser']])
            readout_color = tool_belt.get_registry_entry_no_cxn('wavelength',
                              ['Config', 'Optics', nv_sig['charge_readout_laser']])
        
            ax.plot(bin_centers, binned_samples_sub_green, 'g-')
            ax.set_xlabel('Readout time (ns)')
            ax.set_ylabel('Counts')
            ax.set_title('{} initial pulse, {} test,  {} readout, backround subtracted'.format(init_color,
                                                                                      test_color,
                                                                                      readout_color))
            time.sleep(1)                                                  
            timestamp =tool_belt.get_time_stamp()
            file_path = tool_belt.get_file_path('time_resolved_readout', timestamp, nv_sig['name'])
            tool_belt.save_figure(fig_g, file_path)
            
            
            binned_samples_sub = binned_samples_sub_red - binned_samples_sub_green
            # binned_samples_sub =  binned_samples_sub_green - binned_samples_sub_red
            fig_s, ax = plt.subplots(1, 1, figsize=(10, 8.5))
            ax.plot(bin_centers, binned_samples_sub, 'b-')
            ax.set_xlabel('Readout time (ns)')
            ax.set_ylabel('Counts')
            ax.set_title('Subtracted measurement, {} readout'.format(readout_color))
            time.sleep(1)
            timestamp =tool_belt.get_time_stamp()
            file_path = tool_belt.get_file_path('time_resolved_readout', timestamp, nv_sig['name'])
            tool_belt.save_figure(fig_s, file_path)


        
        # 
        # do_optimize_magnet_angle(nv_sig, apd_indices)
        # do_resonance(nv_sig, nv_sig, apd_indices,  2.875, 0.2)
        # do_resonance(nv_sig, nv_sig, apd_indices,  2.875, 0.1)
        # do_resonance_state(nv_sig,nv_sig, apd_indices, States.LOW)
        
        # do_rabi(nv_sig, nv_sig, apd_indices, States.LOW, uwave_time_range=[0, 200])
        # do_rabi(nv_sig, nv_sig,apd_indices, States.HIGH, uwave_time_range=[0, 200])
        
        # do_pulsed_resonance(nv_sig, nv_sig, apd_indices, 2.875, 0.1)
        # do_pulsed_resonance_state(nv_sig, nv_sig,apd_indices, States.LOW)
        # do_ramsey(nv_sig, opti_nv_sig,apd_indices)
        # do_spin_echo(nv_sig, apd_indices)
        
        num_runs = int(1e2)
        num_steps_a = 51
        num_steps_b = num_steps_a
        img_range_1D = None#[[0,0,0],[0.075,0,0]]
        img_range_2D = [0.05, 0.05, 0]
        offset = [0.2/80,0.4/80,0]
        # do_SPaCE(nv_sig, nv_sig,apd_indices, num_runs, num_steps_a, num_steps_b,
        #         img_range_1D, img_range_2D, offset, charge_state_threshold = None)
        # do_image_sample(nv_sig, apd_indices)
        
        
        # drift = tool_belt.get_drift()
        # tool_belt.set_drift([0.0, 0.0, drift[2]])  # Keep z
        # tool_belt.set_drift([drift[0], drift[1], 0.0])  # Keep xy
        # do_g2_measurement(nv_sig, 0, 1) 
        # do_resonance(nv_sig, apd_indices, 2.875, 0.15)
        # do_resonance_state(nv_sig, apd_indices, States.HIGH)
        # do_pulsed_resonance(nv_sig, apd_indices, 2.875, 0.25)
        # nv_sig['magnet_angle'] = 75
        # do_pulsed_resonance(nv_sig, apd_indices, 2.875, 0.25)
        # do_resonance_state(nv_sig, apd_indices, States.LOW)
        # do_resonance_state(nv_sig, apd_indices, States.HIGH)
        # do_pulsed_resonance_state(nv_sig, nv_sig, apd_indices, States.LOW)
        # do_pulsed_resonance_state(nv_sig, apd_indices, States.HIGH)
        #do_optimize_magnet_angle(nv_sig, apd_indices)
        # do_rabi(nv_sig, apd_indices, States.LOW, uwave_time_range=[0, 300])
        # do_rabi(nv_sig, apd_indices, States.HIGH, uwave_time_range=[0, 300])
        # do_spin_echo(nv_sig, apd_indices)
        # do_spin_echo_battery(nv_sig, apd_indices)
        # do_t1_battery(nv_sig, apd_indices)
        # do_t1_interleave_knill(nv_sig, apd_indices)

        # Operations that don't need an NV
        # tool_belt.set_drift([0.0, 0.0, 0.0])  # Totally reset
        # tool_belt.set_drift([0.0, 0.0, tool_belt.get_drift()[2]])  # Keep z
        # tool_belt.set_xyz(labrad.connect(), [0,0,5])
#-0.243, -0.304,5.423
#ML -0.216, -0.115,5.417
    except Exception as exc:
        # Intercept the exception so we can email it out and re-raise it
        if not debug_mode:
            tool_belt.send_exception_email()
        raise exc

    finally:
        # Reset our hardware - this should be done in each routine, but
        # let's double check here
        tool_belt.reset_cfm()
        # Kill safe stop
        if tool_belt.check_safe_stop_alive():
            print("\n\nRoutine complete. Press enter to exit.")
            tool_belt.poll_safe_stop()
