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
    
    # 35 um / V
    
    # scan_range = 5.0
    # scan_range = 3.0
    # scan_range = 1.5
    # scan_range =1
    # scan_range = 0.8
    #scan_range = 0.5
    # scan_range = 0.35
    # scan_range = 0.25
    scan_range = 0.15
    #scan_range = 0.1
    # scan_range = 0.05
    # scan_range = 0.025
    #
    # num_steps = 400
    # num_steps = 300
    # num_steps = 200
    # num_steps = 175
    # num_steps = 135
    #num_steps =120
    num_steps = 90
    # num_steps = 60
    # num_steps = 31
    #num_steps = 15

    # For now we only support square scans so pass scan_range twice
    image_sample.main(nv_sig, scan_range, scan_range, num_steps, apd_indices)


def do_image_sample_xz(nv_sig, apd_indices):

    scan_range_x = 0.2

    scan_range_z = 2

    num_steps = 90

    image_sample_xz.main(
        nv_sig,
        scan_range_x,
        scan_range_z,
        num_steps,
        apd_indices,
        um_scaled=False,
    )


def do_image_charge_states(nv_sig, apd_indices):

    scan_range = 0.2

    num_steps = 90

    image_sample_charge_state_compare.main(
        nv_sig, scan_range, scan_range, num_steps, apd_indices
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
    diff_window = 150  # ns

    # g2_measurement.main(
    g2_SCC_branch.main(
        nv_sig, run_time, diff_window, apd_a_index, apd_b_index
    )


def do_resonance(nv_sig, opti_nv_sig,apd_indices, freq_center=2.87, freq_range=0.2):

    num_steps = 101
    num_runs = 10
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
    num_runs = 15

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
    uwave_pulse_dur = 75

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
    num_reps = 1e4
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
    num_freq_runs = 5

    # Pulsed
    uwave_power = 14.5
    uwave_pulse_dur = 140/2
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
    num_runs = 10
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


def do_SPaCE(nv_sig, opti_nv_sig, num_runs, num_steps_a, num_steps_b,
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
    SPaCE.main(nv_sig, opti_nv_sig, num_runs, num_steps_a, num_steps_b,
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

    apd_indices = [0]
    # apd_indices = [1]
    # apd_indices = [0,1]

    nd_yellow = "nd_1.0"
    green_power = 10
    red_power = 120
    sample_name = "lovelace"
    green_laser = "cobolt_515"
    yellow_laser = "laserglow_589"
    red_laser = "cobolt_638"

    nv_sig_search = {
        "coords": [-0.068, 0.019,  5.869],
        "name": "{}-search".format(sample_name),
        "disable_opt": False,
        "ramp_voltages": False,
        "expected_count_rate": None,
        "imaging_laser": green_laser,
        "imaging_laser_power": green_power,
        "imaging_readout_dur": 1e7,
        "collection_filter": "630_lp",
        "magnet_angle": None,
        "resonance_LOW": 2.8012,
        "rabi_LOW": 141.5,
        "uwave_power_LOW": 15.5,  # 15.5 max
        "resonance_HIGH": 2.9445,
        "rabi_HIGH": 191.9,
        "uwave_power_HIGH": 14.5,
    }  # 14.5 max

    
    
    nv_sig = {
        "coords": [0.167, 0.008,5.222],
        "name": "{}-nv2_2022_01_11".format(sample_name,),
        "disable_opt": False,
        "ramp_voltages": True,
        "expected_count_rate": 50,
        
        # "coords": [-0.063, -0.145, 5.0],
        # "name": "{}-nv0_2021_11_08".format(sample_name,),
        # "disable_opt": False,
        # "ramp_voltages": False,
        # "expected_count_rate": 65,
        
        "spin_laser": green_laser,
        "spin_laser_power": green_power,
        "spin_pol_dur": 1e5,
        "spin_readout_laser_power": green_power,
        "spin_readout_dur": 350,
        
        "imaging_laser":green_laser,
        "imaging_laser_power": green_power,
        "imaging_readout_dur": 1e7,
        
        
        'nv-_reionization_laser': green_laser, 'nv-_reionization_laser_power': green_power, 
        'nv-_reionization_dur': 1E5,
        'nv0_ionization_laser': red_laser, 'nv0_ionization_laser_power': red_power,
        'nv0_ionization_dur':500,
        
        'spin_shelf_laser': yellow_laser, 'spin_shelf_laser_filter': nd_yellow, 
        'spin_shelf_laser_power': 0.4, 'spin_shelf_dur':0,
            
        "initialize_laser": green_laser,
        "initialize_laser_power": green_power,
        "initialize_dur": 1e4,
        "CPG_laser": red_laser,
        'CPG_laser_power': red_power,
        "CPG_laser_dur": 1e5,
        "charge_readout_laser": yellow_laser,
        "charge_readout_laser_filter": nd_yellow,
        "charge_readout_laser_power": 0.15,
        "charge_readout_dur": 50e6,
        
        "collection_filter": "630_lp",
        "magnet_angle": None,
        "resonance_LOW":2.8068,"rabi_LOW": 123.0,
        "uwave_power_LOW": 15.5,  # 15.5 max
        "resonance_HIGH": 2.9496,
        "rabi_HIGH": 215,
        "uwave_power_HIGH": 14.5,
    }  # 14.5 max
    
    
    
    
      
    
    nv_sig = nv_sig_search
    
    
    # %% Functions to run

    try:

        tool_belt.init_safe_stop()
        # for dz in [-0.5, -0.25, 0.25, 0.5]:
        #     nv_sig_copy = copy.deepcopy(nv_sig)
        #     coords = nv_sig["coords"]
        #     new_coords = numpy.array(coords) +[0, 0, dz]
        #     print(new_coords)
        #     do_image_sample(nv_sig_copy, apd_indices)
        
        do_optimize(nv_sig, apd_indices)
        # do_image_sample(nv_sig, apd_indices)
        # do_stationary_count(nv_sig, apd_indices)
        # do_image_sample_xz(nv_sig, apd_indices)
        # do_image_charge_states(nv_sig, apd_indices)
        
        # do_g2_measurement(nv_sig, 0, 1)
        
        #do_optimize_magnet_angle(nv_sig, apd_indices)
        # do_resonance(nv_sig, nv_sig, apd_indices,  2.875, 0.2)
        # do_resonance_state(nv_sig,opti_nv_sig, apd_indices, States.LOW)
        
        # do_rabi(nv_sig, nv_sig, apd_indices, States.LOW, uwave_time_range=[0, 300])
        # do_rabi(nv_sig, opti_nv_sig,apd_indices, States.HIGH, uwave_time_range=[0, 300])
        
        # do_pulsed_resonance(nv_sig, nv_sig, apd_indices, 2.875, 0.2)
        #do_pulsed_resonance_state(nv_sig, nv_sig,apd_indices, States.LOW)
        # do_ramsey(nv_sig, opti_nv_sig,apd_indices)
        # do_spin_echo(nv_sig, apd_indices)
        
        offset_x = 0
        offset_y = 0
        offset_z = 0
        offset_list = [offset_x, offset_y, offset_z]
        num_steps_x = 51
        num_steps_y = 51
        num_steps_z = 101
    
        for t in [1e3]:
            # nv_sig['CPG_laser_dur'] = t
            img_range_2D = [0.04,0.04, 0 ]
            # do_SPaCE(nv_sig, nv_sig, 1, num_steps_x, num_steps_y, 
            #             None,  img_range_2D, offset_list)
            # img_range_2D = [0.05,0, 4/16 ]
            #do_SPaCE(nv_sig, nv_sig, 3, num_steps_x, num_steps_z, 
            #          None,  img_range_2D, [offset_x, offset_y, +6/16])
            #do_SPaCE(nv_sig, nv_sig, 3, num_steps_x, num_steps_z, 
            #            None,  img_range_2D, [offset_x, offset_y, -6/16])
            # img_range_2D = [0,0.05, 4/16 ]
            # do_SPaCE(nv_sig, nv_sig, 5, num_steps_y, num_steps_z, 
            #           None,  img_range_2D, offset_list)
            
            
        # 1st airy ring power
        t_list = [750e3] #1e3, 1e4, 1e5

        for t in t_list:
            nv_sig['CPG_laser_dur'] = t
            num_steps = 301
            num_runs = 25
            ## +x
            # do_SPaCE(nv_sig, nv_sig, num_runs, num_steps, None, 
            #         [[-0.275/50, -0.25/50,0 ], [-0.575/50, -0.25/50,0 ]],  None, offset_list)
            # #-x
            # do_SPaCE(nv_sig, nv_sig, num_runs, num_steps, None, 
            #         [[0.15/50, -0.25/50,0 ], [0.45/50, -0.25/50,0 ]],  None, offset_list)
            # #-y
            # do_SPaCE(nv_sig, nv_sig, num_runs, num_steps, None, 
            #         [[-0.055/50, -0.475/50,0 ], [-0.055/50, -0.775/50,0 ]],  None, offset_list)
            # #+y
            # do_SPaCE(nv_sig, nv_sig, num_runs, num_steps, None, 
            #         [[-0.055/50, 0.05/50,0 ], [-0.055/50, 0.35/50,0 ]],  None, offset_list)
            
          
          
            
         
        # do_scc_resonance(nv_sig, opti_nv_sig, apd_indices)
        #do_scc_spin_echo(nv_sig, opti_nv_sig, apd_indices, 0, 315)
        
        z = nv_sig['coords'][2]
        A = [-0.001, -0.008, z]
        B = [-0.007, -0.008, z]
        
        depletion_point = [A, B]
        # depletion_point = [ B]
        
        depletion_times = [10e3, 7.5e3]
        for i in range(1): #do the measurement 4 times over
        #for t in [8,10]:
            for p in range(len(depletion_point)):   
                nv_sig['depletion_coords'] = depletion_point[p]
                nv_sig['CPG_laser_dur'] = depletion_times[p]
        
                # do_super_resolution_resonance(nv_sig, opti_nv_sig, apd_indices)
                # do_super_resolution_spin_echo(nv_sig, opti_nv_sig, apd_indices, 0, 5 )
                # do_super_resolution_spin_echo(nv_sig, opti_nv_sig, apd_indices, 0, 40 )
                # do_super_resolution_spin_echo(nv_sig, opti_nv_sig, apd_indices, 0, 315 )
                
                # do_super_resolution_ramsey(nv_sig, opti_nv_sig, apd_indices, 0, 2)
        
        #drift = tool_belt.get_drift()
        #tool_belt.set_drift([0.0, 0.0, drift[2]])  # Keep z
        # tool_belt.set_drift([drift[0], drift[1], 0.0])  # Keep xy
        # do_g2_measurement(nv_sig, 0, 1) 
        # do_resonance(nv_sig, apd_indices, 2.875, 0.15)
        # do_resonance_state(nv_sig, apd_indices, States.HIGH)
        # do_pulsed_resonance(nv_sig, apd_indices, 2.875, 0.25)
        # nv_sig['magnet_angle'] = 75
        # do_pulsed_resonance(nv_sig, apd_indices, 2.875, 0.25)
        # do_resonance_state(nv_sig, apd_indices, States.LOW)
        # do_resonance_state(nv_sig, apd_indices, States.HIGH)
        # do_pulsed_resonance_state(opti_nv_sig, apd_indices, States.LOW)
        # do_pulsed_resonance_state(nv_sig, apd_indices, States.HIGH)
        # do_optimize_magnet_angle(nv_sig, apd_indices)
        # do_rabi(nv_sig, apd_indices, States.LOW, uwave_time_range=[0, 300])
        # do_rabi(nv_sig, apd_indices, States.HIGH, uwave_time_range=[0, 300])
        # do_discrete_rabi(nv_sig, apd_indices, States.LOW, 4)
        # do_discrete_rabi(nv_sig, apd_indices, States.HIGH, 4)
        # do_spin_echo(nv_sig, apd_indices)
        # do_spin_echo_battery(nv_sig, apd_indices)
        # do_t1_battery(nv_sig, apd_indices)
        # do_t1_interleave_knill(nv_sig, apd_indices)

        # Operations that don't need an NV
        # tool_belt.set_drift([0.0, 0.0, 0.0])  # Totally reset
        # tool_belt.set_drift([0.0, 0.0, tool_belt.get_drift()[2]])  # Keep z
        # tool_belt.set_xyz(labrad.connect(), [0, 0, 5])
        # tool_belt.set_xyz(labrad.connect(), [-0.169, -0.006, 5.086])

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
