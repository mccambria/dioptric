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
import majorroutines.image_sample_digital as image_sample_digital
import majorroutines.image_sample_xz_digital as image_sample_xz_digital
import majorroutines.optimize_digital as optimize_digital
import majorroutines.stationary_count as stationary_count
import majorroutines.resonance as resonance
import majorroutines.optimize_magnet_angle as optimize_magnet_angle
import majorroutines.pulsed_resonance as pulsed_resonance
import majorroutines.rabi as rabi
import majorroutines.ramsey as ramsey
import majorroutines.spin_echo as spin_echo
import minorroutines.test_routine_opx as test_routine_opx
import minorroutines.determine_delays as determine_delays
from utils.tool_belt import States
import time
import copy
import matplotlib.pyplot as plt


# %% Major Routines

def do_test_routine_opx(nv_sig, apd_indices, delay, readout_time, laser_name, laser_power, num_reps):
    apd_index = apd_indices[0]
    counts, times, channels = test_routine_opx.main(nv_sig, delay, readout_time, apd_index, laser_name, laser_power, num_reps)
    
    return counts, times, channels
    
    

def do_image_sample(nv_sig, apd_indices,scan_range=2,num_steps=30,cmin=None,cmax=None):
    scale = 1 #um / V
   
    # For now we only support square scans so pass scan_range twice
    image_sample_digital.main(nv_sig, scan_range, scan_range, num_steps, apd_indices,save_data=True,cbarmin=cmin,cbarmax=cmax)

def do_image_sample_xz(nv_sig, apd_indices,scan_range=2,num_steps=30,cmin=None,cmax=None):
    scale = 1 #um / V
   
    # For now we only support square scans so pass scan_range twice
    image_sample_xz_digital.main(nv_sig, scan_range, scan_range, num_steps, apd_indices,save_data=True,cbarmin=cmin,cbarmax=cmax)


def do_optimize(nv_sig, apd_indices):

    optimize_coords = optimize_digital.main(
        nv_sig,
        apd_indices,
        set_to_opti_coords=False,
        save_data=True,
        plot_data=True,
    )
    return optimize_coords



def do_optimize_z(nv_sig, apd_indices):
    
    adj_nv_sig = copy.deepcopy(nv_sig)
    adj_nv_sig["only_z_opt"] = True

    optimize_coords = optimize_digital.main(
        adj_nv_sig,
        apd_indices,
        set_to_opti_coords=False,
        save_data=True,
        plot_data=True,
    )
    return optimize_coords

def do_stationary_count(nv_sig, apd_indices,disable_opt=False):

    run_time = 1 * 60 * 10 ** 9  # ns

    stationary_count.main(nv_sig, run_time, apd_indices,disable_opt)
    
def do_laser_delay_calibration(nv_sig,laser_name,apd_indices,num_reps = int(2e6),
                              delay_range = [50, 500],num_steps=21):
    # laser_delay
    # num_reps = int(2e6)
    # delay_range = [50, 500]
    # num_steps = 21
    with labrad.connect() as cxn:
        determine_delays.aom_delay(
            cxn,
            nv_sig,
            apd_indices,
            delay_range,
            num_steps,
            num_reps,
            laser_name,
            1,
        )
        
def do_resonance(nv_sig, apd_indices, freq_center=2.87, freq_range=0.2,num_steps = 51, num_runs = 20):

    # num_steps = 51
    # num_runs = 20
    uwave_power = -5.0
    nv_sig['spin_pol_dur']=2e3

    resonance.main(
        nv_sig,
        apd_indices,
        freq_center,
        freq_range,
        num_steps,
        num_runs,
        uwave_power,
        state=States.HIGH,
    )

def do_rabi(nv_sig, apd_indices, uwave_time_range, state ,num_steps = 51, num_reps = 1e4, num_runs = 20):

    # num_steps = 51
    # num_runs = 20
    nv_sig['spin_pol_dur']=2e3

    rabi.main(
        nv_sig,
        apd_indices,
        uwave_time_range,
        state,
        num_steps,
        num_reps,
        num_runs,
    )    

def do_spin_echo(nv_sig, apd_indices,max_time=120,num_reps=4e3,num_runs=5,state=States.LOW):

    # T2* in nanodiamond NVs is just a couple us at 300 K
    # In bulk it's more like 100 us at 300 K
    # max_time = 120  # us
    num_steps = int(max_time)  # 1 point per us
    precession_time_range = [1e3, max_time * 10 ** 3]
    # num_reps = 4e3
    # num_runs = 5
    # num_runs = 20
    
    # state = States.LOW

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

def do_ramsey(nv_sig, opti_nv_sig, apd_indices,detuning=4):

    # detuning = 5  # MHz
    precession_time_range = [0, 2500]
    num_steps = 51
    num_reps = int(2e4)
    num_runs = 50

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

def do_pulsed_resonance(nv_sig, opti_nv_sig, apd_indices, freq_center=2.87, freq_range=0.2, uwave_pulse_dur=100, num_steps=51, num_reps=1e4, num_runs=10):

    # num_steps =101
    # num_reps = 1e4
    # num_runs = 10
    uwave_power = 16.5
    nv_sig['spin_pol_dur']=2e3
    # uwave_pulse_dur = int(30)

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

def do_optimize_magnet_angle(nv_sig, apd_indices):

    angle_range = [0,150]
    num_angle_steps = 6
    freq_center = 2.87
    freq_range = 0.250
    num_freq_steps = 51
    # num_freq_runs = 30
    num_freq_runs = 4

    # Pulsed
    uwave_power = 16.5
    uwave_pulse_dur = 85
    num_freq_reps = 2e4

    # CW
    # uwave_power = -5.0
    # uwave_pulse_dur = None
    # num_freq_reps = None

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
    


# %% Run the file


if __name__ == "__main__":

    # In debug mode, don't bother sending email notifications about exceptions
    debug_mode = True
    # %% Shared parameters
    
    with labrad.connect() as cxn:
        apd_indices = tool_belt.get_registry_entry(cxn, "apd_indices", ["","Config"])
        apd_indices = apd_indices.astype(list).tolist()
        

    sample_name = "johnson"
    green_laser = "cobolt_515"

    nv_sig = {
        'coords': [89.664, 39.141, 76.5], 'name': '{}-search'.format(sample_name),
        'ramp_voltages': False, "only_z_opt": False, 'disable_opt': False, "disable_z_opt": False, 
        'expected_count_rate': 50,
        "imaging_laser": green_laser, "imaging_laser_filter": "nd_0", 
        "imaging_readout_dur": 10e6,
        "spin_laser": green_laser,
        "spin_laser_filter": "nd_0",
        "spin_pol_dur": 10e3,
        "spin_readout_dur": 350,
        "nv-_reionization_laser": green_laser,
        "nv-_reionization_dur": 1e6,
        "nv-_reionization_laser_filter": "nd_0",
        "nv-_prep_laser": green_laser,
        "nv-_prep_laser_dur": 1e6,
        "nv-_prep_laser_filter": "nd_0",
        "initialize_laser": green_laser,
        "initialize_dur": 1e4,
        'collection_filter': None, 'magnet_angle': 155,
        'resonance_LOW': 2.7842, 'rabi_LOW': 192, 'uwave_power_LOW': 16.5,
        'resonance_HIGH': 2.9583, 'rabi_HIGH': 148.6, 'uwave_power_HIGH': 16.5,
        }
    
    
    
    # %% Functions to run

    try:
        # tool_belt.reset_drift()

        # tool_belt.init_safe_stop()
        # do_test_routine_opx(nv_sig, apd_indices, laser_name=green_laser, laser_power=1, 
                            # delay=2e9, readout_time=1e9, num_reps=10)
                            
        # do_image_sample_xz(nv_sig, apd_indices,num_steps=80,scan_range=8)#,cmin=0,cmax=50)
        # do_image_sample(nv_sig, apd_indices,num_steps=40,scan_range=4)#,cmin=0,cmax=75)
        
        # do_optimize(nv_sig, apd_indices)
        # do_optimize_z(nv_sig, apd_indices)
        
        # do_stationary_count(nv_sig, apd_indices,disable_opt=True)
        
        # do_laser_delay_calibration(nv_sig,'cobolt_515',apd_indices,num_reps=int(8e5),
        #                             delay_range=[300,500],num_steps=101)
        
        # do_resonance(nv_sig, apd_indices,num_steps = 51, num_runs = 15)
        # do_resonance_modulo(nv_sig, apd_indices,num_steps = 51, num_runs = 5)
        # do_rabi(nv_sig, apd_indices, uwave_time_range = [16,320], state=States.LOW,num_reps=2e4,num_runs=8,num_steps=51)
        # do_rabi(nv_sig, apd_indices, uwave_time_range = [16,1000], state=States.HIGH,num_reps=2e4,num_runs=10,num_steps=201)
        # do_pulsed_resonance(nv_sig, nv_sig, apd_indices,uwave_pulse_dur=84, num_steps=51, num_reps=2e4, num_runs=5)
        # do_spin_echo(nv_sig, apd_indices,max_time=40,num_reps=2e4,num_runs=20,state=States.LOW)
        # do_ramsey(nv_sig, nv_sig, apd_indices,detuning=4)
        # do_optimize_magnet_angle(nv_sig, apd_indices)
        
        # for readout_dur in [200,250,300,350,400,450,500]:
        #     nv_sig['spin_readout_dur'] = readout_dur
        #     do_rabi(nv_sig, apd_indices, uwave_time_range = [16,500], state=States.LOW,num_reps=2e4,num_runs=30,num_steps=51)
            
        
        # do_spin_echo(nv_sig, apd_indices,max_time=100,num_reps=2e4,num_runs=10,state=States.LOW)


    except Exception as exc:
        # Intercept the exception so we can email it out and re-raise it
        if not debug_mode:
            tool_belt.send_exception_email(email_to="cdfox@wisc.edu")
        raise exc

    finally:
        # Reset our hardware - this should be done in each routine, but
        # let's double check here
        tool_belt.reset_cfm()
        # Kill safe stop
        tool_belt.reset_safe_stop()