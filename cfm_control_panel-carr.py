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
# import chargeroutines.SPaCE_digital as SPaCE_digital
# import chargeroutines.SPaCE_digital_annulus as SPaCE_digital_annulus
# import chargeroutines.g2_measurement as g2_SCC_branch
import majorroutines.stationary_count as stationary_count
import minorroutines.test_routine_opx as test_routine_opx
# import majorroutines.set_drift_from_reference_image as set_drift_from_reference_image
# import debug.test_major_routines as test_major_routines
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

# def do_g2_measurement(nv_sig, apd_a_index, apd_b_index):

#     run_time = 5*60  # s
#     diff_window = 150  # ns

#     # g2_measurement.main(
#     g2_SCC_branch.main(
#         nv_sig, run_time, diff_window, apd_a_index, apd_b_index
#     )


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
        'coords': [84.605, 37.951, 69.28], 'name': '{}-search'.format(sample_name),
        'ramp_voltages': False,
        "only_z_opt": False,
        'disable_opt': False, "disable_z_opt": False, 'expected_count_rate': None,
        "imaging_laser": green_laser, "imaging_laser_filter": "nd_0", "imaging_readout_dur": 1e7,
        "spin_laser": green_laser,
        "spin_laser_filter": "nd_0",
        "spin_pol_dur": 100e3,
        "spin_readout_dur": 2e3,
        "nv-_reionization_laser": green_laser,
        "nv-_reionization_dur": 1e6,
        "nv-_reionization_laser_filter": "nd_0",
        "nv-_prep_laser": green_laser,
        "nv-_prep_laser_dur": 1e6,
        "nv-_prep_laser_filter": "nd_0",
        # "nv0_ionization_laser": red_laser,
        # "nv0_ionization_dur": 75,
        # "nv0_prep_laser": red_laser,
        # "nv0_prep_laser_dur": 75,
        # "spin_shelf_laser": yellow_laser,
        # "spin_shelf_dur": 0,
        # "spin_shelf_laser_power": 1.0,
        "initialize_laser": green_laser,
        "initialize_dur": 1e4,
        # "charge_readout_laser": yellow_laser,
        # "charge_readout_dur": 100e6,
        # "charge_readout_laser_power": 1.0,
        'collection_filter': None, 'magnet_angle': None,
        'resonance_LOW': 2.8059, 'rabi_LOW': 226.9, 'uwave_power_LOW': 16.5,
        'resonance_HIGH': 2.9363, 'rabi_HIGH': 300, 'uwave_power_HIGH': 16.5,
        }
    
    
    # %% Functions to run

    try:
        # tool_belt.reset_drift()
        tool_belt.init_safe_stop()
        # do_test_routine_opx(nv_sig, apd_indices, laser_name=green_laser, laser_power=1, 
                            # delay=2e9, readout_time=1e9, num_reps=10)
        # do_image_sample_xz(nv_sig, apd_indices,num_steps=50,scan_range=10)#,cmin=0,cmax=50)
        # do_image_sample(nv_sig, apd_indices,num_steps=20,scan_range=2)#,cmin=0,cmax=75)
        # do_optimize(nv_sig, apd_indices)
        # do_optimize_z(nv_sig, apd_indices)
        # do_stationary_count(nv_sig, apd_indices,disable_opt=True)

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