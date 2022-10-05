# -*- coding: utf-8 -*-
"""
Measure the temperature dependence of the zero field splitting

Created on October 3rd, 2022

@author: mccambria
"""

### Imports

from isort import file
import utils.tool_belt as tool_belt
import majorroutines.optimize as optimize
import majorroutines.pulsed_resonance as pulsed_resonance
import majorroutines.set_drift_from_ref_image as set_drift_from_ref_image
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import labrad
from utils.tool_belt import States
from random import shuffle
import sys
from utils import kplotlib as kpl


# region Functions





# endregion

# region Main


def main(
    nv_sig,
    apd_indices,
    num_reps,
    num_runs,
    state,
    detuning=0.005,
    d_omega=0.002,
    opti_nv_sig=None,
    ret_file_name=False,
):

    with labrad.connect() as cxn:
        ret_vals = main_with_cxn(
            cxn,
            nv_sig,
            apd_indices,
            num_reps,
            num_runs,
            state,
            detuning,
            d_omega,
            opti_nv_sig,
            ret_file_name,
        )
    return ret_vals


def main_with_cxn(
    cxn,
    nv_list,
    apd_indices,
    num_reps,
    num_runs,
    state,
    temp_range,
    d_temp,
    ref_image,
    d_z,
    esr_freq_range,
    esr_num_steps,
    esr_num_reps,
    esr_num_runs,
):

    ### Initial calculations and setup

    tool_belt.reset_cfm(cxn)

    # Get the temps to measure including the endpoint
    temp_linspace = np.arange(temp_range[0], temp_range[1]+0.1, d_temp)
    
    temp_controller = tool_belt.get_temp_controller(cxn)
    temp_monitor = tool_belt.get_temp_monitor(cxn)
    
    # Set up data structure
    controller_temp_list = []
    monitor_temp_list = []
    zfs_list = []
    zfs_err_list = []
    zfs_file_list = []
    
    temp_controller.set_temp(temp_linspace[0])
    temp_controller.activate_temp_control()
    
    for set_point in temp_linspace:
        
        ### Switch the temperature
        
        temp_controller.set_temp(set_point)
        
        # Check stability
        # Move on if the standard deviation of the last 10 temps is < thresh
        stability_thresh = 0.1
        recent_temps = []
        while True:
            time.sleep(1)
            temp = temp_monitor.get_temp()
            recent_temps.append(temp)
            if len(recent_temps) > 10:
                recent_temps.pop(0)
            temp_noise = np.std(recent_temps)
            if temp_noise < stability_thresh:
                break
            
        ### Relocate NVs
        
        success = False
        attempt_count = 0
        while attempt_count < 10:
            attempt_count += 1
            drift = tool_belt.get_drift(cxn)
            z_drift = drift[2]
            adj_z_drift = z_drift + d_z
            adj_drift = [drift[0], drift[1], adj_z_drift]
            tool_belt.set_drift(adj_drift, cxn)
            success = set_drift_from_ref_image.main_with_cxn(cxn, ref_image, apd_indices)
            if success: 
                break
            
        if not success:
            print("Failed to relocate NVs. Stopping...")
            break
            
        ### Measure the zfs
        
        # Set up sub-lists
        controller_temp_list_sub = []
        monitor_temp_list_sub = []
        zfs_list_sub = []
        zfs_err_list_sub = []
        zfs_file_list_sub = []
        
        # Loop through each NV
        for nv_sig in nv_list:
            
            # Get the temps
            controller_temp = temp_controller.get_temp()
            monitor_temp = temp_monitor.get_temp()
            controller_temp_list_sub.append(controller_temp)
            monitor_temp_list_sub.append(monitor_temp)
            
            # Measure the zfs
            pesr_lambda = lambda adj_nv_sig: pulsed_resonance.state(
                nv_sig,
                apd_indices,
                States.LOW,
                esr_freq_range,
                esr_num_steps,
                esr_num_reps,
                esr_num_runs,
                ret_file_name=True,
            )
            res, res_err, file_name = pesr_lambda(nv_sig)
            zfs_list_sub.append(res)
            zfs_err_list_sub.append(res_err)
            zfs_file_list_sub.append(file_name)
            
        controller_temp_list.append(controller_temp_list_sub)
        monitor_temp_list.append(monitor_temp_list_sub)
        zfs_list.append(zfs_list_sub)
        zfs_err_list.append(zfs_err_list_sub)
        zfs_file_list.append(zfs_file_list_sub)
        
    temp_controller.deactivate_temp_control()
    
    if len(controller_temp_list_sub) == 0:
        print("Crashed out before any data was collected!")
        return
    
    ### Plot the results
    
    # Average over NVs
    avg_temp = [np.average(el) for el in monitor_temp_list]
    avg_zfs = [np.average(el) for el in zfs_list]
    avg_zfs_std = [np.std(el) for el in zfs_list]
    
    kpl.init_kplotlib()
    raw_fig, ax = plt.subplots()
    kpl.plot_data(avg_temp, avg_zfs, avg_zfs_std)
            
    ### Clean up and save the data

    tool_belt.reset_cfm(cxn)

    timestamp = tool_belt.get_time_stamp()
    raw_data = {
        'timestamp': timestamp,
        "controller_temp_list": controller_temp_list,
        "monitor_temp_list": monitor_temp_list,
        "zfs_list": zfs_list,
        "zfs_err_list": zfs_err_list,
        "zfs_file_list": zfs_file_list,
    }

    name = "D_vs_T"
    file_path = tool_belt.get_file_path(__file__, timestamp, name)
    tool_belt.save_figure(raw_fig, file_path)
    tool_belt.save_raw_data(raw_data, file_path)
    
    
# endregion


### Run the file


if __name__ == "__main__":

    file_name = ""
    data = tool_belt.get_raw_data(file_name)

