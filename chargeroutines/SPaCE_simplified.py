# -*- coding: utf-8 -*-
"""
Created on Sat May 14 08:54:38 2022

a measurement that mimics the SPaCE measuremnet, but specifically uses two points,
the starting coordinates and one other coordinate.

@author: agard
"""

# import labrad
import scipy.stats
import scipy.special
import numpy
import math
import copy
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import random
import photonstatistics as model
import scipy.stats as stats
import labrad

import utils.tool_belt as tool_belt
import majorroutines.optimize as optimize


def build_voltages_from_list(start_coords_drift, coords_list_drift, num_reps):

    # calculate the x values we want to step thru
    start_x_value = start_coords_drift[0]
    start_y_value = start_coords_drift[1]
    
    num_samples = len(coords_list_drift)
    
    # we want this list to have the pattern [[readout], [target], [readout], [readout], 
    #                                                   [target], [readout], [readout],...]
    # The glavo needs a 0th coord, so we'll pass the readout NV as the "starting" point
    x_points = [start_x_value]
    y_points = [start_y_value]
    
    # now create a list of all the coords we want to feed to the galvo
    for n in range(num_reps):
        for i in range(num_samples):
            x_points.append(coords_list_drift[i][0])
            x_points.append(start_x_value)
            x_points.append(start_x_value) 
            
            y_points.append(coords_list_drift[i][1])
            y_points.append(start_y_value)
            y_points.append(start_y_value) 
        
    return x_points, y_points

def collect_counts(cxn, num_reps, num_samples, seq_args_string,apd_indices):
        
    #  Set up the APD
    cxn.apd_tagger.start_tag_stream(apd_indices)
    # prepare and run the sequence
    file_name = 'SPaCE.py'
    cxn.pulse_streamer.stream_load(file_name, seq_args_string)
    cxn.pulse_streamer.stream_start(num_reps)
        
    total_samples_list = []
    num_read_so_far = 0

    while num_read_so_far < num_samples:

        if tool_belt.safe_stop():
            break

        # Read the samples and update the image
        new_samples = cxn.apd_tagger.read_counter_simple()
        num_new_samples = len(new_samples)

        if num_new_samples > 0:
            for el in new_samples:
                total_samples_list.append(int(el))
            num_read_so_far += num_new_samples

    # print(total_samples_list)
    # readout_counts = total_samples_list[0::3] #init pulse
    # readout_counts = total_samples_list[1::3] #depletion pulse
    readout_counts = total_samples_list[2::3] #readout pulse
    readout_counts_list = [int(el) for el in readout_counts]
    
    cxn.apd_tagger.stop_tag_stream()
            
    return readout_counts_list

# %%
# Apply a gren or red pulse, then measure the counts under yellow illumination.
# Repeat num_reps number of times and returns the list of counts after red illumination, then green illumination
# Use with DM on red and green
def measure(nv_sig, pulse_coords, apd_indices, num_reps):

    with labrad.connect() as cxn:
        sig_counts, ref_counts = measure_with_cxn(cxn, nv_sig, pulse_coords, apd_indices, num_reps)

    return sig_counts, ref_counts
def measure_with_cxn(cxn, nv_sig,pulse_coords, apd_indices, num_reps):

    tool_belt.reset_cfm(cxn)


    # Initial Calculation and setup

    tool_belt.set_filter(cxn, nv_sig, 'charge_readout_laser')

    start_coords = numpy.array(nv_sig['coords'])

    init_color = tool_belt.get_registry_entry_no_cxn('wavelength',
                      ['Config', 'Optics', nv_sig['initialize_laser']])
    pulse_color = tool_belt.get_registry_entry_no_cxn('wavelength',
                      ['Config', 'Optics', nv_sig['CPG_laser']])
    readout_color = tool_belt.get_registry_entry_no_cxn('wavelength',
                      ['Config', 'Optics', nv_sig['charge_readout_laser']])
    pulse_time = nv_sig['CPG_laser_dur']
    initialization_time = nv_sig['initialize_laser_dur']
    charge_readout_time = nv_sig['charge_readout_laser_dur']
    
    
    initialization_laser_power = nv_sig['initialize_laser_power']
    pulse_laser_power = nv_sig['CPG_laser_power']
    charge_readout_laser_power = nv_sig['charge_readout_laser_power']



    # Pulse sequence to do a single pulse followed by readout
    seq_file = 'SPaCE.py'


    ################## Load the measuremnt with green laser ##################  
    seq_args = [initialization_time, pulse_time, charge_readout_time,
                initialization_laser_power, pulse_laser_power,charge_readout_laser_power,
                apd_indices[0], init_color, pulse_color, readout_color]
    seq_args_string = tool_belt.encode_seq_args(seq_args)
    ret_vals=cxn.pulse_streamer.stream_load(seq_file, seq_args_string)
    period = ret_vals[0]
    print(seq_args)       
    
    tool_belt.init_safe_stop()
    
    # Optimize
    optimize.main_with_cxn(cxn, nv_sig, apd_indices)
    drift = numpy.array(tool_belt.get_drift())

    # get the readout coords with drift
    start_coords_drift = start_coords + drift
    pulse_coords_drift = numpy.array(pulse_coords) + drift
    # Build the list to step through the coords on readout NV and targets
    x_voltages, y_voltages = build_voltages_from_list(start_coords_drift, 
                                              [start_coords_drift, pulse_coords_drift], num_reps)
    # print(y_voltages)
    # return
    # Load the galvo
    xyz_server = tool_belt.get_xyz_server(cxn)
    xyz_server.load_arb_scan_xy(x_voltages, y_voltages, int(period))

    # We'll be lookign for three samples each repetition with how I have
    # the sequence set up
    total_num_samples = 3*num_reps
    readout_counts = collect_counts(cxn, num_reps, total_num_samples, seq_args_string,apd_indices)   
    start_counts = readout_counts[0::2]
    target_counts = readout_counts[1::2]
    
    avg_start_counts = numpy.average(start_counts)
    ste_start_counts = stats.sem(start_counts)
    avg_target_counts = numpy.average(target_counts)
    ste_target_counts = stats.sem(target_counts)
    
    dx = pulse_coords_drift[0] - start_coords_drift[0]
    dy = pulse_coords_drift[1] - start_coords_drift[1]
    pulse_r = numpy.sqrt((dx)**2 + dy**2)
    # print(pulse_r)
    
    fig_1D, ax_1D = plt.subplots(1, 1, figsize=(6, 6))
    ax_1D.errorbar([0, pulse_r], 
               [avg_start_counts, avg_target_counts],
               yerr = [ste_start_counts,ste_target_counts ], marker = 'o',  ls = 'none')
    ax_1D.set_xlabel('r (V)')
    ax_1D.set_ylabel('Average counts')
    ax_1D.set_title('{} nm {} ms init pulse \n{} nm {} ms CPG pulse\n{} nm {} ms {} V readout pulse'.\
                    format(init_color, initialization_time*1e-6,
                           pulse_color, pulse_time/10**6,
                           readout_color, charge_readout_time/10**6, charge_readout_laser_power))
    
    timestamp = tool_belt.get_time_stamp()
    raw_data = {'timestamp': timestamp,
            'nv_sig': nv_sig,
            'pulse_coords': pulse_coords,
            'num_reps':num_reps,
            'avg_start_counts': avg_start_counts.tolist(),
            'avg_start_counts-units': 'counts',
            'avg_target_counts': avg_target_counts.tolist(),
            'avg_target_counts-units': 'counts',
            'start_counts': start_counts,
            'start_counts-units': 'counts',
            'target_counts': target_counts,
            'target_counts-units': 'counts',
            }

    file_path = tool_belt.get_file_path('SPaCE.py', timestamp, nv_sig['name'])
    tool_belt.save_raw_data(raw_data, file_path)
    tool_belt.save_figure(fig_1D, file_path)
    return start_counts, target_counts

#%%
if __name__ == '__main__':
    # load the data here
    sample_name = 'sandia'
    apd_indices = [1]
    green_laser = "integrated_520"
    yellow_laser = 'laserglow_589'
    red_laser = 'cobolt_638'
    green_power= 10    

            
    
    nv_sig = {  
        "coords":[-0.732, 0.198, 6.617],
        "name": "{}-NV-R21_a3_r10_c6".format(sample_name,),
        "disable_opt":False,
        "ramp_voltages": True,
        "expected_count_rate":15,
        
        "imaging_laser":green_laser,
        "imaging_laser_power": None,
        "imaging_readout_dur": 1e7,        
        
        # "initialize_laser": red_laser, 
        # "initialize_laser_power": 0.67,
        # "initialize_laser_dur":  1e5,
        # "CPG_laser": green_laser, 
        # "CPG_laser_power": None,
        # "CPG_laser_dur":  1e5,
        
       #"initialize_laser": red_laser, 
       # "initialize_laser_power": 0.67,
       # "initialize_laser_dur":  1e5,
       #  "CPG_laser": red_laser, 
       #  "CPG_laser_power":0.67,
       #  "CPG_laser_dur": 1e5,
        
        
       "initialize_laser": green_laser, 
        "initialize_laser_power": None,
         "initialize_laser_dur":  1e4,
        "CPG_laser": green_laser, 
        "CPG_laser_power":None,
         "CPG_laser_dur":  1e8,
        
       # "initialize_laser": green_laser, 
       #   "initialize_laser_power": None,
       #   "initialize_laser_dur":  1e7,
       #   "CPG_laser": red_laser, 
       #   "CPG_laser_power":0.67,
       #   "CPG_laser_dur":  1e6,       
        
        "charge_readout_laser": yellow_laser,
        'charge_readout_laser_filter': 'nd_1.0',
        "charge_readout_laser_power": 0.15,
        "charge_readout_laser_dur":50e6,
        
        # "collection_filter": "715_lp",
        "collection_filter": "715_sp+630_lp",
        "magnet_angle": None,
        "resonance_LOW":2.87,"rabi_LOW": 150,
        "uwave_power_LOW": 15.5,  # 15.5 max
        "resonance_HIGH": 2.932,
        "rabi_HIGH": 59.6,
        "uwave_power_HIGH": 14.5,
    }  # 14.5 max
    
    
    try:
        #for SiV spot r10 c6
        # pulse_coords = [-0.739, 0.191, 6.617] # closer NV
        # pulse_coords = [-0.732, 0.198, 6.617] #farther away
        # pulse_coords = [-0.752, 0.199, 6.617] #no NV
        
        # pulse_coords = [-0.720, 0.208, 6.617] #nothing
        pulse_coords = [-0.743, 0.187, 6.617] # siv
        
        num_reps = int(1e2)
        for t in [1e5,]:
        # for t in [0.52, 0.53, 0.54, 0.55]:
            nv_sig_copy = copy.deepcopy(nv_sig)
            # nv_sig_copy['charge_readout_laser_power'] = t
            measure(nv_sig_copy, pulse_coords, apd_indices, num_reps)
    
    finally:
        # Reset our hardware - this should be done in each routine, but
        # let's double check here
        tool_belt.reset_cfm()
        # Kill safe stop
        if tool_belt.check_safe_stop_alive():
            print('\n\nRoutine complete. Press enter to exit.')
            tool_belt.poll_safe_stop()