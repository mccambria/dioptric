#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 20:39:55 2019

@author: yanfeili
"""

# %% import
import utils.tool_belt as tool_belt
import majorroutines.optimize as optimize
import numpy
import os
import matplotlib.pyplot as plt
import labrad

def get_Probability_distribution(aList):
        
    def get_unique_value(aList):
        unique_value_list = []
        for i in range(0,len(aList)):
            if aList[i] not in unique_value_list:
                unique_value_list.append(aList[i])
        return unique_value_list
    unique_value = get_unique_value(aList)
    relative_frequency = []
    for i in range(0,len(unique_value)):
        relative_frequency.append(aList.count(unique_value[i])/ (len(aList)))
        
    return unique_value, relative_frequency

#def create_figure():
#    fig, ax = plt.subplots(1, 1, figsize=(10, 8.5))
#    ax.set_xlabel('number of photons (n)')
#    ax.set_ylabel('P(n)')
#    
#    return fig
    
#%% Main
# Connect to labrad in this file, as opposed to control panel
def main(nv_sig, apd_indices, aom_ao_589_pwr,readout_time,num_runs, num_reps):
    
    with labrad.connect() as cxn:
        main_with_cxn(cxn, nv_sig, apd_indices, aom_ao_589_pwr,readout_time,num_runs, num_reps)

def main_with_cxn(cxn, nv_sig, apd_indices, aom_ao_589_pwr,readout_time,num_runs, num_reps):

    tool_belt.reset_cfm(cxn)

# %% Initial Calculation and setup
#    apd_indices = [0]
    
    shared_params = tool_belt.get_shared_parameters_dict(cxn)

    #delay of aoms and laser
    aom_delay = shared_params['532_aom_delay'] 
        
#    # Analyze the sequence
#    seq_args = [gate_time, aom_delay589 ,apd_indices[0], aom_power]
#    seq_args_string = tool_belt.encode_seq_args(seq_args)
#    cxn.pulse_streamer.stream_load('photon_collections_under_589nm_sequence.py', seq_args_string)

    sig_counts = []
    
    # create a list to store the optimized coordinates
    opti_coords_list = []
    
#%% Collect data
    tool_belt.init_safe_stop()

        
    # Optimize
    opti_coords = optimize.main_with_cxn(cxn, nv_sig, apd_indices, 532)
    opti_coords_list.append(opti_coords)    
    
    for run_ind in range(num_runs):

        print('Run index: {}'. format(run_ind))

        # Break out of the while if the user says stop
        if tool_belt.safe_stop():
            break
        
        # Load the APD
        cxn.apd_tagger.start_tag_stream(apd_indices)
        
        seq_args = [readout_time, aom_delay, apd_indices[0], aom_ao_589_pwr]
    #        seq_args = [int(el) for el in seq_args]
        seq_args_string = tool_belt.encode_seq_args(seq_args)
        cxn.pulse_streamer.stream_immediate('photon_collections_under_589nm_sequence.py', num_reps, seq_args_string)
    
        # Get the counts
        new_counts = cxn.apd_tagger.read_counter_simple(num_reps)
        
        sig_counts.extend(new_counts)

    cxn.apd_tagger.stop_tag_stream()
    
#%% plot the data
    
    unique_value, relative_frequency = get_Probability_distribution(list(sig_counts))
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 8.5))
    
    ax.plot(unique_value, relative_frequency, 'bo')
    ax.set_xlabel('number of photons (n)')
    ax.set_ylabel('P(n)')

#%% Save data 
    timestamp = tool_belt.get_time_stamp()
    
    # turn the list of unique_values into pure integers, for saving
    unique_value = [int(el) for el in unique_value]
    sig_counts = [int(el) for el in sig_counts]
    
    raw_data = {'timestamp': timestamp,
            'nv_sig': nv_sig,
            'aom_ao_589_pwr': aom_ao_589_pwr,
            'aom_ao_589_pwr-unit':'V',
            'readout_time':readout_time,
            'readout_time_unit':'ns',
            'nv_sig-units': tool_belt.get_nv_sig_units(),
            'num_runs': num_runs,
            'sig_counts': sig_counts,
            'sig_counts-units': 'counts',
            'unique_values': unique_value,
            'unique_values-units': 'num of photons',
            'relative_frequency': relative_frequency,
            'relative_frequency-units': 'occurrences'
            }

    file_path = tool_belt.get_file_path(__file__, timestamp, nv_sig['name'])
    tool_belt.save_raw_data(raw_data, file_path)    
        
    tool_belt.save_figure(fig, file_path)
