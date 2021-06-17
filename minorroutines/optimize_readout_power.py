#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 20:29:49 2020

@author: yanfeili
"""

# %% import
import utils.tool_belt as tool_belt
import majorroutines.optimize as optimize
import numpy
import matplotlib.pyplot as plt
import labrad
import time
from scipy.optimize import curve_fit
import scipy.stats
import scipy.special
import math  
import photonstatistics as ps

#%% function

def get_average_photon_counts(readout_time, photon_number_list):
    photon_counts = numpy.array(photon_number_list)/numpy.array(readout_time)
    average_photon_counts = numpy.average(photon_counts)
    return average_photon_counts

def get_optimized_power(power_range,average_count_list):
    max_index = average_count_list.index(max(average_count_list))
    if max_index == 0 or 999:
        print('optimal power is out of range')
    else:
        return power_range[max_index]
    

#%% Main
# Connect to labrad in this file, as opposed to control panel
def main(nv_sig, apd_indices, aom_ao_589_pwr_range,readout_time):
    
    with labrad.connect() as cxn:
        main_with_cxn(cxn, nv_sig, apd_indices, aom_ao_589_pwr_range,readout_time)

def main_with_cxn(cxn, nv_sig, apd_indices, aom_ao_589_pwr_range,readout_time):

    tool_belt.reset_cfm(cxn)

# %% Initial Calculation and setup  
    shared_params = tool_belt.get_shared_parameters_dict(cxn)
    
    #Define some parameters
    
    #delay of aoms
    aom_delay = shared_params['532_aom_delay'] 
    
    readout_power = 0 
    #readout_power in unit of microwatts TBD'
    aom_power = numpy.sqrt((readout_power - 0.432)/1361.811) #uW

    
    reionization_time = 1*10**6
    illumination_time = readout_time + 10**3
    
    
    aom_ao_589_pwr = numpy.linspace(aom_ao_589_pwr_range[0],aom_ao_589_pwr_range[1], 1000)
    
    #iterate determine_n_thresh over the power interval 
    #to find the greatest contrast between sig and background
    
    average_count_list = []

    for i in range(len(aom_ao_589_pwr)):        
        
    # Set up our data structure, an array of NaNs that we'll fill
    # we repeatively collect photons for tR 
        
        sig_counts=[]
        opti_coords_list = []
        
        num_runs = 2
        num_reps = 100
    
    
 #%% Collect data
        tool_belt.init_safe_stop()
    
        
        for run_ind in range(num_runs):
    
            print('Run index: {}'. format(run_ind))
                    
            # Optimize
            opti_coords = optimize.main_with_cxn(cxn, nv_sig, apd_indices, 532)
            opti_coords_list.append(opti_coords)   
            
            drift = numpy.array(tool_belt.get_drift())
            coords = numpy.array(nv_sig['coords'])
            
            coords_drift = coords - drift
            
            cxn.galvo.write(coords_drift[0], coords_drift[1])
            cxn.objective_piezo.write(coords_drift[2])
            
            #  set filter slider according to nv_sig
            ND_filter = nv_sig['nd_filter']
            cxn.filter_slider_ell9k.set_filter(ND_filter)
            time.sleep(0.1)
            
    
            # Break out of the while if the user says stop
            if tool_belt.safe_stop():
                break
            
            # Load the APD
            cxn.apd_tagger.start_tag_stream(apd_indices)
            
            seq_args = [readout_time, reionization_time, illumination_time, 
                        aom_delay ,apd_indices[0], aom_ao_589_pwr[i]]
            seq_args_string = tool_belt.encode_seq_args(seq_args)
            cxn.pulse_streamer.stream_immediate('determine_n_thresh.py', num_reps, seq_args_string)
        
            # Get the counts
            new_counts = cxn.apd_tagger.read_counter_simple(num_reps)
            
            sig_counts.extend(new_counts)
    
        cxn.apd_tagger.stop_tag_stream()
        
        average_count_list.append(get_average_photon_counts(readout_time, sig_counts))
     
    optimal_readout_power = get_optimized_power(aom_ao_589_pwr, average_count_list)
    print('optimal readout power: '+str(optimal_readout_power))
    print('optimal average counts: '+str(max(average_count_list)))