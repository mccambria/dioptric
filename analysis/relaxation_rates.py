# -*- coding: utf-8 -*-
"""
Created on Fri May 31 11:06:46 2019

This routine takes the sets of data we take for relaxation measurments (prepare
in +1, readout in -1, etc) and calculates the relaxation rates, omega and
gamma. It calculates the values for each run of the data (num_runs). It will
then allow us to average the value for the relaxation rate and take a standard
deviation.

This file only works if all the experiments in a folder have the same number
of num_runs

@author: Aedan
"""
import os
import numpy
import json
from scipy import asarray as ar,exp
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

#%%

def zero_relaxation_eq(t, omega, amp, offset):
    return offset + amp * exp(-3 * omega * t)

#%%
    
def plus_relaxation_eq(t, gamma, omega, amp, offset):
    return offset + amp * exp(-(omega + gamma * 2) * t)

# %%

def relaxation_rate_analysis(folder_name):
    
    directory = 'G:/Shared drives/Kolkowitz Lab Group/nvdata/t1_double_quantum/' 
    
    # Create a list of all the files in the folder for one experiment
    file_list = []
    for file in os.listdir('{}/{}'.format(directory, folder_name)):
        if file.endswith(".txt"):
            file_list.append(file)
      
    # Get the number of runs to create the empty arrays from the first file in 
    # the list. This requires all the relaxation measurements to have the same
    # num_runs
    file = file_list[0]
    with open('{}/{}/{}'.format(directory, folder_name, file)) as json_file:
        data = json.load(json_file)
        num_runs = data['num_runs']
        
    # Prepare the arrays to fill with data. NaN will be first value
    zero_zero_sig_counts = numpy.ones((num_runs, 1)) * numpy.nan
    zero_zero_ref_counts = numpy.copy(zero_zero_sig_counts)
    zero_plus_sig_counts = numpy.copy(zero_zero_sig_counts)
    zero_plus_ref_counts = numpy.copy(zero_zero_sig_counts)
    plus_plus_sig_counts = numpy.copy(zero_zero_sig_counts)
    plus_plus_ref_counts = numpy.copy(zero_zero_sig_counts)
    plus_minus_sig_counts = numpy.copy(zero_zero_sig_counts)
    plus_minus_ref_counts = numpy.copy(zero_zero_sig_counts)
    
    zero_zero_time = numpy.ones(1) * numpy.nan
    zero_plus_time = numpy.copy(zero_zero_time)
    plus_plus_time = numpy.copy(zero_zero_time)
    plus_minus_time = numpy.copy(zero_zero_time)
    
    # Create lists to store the omega and gamma rates
    omega_rate_list = []
    omega_amp_list = []
    omega_offset_list = []
    gamma_list = []
    gamma_amp_list = []
    gamma_offset_list = []
    
    # %% Unpack the data
    
    # Unpack the data and sort into arrays. This allows multiple experiments of 
    # the same type (ie (1,-1) to be correctly sorted into one array
    for file in file_list:
        with open('{}/{}/{}'.format(directory, folder_name, file)) as json_file:
            data = json.load(json_file)
            
            init_state = data['init_state']
            read_state = data['read_state']
            
            sig_counts = numpy.array(data['sig_counts'])
            ref_counts = numpy.array(data['ref_counts'])
            
            relaxation_time_range = numpy.array(data['relaxation_time_range'])
            min_relaxation_time, max_relaxation_time = relaxation_time_range / 10**6
            num_steps = data['num_steps']

            # time should be in microseconds
            time_array = numpy.linspace(min_relaxation_time, max_relaxation_time,
                          num=num_steps) 
            
            # Check to see which data set the file is for, and append the data
            # to the corresponding array
            if init_state == 0 and read_state == 0:
                zero_zero_sig_counts = numpy.append(zero_zero_sig_counts, 
                                                    sig_counts, axis = 1)
                zero_zero_ref_counts = numpy.append(zero_zero_ref_counts, 
                                                    ref_counts, axis = 1)
                zero_zero_time = numpy.append(zero_zero_time, time_array)
                
                zero_zero_time_linspace = numpy.linspace(min_relaxation_time, 
                                             max_relaxation_time, num=1000)
                
            if init_state == 0 and read_state == 1:
                zero_plus_sig_counts = numpy.append(zero_plus_sig_counts, 
                                                    sig_counts, axis = 1)
                zero_plus_ref_counts = numpy.append(zero_plus_ref_counts, 
                                                    ref_counts, axis = 1)
                zero_plus_time = numpy.append(zero_plus_time, time_array)

            if init_state == 1 and read_state == 1:
                plus_plus_sig_counts = numpy.append(plus_plus_sig_counts, 
                                                    sig_counts, axis = 1)
                plus_plus_ref_counts = numpy.append(plus_plus_ref_counts, 
                                                    ref_counts, axis = 1)
                plus_plus_time = numpy.append(plus_plus_time, time_array)
                
                plus_plus_time_linspace = numpy.linspace(min_relaxation_time, 
                                             max_relaxation_time, num=1000)
                
            if init_state == 1 and read_state == -1:
                plus_minus_sig_counts = numpy.append(plus_minus_sig_counts, 
                                                    sig_counts, axis = 1)
                plus_minus_ref_counts = numpy.append(plus_minus_ref_counts, 
                                                    ref_counts, axis = 1)
                plus_minus_time = numpy.append(plus_minus_time, time_array)
    
    # Delete the NaNs from all the arrays. There might be a better way to fill
    # the arrays, but this should work for now
    zero_zero_sig_counts = numpy.delete(zero_zero_sig_counts, 0, axis = 1)
    zero_zero_ref_counts = numpy.delete(zero_zero_ref_counts, 0, axis = 1)
    zero_zero_time = numpy.delete(zero_zero_time, 0)
    
    zero_plus_sig_counts = numpy.delete(zero_plus_sig_counts, 0, axis = 1)
    zero_plus_ref_counts = numpy.delete(zero_plus_ref_counts, 0, axis = 1)
    zero_plus_time = numpy.delete(zero_plus_time, 0)
    
    plus_plus_sig_counts = numpy.delete(plus_plus_sig_counts, 0, axis = 1)
    plus_plus_ref_counts = numpy.delete(plus_plus_ref_counts, 0, axis = 1)
    plus_plus_time = numpy.delete(plus_plus_time, 0)
    
    plus_minus_sig_counts = numpy.delete(plus_minus_sig_counts, 0, axis = 1)
    plus_minus_ref_counts = numpy.delete(plus_minus_ref_counts, 0, axis = 1)
    plus_minus_time = numpy.delete(plus_minus_time, 0)

# %% Fit to the (0,0) - (0,1) data to find Omega

    # let's first just try finding the relaxation rate for all the data together.
    
    zero_zero_avg_sig_counts = numpy.average(zero_zero_sig_counts, axis=0)
    zero_zero_avg_ref_counts = numpy.average(zero_zero_ref_counts, axis=0)
    
    zero_zero_norm_avg_sig = zero_zero_avg_sig_counts / zero_zero_avg_ref_counts
           
    zero_plus_avg_sig_counts = numpy.average(zero_plus_sig_counts, axis=0)
    zero_plus_avg_ref_counts = numpy.average(zero_plus_ref_counts, axis=0)
    
    zero_plus_norm_avg_sig = zero_plus_avg_sig_counts / zero_plus_avg_ref_counts 
    
    # Define the counts for the zero relaxation equation
    zero_relaxation_counts =  zero_zero_norm_avg_sig - zero_plus_norm_avg_sig
    
    init_params = (1.2, 0.36, 0)
    opti_params, cov_arr = curve_fit(zero_relaxation_eq, zero_zero_time,
                                        zero_relaxation_counts, p0 = init_params)

    omega_rate_list.append(opti_params[0])
    omega_amp_list.append(opti_params[1])
    omega_offset_list.append(opti_params[2])
    omega = opti_params[0]
    
    fig, axes_pack = plt.subplots(1, 2, figsize=(17, 8))
    ax = axes_pack[0]
    ax.plot(zero_zero_time, zero_relaxation_counts, 'bo', label = 'data')
    ax.plot(zero_zero_time_linspace, 
            zero_relaxation_eq(zero_zero_time_linspace, *opti_params), 
            'r', label = 'fit')  
    
# %% Fit to the (1,1) - (1,-1) data to find Gamma
    
    plus_plus_avg_sig_counts = numpy.average(plus_plus_sig_counts, axis=0)
    plus_plus_avg_ref_counts = numpy.average(plus_plus_ref_counts, axis=0)
    
    plus_plus_norm_avg_sig = plus_plus_avg_sig_counts / plus_plus_avg_ref_counts
           
    plus_minus_avg_sig_counts = numpy.average(plus_minus_sig_counts, axis=0)
    plus_minus_avg_ref_counts = numpy.average(plus_minus_ref_counts, axis=0)
    
    plus_minus_norm_avg_sig = plus_minus_avg_sig_counts / plus_minus_avg_ref_counts
    
    # Define the counts for the plus relaxation equation
    plus_relaxation_counts =  plus_plus_norm_avg_sig - plus_minus_norm_avg_sig
    
    
    init_params = (100, 0.50, 0)
    # create a temporary fitting equation that passes in the omega value just found
    plus_relaxation_tmp = lambda t, gamma, amp, offset: plus_relaxation_eq(t, gamma, omega, amp, offset)
    opti_params, cov_arr = curve_fit(plus_relaxation_tmp, 
                                     plus_plus_time, plus_relaxation_counts, p0 = init_params)
    
    gamma_rate_list.append(opti_params[0])
    gamma_amp_list.append(opti_params[2])
    gamma_offset_list.append(opti_params[3])
    
    ax = axes_pack[1]
    ax.plot(plus_plus_time, plus_relaxation_counts, 'bo')
    ax.plot(plus_plus_time_linspace, 
            plus_relaxation_tmp(plus_plus_time_linspace, *opti_params), 
            'r', label = 'fit')   

    fig.canvas.draw()

# %%
    
if __name__ == '__main__':
    
    relaxation_rate_analysis('2019-05-10-NV1_32MHzSplitting_important_data')