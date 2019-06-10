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

import utils.tool_belt as tool_belt

#%%

def zero_relaxation_eq(t, omega, amp, offset):
    return offset + amp * exp(-3 * omega * t)

#%%
    
def plus_relaxation_eq(t, gamma, omega, amp, offset):
    return offset + amp * exp(-(omega + gamma * 2) * t)

# %%

def relaxation_rate_analysis(folder_name, bin_size, doPlot = False,
                             save_data = True):
    
    directory = 'G:/Shared drives/Kolkowitz Lab Group/nvdata/t1_double_quantum/' 
    
    # Create a list of all the files in the folder for one experiment
    file_list = []
    for file in os.listdir('{}/{}'.format(directory, folder_name)):
        if file.endswith(".txt") and not file.endswith("bins.txt"):
            file_list.append(file)
      
    # Get the number of runs to create the empty arrays from the first file in 
    # the list. This requires all the relaxation measurements to have the same
    # num_runs
    file = file_list[0]
    with open('{}/{}/{}'.format(directory, folder_name, file)) as json_file:
        data = json.load(json_file)
        num_runs_set = data['num_runs']
        
    # Prepare the arrays to fill with data. NaN will be first value
    zero_zero_sig_counts = numpy.ones((num_runs_set, 1)) * numpy.nan
    zero_zero_ref_counts = numpy.copy(zero_zero_sig_counts)
    zero_plus_sig_counts = numpy.copy(zero_zero_sig_counts)
    zero_plus_ref_counts = numpy.copy(zero_zero_sig_counts)
    plus_plus_sig_counts = numpy.copy(zero_zero_sig_counts)
    plus_plus_ref_counts = numpy.copy(zero_zero_sig_counts)
    plus_minus_sig_counts = numpy.copy(zero_zero_sig_counts)
    plus_minus_ref_counts = numpy.copy(zero_zero_sig_counts)
    
    zero_zero_time = numpy.ones(1) * numpy.nan
#    zero_plus_time = numpy.copy(zero_zero_time)
    plus_plus_time = numpy.copy(zero_zero_time)
#    plus_minus_time = numpy.copy(zero_zero_time)
    
    # Create lists to store the omega and gamma rates
    omega_rate_list = []
    omega_amp_list = []
    omega_offset_list = []
    gamma_rate_list = []
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
            num_runs = data['num_runs']

            # time should be in microseconds
            time_array = numpy.linspace(min_relaxation_time, max_relaxation_time,
                          num=num_steps) 
            
            # We will want to put the MHz splitting in the file metadata
            uwave_freq_init = data['uwave_freq_init']
            uwave_freq_read = data['uwave_freq_read']
            
            # Check that the num_runs is consistent. If not, raise an error
            if num_runs_set != num_runs:
                print('Error, num_runs not consistent in file {}'.format(file))
                break
            
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
                
                splitting_MHz = abs(uwave_freq_init - uwave_freq_read) * 10**3
    
# Some error handeling if the count arras don't match up            
    if len(zero_zero_sig_counts) != len(zero_plus_sig_counts): 
                    
         print('Error: length of zero_zero_sig_counts and zero_plus_sig_counts do not match')
       
    if len(plus_plus_sig_counts) != len(plus_minus_sig_counts):
        print('Error: length of plus_plus_sig_counts and plus_minus_sig_counts do not match')
        
    # Delete the NaNs from all the arrays. There might be a better way to fill
    # the arrays, but this should work for now
    zero_zero_sig_counts = numpy.delete(zero_zero_sig_counts, 0, axis = 1)
    zero_zero_ref_counts = numpy.delete(zero_zero_ref_counts, 0, axis = 1)
    zero_zero_time = numpy.delete(zero_zero_time, 0)
    
    zero_plus_sig_counts = numpy.delete(zero_plus_sig_counts, 0, axis = 1)
    zero_plus_ref_counts = numpy.delete(zero_plus_ref_counts, 0, axis = 1)
    
    plus_plus_sig_counts = numpy.delete(plus_plus_sig_counts, 0, axis = 1)
    plus_plus_ref_counts = numpy.delete(plus_plus_ref_counts, 0, axis = 1)
    plus_plus_time = numpy.delete(plus_plus_time, 0)
    
    plus_minus_sig_counts = numpy.delete(plus_minus_sig_counts, 0, axis = 1)
    plus_minus_ref_counts = numpy.delete(plus_minus_ref_counts, 0, axis = 1)

# %% Fit the data based on the bin size
    
    i = 0
    
    while i < (num_runs - 1):
        #Fit to the (0,0) - (0,1) data to find Omega
     
        zero_zero_avg_sig_counts = numpy.average(zero_zero_sig_counts[i:i+bin_size, ::], axis=0)
        zero_zero_avg_ref_counts = numpy.average(zero_zero_ref_counts[i:i+bin_size, ::], axis=0)
        
        zero_zero_norm_avg_sig = zero_zero_avg_sig_counts / zero_zero_avg_ref_counts
               
        zero_plus_avg_sig_counts = numpy.average(zero_plus_sig_counts[i:i+bin_size, ::], axis=0)
        zero_plus_avg_ref_counts = numpy.average(zero_plus_ref_counts[i:i+bin_size, ::], axis=0)
        
        zero_plus_norm_avg_sig = zero_plus_avg_sig_counts / zero_plus_avg_ref_counts 
    
        # Define the counts for the zero relaxation equation
        zero_relaxation_counts =  zero_zero_norm_avg_sig - zero_plus_norm_avg_sig
        
        init_params = (1.0, 0.4, 0)
        opti_params, cov_arr = curve_fit(zero_relaxation_eq, zero_zero_time,
                                            zero_relaxation_counts, p0 = init_params)

        omega_rate_list.append(opti_params[0])
        omega_amp_list.append(opti_params[1])
        omega_offset_list.append(opti_params[2])
        omega = opti_params[0]
        
        # Plotting the data
        if doPlot:
            fig, axes_pack = plt.subplots(1, 2, figsize=(17, 8))
            ax = axes_pack[0]
            ax.plot(zero_zero_time, zero_relaxation_counts, 'bo', label = 'data')
            ax.plot(zero_zero_time_linspace, 
                    zero_relaxation_eq(zero_zero_time_linspace, *opti_params), 
                    'r', label = 'fit') 
            ax.set_xlabel('Relaxation time (ms)')
            ax.set_ylabel('Normalized signal Counts')
            ax.set_title('(0,0) - (0,+1)')
            ax.legend()
            text = r'$\Omega = $ {} kHz'.format('%.2f'%opti_params[0])

            props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
            ax.text(0.55, 0.95, text, transform=ax.transAxes, fontsize=12,
                    verticalalignment='top', bbox=props)
    
    # %% Fit to the (1,1) - (1,-1) data to find Gamma
        
        plus_plus_avg_sig_counts = numpy.average(plus_plus_sig_counts[i:i+bin_size, ::], axis=0)
        plus_plus_avg_ref_counts = numpy.average(plus_plus_ref_counts[i:i+bin_size, ::], axis=0)
        
        plus_plus_norm_avg_sig = plus_plus_avg_sig_counts / plus_plus_avg_ref_counts
               
        plus_minus_avg_sig_counts = numpy.average(plus_minus_sig_counts[i:i+bin_size, ::], axis=0)
        plus_minus_avg_ref_counts = numpy.average(plus_minus_ref_counts[i:i+bin_size, ::], axis=0)
        
        plus_minus_norm_avg_sig = plus_minus_avg_sig_counts / plus_minus_avg_ref_counts
        
        # Define the counts for the plus relaxation equation
        plus_relaxation_counts =  plus_plus_norm_avg_sig - plus_minus_norm_avg_sig
        
        
        init_params = (100, 0.40, 0)
        
        # create a temporary fitting equation that passes in the omega value just found
        plus_relaxation_tmp = lambda t, gamma, amp, offset: plus_relaxation_eq(t, gamma, omega, amp, offset)
        opti_params, cov_arr = curve_fit(plus_relaxation_tmp, 
                                         plus_plus_time, plus_relaxation_counts, p0 = init_params)
        
        gamma_rate_list.append(opti_params[0])
        gamma_amp_list.append(opti_params[1])
        gamma_offset_list.append(opti_params[2])
        
        # Advance the index
        i = i + bin_size
        
        # Plotting
        if doPlot:
            ax = axes_pack[1]
            ax.plot(plus_plus_time, plus_relaxation_counts, 'bo')
            ax.plot(plus_plus_time_linspace, 
                    plus_relaxation_tmp(plus_plus_time_linspace, *opti_params), 
                    'r', label = 'fit')   
            ax.set_xlabel('Relaxation time (ms)')
            ax.set_ylabel('Normalized signal Counts')
            ax.set_title('(+1,+1) - (+1,-1)')
            ax.legend()
            text = r'$\gamma = $ {} kHz'.format('%.2f'%opti_params[0])

            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            ax.text(0.55, 0.95, text, transform=ax.transAxes, fontsize=12,
                    verticalalignment='top', bbox=props)
            
            fig.canvas.draw()
            fig.canvas.flush_events()
    
    omega_average = numpy.average(omega_rate_list)
    omega_stdev = numpy.std(omega_rate_list)
    
    gamma_average = numpy.average(gamma_rate_list)
    gamma_stdev = numpy.std(gamma_rate_list)
    
    print('Omega list: {} \nGamma list: {}'.format(omega_rate_list, gamma_rate_list))
    
# %% Saving data
    
    num_bins = int(num_runs / bin_size)
    if save_data: 
        time_stamp = tool_belt.get_time_stamp()
        raw_data = {'time_stamp': time_stamp,
                    'level_splitting': splitting_MHz,
                    'level_splitting-units': 'MHz',
                    'num_runs': num_runs,
                    'bin_size': bin_size,
                    'num_bins': num_bins,
                    'omega_average': omega_average,
                    'omega_average-units': 'kHz',
                    'omega_stdev': omega_stdev,
                    'omega_stdev-units': 'kHz',
                    'gamma_average': gamma_average,
                    'gamma_average-units': 'kHz',
                    'gamma_stdev': gamma_stdev,
                    'gamma_stdev-units': 'kHz',
                    'omega_rate_list': omega_rate_list,
                    'omega_rate_list-units': 'kHz',
                    'omega_amp_list': omega_rate_list,
                    'omega_amp_list-units': 'arb',
                    'omega_offset_list': omega_offset_list,
                    'omega_offset_list-units': 'arb',
                    'gamma_rate_list': gamma_rate_list,
                    'gamma_rate_list-units': 'kHz',
                    'gamma_amp_list': gamma_rate_list,
                    'gamma_amp_list-units': 'arb',
                    'gamma_offset_list': gamma_offset_list,
                    'gamma_offset_list-units': 'arb'}
        
        file_name = str('%.1f'%splitting_MHz) + '_MHz_splitting_' + str(num_bins) + '_bins' 
        file_path = '{}/{}/{}'.format(directory, folder_name, file_name)
        print(file_path)
        # tool_belt.save_raw_data(raw_data, file_path)
        
        with open(file_path + '.txt', 'w') as file:
            json.dump(raw_data, file, indent=2)
      
        return num_bins, omega_average, omega_stdev, gamma_average, gamma_stdev
    
# %%
    
if __name__ == '__main__':
    
    bin_size_list =  numpy.append(numpy.array([1]), numpy.linspace(2,40,20)).tolist()
    
    num_bins_list = []
    omega_value_list = []
    omega_stdev_list = []
    gamma_value_list = []
    gamma_stdev_list = []
    
    
    for bin_size in bin_size_list:
        retvals = relaxation_rate_analysis('2019-05-10-NV1_32MHzSplitting_important_data', bin_size,
                            False, False)
        num_bins_list.append(retvals[0])
        omega_value_list.append(retvals[1])
        omega_stdev_list.append(retvals[2])
        gamma_value_list.append(retvals[3])
        gamma_stdev_list.append(retvals[4])
        
    fig, ax = plt.subplots(1, 2, figsize=(14, 8))
    ax.plot(num_bins_list, omega_value_list, 'bo')
        
        
        