# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 09:52:43 2019

This analysis script will take a set of T1 experiments and fit the fucntions
defined in the Myer's paper ((0,0) - (0,1) and (1,1) - (1,-1)) to extract a 
rate for the two modified data set exponential fits. It allows the data to be
split into different bins, which is used for the analysis of stdev. This file 
does not convert these rates into omega and gamma, this function passes these 
basic rates onto the stdev analysis file.

@author: Aedan
"""

# %% Imports

import numpy
from scipy import exp
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

import utils.tool_belt as tool_belt

# %% Constants

data_folder = 't1_double_quantum'

# %% Functions

# The exponential function used to fit the data

def exp_eq(t, rate, amp, offset):
    return offset + amp * exp(- rate * t)

#def zero_relaxation_eq(t, omega, amp, offset):
#    return offset + amp * exp(-3 * omega * t)
#
#    
#def plus_relaxation_eq(t, gamma, omega, amp, offset):
#    return offset + amp * exp(-(omega + gamma * 2) * t)

# %% Main
    
def main(folder_name, num_bins, doPlot = False, save_data = True):
    
    print('Number of bins: {}'.format(num_bins))

    # Get the file list from this folder
    file_list = tool_belt.get_file_list(data_folder, '.txt', folder_name)

    # Get the number of runs to create the empty arrays from the last file in 
    # the list. This requires all the relaxation measurements to have the same
    # num_runs
    for file in file_list:
        data = tool_belt.get_raw_data(data_folder, file[:-4], folder_name)

        try:
            num_runs_set = data['num_runs']
        except Exception:
            continue
        
    bin_size = int(num_runs_set / num_bins)
        
    # Define booleans to be used later in putting data into usable arrays
    zero_zero_bool = False
    zero_plus_bool = False
    plus_plus_bool = False
    plus_minus_bool = False
    
    o_fit_failed_list = []
    g_fit_failed_list = []
    
    # Create lists to store the omega and gamma rates
    o_rate_list = []
#    omega_unc_list = []
    o_amp_list = []
    o_offset_list = []
    g_rate_list = []
#    gamma_unc_list = []
    g_amp_list = []
    g_offset_list = []
    
    # %% Unpack the data
    
    # Unpack the data and sort into arrays. This allows multiple experiments of 
    # the same type (ie (1,-1)) to be correctly sorted into one array
    for file in file_list:
        data = tool_belt.get_raw_data(data_folder, file[:-4], folder_name)
        try:
                
            init_state = data['init_state']
            read_state = data['read_state']
            
            sig_counts = numpy.array(data['sig_counts'])
            ref_counts = numpy.array(data['ref_counts'])
            
            relaxation_time_range = numpy.array(data['relaxation_time_range'])
            # time is in microseconds
            min_relaxation_time, max_relaxation_time = relaxation_time_range / 10**6
            num_steps = data['num_steps']
            num_runs = data['num_runs']

            time_array = numpy.linspace(min_relaxation_time, 
                                        max_relaxation_time, num=num_steps) 
            
            
            # Check that the num_runs is consistent. If not, raise an error
            if num_runs_set != num_runs:
                print('Error, num_runs not consistent in file {}'.format(file))
                break
            
            # Check to see which data set the file is for, and append the data
            # to the corresponding array
            if init_state == 0 and read_state == 0:
                # Check to see if data has already been taken of this experiment
                # If it hasn't, then create arrays of the data.
                if zero_zero_bool == False:
                    zero_zero_sig_counts = sig_counts
                    zero_zero_ref_counts = ref_counts
                    zero_zero_time = time_array
                    
                    zero_zero_ref_max_time = max_relaxation_time
                    zero_zero_bool = True
                # If data has already been taken for this experiment, then check
                # to see if this current data is the shorter or longer measurement,
                # and either append before or after the prexisting data
                else:
                    
                    if max_relaxation_time > zero_zero_ref_max_time:
                        zero_zero_sig_counts = numpy.concatenate((zero_zero_sig_counts, 
                                                        sig_counts), axis = 1)
                        zero_zero_ref_counts = numpy.concatenate((zero_zero_ref_counts, 
                                                        ref_counts), axis = 1)
                        zero_zero_time = numpy.concatenate((zero_zero_time, time_array))
                        
                    elif max_relaxation_time < zero_zero_ref_max_time:
                        zero_zero_sig_counts = numpy.concatenate((sig_counts, 
                                              zero_zero_sig_counts), axis = 1)
                        zero_zero_ref_counts = numpy.concatenate((ref_counts, 
                                              zero_zero_ref_counts), axis = 1)
                        zero_zero_time = numpy.concatenate((time_array, zero_zero_time))
                
            if init_state == 0 and read_state == 1:
                # Check to see if data has already been taken of this experiment
                # If it hasn't, then create arrays of the data.
                if zero_plus_bool == False:
                    zero_plus_sig_counts = sig_counts
                    zero_plus_ref_counts = ref_counts
                    
                    zero_plus_ref_max_time = max_relaxation_time
                    zero_plus_bool = True
                # If data has already been taken for this experiment, then check
                # to see if this current data is the shorter or longer measurement,
                # and either append before or after the prexisting data
                else:
                    
                    if max_relaxation_time > zero_plus_ref_max_time:
                        zero_plus_sig_counts = numpy.concatenate((zero_plus_sig_counts, 
                                                        sig_counts), axis = 1)
                        zero_plus_ref_counts = numpy.concatenate((zero_plus_ref_counts, 
                                                        ref_counts), axis = 1)
                        
                    elif max_relaxation_time < zero_plus_ref_max_time:
                        zero_plus_sig_counts = numpy.concatenate((sig_counts, 
                                              zero_plus_sig_counts), axis = 1)
                        zero_plus_ref_counts = numpy.concatenate((ref_counts, 
                                              zero_plus_ref_counts), axis = 1)

            if init_state == 1 and read_state == 1:              
                # Check to see if data has already been taken of this experiment
                # If it hasn't, then create arrays of the data.
                if plus_plus_bool == False:
                    plus_plus_sig_counts = sig_counts
                    plus_plus_ref_counts = ref_counts
                    plus_plus_time = time_array
                    
                    plus_plus_ref_max_time = max_relaxation_time
                    plus_plus_bool = True
                # If data has already been taken for this experiment, then check
                # to see if this current data is the shorter or longer measurement,
                # and either append before or after the prexisting data
                else:
                    
                    if max_relaxation_time > plus_plus_ref_max_time:
                        plus_plus_sig_counts = numpy.concatenate((plus_plus_sig_counts, 
                                                        sig_counts), axis = 1)
                        plus_plus_ref_counts = numpy.concatenate((plus_plus_ref_counts, 
                                                        ref_counts), axis = 1)
                        plus_plus_time = numpy.concatenate((plus_plus_time, time_array))
                        
                    elif max_relaxation_time < plus_plus_ref_max_time:
                        plus_plus_sig_counts = numpy.concatenate((sig_counts, 
                                                          plus_plus_sig_counts), axis = 1)
                        plus_plus_ref_counts = numpy.concatenate((ref_counts, 
                                                          plus_plus_ref_counts), axis = 1)
                        plus_plus_time = numpy.concatenate((time_array, plus_plus_time))
                
            if init_state == 1 and read_state == -1:
                # We will want to put the MHz splitting in the file metadata
                uwave_freq_init = data['uwave_freq_init']
                uwave_freq_read = data['uwave_freq_read']
                
                # Check to see if data has already been taken of this experiment
                # If it hasn't, then create arrays of the data.
                if plus_minus_bool == False:
                    plus_minus_sig_counts = sig_counts
                    plus_minus_ref_counts = ref_counts
                    
                    plus_minus_ref_max_time = max_relaxation_time
                    plus_minus_bool = True
                # If data has already been taken for this experiment, then check
                # to see if this current data is the shorter or longer measurement,
                # and either append before or after the prexisting data
                else:
                    
                    if max_relaxation_time > plus_minus_ref_max_time:
                        plus_minus_sig_counts = numpy.concatenate((plus_minus_sig_counts, 
                                                        sig_counts), axis = 1)
                        plus_minus_ref_counts = numpy.concatenate((plus_minus_ref_counts, 
                                                        ref_counts), axis = 1)
                        
                    elif max_relaxation_time < plus_minus_ref_max_time:
                        plus_minus_sig_counts = numpy.concatenate((sig_counts, 
                                              plus_minus_sig_counts), axis = 1)
                        plus_minus_ref_counts = numpy.concatenate((ref_counts, 
                                              plus_minus_ref_counts), axis = 1)
                
                splitting_MHz = abs(uwave_freq_init - uwave_freq_read) * 10**3
                
        except Exception:
            continue
    
    # Some error handeling if the count arras don't match up            
    if len(zero_zero_sig_counts) != len(zero_plus_sig_counts): 
                    
         print('Error: length of zero_zero_sig_counts and zero_plus_sig_counts do not match')
       
    if len(plus_plus_sig_counts) != len(plus_minus_sig_counts):
        print('Error: length of plus_plus_sig_counts and plus_minus_sig_counts do not match')
    
    # %% Fit the data based on the bin size
        
    i = 0
    
    # For any number of bins except the maximum amount, we want to slice the
    # arrays from [i:i+bin_size-1]. However, this doesn't work for the maximum
    # amount of bins, when the bin_size is 1. In that case, we do want to take
    # slices [i:i+bin_size]
    if num_bins == num_runs:
        slice_size = bin_size
    else:
        slice_size = bin_size - 1
    
    while i < (num_runs):
        if doPlot:
            fig, axes_pack = plt.subplots(1, 2, figsize=(17, 8))
        
        #Fit to the (0,0) - (0,1) data to find Omega
        zero_zero_avg_sig_counts =  \
            numpy.average(zero_zero_sig_counts[i:i+slice_size, ::], axis=0)
        zero_zero_avg_ref_counts =  \
            numpy.average(zero_zero_ref_counts[i:i+slice_size, ::], axis=0)
        
        zero_zero_norm_avg_sig = zero_zero_avg_sig_counts / zero_zero_avg_ref_counts
               
        zero_plus_avg_sig_counts = \
            numpy.average(zero_plus_sig_counts[i:i+slice_size, ::], axis=0)
        zero_plus_avg_ref_counts = \
            numpy.average(zero_plus_ref_counts[i:i+slice_size, ::], axis=0)
        
        zero_plus_norm_avg_sig = zero_plus_avg_sig_counts / zero_plus_avg_ref_counts 
    
        # Define the counts for the zero relaxation equation
        zero_relaxation_counts =  zero_zero_norm_avg_sig - zero_plus_norm_avg_sig
        
        o_fit_failed = False
        g_fit_failed = False
    
        try:

            init_params = (1.0, 0.4, 0)
            opti_params, cov_arr = curve_fit(exp_eq, zero_zero_time,
                                         zero_relaxation_counts, p0 = init_params)
           
        except Exception:
            
            o_fit_failed = True
            o_fit_failed_list.append(o_fit_failed)
            
#            if doPlot:
#                ax = axes_pack[0]
#                ax.plot(zero_zero_time, zero_relaxation_counts, 'bo', label = 'data')
#                ax.set_xlabel('Relaxation time (ms)')
#                ax.set_ylabel('Normalized signal Counts')
#                ax.set_title('(0,0) - (0,+1)')
#                ax.legend()

        if not o_fit_failed:
            o_fit_failed_list.append(o_fit_failed)
            
#            o_rate = opti_params[0] / 3.0
#            omega_unc= cov_arr[0,0] / 3.0
            
            o_rate_list.append(opti_params[0])
#            omega_unc_list.append(omega_unc)
            o_amp_list.append(opti_params[1])
            o_offset_list.append(opti_params[2])
            

        
#            # Plotting the data
#            if doPlot:
#                zero_time_linspace = numpy.linspace(0, zero_zero_time[-1], num=1000)
#                ax = axes_pack[0]
#                ax.plot(zero_zero_time, zero_relaxation_counts, 'bo', label = 'data')
#                ax.plot(zero_time_linspace, 
#                        exp_eq(zero_time_linspace, *opti_params), 
#                        'r', label = 'fit') 
#                ax.set_xlabel('Relaxation time (ms)')
#                ax.set_ylabel('Normalized signal Counts')
#                ax.set_title('(0,0) - (0,+1)')
#                ax.legend()
#                text = r'$\Omega = $ {} kHz'.format('%.2f'%omega)
#    
#                props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
#                ax.text(0.55, 0.95, text, transform=ax.transAxes, fontsize=12,
#                        verticalalignment='top', bbox=props)

# %% Fit to the (1,1) - (1,-1) data to find Gamma, only if Omega waas able
# to fit
        
        plus_plus_avg_sig_counts = \
            numpy.average(plus_plus_sig_counts[i:i+slice_size, ::], axis=0)
        plus_plus_avg_ref_counts = \
            numpy.average(plus_plus_ref_counts[i:i+slice_size, ::], axis=0)
        
        plus_plus_norm_avg_sig = plus_plus_avg_sig_counts / plus_plus_avg_ref_counts
               
        plus_minus_avg_sig_counts = \
            numpy.average(plus_minus_sig_counts[i:i+slice_size, ::], axis=0)
        plus_minus_avg_ref_counts = \
            numpy.average(plus_minus_ref_counts[i:i+slice_size, ::], axis=0)
        
        plus_minus_norm_avg_sig = plus_minus_avg_sig_counts / plus_minus_avg_ref_counts
        
        # Define the counts for the plus relaxation equation
        plus_relaxation_counts =  plus_plus_norm_avg_sig - plus_minus_norm_avg_sig
        
#        # If omega failed, we can't fit gamma to this data, so we will set the
#        # gamma fail to True and just plot the points
#        if omega_fit_failed:
#            gamma_fit_failed = True
#            gamma_fit_failed_list.append(gamma_fit_failed)
            
#            if doPlot:
#                ax = axes_pack[1]
#                ax.plot(plus_plus_time, plus_relaxation_counts, 'bo')
#                ax.set_xlabel('Relaxation time (ms)')
#                ax.set_ylabel('Normalized signal Counts')
#                ax.set_title('(+1,+1) - (+1,-1)')
                
        # we will use the omega found to fit to, and add bounds to the 
        # omega param given by +/- the covariance of the fit.
        
#            omega_max = omega + omega_unc
#            omega_min = omega - omega_unc
        try:
            init_params = (200, 0.40, 0)
            opti_params, cov_arr = curve_fit(exp_eq, 
                             plus_plus_time, plus_relaxation_counts, 
                             p0 = init_params)

        except Exception:
            g_fit_failed = True
            g_fit_failed_list.append(g_fit_failed)
            
#            if doPlot:
#                ax = axes_pack[1]
#                ax.plot(plus_plus_time, plus_relaxation_counts, 'bo')
#                ax.set_xlabel('Relaxation time (ms)')
#                ax.set_ylabel('Normalized signal Counts')
#                ax.set_title('(+1,+1) - (+1,-1)')
            
        if not g_fit_failed:
            g_fit_failed_list.append(g_fit_failed)
            
#            gamma = (opti_params[0] - omega)/2
#            gamma_unc = numpy.sqrt((cov_arr[0,0])**2 + (omega_unc)**2)/2.0
            
            g_rate_list.append(opti_params[0])
#            gamma_unc_list.append(gamma_unc) 
            g_amp_list.append(opti_params[1])
            g_offset_list.append(opti_params[2])
        
       
#            # Plotting
#            if doPlot:
#                plus_time_linspace = numpy.linspace(0, plus_plus_time[-1], num=1000)
#                ax = axes_pack[1]
#                ax.plot(plus_plus_time, plus_relaxation_counts, 'bo')
#                ax.plot(plus_time_linspace, 
#                        exp_eq(plus_time_linspace, *opti_params), 
#                        'r', label = 'fit')   
##                    ax.set_xlim(0,0.1)
#                ax.set_xlabel('Relaxation time (ms)')
#                ax.set_ylabel('Normalized signal Counts')
#                ax.set_title('(+1,+1) - (+1,-1)')
#                ax.legend()
#                text = r'$\gamma = $ {} kHz'.format('%.2f'%gamma)
#    
#                props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
#                ax.text(0.55, 0.95, text, transform=ax.transAxes, fontsize=12,
#                        verticalalignment='top', bbox=props)
#        if doPlot:
#            fig.canvas.draw()
#            fig.canvas.flush_events()
                    
        # Advance_ the index
        i = i + bin_size
    
    o_average = numpy.average(o_rate_list)
    o_stdev = numpy.std(o_rate_list)
    
    g_average = numpy.average(g_rate_list)
    g_stdev = numpy.std(g_rate_list)
    
#    print('Omega list: {} \nGamma list: {}'.format(omega_rate_list, gamma_rate_list))
    
# %% Saving data
    
    if save_data: 
        time_stamp = tool_belt.get_time_stamp()
        raw_data = {'time_stamp': time_stamp,
                    'level_splitting': splitting_MHz,
                    'level_splitting-units': 'MHz',
                    'num_runs': num_runs,
                    'num_bins': num_bins,
                    'bin_size': bin_size,
                    'o_fit_failed_list': o_fit_failed_list,
                    'g_fit_failed_list': g_fit_failed_list,
                    'o_average': o_average,
                    'o_average-units': 'kHz',
                    'o_stdev': o_stdev,
                    'o_stdev-units': 'kHz',
                    'g_average': g_average,
                    'g_average-units': 'kHz',
                    'g_stdev': g_stdev,
                    'g_stdev-units': 'kHz',
                    'o_rate_list': o_rate_list,
                    'o_rate_list-units': 'kHz',
#                    'omega_unc_list': omega_unc_list,
#                    'omega_unc_list-units': 'kHz',
                    'o_amp_list': o_amp_list,
                    'o_amp_list-units': 'arb',
                    'o_offset_list': o_offset_list,
                    'o_offset_list-units': 'arb',
                    'g_rate_list': g_rate_list,
                    'g_rate_list-units': 'kHz',
#                    'gamma_unc_list': gamma_unc_list,
#                    'gamma_unc_list-units': 'kHz',
                    'g_amp_list': g_amp_list,
                    'g_amp_list-units': 'arb',
                    'g_offset_list': g_offset_list,
                    'g_offset_list-units': 'arb'}
        
        data_dir='E:/Shared drives/Kolkowitz Lab Group/nvdata'
        
        file_name = str('%.1f'%splitting_MHz) + '_MHz_splitting_' + str(num_bins) + '_bins_v2' 
        file_path = '{}/{}/{}/{}'.format(data_dir, data_folder, folder_name, 
                                                         file_name)
    
        tool_belt.save_raw_data(raw_data, file_path)

    return o_average, o_stdev, g_average, g_stdev, \
                  splitting_MHz, o_fit_failed_list, g_fit_failed_list
                  
# %% Run the file
                  
if __name__ == '__main__':
    
    folder = 'nv2_2019_04_30_57MHz'

    
    main(folder, 10, False, True)

