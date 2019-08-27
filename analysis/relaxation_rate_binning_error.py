# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 09:52:43 2019

This analysis script will take a set of T1 experiments and fit the fucntions
defined in the Myer's paper ((0,0) - (0,1) and (1,1) - (1,-1)) to extract a 
rate for the two modified data set exponential fits. It allows the data to be
split into different bins, which is used for the analysis of stdev. This file 
does not convert these rates into omega and gamma, this function passes these 
basic rates onto the stdev analysis file.

This file averages the reference counts in a bin and uses the single value as 
the reference.

This file also allows the user to specify if the offset shoudl be a free param

Now allows a value and uncertainty for omega to be passed in. If that is passed 
in, then the program will use those values instead. Otherwise, it still will
try to evaluate an omega

@author: Aedan
"""

# %% Imports

import numpy
from scipy import exp
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

import utils.tool_belt as tool_belt
from utils.tool_belt import States

# %% Constants

data_folder = 't1_double_quantum'

# %% Functions

# The exponential function used to fit the data

def exp_eq(t, rate, amp):
    return amp * exp(- rate * t)

def exp_eq_offset(t, rate, amp, offset):
    return offset + amp * exp(- rate * t)

# %% Main
    
def main(folder_name, num_bins, omega = None, omega_ste = None, save_data = True, offset = True):
    
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
    if num_bins > num_runs_set:
        print('num_bins > num_runs. bin_size will be set to 1')
        
        bin_size = 1
        
    else: 
        bin_size = int(num_runs_set / num_bins)
        
    # Define booleans to be used later in putting data into usable arrays
    zero_zero_bool = False
    zero_plus_bool = False
    plus_plus_bool = False
    plus_minus_bool = False
    
    zero_relaxation_count_list = []
    zero_relaxation_count_ste_list = []
    plus_relaxation_count_list = []
    plus_relaxation_count_ste_list = []
    zero_zero_time_list = []
    plus_plus_time_list = []
    
    o_fit_failed_list = []
    g_fit_failed_list = []
    
    # Create lists to store the omega and gamma rates
    o_rate_list = []
    o_amp_list = []
    o_offset_list = []
    
    g_rate_list = []
    g_amp_list = []
    g_offset_list = []
    
    omega_list = []
    omega_ste_list = []
    gamma_list =[]
    gamma_ste_list = []
    # %% Unpack the data
    
    # Unpack the data and sort into arrays. This allows multiple experiments of 
    # the same type (ie (1,-1)) to be correctly sorted into one array
    for file in file_list:
        data = tool_belt.get_raw_data(data_folder, file[:-4], folder_name)
        try:
                
            init_state_name = data['init_state']
            read_state_name = data['read_state']
            
            sig_counts = numpy.array(data['sig_counts'])
            ref_counts = numpy.array(data['ref_counts'])
            
            avg_ref = numpy.average(ref_counts)
            
            norm_counts = sig_counts / avg_ref           
            
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
            if init_state_name == States.ZERO.name and read_state_name == States.ZERO.name:
                # Check to see if data has already been taken of this experiment
                # If it hasn't, then create arrays of the data.
                if zero_zero_bool == False:
                    zero_zero_counts = norm_counts
#                    zero_zero_ref_counts = ref_counts
                    zero_zero_time = time_array
                    
                    zero_zero_ref_max_time = max_relaxation_time
                    zero_zero_bool = True
                # If data has already been taken for this experiment, then check
                # to see if this current data is the shorter or longer measurement,
                # and either append before or after the prexisting data
                else:
                    
                    if max_relaxation_time > zero_zero_ref_max_time:
                        zero_zero_counts = numpy.concatenate((zero_zero_counts, 
                                                        norm_counts), axis = 1)
#                        zero_zero_ref_counts = numpy.concatenate((zero_zero_ref_counts, 
#                                                        ref_counts), axis = 1)
                        zero_zero_time = numpy.concatenate((zero_zero_time, time_array))
                        
                    elif max_relaxation_time < zero_zero_ref_max_time:
                        zero_zero_counts = numpy.concatenate((norm_counts, 
                                              zero_zero_counts), axis = 1)
#                        zero_zero_ref_counts = numpy.concatenate((ref_counts, 
#                                              zero_zero_ref_counts), axis = 1)
                        zero_zero_time = numpy.concatenate((time_array, zero_zero_time))
                
            if init_state_name == States.ZERO.name and read_state_name == States.HIGH.name:
                # Check to see if data has already been taken of this experiment
                # If it hasn't, then create arrays of the data.
                if zero_plus_bool == False:
                    zero_plus_counts = norm_counts
#                    zero_plus_ref_counts = ref_counts
                    
                    zero_plus_ref_max_time = max_relaxation_time
                    zero_plus_bool = True
                # If data has already been taken for this experiment, then check
                # to see if this current data is the shorter or longer measurement,
                # and either append before or after the prexisting data
                else:
                    
                    if max_relaxation_time > zero_plus_ref_max_time:
                        zero_plus_counts = numpy.concatenate((zero_plus_counts, 
                                                        norm_counts), axis = 1)
#                        zero_plus_ref_counts = numpy.concatenate((zero_plus_ref_counts, 
#                                                        ref_counts), axis = 1)
                        
                    elif max_relaxation_time < zero_plus_ref_max_time:
                        zero_plus_counts = numpy.concatenate((norm_counts, 
                                              zero_plus_counts), axis = 1)
#                        zero_plus_ref_counts = numpy.concatenate((ref_counts, 
#                                              zero_plus_ref_counts), axis = 1)

            if init_state_name == States.HIGH.name and read_state_name == States.HIGH.name:              
                # Check to see if data has already been taken of this experiment
                # If it hasn't, then create arrays of the data.
                if plus_plus_bool == False:
                    plus_plus_counts = norm_counts
#                    plus_plus_ref_counts = ref_counts
                    plus_plus_time = time_array
                    
                    plus_plus_ref_max_time = max_relaxation_time
                    plus_plus_bool = True
                    
                # If data has already been taken for this experiment, then check
                # to see if this current data is the shorter or longer measurement,
                # and either append before or after the prexisting data
                else:
                    
                    if max_relaxation_time > plus_plus_ref_max_time:
                        plus_plus_counts = numpy.concatenate((plus_plus_counts, 
                                                        norm_counts), axis = 1)
#                        plus_plus_ref_counts = numpy.concatenate((plus_plus_ref_counts, 
#                                                        ref_counts), axis = 1)
                        plus_plus_time = numpy.concatenate((plus_plus_time, time_array))
                        
                    elif max_relaxation_time < plus_plus_ref_max_time:
                        plus_plus_counts = numpy.concatenate((norm_counts, 
                                                          plus_plus_counts), axis = 1)
#                        plus_plus_ref_counts = numpy.concatenate((ref_counts, 
#                                                          plus_plus_ref_counts), axis = 1)
                        plus_plus_time = numpy.concatenate((time_array, plus_plus_time))
                        
            if init_state_name == States.HIGH.name and read_state_name == States.LOW.name:
                # We will want to put the MHz splitting in the file metadata
                uwave_freq_init = data['uwave_freq_init']
                uwave_freq_read = data['uwave_freq_read']
                
                # Check to see if data has already been taken of this experiment
                # If it hasn't, then create arrays of the data.
                if plus_minus_bool == False:
                    plus_minus_counts = norm_counts
#                    plus_minus_ref_counts = ref_counts
                    
                    plus_minus_ref_max_time = max_relaxation_time
                    plus_minus_bool = True

                # If data has already been taken for this experiment, then check
                # to see if this current data is the shorter or longer measurement,
                # and either append before or after the prexisting data
                else:
                    
                    if max_relaxation_time > plus_minus_ref_max_time:
                        plus_minus_counts = numpy.concatenate((plus_minus_counts, 
                                                        norm_counts), axis = 1)
#                        plus_minus_ref_counts = numpy.concatenate((plus_minus_ref_counts, 
#                                                        ref_counts), axis = 1)

                    elif max_relaxation_time < plus_minus_ref_max_time:
                        plus_minus_counts = numpy.concatenate((norm_counts, 
                                              plus_minus_counts), axis = 1)
#                        plus_minus_ref_counts = numpy.concatenate((ref_counts, 
#                                              plus_minus_ref_counts), axis = 1)

                splitting_MHz = abs(uwave_freq_init - uwave_freq_read) * 10**3
                
        except Exception:
            continue
    
    # Some error handeling if the count arras don't match up            
#    if len(zero_zero_sig_counts) != len(zero_plus_sig_counts): 
#                    
#         print('Error: length of zero_zero_sig_counts and zero_plus_sig_counts do not match')
#       
#    if len(plus_plus_sig_counts) != len(plus_minus_sig_counts):
#        print('Error: length of plus_plus_sig_counts and plus_minus_sig_counts do not match')
    
    # %% Fit the data based on the bin size
        
    i = 0
    
    # We want to slice the arrays using [i:i+bin_size].
    
    slice_size = bin_size
    
    while i < (num_runs):
        o_fit_failed = False
        g_fit_failed = False
        
        #Fit to the (0,0) - (0,1) data to find Omega if we have note passed in an omega value
        if omega is None and omega_ste is None:
            zero_zero_counts_slice = \
                numpy.average(zero_zero_counts[i:i+slice_size, ::], axis = 0)
            zero_zero_counts_slice_std = \
                numpy.std(zero_zero_counts[i:i+slice_size, ::], axis = 0)
            
            zero_plus_counts_slice = \
                numpy.average(zero_plus_counts[i:i+slice_size, ::], axis = 0)
            zero_plus_counts_slice_std = \
                numpy.std(zero_plus_counts[i:i+slice_size, ::], axis = 0)
                
            # Define the counts for the zero relaxation equation
            zero_relaxation_counts =  zero_zero_counts_slice - zero_plus_counts_slice
            zero_relaxation_std = numpy.sqrt(zero_zero_counts_slice_std**2 + zero_plus_counts_slice_std**2)
            zero_relaxation_ste = zero_relaxation_std / numpy.sqrt(slice_size)
            
            #save all the data for possible replotting later
            zero_relaxation_count_list.append(zero_relaxation_counts.tolist())
            zero_relaxation_count_ste_list.append(zero_relaxation_ste)
            zero_zero_time_list.append(zero_zero_time.tolist())
            
            
            init_params_list = [1.0, 0.4]
        
            try:
                if offset:
                    init_params_list.append(0)
                    init_params = tuple(init_params_list)
                    omega_opti_params, cov_arr = curve_fit(exp_eq_offset, zero_zero_time,
                                                 zero_relaxation_counts, p0 = init_params,
                                                 sigma = zero_relaxation_std, 
                                                 absolute_sigma=True)
                    
                else: 
                    init_params = tuple(init_params_list)
                    omega_opti_params, cov_arr = curve_fit(exp_eq, zero_zero_time,
                                                 zero_relaxation_counts, p0 = init_params,
                                                 sigma = zero_relaxation_std, 
                                                 absolute_sigma=True)
                
            except Exception:
                
                o_fit_failed = True
                o_fit_failed_list.append(o_fit_failed)
    
            if not o_fit_failed:
                omega = omega_opti_params[0] / 3
                omega_ste = numpy.sqrt(cov_arr[0,0]) / 3.0
                
                omega_list.append(omega)
                omega_ste_list.append(omega_ste)
                            
                o_fit_failed_list.append(o_fit_failed)
                
                o_rate_list.append(omega_opti_params[0])
                o_amp_list.append(omega_opti_params[1])
                if offset:
                    o_offset_list.append(omega_opti_params[2])

# %% Fit to the (1,1) - (1,-1) data to find Gamma, only if Omega waas able
# to fit

        plus_plus_counts_slice = \
            numpy.average(plus_plus_counts[i:i+slice_size, ::], axis = 0)
        plus_plus_counts_slice_std = \
            numpy.std(plus_plus_counts[i:i+slice_size, ::], axis = 0, ddof = 1)
            
        plus_minus_counts_slice = \
            numpy.average(plus_minus_counts[i:i+slice_size, ::], axis = 0)
        plus_minus_counts_slice_std = \
            numpy.std(plus_minus_counts[i:i+slice_size, ::], axis = 0, ddof = 1)
        
        # Define the counts for the plus relaxation equation, propegate error
        plus_relaxation_counts =  plus_plus_counts_slice - plus_minus_counts_slice
        plus_relaxation_std = numpy.sqrt(plus_plus_counts_slice_std**2 + plus_minus_counts_slice_std**2)
        plus_relaxation_ste = plus_relaxation_std / numpy.sqrt(slice_size)
        
        #save all the data for possible replotting later
        plus_relaxation_count_list.append(plus_relaxation_counts.tolist())
        plus_relaxation_count_ste_list.append(plus_relaxation_ste)
        plus_plus_time_list.append(plus_plus_time.tolist())
        
    
        init_params_list = [10, 0.40]
        try:
            if offset:
                init_params_list.append(0)
                init_params = tuple(init_params_list)
                gamma_opti_params, cov_arr = curve_fit(exp_eq_offset,
                                 plus_plus_time, plus_relaxation_counts,
                                 p0 = init_params, sigma = plus_relaxation_std, 
                                 absolute_sigma=True)
                    
                
            else:
                init_params = tuple(init_params_list)
                gamma_opti_params, cov_arr = curve_fit(exp_eq,
                                 plus_plus_time, plus_relaxation_counts,
                                 p0 = init_params, sigma = plus_relaxation_std, 
                                 absolute_sigma=True)

        except Exception:
            g_fit_failed = True
            g_fit_failed_list.append(g_fit_failed)
            
        if not g_fit_failed:
            gamma = (gamma_opti_params[0] - omega) / 2
            gamma_ste = 0.5 * numpy.sqrt(cov_arr[0,0]+omega_ste**2)
            
            gamma_list.append(gamma)
            gamma_ste_list.append(gamma_ste)
            
            g_fit_failed_list.append(g_fit_failed)
              
            g_rate_list.append(gamma_opti_params[0])
            g_amp_list.append(gamma_opti_params[1])
            if offset:
                g_offset_list.append(gamma_opti_params[2])
        
        # plot data if saving the data
        if save_data:
            
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            plus_time_linspace = numpy.linspace(0, plus_plus_time[-1], num=1000)
            #    ax = axes_pack[1]
            ax.errorbar(plus_plus_time, plus_relaxation_counts,                         
                    yerr = plus_relaxation_std, 
                    label = 'data', fmt = 'o', color = 'blue')
            if offset:
                ax.plot(plus_time_linspace,
                    exp_eq_offset(plus_time_linspace, *gamma_opti_params),
                    'r', label = 'fit')
                
            else:
                ax.plot(plus_time_linspace,
                    exp_eq(plus_time_linspace, *gamma_opti_params),
                    'r', label = 'fit')
            ax.set_xlabel('Relaxation time (ms)')
            ax.set_ylabel('Normalized signal Counts')
            ax.set_title('(+1,+1) - (+1,-1)')
            ax.legend()
            text = r'$\gamma = $ {}$\pm${}kHz'.format('%.2f'%gamma, '%.2f'%gamma_ste)
            
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            ax.text(0.55, 0.95, text, transform=ax.transAxes, fontsize=12,
                    verticalalignment='top', bbox=props)
        # Advance_ the index
        i = i + bin_size
    
    if omega is None and omega_ste is None:
        o_average = numpy.average(o_rate_list)
        o_stdev = numpy.std(o_rate_list)
    else:
        o_average = omega * 3
        o_stdev = omega_ste * 3
    
    g_average = numpy.average(g_rate_list)
    g_stdev = numpy.std(g_rate_list)
    
#    print(o_average / 3)
#    print((g_average - o_average / 3) / 2)


      
# %% Saving data
    if save_data:

        time_stamp = tool_belt.get_time_stamp()
        raw_data = {'time_stamp': time_stamp,
                    'level_splitting': splitting_MHz,
                    'level_splitting-units': 'MHz',
                    'offset_free_param?': offset,
                    'num_runs': num_runs,
                    'num_bins': num_bins,
                    'bin_size': bin_size,
                    'omega_list': omega_list,
                    'omega_list-units': 'kHz',
                    'omega_ste_list': omega_ste_list,
                    'omega_ste_list-units': 'kHz',
                    'gamma_list': gamma_list,
                    'gamma_list-units': 'kHz',
                    'gamma_ste_list': gamma_ste_list,
                    'gamma_ste_list-units': 'kHz',
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
                    'o_amp_list': o_amp_list,
                    'o_amp_list-units': 'arb',
                    'o_offset_list': o_offset_list,
                    'o_offset_list-units': 'arb',
                    'g_rate_list': g_rate_list,
                    'g_rate_list-units': 'kHz',
                    'g_amp_list': g_amp_list,
                    'g_amp_list-units': 'arb',
                    'g_offset_list': g_offset_list,
                    'g_offset_list-units': 'arb',
                    'zero_relaxation_count_list': zero_relaxation_count_list,
                    'plus_relaxation_count_list': plus_relaxation_count_list,
                    'zero_zero_time_list': zero_zero_time_list,
                    'zero_zero_time_list-units': 'ms', 
                    'plus_plus_time_list': plus_plus_time_list,
                    'plus_plus_time_list-units': 'ms'}
        
        data_dir='E:/Shared drives/Kolkowitz Lab Group/nvdata'
        
        file_name = str('%.1f'%splitting_MHz) + '_MHz_splitting_' + str(num_bins) + '_bins_v2' 
        file_path = '{}/{}/{}/{}'.format(data_dir, data_folder, folder_name, 
                                                         file_name)
        tool_belt.save_raw_data(raw_data, file_path)
    
    return o_average, o_stdev, g_average, g_stdev, \
                  splitting_MHz, o_fit_failed_list, g_fit_failed_list
                  
# %% Run the file
                  
if __name__ == '__main__':
    
    folder = 'nv1_2019_05_10_28MHz_4'

    
    main(folder, 15,  1.6, 0.6, True,  True)

