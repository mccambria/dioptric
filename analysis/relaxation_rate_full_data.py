# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 09:52:43 2019

This analysis script will plot and evaluate the omega and gamma rates for the
modified rate equations from the Myer's paper (ex: (0,0) - (0,1) and 
(1,1) - (1,-1)) for the whole data set. It uses the norm_avg_sig counts from 
the data. This file does not allow us to break the data into different sized 
bins

This file will automatically save the figure created in the folder of the data
used.

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


# %% Main
    
def main(folder_name, doPlot = False):

    # Get the file list from this folder
    file_list = tool_belt.get_file_list(data_folder, '.txt', folder_name)
       
    # Define booleans to be used later in putting data into arrays in the 
    # correct order
    zero_zero_bool = False
    zero_plus_bool = False
    plus_plus_bool = False
    plus_minus_bool = False
    
    # %% Unpack the data
    
    # Unpack the data and sort into arrays. This allows multiple experiments of 
    # the same type (ie (1,-1)) to be correctly sorted into one array
    for file in file_list:
        data = tool_belt.get_raw_data(data_folder, file[:-4], folder_name)
        try:
                
            init_state = data['init_state']
            read_state = data['read_state']
            
            norm_avg_sig = numpy.array(data['norm_avg_sig'])
            
            relaxation_time_range = numpy.array(data['relaxation_time_range'])
            # time is in microseconds
            min_relaxation_time, max_relaxation_time = relaxation_time_range / 10**6
            num_steps = data['num_steps']

            time_array = numpy.linspace(min_relaxation_time, 
                                        max_relaxation_time, num=num_steps) 
                       
            # Check to see which data set the file is for, and append the data
            # to the corresponding array
            if init_state == 0 and read_state == 0:
                # Check to see if data has already been taken of this experiment
                # If it hasn't, then create arrays of the data.
                if zero_zero_bool == False:
                    zero_zero_counts = norm_avg_sig
                    zero_zero_time = time_array
                    
                    zero_zero_ref_max_time = max_relaxation_time
                    zero_zero_bool = True
                # If data has already been taken for this experiment, then check
                # to see if this current data is the shorter or longer measurement,
                # and either append before or after the prexisting data
                else:
                    
                    if max_relaxation_time > zero_zero_ref_max_time:
                        zero_zero_counts = numpy.concatenate((zero_zero_counts, 
                                                        norm_avg_sig))
                        zero_zero_time = numpy.concatenate((zero_zero_time, time_array))
                        
                    elif max_relaxation_time < zero_zero_ref_max_time:
                        zero_zero_counts = numpy.concatenate((norm_avg_sig, 
                                              zero_zero_counts))
                        zero_zero_time = numpy.concatenate((time_array, zero_zero_time))
                
            if init_state == 0 and read_state == 1:
                # Check to see if data has already been taken of this experiment
                # If it hasn't, then create arrays of the data.
                if zero_plus_bool == False:
                    zero_plus_counts = norm_avg_sig
                    
                    zero_plus_ref_max_time = max_relaxation_time
                    zero_plus_bool = True
                # If data has already been taken for this experiment, then check
                # to see if this current data is the shorter or longer measurement,
                # and either append before or after the prexisting data
                else:
                    
                    if max_relaxation_time > zero_plus_ref_max_time:
                        zero_plus_counts = numpy.concatenate((zero_plus_counts, 
                                                        norm_avg_sig))
                        
                    elif max_relaxation_time < zero_plus_ref_max_time:
                        zero_plus_counts = numpy.concatenate((norm_avg_sig, 
                                              zero_plus_counts))

            if init_state == 1 and read_state == 1:              
                # Check to see if data has already been taken of this experiment
                # If it hasn't, then create arrays of the data.
                if plus_plus_bool == False:
                    plus_plus_counts = norm_avg_sig
                    plus_plus_time = time_array
                    
                    plus_plus_ref_max_time = max_relaxation_time
                    plus_plus_bool = True
                # If data has already been taken for this experiment, then check
                # to see if this current data is the shorter or longer measurement,
                # and either append before or after the prexisting data
                else:
                    
                    if max_relaxation_time > plus_plus_ref_max_time:
                        plus_plus_counts = numpy.concatenate((plus_plus_counts, 
                                                        norm_avg_sig))
                        plus_plus_time = numpy.concatenate((plus_plus_time, time_array))
                        
                    elif max_relaxation_time < plus_plus_ref_max_time:
                        plus_plus_counts = numpy.concatenate((norm_avg_sig, 
                                                          plus_plus_counts))
                        plus_plus_time = numpy.concatenate((time_array, plus_plus_time))
                
            if init_state == 1 and read_state == -1:
                # We will want to put the MHz splitting in the file metadata
                uwave_freq_init = data['uwave_freq_init']
                uwave_freq_read = data['uwave_freq_read']
                
                # Check to see if data has already been taken of this experiment
                # If it hasn't, then create arrays of the data.
                if plus_minus_bool == False:
                    plus_minus_counts = norm_avg_sig
                    
                    plus_minus_ref_max_time = max_relaxation_time
                    plus_minus_bool = True
                # If data has already been taken for this experiment, then check
                # to see if this current data is the shorter or longer measurement,
                # and either append before or after the prexisting data
                else:
                    
                    if max_relaxation_time > plus_minus_ref_max_time:
                        plus_minus_counts = numpy.concatenate((plus_minus_counts, 
                                                        norm_avg_sig))
                        
                    elif max_relaxation_time < plus_minus_ref_max_time:
                        plus_minus_counts = numpy.concatenate((norm_avg_sig, 
                                              plus_minus_counts))
                
                splitting_MHz = abs(uwave_freq_init - uwave_freq_read) * 10**3
                
        except Exception:
            continue
    
    # Some error handeling if the count arras don't match up            
    if len(zero_zero_counts) != len(zero_plus_counts): 
                    
         print('Error: length of zero_zero_sig_counts and zero_plus_sig_counts do not match')
       
    if len(plus_plus_counts) != len(plus_minus_counts):
        print('Error: length of plus_plus_sig_counts and plus_minus_sig_counts do not match')
    
    # %% Fit the data

    if doPlot:
        fig, axes_pack = plt.subplots(1, 2, figsize=(17, 8))
    
    #Fit to the (0,0) - (0,1) data to find Omega

    # Define the counts for the zero relaxation equation
    zero_relaxation_counts =  zero_zero_counts - zero_plus_counts
    
    omega_fit_failed = False
    gamma_fit_failed = False

    try:

        init_params = (1.0, 0.4, 0)
        opti_params, cov_arr = curve_fit(exp_eq, zero_zero_time,
                                     zero_relaxation_counts, p0 = init_params)
       
    except Exception:
        
        omega_fit_failed = True
        
        if doPlot:
            ax = axes_pack[0]
            ax.plot(zero_zero_time, zero_relaxation_counts, 'bo', label = 'data')
            ax.set_xlabel('Relaxation time (ms)')
            ax.set_ylabel('Normalized signal Counts')
            ax.set_title('(0,0) - (0,+1)')
            ax.legend()

    if not omega_fit_failed:
        
        omega = opti_params[0] / 3.0

        # Plotting the data
        if doPlot:
            zero_time_linspace = numpy.linspace(0, zero_zero_time[-1], num=1000)
            ax = axes_pack[0]
            ax.plot(zero_zero_time, zero_relaxation_counts, 'bo', label = 'data')
            ax.plot(zero_time_linspace, 
                    exp_eq(zero_time_linspace, *opti_params), 
                    'r', label = 'fit') 
            ax.set_xlabel('Relaxation time (ms)')
            ax.set_ylabel('Normalized signal Counts')
            ax.set_title('(0,0) - (0,+1)')
            ax.legend()
            text = r'$\Omega = $ {} kHz'.format('%.2f'%omega)

            props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
            ax.text(0.55, 0.95, text, transform=ax.transAxes, fontsize=12,
                    verticalalignment='top', bbox=props)

# %% Fit to the (1,1) - (1,-1) data to find Gamma, only if Omega waas able
# to fit
    
    # Define the counts for the plus relaxation equation
    plus_relaxation_counts =  plus_plus_counts - plus_minus_counts
                        
    try:
        init_params = (200, 0.40, 0)
        opti_params, cov_arr = curve_fit(exp_eq, 
                         plus_plus_time, plus_relaxation_counts, 
                         p0 = init_params)

    except Exception:
        gamma_fit_failed = True
        
        if doPlot:
            ax = axes_pack[1]
            ax.plot(plus_plus_time, plus_relaxation_counts, 'bo')
            ax.set_xlabel('Relaxation time (ms)')
            ax.set_ylabel('Normalized signal Counts')
            ax.set_title('(+1,+1) - (+1,-1)')
        
    if not gamma_fit_failed:
        
        gamma = (opti_params[0] - omega)/2
   
        # Plotting
        if doPlot:
            plus_time_linspace = numpy.linspace(0, plus_plus_time[-1], num=1000)
            ax = axes_pack[1]
            ax.plot(plus_plus_time, plus_relaxation_counts, 'bo')
            ax.plot(plus_time_linspace, 
                    exp_eq(plus_time_linspace, *opti_params), 
                    'r', label = 'fit')   
#                    ax.set_xlim(0,0.1)
            ax.set_xlabel('Relaxation time (ms)')
            ax.set_ylabel('Normalized signal Counts')
            ax.set_title('(+1,+1) - (+1,-1)')
            ax.legend()
            text = r'$\gamma = $ {} kHz'.format('%.2f'%gamma)

            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            ax.text(0.55, 0.95, text, transform=ax.transAxes, fontsize=12,
                    verticalalignment='top', bbox=props)
    if doPlot:
        fig.canvas.draw()
        fig.canvas.flush_events()
    
#    print('Omega list: {} \nGamma list: {}'.format(omega_rate_list, gamma_rate_list))
    
# %% Saving the figure
    
#    if save_data: 
#        time_stamp = tool_belt.get_time_stamp()
#        raw_data = {'time_stamp': time_stamp,
#                    'level_splitting': splitting_MHz,
#                    'level_splitting-units': 'MHz',
#                    'num_runs': num_runs,
#                    'num_bins': num_bins,
#                    'bin_size': bin_size,
#                    'omega_fit_failed_list': omega_fit_failed_list,
#                    'gamma_fit_failed_list': gamma_fit_failed_list,
#                    'omega_average': omega_average,
#                    'omega_average-units': 'kHz',
#                    'omega_stdev': omega_stdev,
#                    'omega_stdev-units': 'kHz',
#                    'gamma_average': gamma_average,
#                    'gamma_average-units': 'kHz',
#                    'gamma_stdev': gamma_stdev,
#                    'gamma_stdev-units': 'kHz',
#                    'omega_rate_list': omega_rate_list,
#                    'omega_rate_list-units': 'kHz',
#                    'omega_unc_list': omega_unc_list,
#                    'omega_unc_list-units': 'kHz',
#                    'omega_amp_list': omega_rate_list,
#                    'omega_amp_list-units': 'arb',
#                    'omega_offset_list': omega_offset_list,
#                    'omega_offset_list-units': 'arb',
#                    'gamma_rate_list': gamma_rate_list,
#                    'gamma_rate_list-units': 'kHz',
#                    'gamma_unc_list': gamma_unc_list,
#                    'gamma_unc_list-units': 'kHz',
#                    'gamma_amp_list': gamma_rate_list,
#                    'gamma_amp_list-units': 'arb',
#                    'gamma_offset_list': gamma_offset_list,
#                    'gamma_offset_list-units': 'arb'}
        
        data_dir='E:/Shared drives/Kolkowitz Lab Group/nvdata'
        
        file_name = str('%.1f'%splitting_MHz) + '_MHz_splitting_1_bins_v2' 
        file_path = '{}/{}/{}/{}'.format(data_dir, data_folder, folder_name, 
                                                         file_name)
    
#        tool_belt.save_figure(fig, file_path)
                  
# %% Run the file
                  
if __name__ == '__main__':
    
    folder = 'nv2_2019_04_30_57MHz'
    
    main(folder, True)

