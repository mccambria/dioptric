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

This file allows the user to specify if the offset should be a free parameter
or if it should be set to 0.

The time used is in milliseconds

This version of the file calculates the standard deviation of the measurmenets and
plots the standard error. It also uses those standard errors in the curve_fit, 
which changes how the data is weighted.

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
    return  amp * exp(- rate * t)

def exp_eq_offset(t, rate, amp, offset):
    return  offset + amp * exp(- rate * t)

#def error_prop(norm_avg_signal_array, sig_array, sig_err_array, ref, ref_err):
#    norm_avg_signal_err_array = []
#    ref_frac_unc = ref_err / ref
#    for ind in range(len(norm_avg_signal_array)):
#        sig_frac_unc = sig_err_array[ind] / sig_array[ind]
#        error_prop = norm_avg_signal_array[ind] \
#                * numpy.sqrt(sig_frac_unc**2 + ref_frac_unc**2)
#                    
#        norm_avg_signal_err_array.append(error_prop) 
#    
#    return numpy.array(norm_avg_signal_err_array)

# %% Main

def main(folder_name, omega = None, omega_std = None, doPlot = False, offset = True):

    # Get the file list from this folder
    file_list = tool_belt.get_file_list(data_folder, '.txt', folder_name)

    # Define booleans to be used later in putting data into arrays in the
    # correct order
    zero_zero_bool = False
    zero_plus_bool = False
    plus_plus_bool = False
    plus_minus_bool = False
    minus_minus_bool = False
    zero_minus_bool = False

    # %% Unpack the data

    # Unpack the data and sort into arrays. This allows multiple experiments of
    # the same type (ie (1,-1)) to be correctly sorted into one array
    for file in file_list:
        data = tool_belt.get_raw_data(data_folder, file[:-4], folder_name)
        try:

            init_state_name = data['init_state']
            read_state_name = data['read_state']

            sig_counts  = numpy.array(data['sig_counts'])
            ref_counts = numpy.array(data['ref_counts'])

            relaxation_time_range = numpy.array(data['relaxation_time_range'])
            num_steps = data['num_steps']
            num_runs = data['num_runs']
            
            # Calculate some arrays
            min_relaxation_time, max_relaxation_time = relaxation_time_range / 10**6
            time_array = numpy.linspace(min_relaxation_time,
                                        max_relaxation_time, num=num_steps)
            
            avg_sig_counts = numpy.average(sig_counts[::], axis=0)
            std_sig_counts = numpy.std(sig_counts[::], axis=0, ddof = 1) 
            
            avg_ref = numpy.average(ref_counts[::]) 
#            std_ref = numpy.std(ref_counts)
            std_ref = 0
            
            norm_avg_sig = avg_sig_counts / avg_ref
#            norm_avg_err_0 = norm_avg_sig * \
#                    numpy.sqrt((std_sig_counts/avg_sig_counts)**2 + \
#                    (std_ref/avg_ref)**2)
            norm_avg_err = std_sig_counts / avg_ref 
            
            # take the average of the reference for 

            # Check to see which data set the file is for, and append the data
            # to the corresponding array
            if init_state_name == States.ZERO.name and read_state_name == States.ZERO.name:
                # Check to see if data has already been taken of this experiment
                # If it hasn't, then create arrays of the data.
                if zero_zero_bool == False:
                    zero_zero_counts = norm_avg_sig
                    zero_zero_err = norm_avg_err
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
                        zero_zero_err = numpy.concatenate((zero_zero_err,
                                                        norm_avg_err))
                        zero_zero_time = numpy.concatenate((zero_zero_time, time_array))

                    elif max_relaxation_time < zero_zero_ref_max_time:
                        zero_zero_counts = numpy.concatenate((norm_avg_sig,
                                              zero_zero_counts))
                        zero_zero_err = numpy.concatenate((norm_avg_err,
                                              zero_zero_err))
                        zero_zero_time = numpy.concatenate((time_array, zero_zero_time))

            if init_state_name == States.ZERO.name and read_state_name == States.HIGH.name:
                # Check to see if data has already been taken of this experiment
                # If it hasn't, then create arrays of the data.
                if zero_plus_bool == False:
                    zero_plus_counts = norm_avg_sig
                    zero_plus_err = norm_avg_err
                    zero_plus_time = time_array

                    zero_plus_ref_max_time = max_relaxation_time
                    zero_plus_bool = True
                # If data has already been taken for this experiment, then check
                # to see if this current data is the shorter or longer measurement,
                # and either append before or after the prexisting data
                else:

                    if max_relaxation_time > zero_plus_ref_max_time:
                        zero_plus_counts = numpy.concatenate((zero_plus_counts,
                                                        norm_avg_sig))
                        zero_plus_err = numpy.concatenate((zero_plus_err,
                                                        norm_avg_err))

                        zero_plus_time = numpy.concatenate((zero_plus_time, time_array))

                    elif max_relaxation_time < zero_plus_ref_max_time:
                        zero_plus_counts = numpy.concatenate((norm_avg_sig,
                                              zero_plus_counts))
                        zero_plus_err = numpy.concatenate((norm_avg_err,
                                              zero_plus_err))

                        zero_plus_time = numpy.concatenate(time_array, zero_plus_time)

            if init_state_name == States.ZERO.name and read_state_name == States.LOW.name:
                # Check to see if data has already been taken of this experiment
                # If it hasn't, then create arrays of the data.
                if zero_minus_bool == False:
                    zero_minus_counts = norm_avg_sig
                    zero_minus_err = norm_avg_err
                    zero_minus_time = time_array

                    zero_minus_ref_max_time = max_relaxation_time
                    zero_minus_bool = True
                # If data has already been taken for this experiment, then check
                # to see if this current data is the shorter or longer measurement,
                # and either append before or after the prexisting data
                else:

                    if max_relaxation_time > zero_minus_ref_max_time:
                        zero_minus_counts = numpy.concatenate((zero_minus_counts,
                                                        norm_avg_sig))
                        zero_minus_err = numpy.concatenate((zero_minus_err,
                                                        norm_avg_err))

                        zero_minus_time = numpy.concatenate((zero_minus_time, time_array))

                    elif max_relaxation_time < zero_minus_ref_max_time:
                        zero_minus_counts = numpy.concatenate((norm_avg_sig,
                                              zero_minus_counts))
                        zero_minus_err = numpy.concatenate((norm_avg_err,
                                              zero_minus_err))

                        zero_minus_time = numpy.concatenate(time_array, zero_minus_time)


            if init_state_name == States.HIGH.name and read_state_name == States.HIGH.name:
                # Check to see if data has already been taken of this experiment
                # If it hasn't, then create arrays of the data.
                if plus_plus_bool == False:
                    plus_plus_counts = norm_avg_sig
                    plus_plus_err = norm_avg_err
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
                        plus_plus_err = numpy.concatenate((plus_plus_err,
                                                        norm_avg_err))
                        plus_plus_time = numpy.concatenate((plus_plus_time, time_array))

                    elif max_relaxation_time < plus_plus_ref_max_time:
                        plus_plus_counts = numpy.concatenate((norm_avg_sig,
                                                          plus_plus_counts))
                        plus_plus_err = numpy.concatenate((norm_avg_err,
                                                          plus_plus_err))
                        plus_plus_time = numpy.concatenate((time_array, plus_plus_time))
            
            if init_state_name == States.LOW.name and read_state_name == States.LOW.name:
                # Check to see if data has already been taken of this experiment
                # If it hasn't, then create arrays of the data.
                if minus_minus_bool == False:
                    minus_minus_counts = norm_avg_sig
                    minus_minus_err = norm_avg_err
                    minus_minus_time = time_array

                    minus_minus_ref_max_time = max_relaxation_time
                    minus_minus_bool = True
                # If data has already been taken for this experiment, then check
                # to see if this current data is the shorter or longer measurement,
                # and either append before or after the prexisting data
                else:

                    if max_relaxation_time > minus_minus_ref_max_time:
                        minus_minus_counts = numpy.concatenate((minus_minus_counts,
                                                        norm_avg_sig))
                        minus_minus_err = numpy.concatenate((minus_minus_err,
                                                        norm_avg_err))
                        minus_minus_time = numpy.concatenate((minus_minus_time, time_array))

                    elif max_relaxation_time < minus_minus_ref_max_time:
                        minus_minus_counts = numpy.concatenate((norm_avg_sig,
                                                          minus_minus_counts))
                        minus_minus_err = numpy.concatenate((norm_avg_err,
                                                          minus_minus_err))
                        minus_minus_time = numpy.concatenate((time_array, minus_minus_time))
                        
            if init_state_name == States.HIGH.name and read_state_name == States.LOW.name:
                # We will want to put the MHz splitting in the file metadata
                uwave_freq_init = data['uwave_freq_init']
                uwave_freq_read = data['uwave_freq_read']

                # Check to see if data has already been taken of this experiment
                # If it hasn't, then create arrays of the data.
                if plus_minus_bool == False:
                    plus_minus_counts = norm_avg_sig
                    plus_minus_err = norm_avg_err
                    plus_minus_time = time_array

                    plus_minus_ref_max_time = max_relaxation_time
                    plus_minus_bool = True
                # If data has already been taken for this experiment, then check
                # to see if this current data is the shorter or longer measurement,
                # and either append before or after the prexisting data
                else:

                    if max_relaxation_time > plus_minus_ref_max_time:
                        plus_minus_counts = numpy.concatenate((plus_minus_counts,
                                                        norm_avg_sig))
                        plus_minus_err = numpy.concatenate((plus_minus_err,
                                                        norm_avg_err))
                        plus_minus_time = numpy.concatenate((plus_minus_time, time_array))


                    elif max_relaxation_time < plus_minus_ref_max_time:
                        plus_minus_counts = numpy.concatenate((norm_avg_sig,
                                              plus_minus_counts))
                        plus_minus_err = numpy.concatenate((norm_avg_err,
                                              plus_minus_err))
                        plus_minus_time = numpy.concatenate((time_array, plus_minus_time))


                splitting_MHz = abs(uwave_freq_init - uwave_freq_read) * 10**3

        except Exception:
            continue
        
    # %% Fit the data

    if doPlot:
        fig, axes_pack = plt.subplots(1, 2, figsize=(17, 8))

    omega_HIGH_fit_failed = False
    
    #Fit to the (0,0) - (0,1) data to find Omega

    # Define the counts for the zero relaxation equation
    
    omega_HIGH_relaxation_counts =  zero_zero_counts - zero_plus_counts
    omega_HIGH_relaxation_std = numpy.sqrt(zero_zero_err**2 + zero_plus_err**2)
    omega_HIGH_relaxation_ste = omega_HIGH_relaxation_std / numpy.sqrt(num_runs)

    init_params_list = [1.0, 0.4]
    
    try:
        if offset:
            init_params_list.append(0)
            init_params = tuple(init_params_list)
            omega_HIGH_opti_params, cov_arr = curve_fit(exp_eq_offset, zero_zero_time,
                                         omega_HIGH_relaxation_counts, p0 = init_params,
                                         sigma =  omega_HIGH_relaxation_ste, 
                                         absolute_sigma=True)
            
        else: 
            init_params = tuple(init_params_list)
            omega_HIGH_opti_params, cov_arr = curve_fit(exp_eq, zero_zero_time,
                                         omega_HIGH_relaxation_counts, p0 = init_params,
                                         sigma = omega_HIGH_relaxation_ste, 
                                         absolute_sigma=True)

    except Exception:

        omega_HIGH_fit_failed = True

        if doPlot:
            ax = axes_pack[0]
            ax.errorbar(zero_zero_time, omega_HIGH_relaxation_counts, 
                        yerr = omega_HIGH_relaxation_ste, 
                        label = 'data', fmt = 'o', color = 'blue')
            ax.set_xlabel('Relaxation time (ms)')
            ax.set_ylabel('Normalized signal Counts')
            ax.set_title('(0,0) - (0,+1)')
            ax.legend()

    if not omega_HIGH_fit_failed:

        omega_HIGH = omega_HIGH_opti_params[0] / 3.0
        omega_HIGH_ste = numpy.sqrt(cov_arr[0,0]) / 3.0
        
        print('Omega HIGH: {} +/- {} kHz'.format('%.3f'%omega, 
                  '%.3f'%omega_std))
        # Plotting the data
        if doPlot:
            zero_time_linspace = numpy.linspace(0, zero_zero_time[-1], num=1000)
            ax = axes_pack[0]
            ax.errorbar(zero_zero_time, omega_HIGH_relaxation_counts, 
                        yerr = omega_HIGH_relaxation_ste, 
                        label = 'data', fmt = 'o', color = 'blue')
            if offset:
                ax.plot(zero_time_linspace,
                    exp_eq_offset(zero_time_linspace, *omega_HIGH_opti_params),
                    'r', label = 'fit')
            else:
                ax.plot(zero_time_linspace,
                    exp_eq(zero_time_linspace, *omega_HIGH_opti_params),
                    'r', label = 'fit')
            ax.set_xlabel('Relaxation time (ms)')
            ax.set_ylabel('Normalized signal Counts')
            ax.set_title('(0,0) - (0,+1)')
            ax.legend()
            text = r'$\Omega HIGH= $ {} $\pm$ {} kHz'.format('%.3f'%omega_HIGH, 
                  '%.3f'%omega_HIGH_ste)

            props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
            ax.text(0.55, 0.9, text, transform=ax.transAxes, fontsize=12,
                    verticalalignment='top', bbox=props)

    # %% Fit to the (0,0) - (0,-1) data to find Gamma, only if Omega waas able
    # to fit
    omega_LOW_fit_failed = False
    
    #Fit to the (0,0) - (0,1) data to find Omega

    # Define the counts for the zero relaxation equation
    
    omega_LOW_relaxation_counts =  zero_zero_counts - zero_minus_counts
    omega_LOW_relaxation_std = numpy.sqrt(zero_zero_err**2 + zero_minus_err**2)
    omega_LOW_relaxation_ste = omega_LOW_relaxation_std / numpy.sqrt(num_runs)

    init_params_list = [1.0, 0.4]
    
    try:
        if offset:
            init_params_list.append(0)
            init_params = tuple(init_params_list)
            omega_LOW_opti_params, cov_arr = curve_fit(exp_eq_offset, zero_zero_time,
                                         omega_LOW_relaxation_counts, p0 = init_params,
                                         sigma = omega_LOW_relaxation_ste, 
                                         absolute_sigma=True)
            
        else: 
            init_params = tuple(init_params_list)
            omega_LOW_opti_params, cov_arr = curve_fit(exp_eq, zero_zero_time,
                                         omega_LOW_relaxation_counts, p0 = init_params,
                                         sigma = omega_LOW_relaxation_ste, 
                                         absolute_sigma=True)
    except Exception:

        omega_LOW_fit_failed = True

        if doPlot:
            ax = axes_pack[0]
            ax.errorbar(zero_zero_time, omega_LOW_relaxation_counts, 
                        yerr = omega_LOW_relaxation_ste, 
                        label = 'data', fmt = 'o', color = 'blue')
            ax.set_xlabel('Relaxation time (ms)')
            ax.set_ylabel('Normalized signal Counts')
            ax.set_title('(0,0) - (0,-1)')
            ax.legend()

    if not omega_LOW_fit_failed:

#        print(opti_params[0])
        omega_LOW = omega_LOW_opti_params[0] / 3.0
        omega_LOW_ste = numpy.sqrt(cov_arr[0,0]) / 3.0
        
        print('Omega LOW: {} +/- {} kHz'.format('%.3f'%omega, 
                  '%.3f'%omega_std))
        
    # Plotting the data
    if doPlot:
        zero_time_linspace = numpy.linspace(0, zero_zero_time[-1], num=1000)
        ax = axes_pack[0]
        ax.errorbar(zero_zero_time, omega_LOW_relaxation_counts, 
                    yerr = omega_LOW_relaxation_ste, 
                    label = 'data', fmt = 'o', color = 'blue')
        if offset:
            ax.plot(zero_time_linspace,
                exp_eq_offset(zero_time_linspace, *omega_LOW_opti_params),
                'r', label = 'fit')
        else:
            ax.plot(zero_time_linspace,
                exp_eq(zero_time_linspace, *omega_LOW_opti_params),
                'r', label = 'fit')
            
        ax.set_xlabel('Relaxation time (ms)')
        ax.set_ylabel('Normalized signal Counts')
        ax.set_title('(0,0) - (0,-1)')
        ax.legend()
        text = r'$\Omega LOW= $ {} $\pm$ {} kHz'.format('%.3f'%omega_LOW, 
              '%.3f'%omega_LOW_ste)

        props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
        ax.text(0.55, 0.9, text, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=props)    

        fig.canvas.draw()
        fig.canvas.flush_events()

        # %% Saving the data 
        
        data_dir='E:/Shared drives/Kolkowitz Lab Group/nvdata'

        time_stamp = tool_belt.get_time_stamp()
        raw_data = {'time_stamp': time_stamp,
                    'splitting_MHz': splitting_MHz,
                    'splitting_MHz-units': 'MHz',
                    'offset_free_param?': offset,
                    'omega_HIGH_relaxation_counts': omega_HIGH_relaxation_counts.tolist(),
                    'omega_HIGH_relaxation_counts-units': 'counts',
                    'omega_HIGH_relaxation_ste': omega_HIGH_relaxation_ste.tolist(),
                    'omega_HIGH_relaxation_ste-units': 'counts',
                    'zero_zero_time': zero_zero_time.tolist(),
                    'zero_zero_time-units': 'ms',
                    'omega_LOW_relaxation_counts': omega_LOW_relaxation_counts.tolist(),
                    'omega_LOW_relaxation_counts-units': 'counts',
                    'omega_LOW_relaxation_ste': omega_LOW_relaxation_ste.tolist(),
                    'omega_LOW_relaxation_ste-units': 'counts',
                    'omega_HIGH_opti_params': omega_HIGH_opti_params.tolist(),
                    'omega_LOW_opti_params': omega_LOW_opti_params.tolist(),
                    'omega_HIGH_ste': omega_HIGH_ste,
                    'omega_HIGH_ste-units': 'khz',
                    'omega_LOW_ste': omega_LOW_ste,
                    'omega_LOW_ste-units': 'khz'
                    }
        

        
        file_name = str('%.1f'%splitting_MHz) + '_MHz_splitting_omega_comparison' 
        file_path = '{}/{}/{}/{}'.format(data_dir, data_folder, folder_name, 
                                                             file_name)
        
        tool_belt.save_raw_data(raw_data, file_path)
    
    # %% Saving the figure

   
        file_name = str('%.1f'%splitting_MHz) + '_MHz_splitting_omega_comparison'
        file_path = '{}/{}/{}/{}'.format(data_dir, data_folder, folder_name,
                                                             file_name)

        tool_belt.save_figure(fig, file_path)

# %% Run the file

if __name__ == '__main__':

    folder = 'nv1_2019_05_10_1017MHz'

#    for folder in folder_list:
#    main(folder, True)
    main(folder,  None, None,  True, offset = True)
