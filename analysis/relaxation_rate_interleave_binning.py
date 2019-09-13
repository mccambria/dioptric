# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 17:45:26 2019

This file specifically made to analyze the data of t1_interleave

It works only with an experiment that performed (1,1) and (1,-1), for now

@author: agardill
"""
# %% Imports

import numpy
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

import utils.tool_belt as tool_belt
from utils.tool_belt import States

# %% Presets

data_folder = 't1_double_quantum'

omega =  1.2
omega_ste = 0.6

# %% Minor functions

def expon_decay(t, rate, amp):
    return amp*numpy.exp(-rate*t)

# Function to get data from interleave files from folder
def extract_data(file_name, folder_name):
    # Call the data
    data = tool_belt.get_raw_data(data_folder, file_name, folder_name)
    
    # Define the num_runs
    num_runs = data['num_runs']
    
    # Get the splitting
    nv_sig = data['nv_sig']
    resonance_HIGH = nv_sig['resonance_HIGH']*10**3
    resonance_LOW = nv_sig['resonance_LOW']*10**3
    splitting_MHz = resonance_HIGH - resonance_LOW

    # Get the tau data and the (1,1) and (1,-1) sig and reference counts
    taus = numpy.array(data['tau_master_list'][0]) / 10**6 # us
    plus_minus_sig_counts = data['sig_counts_master_list'][0]
    plus_plus_sig_counts= data['sig_counts_master_list'][1]
    plus_minus_ref = numpy.average(data['ref_counts_master_list'][0])
    plus_plus_ref= numpy.average(data['ref_counts_master_list'][1])
    
    # Normalize the data using one averaged ref value
    plus_plus_norm_counts = plus_plus_sig_counts/plus_plus_ref
    plus_minus_norm_counts = plus_minus_sig_counts/plus_minus_ref
    
    return plus_plus_norm_counts, plus_minus_norm_counts, taus, \
                        num_runs, splitting_MHz
    
def plot_fig(x, y, y_sigma, fit_params, gamma, gamma_unc):
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        plus_time_linspace = numpy.linspace(0, x[-1], num=1000)
        #    ax = axes_pack[1]
        ax.errorbar(x, y, yerr = y_sigma, label = 'data', fmt = 'o', 
                    color = 'blue')
        
        ax.plot(plus_time_linspace,
            expon_decay(plus_time_linspace, *fit_params),
            'r', label = 'fit')
        
        ax.set_xlabel('Relaxation time (ms)')
        ax.set_ylabel('Normalized signal Counts')
        ax.set_title('(+1,+1) - (+1,-1)')
        ax.legend()
        text = r'$\gamma = $ {}$\pm${} kHz'.format('%.2f'%gamma, '%.2f'%gamma_unc)
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.55, 0.95, text, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=props)
        
        fig.canvas.draw()
        fig.canvas.flush_events()
        
    
# %% Main

def main(file_name, folder_name, num_bins, amp = None, offset = None):  
    # Make some lists to save data
    gamma_list = []
    gamma_ste_list = []
    gamma_fit_params_list = []
    gamma_counts_list = []
    
    # get the normalized counts for the two experiments, num_runs, and splitting
    
    plus_plus_norm_counts, plus_minus_norm_counts, taus, \
        num_runs, splitting_MHz = extract_data(file_name, folder_name)
    
    # Calculate the number of runs in each of the bins based on the num_bins
    bin_size = int(num_runs / num_bins)


    for bin_ind in range(num_bins):
        i = bin_ind * bin_size
        if i > num_runs:
            print('last bin_size too large, skipping last data point')
            break

        # For each slice, calculate the average normalized counts, and the std
        plus_plus_sliced_counts = \
            numpy.average(plus_plus_norm_counts[i:i+bin_size, ::], axis = 0)
        plus_minus_sliced_counts = \
            numpy.average(plus_minus_norm_counts[i:i+bin_size, ::], axis = 0)
    
        plus_plus_sliced_std = \
            numpy.std(plus_plus_norm_counts[i:i+bin_size, ::], axis = 0, ddof=1)
        plus_minus_sliced_std = \
            numpy.std(plus_minus_norm_counts[i:i+bin_size, ::], axis = 0, ddof=1)
        
        # Subtract the (1,1) and (1,-1) lists, and propegate error
        plus_subt_counts = plus_plus_sliced_counts - plus_minus_sliced_counts
        
        plus_subt_std = numpy.sqrt(plus_plus_sliced_std**2 + plus_minus_sliced_std**2)
        plus_subt_ste = plus_subt_std/numpy.sqrt(bin_size)
        
        # Save the counts for future use
        gamma_counts_list.append(plus_subt_counts.tolist())
        
        # if any ste are evaluated to 0, force them to be non-zero
        plus_subt_ste = [0.00001 if x==0 else x for x in plus_subt_ste]
        
        if amp == None:
            # Fit the data
            init_params = (10, 0.3)
            g_fit_params, g_pcov = curve_fit(expon_decay, taus, plus_subt_counts, 
                                            p0 = init_params, 
                                            sigma = plus_subt_ste, 
                                            absolute_sigma = True)

        else:
            # redefine the function so the rate is the only free paramter
            expon_decay_simp = lambda t, rate: expon_decay(t, rate, amp)
            
            init_params = (30)
            g_fit_params, g_pcov = curve_fit(expon_decay_simp, taus, plus_subt_counts, 
                                            p0 = init_params, 
                                            sigma = plus_subt_ste, 
                                            absolute_sigma = True)
        
        gamma_fit_params_list.append(g_fit_params.tolist())
        
        # Calculate gamma and the ste
        gamma = (g_fit_params[0] - omega)/2
        gamma_ste = numpy.sqrt(g_pcov[0][0] + omega_ste**2)/2
        
        gamma_list.append(gamma)
        gamma_ste_list.append(gamma_ste)
        
        # Plot the figure        
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        plus_time_linspace = numpy.linspace(0, taus[-1], num=1000)
        #    ax = axes_pack[1]
        ax.errorbar(taus, plus_subt_counts, yerr = plus_subt_ste,
                    label = 'data', fmt = 'o', 
                    color = 'blue')
        
        if amp == None and offset == None: 
            ax.plot(plus_time_linspace,
                expon_decay(plus_time_linspace, *g_fit_params),
                'r', label = 'fit')
        else:
            ax.plot(plus_time_linspace,
                expon_decay_simp(plus_time_linspace, *g_fit_params),
                'r', label = 'fit')
        
        ax.set_xlabel('Relaxation time (ms)')
        ax.set_ylabel('Normalized signal Counts')
        ax.set_title('(+1,+1) - (+1,-1)')
        ax.legend()
        text = r'$\gamma = $ {}$\pm${} kHz'.format('%.2f'%gamma, '%.2f'%gamma_ste)
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.55, 0.95, text, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=props)
        
        fig.canvas.draw()
        fig.canvas.flush_events()
        
    # Save data on each bin
    time_stamp = tool_belt.get_time_stamp()
    
    raw_data = {'time_stamp': time_stamp,
                    'level_splitting': splitting_MHz,
                    'level_splitting-units': 'MHz',
                    'num_runs': num_runs,
                    'num_bins': num_bins,
                    'bin_size': bin_size,
                    'gamma_list': gamma_list,
                    'gamma_ste_list': gamma_ste_list,
                    'gamma_fit_params_list': gamma_fit_params_list,
                    'gamma_counts_list': gamma_counts_list,
                    'taus': taus.tolist()
            }
    
    data_dir='E:/Shared drives/Kolkowitz Lab Group/nvdata'
        
    file_name = str('%.1f'%splitting_MHz) + '_MHz_splitting_' + str(num_bins) + '_bins_error' 
    file_path = '{}/{}/{}/{}'.format(data_dir, data_folder, folder_name, 
                                                     file_name)
    tool_belt.save_raw_data(raw_data, file_path)
    
# %% Run the file

if __name__ == '__main__':

    folder = 'nv1_2019_05_10_28MHz_4'
    file = '2019-08-27-13_45_39-ayrton12-nv1_2019_05_10'
    
    main(file, folder, 18, amp = 0.3019)