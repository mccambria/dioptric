# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 16:00:15 2019

@author: matth
"""


# %% Imports


import numpy
import matplotlib.pyplot as plt
import utils.tool_belt as tool_belt
import json
import matplotlib.patches as mpatches
import matplotlib.lines as mlines


# %% Functions


def process_raw_data(data, ref_range):
    """Pull the relaxation signal and ste out of the raw data."""

    num_runs = data['num_runs']
    num_steps = data['num_steps']
    sig_counts  = numpy.array(data['sig_counts'])
    ref_counts = numpy.array(data['ref_counts'])
    time_range = numpy.array(data['relaxation_time_range'])

    # Calculate time arrays in ms
    min_time, max_time = time_range / 10**6
    times = numpy.linspace(min_time, max_time, num=num_steps)

    # Calculate the average signal counts over the runs, and ste
    avg_sig_counts = numpy.average(sig_counts[::], axis=0)
    ste_sig_counts = numpy.std(sig_counts[::], axis=0, ddof = 1) / numpy.sqrt(num_runs)

    # Assume reference is constant and can be approximated to one value
    avg_ref = numpy.average(ref_counts[::])

    # Divide signal by reference to get normalized counts and st error
    norm_avg_sig = avg_sig_counts / avg_ref
    norm_avg_sig_ste = ste_sig_counts / avg_ref

    # Normalize to the reference range
    diff = ref_range[1] - ref_range[0]
    norm_avg_sig = (norm_avg_sig - ref_range[0]) / diff

    return norm_avg_sig, norm_avg_sig_ste, times


def relaxation_zero_func(t, gamma, omega, infid):

    return (1/3) + (2/3) * numpy.exp(-3 * omega * t)


def relaxation_high_func(t, gamma, omega, infid):

    first_term = (1/3) + (1/2) * ((1-infid)**2) * numpy.exp(-(2 * gamma + omega) * t)
    second_term = (-1/2) * (infid - (1/3)) * numpy.exp(-3 * omega * t) * (1-infid)
    third_term = (infid - (1/3)) * numpy.exp(-3 * omega * t) * infid
    return first_term + second_term + third_term


def get_first_norm_avg_sig(data):

    sig_counts  = numpy.array(data['sig_counts'])
    ref_counts = numpy.array(data['ref_counts'])
    avg_ref = numpy.average(ref_counts[::])
    avg_sig_counts = numpy.average(sig_counts[::], axis=0)
    return avg_sig_counts[0] / avg_ref


def exp_eq(t, rate, amp):
    return  amp * numpy.exp(- rate * t)


def exp_eq_offset(t, rate, amp, offset):
    return  amp * numpy.exp(- rate * t) + offset


def subtraction_plot(axes_pack, analysis_file_path):
    """
    This is adapted from Aedan's function of the same name in
    analysis/Paper Figures\Magnetically Forbidden Rate\supplemental_figures.py
    """
    
    text_font = 16
    title_font = 20
    
    with open(analysis_file_path) as file:
            data = json.load(file)
            
            zero_relaxation_counts = data['zero_relaxation_counts']
            zero_relaxation_ste = numpy.array(data['zero_relaxation_ste'])
            zero_zero_time = data['zero_zero_time']
            
            plus_relaxation_counts = data['plus_relaxation_counts']
            plus_relaxation_ste = numpy.array(data['plus_relaxation_ste'])
            plus_plus_time = data['plus_plus_time']

            omega_opti_params = data['omega_opti_params']
            gamma_opti_params = data['gamma_opti_params']
            manual_offset_gamma = data['manual_offset_gamma']
            
    ax = axes_pack[0]
    
    ax.errorbar(zero_zero_time, zero_relaxation_counts,
                        yerr = zero_relaxation_ste,
                        label = 'data',  fmt = 'o', color = 'blue')
    zero_time_linspace = numpy.linspace(0, zero_zero_time[-1], num=1000)
    ax.plot(zero_time_linspace,
                exp_eq(zero_time_linspace, *omega_opti_params),
                'r', label = 'fit')
    ax.set_xlabel(r'Wait time, $\tau$ (ms)', fontsize=text_font)
    ax.set_ylabel(r'$F_{\Omega}$ (arb. units)', fontsize=text_font)
#    ax.set_title(r'$P_{0,0} - P_{0,1}$', fontsize=title_font)
#    ax.legend(fontsize=20)
    ax.tick_params(which = 'both', length=8, width=2, colors='k',
                direction='in',grid_alpha=0.7, labelsize = text_font)
    
    ax = axes_pack[1]
    
    ax.errorbar(numpy.array(plus_plus_time), plus_relaxation_counts,
                        yerr = plus_relaxation_ste,
                        label = 'data', fmt = 'o', color = 'blue')
    plus_time_linspace = numpy.linspace(0, plus_plus_time[-1], num=1000)
    gamma_rate = gamma_opti_params[0]
    gamma_opti_params[0] = gamma_rate
    gamma_opti_params_offset = gamma_opti_params + [manual_offset_gamma]
    ax.plot(plus_time_linspace,
                exp_eq_offset(plus_time_linspace, *gamma_opti_params_offset),
                'r', label = 'fit')
    ax.set_xlabel(r'Wait time, $\tau$ (ms)', fontsize=text_font)
    ax.set_ylabel(r'$F_{\gamma}$ (arb. units)', fontsize=text_font)
#    ax.set_title(r'$P_{1,1} - P_{1,-1}$', fontsize=title_font)
#    ax.legend(fontsize=20)

    ax.tick_params(which = 'both', length=8, width=2, colors='k',
                direction='in',grid_alpha=0.7, labelsize = text_font)
            

# %% Main


def main(folder, file_high, file_zero, file_high_to_low,
         gamma, omega, pi_pulse_infidelity, analysis_file):

    fig, axes_pack = plt.subplots(3, 1, figsize=(10,15))
    fig.set_tight_layout(True)
    
    source = 't1_double_quantum/paper_data/bulk_dq/'
    path = source + folder

    # %% Relaxation out of plots
    
    ax = axes_pack[0]
    
    # Get reference values for to convert fluorescence to population
    ref_range = [None, None]

    # Reference for 0
    data = tool_belt.get_raw_data(path, file_high_to_low)
    ref_range[0] = get_first_norm_avg_sig(data)

    # Reference for 1
    data = tool_belt.get_raw_data(path, file_zero)
    ref_range[1] = get_first_norm_avg_sig(data)

    raw_data_zero = tool_belt.get_raw_data(path, file_zero)
    signal_zero, ste_zero, times_zero = process_raw_data(raw_data_zero,
                                                         ref_range)
    smooth_t = numpy.linspace(times_zero[0], times_zero[-1], 1000)
    fit_zero = relaxation_zero_func(smooth_t,
                                    gamma, omega, pi_pulse_infidelity)

    raw_data_high = tool_belt.get_raw_data(path, file_high)
    signal_high, ste_high, times_high = process_raw_data(raw_data_high,
                                                         ref_range)
    smooth_t = numpy.linspace(times_high[0], times_high[-1], 1000)
    fit_high = relaxation_high_func(smooth_t,
                                    gamma, omega, pi_pulse_infidelity)

    ax.set_xlabel(r'Wait time, $\tau$ (ms)')
    ax.set_ylabel('Fluorescence (arb. units)')
    # ax.set_xlabel('test')
    # ax.set_yscale('log')

    # Plot zero
    label = r'Relaxation out of $\ket{0}$'
    zero_patch = mlines.Line2D([], [], label=label, linewidth=2.2,
                               marker='^', color='#FFCC33',
                               markeredgecolor='#FF9933', markersize=8)
    ax.plot(smooth_t, fit_zero, color='#FFCC33', linewidth=2.2)
    ax.scatter(times_zero, signal_zero, zorder=5, marker='^',
               color='#FFCC33', edgecolor='#FF9933', s=64)

    # Plot high
    label = r'Relaxation out of $\ket{+1}$'
    high_patch = mlines.Line2D([], [], label=label, linewidth=2.2,
                               marker='o', color='#CC99CC',
                               markeredgecolor='#993399', markersize=8)
    ax.plot(smooth_t, fit_high, color='#CC99CC', linewidth=2.2)
    ax.scatter(times_high, signal_high, zorder=5, marker='o',
               color='#CC99CC', edgecolor='#993399', s=64)
    ax.legend(handles=[zero_patch, high_patch])
    
    # %% F Omega and gamma
    
    source = 'E:/Shared drives/Kolkowitz Lab Group/nvdata/' 
    analysis_file_path = source + path + '/' + analysis_file
    subtraction_plot(axes_pack[1:], analysis_file_path)
    

# %% Run


if __name__ == '__main__':
    
    plt.rc('text', usetex=True)
    plt.rcParams['text.latex.preamble'] = [
        r'\usepackage{physics}',
        r'\usepackage{sfmath}',
       ]  
    plt.rcParams.update({'font.family': 'sans-serif'})  # Increase font size
    plt.rcParams.update({'font.size': 21})  # Increase font size

    # This assumes the num_steps and relaxation_time_range are the same for
    # both data sets
    folder = 'goeppert_mayer-nv7_2019_11_27-167MHz'
    analysis_file = '167MHz_splitting_rate_analysis.txt'
    file_high = '2019_11_29-11_26_00-goeppert_mayer-nv7_2019_11_27'
    file_zero = '2019_12_01-05_31_53-goeppert_mayer-nv7_2019_11_27'
    file_high_to_low = '2019_11_28-14_23_04-goeppert_mayer-nv7_2019_11_27'
    gamma = 0.132
    omega = 0.056
    pi_pulse_infidelity = (1.0 - numpy.exp(-111/1398)) # 7.6%

    main(folder, file_high, file_zero, file_high_to_low,
         gamma, omega, pi_pulse_infidelity, analysis_file)

