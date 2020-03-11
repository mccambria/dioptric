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
            

# %% Main


def main(folder, file_high, file_zero, file_high_to_low,
         gamma, omega, pi_pulse_infidelity):

    # plt.rcParams.update({'font.size': 18})  # Increase font size
    fig, axes_pack = plt.subplots(1,2, figsize=(10,5))
    # fig.set_tight_layout(True)
    
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
    ax.set_xticks([0,5,10,15])
    # ax.set_xlabel('test')
    # ax.set_yscale('log')

    # Plot zero
    ket = r'$\ket{0}$'
    label = 'Relaxation \nout of {}'.format(ket)
    zero_patch = mlines.Line2D([], [], label=label, linewidth=2.2,
                               marker='^', color='#FFCC33',
                               markeredgecolor='#FF9933', markersize=8)
    ax.plot(smooth_t, fit_zero, color='#FFCC33', linewidth=2.2)
    ax.scatter(times_zero, signal_zero, zorder=5, marker='^',
               color='#FFCC33', edgecolor='#FF9933', s=64)

    # Plot high
    ket = r'$\ket{+1}$'
    label = 'Relaxation \nout of {}'.format(ket)
    high_patch = mlines.Line2D([], [], label=label, linewidth=2.2,
                               marker='o', color='#CC99CC',
                               markeredgecolor='#993399', markersize=8)
    ax.plot(smooth_t, fit_high, color='#CC99CC', linewidth=2.2)
    ax.scatter(times_high, signal_high, zorder=5, marker='o',
               color='#CC99CC', edgecolor='#993399', s=64)
    ax.legend(handles=[zero_patch, high_patch], handlelength=1.0)
    
    # %% T2max
    
    ax = axes_pack[1]

    ax.set_xlabel(r'$T_{1}^{(0)}$ (ms)')
    ax.set_ylabel(r'$T_{2}$ (ms)')
    
    t2max_right = lambda T10: 2*((T10**-1) + (2/3)*(T10**-1))**-1
    t2max_wrong = lambda T10: 2*T10
    
    smooth_T10 = numpy.linspace(1, 250, 1000, dtype=float)
    ax.plot(smooth_T10, t2max_right(smooth_T10),
            label=r'$T_{2,\text{max}}$', color='red')
    ax.plot(smooth_T10, t2max_wrong(smooth_T10),
            label=r'$2T_{1}^{(0)}$', color='gray', ls='dashed')
    
    ax.legend(handlelength=1.0)
    
    fig.tight_layout(w_pad=1.5)
    

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
    file_high = '2019_11_29-11_26_00-goeppert_mayer-nv7_2019_11_27'
    file_zero = '2019_12_01-05_31_53-goeppert_mayer-nv7_2019_11_27'
    file_high_to_low = '2019_11_28-14_23_04-goeppert_mayer-nv7_2019_11_27'
    gamma = 0.132
    omega = 0.056
    pi_pulse_infidelity = (1.0 - numpy.exp(-111/1398)) # 7.6%

    main(folder, file_high, file_zero, file_high_to_low,
         gamma, omega, pi_pulse_infidelity)

