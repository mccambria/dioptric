# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 17:35:30 2020

@author: matth
"""


# %% Imports


import numpy
import matplotlib
import matplotlib.pyplot as plt
import utils.tool_belt as tool_belt
import json
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
    
ms = 7
lw = 1.75


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
    # avg_ref = numpy.average(ref_counts[::], axis=0)  # test norm per point

    # Divide signal by reference to get normalized counts and st error
    norm_avg_sig = avg_sig_counts / avg_ref
    norm_avg_sig_ste = ste_sig_counts / avg_ref

    # Normalize to the reference range
    # diff = ref_range[1] - ref_range[0]
    # norm_avg_sig = (norm_avg_sig - ref_range[0]) / diff

    return norm_avg_sig, norm_avg_sig_ste, times
            

# %% Main


def main(folder, file_high, file_low):

    fig, ax = plt.subplots()
    
    source = 't1_double_quantum/data_folders/paper_data/bulk_dq/'
    path = source + folder
    
    # Get reference values for to convert fluorescence to population
    ref_range = [None, None]

    raw_data_high = tool_belt.get_raw_data(path, file_high)
    signal_high, ste_high, times_high = process_raw_data(raw_data_high,
                                                         ref_range)

    raw_data_low = tool_belt.get_raw_data(path, file_low)
    signal_low, ste_low, times_low = process_raw_data(raw_data_low,
                                                      ref_range)

    ax.set_xlabel(r'Wait time $\tau$ (ms)')
    ax.set_ylabel('Fluorescence (arb. units)')

    # Plot high
    high_label = '0,+1'
    ax.errorbar(times_high, signal_high, yerr=ste_high, label=high_label)

    # Plot low
    low_label = '0,-1'
    ax.errorbar(times_low, signal_low, yerr=ste_low, label=low_label)
    
    ax.legend()
    

# %% Run


if __name__ == '__main__':
    
    plt.rcParams['text.latex.preamble'] = [
        r'\usepackage{physics}',
        r'\usepackage{sfmath}',
        r'\usepackage{upgreek}',
        r'\usepackage{helvet}',
       ]  
    plt.rcParams.update({'font.size': 15})
    plt.rcParams.update({'font.family': 'sans-serif'})
    plt.rcParams.update({'font.sans-serif': ['Helvetica']})
    plt.rc('text', usetex=True)
    
    folder = 'goeppert_mayer-nv7_2019_11_27-1662MHz-7deg'
    file_high = '2020_02_01-00_31_35-goeppert_mayer-nv7_2019_11_27'  # 0,+1 run
    file_low = '2020_02_03-12_05_08-goeppert_mayer-nv7_2019_11_27'  # 0,-1 run

    main(folder, file_high, file_low)
