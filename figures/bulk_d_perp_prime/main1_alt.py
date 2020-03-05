# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 17:35:45 2020

@author: matth
"""


# %% Imports


import numpy
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar as scale_bar
import json
import utils.tool_belt as tool_belt
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


def main(file_names):
    """
    2 x 2 figure. Top left ax blank for level structure, next 3 for sample
    scans. file_names should be a list with 3 paths to the appropriate
    raw data files.
    """
    
    plt.rcParams.update({'font.size': 18})  # Increase font size
    fig, axes_pack = plt.subplots(1, 3, figsize=(15,5))
    fig.set_tight_layout(True)
    ticks = [[10,20,30,40,50,60],
             [20,40,60,80,100],
             [15,20,25,30]]
    
    for ind in range(3):
        
        ax = axes_pack[ind]
        name = file_names[ind]
        
        # This next bit is jacked from image_sample.reformat_plot 2/29/2020
        with open(name) as file:

            # Load the data from the file
            data = json.load(file)

            # Build the image array from the data
            # Not sure why we're doing it this way...
            img_array = []
            try:
                file_img_array = data['img_array']
            except:
                file_img_array = data['imgArray']
            for line in file_img_array:
                img_array.append(line)

            # Get the readout in s
            readout = float(data['readout']) / 10**9

            try:
                xScanRange = data['x_range']
                yScanRange = data['y_range']
            except:
                xScanRange = data['xScanRange']
                yScanRange = data['yScanRange']
            
        kcps_array = (numpy.array(img_array) / 1000) / readout

        # ax.set_xlabel('Position ($\mu$m)')
        # ax.set_ylabel('Position ($\mu$m)')
        
        scale = 35  # galvo scaling in microns / volt
        
        # Plot!
        img = ax.imshow(kcps_array, cmap='inferno', interpolation='none')
        ax.set_axis_off()
        
        # Scale bar
        # Find the number of pixels in a micron
        num_steps = kcps_array.shape[0]
        v_resolution = xScanRange / num_steps  # resolution in volts / pixel
        resolution = v_resolution * scale  # resolution in microns / pixel
        px_per_micron = int(1/resolution)
        trans = ax.transData
        bar = scale_bar(trans, 5*px_per_micron, '5 $\mu$m', 'upper right',
                        size_vertical=int(num_steps/100))
        ax.add_artist(bar)

        # Add the color bar
        cbar = fig.colorbar(img, ax=ax, ticks=ticks[ind])
        cbar.ax.set_title('kcps')
    
    
def main2(folder, file_high, file_zero, file_high_to_low,
         gamma, omega, pi_pulse_infidelity, analysis_file):

    plt.rcParams.update({'font.size': 18})  # Increase font size
    fig, ax = plt.subplots(1, 1, figsize=(6,6))
    fig.set_tight_layout(True)
    
    source = 't1_double_quantum/paper_data/bulk_dq/'
    path = source + folder

    # %% Relaxation out of plots
    
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


# %% Run


if __name__ == '__main__':

    path = 'E:/Shared drives/Kolkowitz Lab Group/nvdata/image_sample/'
    file_names = ['2019_07/2019-07-23_17-39-48_johnson1.txt',
                  '2019_10/2019-10-02-15_12_01-goeppert_mayer-nv_search.txt',
                  '2019_04/2019-04-15_16-42-08_Hopper.txt']
    file_names = [path+name for name in file_names]
    # main(file_names)
    
    plt.rc('text', usetex=True)
    plt.rcParams['text.latex.preamble'] = [
       r'\usepackage{physics}',
       ]  

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

    main2(folder, file_high, file_zero, file_high_to_low,
         gamma, omega, pi_pulse_infidelity, analysis_file)
