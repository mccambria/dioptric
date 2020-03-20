# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 16:00:15 2019

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


def subtraction_plot(ax, analysis_file_path):
    """
    This is adapted from Aedan's function of the same name in
    analysis/Paper Figures\Magnetically Forbidden Rate\supplemental_figures.py
    """
    
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
            
    zero_zero_time = numpy.array(zero_zero_time)
    try:
        times_15 = numpy.where(zero_zero_time > 15.0)[0][0]
    except:
        times_15 = None
    color = '#FF9933'
    facecolor = '#FFCC33'
    ax.scatter(zero_zero_time[:times_15], zero_relaxation_counts[:times_15],
               label=r'$F_{\Omega}$', zorder=5, marker='^', s=ms**2,
               color=color, facecolor=facecolor)
    zero_time_linspace = numpy.linspace(0, 15.0, num=1000)
    ax.plot(zero_time_linspace, exp_eq(zero_time_linspace, *omega_opti_params),
            color=color, linewidth=lw)
    
    omega_patch = mlines.Line2D([], [], label=r'$F_{\Omega}$', linewidth=lw,
                               marker='^', markersize=ms,
                               color=color, markerfacecolor=facecolor)
    
    plus_plus_time = numpy.array(plus_plus_time)
    try:
        times_15 = numpy.where(plus_plus_time > 15.0)[0][0]
    except:
        times_15 = None
    x_clip = numpy.array(plus_plus_time[:times_15])
    y_clip = numpy.array(plus_relaxation_counts[:times_15])
    # mask = numpy.array([el.is_integer() for el in x_clip])
    # ax.scatter(x_clip[mask], y_clip[mask])
    color = '#993399'
    facecolor = '#CC99CC'
    ax.scatter(x_clip, y_clip,
               label=r'$F_{\gamma}$', zorder=5, marker='o', s=ms**2,
               color=color, facecolor=facecolor)
    plus_time_linspace = numpy.linspace(0, 15.0, num=1000)
    gamma_rate = gamma_opti_params[0]
    gamma_opti_params[0] = gamma_rate
    gamma_opti_params_offset = gamma_opti_params + [manual_offset_gamma]
    ax.plot(plus_time_linspace, 
            exp_eq_offset(plus_time_linspace, *gamma_opti_params_offset),
            color=color, linewidth=lw)

    # ax.tick_params(which = 'both', length=8, width=2, colors='k',
    #             direction='in',grid_alpha=0.7)
    ax.set_xlabel(r'Wait time $\tau$ (ms)')
    ax.set_ylabel('Subtraction curve (arb. units)')
    ax.set_xlim(-0.5, 15.5)
    ax.set_yscale('log')
    
    gamma_patch = mlines.Line2D([], [], label=r'$F_{\gamma}$', linewidth=lw,
                               marker='o', markersize=ms,
                               color=color, markerfacecolor=facecolor,
                               )
    # ax.legend(handles=[omega_patch, gamma_patch], handlelength=lw)
    ax.legend(handleheight=1.6, handlelength=0.6)
    
    trans = ax.transAxes
    # trans = ax.get_figure().transFigure  # 0.030, 0.46
    ax.text(-0.13, 1.05, '(c)', transform=trans,
            color='black', fontsize=16)
            

# %% Main


def main(folder, file_high, file_zero, file_high_to_low,
         gamma, omega, pi_pulse_infidelity, analysis_file):

    # plt.rcParams.update({'font.size': 18})  # Increase font size
    # fig, axes_pack = plt.subplots(1,2, figsize=(10,5))
    fig = plt.figure(figsize=(6.75,6.75))
    gs = gridspec.GridSpec(2, 2)
    
    source = 't1_double_quantum/paper_data/bulk_dq/'
    path = source + folder
    
    # %% Level structure
    
    # Add a new axes, make it invisible, steal its rect
    ax = fig.add_subplot(gs[0, 0])
    ax.set_axis_off()
    ax.text(-0.295, 1.05, '(a)', transform=ax.transAxes,
            color='black', fontsize=16)
    
    ax = plt.Axes(fig, [0.0, 0.51, 0.5, 0.43])
    ax.set_axis_off()
    fig.add_axes(ax)
    # print(gs)
    # fig.add_axes(gs[0, 0])
    # ax = fig.add_subplot(gs[0, 0])
    
    # ax = axes_pack[0]
    # ax.set_axis_off()
    file = 'C:/Users/matth/Desktop/lab/bulk_dq_relaxation/figures_revision2/main1/level_structure.png'
    img = mpimg.imread(file)
    img_plot = ax.imshow(img)
    
    # l, b, w, h = ax.get_position().bounds
    # ax.set_position([l, b -1.0, w, h])
    # ax.set_axis_off()
    # ax.axis('off')

    # %% Relaxation out of plots
    
    ax = fig.add_subplot(gs[0, 1])
    # ax = axes_pack[1]
    
    # Get reference values for to convert fluorescence to population
    ref_range = [None, None]
    times = [0.0, 15.0]

    # Reference for 0
    data = tool_belt.get_raw_data(path, file_high_to_low)
    # ref_range[0] = get_first_norm_avg_sig(data)
    ref_range[0] = 0.64 

    # Reference for 1
    data = tool_belt.get_raw_data(path, file_zero)
    ref_range[1] = get_first_norm_avg_sig(data)
    # print(ref_range)
    # return

    raw_data_zero = tool_belt.get_raw_data(path, file_zero)
    signal_zero, ste_zero, times_zero = process_raw_data(raw_data_zero,
                                                         ref_range)
    smooth_t = numpy.linspace(times[0], times[-1], 1000)
    fit_zero = relaxation_zero_func(smooth_t,
                                    gamma, omega, pi_pulse_infidelity)

    raw_data_high = tool_belt.get_raw_data(path, file_high)
    signal_high, ste_high, times_high = process_raw_data(raw_data_high,
                                                         ref_range)
    smooth_t = numpy.linspace(times[0], times[-1], 1000)
    fit_high = relaxation_high_func(smooth_t,
                                    gamma, omega, pi_pulse_infidelity)

    ax.set_xlabel(r'Wait time $\tau$ (ms)')
    ax.set_ylabel('Fluorescence (arb. units)')
    ax.set_xticks([0,5,10,15])
    # ax.set_xlabel('test')
    # ax.set_yscale('log')

    # Plot zero
    color = '#0D83C5'
    facecolor = '#56B4E9'
    zero_label = 'Relaxation \nout of {}'.format(r'$\ket{0}$')
    zero_patch = mlines.Line2D([], [], label=zero_label, linewidth=lw,
                               marker='s', color=color,
                               markerfacecolor=facecolor, markersize=ms)
    ax.plot(smooth_t, fit_zero, color=color, linewidth=lw)
    try:
        times_15 = numpy.where(times_zero > 15.0)[0][0]
    except:
        times_15 = None
    ax.scatter(times_zero[:times_15], signal_zero[:times_15], 
               label=zero_label, zorder=5, marker='s',
               color=color, facecolor=facecolor, s=ms**2)

    # Plot high
    color = '#D2C40E'
    facecolor = '#F0E442'
    high_label = 'Relaxation \nout of {}'.format(r'$\ket{+1}$')
    high_patch = mlines.Line2D([], [], label=high_label, linewidth=lw,
                               marker='D', color=color,
                               markerfacecolor=facecolor, markersize=ms)
    ax.plot(smooth_t, fit_high, color=color, linewidth=lw)
    try:
        times_15 = numpy.where(times_high > 15.0)[0][0]
    except:
        times_15 = None
    ax.scatter(times_high[:times_15], signal_high[:times_15],
               label=high_label, zorder=5, marker='D',
               color=color, facecolor=facecolor, s=ms**2)
    # ax.legend(handles=[zero_patch, high_patch], handlelength=lw)
    ax.legend(handleheight=2.0, handlelength=0.6)
    
    ax.text(-0.25, 1.05, '(b)', transform=ax.transAxes,
            color='black', fontsize=16)

    # %% Subtraction plots
    
    ax = fig.add_subplot(gs[1,:])
    source = 'E:/Shared drives/Kolkowitz Lab Group/nvdata/' 
    analysis_file_path = source + path + '/' + analysis_file
    subtraction_plot(ax, analysis_file_path)
    
    # %% Wrap up
    
    fig.tight_layout(pad=0.5)
    # fig.tight_layout()
    

# %% Run


if __name__ == '__main__':
    
    plt.rcParams['text.latex.preamble'] = [
        r'\usepackage{physics}',
        r'\usepackage{sfmath}',
        r'\usepackage{upgreek}',
        r'\usepackage{helvet}',
       ]  
    plt.rcParams.update({'font.size': 13})
    plt.rcParams.update({'font.family': 'sans-serif'})
    plt.rcParams.update({'font.sans-serif': ['Helvetica']})
    plt.rc('text', usetex=True)

    # This assumes the num_steps and relaxation_time_range are the same for
    # both data sets
    
    # folder = 'goeppert_mayer-nv7_2019_11_27-167MHz'
    # analysis_file = '167MHz_splitting_rate_analysis.txt'
    # file_high = '2019_11_29-11_26_00-goeppert_mayer-nv7_2019_11_27'
    # file_zero = '2019_12_01-05_31_53-goeppert_mayer-nv7_2019_11_27'
    # file_high_to_low = '2019_11_28-14_23_04-goeppert_mayer-nv7_2019_11_27'
    # gamma = 0.132
    # omega = 0.056
    # pi_pulse_infidelity = (1.0 - numpy.exp(-111/1398)) # 7.6%
    
    folder = 'johnson-nv0_2020_03_13-122MHz'
    analysis_file = '104MHz_splitting_rate_analysis.txt'
    file_high = '2020_03_16-05_16_47-johnson-nv0_2020_03_13'
    file_zero = '2020_03_17-18_32_12-johnson-nv0_2020_03_13'
    file_high_to_low = '2020_03_15-10_39_07-johnson-nv0_2020_03_13'
    gamma = 0.092
    omega = 0.050
    pi_pulse_infidelity = 1.0 - numpy.exp(-91.6/(2*721))  # 6.2%
    
    # folder = 'goeppert_mayer-nv7_2019_11_27-123MHz-0deg'
    # analysis_file = '123MHz_splitting_rate_analysis.txt'
    # file_high = '2020_01_19-15_39_46-goeppert_mayer-nv7_2019_11_27'
    # file_zero = '2020_01_20-17_10_45-goeppert_mayer-nv7_2019_11_27'
    # file_high_to_low = '2020_01_19-04_06_24-goeppert_mayer-nv7_2019_11_27'
    # gamma = 0.113
    # omega = 0.055
    # pi_pulse_infidelity = (1.0 - numpy.exp(-67/671))
    
    # folder = 'goeppert_mayer-nv7_2019_11_27-48deg_2'
    # analysis_file = '122MHz_splitting_rate_analysis.txt'
    # file_high = '2020_01_16-18_15_45-goeppert_mayer-nv7_2019_11_27'
    # file_zero = '2020_01_17-19_47_11-goeppert_mayer-nv7_2019_11_27'
    # file_high_to_low = '2020_01_16-06_42_21-goeppert_mayer-nv7_2019_11_27'
    # gamma = 0.150
    # omega = 0.055
    # pi_pulse_infidelity = 0.0
    
    # folder = 'johnson-nv1_2019_07_24-130MHz'
    # analysis_file = '130MHz_splitting_rate_analysis.txt'
    # file_high = '2019-07-24_14-25-06_johnson1'
    # file_zero = '2019-07-25_10-30-31_johnson1'
    # file_high_to_low = '2019-07-24_04-22-45_johnson1'
    # gamma = 0.114
    # omega = 0.060
    # pi_pulse_infidelity = 0.0

    main(folder, file_high, file_zero, file_high_to_low,
         gamma, omega, pi_pulse_infidelity, analysis_file)

