# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 13:39:38 2023

@author: kolkowitz
"""


import utils.tool_belt as tool_belt
import utils.kplotlib as kpl
from utils.kplotlib import KplColors
from numpy import pi
import numpy
import time
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def plot(file0, file1, file2):
    kpl.init_kplotlib()
    
    data = tool_belt.get_raw_data(file0)
    sig_counts = data['sig_counts']
    ref_counts = data['ref_counts']
    num_reps = data['num_reps']
    nv_sig = data['nv_sig']
    readout = nv_sig['charge_readout_dur']
    norm_style = tool_belt.NormStyle.SINGLE_VALUED
    
    relaxation_time_range = data['relaxation_time_range']
    num_steps = data['num_steps']
    
    taus0 = numpy.linspace(
        relaxation_time_range[0],
        relaxation_time_range[-1],
        num=num_steps,
    )/1e6
    
    ret_vals = tool_belt.process_counts(sig_counts, ref_counts, num_reps, readout, norm_style)
    (
        sig_counts_avg_kcps,
        ref_counts_avg_kcps,
        norm_avg_sig0,
        norm_avg_sig_ste0,
    ) = ret_vals
    
    
    data = tool_belt.get_raw_data(file1)
    sig_counts = data['sig_counts']
    ref_counts = data['ref_counts']
    num_reps = data['num_reps']
    nv_sig = data['nv_sig']
    readout = nv_sig['charge_readout_dur']
    norm_style = tool_belt.NormStyle.SINGLE_VALUED
    
    relaxation_time_range = data['relaxation_time_range']
    num_steps = data['num_steps']
    
    taus1 = numpy.linspace(
        relaxation_time_range[0],
        relaxation_time_range[-1],
        num=num_steps,
    )/1e6
    
    ret_vals = tool_belt.process_counts(sig_counts, ref_counts, num_reps, readout, norm_style)
    (
        sig_counts_avg_kcps,
        ref_counts_avg_kcps,
        norm_avg_sig1,
        norm_avg_sig_ste1,
    ) = ret_vals
    
    data = tool_belt.get_raw_data(file2)
    sig_counts = data['sig_counts']
    ref_counts = data['ref_counts']
    num_reps = data['num_reps']
    nv_sig = data['nv_sig']
    readout = nv_sig['charge_readout_dur']
    norm_style = tool_belt.NormStyle.SINGLE_VALUED
    
    relaxation_time_range = data['relaxation_time_range']
    num_steps = data['num_steps']
    
    taus2 = numpy.linspace(
        relaxation_time_range[0],
        relaxation_time_range[-1],
        num=num_steps,
    )/1e6
    
    ret_vals = tool_belt.process_counts(sig_counts, ref_counts, num_reps, readout, norm_style)
    (
        sig_counts_avg_kcps,
        ref_counts_avg_kcps,
        norm_avg_sig2,
        norm_avg_sig_ste2,
    ) = ret_vals
    
    x_smooth = numpy.linspace(taus2[0], taus2[-1], 1000)
    
    fit_func = lambda x, amp, decay: tool_belt.exp_decay(x, amp, decay, 1-amp)
    #amp, decay, offset
    init_params = [ -0.02, 5]
    popt0, pcov0 = curve_fit(
        fit_func,
          taus0,
        norm_avg_sig0,
        sigma=norm_avg_sig_ste0,
        absolute_sigma=True,
        p0=init_params,
    )
    print(popt0)
    
    popt1, pcov1 = curve_fit(
        fit_func,
          taus1,
        norm_avg_sig1,
        sigma=norm_avg_sig_ste1,
        absolute_sigma=True,
        p0=init_params,
    )
    print(popt1)
    popt2, pcov2 = curve_fit(
        fit_func,
          taus2,
        norm_avg_sig2,
        sigma=norm_avg_sig_ste2,
        absolute_sigma=True,
        p0=init_params,
    )
    print(popt2)
        

    # Plot setup
    fig, ax = plt.subplots(1, 1)
    ax.set_xlabel('Relaxation time (ms)')
    ax.set_ylabel("Normalized signal")
    # ax.set_title(title)

    # Plotting
    kpl.plot_points(ax,  taus0, norm_avg_sig0,yerr= norm_avg_sig_ste0, label = 'V_offset = -0.01 V', color=KplColors.BLACK)
    # kpl.plot_points(ax,  taus1, norm_avg_sig1,yerr= norm_avg_sig_ste1,  label =  'V_offset = -0.005 V', color=KplColors.RED)
    kpl.plot_points(ax,  taus2, norm_avg_sig2,yerr= norm_avg_sig_ste2,  label =  'V_offset = 0.00 V', color=KplColors.BLUE)
    
    
    label0 = 'Amplitude {:.3f} +/- {:.3f}'.format(popt0[0],numpy.sqrt(pcov0[0][0]) )
    label1 = 'Amplitude {:.3f} +/- {:.3f}'.format(popt1[0],numpy.sqrt(pcov1[0][0]) )
    label2 = 'Amplitude {:.3f} +/- {:.3f}'.format(popt2[0],numpy.sqrt(pcov2[0][0]) )
    kpl.plot_line(ax, x_smooth, fit_func(x_smooth,*popt0 ), label = label0, color=KplColors.BLACK)
    # kpl.plot_line(ax, x_smooth, fit_func(x_smooth,*popt1 ), label = label1, color=KplColors.RED)
    kpl.plot_line(ax, x_smooth, fit_func(x_smooth,*popt2 ), label = label2, color=KplColors.BLUE)
    # text = "Slope {:.4f}, Offset {:.4f} V".format(popt[0], popt[1])
    # kpl.anchored_text(ax, text, kpl.Loc.LOWER_RIGHT, size= kpl.Size.SMALL)
    
    ax.legend()
    

file0 = '2023_03_29-19_56_08-siena-nv0_2023_03_20'
file1='2023_03_29-23_26_03-siena-nv0_2023_03_20'
file2='2023_03_30-05_55_17-siena-nv0_2023_03_20'
plot(file0, file1, file2)