# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 10:08:38 2019

@author: Aedan
"""
import json
import numpy
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import utils.tool_belt as tool_belt
from utils.tool_belt import States
from scipy import exp
from scipy.optimize import curve_fit

#%%

def exp_eq(t, rate, amp):
    return  amp * exp(- rate * t)

# %%

def rate_comaprison_bulk_log(save=False):
    text_font = 30
    orange = '#f7941d'
    purple = '#87479b'
    
    file = '23.9_MHz_splitting_rate_analysis'
    folder = 'nv0_2019_06_27_23MHz'
    
#    file = '233.2_MHz_splitting_rate_analysis'
#    folder = 'nv0_2019_06_27_233MHz'
    
#    file = '125.9_MHz_splitting_rate_analysis'
#    folder = 'nv0_2019_06_27_126MHz'
    data = tool_belt.get_raw_data('t1_double_quantum.py', '{}\\{}'.format(folder, file))
 
    
    counts_o = data['zero_relaxation_counts']
    error_o = numpy.array(data['zero_relaxation_ste'])
    counts_g = data['plus_relaxation_counts']
    error_g = numpy.array(data['plus_relaxation_ste'])
    
    time = numpy.array(data['zero_zero_time']) * 1000
    time_linspace = numpy.linspace(time[0], time[-1], 1000)
    
    
    gamma = data['gamma'] / 1000
    gamma_unc = data['gamma_ste'] / 1000
    amp_g = data['gamma_opti_params'][1]
    omega = data['omega'] / 1000
    omega_unc = data['omega_std_error'] / 1000
    amp_o = data['omega_opti_params'][1]
    
    opti_params_o = [(3*omega), amp_o]
    opti_params_g = [(2*gamma+omega), amp_g]
    
    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(12,8))
    ax.set_yscale("log", nonposy='clip')
    ax.set_xlim([-500,8500])
    ax.set_ylim([3.3*10**-3,5.5*10**-1])
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y,pos: ('{{:.{:1d}f}}'.format(int(numpy.maximum(-numpy.log10(y),0)))).format(y)))
    
    ax.errorbar(time, counts_o, yerr = error_o, 
                fmt = '^', markersize = 12, color = orange)

    yfit = exp_eq(time_linspace, *opti_params_o)
    ax.plot(time_linspace, yfit, '-', color=orange)
    
#    opti_params_f[0] = (2*(gamma_f + gamma_unc_f) + omega + omega_unc)
#    yupper = exp_eq_offset(time_linspace, *opti_params_f)
#    opti_params_f[0] = (2*(gamma_f - gamma_unc_f) + omega - omega_unc)
#    ylower = exp_eq_offset(time_linspace, *opti_params_f)
    
#    ax.fill_between(time_linspace, yupper,  ylower,
#                     color='blue', alpha=0.4)
    
    
    ax.errorbar(time, counts_g, yerr = error_g, 
                fmt = 'o', markersize = 12, color = purple)
#    ax.plot(time, counts_s, '^', color =red)
    
    yfit = exp_eq(time_linspace, *opti_params_g)
    ax.plot(time_linspace, yfit, '-', color=purple)
    
#    opti_params_s[0] = (2*(gamma_s + gamma_unc_s) + omega + omega_unc)
#    yupper = exp_eq_offset(time_linspace, *opti_params_s)
#    opti_params_s[0] = (2*(gamma_s - gamma_unc_s) + omega - omega_unc)
#    ylower = exp_eq_offset(time_linspace, *opti_params_s)
#    
#    ax.fill_between(time_linspace, yupper,  ylower,
#                     color='red', alpha=0.4)
    
    ax.set_xlabel(r'Wait time ($\mu$s)', fontsize=text_font)
    ax.set_ylabel('Signal (arb. units)', fontsize=text_font)
    ax.tick_params(which = 'both', length=8, width=3, colors='k',
                direction = 'in', grid_alpha=0.7, labelsize = text_font)
    ax.tick_params(which = 'major', length=20, width=4)
    ax.grid(axis='y')
#    ax.set_xlim()
#    ax.legend()
#    ax.set_title('Compare NV1 measurements')
    
    fig.canvas.draw()
    fig.canvas.flush_events()
    
    if save:
        fig.savefig("C:/Users/Aedan/Creative Cloud Files/Paper Illustrations/Magnetically Forbidden Rate/supplemental_materials/NV2_rate_compare_log.pdf", bbox_inches='tight')

# %%
        
rate_comaprison_bulk_log()