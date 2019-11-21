# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 18:46:26 2019

Figure for the supplemental mateirals of magnetically forbidden rates paper
showing all 9 possible measurements

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

# %%

purple = '#87479b'

# Time data for NV1
# 1.25 hour incr
file4 = '26.3_MHz_splitting_18_bins_error'
folder4 = 'nv1_2019_05_10_28MHz_4'
data4 = tool_belt.get_raw_data('t1_double_quantum', file4, folder4)

file5 = '26.5_MHz_splitting_15_bins_error'
folder5 = 'nv1_2019_05_10_28MHz_5'
data5 = tool_belt.get_raw_data('t1_double_quantum', file5, folder5)

file6 = '26.2_MHz_splitting_15_bins_error'
folder6 = 'nv1_2019_05_10_28MHz_6'
data6 = tool_belt.get_raw_data('t1_double_quantum', file6, folder6)

gamma_list_NV1 = data4['gamma_list'] + data5['gamma_list'] + data6['gamma_list'] 
gamma_ste_list_NV1 = data4['gamma_ste_list'] + data5['gamma_ste_list'] \
                + data6['gamma_ste_list']
gamma_ste_list_NV1 = numpy.array(gamma_ste_list_NV1)*2
    
# Time data for NV2
# 1.0 hour incr  
file29 = '29.5_MHz_splitting_25_bins_error'
folder29 = 'nv2_2019_04_30_29MHz_29'
data29 = tool_belt.get_raw_data('t1_double_quantum', file29, folder29)

file30 = '29.8_MHz_splitting_10_bins_error'
folder30 = 'nv2_2019_04_30_29MHz_30'
data30 = tool_belt.get_raw_data('t1_double_quantum', file30, folder30)


gamma_list_NV2 = data29['gamma_list'] + data30['gamma_list']
gamma_ste_list_NV2 = data29['gamma_ste_list'] + data30['gamma_ste_list']

gamma_ste_list_NV2 = numpy.array(gamma_ste_list_NV2)*2

nv2_rates = [27.621071292343977, 28.00467103154353, 29.469965453079418, 25.10316827931753, 27.384306752689977, 26.950826695652587, 27.57847277904147, 25.813643197609753, 25.545459621923523, 27.73919964386004, 24.93512434869353, 25.341877755575116, 25.03634200036785, 25.81743330146043, 27.664173722079138, 26.182779332824943, 29.769336662684722, 29.004513033512666, 28.668977681142096, 30.34822157435222, 27.712413403059614, 27.39954530909307, 29.030026517648455, 28.11304948821951, 27.691511309720386, 26.623793229361375]
nv2_error = numpy.array([0.48665910284728936, 0.5136437039949685, 0.5229472135927203, 0.43261891664090063, 0.6756646634091575, 0.46467072686726413, 0.47143103043938, 0.423587832606777, 0.4446911497352184, 0.5366201978752652, 0.4412212934696276, 0.43465587142873024, 0.42048005502218033, 0.4462953842115663, 0.4822572573604373, 0.4403568078813716, 0.5657279159827925, 0.543961952271339, 0.49386476574863025, 0.5473851659364642, 0.5002106035777926, 0.5016894491027237, 0.5087576703572062, 0.4840832910081832, 0.4797090724802973, 0.47941746597351753])

splittings = [29.4, 29.2, 28.9,	29.1, 29.3, 29.5, 29.9, 29.8, 29.8, 30.0, 29.7, 29.3, 29.9, 29.8, 29.7, 29.5, 29.6, 29.6, 29.8, 29.6, 29.0, 29.5, 29.7, 29.0, 29.3, 29.2 ]

    
# %% equations
def exp_eq(t, rate, amp):
    return  amp * exp(- rate * t)

def exp_eq_offset(t, rate, amp, offset):
    return  amp * exp(- rate * t) + offset

def plus_plus(t, gamma, omega, ep, em):
    return 1/3 + (1/2*(1-ep)*exp(-(2*gamma+omega)*t) -1/2*(ep-1/3)*exp(-3*omega*t))*(1-ep) \
                + (ep-1/3)*exp(-3*omega*t)*ep
                
def plus_minus(t, gamma, omega, ep, em):
    return 1/3 + (-1/2*(1-ep)*exp(-(2*gamma+omega)*t) -1/2*(ep-1/3)*exp(-3*omega*t))*(1-em) \
                + (ep-1/3)*exp(-3*omega*t)*em
                
def plus_zero(t, gamma, omega, ep, em):
    return 1/3 +  (ep-1/3)*exp(-3*omega*t)

def minus_plus(t, gamma, omega, ep, em):
    return 1/3 + (-1/2*(1-em)*exp(-(2*gamma+omega)*t) -1/2*(em-1/3)*exp(-3*omega*t))*(1-ep) \
                + (em-1/3)*exp(-3*omega*t)*ep

def minus_minus(t, gamma, omega, ep, em):
    return 1/3 + (1/2*(1-em)*exp(-(2*gamma+omega)*t) -1/2*(em-1/3)*exp(-3*omega*t))*(1-em) \
                + (em-1/3)*exp(-3*omega*t)*em

def minus_zero(t, gamma, omega, ep, em):
    return 1/3 +  (em-1/3)*exp(-3*omega*t)

def zero_plus(t, gamma, omega, ep, em):
    return 1/3 - 1/2* (2/3)*exp(-3*omega*t)*(1-ep) + (2/3)*exp(-3*omega*t)*ep

def zero_minus(t, gamma, omega, ep, em):
    return 1/3 - 1/2* (2/3)*exp(-3*omega*t)*(1-em) + (2/3)*exp(-3*omega*t)*em

def zero_zero(t, gamma, omega, ep, em):
    return 1/3 +  (2/3)*exp(-3*omega*t)

fitting_eq_list = [plus_plus, plus_minus, plus_zero, minus_plus, minus_minus, \
                minus_zero, zero_plus, zero_minus, zero_zero]
# %%
    
def omega_comparison(save=False):
    text_font = 40
    
    linspace = numpy.linspace(-0.05, 2.5, 1000)
    folder = 'E:/Shared drives/Kolkowitz Lab Group/nvdata/t1_double_quantum'
    
    low_plus_file = 'nv1_2019_05_10_28MHz_3/2019-08-26-05_57_27-ayrton12-nv1_2019_05_10.txt'
    low_minus_file = 'nv1_2019_05_10_28MHz_3/2019-08-25-22_26_53-ayrton12-nv1_2019_05_10.txt'
    low_zero_file = 'nv1_2019_05_10_28MHz_3/2019-08-26-13_27_51-ayrton12-nv1_2019_05_10.txt'
    
#    high_plus_file = 'nv1_2019_05_10_116MHz/2019-05-18_11-32-57_ayrton12.txt'
#    high_minus_file = 'nv1_2019_05_10_116MHz/2019-05-18_13-03-58_ayrton12.txt'
#    high_zero_file = 'nv1_2019_05_10_116MHz/2019-05-18_19-25-42_ayrton12.txt'
    
    
    high_plus_file = 'nv1_2019_05_10_1017MHz/high_low_comparison/2019-09-01-05_08_17-ayrton12-nv1_2019_05_10.txt'
    high_minus_file = 'nv1_2019_05_10_1017MHz/high_low_comparison/2019-08-31-19_45_43-ayrton12-nv1_2019_05_10.txt'
    high_zero_file = 'nv1_2019_05_10_1017MHz/high_low_comparison/2019-09-01-14_30_44-ayrton12-nv1_2019_05_10.txt'
    

    file_list = [low_plus_file, low_minus_file, low_zero_file,
                 high_plus_file, high_minus_file, high_zero_file]
    
    relaxation_counts = []
    relaxation_ste = []
    time = []
    
    subtr_counts = []
    subtr_counts_ste = []
    omega_params = []
    
    
    for file_ind in range(len(file_list)):
        file = file_list[file_ind]
        with open('{}/{}'.format(folder, file)) as file:
                data = json.load(file)
                
                sig_counts = numpy.array(data['sig_counts'])
                ref_value = numpy.average(data['ref_counts'])
                num_runs = data['num_runs']
                relaxation_time_range = numpy.array(data['relaxation_time_range'])
                num_steps = data['num_steps']
                
                min_relaxation_time, max_relaxation_time = \
                                        relaxation_time_range / 10**6
                time_array = numpy.linspace(min_relaxation_time,
                                        max_relaxation_time, num=num_steps)
                
                avg_sig_counts = numpy.average(sig_counts[::], axis=0)
                ste_sig_counts = numpy.std(sig_counts[::], axis=0, ddof = 1) / numpy.sqrt(num_runs)

                norm_avg_sig = avg_sig_counts / ref_value
                norm_avg_sig_ste = ste_sig_counts / ref_value

                relaxation_counts.append(norm_avg_sig)
                relaxation_ste.append(norm_avg_sig_ste)
                time.append(time_array)
    
    
    for i in [0,3]:
        #Omega Plus
        omega_plus = relaxation_counts[i+2] - relaxation_counts[i]
        omega_plus_ste = numpy.sqrt(relaxation_ste[i+2] **2 + relaxation_ste[i]**2)
        
        subtr_counts.append(omega_plus)
        subtr_counts_ste.append(omega_plus_ste)
        
        init_params = [0.1, 0.4]
        
        omega_plus_params, cov_arr = curve_fit(exp_eq, time[i],
                                             omega_plus, p0 = init_params,
                                             sigma = omega_plus_ste,
                                             absolute_sigma=True)
        
        omega_plus_value = omega_plus_params[0] / 3.0
        omega_plus_ste = numpy.sqrt(cov_arr[0,0]) / 3.0
        
        omega_params.append(omega_plus_params)
        
        print('Omega Plus: {} +/- {} kHz'.format('%.3f'%omega_plus_value,
                      '%.3f'%omega_plus_ste))
        
        #Omega Minus
        omega_minus = relaxation_counts[i+2] - relaxation_counts[i+1]
        omega_minus_ste = numpy.sqrt(relaxation_ste[i+2] **2 + relaxation_ste[i+1]**2)
        
        subtr_counts.append(omega_minus)
        subtr_counts_ste.append(omega_minus_ste)
        
        omega_minus_params, cov_arr = curve_fit(exp_eq, time[i],
                                             omega_minus, p0 = init_params,
                                             sigma = omega_minus_ste,
                                             absolute_sigma=True)
        
        omega_minus_value = omega_minus_params[0] / 3.0
        omega_minus_ste = numpy.sqrt(cov_arr[0,0]) / 3.0
        
        omega_params.append(omega_minus_params)
        
        print('Omega Minus: {} +/- {} kHz'.format('%.3f'%omega_minus_value,
                      '%.3f'%omega_minus_ste))
        
        
    fig, axes = plt.subplots(1,2, figsize=(16, 8))
    
    ax = axes[0]
    ax.errorbar(time[0], subtr_counts[0],
                        yerr = subtr_counts_ste[0],
                        label = r'P$_{0,0}$ - P$_{0,+1}$', markersize = 10, fmt = 'o', color = 'teal')
    ax.plot(linspace,exp_eq(linspace, *omega_params[0]),
                        'r', linestyle='--', linewidth=3, color = 'teal', label = 'fit')
    ax.errorbar(time[1], subtr_counts[1],
                        yerr = subtr_counts_ste[1],
                        label = r'P$_{0,0}$ - P$_{0,-1}$', markersize = 10,fmt = '^', color = 'orange')
    ax.plot(linspace,exp_eq(linspace, *omega_params[1]),
                        'r', linestyle='--', linewidth=3, color = 'orange', label = 'fit')
    ax.set_xlabel(r'Wait time, $\tau$  (ms)', fontsize=text_font)
    ax.set_ylabel(r'$F_{\Omega}$ (arb. units)', fontsize=text_font)
    ax.set_xlim([-0.01, 0.61])
    ax.set_ylim([-0.01, 0.43])
    
    ax.set_xticks([0.0,0.2,0.4,0.6])
    ax.set_yticks([0.0,0.1,0.2,0.3,0.4])
#    ax.set_ylim([0.6, 0.9])
    ax.tick_params(which = 'both', length=8, width=2, colors='k',
                direction='in',grid_alpha=0.7, labelsize = text_font)
    
    
    ax = axes[1]
    ax.errorbar(time[3], subtr_counts[2],
                        yerr = subtr_counts_ste[2],
                        label = r'P$_{0,0}$ - P$_{0,+1}$', markersize = 10,fmt = 'o', color = 'teal')
    ax.plot(linspace,exp_eq(linspace, *omega_params[2]),
                        'r', linestyle='--', linewidth=3, color = 'teal',label = 'fit')
    ax.errorbar(time[3], subtr_counts[3],
                        yerr = subtr_counts_ste[3],
                        label = r'P$_{0,0}$ - P$_{0,-1}$', markersize = 10,fmt = '^', color = 'orange')
    ax.plot(linspace,exp_eq(linspace, *omega_params[3]),
                        'r', linestyle='--', linewidth=3, color = 'orange',label = 'fit')
    ax.set_xlabel(r'Wait time, $\tau$  (ms)', fontsize=text_font)
    ax.set_ylabel(r'$F_{\Omega}$ (arb. units)', fontsize=text_font)
    ax.set_xlim([-0.1, 2.26])
    ax.set_xticks([0.0,0.5,1,1.5,2])
    ax.set_yticks([-0.05,0.0,0.05,0.1,0.15,0.2])
#    ax.set_ylim([-0.01, 0.43])
#    ax.set_ylim([0.6, 0.9])
    ax.tick_params(which = 'both', length=8, width=2, colors='k',
                direction='in',grid_alpha=0.7, labelsize = text_font)
            
    fig.canvas.draw()
    fig.canvas.flush_events()
    fig.tight_layout()

    if save:
        fig.savefig("C:/Users/Aedan/Creative Cloud Files/Paper Illustrations/Magnetically Forbidden Rate/supplemental_materials/omega_comp.pdf", bbox_inches='tight')
    
# %%
    
def one_hour_rates_NV1(save = False):    
   
    time_inc = 1.25 # hr
    
    time_start_list_4 = []
    for i in range(len(data4['gamma_list'] )):
        time = i*time_inc
        time_start_list_4.append(time)
    time_end_list_4 = []
    for i in range(len(data4['gamma_list']) ):
        time = i*time_inc+ time_inc
        time_end_list_4.append(time)
        
    time_start_list_5 = []
    for i in range(len(data5['gamma_list'] )):
        time = i*time_inc + 3 + time_end_list_4[-1]
        time_start_list_5.append(time)
    time_end_list_5 = []
    for i in range(len(data5['gamma_list'] )):
        time = i*time_inc+ time_inc + 3 + time_end_list_4[-1]
        time_end_list_5.append(time)
        
    time_start_list_6 = []
    for i in range(len(data6['gamma_list'] )):
        time = i*time_inc + 1 + time_end_list_5[-1]
        time_start_list_6.append(time)
    time_end_list_6 = []
    for i in range(len(data6['gamma_list'] )):
        time = i*time_inc+ time_inc + 1 + time_end_list_5[-1]
        time_end_list_6.append(time)
        
    time_start_list = time_start_list_4 + time_start_list_5 + time_start_list_6
    time_end_list = time_end_list_4 + time_end_list_5 + time_end_list_6
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    for i in range(len(time_start_list)):
        ax.hlines(gamma_list_NV1[i], time_start_list[i], time_end_list[i], linewidth=5, colors = purple)
        time_space = numpy.linspace(time_start_list[i], time_end_list[i], 1000)
        ax.fill_between(time_space, gamma_list_NV1[i] + gamma_ste_list_NV1[i],
                        gamma_list_NV1[i] - gamma_ste_list_NV1[i],
                        color=purple, alpha=0.2)
    
    ax.tick_params(which = 'both', length=6, width=2, colors='k',
                    grid_alpha=0.7, labelsize = 18)

    ax.tick_params(which = 'major', length=12, width=2)

    ax.grid()

    ax.set_xlabel('Time (hours)', fontsize=18)
    ax.set_ylabel(r'Relaxation Rate, $\gamma$ (kHz)', fontsize=18)
    ax.set_ylim(33,69)
    fig.canvas.draw()
    fig.canvas.flush_events()
    
    if save:
        fig.savefig("C:/Users/Aedan/Creative Cloud Files/Paper Illustrations/Magnetically Forbidden Rate/supplemental_materials/NV1_hour_rates.pdf", bbox_inches='tight')
# %%
def NV1_histogram(save=False):
    text = 62
    bins = 13
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.hist(gamma_list_NV1, bins = bins, color = purple)
    ax.set_xlabel(r'$\gamma$ (kHz)', fontsize=text)
    ax.set_ylabel('Occurances', fontsize=text)
    ax.tick_params(which = 'both', length=10, width=20, colors='k',
                    grid_alpha=1.2, labelsize = text)

    ax.tick_params(which = 'major', length=12, width=2)
    if save:
        fig.savefig("C:/Users/Aedan/Creative Cloud Files/Paper Illustrations/Magnetically Forbidden Rate/supplemental_materials/NV1_hist.pdf", bbox_inches='tight')


# %%

def one_hour_rates_NV2(save=False):
    time_inc = 1.0 # hr
    
    start_time_list_2 = []
    end_time_list_2 = []
    
    for i in range(len(gamma_list_NV2)):
        time = i*time_inc + 640
        start_time_list_2.append(time)
        
        time = i*time_inc + time_inc + 640
        end_time_list_2.append(time)
        
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    for i in range(len(start_time_list_2)):
        ax.hlines(gamma_list_NV2[i], start_time_list_2[i], end_time_list_2[i], linewidth=5, colors = purple)
        time_space = numpy.linspace(start_time_list_2[i], end_time_list_2[i], 10)
        ax.fill_between(time_space, gamma_list_NV2[i] + gamma_ste_list_NV2[i],
                        gamma_list_NV2[i] - gamma_ste_list_NV2[i],
                        color=purple, alpha=0.2)
        
    ax.tick_params(which = 'both', length=6, width=2, colors='k',
                    grid_alpha=0.7, labelsize = 18)

    ax.tick_params(which = 'major', length=12, width=2)

    ax.grid()
    ax.set_ylim([22, 36])

    plt.xlabel('Time (hours)', fontsize=18)
    plt.ylabel(r'Relaxation Rate, $\gamma$ (kHz)', fontsize=18)
    fig.canvas.draw()
    fig.canvas.flush_events()

    if save:
        fig.savefig("C:/Users/Aedan/Creative Cloud Files/Paper Illustrations/Magnetically Forbidden Rate/supplemental_materials/NV2_hour_rates.pdf", bbox_inches='tight')

# %%
def NV2_histogram(save=False):
    text = 62
    bins = 8
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.hist(gamma_list_NV2, bins = bins, color = purple)
    ax.set_xlabel(r'$\gamma$ (kHz)', fontsize=text)
    ax.set_ylabel('Occurances', fontsize=text)
    ax.tick_params(which = 'both', length=10, width=20, colors='k',
                    grid_alpha=1.2, labelsize = text)

    ax.tick_params(which = 'major', length=12, width=2)
    
    if save:
        fig.savefig("C:/Users/Aedan/Creative Cloud Files/Paper Illustrations/Magnetically Forbidden Rate/supplemental_materials/NV2_hist.pdf", bbox_inches='tight')
    
#%%
    
def rate_comaprison_NV2_subtraction(save=False):
    text_font = 18
    blue = '#2e3192'
    omega = 0.32 / 1000 
    omega_unc = 0.06 / 1000
    offset = 0
    
    file = '28.9_MHz_splitting_rate_analysis'
    folder = 'nv2_2019_04_30_29MHz_5'
    data_f = tool_belt.get_raw_data('t1_double_quantum.py', '{}\\{}'.format(folder, file))
    
    file = '29.1_MHz_splitting_rate_analysis'
    folder = 'nv2_2019_04_30_29MHz_6'
    data_s = tool_belt.get_raw_data('t1_double_quantum.py', '{}\\{}'.format(folder, file))
    
    
    counts_f = numpy.array(data_f['plus_relaxation_counts'])
    error_f = numpy.array(data_f['plus_relaxation_ste'])
    counts_s = numpy.array(data_s['plus_relaxation_counts'])
    error_s = numpy.array(data_s['plus_relaxation_ste'])
    
    time = numpy.array(data_f['plus_plus_time']) * 1000
    time_linspace = numpy.linspace(time[0], time[-1], 1000)

    
    gamma_f = data_f['gamma'] / 1000
    gamma_unc_f = data_f['gamma_ste'] / 1000
    amp_f = data_f['gamma_opti_params'][1]
    gamma_s = data_s['gamma'] / 1000
    gamma_unc_s = data_s['gamma_ste'] / 1000
    amp_s = data_s['gamma_opti_params'][1]
    
    opti_params_f = [(2*gamma_f+omega), amp_f, offset]
    opti_params_s = [(2*gamma_s+omega), amp_s, offset]
    
    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(10,8))

    ax.set_xlim([-1,110])
    ax.errorbar(time, counts_s - counts_f, yerr = numpy.sqrt(error_f**2 + error_s**2), label = 'gamma = {}({}) kHz'.format(gamma_f, gamma_unc_f), 
                fmt = 'o', color = blue)
    ax.hlines(0, -10, 500)
    
    
    ax.set_xlabel(r'Relaxation time ($\mu$s)', fontsize=text_font)
    ax.set_ylabel('Difference between two data sets', fontsize=text_font)
    ax.tick_params(which = 'both', length=8, width=2, colors='k',
                grid_alpha=0.7, labelsize = text_font)
    
    fig.canvas.draw()
    fig.canvas.flush_events()
    
    if save:
        fig.savefig("C:/Users/Aedan/Creative Cloud Files/Paper Illustrations/Magnetically Forbidden Rate/supplemental_materials/NV2_rate_compare_subtraction.pdf", bbox_inches='tight')

#%%
    
def rate_comaprison_NV2_lin(save=False):
    text_font = 30
    blue = '#2e3192'
    red = '#ed1c24'
    omega = 0.32 / 1000 
    omega_unc = 0.06 / 1000
    offset = 0
    
    file = '28.9_MHz_splitting_rate_analysis'
    folder = 'nv2_2019_04_30_29MHz_5'
    data_f = tool_belt.get_raw_data('t1_double_quantum.py', '{}\\{}'.format(folder, file))
    
    file = '29.1_MHz_splitting_rate_analysis'
    folder = 'nv2_2019_04_30_29MHz_6'
    data_s = tool_belt.get_raw_data('t1_double_quantum.py', '{}\\{}'.format(folder, file))
    
    
    counts_f = numpy.array(data_f['plus_relaxation_counts'])
    error_f = numpy.array(data_f['plus_relaxation_ste'])
    counts_s = numpy.array(data_s['plus_relaxation_counts'])
    error_s = numpy.array(data_s['plus_relaxation_ste'])
    
    time = numpy.array(data_f['plus_plus_time']) * 1000
    time_linspace = numpy.linspace(time[0], time[-1], 1000)

    
    gamma_f = data_f['gamma'] / 1000
    gamma_unc_f = data_f['gamma_ste'] / 1000
    amp_f = data_f['gamma_opti_params'][1]
    gamma_s = data_s['gamma'] / 1000
    gamma_unc_s = data_s['gamma_ste'] / 1000
    amp_s = data_s['gamma_opti_params'][1]
    
    opti_params_f = [(2*gamma_f+omega), amp_f, offset]
    opti_params_s = [(2*gamma_s+omega), amp_s, offset]
    
    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(10,8))

    ax.set_xlim([-1,110])
    ax.errorbar(time, counts_f, yerr = error_f, label = 'gamma = {}({}) kHz'.format(gamma_f, gamma_unc_f), 
                fmt = 'o', color = blue, markersize = 10)
    yfit = exp_eq_offset(time_linspace, *opti_params_f)
    ax.plot(time_linspace, yfit, '-', color=blue)
    
    opti_params_f[0] = (2*(gamma_f + gamma_unc_f) + omega + omega_unc)
    yupper = exp_eq_offset(time_linspace, *opti_params_f)
    opti_params_f[0] = (2*(gamma_f - gamma_unc_f) + omega - omega_unc)
    ylower = exp_eq_offset(time_linspace, *opti_params_f)
    
    ax.fill_between(time_linspace, yupper,  ylower,
                     color='blue', alpha=0.4)
    
    
    ax.errorbar(time, counts_s, yerr = error_s, label = 'gamma = {}({}) kHz'.format(gamma_s, gamma_unc_s), 
                fmt = '^', color = red, markersize = 10)
    
    yfit = exp_eq_offset(time_linspace, *opti_params_s)
    ax.plot(time_linspace, yfit, '-', color=red)
    
    opti_params_s[0] = (2*(gamma_s + gamma_unc_s) + omega + omega_unc)
    yupper = exp_eq_offset(time_linspace, *opti_params_s)
    opti_params_s[0] = (2*(gamma_s - gamma_unc_s) + omega - omega_unc)
    ylower = exp_eq_offset(time_linspace, *opti_params_s)
    
    ax.fill_between(time_linspace, yupper,  ylower,
                     color='red', alpha=0.4)
    
    ax.set_xlabel(r'Relaxation time ($\mu$s)', fontsize=text_font)
    ax.set_ylabel('Relaxation signal', fontsize=text_font)
    ax.tick_params(which = 'both', length=8, width=2, colors='k',
                direction = 'in', grid_alpha=0.7, labelsize = text_font)
    
    fig.canvas.draw()
    fig.canvas.flush_events()
    
    if save:
        fig.savefig("C:/Users/Aedan/Creative Cloud Files/Paper Illustrations/Magnetically Forbidden Rate/supplemental_materials/NV2_rate_compare.pdf", bbox_inches='tight')

#%%
    
def rate_comaprison_NV2_log(save=False):
    text_font = 60
    blue = '#2e3192'
    red = '#ed1c24'
    omega = 0.32 / 1000 
    omega_unc = 0.06 / 1000
    offset = 0
    
    file = '28.9_MHz_splitting_rate_analysis'
    folder = 'nv2_2019_04_30_29MHz_5'
    data_f = tool_belt.get_raw_data('t1_double_quantum.py', '{}\\{}'.format(folder, file))
    
    file = '29.1_MHz_splitting_rate_analysis'
    folder = 'nv2_2019_04_30_29MHz_6'
    data_s = tool_belt.get_raw_data('t1_double_quantum.py', '{}\\{}'.format(folder, file))
    
    
    counts_f = data_f['plus_relaxation_counts']
    error_f = numpy.array(data_f['plus_relaxation_ste'])
    counts_s = data_s['plus_relaxation_counts']
    error_s = numpy.array(data_s['plus_relaxation_ste'])
    
    time = numpy.array(data_f['plus_plus_time']) * 1000
    time_linspace = numpy.linspace(time[0], time[-1], 1000)
    
    gamma_f = data_f['gamma'] / 1000
    gamma_unc_f = data_f['gamma_ste'] / 1000
    amp_f = data_f['gamma_opti_params'][1]
    gamma_s = data_s['gamma'] / 1000
    gamma_unc_s = data_s['gamma_ste'] / 1000
    amp_s = data_s['gamma_opti_params'][1]
    
    opti_params_f = [(2*gamma_f+omega), amp_f, offset]
    opti_params_s = [(2*gamma_s+omega), amp_s, offset]
    
    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(10,8))
    ax.set_yscale("log", nonposy='clip')
    ax.set_xlim([-1,48.5])
    ax.set_ylim([7*10**-3,4*10**-1])
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y,pos: ('{{:.{:1d}f}}'.format(int(numpy.maximum(-numpy.log10(y),0)))).format(y)))
    
    ax.errorbar(time, counts_f, yerr = error_f, label = 'gamma = {}({}) kHz'.format(gamma_f, gamma_unc_f), 
                fmt = 'o', markersize = 16, color = blue)
#    ax.plot(time, counts_f, 'o', color =blue)
    yfit = exp_eq_offset(time_linspace, *opti_params_f)
    ax.plot(time_linspace, yfit, '-', color=blue)
    
    opti_params_f[0] = (2*(gamma_f + gamma_unc_f) + omega + omega_unc)
    yupper = exp_eq_offset(time_linspace, *opti_params_f)
    opti_params_f[0] = (2*(gamma_f - gamma_unc_f) + omega - omega_unc)
    ylower = exp_eq_offset(time_linspace, *opti_params_f)
    
    ax.fill_between(time_linspace, yupper,  ylower,
                     color='blue', alpha=0.4)
    
    
    ax.errorbar(time, counts_s, yerr = error_s, label = 'gamma = {}({}) kHz'.format(gamma_s, gamma_unc_s), 
                fmt = '^', markersize = 16, color = red)
#    ax.plot(time, counts_s, '^', color =red)
    
    yfit = exp_eq_offset(time_linspace, *opti_params_s)
    ax.plot(time_linspace, yfit, '-', color=red)
    
    opti_params_s[0] = (2*(gamma_s + gamma_unc_s) + omega + omega_unc)
    yupper = exp_eq_offset(time_linspace, *opti_params_s)
    opti_params_s[0] = (2*(gamma_s - gamma_unc_s) + omega - omega_unc)
    ylower = exp_eq_offset(time_linspace, *opti_params_s)
    
    ax.fill_between(time_linspace, yupper,  ylower,
                     color='red', alpha=0.4)
    
    ax.set_xlabel(r'Relaxation time ($\mu$s)', fontsize=text_font)
    ax.set_ylabel('Relaxation signal', fontsize=text_font)
    ax.tick_params(which = 'both', length=8, width=3, colors='k',
                direction = 'in', grid_alpha=0.7, labelsize = text_font)
    ax.tick_params(which = 'major', length=20, width=5)
    ax.grid(axis='y')
#    ax.set_xlim()
#    ax.legend()
#    ax.set_title('Compare NV1 measurements')
    
    fig.canvas.draw()
    fig.canvas.flush_events()
    
    if save:
        fig.savefig("C:/Users/Aedan/Creative Cloud Files/Paper Illustrations/Magnetically Forbidden Rate/supplemental_materials/NV2_rate_compare_log.pdf", bbox_inches='tight')

#%%
    
def rate_comaprison_NV1_lin(save=False):
    text_font = 40
    blue = '#2e3192'
    red = '#ed1c24'
    omega = 1.17 / 1000 
    omega_unc = 0.18 / 1000
    offset = -0.006
    
    file = '26.5_MHz_splitting_5_bins_error'
    folder = 'nv1_2019_05_10_28MHz_5'
    f_number = 1
    s_number = 2
    data = tool_belt.get_raw_data('t1_double_quantum.py', '{}\\{}'.format(folder, file))
    
    offset = data['gamma_offset']
    
    
    
    counts_f = numpy.array(data['gamma_counts_list'][f_number])
    error_f = numpy.array(data['gamma_counts_ste_list'][f_number])
    counts_s = numpy.array(data['gamma_counts_list'][s_number])
    error_s = numpy.array(data['gamma_counts_ste_list'][s_number])
    
    time = numpy.array(data['taus']) * 1000
    time_linspace = numpy.linspace(time[0], time[-1]+5, 1000)

    
    gamma_f = data['gamma_list'][f_number] / 1000
    gamma_unc_f = data['gamma_ste_list'][f_number] / 1000
    amp_f = 0.302
    gamma_s = data['gamma_list'][s_number] / 1000
    gamma_unc_s = data['gamma_ste_list'][s_number] / 1000
    amp_s = 0.302
    
    opti_params_f = [(2*gamma_f+omega), amp_f, offset]
    opti_params_s = [(2*gamma_s+omega), amp_s, offset]
    
    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(10,8))

    ax.set_xlim([-1,76])
    ax.errorbar(time, counts_f, yerr = error_f, label = 'gamma = {}({}) kHz'.format(gamma_f, gamma_unc_f), 
                fmt = 'o', color = blue, markersize = 10)
    yfit = exp_eq_offset(time_linspace, *opti_params_f)
    ax.plot(time_linspace, yfit, '-', color=blue)
    
    opti_params_f[0] = (2*(gamma_f + gamma_unc_f) + omega + omega_unc)
    yupper = exp_eq_offset(time_linspace, *opti_params_f)
    opti_params_f[0] = (2*(gamma_f - gamma_unc_f) + omega - omega_unc)
    ylower = exp_eq_offset(time_linspace, *opti_params_f)
    
    ax.fill_between(time_linspace, yupper,  ylower,
                     color='blue', alpha=0.4)
    
    
    ax.errorbar(time, counts_s, yerr = error_s, label = 'gamma = {}({}) kHz'.format(gamma_s, gamma_unc_s), 
                fmt = '^', color = red, markersize = 10)
    
    yfit = exp_eq_offset(time_linspace, *opti_params_s)
    ax.plot(time_linspace, yfit, '-', color=red)
    
    opti_params_s[0] = (2*(gamma_s + gamma_unc_s) + omega + omega_unc)
    yupper = exp_eq_offset(time_linspace, *opti_params_s)
    opti_params_s[0] = (2*(gamma_s - gamma_unc_s) + omega - omega_unc)
    ylower = exp_eq_offset(time_linspace, *opti_params_s)
    
    ax.fill_between(time_linspace, yupper,  ylower,
                     color='red', alpha=0.4)
    
    ax.set_xlabel(r'Wait time, $\tau$ ($\mu$s)', fontsize=text_font)
    ax.set_ylabel(r'$F_\gamma$ (arb. units)', fontsize=text_font)
    ax.tick_params(which = 'both', length=8, width=2, colors='k',
                direction='in',grid_alpha=0.7, labelsize = text_font)
    
    fig.canvas.draw()
    fig.canvas.flush_events()
    
    if save:
        fig.savefig("C:/Users/Aedan/Creative Cloud Files/Paper Illustrations/Magnetically Forbidden Rate/supplemental_materials/NV1_rate_compare_lin.pdf", bbox_inches='tight')

#%%
    
def rate_comaprison_NV1_log(save=False):
    text_font = 65
    blue = '#2e3192'
    red = '#ed1c24'
    omega = 1.17 / 1000 
    omega_unc = 0.18 / 1000
    offset = -0.006
    
    file = '26.5_MHz_splitting_5_bins_error'
    folder = 'nv1_2019_05_10_28MHz_5'
    f_number = 1
    s_number = 2
    data = tool_belt.get_raw_data('t1_double_quantum.py', '{}\\{}'.format(folder, file))
    
    offset = data['gamma_offset']
    
    
    
    counts_f = numpy.array(data['gamma_counts_list'][f_number])
    error_f = numpy.array(data['gamma_counts_ste_list'][f_number])
    counts_s = numpy.array(data['gamma_counts_list'][s_number])
    error_s = numpy.array(data['gamma_counts_ste_list'][s_number])
    
    time = numpy.array(data['taus']) * 1000
    time_linspace = numpy.linspace(time[0], time[-1]+5, 1000)

    
    gamma_f = data['gamma_list'][f_number] / 1000
    gamma_unc_f = data['gamma_ste_list'][f_number] / 1000
    amp_f = 0.302
    gamma_s = data['gamma_list'][s_number] / 1000
    gamma_unc_s = data['gamma_ste_list'][s_number] / 1000
    amp_s = 0.302
    
    opti_params_f = [(2*gamma_f+omega), amp_f, offset]
    opti_params_s = [(2*gamma_s+omega), amp_s, offset]
    
    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(10,8))
    ax.set_yscale("log", nonposy='clip')
    ax.set_xlim([-0.5,21.5])
    ax.set_ylim([2*10**-2,3.5*10**-1])
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y,pos: ('{{:.{:1d}f}}'.format(int(numpy.maximum(-numpy.log10(y),0)))).format(y)))
    
    ax.errorbar(time, counts_f, yerr = error_f, label = 'gamma = {}({}) kHz'.format(gamma_f, gamma_unc_f), 
                fmt = 'o', markersize = 16, color = blue)
#    ax.plot(time, counts_f, 'o', color =blue)
    yfit = exp_eq_offset(time_linspace, *opti_params_f)
    ax.plot(time_linspace, yfit, '-', color=blue)
    
    opti_params_f[0] = (2*(gamma_f + gamma_unc_f) + omega + omega_unc)
    yupper = exp_eq_offset(time_linspace, *opti_params_f)
    opti_params_f[0] = (2*(gamma_f - gamma_unc_f) + omega - omega_unc)
    ylower = exp_eq_offset(time_linspace, *opti_params_f)
    
    ax.fill_between(time_linspace, yupper,  ylower,
                     color='blue', alpha=0.4)
    
    
    ax.errorbar(time, counts_s, yerr = error_s, label = 'gamma = {}({}) kHz'.format(gamma_s, gamma_unc_s), 
                fmt = '^', markersize = 16, color = red)
#    ax.plot(time, counts_s, '^', color =red)
    
    yfit = exp_eq_offset(time_linspace, *opti_params_s)
    ax.plot(time_linspace, yfit, '-', color=red)
    
    opti_params_s[0] = (2*(gamma_s + gamma_unc_s) + omega + omega_unc)
    yupper = exp_eq_offset(time_linspace, *opti_params_s)
    opti_params_s[0] = (2*(gamma_s - gamma_unc_s) + omega - omega_unc)
    ylower = exp_eq_offset(time_linspace, *opti_params_s)
    
    ax.fill_between(time_linspace, yupper,  ylower,
                     color='red', alpha=0.4)
    
    ax.set_xlabel(r'Wait time, $\tau$ ($\mu$s)', fontsize=text_font)
    ax.set_ylabel(r'$F_\gamma$ (arb. units)', fontsize=text_font)
    ax.tick_params(which = 'both', length=8, width=3, colors='k',
                direction='in',grid_alpha=0.7, labelsize = text_font)
    ax.tick_params(which = 'major', length=20, width=5)
    ax.grid(axis='y')
    ax.set_xticks([0,10,20])
#    ax.set_xlim()
#    ax.legend()
#    ax.set_title('Compare NV1 measurements')
    
    fig.canvas.draw()
    fig.canvas.flush_events()
    
    if save:
        fig.savefig("C:/Users/Aedan/Creative Cloud Files/Paper Illustrations/Magnetically Forbidden Rate/supplemental_materials/NV1_rate_compare_log.pdf", bbox_inches='tight')

# %%
    
def silicon_sample(save=False):
    
    text_font = 16
    title_font = 20
    folder = 'E:/Shared drives/Kolkowitz Lab Group/nvdata/t1_double_quantum/nv14_2019_10_17_15MHz'
    
    file = '15.1_MHz_splitting_rate_analysis.txt'
    
    with open('{}/{}'.format(folder, file)) as file:
            data = json.load(file)
            
            zero_relaxation_counts = data['zero_relaxation_counts']
            zero_relaxation_ste = numpy.array(data['zero_relaxation_ste'])
            zero_zero_time = data['zero_zero_time']
            
            plus_relaxation_counts = data['plus_relaxation_counts']
            plus_relaxation_ste = data['plus_relaxation_ste']
            plus_plus_time = data['plus_plus_time']

            omega_opti_params = data['omega_opti_params']
            gamma_opti_params = data['gamma_opti_params']
            manual_offset_gamma = data['manual_offset_gamma']
            
#    del plus_relaxation_counts[0], plus_relaxation_counts[50]
#    del plus_relaxation_ste[0], plus_relaxation_ste[50]
#    del plus_plus_time[0], plus_plus_time[50]
    fig, axes_pack = plt.subplots(2,1, figsize=(8.5, 7.5))
    
    ax = axes_pack[0]
    
    ax.errorbar(zero_zero_time, zero_relaxation_counts,
                        yerr = zero_relaxation_ste,
                        label = 'data',  fmt = 'o', color = 'blue')
    zero_time_linspace = numpy.linspace(0, zero_zero_time[-1], num=1000)
    ax.plot(zero_time_linspace,
                exp_eq(zero_time_linspace, *omega_opti_params),
                'r', label = 'fit')
    ax.set_xlabel(r'Wait time, $\tau$  (ms)', fontsize=text_font)
    ax.set_ylabel(r'$F_{\Omega}$ (arb. units)', fontsize=text_font)
#    ax.set_title(r'$P_{0,0} - P_{0,1}$', fontsize=title_font)
#    ax.legend(fontsize=20)
    ax.tick_params(which = 'both', length=8, width=2, colors='k',
                direction='in',grid_alpha=0.7, labelsize = text_font)
    
    ax = axes_pack[1]
    
    ax.errorbar(numpy.array(plus_plus_time)*1000, plus_relaxation_counts,
                        yerr = plus_relaxation_ste,
                        label = 'data', fmt = 'o', color = 'blue')
    plus_time_linspace = numpy.linspace(0, plus_plus_time[-1], num=1000)*1000
    gamma_rate = gamma_opti_params[0]
    gamma_opti_params[0] = gamma_rate/1000
    gamma_opti_params_offset = gamma_opti_params + [manual_offset_gamma]
    ax.plot(plus_time_linspace,
                exp_eq_offset(plus_time_linspace, *gamma_opti_params_offset),
                'r', label = 'fit')
    ax.set_xlabel(r'Wait time, $\tau$  ($\mu$s)', fontsize=text_font)
    ax.set_ylabel(r'$F_{\gamma}$ (arb. units)', fontsize=text_font)
#    ax.set_title(r'$P_{1,1} - P_{1,-1}$', fontsize=title_font)
#    ax.legend(fontsize=20)

    ax.tick_params(which = 'both', length=8, width=2, colors='k',
                direction='in',grid_alpha=0.7, labelsize = text_font)
            
    fig.canvas.draw()
    fig.canvas.flush_events()
    fig.tight_layout()
    
    if save:
        fig.savefig("C:/Users/Aedan/Creative Cloud Files/Paper Illustrations/Magnetically Forbidden Rate/supplemental_materials/silicon_sample.pdf", bbox_inches='tight')
    
    
# %%
    
def subtraction_plot(save=False):
    
    text_font = 16
    title_font = 20
    folder = 'E:/Shared drives/Kolkowitz Lab Group/nvdata/t1_double_quantum/nv1_2019_05_10_20MHz'
    
    file = '19.8_MHz_splitting_rate_analysis.txt'
    
    with open('{}/{}'.format(folder, file)) as file:
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
            
    fig, axes_pack = plt.subplots(2,1, figsize=(8.5, 7.5))
    
    ax = axes_pack[0]
    
    ax.errorbar(zero_zero_time, zero_relaxation_counts,
                        yerr = zero_relaxation_ste,
                        label = 'data',  fmt = 'o', color = 'blue')
    zero_time_linspace = numpy.linspace(0, zero_zero_time[-1], num=1000)
    ax.plot(zero_time_linspace,
                exp_eq(zero_time_linspace, *omega_opti_params),
                'r', label = 'fit')
    ax.set_xlabel('Wait time, $\tau$  (ms)', fontsize=text_font)
    ax.set_ylabel(r'$F_{\Omega}$ (arb. units)', fontsize=text_font)
#    ax.set_title(r'$P_{0,0} - P_{0,1}$', fontsize=title_font)
#    ax.legend(fontsize=20)
    ax.tick_params(which = 'both', length=8, width=2, colors='k',
                direction='in',grid_alpha=0.7, labelsize = text_font)
    
    ax = axes_pack[1]
    
    ax.errorbar(numpy.array(plus_plus_time)*1000, plus_relaxation_counts,
                        yerr = plus_relaxation_ste,
                        label = 'data', fmt = 'o', color = 'blue')
    plus_time_linspace = numpy.linspace(0, plus_plus_time[-1], num=1000)*1000
    gamma_rate = gamma_opti_params[0]
    gamma_opti_params[0] = gamma_rate/1000
    gamma_opti_params_offset = gamma_opti_params + [manual_offset_gamma]
    ax.plot(plus_time_linspace,
                exp_eq_offset(plus_time_linspace, *gamma_opti_params_offset),
                'r', label = 'fit')
    ax.set_xlabel(r'Wait time, $\tau$  ($\mu$s)', fontsize=text_font)
    ax.set_ylabel(r'$F_{\gamma}$ (arb. units)', fontsize=text_font)
#    ax.set_title(r'$P_{1,1} - P_{1,-1}$', fontsize=title_font)
#    ax.legend(fontsize=20)

    ax.tick_params(which = 'both', length=8, width=2, colors='k',
                direction='in',grid_alpha=0.7, labelsize = text_font)
            
    fig.canvas.draw()
    fig.canvas.flush_events()
    fig.tight_layout()
    
    if save:
        fig.savefig("C:/Users/Aedan/Creative Cloud Files/Paper Illustrations/Magnetically Forbidden Rate/supplemental_materials/subtraction.pdf", bbox_inches='tight')
    
# %%
def all_9_meas(save=False):
    folder = 'E:/Shared drives/Kolkowitz Lab Group/nvdata/t1_double_quantum/nv1_2019_05_10_28MHz_3'
    
    file_list = [
    
            '2019-08-24-01_07_22-ayrton12-nv1_2019_05_10.txt', # 1,1
            '2019-08-24-06_51_38-ayrton12-nv1_2019_05_10.txt', # 1,1
            '2019-08-23-17_36_12-ayrton12-nv1_2019_05_10.txt', #1,-1 short
            '2019-08-23-23_20_27-ayrton12-nv1_2019_05_10.txt', # 1,-1 long
            '2019-08-24-08_38_20-ayrton12-nv1_2019_05_10.txt', # 1,0
            '2019-08-24-14_22_27-ayrton12-nv1_2019_05_10.txt', # 1,0
            '2019-08-25-00_29_24-ayrton12-nv1_2019_05_10.txt', # -1,1
            '2019-08-25-06_13_26-ayrton12-nv1_2019_05_10.txt', #-1,1
            '2019-08-24-16_58_56-ayrton12-nv1_2019_05_10.txt', #-1,-1
            '2019-08-24-22_43_02-ayrton12-nv1_2019_05_10.txt', #-1,-1
            '2019-08-25-07_59_35-ayrton12-nv1_2019_05_10.txt', #-1,0
            '2019-08-25-13_43_30-ayrton12-nv1_2019_05_10.txt', #-1,0
            '2019-08-26-00_13_23-ayrton12-nv1_2019_05_10.txt',#0,1
            '2019-08-26-05_57_27-ayrton12-nv1_2019_05_10.txt', #0,1
            '2019-08-25-16_42_47-ayrton12-nv1_2019_05_10.txt', #0,-1
            '2019-08-25-22_26_53-ayrton12-nv1_2019_05_10.txt', #0,-1
            '2019-08-26-07_43_57-ayrton12-nv1_2019_05_10.txt', #0,0
            '2019-08-26-13_27_51-ayrton12-nv1_2019_05_10.txt' # 0,0
            ]
    titles = [r'P$_{+1,+1}$ ($\tau$)',
              r'P$_{+1,-1}$ ($\tau$)',
              r'P$_{+1,0}$ ($\tau$)',
              r'P$_{-1,+1}$ ($\tau$)',
              r'P$_{-1,-1}$ ($\tau$)',
              r'P$_{-1,0}$ ($\tau$)',
              r'P$_{0,+1}$ ($\tau$)',
              r'P$_{0,-1}$ ($\tau$)',
              r'P$_{0,0}$ ($\tau$)'
              ]
    fig , axes = plt.subplots(3, 3, figsize=(16,16))
    r_ind = 0
    c_ind = 0
    
    for i in range(int(len(file_list)/2)):
        if i < 3:
            min_fluor = 0.622735
        elif i > 2 or i < 5:
            min_fluor = 0.609
        else:
            min_fluor = 0.607
        with open('{}/{}'.format(folder, file_list[i*2])) as file_shrt:
            data_shrt = json.load(file_shrt)
            relaxation_time_range = data_shrt['relaxation_time_range']
            num_steps = data_shrt['num_steps']
            
            taus_shrt = numpy.linspace(relaxation_time_range[0],
                                  relaxation_time_range[1], num_steps) / 10**6
            norm_avg_sig_shrt = (numpy.array(data_shrt['norm_avg_sig']) - min_fluor) / (1-min_fluor) #0.622735) / 0.376254
        
        with open('{}/{}'.format(folder, file_list[i*2+1])) as file_long:
            data_long = json.load(file_long)
            
            relaxation_time_range = data_long['relaxation_time_range']
            num_steps = data_long['num_steps']
            
            taus_long = numpy.linspace(relaxation_time_range[0],
                                  relaxation_time_range[1], num_steps) / 10**6
            norm_avg_sig_long = (numpy.array(data_long['norm_avg_sig']) - min_fluor) / (1-min_fluor)
            
        taus = numpy.concatenate((taus_shrt, taus_long))
        norm_sig = numpy.concatenate((norm_avg_sig_shrt, norm_avg_sig_long))
    
        ax = axes[r_ind,c_ind]
        
        gamma = 56.4
        omega = 1.0
        ep = 0.069
        em = 0.013
        lintaus = numpy.linspace(0,0.6,1000)

        ax.plot(lintaus, fitting_eq_list[i](lintaus, gamma, omega, ep, em), 
                'r',  lw = 3)
        
        
        ax.plot(taus, norm_sig, 'bo')
        ax.set_ylim([-0.1,1.1])
        ax.set_xlabel(r'Wait time, $\tau$ (ms)', fontsize=30)
        ax.set_ylabel('Norm. NV Fluor.', fontsize=30)
        ax.set_title(titles[i], fontsize=30)
        ax.tick_params(which = 'both', length=10, width=3, colors='k',
                        direction = 'in', grid_alpha=0.7, labelsize = 30)
        ax.set_xticks([0.0,0.2,0.4,0.6])
        ax.set_yticks([0.0,0.2,0.4,0.6,0.8,1.0])
        if c_ind == 0:
            c_ind = 1
        elif c_ind == 1:
            c_ind = 2
        elif c_ind == 2:
            r_ind = r_ind +  1
            c_ind = 0    
            
    fig.canvas.draw()
    fig.canvas.flush_events()
    fig.tight_layout()
    
    if save:
        fig.savefig("C:/Users/Aedan/Creative Cloud Files/Paper Illustrations/Magnetically Forbidden Rate/supplemental_materials/all_9_measure_figure.pdf", bbox_inches='tight')

# %%
        
def correlation(save=False):
    fig, ax = plt.subplots(1,1,figsize=(8,8))
    ax.scatter(splittings, nv2_rates, color = purple,  s = 200)
    ax.set_ylabel(r'Relaxation rate, $\gamma$ (kHz)', fontsize=20)
    ax.set_xlabel(r'Splitting, $\Delta_\pm$ (MHz)', fontsize=20)
    ax.tick_params(which = 'both', length=10, width=3, colors='k',
                    grid_alpha=0.7, labelsize = 20)
        
    fig.canvas.draw()
    fig.canvas.flush_events()
    fig.tight_layout()
    
    if save:
        fig.savefig("C:/Users/Aedan/Creative Cloud Files/Paper Illustrations/Magnetically Forbidden Rate/supplemental_materials/NV2_correlation.pdf", bbox_inches='tight')

# %%
if __name__ == '__main__':
        
#    all_9_meas(True)
#    correlation(True)
#    silicon_sample(True)
#    subtraction_plot(True) 
    omega_comparison(True)
#    one_hour_rates_NV1(save=True)
#    NV1_histogram(save=True)
#    one_hour_rates_NV2(save=True)
#    NV2_histogram(save=True)
#    rate_comaprison_similar(save=True)
#    rate_comaprison_different(save=True)
#    rate_comaprison_NV2_subtraction(save=False)
    
#    rate_comaprison_NV2_lin(save=True)
#    rate_comaprison_NV2_log(save=True)
    
#    rate_comaprison_NV1_lin(True)
#    rate_comaprison_NV1_log(True)