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
    
# %% equations
def exp_eq(t, rate, amp):
    return  amp * exp(- rate * t)

def exp_eq_offset(t, rate, amp, offset):
    return  amp * exp(- rate * t) + offset

# %%
    
def omega_comparison(save=False):
    text_font = 24
    title_font = 20
    
    linspace = numpy.linspace(-0.05, 1, 1000)
    folder = 'E:/Shared drives/Kolkowitz Lab Group/nvdata/t1_double_quantum'
    
    low_plus_file = 'nv1_2019_05_10_28MHz_3/2019-08-26-05_57_27-ayrton12-nv1_2019_05_10.txt'
    low_minus_file = 'nv1_2019_05_10_28MHz_3/2019-08-25-22_26_53-ayrton12-nv1_2019_05_10.txt'
    low_zero_file = 'nv1_2019_05_10_28MHz_3/2019-08-26-13_27_51-ayrton12-nv1_2019_05_10.txt'
    
    high_plus_file = 'nv1_2019_05_10_116MHz/2019-05-18_11-32-57_ayrton12.txt'
    high_minus_file = 'nv1_2019_05_10_116MHz/2019-05-18_13-03-58_ayrton12.txt'
    high_zero_file = 'nv1_2019_05_10_116MHz/2019-05-18_19-25-42_ayrton12.txt'
    

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
    ax.set_xlabel('Relaxation time (ms)', fontsize=text_font)
    ax.set_ylabel('Relaxation signal', fontsize=text_font)
#    ax.set_title(r'NV1 at 10.3 G', fontsize=title_font)
#    ax.legend(fontsize=20)
    ax.set_xlim([-0.01, 0.61])
    ax.set_ylim([-0.01, 0.43])
    ax.tick_params(which = 'both', length=8, width=2, colors='k',
                grid_alpha=0.7, labelsize = text_font)
    
    
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
    ax.set_xlabel('Relaxation time (ms)', fontsize=text_font)
    ax.set_ylabel('Relaxation signal', fontsize=text_font)
#    ax.set_title(r'NV1 at 41.4 G', fontsize=title_font)
#    ax.legend(fontsize=20)
    ax.set_xlim([-0.01, 0.81])
    ax.set_ylim([-0.01, 0.43])
    ax.tick_params(which = 'both', length=8, width=2, colors='k',
                grid_alpha=0.7, labelsize = text_font)
            
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
    

# %%
    
def rate_comaprison_different(save=False):
    text_font = 24
    title_font = 20
    
    omega = 1.17
    omega_unc = 0.18
    offset = -0.006
    amp = 0.302
    
    file = '26.5_MHz_splitting_15_bins_error'
    folder = 'nv1_2019_05_10_28MHz_5'
    data = tool_belt.get_raw_data('t1_double_quantum.py', '{}\\{}'.format(folder, file))
    first_number = 5
    second_number = 4
    
    num_runs - data['num_runs']
    
    counts_f = data['gamma_counts_list'][first_number]
    error_f = numpy.array(data['gamma_counts_ste_list'][first_number])*2 / numpy.sqrt(num_runs)
    counts_s = data['gamma_counts_list'][second_number] 
    error_s = numpy.array(data['gamma_counts_ste_list'][second_number])*2  / numpy.sqrt(num_runs)
    
    time = numpy.array(data['taus'])*1000
    time_linspace = numpy.linspace(time[0], time[-1], 1000)
    
    gamma_f = data['gamma_list'][first_number]
    gamma_unc_f = data['gamma_ste_list'][first_number]
    gamma_s = data['gamma_list'][second_number]
    gamma_unc_s = data['gamma_ste_list'][second_number]
    
    opti_params_f = [(2*gamma_f+omega), amp, offset]
    opti_params_s = [(2*gamma_s+omega), amp, offset]
    
    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(10,8))
    
    ax.errorbar(time, counts_f, yerr = error_f, label = 'gamma = {}({}) kHz'.format(gamma_f, gamma_unc_f), 
                fmt = 'o', color = 'green')
    yfit = exp_eq_offset(time_linspace, *opti_params_f)
#    ax.plot(time_linspace, yfit, '-', color='green')
    
    opti_params_f[0] = (2*(gamma_f + gamma_unc_f) + omega + omega_unc)
    yupper = exp_eq_offset(time_linspace, *opti_params_f)
    opti_params_f[0] = (2*(gamma_f - gamma_unc_f) + omega - omega_unc)
    ylower = exp_eq_offset(time_linspace, *opti_params_f)
    
#    ax.fill_between(time_linspace, yupper,  ylower,
#                     color='green', alpha=0.2)
    
    
    ax.errorbar(time, counts_s, yerr = error_s, label = 'gamma = {}({}) kHz'.format(gamma_s, gamma_unc_s), 
                fmt = 'o', color = 'blue')
    
    yfit = exp_eq_offset(time_linspace, *opti_params_s)
#    ax.plot(time_linspace, yfit, '-', color='blue')
    
    opti_params_s[0] = (2*(gamma_s + gamma_unc_s) + omega + omega_unc)
    yupper = exp_eq_offset(time_linspace, *opti_params_s)
    opti_params_s[0] = (2*(gamma_s - gamma_unc_s) + omega - omega_unc)
    ylower = exp_eq_offset(time_linspace, *opti_params_s)
    
#    ax.fill_between(time_linspace, yupper,  ylower,
#                     color='blue', alpha=0.2)
    
    ax.set_xlabel(r'Relaxation time ($\mu$s)', fontsize=text_font)
    ax.set_ylabel('Relaxation signal', fontsize=text_font)
    ax.tick_params(which = 'both', length=8, width=2, colors='k',
                grid_alpha=0.7, labelsize = text_font)
    ax.grid()
    ax.set_ylim([-0.06, 0.38])
#    ax.legend()
#    ax.set_title('Compare NV1 measurements')
    
    fig.canvas.draw()
    fig.canvas.flush_events()

    if save:
        fig.savefig("C:/Users/Aedan/Creative Cloud Files/Paper Illustrations/Magnetically Forbidden Rate/supplemental_materials/NV1_comp_rate_diff.pdf", bbox_inches='tight')

# %%
    
def rate_comaprison_similar(save=False):
    text_font = 24
    title_font = 20
    
    omega = 1.17
    omega_unc = 0.18
    offset = -0.006
    amp = 0.302
    
    file = '26.5_MHz_splitting_15_bins_error'
    folder = 'nv1_2019_05_10_28MHz_5'
    data = tool_belt.get_raw_data('t1_double_quantum.py', '{}\\{}'.format(folder, file))
    first_number = 3
    second_number = 4
    
    num_runs - data['num_runs']
    
    counts_f = data['gamma_counts_list'][first_number]
    error_f = numpy.array(data['gamma_counts_ste_list'][first_number])*2 / numpy.sqrt(num_runs)
    counts_s = data['gamma_counts_list'][second_number] 
    error_s = numpy.array(data['gamma_counts_ste_list'][second_number])*2  / numpy.sqrt(num_runs)
    
    time = numpy.array(data['taus'])*1000
    time_linspace = numpy.linspace(time[0], time[-1], 1000)
    
    gamma_f = data['gamma_list'][first_number]
    gamma_unc_f = data['gamma_ste_list'][first_number]
    gamma_s = data['gamma_list'][second_number]
    gamma_unc_s = data['gamma_ste_list'][second_number]
    
    opti_params_f = [(2*gamma_f+omega), amp, offset]
    opti_params_s = [(2*gamma_s+omega), amp, offset]
    
    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    ax.errorbar(time, counts_f, yerr = error_f, label = 'gamma = {}({}) kHz'.format(gamma_f, gamma_unc_f), 
                fmt = 'o', color = 'red')
    yfit = exp_eq_offset(time_linspace, *opti_params_f)
#    ax.plot(time_linspace, yfit, '-', color='red')
    
    opti_params_f[0] = (2*(gamma_f + gamma_unc_f) + omega + omega_unc)
    yupper = exp_eq_offset(time_linspace, *opti_params_f)
    opti_params_f[0] = (2*(gamma_f - gamma_unc_f) + omega - omega_unc)
    ylower = exp_eq_offset(time_linspace, *opti_params_f)
    
#    ax.fill_between(time_linspace, yupper,  ylower,
#                     color='red', alpha=0.2)
    
    
    ax.errorbar(time, counts_s, yerr = error_s, label = 'gamma = {}({}) kHz'.format(gamma_s, gamma_unc_s), 
                fmt = 'o', color = 'blue')
    
    yfit = exp_eq_offset(time_linspace, *opti_params_s)
#    ax.plot(time_linspace, yfit, '-', color='blue')
    
    opti_params_s[0] = (2*(gamma_s + gamma_unc_s) + omega + omega_unc)
    yupper = exp_eq_offset(time_linspace, *opti_params_s)
    opti_params_s[0] = (2*(gamma_s - gamma_unc_s) + omega - omega_unc)
    ylower = exp_eq_offset(time_linspace, *opti_params_s)
    
#    ax.fill_between(time_linspace, yupper,  ylower,
#                     color='blue', alpha=0.2)
    
    ax.set_xlabel(r'Relaxation time ($\mu$s)', fontsize=text_font)
    ax.set_ylabel('Relaxation signal', fontsize=text_font)
    ax.tick_params(which = 'both', length=8, width=2, colors='k',
                grid_alpha=0.7, labelsize = text_font)
    ax.grid()
    ax.set_ylim([-0.06, 0.38])
#    ax.legend()
#    ax.set_title('Compare NV1 measurements')
    
    fig.canvas.draw()
    fig.canvas.flush_events()

    if save:
        fig.savefig("C:/Users/Aedan/Creative Cloud Files/Paper Illustrations/Magnetically Forbidden Rate/supplemental_materials/NV1_comp_rate_same.pdf", bbox_inches='tight')

#%%
    
def rate_comaprison_NV2(save=False):
    text_font = 18
    blue = '#2e3192'
    red = '#ed1c24'
    omega = 0.32 / 1000 
    omega_unc = 0.12 / 1000
    offset = 0
    
#    file = '28.9_MHz_splitting_rate_analysis'
#    folder = 'nv2_2019_04_30_29MHz_5'
#    data_f = tool_belt.get_raw_data('t1_double_quantum.py', '{}\\{}'.format(folder, file))
#    
#    file = '29.1_MHz_splitting_rate_analysis'
#    folder = 'nv2_2019_04_30_29MHz_6'
#    data_s = tool_belt.get_raw_data('t1_double_quantum.py', '{}\\{}'.format(folder, file))
#    
    
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
    
    ax.errorbar(time, counts_f, yerr = error_f, label = 'gamma = {}({}) kHz'.format(gamma_f, gamma_unc_f), 
                fmt = 'o', color = blue)
    yfit = exp_eq_offset(time_linspace, *opti_params_f)
    ax.plot(time_linspace, yfit, '-', color=blue)
    
    opti_params_f[0] = (2*(gamma_f + gamma_unc_f) + omega + omega_unc)
    yupper = exp_eq_offset(time_linspace, *opti_params_f)
    opti_params_f[0] = (2*(gamma_f - gamma_unc_f) + omega - omega_unc)
    ylower = exp_eq_offset(time_linspace, *opti_params_f)
    
    ax.fill_between(time_linspace, yupper,  ylower,
                     color='blue', alpha=0.2)
    
    
    ax.errorbar(time, counts_s, yerr = error_s, label = 'gamma = {}({}) kHz'.format(gamma_s, gamma_unc_s), 
                fmt = '^', color = red)
    
    yfit = exp_eq_offset(time_linspace, *opti_params_s)
    ax.plot(time_linspace, yfit, '-', color=red)
    
    opti_params_s[0] = (2*(gamma_s + gamma_unc_s) + omega + omega_unc)
    yupper = exp_eq_offset(time_linspace, *opti_params_s)
    opti_params_s[0] = (2*(gamma_s - gamma_unc_s) + omega - omega_unc)
    ylower = exp_eq_offset(time_linspace, *opti_params_s)
    
    ax.fill_between(time_linspace, yupper,  ylower,
                     color='red', alpha=0.2)
    
    ax.set_xlabel(r'Relaxation time ($\mu$s)', fontsize=text_font)
    ax.set_ylabel('Relaxation signal', fontsize=text_font)
    ax.tick_params(which = 'both', length=8, width=2, colors='k',
                grid_alpha=0.7, labelsize = text_font)
    ax.grid()
#    ax.set_xlim()
#    ax.legend()
#    ax.set_title('Compare NV1 measurements')
    
    fig.canvas.draw()
    fig.canvas.flush_events()
    
    if save:
        fig.savefig("C:/Users/Aedan/Creative Cloud Files/Paper Illustrations/Magnetically Forbidden Rate/supplemental_materials/NV2_rate_compare.pdf", bbox_inches='tight')

# %%
    
def subtraction_plot(save=False):
    
    text_font = 16
    title_font = 20
    folder = 'E:/Shared drives/Kolkowitz Lab Group/nvdata/t1_double_quantum/nv1_2019_05_10_28MHz_2'
    
    file = '27.7_MHz_splitting_rate_analysis.txt'
    
    with open('{}/{}'.format(folder, file)) as file:
            data = json.load(file)
            
            zero_relaxation_counts = data['zero_relaxation_counts']
            zero_relaxation_ste = numpy.array(data['zero_relaxation_ste'])*2
            zero_zero_time = data['zero_zero_time']
            
            plus_relaxation_counts = data['plus_relaxation_counts']
            plus_relaxation_ste = numpy.array(data['plus_relaxation_ste'])*2
            plus_plus_time = data['plus_plus_time']
            
            omega_opti_params = data['omega_opti_params']
            gamma_opti_params = data['gamma_opti_params']
            
    fig, axes_pack = plt.subplots(2,1, figsize=(8.5, 7.5))
    
    ax = axes_pack[0]
    
    ax.errorbar(zero_zero_time, zero_relaxation_counts,
                        yerr = zero_relaxation_ste,
                        label = 'data', fmt = 'o', color = 'blue')
    zero_time_linspace = numpy.linspace(0, zero_zero_time[-1], num=1000)
    ax.plot(zero_time_linspace,
                exp_eq(zero_time_linspace, *omega_opti_params),
                'r', label = 'fit')
    ax.set_xlabel('Relaxation time (ms)', fontsize=text_font)
    ax.set_ylabel('Relaxation signal', fontsize=text_font)
    ax.set_title(r'$P_{0,0} - P_{0,1}$', fontsize=title_font)
#    ax.legend(fontsize=20)
    ax.tick_params(which = 'both', length=8, width=2, colors='k',
                grid_alpha=0.7, labelsize = text_font)
    
    ax = axes_pack[1]
    
    ax.errorbar(plus_plus_time, plus_relaxation_counts,
                        yerr = plus_relaxation_ste,
                        label = 'data', fmt = 'o', color = 'blue')
    plus_time_linspace = numpy.linspace(0, plus_plus_time[-1], num=1000)
    ax.plot(plus_time_linspace,
                exp_eq(plus_time_linspace, *gamma_opti_params),
                'r', label = 'fit')
    ax.set_xlabel('Relaxation time (ms)', fontsize=text_font)
    ax.set_ylabel('Relaxation signal', fontsize=text_font)
    ax.set_title(r'$P_{1,1} - P_{1,-1}$', fontsize=title_font)
#    ax.legend(fontsize=20)

    ax.tick_params(which = 'both', length=8, width=2, colors='k',
                grid_alpha=0.7, labelsize = text_font)
            
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
        with open('{}/{}'.format(folder, file_list[i*2])) as file_shrt:
            data_shrt = json.load(file_shrt)
            relaxation_time_range = data_shrt['relaxation_time_range']
            num_steps = data_shrt['num_steps']
            
            taus_shrt = numpy.linspace(relaxation_time_range[0],
                                  relaxation_time_range[1], num_steps) / 10**6
            norm_avg_sig_shrt = (numpy.array(data_shrt['norm_avg_sig']) - 0.622735) / 0.376254
        
        with open('{}/{}'.format(folder, file_list[i*2+1])) as file_long:
            data_long = json.load(file_long)
            
            relaxation_time_range = data_long['relaxation_time_range']
            num_steps = data_long['num_steps']
            
            taus_long = numpy.linspace(relaxation_time_range[0],
                                  relaxation_time_range[1], num_steps) / 10**6
            norm_avg_sig_long = (numpy.array(data_long['norm_avg_sig']) - 0.622735) / 0.376254
            
            
        taus = numpy.concatenate((taus_shrt, taus_long))
        norm_sig = numpy.concatenate((norm_avg_sig_shrt, norm_avg_sig_long))
    
        ax = axes[r_ind,c_ind]
        
        ax.plot(taus, norm_sig, 'bo')
        ax.set_ylim([-0.1,1.1])
        ax.set_xlabel(r'Relaxation time, $\tau$ ($\mu$s)', fontsize=20)
        ax.set_ylabel('Normalized NV Fluorescence', fontsize=20)
        ax.set_title(titles[i], fontsize=24)
        ax.tick_params(which = 'both', length=10, width=3, colors='k',
                        grid_alpha=0.7, labelsize = 20)
        
        
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
if __name__ == '__main__':
        
#    all_9_meas(True)
#    subtraction_plot(True) 
#    omega_comparison(True)
#    one_hour_rates_NV1(save=True)
#    NV1_histogram(save=True)
#    one_hour_rates_NV2(save=True)
#    NV2_histogram(save=True)
#    rate_comaprison_similar(save=True)
#    rate_comaprison_different(save=True)
    rate_comaprison_NV2(save=True)