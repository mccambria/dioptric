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
from utils.tool_belt import States
from scipy import exp

# %% equations
def exp_eq(t, rate, amp):
    return  amp * exp(- rate * t)

# %%
    
def omega_comparison(save=False):
    return

# %%
    
def 1_hour_rates(save = False):
    return
# %%
    
def rate_comaprison(save=False):
    return

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
    titles = [r'P$_{1,1}$ ($\tau$)',
              r'P$_{1,-1}$ ($\tau$)',
              r'P$_{1,0}$ ($\tau$)',
              r'P$_{-1,1}$ ($\tau$)',
              r'P$_{-1,-1}$ ($\tau$)',
              r'P$_{-1,0}$ ($\tau$)',
              r'P$_{0,1}$ ($\tau$)',
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
        ax.tick_params(which = 'both', length=10, width=2, colors='k',
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
        
#    all_9_meas()
#    subtraction_plot(True) 
#    omega_comparison()
#    1_hour_rates()
#    rate_comaprison()