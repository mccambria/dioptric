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
        
        init_state = data_shrt['init_state']
        read_state = data_shrt['read_state']

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

fig.savefig("C:/Users/Aedan/Creative Cloud Files/Paper Illustrations/Magnetically Forbidden Rate/supplemental_materials/all_9_measure_figure.pdf", bbox_inches='tight')


    
    
    