# -*- coding: utf-8 -*-
"""
Created on Tue May 14 15:21:51 2019

@author: kolkowitz
"""

import json
import os
import numpy
import matplotlib.pyplot as plt

file_list = os.listdir('E:/Team Drives/Kolkowitz Lab Group/nvdata/optimize_gate_time/run_two')

snr_list = []
min_delay_time = 250
max_delay_time = 400
num_delay_steps = int((max_delay_time - min_delay_time) / 10 + 1)
delay_time_list = numpy.linspace(min_delay_time, max_delay_time, num = num_delay_steps).astype(int)
    
for file in file_list:
    
    with open('E:/Team Drives/Kolkowitz Lab Group/nvdata/optimize_gate_time/run_two/' + file) as json_file:
        
        data = json.load(json_file)
        
        norm_avg_sig = data['norm_avg_sig']
        
    sig_stat = numpy.average(norm_avg_sig) 
    st_dev_stat = numpy.std(norm_avg_sig)    
    sig_to_noise_ratio = sig_stat / st_dev_stat
    
    snr_list.append(sig_to_noise_ratio)
    
fig, ax = plt.subplots(1, 1, figsize=(12, 10))
ax.plot(delay_time_list, snr_list, 'r-')
ax.set_xlabel('Gate time (ns)')
ax.set_ylabel('Signal-to-noise ratio')  

fig.canvas.draw()
fig.canvas.flush_events()
fig.savefig('2019-05-14_15-35-12_Ayrton12_SNR.svg')