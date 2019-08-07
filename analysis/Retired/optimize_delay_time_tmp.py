# -*- coding: utf-8 -*-
"""
Created on Tue May 14 15:21:51 2019

quick routine set up to read the data taken for the signal to noise analysis 
and replot it.

@author: gardill
"""

import utils.tool_belt as tool_belt
import json
import os
import numpy
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

file_list = os.listdir('E:/Team Drives/Kolkowitz Lab Group/nvdata/optimize_gate_time/ND_filter')

snr_list = []
#min_delay_time = 250
#max_delay_time = 400
#num_delay_steps = int((max_delay_time - min_delay_time) / 10 + 1)
#delay_time_array = numpy.linspace(min_delay_time, max_delay_time, num = num_delay_steps).astype(int)

nd_list = [0.5, 1.0, 1.5, 2.0]
    
for file in file_list:
    
    with open('E:/Team Drives/Kolkowitz Lab Group/nvdata/optimize_gate_time/ND_filter/' + file) as json_file:
        
        data = json.load(json_file)
        
        sig_counts = numpy.array(data['sig_counts'])
        ref_counts = numpy.array(data['ref_counts'])
        norm_avg_sig = data['norm_avg_sig']
        
    # using contrast to define signal
#    sig_stat = 1 - numpy.average(norm_avg_sig)# calculate the contrast between the two
#    st_dev_stat = numpy.std(norm_avg_sig)    
    
    # using high - low counts to define signal
    
    sig_stat = numpy.average(ref_counts - sig_counts)
    st_dev_stat = numpy.std(ref_counts - sig_counts)
    sig_to_noise_ratio = sig_stat / st_dev_stat
    
    snr_list.append(sig_to_noise_ratio)


def parabola(t, offset, amplitude, delay_time):
    return offset + amplitude * (t - delay_time)**2
    
offset = 10
amplitude = 100
delay_time = 300

popt,pcov = curve_fit(parabola, nd_list, snr_list, 
                              p0=[offset, amplitude, delay_time]) 

linspace_time = numpy.linspace(0.5, 2.0, num = 1000)
   
fig, ax = plt.subplots(1, 1, figsize=(12, 10))
ax.plot(nd_list, snr_list, 'ro', label = 'data')
ax.plot(linspace_time, parabola(linspace_time,*popt), 'b-', label = 'fit')
ax.set_xlabel('ND Filter')
ax.set_ylabel('Signal-to-noise ratio') 
ax.legend() 

text = ('Optimal ND filter = {:.2f}'.format(popt[2]))

props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
ax.text(0.70, 0.05, text, transform=ax.transAxes, fontsize=12,
                            verticalalignment="top", bbox=props)

fig.canvas.draw()
fig.canvas.flush_events()



timestamp = tool_belt.get_time_stamp()
raw_data = {'timestamp': timestamp,
            'snr_list': snr_list,
            'nd_list': nd_list}
#fig.savefig('2019-05-14_15-35-12_Ayrton12_SNR.svg')

file_path = tool_belt.get_file_path(__file__, timestamp, 'Ayrton12_SNR_fit')
#tool_belt.save_figure(raw_fig, file_path)
tool_belt.save_raw_data(raw_data, file_path)