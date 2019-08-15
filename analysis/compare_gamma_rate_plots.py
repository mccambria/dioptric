# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 09:24:01 2019

Plot t1 data on top of eachother, specifically fo the gamma rate

@author: Aedan
"""

import utils.tool_belt as tool_belt
import matplotlib.pyplot as plt
from scipy import exp

def exp_eq_offset(t, rate, amp, offset):
    return offset + amp * exp(- rate * t)


file = '29.5_MHz_splitting_1_bins'
folder = 'nv2_2019_04_30_29MHz_8'
data_f = tool_belt.get_raw_data('t1_double_quantum.py', '{}\\{}'.format(folder, file))

file = '29.9_MHz_splitting_1_bins'
folder = 'nv2_2019_04_30_29MHz_9'
data_s = tool_belt.get_raw_data('t1_double_quantum.py', '{}\\{}'.format(folder, file))

counts_f = data_f['plus_relaxation_counts']
counts_s = data_s['plus_relaxation_counts']    

time = data_f['plus_plus_time']

opti_params_f = data_f['gamma_opti_params']
opti_params_s = data_s['gamma_opti_params']

fig, ax = plt.subplots(1, 1, figsize=(11, 8.5))

ax.plot(time, counts_f, 'b.', label = 'gamma = 33(1) kHz')
ax.plot(time, counts_s, 'r.', label = 'gamma = 32.9(7) kHz')

ax.set_xlabel('Relaxation time (ms)')
ax.set_ylabel('Contrast (arb. units)')
ax.legend()

fig.canvas.draw()
fig.canvas.flush_events()
    

