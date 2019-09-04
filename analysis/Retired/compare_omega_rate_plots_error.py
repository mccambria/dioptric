# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 09:24:01 2019

Plot t1 data on top of eachother, specifically fo the gamma rate

@author: Aedan
"""

import utils.tool_belt as tool_belt
import matplotlib.pyplot as plt
import numpy
from scipy import exp

# %%

def exp_eq_offset(t, rate, amp, offset):
    return offset + amp * exp(- rate * t)

# %%
    
num_runs = 20
omega = 0.34
omega_unc = 0.07

file = '1015.0_MHz_splitting_omega_comparison'
folder = 'nv1_2019_05_10_1017MHz\high_low_comparison'
data = tool_belt.get_raw_data('t1_double_quantum.py', '{}\\{}'.format(folder, file))
omega_H = data['omega_HIGH_opti_params'][0] / 3
omega_H_ste = data['omega_HIGH_ste']

omega_L = data['omega_LOW_opti_params'][0] / 3
omega_L_ste = data['omega_LOW_ste']

# %%

counts_H = data['omega_HIGH_relaxation_counts']
error_H = data['omega_HIGH_relaxation_ste']
counts_L = data['omega_LOW_relaxation_counts'] 
error_L = data['omega_LOW_relaxation_ste']

time = data['zero_zero_time']
time_linspace = numpy.linspace(time[0], time[-1], 1000)

opti_params_H = data['omega_HIGH_opti_params']
opti_params_L = data['omega_LOW_opti_params']

# Plot
fig, ax = plt.subplots(1, 1, figsize=(11, 8.5))

ax.errorbar(time, counts_H, yerr = error_H, label = 'Omega High = {}({}) kHz'.format('%.3f'%omega_H,'%.3f'%omega_H_ste), 
            fmt = '.', color = 'blue')
yfit = exp_eq_offset(time_linspace, *opti_params_H)
ax.plot(time_linspace, yfit, '-', color='blue')

opti_params_H[0] = (3 *(omega_H + omega_H_ste))
yupper = exp_eq_offset(time_linspace, *opti_params_H)
opti_params_H[0] = (3 *(omega_H - omega_H_ste))
ylower = exp_eq_offset(time_linspace, *opti_params_H)

ax.fill_between(time_linspace, yupper,  ylower,
                 color='blue', alpha=0.2)


ax.errorbar(time, counts_L, yerr = error_L, label = 'Omega Low = {}({}) kHz'.format('%.3f'%omega_L, '%.3f'%omega_L_ste), 
            fmt = '.', color = 'red')

yfit = exp_eq_offset(time_linspace, *opti_params_L)
ax.plot(time_linspace, yfit, '-', color='red')

opti_params_L[0] = (3 *(omega_L + omega_L_ste))
yupper = exp_eq_offset(time_linspace, *opti_params_L)
opti_params_L[0] = (3 *(omega_L - omega_L_ste))
ylower = exp_eq_offset(time_linspace, *opti_params_L)

ax.fill_between(time_linspace, yupper,  ylower,
                 color='red', alpha=0.2)

ax.set_xlabel('Relaxation time (ms)')
ax.set_ylabel('Contrast (arb. units)')
ax.legend()

fig.canvas.draw()
fig.canvas.flush_events()
    

