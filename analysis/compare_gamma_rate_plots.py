# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 09:24:01 2019

Plot t1 data on top of eachother, specifically for the gamma rate

@author: Aedan
"""

import utils.tool_belt as tool_belt
import matplotlib.pyplot as plt
from scipy import exp
import numpy

# %%

def exp_eq_offset(t, rate, amp, offset):
    return offset + amp * exp(- rate * t)

# %%
    
omega = 0.34
omega_unc = 0.07

file = '29.9_MHz_splitting_1_bins'
folder = 'nv2_2019_04_30_29MHz_9'
data_f = tool_belt.get_raw_data('t1_double_quantum.py', '{}\\{}'.format(folder, file))
gamma_f = 32.9
gamma_unc_f = 0.7

file = '29.8_MHz_splitting_1_bins'
folder = 'nv2_2019_04_30_29MHz_10'
data_s = tool_belt.get_raw_data('t1_double_quantum.py', '{}\\{}'.format(folder, file))
gamma_s = 28.9
gamma_unc_s = 0.7

# %%

counts_f = data_f['plus_relaxation_counts']
counts_s = data_s['plus_relaxation_counts']    

time = data_f['plus_plus_time']
time_linspace = numpy.linspace(time[0], time[-1], 1000)

opti_params_f = data_f['gamma_opti_params']
opti_params_s = data_s['gamma_opti_params']

# Plot
fig, ax = plt.subplots(1, 1, figsize=(11, 8.5))

ax.plot(time, counts_f, 'b.', 
        label = 'gamma = {}({}) kHz'.format(gamma_f, gamma_unc_f))
yfit = exp_eq_offset(time_linspace, *opti_params_f)
ax.plot(time_linspace, yfit, '-', color='blue')

opti_params_f[0] = (2*(gamma_f + gamma_unc_f) + omega + omega_unc)
yupper = exp_eq_offset(time_linspace, *opti_params_f)
opti_params_f[0] = (2*(gamma_f - gamma_unc_f) + omega - omega_unc)
ylower = exp_eq_offset(time_linspace, *opti_params_f)

ax.fill_between(time_linspace, yupper,  ylower,
                 color='blue', alpha=0.2)


ax.plot(time, counts_s, 'r.', 
            label = 'gamma = {}({}) kHz'.format(gamma_s, gamma_unc_s))

yfit = exp_eq_offset(time_linspace, *opti_params_s)
ax.plot(time_linspace, yfit, '-', color='red')

opti_params_s[0] = (2*(gamma_s + gamma_unc_s) + omega + omega_unc)
yupper = exp_eq_offset(time_linspace, *opti_params_s)
opti_params_s[0] = (2*(gamma_s - gamma_unc_s) + omega - omega_unc)
ylower = exp_eq_offset(time_linspace, *opti_params_s)

ax.fill_between(time_linspace, yupper,  ylower,
                 color='red', alpha=0.2)

ax.set_xlabel('Relaxation time (ms)')
ax.set_ylabel('Contrast (arb. units)')
ax.legend()

fig.canvas.draw()
fig.canvas.flush_events()

