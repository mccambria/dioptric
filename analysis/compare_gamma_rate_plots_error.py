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

def exp_eq_offset(t, rate, amp):
    return  amp * exp(- rate * t)

# %%
    
num_runs = 20
omega = 0.34
omega_unc = 0.07

file = '29.1_MHz_splitting_rate_analysis'
folder = 'nv2_2019_04_30_29MHz'
data_f = tool_belt.get_raw_data('t1_double_quantum.py', '{}\\{}'.format(folder, file))
gamma_f = 18.7
gamma_unc_f = 0.3*2

file = '29.2_MHz_splitting_rate_analysis'
folder = 'nv2_2019_04_30_29MHz_2'
data_s = tool_belt.get_raw_data('t1_double_quantum.py', '{}\\{}'.format(folder, file))
gamma_s = 31.1
gamma_unc_s = 0.4*2

# %%

counts_f = data_f['plus_relaxation_counts']
error_f = numpy.array(data_f['plus_relaxation_ste']) / numpy.sqrt(num_runs)
counts_s = data_s['plus_relaxation_counts'] 
error_s = numpy.array(data_s['plus_relaxation_ste'])  / numpy.sqrt(num_runs)

time_f = data_f['plus_plus_time']
time_linspace = numpy.linspace(time_f[0], time_f[-1], 1000)
time_s = data_s['plus_plus_time']

opti_params_f = data_f['gamma_opti_params']
opti_params_s = data_s['gamma_opti_params']

# Plot
fig, ax = plt.subplots(1, 1, figsize=(11, 8.5))

ax.errorbar(time_f, counts_f, yerr = error_f, label = 'gamma = {}({}) kHz'.format(gamma_f, gamma_unc_f), 
            fmt = '.', color = 'blue')
yfit = exp_eq_offset(time_linspace, *opti_params_f)
ax.plot(time_linspace, yfit, '-', color='blue')

opti_params_f[0] = (2*(gamma_f + gamma_unc_f) + omega + omega_unc)
yupper = exp_eq_offset(time_linspace, *opti_params_f)
opti_params_f[0] = (2*(gamma_f - gamma_unc_f) + omega - omega_unc)
ylower = exp_eq_offset(time_linspace, *opti_params_f)

ax.fill_between(time_linspace, yupper,  ylower,
                 color='blue', alpha=0.2)


ax.errorbar(time_s, counts_s, yerr = error_s, label = 'gamma = {}({}) kHz'.format(gamma_s, gamma_unc_s), 
            fmt = '.', color = 'red')

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
ax.set_title('Compare NV2 29 MHz measurements')

fig.canvas.draw()
fig.canvas.flush_events()
    

