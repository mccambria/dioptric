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
    
omega = 1.6
omega_unc = 1.6

first_ind = 5
second_ind = 2

folder = 'nv1_2019_05_10_28MHz_4'
file = '26.3_MHz_splitting_30_bins_error'
data_f = tool_belt.get_raw_data('t1_double_quantum.py', '{}\\{}'.format(folder, file))


gamma_f = data_f['gamma_list'][first_ind]
gamma_unc_f = data_f['gamma_ste_list'][first_ind]

gamma_s = data_f['gamma_list'][second_ind]
gamma_unc_s = data_f['gamma_ste_list'][second_ind]

# %%

counts_f = data_f['gamma_counts_list'][first_ind]
counts_s = data_f['gamma_counts_list'][second_ind]  

time = data_f['taus']
time_linspace = numpy.linspace(time[0], time[-1], 1000)

amp, offset = 0.3038, -0.0086

opti_params_f = [data_f['gamma_fit_params_list'][first_ind][0], amp, offset]
opti_params_s = [data_f['gamma_fit_params_list'][second_ind][0], amp, offset]

# Plot
fig, ax = plt.subplots(1, 1, figsize=(11, 8.5))

ax.plot(time, counts_f, 'b.', 
        label = 'gamma = {}({}) kHz'.format('%.1f'%gamma_f, '%.1f'%gamma_unc_f))
yfit = exp_eq_offset(time_linspace, *opti_params_f)
ax.plot(time_linspace, yfit, '-', color='blue')

opti_params_f[0] = (2*(gamma_f + gamma_unc_f) + omega + omega_unc)
yupper = exp_eq_offset(time_linspace, *opti_params_f)
opti_params_f[0] = (2*(gamma_f - gamma_unc_f) + omega - omega_unc)
ylower = exp_eq_offset(time_linspace, *opti_params_f)

ax.fill_between(time_linspace, yupper,  ylower,
                 color='blue', alpha=0.2)


ax.plot(time, counts_s, 'r.', 
            label = 'gamma = {}({}) kHz'.format('%.1f'%gamma_s, '%.1f'%gamma_unc_s))

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

