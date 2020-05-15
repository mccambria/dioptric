# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 09:51:31 2019

Plot the t1 data from two different runs, both gamma and omega

@author: Aedan
"""

# %% Imports

import numpy
from scipy import exp
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

import utils.tool_belt as tool_belt

# %% Functions

# The exponential function used to fit the data

def exp_eq(t, rate, amp):
    return amp * exp(- rate * t)

# %%
    
# Area A1 
data = tool_belt.get_raw_data('t1_double_quantum/data_folders/other_data/bachman-ensemble-B1/', 
                              '139MHz_splitting_rate_analysis')

zero_relaxation_counts_A1 = data['zero_relaxation_counts']
zero_relaxation_ste_A1 = data['zero_relaxation_ste']
zero_zero_time_A1 = data['zero_zero_time']
plus_relaxation_counts_A1 = data['plus_relaxation_counts']
plus_relaxation_ste_A1 = data['plus_relaxation_ste']
plus_plus_time_A1 = data['plus_plus_time']
omega_opti_params_A1 = data['omega_opti_params']
omega_A1 = data['omega']
omega_ste_A1 = data['omega_ste']
gamma_opti_params_A1 = data['gamma_opti_params']
gamma_A1 = data['gamma']
gamma_ste_A1 = data['gamma_ste']

omega_time_linspace_A1 = numpy.linspace(zero_zero_time_A1[0],zero_zero_time_A1[-1],1000)
gamma_time_linspace_A1 = numpy.linspace(plus_plus_time_A1[0],plus_plus_time_A1[-1],1000)
    
# Area B1
data = tool_belt.get_raw_data('t1_double_quantum/data_folders/other_data/bachman-ensemble-A1/', 
                              '142MHz_splitting_rate_analysis')

zero_relaxation_counts_B1 = data['zero_relaxation_counts']
zero_relaxation_ste_B1 = data['zero_relaxation_ste']
zero_zero_time_B1 = data['zero_zero_time']
plus_relaxation_counts_B1 = data['plus_relaxation_counts']
plus_relaxation_ste_B1 = data['plus_relaxation_ste']
plus_plus_time_B1 = data['plus_plus_time']
omega_opti_params_B1 = data['omega_opti_params']
omega_B1 = data['omega']
omega_ste_B1 = data['omega_ste']
gamma_opti_params_B1 = data['gamma_opti_params']
gamma_B1 = data['gamma']
gamma_ste_B1 = data['gamma_ste']

omega_time_linspace_B1 = numpy.linspace(zero_zero_time_B1[0],zero_zero_time_B1[-1],1000)
gamma_time_linspace_B1 = numpy.linspace(plus_plus_time_B1[0],plus_plus_time_B1[-1],1000)
    
fig, axes = plt.subplots(1, 2, figsize=(17, 8))
ax = axes[0]
ax.errorbar(zero_zero_time_A1, zero_relaxation_counts_A1,
                    yerr = zero_relaxation_ste_A1,
                    label = 'A1', fmt = 'o', color = 'orange')
ax.plot(omega_time_linspace_A1,
                    exp_eq(omega_time_linspace_A1, *omega_opti_params_A1),
                    'orange', label = 'A1 fit')
ax.errorbar(zero_zero_time_B1, zero_relaxation_counts_B1,
                    yerr = zero_relaxation_ste_B1,
                    label = 'B1', fmt = 'o', color = 'red')
ax.plot(omega_time_linspace_B1,
                    exp_eq(omega_time_linspace_B1, *omega_opti_params_B1),
                    'red', label = 'B1 fit')
ax.set_xlabel('Wait time (ms)')
ax.set_ylabel('Normalized signal Counts')
ax.legend()
text = "\n".join((r'Area A1 $\Omega = {:.2f} \pm {:.2f}$ kHz'.format(omega_A1, omega_ste_A1),
                  r'Area B1 $\Omega = {:.2f} \pm {:.2f}$ kHz'.format(omega_B1, omega_ste_B1),))


props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
ax.text(0.55, 0.8, text, transform=ax.transAxes, fontsize=12,
                        verticalalignment="top", bbox=props)

ax = axes[1]
ax.errorbar(plus_plus_time_A1, plus_relaxation_counts_A1,
                    yerr = plus_relaxation_ste_A1,
                    label = 'A1', fmt = 'o', color = 'purple')
ax.plot(gamma_time_linspace_A1,
                    exp_eq(gamma_time_linspace_A1, *gamma_opti_params_A1),
                    'purple', label = 'A1 fit')
ax.errorbar(plus_plus_time_B1, plus_relaxation_counts_B1,
                    yerr = plus_relaxation_ste_B1,
                    label = 'B1', fmt = 'o', color = 'blue')
ax.plot(gamma_time_linspace_B1,
                    exp_eq(gamma_time_linspace_B1, *gamma_opti_params_B1),
                    'blue', label = 'B1 fit')

ax.set_xlabel('Wait time (ms)')
ax.set_ylabel('Normalized signal Counts')
#ax.set_title('Comparing omega rate of bulk diamond at two splittings')
ax.legend()

text = "\n".join((r'Area A1 $\gamma = {:.2f} \pm {:.2f}$ kHz'.format(gamma_A1, gamma_ste_A1),
                  r'Area B1 $\gamma = {:.2f} \pm {:.2f}$ kHz'.format(gamma_B1, gamma_ste_B1),))


props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
ax.text(0.55, 0.8, text, transform=ax.transAxes, fontsize=12,
                        verticalalignment="top", bbox=props)


