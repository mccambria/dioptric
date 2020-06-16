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
    
init_params = [0.3, 1]

# %%
    
# Area a5 
area_name_1 = 'Bachman A5 - 240 MHz splitting'
data = tool_belt.get_raw_data('t1_double_quantum/data_folders/other_data/bachman/bachman-A5-ensemble-B1-234MHz/', 
                              '234MHz_splitting_rate_analysis')

zero_relaxation_counts_1 = data['zero_relaxation_counts']
zero_relaxation_ste_1 = data['zero_relaxation_ste']
zero_zero_time_1 = data['zero_zero_time']
plus_relaxation_counts_1 = data['plus_relaxation_counts']
plus_relaxation_ste_1 = data['plus_relaxation_ste']
plus_plus_time_1 = data['plus_plus_time']
omega_opti_params_1 = data['omega_opti_params']
omega_1 = data['omega']
omega_ste_1 = data['omega_ste']
gamma_opti_params_1 = data['gamma_opti_params']
gamma_1 = data['gamma']
gamma_ste_1 = data['gamma_ste']

omega_time_linspace_1 = numpy.linspace(zero_zero_time_1[0],zero_zero_time_1[-1],1000)
gamma_time_linspace_1 = numpy.linspace(plus_plus_time_1[0],plus_plus_time_1[-1],1000)
    
# manipulate the data to normalize 
first_point = zero_relaxation_counts_1[0]
zero_relaxation_counts_1 = numpy.array(zero_relaxation_counts_1)/ first_point
zero_relaxation_ste_1 = numpy.array(zero_relaxation_ste_1)/ first_point

first_point = plus_relaxation_counts_1[0]
plus_relaxation_counts_1 = numpy.array(plus_relaxation_counts_1)/ first_point
plus_relaxation_ste_1 = numpy.array(plus_relaxation_ste_1)/ first_point

# fit the normalized data
omega_popt_1, _ = curve_fit(exp_eq, zero_zero_time_1, zero_relaxation_counts_1,
                               p0=init_params)
gamma_popt_1, _ = curve_fit(exp_eq, plus_plus_time_1, plus_relaxation_counts_1,
                               p0=init_params)

# Area A1
area_name_2 = 'Bachman A1 - 240 MHz splitting'
data = tool_belt.get_raw_data('t1_double_quantum/data_folders/other_data/bachman/bachman-A1-ensemble-B1-232MHz/', 
                              '232MHz_splitting_rate_analysis')

zero_relaxation_counts_2 = data['zero_relaxation_counts']
zero_relaxation_ste_2 = data['zero_relaxation_ste']
zero_zero_time_2 = data['zero_zero_time']
#plus_relaxation_counts_2 = data['plus_relaxation_counts']
#plus_relaxation_ste_2 = data['plus_relaxation_ste']
#plus_plus_time_2 = data['plus_plus_time']
#omega_opti_params_2 = data['omega_opti_params']
omega_2 = data['omega']
omega_ste_2 = data['omega_ste']
#gamma_opti_params_2 = data['gamma_opti_params']
#gamma_2 = data['gamma']
#gamma_ste_2 = data['gamma_ste']

omega_time_linspace_2 = numpy.linspace(zero_zero_time_2[0],zero_zero_time_2[-1],1000)
#gamma_time_linspace_2 = numpy.linspace(plus_plus_time_2[0],plus_plus_time_2[-1],1000)

# manipulate the data to normalize 
first_point = zero_relaxation_counts_2[0]
zero_relaxation_counts_2 = numpy.array(zero_relaxation_counts_2)/ first_point
zero_relaxation_ste_2 = numpy.array(zero_relaxation_ste_2)/ first_point

#first_point = plus_relaxation_counts_2[0]
#plus_relaxation_counts_2 = numpy.array(plus_relaxation_counts_2)/ first_point
#plus_relaxation_ste_2 = numpy.array(plus_relaxation_ste_2)/ first_point

# fit the normalized data
omega_popt_2, _ = curve_fit(exp_eq, zero_zero_time_2, zero_relaxation_counts_2,
                               p0=init_params)
#gamma_popt_2, _ = curve_fit(exp_eq, plus_plus_time_2, plus_relaxation_counts_2,
#                               p0=init_params)

fig, axes = plt.subplots(1, 2, figsize=(17, 8))
ax = axes[0]
ax.errorbar(zero_zero_time_1, zero_relaxation_counts_1,
                    yerr = zero_relaxation_ste_1,
                    label = area_name_1, fmt = 'o', color = 'orange')
ax.plot(omega_time_linspace_1,
                    exp_eq(omega_time_linspace_1, *omega_popt_1),
                    'orange', label = '{} fit'.format(area_name_1))
ax.errorbar(zero_zero_time_2, zero_relaxation_counts_2,
                    yerr = zero_relaxation_ste_2,
                    label = area_name_2, fmt = 'o', color = 'red')
ax.plot(omega_time_linspace_2,
                    exp_eq(omega_time_linspace_2, *omega_popt_2),
                    'red', label = '{} fit'.format(area_name_2))
ax.set_xlabel('Wait time (ms)')
ax.set_ylabel('Normalized signal Counts')
ax.legend()
text = "\n".join((r'{} $\Omega = {:.2f} \pm {:.2f}$ kHz'.format(area_name_1, omega_1, omega_ste_1),
                  r'{} $\Omega = {:.2f} \pm {:.2f}$ kHz'.format(area_name_2, omega_2, omega_ste_2),))


props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
ax.text(0.40, 0.8, text, transform=ax.transAxes, fontsize=12,
                        verticalalignment="top", bbox=props)

#ax = axes[1]
#ax.errorbar(plus_plus_time_1, plus_relaxation_counts_1,
#                    yerr = plus_relaxation_ste_1,
#                    label = area_name_1, fmt = 'o', color = 'purple')
#ax.plot(gamma_time_linspace_1,
#                    exp_eq(gamma_time_linspace_1, *gamma_popt_1),
#                    'purple', label = '{} fit'.format(area_name_1))
#ax.errorbar(plus_plus_time_2, plus_relaxation_counts_2,
#                    yerr = plus_relaxation_ste_2,
#                    label = area_name_2, fmt = 'o', color = 'blue')
#ax.plot(gamma_time_linspace_2,
#                    exp_eq(gamma_time_linspace_2, *gamma_popt_2),
#                    'blue', label = '{} fit'.format(area_name_2))

ax.set_xlabel('Wait time (ms)')
ax.set_ylabel('Normalized signal Counts')
#ax.set_title('Comparing omega rate of bulk diamond at two splittings')
ax.legend()

#text = "\n".join((r'{} $\gamma = {:.2f} \pm {:.2f}$ kHz'.format(area_name_1, gamma_1, gamma_ste_1),
#                  r'{} $\gamma = {:.2f} \pm {:.2f}$ kHz'.format(area_name_2, gamma_2, gamma_ste_2),))


#props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
#ax.text(0.55, 0.8, text, transform=ax.transAxes, fontsize=12,
#                        verticalalignment="top", bbox=props)


