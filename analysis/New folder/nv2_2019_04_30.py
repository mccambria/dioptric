# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 14:36:23 2019

This file plots the relaxation rate data collected for the nv2_2019_04_30.

The data is input manually, and plotted on a loglog plot with error bars along
the y-axis. A 1/f**2 line is also fit to the gamma rates to show the behavior.

@author: Aedan
"""

'''
nv2_2019_04_30

files used:
    2019-06-13_15-45-15_29.1_MHz_splitting_rate_analysis
    2019-06-13_15-46-41_44.8_MHz_splitting_rate_analysis
    2019-06-13_15-47-36_56.2_MHz_splitting_rate_analysis
    2019-06-13_15-48-16_56.9_MHz_splitting_rate_analysis
    2019-06-12_12-46-44_69.8_MHz_splitting_rate_analysis
    2019-06-12_12-49-10_85.1_MHz_splitting_rate_analysis
    2019-06-12_12-50-06_101.6_MHz_splitting_rate_analysis
'''
# %%
def fit_eq(f, amp):
    return amp*f**(-2)

# %%

import matplotlib.pyplot as plt
from scipy import asarray as ar, exp
from scipy.optimize import curve_fit
import numpy

# The data
splitting_list = [29.1, 44.8, 56.2, 56.9, 69.8, 85.1, 101.6]

omega_avg_list = [0.39, 0.51, 0.32, 0.41, 0.33, 0.33, 0.27]

omega_error_list = [0.09, 0.12, 0.08, 0.09, 0.06, 0.06, 0.04]

gamma_avg_list = [20.5, 7.2, 3.9, 3.9, 2.48, 3.0, 1.6]

gamma_error_list = [0.9, 0.4, 0.3, 0.2, 0.17, 0.3, 0.3]

# Try to fit the gamma to a 1/f^2

fit_params, cov_arr = curve_fit(fit_eq, splitting_list, gamma_avg_list, 
                                p0 = 100)

splitting_linspace = numpy.linspace(splitting_list[0], splitting_list[-1],
                                    1000)


fig, ax = plt.subplots(1, 1, figsize=(10, 8))

#ax.errorbar(splitting_list, omega_avg_list, yerr = omega_error_list)
ax.set_xscale("log", nonposx='clip')
ax.set_yscale("log", nonposy='clip')
ax.errorbar(splitting_list, gamma_avg_list, yerr = gamma_error_list, 
            label = 'Gamma', fmt='o', color='blue')
ax.errorbar(splitting_list, omega_avg_list, yerr = omega_error_list, 
            label = 'Omega', fmt='o', color='red')
ax.plot(splitting_linspace, fit_eq(splitting_linspace, *fit_params), 
            label = '1/f^2')
ax.grid()

ax.set_xlabel('Splitting (MHz)')
ax.set_ylabel('Relaxation Rate (kHz)')
ax.set_title('NV2_2019_04_30')
ax.legend()
