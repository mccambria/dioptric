# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 14:36:23 2019

This file plots the relaxation rate data collected for the nv1_2019_05_10.

The data is input manually, and plotted on a loglog plot with error bars along
the y-axis. A 1/f**2 line is also fit to the gamma rates to show the behavior.

@author: Aedan
"""

'''
nv1_2019_05_10

files used:
    
    
2019-06-14_09-48-56_116.0_MHz_splitting_rate_analysis

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
splitting_list = [29.8, 72.4]

omega_avg_list = [0.88, 0.8]

omega_error_list = [0.16, 0.6]

gamma_avg_list = [27, 21]

gamma_error_list = [4, 2]

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
ax.set_title('NV13_2019_06_10')
ax.legend()
