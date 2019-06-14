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
splitting_list = [19.8, 32.7, 51.8, 97.8]

omega_avg_list = [1.3, 1.50, 2.3, 1.7]

omega_error_list = [0.3, 0.11, 0.5, 0.2]

gamma_avg_list = [135, 50, 13.1, 3.5]

gamma_error_list = [11, 3, 0.7, 0.2]

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
ax.set_title('NV1_2019_05_10')
ax.legend()
