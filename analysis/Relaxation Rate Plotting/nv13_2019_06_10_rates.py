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


'''
# %%
def fit_eq(f, amp):
    return amp*f**(-1)

# %%

import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy

# The data
nv13_splitting_list = [23.1, 29.8, 51.9, 72.4, 112.9, 164.1]
nv13_omega_avg_list = [0.42, 0.9, 1.2, 0.8, 1.4, 1.2]
nv13_omega_error_list = [0.12, 0.4, 0.6, 0.4, 0.3, 0.2]
nv13_gamma_avg_list = [90, 25, 30, 22, 13, 5.8]
nv13_gamma_error_list = [30, 5, 5, 3, 3, 0.5]

# Try to fit the gamma to a 1/f^2

fit_params, cov_arr = curve_fit(fit_eq, nv13_splitting_list, nv13_gamma_avg_list, 
                                p0 = 100)

splitting_linspace = numpy.linspace(nv13_splitting_list[0], nv13_splitting_list[-1],
                                    1000)


fig, ax = plt.subplots(1, 1, figsize=(10, 8))

#ax.errorbar(splitting_list, omega_avg_list, yerr = omega_error_list)
ax.set_xscale("log", nonposx='clip')
ax.set_yscale("log", nonposy='clip')
ax.errorbar(nv13_splitting_list, nv13_gamma_avg_list, yerr = nv13_gamma_error_list, 
            label = 'Gamma', fmt='o', color='blue')
ax.errorbar(nv13_splitting_list, nv13_omega_avg_list, yerr = nv13_omega_error_list, 
            label = 'Omega', fmt='o', color='red')
ax.plot(splitting_linspace, fit_eq(splitting_linspace, *fit_params), 
            label = '1/f')
ax.grid()

ax.set_xlabel('Splitting (MHz)')
ax.set_ylabel('Relaxation Rate (kHz)')
ax.set_title('NV13_2019_06_10')
ax.legend()
