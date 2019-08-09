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

'''
# %%
def fit_eq_1(f, amp):
    return amp*f**(-1)

def fit_eq_2(f, amp):
    return amp*f**(-2)

def fit_eq_alpha(f, amp, alpha):
    return amp*f**(-alpha)

# %%

import matplotlib.pyplot as plt
from scipy import asarray as ar, exp
from scipy.optimize import curve_fit
import numpy

# The data
nv2_splitting_list = [29.1, 44.8, 56.2, 56.9, 69.8, 85.1, 101.6]
nv2_omega_avg_list = [0.37, 0.52, 0.33, 0.41, 0.33, 0.32, 0.27]
nv2_omega_error_list = [0.06, 0.11, 0.07, 0.06, 0.06, 0.04, 0.04]
nv2_gamma_avg_list = [20.8, 7.2, 3.9, 3.9, 2.46, 2.9, 1.6]
nv2_gamma_error_list = [0.9, 0.3, 0.2, 0.2, 0.14, 0.2, 0.2]

# Data for the second round of measurements
nv2_splitting_list_2 = [45.5, 85.2, 280.4]
nv2_omega_avg_list_2 = [0.25, 0.35, 0.275]
nv2_omega_error_list_2 = [0.03, 0.02, 0.008]
nv2_gamma_avg_list_2 = [9.7, 3.18, 0.441]
nv2_gamma_error_list_2 = [0.2, 0.12, 0.015]

# Fit the gamma to a 1/f^alpha

fit_alpha_params, cov_arr = curve_fit(fit_eq_alpha, nv2_splitting_list, nv2_gamma_avg_list, 
                                p0 = (100, 1), sigma = nv2_gamma_error_list,
                                absolute_sigma = True)

fit_alpha_params_2, cov_arr = curve_fit(fit_eq_alpha, nv2_splitting_list_2, nv2_gamma_avg_list_2, 
                                p0 = (100, 1), sigma = nv2_gamma_error_list_2,
                                absolute_sigma = True)

splitting_linspace_1 = numpy.linspace(nv2_splitting_list[0], nv2_splitting_list[-1],
                                    1000)

splitting_linspace_2 = numpy.linspace(nv2_splitting_list_2[0], nv2_splitting_list_2[-1],
                                    1000)


fig, ax = plt.subplots(1, 1, figsize=(10, 8))

#ax.errorbar(splitting_list, omega_avg_list, yerr = omega_error_list)
ax.set_xscale("log", nonposx='clip')
ax.set_yscale("log", nonposy='clip')
#ax.errorbar(nv2_splitting_list, nv2_gamma_avg_list, yerr = nv2_gamma_error_list, 
#            label = r'$\gamma$', fmt='o', markersize = 10, color='blue')
ax.errorbar(nv2_splitting_list_2, nv2_gamma_avg_list_2, yerr = nv2_gamma_error_list_2, 
            label =  r'$\gamma$', fmt='v', markersize = 10, color='orange')
#ax.errorbar(nv2_splitting_list, nv2_omega_avg_list, yerr = nv2_omega_error_list, 
#            label = r'$\Omega$', fmt='o', markersize = 10, color='red')
ax.errorbar(nv2_splitting_list_2, nv2_omega_avg_list_2, yerr = nv2_omega_error_list_2, 
            label = r'$\Omega$', fmt='o', markersize = 10, color='purple')
#ax.plot(splitting_linspace_1, fit_eq_alpha(splitting_linspace_1, *fit_alpha_params), 
#            label = 'fit', color ='blue')

ax.plot(splitting_linspace_2, fit_eq_alpha(splitting_linspace_2, *fit_alpha_params_2), 
            label = 'fit', color = 'orange')

# %%

#ax.plot(splitting_linspace, fit_eq_alpha(splitting_linspace, *fit_alpha_params), 
#            label = r'$1/f^\alpha$')
#
text_1 = '\n'.join((r'$1/f^{\alpha}$ fit:',
                  r'$\alpha = $' + '%.2f'%(fit_alpha_params_2[1]),
#                  r'$\alpha_{recent} = $' + '%.2f'%(fit_alpha_params_2[1])
                  r'$A_0 = $' + '%.0f'%(fit_alpha_params_2[0])
#                  ,r'$a = $' + '%.2f'%(fit_params[2])
                  ))
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax.text(0.85, 0.7, text_1, transform=ax.transAxes, fontsize=12,
        verticalalignment='top', bbox=props)

# %%

ax.tick_params(which = 'both', length=6, width=2, colors='k',
                grid_alpha=0.7, labelsize = 18)

ax.tick_params(which = 'major', length=12, width=2)

ax.grid()

plt.xlabel('Splitting (MHz)', fontsize=18)
plt.ylabel('Relaxation Rate (kHz)', fontsize=18)
plt.title(r'NV2 (data taken during August, 2019)', fontsize=18)
ax.legend(fontsize=18)
