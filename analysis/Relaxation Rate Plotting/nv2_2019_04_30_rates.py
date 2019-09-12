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
def fit_eq_alpha(f, amp, offset):
    return amp*f**(-2) + offset

# %%

import matplotlib.pyplot as plt
from scipy import asarray as ar, exp
from scipy.optimize import curve_fit
import numpy
from scipy.stats import chisquare

# The data
nv2_splitting_list = [29.1, 44.8, 56.2, 56.9, 69.8, 85.1, 101.6]
nv2_omega_avg_list = [0.37, 0.52, 0.33, 0.41, 0.33, 0.32, 0.27]
nv2_omega_error_list = [0.06, 0.11, 0.07, 0.06, 0.06, 0.04, 0.04]
nv2_gamma_avg_list = [20.8, 7.2, 3.9, 3.9, 2.46, 2.9, 1.6]
nv2_gamma_error_list = [0.9, 0.3, 0.2, 0.2, 0.14, 0.2, 0.2]

# Data for the second round of measurements
nv2_splitting_list_2 = [29.2, 45.5, 85.2, 280.4,697.4]
nv2_omega_avg_list_2 = [0.34, 0.25, 0.35, 0.30, 0.34]
nv2_omega_error_list_2 = [0.02, 0.03, 0.02, 0.03, 0.07]
nv2_gamma_avg_list_2 = [34.3, 9.7, 3.18, 0.56, 0.68]
nv2_gamma_error_list_2 = [1.0, 0.2, 0.12, 0.06, 0.09]

nv2_splitting_list_all = nv2_splitting_list + nv2_splitting_list_2
nv2_omega_avg_list_all = nv2_omega_avg_list + nv2_omega_avg_list_2
nv2_omega_error_list_all = nv2_omega_error_list + nv2_omega_error_list_2
nv2_gamma_avg_list_all = nv2_gamma_avg_list + nv2_gamma_avg_list_2
nv2_gamma_error_list_all = nv2_gamma_error_list + nv2_gamma_error_list_2

# %% Seperate analysis of data

#combine all data, omega limited by electric field noise

# Fit the gamma to a 1/f^alpha

fit_alpha_params, cov_arr = curve_fit(fit_eq_alpha, nv2_splitting_list_all, nv2_gamma_avg_list_all, 
                                p0 = (100, 1), sigma = nv2_gamma_error_list_all,
                                absolute_sigma = True)



splitting_linspace = numpy.linspace(10, 2000,
                                    1000)
omega_constant_array = numpy.empty([1000]) 
omega_constant_array[:] = numpy.average(nv2_omega_avg_list_all)

fig, ax = plt.subplots(1, 1, figsize=(10, 8))

orange = '#f7941d'
purple = '#87479b'

print(fit_alpha_params)

ax.set_xscale("log", nonposx='clip')
ax.set_yscale("log", nonposy='clip')
ax.errorbar(nv2_splitting_list_all, nv2_gamma_avg_list_all, yerr = nv2_gamma_error_list_all, 
            label = r'$\gamma$', fmt='o',markersize = 12, color = purple)
ax.errorbar(nv2_splitting_list_all, nv2_omega_avg_list_all, yerr = nv2_omega_error_list_all, 
            label = r'$\Omega$', fmt='^', markersize = 12, color=orange)

ax.plot(splitting_linspace, fit_eq_alpha(splitting_linspace, *fit_alpha_params), 
            linestyle='dashed', linewidth=3, color =purple)
ax.plot(splitting_linspace, omega_constant_array, color = orange,
            linestyle='dashed', linewidth=3)

# %% Chi Squared

expected = []

for el in range(len(nv2_splitting_list)):
    expected_value = fit_eq_alpha(nv2_splitting_list[el], *fit_alpha_params)
    expected.append(expected_value)
    
ret_vals = chisquare(nv2_gamma_avg_list, f_exp=expected)
chi_sq = ret_vals[0]

# %%


text = '\n'.join((r'$A_0/f^{2} + \gamma_\infty$ fit:',
#                  r'$\alpha = {} \pm {}$'.format('%.2f'%(fit_alpha_params[1]), '%.2f'%(numpy.sqrt(cov_arr[1][1]))),
                  r'$A_0 = {} \pm {}$'.format('%.0f'%(fit_alpha_params[0]), '%.0f'%(numpy.sqrt(cov_arr[0][0]))),
                  r'$\gamma_\infty = {} \pm {}$'.format('%.2f'%(fit_alpha_params[1]), '%.2f'%(numpy.sqrt(cov_arr[1][1]))),
                  r'$\chi^2 = $' + '%.2f'%(chi_sq)
                  ))
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax.text(0.75, 0.8, text, transform=ax.transAxes, fontsize=12,
        verticalalignment='top', bbox=props)

# %%

ax.tick_params(which = 'both', length=6, width=2, colors='k',
                grid_alpha=0.7, labelsize = 18)

ax.tick_params(which = 'major', length=12, width=2)

ax.grid()

ax.set_xlim([10,1200])
ax.set_ylim([0.1,300])

#ax.set_ylim([-5,40])

plt.xlabel('Splitting (MHz)', fontsize=18)
plt.ylabel('Relaxation Rate (kHz)', fontsize=18)

plt.title(r'NV2', fontsize=18)
ax.legend(fontsize=18)

#fig.savefig("C:/Users/Aedan/Creative Cloud Files/Paper Illustrations/Magnetically Forbidden Rate/fig_3a.pdf", bbox_inches='tight')

