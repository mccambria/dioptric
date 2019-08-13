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

def fit_eq_alpha(f, amp, alpha, offset):
    return amp*f**(-alpha)+offset

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
                                p0 = (100, 1, 0.4), sigma = nv2_gamma_error_list_all,
                                absolute_sigma = True)


splitting_linspace = numpy.linspace(nv2_splitting_list_all[0], nv2_splitting_list_all[-1],
                                    1000)

fig, ax = plt.subplots(1, 1, figsize=(10, 8))


ax.set_xscale("log", nonposx='clip')
ax.set_yscale("log", nonposy='clip')
ax.errorbar(nv2_splitting_list_all, nv2_gamma_avg_list_all, yerr = nv2_gamma_error_list_all, 
            label = r'$\gamma$', fmt='o', markersize = 10, color='blue')
ax.errorbar(nv2_splitting_list_all, nv2_omega_avg_list_all, yerr = nv2_omega_error_list_all, 
            label = r'$\Omega$', fmt='o', markersize = 10, color='red')

ax.plot(splitting_linspace, fit_eq_alpha(splitting_linspace, *fit_alpha_params), 
            label = 'fit', color ='blue')

#ax.set_xscale("log", nonposx='clip')
#ax.set_yscale("log", nonposy='clip')
#ax.errorbar(nv2_splitting_list, nv2_gamma_avg_list, yerr = nv2_gamma_error_list, 
#            label = r'$\gamma$ (past)', fmt='v', markersize = 10, color='blue')
#ax.errorbar(nv2_splitting_list_all, nv2_omega_avg_list_all, yerr = nv2_omega_error_list_all, 
#            label = r'$\Omega$ (past)', fmt='v', markersize = 10, color='red')

#ax.plot(splitting_linspace, fit_eq_alpha(splitting_linspace, *fit_alpha_params), 
#            label = 'fit (past)', color ='blue')


#ax.set_xscale("log", nonposx='clip')
#ax.set_yscale("log", nonposy='clip')
#ax.errorbar(nv2_splitting_list_2, nv2_gamma_avg_list_2, yerr = nv2_gamma_error_list_2, 
#            label = r'$\gamma$ (recent)', fmt='o', markersize = 10, color='blue')
#ax.errorbar(nv2_splitting_list_all, nv2_omega_avg_list_all, yerr = nv2_omega_error_list_all, 
#            label = r'$\Omega$ (recent)', fmt='o', markersize = 10, color='red')

#ax.plot(splitting_linspace, fit_eq_alpha(splitting_linspace, *fit_alpha_params), 
#            label = 'fit', color ='blue')



# %%


text_1 = '\n'.join((r'$A_0/f^{\alpha} + \gamma_\infty$ fit:',
                  r'$\alpha = $' + '%.2f'%(fit_alpha_params[1]),
#                  r'$\alpha_{recent} = $' + '%.2f'%(fit_alpha_params_2[1])
                  r'$A_0 = $' + '%.0f'%(fit_alpha_params[0])
                  ,r'$\gamma_\infty = $' + '%.2f'%(fit_alpha_params[2])
                  ))

props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax.text(0.65, 0.95, text_1, transform=ax.transAxes, fontsize=12,
        verticalalignment='top', bbox=props)

# %%

ax.tick_params(which = 'both', length=6, width=2, colors='k',
                grid_alpha=0.7, labelsize = 18)

ax.tick_params(which = 'major', length=12, width=2)

ax.grid()

plt.xlabel('Splitting (MHz)', fontsize=18)
plt.ylabel('Relaxation Rate (kHz)', fontsize=18)
plt.title(r'NV2', fontsize=18)
ax.legend(fontsize=18)
