# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 14:36:23 2019

This file plots the relaxation rate data collected for the bulk diamond rates.

I think it'd be best to just plot the three middle points (two different NVs
and one misaligned) to show that they all agree. And then plot just the same
data from NV0 of the three points with the magnet completely aligned.

@author: Aedan
"""

'''
nv1_2019_05_10


'''
# %%
def fit_eq_alpha(f, amp, alpha):
    return amp*f**(-alpha)

# %%

import matplotlib.pyplot as plt
#from scipy import asarray as ar, exp
#from scipy.optimize import curve_fit
#import numpy

# The data at around one splitting
NV0_opt_splitting_list = [125.9]
NV0_opt_omega_avg_list = [0.053]
NV0_opt_omega_error_list = [0.003]
NV0_opt_gamma_avg_list = [0.111]
NV0_opt_gamma_error_list = [0.009]

NV0_mis_splitting_list = [128.1]
NV0_mis_omega_avg_list = [0.059]
NV0_mis_omega_error_list = [0.006]
NV0_mis_gamma_avg_list = [0.144]
NV0_mis_gamma_error_list = [0.025]

nv1_splitting_list = [129.7]
nv1_omega_avg_list = [0.060]
nv1_omega_error_list = [0.004]
nv1_gamma_avg_list = [0.114]
nv1_gamma_error_list = [0.012]

# The data of NV_0
nv0_splitting_list = [23.9, 125.9, 233.2]
nv0_omega_avg_list = [0.063, 0.053, 0.061]
nv0_omega_error_list = [0.009, 0.003, 0.006]
nv0_gamma_avg_list = [0.127, 0.111, 0.132]
nv0_gamma_error_list = [0.023, 0.009, 0.017]

# Fig with just the magnet optimized data from NV0

fig_nv0, ax = plt.subplots(1, 1, figsize=(5, 4))

axis_font = {'size':'14'}

#ax = axes[0]

## Data for the single NV at different splittings
#ax.errorbar(nv0_splitting_list, nv0_gamma_avg_list, yerr = nv0_gamma_error_list, 
#            label = r'$\gamma$',  markersize = 10, fmt='o', color='blue')
#ax.errorbar(nv0_splitting_list, nv0_omega_avg_list, yerr = nv0_omega_error_list, 
#            label = r'$\Omega$',  markersize = 10, fmt='o', color='orange')
#
## Formatting
#ax.tick_params(which = 'both', length=6, width=2, colors='k',
#                grid_alpha=0.7, labelsize = 18)
#ax.tick_params(which = 'major', length=12, width=2)
#ax.grid()
#ax.set_xlabel('Splitting (MHz)', fontsize=18)
#ax.set_ylabel('Relaxation Rate (kHz)', fontsize=18)
#ax.legend(fontsize=18)


# %%
#ax = axes[1]

# Data for a second NV
ax.errorbar(nv1_splitting_list, nv1_gamma_avg_list, yerr = nv1_gamma_error_list, 
            label = r'$\gamma_{NV_2}$', fmt='^', markersize = 10, color='blue')
ax.errorbar(nv1_splitting_list, nv1_omega_avg_list, yerr = nv1_omega_error_list, 
            label = r'$\Omega_{NV_2}$', fmt='^', markersize = 10, color='orange')

# Data for the original Nv, with misaligned magnet
ax.errorbar(NV0_mis_splitting_list, NV0_mis_gamma_avg_list, yerr = NV0_mis_gamma_error_list, 
            label = r'$\gamma_{NV_1}$, misaligned', markersize = 10, fmt='o', color='blue',  markerfacecolor='none')
ax.errorbar(NV0_mis_splitting_list, NV0_mis_omega_avg_list, yerr = NV0_mis_omega_error_list, 
            label = r'$\Omega_{NV_1}$, misaligned', markersize = 10, fmt='o', color='orange',  markerfacecolor='none')

# Data for the original NV, with optimized magnet
ax.errorbar(NV0_opt_splitting_list, NV0_opt_gamma_avg_list, yerr = NV0_opt_gamma_error_list, 
            label = r'$\gamma_{NV_1}$, optimized', markersize = 10, fmt='o', color='blue')
ax.errorbar(NV0_opt_splitting_list, NV0_opt_omega_avg_list, yerr = NV0_opt_omega_error_list, 
            label = r'$\Omega_{NV_1}$, optimized', markersize = 10, fmt='o', color='orange')


# NV0, alignd magnet data
#ax.errorbar(nv0_splitting_list, nv0_gamma_avg_list, yerr = nv0_gamma_error_list, 
#            label = r'$\gamma_{NV_1}$, optimum magnet', markersize = 10, fmt='o', color='blue')
#ax.errorbar(nv0_splitting_list, nv0_omega_avg_list, yerr = nv0_omega_error_list, 
#            label = r'$\Omega_{NV_1}$, optimum magnet', markersize = 10, fmt='o', color='orange')

# Formatting
ax.tick_params(which = 'both', length=6, width=2, colors='k',
                grid_alpha=0.7, labelsize = 18)
ax.tick_params(which = 'major', length=12, width=2)
ax.grid()
ax.set_xlabel('Splitting (MHz)', fontsize=18)
ax.set_ylabel('Relaxation Rate (kHz)', fontsize=18)
#ax.legend(fontsize=18)
##ax.title('Bulk Diamond', fontsize=18)

