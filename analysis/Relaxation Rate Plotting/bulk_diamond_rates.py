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
NV0_opt_omega_avg_list = [0.057]
NV0_opt_omega_error_list = [0.010]
NV0_opt_gamma_avg_list = [0.139]
NV0_opt_gamma_error_list = [0.014]

NV0_mis_splitting_list = [128.1]
NV0_mis_omega_avg_list = [0.059]
NV0_mis_omega_error_list = [0.014]
NV0_mis_gamma_avg_list = [0.13]
NV0_mis_gamma_error_list = [0.02]

nv1_splitting_list = [129.7]
nv1_omega_avg_list = [0.058]
nv1_omega_error_list = [0.008]
nv1_gamma_avg_list = [0.11]
nv1_gamma_error_list = [0.02]

# The data of NV_0
nv0_splitting_list = [23.9, 125.9, 233.2]
nv0_omega_avg_list = [0.065, 0.057, 0.046]
nv0_omega_error_list = [0.013, 0.010, 0.008]
nv0_gamma_avg_list = [0.15, 0.139,0.13]
nv0_gamma_error_list = [0.03, 0.014, 0.02]

# Fig with just the magnet optimized data from NV0

fig_nv0, axes = plt.subplots(1, 2, figsize=(17, 8))

axis_font = {'size':'14'}

ax = axes[0]
ax.errorbar(nv0_splitting_list, nv0_gamma_avg_list, yerr = nv0_gamma_error_list, 
            label = r'$\gamma$', fmt='o', color='blue')
ax.errorbar(nv0_splitting_list, nv0_omega_avg_list, yerr = nv0_omega_error_list, 
            label = r'$\Omega$', fmt='o', color='orange')

ax.tick_params(which = 'both', length=6, width=2, colors='k',
                grid_alpha=0.7, labelsize = 18)

ax.tick_params(which = 'major', length=12, width=2)

ax.grid()
ax.legend(fontsize=18)

#ax.errorbar(nv0m_splitting_list, nv0m_gamma_avg_list, yerr = nv0m_gamma_error_list, 
#            label = r'$\gamma_{NV_1}$, magnet misaligned', markersize = 10, fmt='o', color='blue',  markerfacecolor='none')
#ax.errorbar(nv0m_splitting_list, nv0m_omega_avg_list, yerr = nv0m_omega_error_list, 
#            label = r'$\Omega_{NV_1}$, magnet misaligned', markersize = 10, fmt='o', color='orange',  markerfacecolor='none')

ax = axes[1]
ax.errorbar(nv1_splitting_list, nv1_gamma_avg_list, yerr = nv1_gamma_error_list, 
            label = r'$\gamma_{NV_2}$', fmt='^', markersize = 10, color='blue')
ax.errorbar(nv1_splitting_list, nv1_omega_avg_list, yerr = nv1_omega_error_list, 
            label = r'$\Omega_{NV_2}$', fmt='^', markersize = 10, color='orange')

ax.errorbar(NV0_mis_splitting_list, NV0_mis_gamma_avg_list, yerr = NV0_mis_gamma_error_list, 
            label = r'$\gamma_{NV_1}$, misaligned', markersize = 10, fmt='o', color='blue',  markerfacecolor='none')
ax.errorbar(NV0_mis_splitting_list, NV0_mis_omega_avg_list, yerr = NV0_mis_omega_error_list, 
            label = r'$\Omega_{NV_1}$, misaligned', markersize = 10, fmt='o', color='orange',  markerfacecolor='none')

ax.errorbar(NV0_opt_splitting_list, NV0_opt_gamma_avg_list, yerr = NV0_opt_gamma_error_list, 
            label = r'$\gamma_{NV_1}$, optimum magnet', markersize = 10, fmt='o', color='blue')
ax.errorbar(NV0_opt_splitting_list, NV0_opt_omega_avg_list, yerr = NV0_opt_omega_error_list, 
            label = r'$\Omega_{NV_1}$, optimum magnet', markersize = 10, fmt='o', color='orange')


ax.tick_params(which = 'both', length=6, width=2, colors='k',
                grid_alpha=0.7, labelsize = 18)

ax.tick_params(which = 'major', length=12, width=2)

ax.grid()
ax.legend(fontsize=18)

plt.xlabel('Splitting (MHz)', fontsize=18)
plt.ylabel('Relaxation Rate (kHz)', fontsize=18)
plt.title('Bulk Diamond', fontsize=18)

