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

# nv1_2019_05_10
nv1_splitting_list = [19.8, 28, 32.7, 51.8, 97.8, 116]
nv1_omega_avg_list = [1.3, 1.7, 1.50, 2.3, 1.7, 1.21]
nv1_omega_error_list = [0.3, 0.4, 0.11, 0.5, 0.2, 0.13]
nv1_gamma_avg_list = [135, 71, 50, 13.1, 3.5, 4.6]
nv1_gamma_error_list = [11, 7, 3, 0.7, 0.2, 0.3]

# nv2_2019_04_30
nv2_splitting_list = [29.1, 44.8, 56.2, 56.9, 69.8, 85.1, 101.6]
nv2_omega_avg_list = [0.39, 0.51, 0.32, 0.41, 0.33, 0.33, 0.27]
nv2_omega_error_list = [0.09, 0.12, 0.08, 0.09, 0.06, 0.06, 0.04]
nv2_gamma_avg_list = [20.5, 7.2, 3.9, 3.9, 2.48, 3.0, 1.6]
nv2_gamma_error_list = [0.9, 0.4, 0.3, 0.2, 0.17, 0.3, 0.2]

# nv13_2019_06_10
nv13_splitting_list = [29.8]
nv13_omega_avg_list = [0.88]
nv13_omega_error_list = [0.16]
nv13_gamma_avg_list = [27]
nv13_gamma_error_list = [4]

# nv0_2019_06_06
nv0_splitting_list = [48.1]
nv0_omega_avg_list = [0.47]
nv0_omega_error_list = [0.16]
nv0_gamma_avg_list = [17.2]
nv0_gamma_error_list = [0.8]

# nv4_2019_06_06
nv4_splitting_list = [28.5]
nv4_omega_avg_list = [1.4]
nv4_omega_error_list = [0.3]
nv4_gamma_avg_list = [200]
nv4_gamma_error_list = [22]

fig, ax = plt.subplots(1, 1, figsize=(10, 8))

#ax.errorbar(splitting_list, omega_avg_list, yerr = omega_error_list)
ax.set_xscale("log", nonposx='clip')
ax.set_yscale("log", nonposy='clip')
ax.errorbar(nv1_splitting_list, nv1_gamma_avg_list, yerr = nv1_gamma_error_list, 
            label = 'nv1_2019_05_10', fmt='o', color='blue')
#ax.errorbar(nv1_splitting_list, nv1_omega_avg_list, yerr = nv1_omega_error_list, 
#            label = 'nv1 Omega', fmt='o', color='red')

ax.errorbar(nv2_splitting_list, nv2_gamma_avg_list, yerr = nv2_gamma_error_list, 
            label = 'nv2_2019_04_30', fmt='o', color='purple')
#ax.errorbar(nv2_splitting_list, nv2_omega_avg_list, yerr = nv2_omega_error_list, 
#            label = 'nv2 Omega', fmt='o', color='orange')

ax.errorbar(nv13_splitting_list, nv13_gamma_avg_list, yerr = nv13_gamma_error_list, 
            label = 'nv13_2019_06_10', fmt='o', color='green')
#ax.errorbar(nv13_splitting_list, nv13_omega_avg_list, yerr = nv13_omega_error_list, 
#            label = 'nv13 Omega', fmt='o', color='red')

ax.errorbar(nv0_splitting_list, nv0_gamma_avg_list, yerr = nv0_gamma_error_list, 
            label = 'nv0_2019_06_06', fmt='o', color='orange')
#ax.errorbar(nv0_splitting_list, nv0_omega_avg_list, yerr = nv0_omega_error_list, 
#            label = 'nv0 Omega', fmt='o', color='red')

ax.errorbar(nv4_splitting_list, nv4_gamma_avg_list, yerr = nv4_gamma_error_list, 
            label = 'nv4_2019_06_06', fmt='o', color='pink')
#ax.errorbar(nv4_splitting_list, nv4_omega_avg_list, yerr = nv4_omega_error_list, 
#            label = 'nv4 Omega', fmt='o', color='red')

ax.grid()

ax.set_xlabel('Splitting (MHz)')
ax.set_ylabel('Relaxation Rate (kHz)')
ax.set_title('Double quantum relaxation rates')
ax.legend()
