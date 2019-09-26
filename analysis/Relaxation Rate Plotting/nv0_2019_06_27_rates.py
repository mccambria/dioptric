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
def fit_eq_alpha(f, amp, alpha):
    return amp*f**(-alpha)

# %%

import matplotlib.pyplot as plt
#from scipy import asarray as ar, exp
#from scipy.optimize import curve_fit
#import numpy

# The data
sample_name = 'NV 0'
splitting_list = [23.9, 125.9, 128.1, 233.2]
omega_avg_list = [0.063, 0.053, 0.059, 0.061]
omega_error_list = [0.009, 0.003, 0.006, 0.006]
gamma_avg_list = [0.127, 0.111, 0.144, 0.132]
gamma_error_list = [0.023, 0.009, 0.025, 0.017]

# sample_name = 'NV 1'
# splitting_list = [129.7]
# omega_avg_list = [0.060]
# omega_error_list = [0.004]
# gamma_avg_list = [0.114]
# gamma_error_list = [0.012]

# Try to fit the gamma to a 1/f^2

#fit_1_params, cov_arr = curve_fit(fit_eq_1, nv1_splitting_list, nv1_gamma_avg_list, 
#                                p0 = 100)
#
#fit_2_params, cov_arr = curve_fit(fit_eq_2, nv1_splitting_list, nv1_gamma_avg_list, 
#                                p0 = 100)
#
#fit_alpha_params, cov_arr = curve_fit(fit_eq_alpha, nv1_splitting_list, nv1_gamma_avg_list, 
#                                p0 = (100, 1))
#
#splitting_linspace = numpy.linspace(nv1_splitting_list[0], nv1_splitting_list[-1],
#                                    1000)


font = {'size': 14}
plt.rc('font', **font)
fig, ax = plt.subplots(figsize=(8.5, 8.5))
fig.set_tight_layout(True)

#ax.errorbar(splitting_list, omega_avg_list, yerr = omega_error_list)

axis_font = {'size':'14'}

# ax.set_xscale("log", nonposx='clip')
# ax.set_yscale("log", nonposy='clip')
ax.errorbar(splitting_list, gamma_avg_list, yerr = gamma_error_list,
            label = r'$\gamma$', fmt='o', color='blue')
ax.errorbar(splitting_list, omega_avg_list, yerr = omega_error_list,
            label = r'$\Omega$', fmt='o', color='red')
# ax.set_xlim(left=0.0)
ax.set_ylim(bottom=0.0)

#ax.plot(splitting_linspace, fit_eq_2(splitting_linspace, *fit_2_params), 
#            label = r'$f^{-2}$', color ='teal')
#ax.plot(splitting_linspace, fit_eq_1(splitting_linspace, *fit_1_params), 
#            label = r'$f^{-1}$', color = 'orange')

# %%

#ax.plot(splitting_linspace, fit_eq_alpha(splitting_linspace, *fit_alpha_params), 
#            label = r'$1/f^\alpha$')

#text = '\n'.join((r'$1/f^{\alpha}$ fit:',
#                  r'$\alpha = $' + '%.2f'%(fit_alpha_params[1])
#                  r'$A_0 = $' + '%.4f'%(fit_params[1] * 10**6) + ' kHz'
#                  ,r'$a = $' + '%.2f'%(fit_params[2])
#                  ))

#props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
#ax.text(0.85, 0.8, text, transform=ax.transAxes, fontsize=12,
#        verticalalignment='top', bbox=props)

# %%

ax.tick_params(which = 'both', length=6, width=2, colors='k', grid_alpha=0.7)

ax.tick_params(which = 'major', length=12, width=2)

ax.grid()

ax.set_ylabel('Relaxation Rate (kHz)')
ax.set_xlabel('Splitting (MHz)')
ax.set_title('Bulk Diamond, {}'.format(sample_name))
ax.legend()
