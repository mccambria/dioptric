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

def fit_eq_alpha(f, amp, offset):
    return amp*f**(-2) + offset


# %%

import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy

# The data
nv13_splitting_list = numpy.array([10.9,  23.1,  29.8, 51.9, 72.4, 112.9, 164.1, 256.2])
nv13_splitting_error = numpy.full(len(nv13_splitting_list), 0.3)*2
nv13_omega_avg_list = numpy.array([0.45,  1.01, 1.01, 0.39, 0.76, 0.92, 0.66, 0.23])
nv13_omega_error_list = numpy.array([0.06,  0.16,   0.09, 0.04, 0.1, 0.14, 0.11, 0.04])*2
nv13_gamma_avg_list = numpy.array([240,  62, 19.3, 17.7, 16.2, 12.1, 5.6, 2.1])
nv13_gamma_error_list = numpy.array([25,  8,   1.1, 1.4, 1.1, 0.9, 0.5, 0.3])*2

# Try to fit the gamma to a 1/f^2

fit_alpha_params, cov_arr = curve_fit(fit_eq_alpha, nv13_splitting_list, nv13_gamma_avg_list, 
                                p0 = [1000, 1], sigma = nv13_gamma_error_list,
                                absolute_sigma = True)

splitting_linspace = numpy.linspace(1, 2000,
                                    1000)
omega_constant_array = numpy.empty([1000]) 
omega_constant_array[:] = numpy.average(nv13_omega_avg_list)
print(numpy.average(nv13_omega_avg_list))
print( numpy.sqrt(numpy.sum(numpy.array(nv13_omega_error_list)**2)))


fig, ax = plt.subplots(1, 1, figsize=(10, 8))


axis_font = {'size':'14'}

orange = '#f7941d'
purple = '#87479b'

print(fit_alpha_params)

ax.set_xscale("log", nonposx='clip')
ax.set_yscale("log", nonposy='clip')
ax.errorbar(nv13_splitting_list, nv13_gamma_avg_list, yerr = nv13_gamma_error_list, 
             xerr = nv13_splitting_error, label = r'$\gamma$', fmt='o', markersize = 15, color=purple)
ax.errorbar(nv13_splitting_list, nv13_omega_avg_list, yerr = nv13_omega_error_list, 
             xerr = nv13_splitting_error, label = r'$\Omega$', fmt='^', markersize = 15, color=orange)

ax.plot(splitting_linspace, fit_eq_alpha(splitting_linspace, *fit_alpha_params), 
            linestyle='dashed', linewidth=3, color = purple)
ax.plot(splitting_linspace, omega_constant_array, color= orange,
            linestyle='dashed', linewidth=3)

# %% Residuals

#ax.plot(nv13_splitting_list, nv13_gamma_avg_list - fit_eq_alpha(nv13_splitting_list, *fit_alpha_params), color='b', label = r'$\gamma$ residual')
#ax.set_ylabel('Residuals', color='b', fontsize=18)
#ax.set_xlabel('Splitting (MHz)', color='k', fontsize=18)
#
##ax2 = ax.twinx()
#ax.plot(nv13_splitting_list, nv13_omega_avg_list - numpy.average(nv13_omega_avg_list), color = 'r', label = r'$\Omega$ residual')
##ax.set_ylabel('Omega Residuals', color='r', fontsize=18)
#ax.tick_params(which = 'both', length=6, width=2, colors='k',
#                grid_alpha=0.7, labelsize = 18)
#ax.set_ylim([-5,10])
##ax2.tick_params(which = 'major', length=12, width=2)
# %%


text = '\n'.join((r'$A_0/f^{2} + \gamma_\infty$ fit:',
#                  r'$\alpha = {} \pm {}$'.format('%.2f'%(fit_alpha_params[1]), '%.2f'%(numpy.sqrt(cov_arr[1][1]))),
                  r'$A_0 = {} \pm {}$'.format('%.0f'%(fit_alpha_params[0]), '%.0f'%(numpy.sqrt(cov_arr[0][0]))),
                  r'$\gamma_\infty = {} \pm {}$'.format('%.2f'%(fit_alpha_params[1]), '%.2f'%(numpy.sqrt(cov_arr[1][1])))
                  ))
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax.text(0.75, 0.8, text, transform=ax.transAxes, fontsize=12,
        verticalalignment='top', bbox=props)


# %%

ax.tick_params(which = 'both', length=6, width=2, colors='k',
                direction='in',grid_alpha=0.7, labelsize = 18)

ax.tick_params(which = 'major', length=12, width=2)

ax.grid()

ax.set_xlim([8,400])
ax.set_ylim([0.1,800])



plt.xlabel('Splitting (MHz)', fontsize=18)
plt.ylabel('Relaxation Rate (kHz)', fontsize=18)
#ax.legend(fontsize=18)

#fig.savefig("C:/Users/Aedan/Creative Cloud Files/Paper Illustrations/Magnetically Forbidden Rate/NV13.pdf", bbox_inches='tight')
