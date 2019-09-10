# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 14:36:23 2019

This file plots the relaxation rate data collected for the nv0_2019_09_09.

@author: Aedan
"""

'''
nv0_2019_09_09


'''
# %%
def fit_eq_alpha(f, amp, alpha):
    return amp*f**(-alpha)

# %%

import matplotlib.pyplot as plt
from scipy import asarray as ar, exp
from scipy.optimize import curve_fit
import numpy

# The data
splitting_list = [48.1, 92.3]
omega_avg_list = [0.45, 0.24]
omega_error_list = [0.14, 0.03]
gamma_avg_list = [17.5, 6.7]
gamma_error_list = [0.7, 0.2]

# Try to fit the gamma to a 1/f^alpha

fit_alpha_params, cov_arr = curve_fit(fit_eq_alpha, splitting_list, gamma_avg_list, 
                                p0 = [1000, 2], sigma = gamma_error_list,
                                absolute_sigma = True)

splitting_linspace = numpy.linspace(10, 2000,
                                    1000)
omega_constant_array = numpy.empty([1000]) 
omega_constant_array[:] = numpy.average(omega_avg_list)

fig, ax = plt.subplots(1, 1, figsize=(10, 8))


axis_font = {'size':'14'}

orange = '#f7941d'
purple = '#87479b'

print(fit_alpha_params)

ax.set_xscale("log", nonposx='clip')
ax.set_yscale("log", nonposy='clip')
ax.errorbar(splitting_list, gamma_avg_list, yerr = gamma_error_list, 
            label = r'$\gamma$', fmt='o', markersize = 12, color=purple)
ax.errorbar(splitting_list, omega_avg_list, yerr = omega_error_list, 
            label = r'$\Omega$', fmt='^', markersize = 12, color=orange)
ax.plot(splitting_linspace, omega_constant_array, color= orange,
            linestyle='dashed', linewidth=3, label = r'$\Omega$')


#ax.plot(splitting_linspace, fit_eq_2(splitting_linspace, *fit_2_params), 
#            label = r'$f^{-2}$', color ='teal')
#ax.plot(splitting_linspace, fit_eq_1(splitting_linspace, *fit_1_params), 
#            label = r'$f^{-1}$', color = 'orange')

# %%

ax.plot(splitting_linspace, fit_eq_alpha(splitting_linspace, *fit_alpha_params), 
            linestyle='dashed', linewidth=3, color = purple, label = 'fit')

text = '\n'.join((r'$A_0/f^{\alpha}$ fit:',
                  r'$\alpha = $' + '%.2f'%(fit_alpha_params[1]),
                  r'$A_0 = $' + '%.0f'%(fit_alpha_params[0])
#                  r'$\gamma_\infty = $' + '%.2f'%(fit_alpha_params[2])
                  ))
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax.text(0.85, 0.7, text, transform=ax.transAxes, fontsize=12,
        verticalalignment='top', bbox=props)

# %%

ax.tick_params(which = 'both', length=6, width=2, colors='k',
                grid_alpha=0.7, labelsize = 18)

ax.tick_params(which = 'major', length=12, width=2)

ax.grid()

ax.set_xlim([10,1200])
ax.set_ylim([0.1,300])

plt.xlabel('Splitting (MHz)', fontsize=18)
plt.ylabel('Relaxation Rate (kHz)', fontsize=18)
plt.title('NV0', fontsize=18)
#ax.legend(fontsize=18)
#fig.savefig("C:/Users/Aedan/Creative Cloud Files/Paper Illustrations/Magnetically Forbidden Rate/fig_3b.pdf", bbox_inches='tight')
