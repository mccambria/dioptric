# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 14:36:23 2019

This file plots the relaxation rate data collected for the nv16_2019_07_25.

The data is input manually, and plotted on a loglog plot with error bars along
the y-axis. A 1/f**2 line is also fit to the gamma rates to show the behavior.

@author: Aedan
"""

'''
nv16_2019_07_25


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
nv16_splitting_list = [28.6, 81.2, 128.0, 283.7]
nv16_omega_avg_list = [0.55, 1.75, 0.69, 0.64]
nv16_omega_error_list = [0.08, 0.19, 0.10, 0.18]
nv16_gamma_avg_list = [94, 18.2, 12.1, 6.8]
nv16_gamma_error_list = [18, 1.0, 0.8, 0.4]

# Try to fit the gamma to a 1/f^alpha

fit_alpha_params, cov_arr = curve_fit(fit_eq_alpha, nv16_splitting_list, nv16_gamma_avg_list, 
                                p0 = (100, 1))

splitting_linspace = numpy.linspace(nv16_splitting_list[0], nv16_splitting_list[-1],
                                    1000)


fig, ax = plt.subplots(1, 1, figsize=(10, 8))


axis_font = {'size':'14'}

ax.set_xscale("log", nonposx='clip')
ax.set_yscale("log", nonposy='clip')
ax.errorbar(nv16_splitting_list, nv16_gamma_avg_list, yerr = nv16_gamma_error_list, 
            label = r'$\gamma$', fmt='o', color='blue')
ax.errorbar(nv16_splitting_list, nv16_omega_avg_list, yerr = nv16_omega_error_list, 
            label = r'$\Omega$', fmt='o', color='red')

#ax.plot(splitting_linspace, fit_eq_2(splitting_linspace, *fit_2_params), 
#            label = r'$f^{-2}$', color ='teal')
#ax.plot(splitting_linspace, fit_eq_1(splitting_linspace, *fit_1_params), 
#            label = r'$f^{-1}$', color = 'orange')

# %%

ax.plot(splitting_linspace, fit_eq_alpha(splitting_linspace, *fit_alpha_params), 
            label = r'$1/f^\alpha$')

text = '\n'.join((r'$1/f^{\alpha}$ fit:',
                  r'$\alpha = $' + '%.2f'%(fit_alpha_params[1])
#                  ,r'$A_0 = $' + '%.4f'%(fit_alpha_params[0])
                  ))

props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax.text(0.85, 0.7, text, transform=ax.transAxes, fontsize=12,
        verticalalignment='top', bbox=props)

# %%

ax.tick_params(which = 'both', length=6, width=2, colors='k',
                grid_alpha=0.7, labelsize = 18)

ax.tick_params(which = 'major', length=12, width=2)

ax.grid()

plt.xlabel('Splitting (MHz)', fontsize=18)
plt.ylabel('Relaxation Rate (kHz)', fontsize=18)
plt.title('NV16_2019_07_25', fontsize=18)
ax.legend(fontsize=18)
