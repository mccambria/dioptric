# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 14:36:23 2019

This file plots the relaxation rate data collected for the nv0_2019_06_27.

@author: Aedan
"""

'''
nv0_2019_06_27


'''
# %%
def fit_eq_alpha(f, amp, offset):
    return amp*f**(-2) + offset

# %%

import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy

# The data
splitting_list = [23.4, 26.2, 36.2, 48.1, 60.5, 92.3, 150.8, 329.6, 884.9, 1080.5, 1148.4]
omega_avg_list = numpy.array([0.283, 0.33,0.32,  0.314, 0.24, 0.253, 0.29, 0.33, 0.29, 0.28, 0.38])
omega_error_list = numpy.array([0.017, 0.03,0.03,  0.01, 0.02, 0.012, 0.02, 0.02, 0.02, 0.05, 0.04])*2
gamma_avg_list = numpy.array([	34.5, 29.0, 20.4,  15.8, 9.1, 6.4, 4.08, 1.23, 0.45, 0.69, 0.35])
gamma_error_list = numpy.array([1.3, 1.1, 0.5, 0.3, 0.3, 0.1, 0.15, 0.07, 0.03, 0.12, 0.03])*2

# Try to fit the gamma to a 1/f^2

fit_alpha_params, cov_arr = curve_fit(fit_eq_alpha, splitting_list, gamma_avg_list, 
                                p0 = (100, 0.1), sigma = gamma_error_list,
                                absolute_sigma = True)

splitting_linspace = numpy.linspace(10, 2000,
                                    1000)

omega_constant_array = numpy.empty([1000]) 
omega_constant_array[:] = numpy.average(omega_avg_list)
print(numpy.average(omega_avg_list))
print( numpy.sqrt(numpy.sum(numpy.array(omega_error_list)**2)))

fig, ax = plt.subplots(1, 1, figsize=(10, 8))


axis_font = {'size':'14'}

orange = '#f7941d'
purple = '#87479b'

ax.set_xscale("log", nonposx='clip')
ax.set_yscale("log", nonposy='clip')
ax.errorbar(splitting_list, gamma_avg_list, yerr = gamma_error_list, 
            label = r'$\gamma$',  fmt='o',markersize = 12, color = purple)
ax.errorbar(splitting_list, omega_avg_list, yerr = omega_error_list, 
            label = r'$\Omega$', fmt='^', markersize = 12, color=orange)


# %%

ax.plot(splitting_linspace, fit_eq_alpha(splitting_linspace, *fit_alpha_params), 
             label = r'fit',linestyle='dashed', linewidth=3, color =purple)
ax.plot(splitting_linspace, omega_constant_array, color = orange,
            label = r'$\Omega$', linestyle='dashed', linewidth=3)

text = '\n'.join((r'$A_0/f^{2} + \gamma_\infty$ fit:',
#                  r'$\alpha = {} \pm {}$'.format('%.2f'%(fit_alpha_params[1]), '%.2f'%(numpy.sqrt(cov_arr[1][1]))),
                  r'$A_0 = {} \pm {}$'.format('%.0f'%(fit_alpha_params[0]), '%.0f'%(numpy.sqrt(cov_arr[0][0]))),
                  r'$\gamma_\infty = {} \pm {}$'.format('%.2f'%(fit_alpha_params[1]), '%.2f'%(numpy.sqrt(cov_arr[1][1])))
#                  r'$\chi^2 = $' + '%.2f'%(chi_sq)
                                           
                  ))
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
#ax.text(0.85, 0.7, text, transform=ax.transAxes, fontsize=12,
#        verticalalignment='top', bbox=props)

# %%

ax.tick_params(which = 'both', length=6, width=2, colors='k',
                grid_alpha=0.7, labelsize = 18)

ax.tick_params(which = 'major', length=12, width=2)

ax.grid()

ax.set_xlim([10,1600])
ax.set_ylim([0.1,300])

plt.xlabel('Splitting (MHz)', fontsize=18)
plt.ylabel('Relaxation Rate (kHz)', fontsize=18)
#plt.title('NV 0', fontsize=18)
#ax.legend(fontsize=18)
fig.canvas.draw()
fig.canvas.flush_events()

fig.savefig("C:/Users/Aedan/Creative Cloud Files/Paper Illustrations/Magnetically Forbidden Rate/fig_3d.pdf", bbox_inches='tight')
