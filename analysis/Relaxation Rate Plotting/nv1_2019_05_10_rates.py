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

import numpy
import matplotlib.pyplot as plt
from scipy import asarray as ar, exp
from scipy.optimize import curve_fit
from scipy.stats import chisquare

# The data
nv1_splitting_list = [ 27.7, 28.9, 41.9, 32.7, 51.8, 97.8, 116, 268, 563.6, 1016.8]
nv1_omega_avg_list = [ 1.30,  1.000, 1.11, 1.42, 1.85, 1.41, 1.18, 1.04, 1.19, 0.58]
nv1_omega_error_list = numpy.array([ 0.06, 0.016, 0.05, 0.05, 0.08, 0.05, 0.06, 0.04, 0.06, 0.03])*2
nv1_gamma_avg_list = [64.5, 56.4, 23.5, 42.6, 13.1, 3.91, 4.67, 1.98, 0.70, 0.41]
nv1_gamma_error_list = numpy.array([1.4, 1.3, 0.5, 0.9, 0.2, 0.1, 0.11, 0.1, 0.05, 0.05])*2

# Try to fit the gamma to a 1/f^2

fit_alpha_params, cov_arr = curve_fit(fit_eq_alpha, nv1_splitting_list, nv1_gamma_avg_list, 
                                p0 = (100, 0.66), sigma = nv1_gamma_error_list,
                                absolute_sigma = True)

splitting_linspace = numpy.linspace(10, 2000,
                                    1000)

omega_constant_array = numpy.empty([1000]) 
omega_constant_array[:] = numpy.average(nv1_omega_avg_list)

print(numpy.average(nv1_omega_avg_list))
print( numpy.sqrt(numpy.sum(numpy.array(nv1_omega_error_list)**2)))

fig, ax = plt.subplots(1, 1, figsize=(10, 8))

#ax.errorbar(splitting_list, omega_avg_list, yerr = omega_error_list)

axis_font = {'size':'14'}

orange = '#f7941d'
purple = '#87479b'

#print(fit_alpha_params)

ax.set_xscale("log", nonposx='clip')
ax.set_yscale("log", nonposy='clip')
ax.errorbar(nv1_splitting_list, nv1_gamma_avg_list, yerr = nv1_gamma_error_list, 
            label = r'$\gamma$',  fmt='o',markersize = 12, color = purple)
ax.errorbar(nv1_splitting_list, nv1_omega_avg_list, yerr = nv1_omega_error_list, 
            label = r'$\Omega$', fmt='^', markersize = 12, color=orange)

# %% Chi Squared

expected = []

for el in range(len(nv1_splitting_list)):
    expected_value = fit_eq_alpha(nv1_splitting_list[el], *fit_alpha_params)
    expected.append(expected_value)
    
ret_vals = chisquare(nv1_gamma_avg_list, f_exp=expected)
chi_sq = ret_vals[0]


# %%

ax.plot(splitting_linspace, fit_eq_alpha(splitting_linspace, *fit_alpha_params), 
             linestyle='dashed', linewidth=3,color =purple)
ax.plot(splitting_linspace, omega_constant_array, color = orange,
            linestyle='dashed', linewidth=3)

text = '\n'.join((r'$A_0/f^{2} + \gamma_\infty$ fit:',
#                  r'$\alpha = {} \pm {}$'.format('%.2f'%(fit_alpha_params[1]), '%.2f'%(numpy.sqrt(cov_arr[1][1]))),
                  r'$A_0 = {} \pm {}$'.format('%.0f'%(fit_alpha_params[0]), '%.0f'%(numpy.sqrt(cov_arr[0][0]))),
                  r'$\gamma_\infty = {} \pm {}$'.format('%.2f'%(fit_alpha_params[1]), '%.2f'%(numpy.sqrt(cov_arr[1][1]))),
                  r'$\chi^2 = $' + '%.2f'%(chi_sq)
#                  ,r'$a = $' + '%.2f'%(fit_params[2])
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
#ax.set_ylim([-10,150])

plt.xlabel('Splitting (MHz)', fontsize=18)
plt.ylabel('Relaxation Rate (kHz)', fontsize=18)
#plt.title('NV 1', fontsize=18)
#ax.legend(fontsize=18)

#fig.savefig("C:/Users/Aedan/Creative Cloud Files/Paper Illustrations/Magnetically Forbidden Rate/fig_3c.pdf", bbox_inches='tight')


