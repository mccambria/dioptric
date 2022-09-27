# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 11:33:28 2022

@author: kolkowitz
"""

import utils.tool_belt as tool_belt
import majorroutines.optimize as optimize
import numpy
import os
import time
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import json
import labrad

def T2_scale(N, T2_0):
    return (N/2)**(2/3)*T2_0
    
N = [4,8,16]
T2=[191, 247, 295]
 
fig, ax = plt.subplots()   
ax.plot(N, T2, 'bo', 
            label = 'data')

centers_lin = numpy.linspace(1, 20, 100)
# print(centers_lin)
fit_func = T2_scale
init_params = [ 150]
popt, pcov = curve_fit(
    fit_func,
    N,
    T2,
    # sigma=norm_avg_sig_ste,
    # absolute_sigma=True,
    p0=init_params,
    # bounds=(0, numpy.inf),
)
print('T2_0 = {} +/- {} us'.format(popt[0], numpy.sqrt(pcov[0][0])))
# print(popt)
# print(pcov)
ax.plot(
        centers_lin,
        fit_func(centers_lin, *popt),
        "r-",
        label="fit",
    ) 

# text_popt = '\n'.join((
#                     r'y = exp(-t / d)',
#                     r'd = ' + '%.2f'%(popt[0]) + ' +/- ' + '%.2f'%(numpy.sqrt(pcov[0][0])) + ' ns'
#                     ))

# props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
# ax.text(0.1, 0.3, text_popt, transform=ax.transAxes, fontsize=12,
#         verticalalignment='top', bbox=props)



ax.set_xlabel('Number of pi pulses')
ax.set_ylabel('T2 time')
# ax.set_title('Lifetime for {}'.format(nv_name))
# ax.set_ylim([5e-4, 1.7])
# ax.set_yscale("log")
ax.legend()
    