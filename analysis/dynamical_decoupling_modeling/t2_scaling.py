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

def T2_scale(N, T2_0): #https://journals.aps.org/prb/pdf/10.1103/PhysRevB.85.155204
    return (N/2)**(2/3)*T2_0
    
def T2_scale_alt(N,n, T2_0): #https://www.nature.com/articles/s41467-019-11776-8
    return (N)**(n)*T2_0
    

N = [1, 2, 4, 8, 16]
T2=[934, 1591, 2165, 2087, 2600]
 
#Bar Gill T=300 K
# N = [1, 8, 16, 32]
# T2=[800, 1500, 2000,2200]
 
fig, ax = plt.subplots()   
ax.plot(N, T2, 'bo', 
            label = 'data')

centers_lin = numpy.linspace(N[0], N[-1], 100)
# print(centers_lin)


# fit_func = T2_scale
# init_params = [ 150]
# popt, pcov = curve_fit(
#     fit_func,
#     N,
#     T2,
#     # sigma=norm_avg_sig_ste,
#     # absolute_sigma=True,
#     p0=init_params,
#     # bounds=(0, numpy.inf),
# )


fit_func = T2_scale_alt
init_params = [ 0.05, 1500]
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
print(popt)
# print(pcov)
ax.plot(
        centers_lin,
        fit_func(centers_lin, *popt),
        "r-",
        label="fit",
    ) 

text_popt = '\n'.join((
                    r'$T_2 = (N/2)^n * T_{2,0}$',
                    r'n = ' + '%.2f'%(popt[0]) + ' +/- ' + '%.2f'%(numpy.sqrt(pcov[0][0])),
                    r'$T_{2,0} = $' + '%.0f'%(popt[1]) + ' +/- ' + '%.0f'%(numpy.sqrt(pcov[1][1])) + ' us',
                    
                    
                    ))

props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax.text(0.4, 0.3, text_popt, transform=ax.transAxes, fontsize=12,
        verticalalignment='top', bbox=props)



ax.set_xlabel('Number of pi pulses, N')
ax.set_ylabel('T2 time (us)')
# ax.set_title('Lifetime for {}'.format(nv_name))
# ax.set_ylim([5e-4, 1.7])
ax.set_yscale("log")
ax.set_xscale("log")
ax.legend()
    