# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 13:19:47 2022

@author: kolkowitz
"""

import numpy
import utils.tool_belt as tool_belt
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.integrate import quad
from numpy import pi

dd_model_coeff_dict = {
    '1': [6, -8, 2],
    '2': [10, -8, -8, 8, -2],
    '4': [18, -8, -24, 8, 16, -8, -8, 8, -2],
    '8': [34, -8, -56, 8, 48, 
               -8, -40, 8, 32,  -8, -24, 8, 16, -8, -8, 8, -2]
    }
# a_list_se = [6, -8, 2]
# a_list_2 = [10, -8, -8, 8, -2]
# a_list_4 = [18, -8, -24, 8, 16, -8, -8, 8, -2]
# a_list_8 = [34, -8, -56, 8, 48, 
#            -8, -40, 8, 32,  -8, -24, 8, 16, -8, -8, 8, -2]


def S_bath_test(t, lambd, sigma, a_list ):
    sum_expr = a_list[0]
    
    for i in range(len(a_list)-1):
        n=i+1
        # print(n)
        sum_expr += a_list[n]*numpy.exp(-n**2 * (t*2*pi)**2 * (sigma)**2 / 2) * numpy.cos(n*t*2*pi)

    X = 4*lambd**2 * sum_expr
    return numpy.exp(-X)



fig, ax = plt.subplots()
taus_lin = numpy.linspace(0, 3,600)
params = [0.4, 0.03]

#######________________________#############
fit_func = lambda  t, lambd, sigma: S_bath_test(t, lambd, sigma,  dd_model_coeff_dict['2'] ) 
ax.plot(
        taus_lin,
        fit_func(taus_lin, *params),
        "-",
        color="black",
        label=r'2 $\pi$-pulses',
    ) 

fit_func = lambda  t, lambd, sigma: S_bath_test(t, lambd, sigma,  dd_model_coeff_dict['4'] ) 
ax.plot(
        taus_lin,
        fit_func(taus_lin, *params),
        "-",
        color="blue",
        label=r'4 $\pi$-pulses',
    ) 

fit_func = lambda  t, lambd, sigma: S_bath_test(t, lambd, sigma,  dd_model_coeff_dict['8'] ) 
ax.plot(
        taus_lin,
        fit_func(taus_lin, *params),
        "-",
        color="red",
        label=r'8 $\pi$-pulses',
    ) 

ax.set_xlabel(r"Evolution time ($2 \pi/ \omega_L$)")
ax.set_ylabel("Coherence")

# ax.legend()

