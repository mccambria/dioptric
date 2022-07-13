# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 09:19:51 2022

@author: agard
"""

import utils.tool_belt as tool_belt
import numpy
import matplotlib.pyplot as plt
import airy_disk_simulation 
from scipy.optimize import curve_fit
from scipy.special import j1
from scipy.special import jv
import copy

# %%
NA = 1.3
wavelength = 638
fwhm =1.825 # 2* (ln(2))^1/4
scale = 0.99e3


def bessel_scnd_der(x):
    term_1 = 24*j1(x)**2
    term_2 = 16*j1(x)*(jv(0,x) - jv(2,x))
    term_3 = 4* (0.5* (jv(0,x) - jv(2,x))**2 + j1(x)* (0.5*(jv(3,x) - j1(x)) - j1(x)))
    
    return term_1/x**4 - term_2/x**3 + term_3/x**2
# %% Data from fitting
y = numpy.array([0.21638739, 0.23514312, 0.270726 ,  0.27750125, 0.39541004, 0.59534277,
 0.58171584, 0.69582389, 0.86908403, 1.11268992, 1.22341671, 1.36458656,
 2.36018764]) #fwhm list, in dimensionless units
y_err = numpy.array([0.01053939, 0.01709594, 0.01140871, 0.0106756,
                   0.01095163, 
                   0.01374391, 0.01250901 ,0.01245438, 0.01452769, 
                   0.02224011 ,0.0237046 , 0.02058593, 0.04687215])

t = numpy.array([10.0, 11.0, 7.5, 5.0, 2.5, 1.0, 0.75, 0.5,
              0.25, 0.1, 0.075, 0.05, 0.01]) # in ms

lin_x_vals = numpy.logspace(numpy.log10(0.01), numpy.log10(11), 100)
# %%
def width_scaling_w_mods(t, e, alpha, R):

    x0=7.0156 #Position of this Airy disk (n2), in dimensionless units
    C = bessel_scnd_der(x0) #Calculate a constant to use in fit
    
    return numpy.sqrt(4/C* (-e + numpy.sqrt(e**2+ alpha/t)) + R**2)


fit_func = width_scaling_w_mods
e =  0.0008723132950598539
alpha = 0.00000309380937
R_nm = 6.1
params = [e, alpha, R_nm*(2*numpy.pi*NA)/wavelength]

fig, ax = plt.subplots()
ax.plot(lin_x_vals, fit_func(lin_x_vals, *params)*wavelength/(2*numpy.pi*NA), 
            color = 'red',  linestyle = 'dashed' , linewidth = 1)


ax.errorbar(t, numpy.array(y)*wavelength/(2*numpy.pi*NA),  
            yerr = numpy.array(y_err)*wavelength/(2*numpy.pi*NA), 
                fmt='o', color = 'black', 
                linewidth = 1, markersize = 5, mfc='#d6d6d6')

ax.set_xlabel(r'Depletion pulse duration, $\tau$ (ms)')
ax.set_ylabel('FWHM (nm)')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_ylim([13.8,208.5])