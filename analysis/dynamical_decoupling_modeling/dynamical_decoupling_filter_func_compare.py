# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 15:30:18 2022

@author: kolkowitz
"""


import numpy
import utils.tool_belt as tool_belt
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.integrate import quad
from numpy import pi

def SE1(z, a0, a1, a2):
    return a0 + a1*numpy.cos(z/2*2) + a2*numpy.cos(2*z/2*2)

def SE2(z):
    return 8*numpy.sin(z/2)**4

z_list = numpy.linspace(0,20,1001)
s_gen = SE2(z_list)


fig, ax = plt.subplots()

ax.plot(
        z_list,
        s_gen,
        "-",
        color="black",
        label="Simplified expression",
    ) 

init_params = [-3,1]
fit_func = lambda z, :SE1(z, 3, -4, 1) 
# popt, pcov = curve_fit(
#     fit_func,
#     z_list,
#     s_gen,
#     p0=init_params,
# )
# print(popt)

ax.plot(
        z_list,
        fit_func(z_list,),
        "-",
        color="red",
        label="Summed expression",
        ) 
# ax.set_xlabel(r"t (us)")
# ax.set_ylabel(r"Signal")
ax.legend()