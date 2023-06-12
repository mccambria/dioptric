# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 10:17:47 2023

@author: gardill
"""

import numpy
import utils.kplotlib as kpl
from utils.kplotlib import KplColors
import utils.tool_belt as tool_belt
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def inv_sq_rt_fit(x, amp, offset):
    return offset + amp / numpy.sqrt(x)

kpl.init_kplotlib
    
# powers_dbm = numpy.array([4,7,10,13,14, 14.5,16, 17])
# pi_pulse = numpy.array([171.82 ,133,  112.60, 101.98,93.15,  87.39,  87.88, 90.86])

powers_dbm = numpy.array([-15, -12,-9, -6, -5, -4, -3, -2 ,-1, 0])
pi_pulse = numpy.array([186, 151, 133,102,  97, 94, 93, 89, 80.10, 86.21])


rabi_period = 2*pi_pulse
powers_mw = 10**(powers_dbm/10)
x_smooth = numpy.linspace(powers_mw[0], powers_mw[-1], 1000)

# fit_func = inv_sq_rt_fit
# init_params = [ 1000, 175]
# popt, pcov = curve_fit(
#     fit_func,
#       powers_mw,
#     rabi_period,
#     p0=init_params,
# )

# print(popt)
    
# fig, ax = plt.subplots()

# kpl.plot_points(ax,  powers_mw, rabi_period, label = 'data', color=KplColors.BLACK)
# kpl.plot_line(ax, x_smooth, fit_func(x_smooth,*popt ), label = 'fit', color=KplColors.RED)
# ax.set_xlabel('MW power (mW)')
# ax.set_ylabel('Rabi period (ns)')
# ax.legend()

rabi_freq_mhz = 1/rabi_period*1e3
x_smooth = numpy.linspace(powers_mw[0]**0.5, powers_mw[-1]**0.5, 1000)
fit_func = tool_belt.linear
init_params = [ 1,0]

popt, pcov = curve_fit(
    fit_func,
      powers_mw[:-5]**0.5,
    rabi_freq_mhz[:-5],
    p0=init_params,
)
print(popt)
    
fig, ax = plt.subplots()

kpl.plot_points(ax,  powers_mw**0.5, rabi_freq_mhz, label = 'data', color=KplColors.BLACK)
kpl.plot_line(ax, x_smooth, fit_func(x_smooth,*popt ), label = 'fit', color=KplColors.RED)
ax.set_xlabel('sqrt(MW power) (mW^(1/2))')
ax.set_ylabel('Rabi freq (MHz)')
ax.legend()