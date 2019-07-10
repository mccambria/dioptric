# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 17:53:37 2019

@author: mccambria
"""

import numpy
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def exp_dec(t, rate, amp):
    return  amp * numpy.exp(-rate * t)

def exp_dec_offset(t, rate, amp, offset):
    return  offset + (amp * numpy.exp(-rate * t))

t = numpy.linspace(0, 10, 6)
y_vals = [9, 6, 4, 3, 2, 1.5]
sigmas = [2, 1, 1, 0.5, 1, 0.5]

plt.plot(t, y_vals)
plt.errorbar(t, y_vals, yerr=sigmas, linestyle='None')

# popt, pcov = curve_fit(exp_dec, t, y_vals)
popt, pcov = curve_fit(exp_dec, t, y_vals, sigma=sigmas, absolute_sigma=True)
print(pcov)

smooth_t = numpy.linspace(0, 10, 1000)
plt.plot(smooth_t, exp_dec(smooth_t, *popt))
