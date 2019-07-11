# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 17:53:37 2019

@author: mccambria
"""

import numpy
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def line(t, a, b):
    return a + (b * t)

def exp_dec(t, rate, amp):
    return amp * numpy.exp(-rate * t)

def exp_dec_d_rate(t, rate, amp):
    return (-rate) * amp * numpy.exp(-rate * t)

def exp_dec_d_amp(t, rate, amp):
    return numpy.exp(-rate * t)

def exp_dec_offset(t, rate, amp, offset):
    return  offset + (amp * numpy.exp(-rate * t))

t = numpy.linspace(0, 10, 6)
y_vals = [9, 6, 4, 3, 2, 1.5]
sigmas = [2, 1, 1, 0.5, 1, 0.5]

plt.plot(t, y_vals)
plt.errorbar(t, y_vals, yerr=sigmas, linestyle='None')

# fit_func = line
fit_func = exp_dec

popt, pcov = curve_fit(fit_func, t, y_vals)
# popt, pcov = curve_fit(fit_func, t, y_vals, sigma=sigmas, absolute_sigma=True)
print(pcov)

# Get the chi squared
residuals = y_vals - fit_func(t, *popt)
chi_squared = sum((residuals) ** 2)  # without sigmas
# chi_squared = chisq = sum((residuals / sigmas) ** 2)  # with sigmas
reduced_chi_squared = chi_squared / 2  # chi squared per degree of freedom

# Manually calculate the covariances to check what's really happening
# Where f is the fit_func and pi is the ith fit param, the covariances are:
# cov_ij = reduced_chi_squared / sum_t((df/dpi)(df/dpj))
print(reduced_chi_squared * sum(exp_dec_d_rate(t, *popt) * exp_dec_d_rate(t, *popt)))

smooth_t = numpy.linspace(0, 10, 1000)
# plt.plot(smooth_t, fit_func(smooth_t, *popt))
