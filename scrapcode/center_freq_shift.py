# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 13:58:57 2019

@author: Aedan
"""

from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy

def parab(s, a, b, c):
    return a*s**2 + b*s + c

splittings = [45.5, 85.2, 280.4]
center_freq = [2.86625, 2.86915, 2.9128]

fit_params, cov_arr = curve_fit(parab, splittings, center_freq, 
                                p0 = [10**-7, 10**-5, 2.86])


splitting_linspace = numpy.linspace(splittings[0], splittings[-1],
                                    1000)

fig, ax = plt.subplots(1, 1, figsize=(10, 8))

ax.plot(splittings, center_freq, 'bo')
ax.plot(splitting_linspace, parab(splitting_linspace, *fit_params))

print(fit_params)

expected_splitting = parab(600, *fit_params)
freq2 = expected_splitting + .300
freq1 = expected_splitting - .300
print(freq1, freq2)
