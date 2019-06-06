# -*- coding: utf-8 -*-
"""
Simple scatter plotting script

Created on Fri May 31 10:38:04 2019

@author: mccambria
"""

import matplotlib.pyplot as plt
import numpy
from scipy.optimize import curve_fit

# %% Fitting functions

def AbsCos(angle, offset, amp, phase):
    return offset + abs(amp * numpy.cos(angle * numpy.pi / 180 + phase * numpy.pi / 180))

# %% Scatter raw data

angles = [0, 30, 60, 90, 120, 150]
splittings = [25, 28, 41, 49, 48, 38]

fig, ax = plt.subplots()

ax.set_title('ESR splitting versus magnet angle')
ax.set_xlabel('Angle (deg)')
ax.set_ylabel('Splitting (MHz)')

ax.scatter(angles, splittings, c='r')

# %% Fitting

offset = 0
amp = 110
phase = 10

popt, pcov = curve_fit(AbsCos, angles, splittings, 
                       p0=[offset, amp, phase])

x_vals = numpy.linspace(min(angles), max(angles), 1000)
y_vals = AbsCos(x_vals, *popt)

ax.plot(x_vals, y_vals)
