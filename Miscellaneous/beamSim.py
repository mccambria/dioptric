# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import matplotlib.pyplot as plt
import numpy as np

waist = 0.0011
wavelength = 532 * (10**-9)
samples = 6
sampleDensity = 1000  # 1000 samples/meter

rayleigh = np.pi * (waist**2) / wavelength
print(rayleigh)

xVals = np.arange(samples)
xVals = xVals - (samples//2)
print(xVals)
xVals = xVals / sampleDensity

yVals = waist * np.sqrt(1 + (xVals / rayleigh)**2)
fig, ax = plt.subplots()
ax.plot(xVals, yVals)
negYVals = -1 * yVals
ax.plot(xVals, -negYVals)
