# -*- coding: utf-8 -*-
"""

Plotting gamma, omega, and the ratio for chinese sample Hopper
Created on Thu Feb 13 10:59:20 2020

@author: Aedan
"""

import matplotlib.pyplot as plt
import numpy

angles = [0, 11, 26, 32, 42, 49, 52, 65, 82, 85.5, 84.8]

gamma = numpy.array([0.120, 0.140, 0.151, 0.149, 0.140, 0.154, 0.147, 0.185, 0.181, 0.235, 0.20])
gamma_unc = numpy.array([0.011, 0.012, 0.015, 0.013, 0.014, 0.015, 0.014, 0.017, 0.019, 0.034, 0.03])

omega = numpy.array([0.061, 0.047, 0.065, 0.067, 0.053, 0.064, 0.065, 0.057, 0.055, 0.064, 0.070])
omega_unc = numpy.array([0.004, 0.003, 0.005, 0.006, 0.004, 0.005, 0.005, 0.005, 0.004, 0.007, 0.006])

ratio = gamma / omega
ratio_unc = ratio*numpy.sqrt( (gamma_unc / gamma)**2 + (omega_unc / omega)**2)

fig, axes = plt.subplots(2,1,figsize=(8.5, 8.5))

ax = axes[0]
ax.errorbar(angles, gamma, yerr = gamma_unc, label = 'gamma', fmt = 'o')
ax.errorbar(angles, omega, yerr = omega_unc, label = 'omega', fmt = 'o')
ax.set_xlabel('Angle (degree)')
ax.set_ylabel('Relaxation rate (kHz)')
ax.legend()  

ax = axes[1]
ax.errorbar(angles, ratio, yerr = ratio_unc, label = 'ratio gamma/omega', fmt = 'o')
ax.set_xlabel('Angle (degree)')
ax.set_ylabel('Ratio gamma/omega')
ax.legend()  
