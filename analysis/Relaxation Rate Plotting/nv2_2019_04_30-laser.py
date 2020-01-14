# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 14:36:23 2019

This file plots the relaxation rate data collected for the nv2_2019_04_30.

The data is input manually, and plotted on a loglog plot with error bars along
the y-axis. A 1/f**2 line is also fit to the gamma rates to show the behavior.

@author: mccambria
"""


import matplotlib.pyplot as plt
import numpy


# %% Data


# Laser powers in uW, rates in kHz
laser_powers = numpy.array([44, 153, 582, 1640])/1000
omegas = [0.215, 0.337, 0.278, 0.177]
omega_errors = numpy.array([0.010, 0.016, 0.042, 0.018])*2
gammas = [24.254, 21.692, 26.103, 25.415]
gamma_errors = numpy.array([0.335, 0.313, 0.665, 0.598])*2

omega_twice_errors = [2*error for error in omega_errors]
gamma_twice_errors = [2*error for error in gamma_errors]


# %% Plotting



fig, ax = plt.subplots(1, 1, figsize=(7, 8))


orange = '#f7941d'
purple = '#87479b'


x_range = [30/1000, 2000/1000]
laser_powers_linspace = numpy.linspace(*x_range, 1000)

ax.set_xscale("log", nonposx='clip')
#ax.set_yscale("log", nonposy='clip')
#ax.errorbar(laser_powers, gammas, yerr=gamma_errors, 
#            label=r'$\gamma$', fmt='o',markersize=15, color=purple)
ax.errorbar(laser_powers, omegas, yerr=omega_errors, 
            label=r'$\Omega$', fmt='^', markersize=15, color=orange)
#ax.plot(laser_powers_linspace, [numpy.average(gammas)]*1000, 
#        linestyle='dashed', linewidth=3, color=purple)
ax.plot(laser_powers_linspace, [numpy.average(omegas)]*1000,
        linestyle='dashed', linewidth=3, color=orange)
omega_ax.set_ylim([0.1, 0.4])


ax.tick_params(which='both', length=6, width=2, colors='k',
               direction='in', grid_alpha=0.7, labelsize=18)

ax.tick_params(which='major', length=12, width=2)

#ax.grid(axis='x')

ax.set_xlim(x_range)
#ax.set_ylim([20, 28])

ax.set_ylim([0.1,0.4])

ax.set_xlabel(r'Laser Power (mW)', fontsize=18)
ax.set_ylabel('Relaxation Rate (kHz)', fontsize=18)

fig.savefig("C:/Users/Aedan/Creative Cloud Files/Paper Illustrations/Magnetically Forbidden Rate/supplemental_materials/laser_omega.svg", bbox_inches='tight')

