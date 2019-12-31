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
laser_powers = [44, 153, 582, 1640]
omegas = [0.215, 0.337, 0.278, 0.177]
omega_errors = [0.010, 0.016, 0.042, 0.018]
gammas = [24.254, 21.692, 26.103, 25.415]
gamma_errors = [0.335, 0.313, 0.665, 0.598]

omega_twice_errors = [2*error for error in omega_errors]
gamma_twice_errors = [2*error for error in gamma_errors]


# %% Plotting


fig, axes_pack = plt.subplots(1, 2, figsize=(20, 8))

orange = '#f7941d'
purple = '#87479b'

# x_range = [30, 2000]
x_range = [-50, 1800]
laser_powers_linspace = numpy.linspace(*x_range, 1000)

omega_ax = axes_pack[0]
omega_ax.errorbar(laser_powers, omegas, yerr=omega_twice_errors, 
            label=r'$\Omega$', fmt='^', markersize=15, color=orange)
omega_ax.plot(laser_powers_linspace, [numpy.average(omegas)]*1000,
        linestyle='dashed', linewidth=3, color=orange)
omega_ax.set_ylim([0.1, 0.4])

gamma_ax = axes_pack[1]
gamma_ax.errorbar(laser_powers, gammas, yerr=gamma_twice_errors, 
            label=r'$\gamma$', fmt='o',markersize=15, color=purple)
gamma_ax.plot(laser_powers_linspace, [numpy.average(gammas)]*1000, 
        linestyle='dashed', linewidth=3, color=purple)
gamma_ax.set_ylim([21, 28])

for ax in axes_pack:
    # ax.set_xscale("log", nonposx='clip')

    ax.tick_params(which='both', length=6, width=2, colors='k',
                   direction='in', grid_alpha=0.7, labelsize=18)
    
    ax.tick_params(which='major', length=12, width=2)
    
    ax.grid()
    
    ax.set_xlim(x_range)
    
    ax.set_xlabel(r'Laser Power ($\mathrm{\mu}$W)', fontsize=18)
    ax.set_ylabel('Relaxation Rate (kHz)', fontsize=18)
