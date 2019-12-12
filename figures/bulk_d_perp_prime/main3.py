# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 17:04:19 2019

@author: matth
"""


# %% Imports


import numpy
import matplotlib.pyplot as plt


# %% Functions


# %% Main


def main(nv_data):

    # %% Plotting

    plt.rcParams.update({'font.size': 18})  # Increase font size
    # fig, axes_pack = plt.subplots(1, 2, figsize=(12,5))
    fig, ax = plt.subplots(figsize=(6,5))
    fig.set_tight_layout(True)
    
    # Splittings
    # ax = axes_pack[0]
    ax.set_xlabel(r'Splitting, $\Delta_{\pm}$ (MHz)')
    ax.set_ylabel(r'$\gamma / \Omega$')
    ax.set_ylim(0, 3.0)
    
    # Angles
    # ax = axes_pack[1]
    # ax.set_xlabel(r'Magnet angle, $\theta_{B}$ ($\degree$)')
    # ax.set_ylabel(r'$\gamma / \Omega$')
    # ax.set_xlim(0, 180)
    # ax.set_xticks(numpy.linspace(0, 180, 5))
    # ax.set_ylim(0, 3.0)
    
    # Marker and color combination to distinguish NVs
    markers = ['^', 'o', 's', 'D', ]
    colors = ['#009E73', '#E69F00', '#0072B2', '#CC79A7', ]

    for ind in range(len(nv_data)):
        
        nv = nv_data[ind]
        marker = markers[ind]
        color = colors[ind]  # Wong colorblind-safe palette
        
        # Calculate ratios
        name = nv['name']
        gammas = numpy.array(nv['gammas'])
        omegas = numpy.array(nv['omegas'])
        gamma_errors = numpy.array(nv['gamma_errors'])
        omega_errors = numpy.array(nv['omega_errors'])
        ratios = gammas / omegas
        ratio_errors = ratios * numpy.sqrt((gamma_errors / gammas)**2 +
                                           (omega_errors / omegas)**2)
        
        # Plot splittings
        # ax = axes_pack[0]
        splittings = numpy.array(nv['splittings'])
        ax.errorbar(splittings, ratios, yerr=ratio_errors, label=name,
                    marker=marker, color=color, linestyle='None', ms=9, lw=2.5)
    
        # Plot angles
        # ax = axes_pack[1]
        # ax.scatter(times_high, signal_high, marker='o',
        #            color='#CC99CC', edgecolor='#993399', ms=60)

    ax.legend(loc='lower right')


# %% Run


if __name__ == '__main__':


    # List of data for each NV. Each element should be a dictionary
    # containing NV name and lists of splittings, angles, gammas,
    # gamma_errors, omegas, omega_errors for each point. Put None if unknown.
    nv_data = [
            {
                'name': 'NVA1',
                'splittings': [23.9, 125.9, 128.1, 233.2],
                'angles': [None, None, None, None],
                'gammas': [0.13, 0.111, 0.14, 0.132],
                'gamma_errors': [0.02, 0.009, 0.03, 0.017],
                'omegas': [0.063, 0.053, 0.059, 0.061],
                'omega_errors': [0.009, 0.003, 0.006, 0.006],
                },
            {
                'name': 'NVA2',
                'splittings': [129.7],
                'angles': [None],
                'gammas': [0.114],
                'gamma_errors': [0.012],
                'omegas': [0.060],
                'omega_errors': [0.004],
                },
            {
                'name': 'NVB1',
                'splittings': [167.1, 831.6],
                'angles': [None, None],
                'gammas': [0.132, 0.169],
                'gamma_errors': [0.04],
                'omegas': [0.083],
                'omega_errors': [0.02],
                },
        ]
    
    main(nv_data)

