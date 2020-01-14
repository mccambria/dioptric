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


def main(nv_data, just_splittings=False):
    
    # %% Setup
    
    all_ratios = []
    all_ratio_errors = []

    # %% Plotting

    plt.rcParams.update({'font.size': 18})  # Increase font size
    if just_splittings:
        fig, ax = plt.subplots(figsize=(6,5))
    else:
        fig, axes_pack = plt.subplots(1, 2, figsize=(12,5))
    fig.set_tight_layout(True)
    
    # Splittings
    if not just_splittings:
        ax = axes_pack[0]
    ax.set_xlabel(r'Splitting, $\Delta_{\pm}$ (MHz)')
    ax.set_ylabel(r'$\gamma / \Omega$')
    ax.set_ylim(0, 3.0)
    
    # Angles
    if not just_splittings:
        ax = axes_pack[1]
        ax.set_xlabel(r'Magnet angle, $\theta_{B}$ ($\degree$)')
        ax.set_ylabel(r'$\gamma / \Omega$')
        ax.set_xlim(0, 180)
        ax.set_xticks(numpy.linspace(0, 180, 5))
        ax.set_ylim(0, 3.0)
    
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
        all_ratios.extend(ratios)
        all_ratio_errors.extend(ratio_errors)
        
        # Plot splittings
        if not just_splittings:
            ax = axes_pack[0]
        splittings = numpy.array(nv['splittings'])
        ax.errorbar(splittings, ratios, yerr=ratio_errors, label=name,
                    marker=marker, color=color, linestyle='None', ms=9, lw=2.5)
    
        # Plot angles
        if not just_splittings:
            ax = axes_pack[1]

    all_ratios = numpy.array(all_ratios)
    all_ratio_errors = numpy.array(all_ratio_errors)
    if not just_splittings:
        ax = axes_pack[0]
    ax.legend(loc='lower right')
    wavg_ratio = numpy.average(all_ratios, weights=(1/all_ratio_errors**2))
    ste_ratio = numpy.sqrt(1/numpy.sum(all_ratio_errors**-2))
    print(wavg_ratio)
    print(ste_ratio)
    xlim = ax.get_xlim()
    ax.fill_between([xlim[0], xlim[1]],
                    wavg_ratio+ste_ratio, wavg_ratio-ste_ratio,
                    alpha=0.5, color=colors[-1])
    ax.plot([xlim[0], xlim[1]], [wavg_ratio, wavg_ratio], color=colors[-1])


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
                'splittings': [40.6, 167.1, 412.7, 623.8, 831.6, 1207.1],
                'angles': [None, None, None, None, None, None],
                'gammas': [0.115, 0.132, 0.135, 0.138, 0.169, 0.103],
                'gamma_errors': [0.010, 0.011, 0.013, 0.014, 0.04, 0.026],
                'omegas': [0.049, 0.056, 0.065, 0.061, 0.083, 0.048],
                'omega_errors': [0.003, 0.003, 0.005, 0.005, 0.02, 0.013],
                },
        ]
    
    main(nv_data, just_splittings=True)

