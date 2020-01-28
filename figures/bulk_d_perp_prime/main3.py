# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 17:04:19 2019

@author: matth
"""


# %% Imports


import numpy
from numpy import pi
import matplotlib.pyplot as plt
from analysis.extract_hamiltonian import calc_splitting


# %% Constants


im = 0+1j
inv_sqrt_2 = 1/numpy.sqrt(2)
gmuB = 2.8e-3  # gyromagnetic ratio in GHz / G


# %% Functions


def plot_gamma_omega_vs_angle(nv_data):

    plt.rcParams.update({'font.size': 18})  # Increase font size
    fig, axes_pack = plt.subplots(1, 2, figsize=(12,5))
    fig.set_tight_layout(True)
    for ax in axes_pack:
        ax.set_xlabel(r'Magnet angle, $\theta_{B}$ ($\degree$)')
        ax.set_xlim(-5, 95)
        ax.set_xticks(numpy.linspace(0,90,7))
        ax.set_ylim(0.78, 1.25)
        
    gamma_ax = axes_pack[0]
    omega_ax = axes_pack[1]
    gamma_ax.set_title('NVB1 gamma vs angle')
    omega_ax.set_title('NVB1 Omega vs angle')
    
    for ind in range(len(nv_data)):
        
        nv = nv_data[ind]
        
        name = nv['name']
        if name in ['NVA1', 'NVA2']:
            continue
        
        gammas = numpy.array(nv['gammas'])
        omegas = numpy.array(nv['omegas'])
        gamma_errors = numpy.array(nv['gamma_errors'])
        omega_errors = numpy.array(nv['omega_errors'])
        
        angles = numpy.array(nv['angles'])
        mask = angles != None
        if True in mask:
            
            # gamma_ax.errorbar(angles[mask], gammas[mask],
            #             yerr=gamma_errors[mask], label='gamma',
            #             linestyle='None', marker='o', ms=9, lw=2.5)
            # omega_ax.errorbar(angles[mask], omegas[mask],
            #             yerr=omega_errors[mask], label='Omega',
            #             linestyle='None', marker='o', ms=9, lw=2.5)
            
            gamma_wavg = numpy.average(gammas[mask], weights=(1/gamma_errors[mask]**2))
            gamma_ax.errorbar(angles[mask], gammas[mask]/gamma_wavg,
                              yerr=gamma_errors[mask]/gamma_wavg,
                              linestyle='None', marker='o', ms=9, lw=2.5)
            omega_wavg = numpy.average(omegas[mask], weights=(1/omega_errors[mask]**2))
            omega_ax.errorbar(angles[mask], omegas[mask]/omega_wavg,
                              yerr=omega_errors[mask]/omega_wavg,
                              linestyle='None', marker='o', ms=9, lw=2.5)


def plot_splittings_vs_angle(nv_data):

    plt.rcParams.update({'font.size': 18})  # Increase font size
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    fig.set_tight_layout(True)
    ax.set_xlabel(r'Magnet angle, $\theta_{B}$ ($\degree$)')
    ax.set_xlim(-5, 95)
    ax.set_xticks(numpy.linspace(0,90,7))
    ax.set_ylabel('Splitting (MHz)')
    mag_B = 33 * gmuB
        
    for ind in range(len(nv_data)):
        
        nv = nv_data[ind]
        
        name = nv['name']
        # if name in ['NVA1', 'NVA2']:
        #     continue
        if name != 'NVB1':
            continue
        
        all_splittings = numpy.array(nv['all_splittings'])
        angles = numpy.array(nv['angles'])
        mask = angles != None
        if True in mask:
            smooth_theta_Bs = numpy.linspace(0, pi/2, 1000)
            smooth_theta_Bs_deg = smooth_theta_Bs * (180/pi)
            splittings = [calc_splitting(mag_B, val, 0, 0, 0, 0) * 1000
                          for val in smooth_theta_Bs]  # Scaled to MHz
            ax.plot(smooth_theta_Bs_deg, splittings,
                    c='orange', label='33 G predicted splittings')
            ax.scatter(angles[mask], all_splittings[mask],
                       label='Measured splittings')
            ax.legend()


# %% Main


def main(nv_data, just_splittings=False):
    
    # %% Setup
    
    all_ratios = []
    all_ratio_errors = []

    # %% Plotting

    plt.rcParams.update({'font.size': 18})  # Increase font size
    if just_splittings:
        fig, ax = plt.subplots(figsize=(6,5))
        # Pack the one ax up so we don't have to make too many changes
        # for just_splittings
        axes_pack = [ax]
    else:
        fig, axes_pack = plt.subplots(1, 2, figsize=(12,5))
    fig.set_tight_layout(True)
    
    # Splittings
    ax = axes_pack[0]
    ax.set_xlabel(r'Splitting, $\Delta_{\pm}$ (MHz)')
    ax.set_xlim(-50, 1300)
    
    # Angles
    if not just_splittings:
        ax = axes_pack[1]
        ax.set_xlabel(r'Magnet angle, $\theta_{B}$ ($\degree$)')
        ax.set_xlim(-5, 95)
        ax.set_xticks(numpy.linspace(0,90,7))
        
    # Both
    for ax in axes_pack:
        ax.set_ylabel(r'$\gamma / \Omega$')
        ax.set_ylim(0, 3.5)
        
    
    # Marker and color combination to distinguish NVs
    markers = ['^', 'o', 's', 'D', ]
    colors = ['#009E73', '#E69F00', '#0072B2', '#CC79A7', ]

    for ind in range(len(nv_data)):
        
        nv = nv_data[ind]
        marker = markers[ind]
        color = colors[ind]  # Wong colorblind-safe palette
        
        name = nv['name']
        # if name in ['NVA1', 'NVA2']:
        #     continue
        if name == 'test':
            continue
        
        # Calculate ratios
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
        ax = axes_pack[0]
        splittings = numpy.array(nv['splittings'])
        mask = splittings != None
        if True in mask:
            ax.errorbar(splittings[mask], ratios[mask],
                        yerr=ratio_errors[mask], label=name,
                        marker=marker, color=color, linestyle='None',
                        ms=9, lw=2.5)
    
        # Plot angles
        if not just_splittings:
            ax = axes_pack[1]
            angles = numpy.array(nv['angles'])
            mask = angles != None
            if True in mask:
                print(angles[mask])
                print(ratios[mask])
                print(ratio_errors[mask])
                ax.errorbar(angles[mask], ratios[mask],
                            yerr=ratio_errors[mask], label=name,
                            marker=marker, color=color, linestyle='None',
                            ms=9, lw=2.5)
        # print(ratios)
        # print(ratio_errors)
        # print(angles)

    all_ratios = numpy.array(all_ratios)
    all_ratio_errors = numpy.array(all_ratio_errors)
    
    wavg_ratio = numpy.average(all_ratios, weights=(1/all_ratio_errors**2))
    ste_ratio = numpy.sqrt(1/numpy.sum(all_ratio_errors**-2))
    # print(all_ratios)
    # print(all_ratio_errors)
    # print(wavg_ratio)
    # print(ste_ratio)
    
    # For both axes, plot the same weighted average and display a legend.
    for ax in axes_pack:
        ax.legend(loc='lower right')
        xlim = ax.get_xlim()
        ax.fill_between([0, xlim[1]],
                        wavg_ratio+ste_ratio, wavg_ratio-ste_ratio,
                        alpha=0.5, color=colors[-1])
        ax.plot([0, xlim[1]], [wavg_ratio, wavg_ratio], color=colors[-1])


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
                'splittings': [40.6, 167.1, 412.7, 623.8, 831.6, 1207.1,
                               None, None, None, None, None, None,
                               None, None, None],
                'all_splittings': [40.6, 167.1, 412.7, 623.8, 831.6, 1207.1,
                                   187.2, 80.4, 177.2, 121.4, 33.7, 16.6,
                                   154.5, 122.1, 555.7],
                'angles': [None, None, None, None, None, None,
                           0.000, 63.819, 16.345, 47.524, 79.508, 84.801,
                           32.145, 48.258, 47.578],
                'gammas': [0.115, 0.132, 0.135, 0.138, 0.169, 0.103,
                           0.121, 0.145, 0.130, 0.158, 0.139, 0.128,
                           0.135, 0.150, 0.162],
                'gamma_errors': [0.010, 0.011, 0.013, 0.014, 0.04, 0.026,
                                 0.010, 0.011, 0.010, 0.012, 0.013, 0.019,
                                 0.011, 0.012, 0.024],
                'omegas': [0.049, 0.056, 0.065, 0.061, 0.083, 0.048,
                           0.062, 0.060, 0.058, 0.052, 0.059, 0.054,
                           0.056, 0.055, 0.064],
                'omega_errors': [0.003, 0.003, 0.005, 0.005, 0.02, 0.013,
                                 0.005, 0.005, 0.005, 0.004, 0.005, 0.005,
                                 0.005, 0.005, 0.009],
                },
            # {
            #     'name': 'test',
            #     'splittings': [],
            #     'angles': [],
            #     'gammas': [0.113],
            #     'gamma_errors': [0.010],
            #     'omegas': [0.055],
            #     'omega_errors': [0.004],
            #     },
        ]
    
    main(nv_data, just_splittings=False)
    # plot_gamma_omega_vs_angle(nv_data)
    # plot_splittings_vs_angle(nv_data)

