# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 17:04:19 2019

@author: matth
"""


# %% Imports


import csv
import numpy
import numpy
from numpy import pi
import matplotlib.pyplot as plt
from analysis.extract_hamiltonian import calc_splitting


# %% Constants


im = 0+1j
inv_sqrt_2 = 1/numpy.sqrt(2)
gmuB = 2.8e-3  # gyromagnetic ratio in GHz / G


# %% Functions


def get_nv_data_csv(file):
    """
    Parses a csv into a list of dictionaries for each NV. Assumes the data 
    for an NV is grouped together, ie there are no gaps.
    """
    
    # Marker and color combination to distinguish NVs
    # Colors are from the Wong colorblind-safe palette
    marker_ind = 0
    markers = ['^', 'o', 's', 'D', 'X']
    colors = ['#009E73', '#E69F00', '#0072B2', '#CC79A7', '#D55E00',]
    
    # Columns to loop through, so this exludes name, column 0
    # res in GHz, splitting in MHz, B mags in G, angles in deg, 
    # rabis in ns, rates in kHz
    columns = ['res_minus', 'res_plus', 'splitting', 'mag_B',
               'theta_B', 'perp_B', 'contrast_minus', 'contrast_plus', 
               'rabi_minus', 'rabi_plus', 'gamma', 'gamma_error',
               'omega', 'omega_error', 'ratio', 'ratio_error']

    nv_data = []
    header = True
    current_name = None
    nv = None
    with open(file, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            # Skip the header (first row)
            if header:
                header = False
                continue
            # Set up for a new NV if we're in a new block
            if row[0] != current_name: 
                # Append the current NV if there is one
                if current_name is not None:
                    nv_data.append(nv)
                current_name = row[0]
                nv = {}
                nv['name'] = current_name
                nv['marker'] = markers[marker_ind]
                nv['color'] = colors[marker_ind]
                marker_ind += 1
                # Initialize a new list for each column
                for column in columns:
                    nv[column] = []
            for ind in range(len(columns)):
                column = columns[ind]
                val = row[ind+1]
                if val == 'None':
                    val = None
                else:
                    val = float(val)
                nv[column].append(val)
    # Don't forget the last NV!
    nv_data.append(nv)
            
    return nv_data


def plot_gamma_omega_vs_angle(nv_data):

    plt.rcParams.update({'font.size': 18})  # Increase font size
    fig, axes_pack = plt.subplots(3, 1, figsize=(9,18))
    fig.set_tight_layout(True)
    for ax in axes_pack:
        ax.set_xlabel(r'Magnet angle, $\theta_{B}$ ($\degree$)')
        ax.set_xlim(-5, 95)
        ax.set_xticks(numpy.linspace(0,90,7))
        # ax.set_xlim(47, 50)
        
    # Ax-specific setup
    
    gamma_ax = axes_pack[0]
    gamma_ax.set_title('Gamma vs Angle')
    # gamma_ax.set_ylim(0.1, 0.2)
    gamma_ax.set_ylim(0.1, 0.3)
    gamma_ax.set_ylabel('Gamma (kHz)')
    
    omega_ax = axes_pack[1]
    omega_ax.set_title('Omega vs Angle')
    omega_ax.set_ylim(0.04, 0.08)
    omega_ax.set_ylabel('Omega (kHz)')
    
    ratio_ax = axes_pack[2]
    ratio_ax.set_title('Gamma/Omega vs Angle')
    # ratio_ax.set_ylim(1.5, 3.5)
    ratio_ax.set_ylim(1.5, 4.5)
    ratio_ax.set_ylabel('Ratio')
    
    for ind in range(len(nv_data)):
        
        nv = nv_data[ind]
        
        name = nv['name']
        if name not in ['NVE']:
            continue
        
        gammas = numpy.array(nv['gamma'])
        gamma_errors = numpy.array(nv['gamma_error'])
        
        omegas = numpy.array(nv['omega'])
        omega_errors = numpy.array(nv['omega_error'])
        
        ratios = numpy.array(nv['ratio'])
        ratio_errors = numpy.array(nv['ratio_error'])
        
        angles = numpy.array(nv['theta_B'])
        mask = angles != None
        if True in mask:
            
            gamma_ax.errorbar(angles[mask], gammas[mask],
                    yerr=gamma_errors[mask], linestyle='None', ms=9, lw=2.5,
                    marker=nv['marker'], color=nv['color'], label=nv['name'])
            omega_ax.errorbar(angles[mask], omegas[mask],
                    yerr=omega_errors[mask], linestyle='None', ms=9, lw=2.5,
                    marker=nv['marker'], color=nv['color'], label=nv['name'])
            ratio_ax.errorbar(angles[mask], ratios[mask],
                    yerr=ratio_errors[mask], linestyle='None', ms=9, lw=2.5,
                    marker=nv['marker'], color=nv['color'], label=nv['name'])
            
            # gamma_wavg = numpy.average(gammas[mask], weights=(1/gamma_errors[mask]**2))
            # gamma_ax.errorbar(angles[mask], gammas[mask]/gamma_wavg,
            #                   yerr=gamma_errors[mask]/gamma_wavg,
            #                   linestyle='None', marker='o', ms=9, lw=2.5)
            # omega_wavg = numpy.average(omegas[mask], weights=(1/omega_errors[mask]**2))
            # omega_ax.errorbar(angles[mask], omegas[mask]/omega_wavg,
            #                   yerr=omega_errors[mask]/omega_wavg,
            #                   linestyle='None', marker='o', ms=9, lw=2.5)
            
    gamma_ax.legend()


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
        # if name != 'NVB1':
        #     continue
        
        all_splittings = numpy.array(nv['all_splittings'])
        angles = numpy.array(nv['angle'])
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
            
    
def hist_gamma_omega(nv_data):
    
    plt.rcParams.update({'font.size': 18})  # Increase font size
    fig, axes_pack = plt.subplots(3, 1, figsize=(9,18))
    fig.set_tight_layout(True)
    for ax in axes_pack:
        ax.set_ylabel('Occurrences')
        
    # Ax-specific setup
    
    gamma_ax = axes_pack[0]
    gamma_ax.set_title('Gamma Histogram')
    # gamma_ax.set_ylim(0.1, 0.2)
    # gamma_ax.set_xlim(0.1, 0.3)
    gamma_ax.set_xlabel('Gamma (kHz)')
    
    omega_ax = axes_pack[1]
    omega_ax.set_title('Omega Histogram')
    # omega_ax.set_xlim(0.04, 0.08)
    omega_ax.set_xlabel('Omega (kHz)')
    
    ratio_ax = axes_pack[2]
    ratio_ax.set_title('Gamma/Omega Histogram')
    # ratio_ax.set_xlim(1.5, 3.5)
    # ratio_ax.set_xlim(1.5, 4.5)
    ratio_ax.set_xlabel('Ratio')
    
    all_gammas = []
    all_omegas = []
    all_ratios = []
    
    for ind in range(len(nv_data)):
        
        nv = nv_data[ind]
        
        name = nv['name']
        if name not in ['NVE']:
            continue
        
        all_gammas.extend(nv['gamma'])
        all_omegas.extend(nv['omega'])
        all_ratios.extend(nv['ratio'])
        
    all_gammas = numpy.array(all_gammas)
    all_omegas = numpy.array(all_omegas)
    all_ratios = numpy.array(all_ratios)
    
    num_bins = 10
    gamma_hist, gamma_bin_edges, gamma_patches = gamma_ax.hist(all_gammas, num_bins)
    omega_hist, omega_bin_edges, omega_patches = omega_ax.hist(all_omegas, num_bins)
    ratio_hist, ratio_bin_edges, ratio_patches = ratio_ax.hist(all_ratios, num_bins)
    

# %% Main


def main(nv_data, mode='both'):
    
    if mode not in ['splittings', 'angles', 'both']:
        print('Allowed modes are splittings, angles, or both.')
        return
    
    # %% Setup
    
    all_ratios = []
    all_ratio_errors = []

    # %% Plotting

    plt.rcParams.update({'font.size': 18})  # Increase font size
    if mode != 'both':
        fig, ax = plt.subplots(figsize=(6,5))
        axes_pack = [ax]  # Pack up to allow for common codepaths
        if mode == 'splittings':
            splittings_ax_ind = 0
        elif mode == 'angles':
            angles_ax_ind = 0
    else:
        fig, axes_pack = plt.subplots(1, 2, figsize=(12,5))
        splittings_ax_ind = 0
        angles_ax_ind = 1
    fig.set_tight_layout(True)
    
    # Splitting setup
    if mode in ['splittings', 'both']:
        ax = axes_pack[splittings_ax_ind]
        ax.set_xlabel(r'Splitting, $\Delta_{\pm}$ (MHz)')
        ax.set_xlim(-50, 1300)
        ax.set_ylabel(r'$\gamma / \Omega$')
        ax.set_ylim(0, 3.5)
            
    # Angles setup
    if mode in ['angles', 'both']:
        ax = axes_pack[angles_ax_ind]
        ax.set_xlabel(r'Magnet angle, $\theta_{B}$ ($\degree$)')
        # ax.set_xlim(-5, 95)
        ax.set_xlim(47, 49)
        # ax.set_xticks(numpy.linspace(0,90,7))
        ax.set_ylabel(r'$\gamma / \Omega$')
        ax.set_ylim(0, 3.5)

    for ind in range(len(nv_data)):
        
        nv = nv_data[ind]
        marker = markers[ind]
        color = colors[ind]  
        
        name = nv['name']
        # if name in ['NVA1', 'NVA2']:
        #     continue
        # if name != 'test':
        #     continue
        
        # Calculate ratios
        ratios = numpy.array(nv['ratio'])
        ratio_errors = numpy.array(nv['ratio_error'])
        all_ratios.extend(ratios)
        all_ratio_errors.extend(ratio_errors)
        
        # Plot splittings
        if mode in ['splittings', 'both']:
            ax = axes_pack[splittings_ax_ind]
            splittings = numpy.array(nv['splitting'])
            mask = splittings != None
            if True in mask:
                ax.errorbar(splittings[mask], ratios[mask],
                            yerr=ratio_errors[mask], label=name,
                            marker=marker, color=color, linestyle='None',
                            ms=9, lw=2.5)
    
        # Plot angles
        if mode in ['angles', 'both']:
            ax = axes_pack[angles_ax_ind]
            angles = numpy.array(nv['theta_B'])
            mask = angles != None
            if True in mask:
                print(angles[mask])
                print(ratios[mask])
                print(ratio_errors[mask])
                ax.errorbar(angles[mask], ratios[mask],
                            yerr=ratio_errors[mask], label=name,
                            marker=marker, color=color, linestyle='None',
                            ms=9, lw=2.5)

    all_ratios = numpy.array(all_ratios)
    all_ratio_errors = numpy.array(all_ratio_errors)
    
    wavg_ratio = numpy.average(all_ratios, weights=(1/all_ratio_errors**2))
    ste_ratio = numpy.sqrt(1/numpy.sum(all_ratio_errors**-2))
    print(all_ratios)
    print(all_ratio_errors)
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
    
    path = 'E:/Shared drives/Kolkowitz Lab Group/nvdata/papers/bulk_dq_relaxation/'
    file = path + 'compiled_data_markup.csv'
    nv_data = get_nv_data_csv(file)
    
    # main(nv_data)
    # plot_gamma_omega_vs_angle(nv_data)
    hist_gamma_omega(nv_data)
    # plot_splittings_vs_angle(nv_data)

