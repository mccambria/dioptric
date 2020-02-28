# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 17:04:19 2019

@author: matth
"""


# %% Imports


import csv
import numpy
from numpy import pi
import matplotlib.pyplot as plt
from analysis.extract_hamiltonian import calc_splitting
import scipy.stats as stats


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


def weighted_corrcoeff(x, y, y_errors=None):
    """
    Returns Pearson correlation coefficient for dependent variable y and
    independent variable x. Optionally weighted by squared errors on y 
    (variance weights).
    """
    
    # Create a mask for elements that are None in neither array
    x_mask = x != None
    y_mask = y != None
    mask = x_mask * y_mask
    
    if y_errors is not None:
        cov_mat = numpy.cov(x[mask].astype(float), y[mask].astype(float),
                            aweights=y_errors[mask]**-2)
    else:
        cov_mat = numpy.cov(x[mask].astype(float), y[mask].astype(float))
    
    return cov_mat[0,1] / numpy.sqrt(cov_mat[0,0]*cov_mat[1,1])
    
    
def correlations_OLD(nv_data):
    """
    Return Pearson product-moment correlation coefficients for various
    combinations of measured quantities.
    """
    
    mode = 'all'
    # mode = 'single nv'
    # mode = 'ensemble'
    
    # columns = ['res_minus', 'res_plus', 'splitting', 'mag_B',
    #            'theta_B', 'perp_B', 'contrast_minus', 'contrast_plus', 
    #            'rabi_minus', 'rabi_plus', 'gamma', 'gamma_error',
    #            'omega', 'omega_error', 'ratio', 'ratio_error']
    
    res_minus = []
    res_plus = []
    splitting = []
    mag_B = []
    theta_B = []
    perp_B = []
    perp_B_frac = []
    contrast_minus = []
    contrast_plus = []
    rabi_minus = []
    rabi_plus = []
    gamma = []
    gamma_error = []
    omega = []
    omega_error = []
    ratio = []
    ratio_error = []
    
    if mode == 'all':
        inclusion_check = lambda name: True
    elif mode == 'single nv':
        inclusion_check = lambda name: name not in ['NVE']
    elif mode == 'ensemble':
        inclusion_check = lambda name: name == 'NVE'
    
    for ind in range(len(nv_data)):
        
        nv = nv_data[ind]
        
        name = nv['name']
        if not inclusion_check(name):
            continue
        
        res_minus.extend(nv['res_minus'])
        res_plus.extend(nv['res_plus'])
        splitting.extend(nv['splitting'])
        mag_B.extend(nv['mag_B'])
        theta_B.extend(nv['theta_B'])
        perp_B.extend(nv['perp_B'])
        
        nv_perp_B = nv['perp_B']
        nv_mag_B = nv['mag_B']
        nv_perp_B_frac = []
        for ind in range(len(nv_perp_B)):
            if nv_perp_B[ind] is None:
               nv_perp_B_frac.append(None)
            else:
               nv_perp_B_frac.append(nv_perp_B[ind] / nv_mag_B[ind])
        perp_B_frac.extend(nv_perp_B_frac)
        
        contrast_minus.extend(nv['contrast_minus'])
        contrast_plus.extend(nv['contrast_plus'])
        rabi_minus.extend(nv['rabi_minus'])
        rabi_plus.extend(nv['rabi_plus'])
        gamma.extend(nv['gamma'])
        gamma_error.extend(nv['gamma_error'])
        omega.extend(nv['omega'])
        omega_error.extend(nv['omega_error'])
        ratio.extend(nv['ratio'])
        ratio_error.extend(nv['ratio_error'])
        
    res_minus = numpy.array(res_minus)
    res_plus = numpy.array(res_plus)
    splitting = numpy.array(splitting)
    mag_B = numpy.array(mag_B)
    theta_B = numpy.array(theta_B)
    perp_B = numpy.array(perp_B)
    perp_B_frac = numpy.array(perp_B_frac)
    contrast_minus = numpy.array(contrast_minus)
    contrast_plus = numpy.array(contrast_plus)
    rabi_minus = numpy.array(rabi_minus)
    rabi_plus = numpy.array(rabi_plus)
    gamma = numpy.array(gamma)
    gamma_error = numpy.array(gamma_error)
    omega = numpy.array(omega)
    omega_error = numpy.array(omega_error)
    ratio = numpy.array(ratio)
    ratio_error = numpy.array(ratio_error)
    
    # Calculate correlations
    p_corr_fun = lambda x,y: numpy.corrcoef(x,y)[0,1]  # Pearson
    s_corr_fun = lambda x,y: stats.spearmanr(x,y).correlation  # Spearman
    w_corr_fun = weighted_corrcoeff  # test
    
    title = '\nCorrelation coefficients for {} measurements, Pearson, Spearman, then weighted Pearson\n'
    print(title.format(mode))
    
    # rabi_plus, rabi_minus
    p_corr = p_corr_fun(rabi_plus, rabi_minus)
    s_corr = s_corr_fun(rabi_plus, rabi_minus)
    w_corr = w_corr_fun(rabi_plus, rabi_minus)
    print('rabi_plus, rabi_minus correlation: {:.2}, {:.2}, {:.2}'.format(p_corr, s_corr, w_corr))
    
    # contrast_plus, contrast_minus
    p_corr = p_corr_fun(contrast_plus, contrast_minus)
    s_corr = s_corr_fun(contrast_plus, contrast_minus)
    w_corr = w_corr_fun(contrast_plus, contrast_minus)
    print('contrast_plus, contrast_minus correlation: {:.2}, {:.2}, {:.2}'.format(p_corr, s_corr, w_corr))
    
    # contrast_plus, perp_B
    mask = perp_B != None
    p_corr = p_corr_fun(contrast_plus[mask].astype(float),
                        perp_B[mask].astype(float))
    s_corr = s_corr_fun(contrast_plus[mask].astype(float),
                        perp_B[mask].astype(float))
    w_corr = w_corr_fun(contrast_plus, perp_B)
    print('contrast_plus, perp_B correlation: {:.2}, {:.2}, {:.2}'.format(p_corr, s_corr, w_corr))
    
    # ratio, theta_B
    mask = theta_B != None
    p_corr = p_corr_fun(ratio[mask].astype(float),
                        theta_B[mask].astype(float))
    s_corr = s_corr_fun(ratio[mask].astype(float),
                        theta_B[mask].astype(float))
    w_corr = w_corr_fun(ratio, theta_B, ratio_error)
    print('ratio, theta_B correlation: {:.2}, {:.2}, {:.2}'.format(p_corr, s_corr, w_corr))
    
    # ratio, perp_B
    mask = perp_B != None
    p_corr = p_corr_fun(ratio[mask].astype(float),
                        perp_B[mask].astype(float))
    s_corr = s_corr_fun(ratio[mask].astype(float),
                        perp_B[mask].astype(float))
    w_corr = w_corr_fun(ratio, perp_B, ratio_error)
    print('ratio, perp_B correlation: {:.2}, {:.2}, {:.2}'.format(p_corr, s_corr, w_corr))
    
    # gamma, theta_B
    mask = theta_B != None
    p_corr = p_corr_fun(gamma[mask].astype(float),
                        theta_B[mask].astype(float))
    s_corr = s_corr_fun(gamma[mask].astype(float),
                        theta_B[mask].astype(float))
    w_corr = w_corr_fun(gamma, theta_B, gamma_error)
    print('gamma, theta_B correlation: {:.2}, {:.2}, {:.2}'.format(p_corr, s_corr, w_corr))
    
    # gamma, perp_B
    mask = perp_B != None
    p_corr = p_corr_fun(gamma[mask].astype(float),
                        perp_B[mask].astype(float))
    s_corr = s_corr_fun(gamma[mask].astype(float),
                        perp_B[mask].astype(float))
    w_corr = w_corr_fun(gamma, perp_B, gamma_error)
    print('gamma, perp_B correlation: {:.2}, {:.2}, {:.2}'.format(p_corr, s_corr, w_corr))
    
    # omega, theta_B
    mask = theta_B != None
    p_corr = p_corr_fun(omega[mask].astype(float),
                        theta_B[mask].astype(float))
    s_corr = s_corr_fun(omega[mask].astype(float),
                        theta_B[mask].astype(float))
    w_corr = w_corr_fun(omega, theta_B, omega_error)
    print('omega, theta_B correlation: {:.2}, {:.2}, {:.2}'.format(p_corr, s_corr, w_corr))
    
    # omega, perp_B
    mask = perp_B != None
    p_corr = p_corr_fun(omega[mask].astype(float),
                        perp_B[mask].astype(float))
    s_corr = s_corr_fun(omega[mask].astype(float),
                        perp_B[mask].astype(float))
    w_corr = w_corr_fun(omega, perp_B, omega_error)
    print('omega, perp_B correlation: {:.2}, {:.2}, {:.2}'.format(p_corr, s_corr, w_corr))
    
    
    
    # # contrast_plus, theta_B
    # mask = theta_B != None
    # p_corr = p_corr_fun(contrast_plus[mask].astype(float),
    #                     theta_B[mask].astype(float))
    # s_corr = s_corr_fun(contrast_plus[mask].astype(float),
    #                     theta_B[mask].astype(float))
    # print('contrast_plus, theta_B correlation: {:.2}, {:.2}'.format(p_corr, s_corr))
    
    # # ratio, contrast_minus
    # p_corr = p_corr_fun(ratio, contrast_minus)
    # s_corr = s_corr_fun(ratio, contrast_minus)
    # print('ratio, contrast_minus correlation: {:.2}, {:.2}'.format(p_corr, s_corr))
    
    # # ratio, contrast_plus
    # p_corr = p_corr_fun(ratio, contrast_plus)
    # s_corr = s_corr_fun(ratio, contrast_plus)
    # print('ratio, contrast_plus correlation: {:.2}, {:.2}'.format(p_corr, s_corr))
    
    # # ratio, rabi_minus
    # p_corr = p_corr_fun(ratio, rabi_minus)
    # s_corr = s_corr_fun(ratio, rabi_minus)
    # print('ratio, rabi_minus correlation: {:.2}, {:.2}'.format(p_corr, s_corr))
    
    # # ratio, rabi_plus
    # p_corr = p_corr_fun(ratio, rabi_plus)
    # s_corr = s_corr_fun(ratio, rabi_plus)
    # print('ratio, rabi_plus correlation: {:.2}, {:.2}'.format(p_corr, s_corr))
    
    # # ratio, splitting
    # p_corr = p_corr_fun(ratio, splitting)
    # s_corr = s_corr_fun(ratio, splitting)
    # print('ratio, splitting correlation: {:.2}, {:.2}'.format(p_corr, s_corr))
    
    # # ratio, perp_B_frac
    # mask = perp_B != None
    # p_corr = p_corr_fun(ratio[mask].astype(float),
    #                     perp_B_frac[mask].astype(float))
    # s_corr = s_corr_fun(ratio[mask].astype(float),
    #                     perp_B_frac[mask].astype(float))
    # print('ratio, perp_B_frac correlation: {:.2}, {:.2}'.format(p_corr, s_corr))
    
    # # ratio, res_minus
    # p_corr = p_corr_fun(ratio, res_minus)
    # s_corr = s_corr_fun(ratio, res_minus)
    # print('ratio, res_minus correlation: {:.2}, {:.2}'.format(p_corr, s_corr))
    
    # # ratio, res_plus
    # p_corr = p_corr_fun(ratio, res_plus)
    # s_corr = s_corr_fun(ratio, res_plus)
    # print('ratio, res_plus correlation: {:.2}, {:.2}'.format(p_corr, s_corr))
    return
    
    
def correlations(nv_data):
    """
    Return Pearson product-moment correlation coefficients for various
    combinations of measured quantities.
    """
    
    # mode = 'all'
    # mode = 'single nv'
    mode = 'ensemble'
    
    columns = ['res_minus', 'res_plus', 'splitting', 'mag_B',
                'theta_B', 'perp_B', 'contrast_minus', 'contrast_plus', 
                'rabi_minus', 'rabi_plus', 'gamma', 'omega', 'ratio', ]
    error_columns = ['gamma', 'omega', 'ratio']
    
    res_minus = []
    res_plus = []
    splitting = []
    mag_B = []
    theta_B = []
    perp_B = []
    perp_B_frac = []
    contrast_minus = []
    contrast_plus = []
    rabi_minus = []
    rabi_plus = []
    gamma = []
    gamma_error = []
    omega = []
    omega_error = []
    ratio = []
    ratio_error = []
    
    if mode == 'all':
        inclusion_check = lambda name: True
    elif mode == 'single nv':
        inclusion_check = lambda name: name not in ['NVE']
    elif mode == 'ensemble':
        inclusion_check = lambda name: name == 'NVE'
    
    for ind in range(len(nv_data)):
        
        nv = nv_data[ind]
        
        name = nv['name']
        if not inclusion_check(name):
            continue
        
        res_minus.extend(nv['res_minus'])
        res_plus.extend(nv['res_plus'])
        splitting.extend(nv['splitting'])
        mag_B.extend(nv['mag_B'])
        theta_B.extend(nv['theta_B'])
        perp_B.extend(nv['perp_B'])
        
        nv_perp_B = nv['perp_B']
        nv_mag_B = nv['mag_B']
        nv_perp_B_frac = []
        for ind in range(len(nv_perp_B)):
            if nv_perp_B[ind] is None:
               nv_perp_B_frac.append(None)
            else:
               nv_perp_B_frac.append(nv_perp_B[ind] / nv_mag_B[ind])
        perp_B_frac.extend(nv_perp_B_frac)
        
        contrast_minus.extend(nv['contrast_minus'])
        contrast_plus.extend(nv['contrast_plus'])
        rabi_minus.extend(nv['rabi_minus'])
        rabi_plus.extend(nv['rabi_plus'])
        gamma.extend(nv['gamma'])
        gamma_error.extend(nv['gamma_error'])
        omega.extend(nv['omega'])
        omega_error.extend(nv['omega_error'])
        ratio.extend(nv['ratio'])
        ratio_error.extend(nv['ratio_error'])
        
    res_minus = numpy.array(res_minus)
    res_plus = numpy.array(res_plus)
    splitting = numpy.array(splitting)
    mag_B = numpy.array(mag_B)
    theta_B = numpy.array(theta_B)
    perp_B = numpy.array(perp_B)
    perp_B_frac = numpy.array(perp_B_frac)
    contrast_minus = numpy.array(contrast_minus)
    contrast_plus = numpy.array(contrast_plus)
    rabi_minus = numpy.array(rabi_minus)
    rabi_plus = numpy.array(rabi_plus)
    gamma = numpy.array(gamma)
    gamma_error = numpy.array(gamma_error)
    omega = numpy.array(omega)
    omega_error = numpy.array(omega_error)
    ratio = numpy.array(ratio)
    ratio_error = numpy.array(ratio_error)
    
    # Calculate correlations
    corr_fun = weighted_corrcoeff
    
    for x_name in columns:
        print()
        
        for y_name in columns:
            
            x_column = eval(x_name)
            y_column = eval(y_name)
            
            x_error = None
            if x_name in error_columns:
                x_error = eval('{}_error'.format(x_name))
            y_error = None
            if y_name in error_columns:
                y_error = eval('{}_error'.format(y_name))
                
            # If we have errors on both columns, add in quadrature. This is
            # just what my intuition says is correct! I haven't checked this.
            if (x_error is not None) and (y_error is not None):
                error = numpy.sqrt(x_error**2 + y_error**2)
            elif x_error is not None:
                error = x_error
            elif y_error is not None:
                error = y_error
            else:
                error = None
                
            corrcoeff = corr_fun(x_column, y_column, error)
            # print('{}, {}: {:.2}'.format(x_name, y_name, corrcoeff))
            print(corrcoeff)
        


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
    # hist_gamma_omega(nv_data)
    correlations(nv_data)
    # plot_splittings_vs_angle(nv_data)

