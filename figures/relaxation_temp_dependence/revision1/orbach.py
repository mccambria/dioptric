# -*- coding: utf-8 -*-
"""
Reproduce Jarmola 2012 temperature scalings

Created on Fri Jun 26 17:40:09 2020

@author: matth
"""


# %% Imports


import numpy
import matplotlib.pyplot as plt
import csv
import matplotlib.patches as patches
import matplotlib.lines as mlines
from scipy.optimize import curve_fit
import pandas as pd


# %% Constants


Boltzmann = 8.617e-2  # meV / K
# from scipy.constants import Boltzmann  # J / K

# Rate coefficients in s^-1 from Jarmola. Not accurate right now
# A_1 = 0.007  # Constant for S3
# A_2 = 5.10e2  # Orbach
# # A_2 = 1.7e3  # Test
# A_3 = 1.38e-11  # T^5
# # A_3 = 2.5e-11  # Test
# A_4 = 4.3e-6  # T^3
# A_7 = 2.55e-20

A_1 = 0.007  # Constant for S3
A_2 = 5.10e2  # Orbach
# A_2 = 1.7e3  # Test
A_3 = 1.38e-11  # T^5
# A_3 = 2.5e-11  # Test
A_4 = 4.3e-6  # T^3
A_7 = 2.55e-20

# Quasilocalized mode activation energy
quasi = 76.0  # meV, empirical fit
# quasi = 69.0  # meV, empirical fit
# quasi = 65.0  # meV, quasilocalized resonance
# quasi = 1.17e-20  # J

ms = 7
lw = 1.75

gamma_face_color = '#CC99CC'
gamma_edge_color = '#993399'
omega_face_color = '#FFCC33'
omega_edge_color = '#FF9933'
ratio_face_color = '#FB9898'
ratio_edge_color = '#EF2424'

sample_column_title = 'Sample'
skip_column_title = 'Skip'
temp_column_title = 'Nominal temp (K)'
# temp_column_title = 'ZFS temp (K)'
omega_column_title = 'Omega (s^-1)'
omega_err_column_title = 'Omega err (s^-1)'
gamma_column_title = 'gamma (s^-1)'
gamma_err_column_title = 'gamma err (s^-1)'


# %% Processes and sum functions


def bose(energy, temp):
    return 1 / (numpy.exp(energy / (Boltzmann * temp)) - 1)

def orbach(temp):
    """
    This is for quasilocalized phonons interacting by a Raman process, which
    reproduces an Orbach scaling even though it's not really an Orbach.
    process. As such, the proper scaling is
    n(omega)(n(omega)+1) approx n(omega) for omega << kT
    """
    # return A_2 * bose(quasi, temp) * (bose(quasi, temp) + 1)
    return A_2 * bose(quasi, temp)
    # return A_2 / (numpy.exp(quasi / (Boltzmann * temp)))

def orbach_free(temp, coeff, activation):
    return coeff * bose(activation, temp)

def raman(temp):
    return A_3 * (temp**5)

def test_T_cubed(temp):
    return A_4 * (temp**3)

def test_T_seventh(temp):
    return A_7 * (temp**7)

def omega_calc(temp):
    """Using fit from May 12th, 2021"""
    return orbach_T5_free(temp, 510, 76.0, 1.38e-11)
    # return A_1 + test_T_cubed(temp) + raman(temp)
    # return (A_1 + orbach(temp) + raman(temp) + test_T_seventh(temp)) / 3

def orbach_T5_free(temp, coeff_orbach, activation, coeff_T5):
    # activation = 78
    # coeff_T5 = 0
    return (coeff_orbach * bose(activation, temp)) + (coeff_T5 * temp**5)

def T5_free(temp, coeff_T5):
    return coeff_T5 * temp**5

def gamma_calc(temp):
    """Using fit from May 12th, 2021"""
    return orbach_free(temp, 2200, 75.2)


# %% Other functions


def fit_omega_orbach_T5(data_points):

    temps = []
    omegas = []
    omega_errs = []
    for point in data_points:
        if point[omega_column_title] is not None:
            temps.append(point[temp_column_title])
            omegas.append(point[omega_column_title])
            omega_errs.append(point[omega_err_column_title])

    # fit_func = orbach_free
    # init_params = (A_2 / 3, quasi)

    fit_func = orbach_T5_free
    init_params = (A_2 / 3, quasi, A_3 / 3)

    # fit_func = T5_free
    # init_params = (2 * A_3 / 3, )

    num_params = len(init_params)
    popt, pcov = curve_fit(fit_func, temps, omegas, p0=init_params,
                           sigma=omega_errs, absolute_sigma=True,
                           bounds=([0]*num_params,[numpy.inf]*num_params),
                           method='dogbox')

    return popt, pcov, fit_func


def fit_gamma_orbach(data_points):

    temps = []
    gammas = []
    gamma_errs = []
    for point in data_points:
        if point[gamma_column_title] is not None:
            temps.append(point[temp_column_title])
            gammas.append(point[gamma_column_title])
            gamma_errs.append(point[gamma_err_column_title])

    fit_func = orbach_free
    init_params = ((2/3) * A_2, quasi)

    # fit_func = orbach_T5_free
    # init_params = ((2/3) * A_2, quasi, 1E-11)

    # fit_func = T5_free
    # init_params = ((2/3) * A_3)

    num_params = len(init_params)
    popt, pcov = curve_fit(fit_func, temps, gammas, p0=init_params,
                           sigma=gamma_errs, absolute_sigma=True,
                           bounds=([0]*num_params,[numpy.inf]*num_params),
                           method='dogbox')

    return popt, pcov, fit_func


def get_data_points_csv(file):

    # Marker and color combination to distinguish samples
    marker_ind = 0
    markers = ['o', 's', '^', 'X', ]

    data_points = []
    samples = []
    sample_markers = {}
    header = True
    with open(file, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            # Create columns from the header (first row)
            if header:
                columns = row[1:]
                header = False
                continue
            point = {}
            sample = row[0]
            if sample not in samples:
                sample_markers[sample] = markers[marker_ind]
                marker_ind += 1
                samples.append(sample)
            point['marker'] = sample_markers[sample]
            point[sample_column_title] = sample
            for ind in range(len(columns)):
                column = columns[ind]
                raw_val = row[1+ind]
                if raw_val == 'TRUE':
                    val = True
                else:
                    try:
                        val = eval(raw_val)
                    except Exception:
                        val = raw_val
                point[column] = val
            if not point[skip_column_title]:
                data_points.append(point)

    return data_points


# %% Main


def main(file_name, path, plot_type, rates_to_plot,
         temp_range=[190, 310], xscale='linear', yscale='linear'):

    # %% Setup

    plt.rcParams['text.latex.preamble'] = [
        r'\usepackage{physics}',
        r'\usepackage{sfmath}',
        r'\usepackage{upgreek}',
        r'\usepackage{helvet}',
       ]
    plt.rcParams.update({'font.size': 11.25})
    plt.rcParams.update({'font.family': 'sans-serif'})
    plt.rcParams.update({'font.sans-serif': ['Helvetica']})
    plt.rc('text', usetex=True)
    file_path = path + '{}.xlsx'.format(file_name)
    csv_file_path = path + '{}.csv'.format(file_name)

    file = pd.read_excel(file_path)
    file.to_csv(csv_file_path, index=None, header=True)

    data_points = get_data_points_csv(csv_file_path)

    min_temp = temp_range[0]
    max_temp = temp_range[1]

    # temp_linspace = numpy.linspace(5, 600, 1000)
    temp_linspace = numpy.linspace(min_temp, max_temp, 1000)
    # temp_linspace = numpy.linspace(5, 300, 1000)
    # temp_linspace = numpy.linspace(5, 5000, 1000)
    fig, ax = plt.subplots()
    fig.set_tight_layout(True)
    # ax.set_title('Relaxation rates')

    # Fit to Omega
    omega_popt, omega_pcov, omega_fit_func = fit_omega_orbach_T5(data_points)
    omega_lambda = lambda temp: omega_fit_func(temp, *omega_popt)
    # omega_lambda = lambda temp: omega_calc(temp)
    # omega_popt[2] = 0
    # omega_popt[1] = 78
    print(omega_popt)
    if (plot_type == 'rates') and (rates_to_plot in ['both', 'Omega']):
        ax.plot(temp_linspace, omega_lambda(temp_linspace),
                label=r'$\Omega$ fit', color=omega_edge_color)
        # Plot Jarmola 2012 Eq. 1 for S3
        # ax.plot(temp_linspace, omega_calc(temp_linspace),
        #         label=r'$\Omega$ fit', color=omega_edge_color)

    # Fit to gamma
    gamma_popt, gamma_pcov, gamma_fit_func = fit_gamma_orbach(data_points)
    gamma_lambda = lambda temp: gamma_fit_func(temp, *gamma_popt)
    # gamma_lambda = lambda temp: gamma_calc(temp)
    # gamma_popt[1] = omega_popt[1]
    print(gamma_popt)
    if (plot_type == 'rates') and (rates_to_plot in ['both', 'gamma']):
        ax.plot(temp_linspace, gamma_lambda(temp_linspace),
                label=r'$\gamma$ fit', color=gamma_edge_color)

    ratio_lambda = lambda temp: gamma_lambda(temp_linspace) / omega_lambda(temp_linspace)
    if plot_type == 'ratio_fits':
        ax.plot(temp_linspace, ratio_lambda(temp_linspace),
                label=r'$\gamma/\Omega$', color=gamma_edge_color)


    # ax.plot(temp_linspace, orbach(temp_linspace) * 0.7, label='Orbach')
    # ax.plot(temp_linspace, raman(temp_linspace)/3, label='Raman')

    ax.set_xlabel(r'T (K)')
    if plot_type == 'rates':
        ax.set_ylabel(r'Relaxation rates (s$^{-1}$)')
    elif plot_type == 'ratios':
        ax.set_ylabel(r'Ratios')
    elif plot_type == 'ratio_fits':
        ax.set_ylabel(r'Ratio of fits')
    elif plot_type == 'residuals':
        ax.set_ylabel(r'Residuals (s$^{-1}$)')
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.set_xlim(min_temp, max_temp)
    # ax.set_ylim(1e-2, 1e4)
    # ax.set_ylim(-10, 130)
    # ax.set_ylim(-10, 800)
    # ax.set_ylim(2e-3, None)

    # ind in range(len(nv_data)):

    #     nv = nv_data[ind]

    # %% Plot the points

    samples = []
    markers = []

    for point in data_points:

        # print(point)
        sample = point[sample_column_title]
        marker = point['marker']

        if sample not in samples:
            samples.append(sample)
        if marker not in markers:
            markers.append(marker)

        temp = point[temp_column_title]

        if plot_type in ['rates', 'residuals']:
            # Omega
            rate = point[omega_column_title]
            if (rate is not None) and (rates_to_plot in ['both', 'Omega']):
                if plot_type == 'rates':
                    val = rate
                elif plot_type == 'residuals':
                    val = rate - omega_lambda(temp)
                ax.errorbar(temp, val,
                            yerr=point[omega_err_column_title],
                            label=r'$\Omega$', marker=marker,
                            color=omega_edge_color,
                            markerfacecolor=omega_face_color,
                            linestyle='None', ms=ms, lw=lw)
            # gamma
            rate = point[gamma_column_title]
            if (rate is not None) and (rates_to_plot in ['both', 'gamma']):
                if plot_type == 'rates':
                    val = rate
                elif plot_type == 'residuals':
                    val = rate - gamma_lambda(temp)
                ax.errorbar(temp, val,
                            yerr=point[gamma_err_column_title],
                            label= r'$\gamma$', marker=marker,
                            color=gamma_edge_color,
                            markerfacecolor=gamma_face_color,
                            linestyle='None', ms=ms, lw=lw)

        elif plot_type == 'ratios':
            omega_val = point[omega_column_title]
            omega_err = point[omega_err_column_title]
            gamma_val = point[gamma_column_title]
            gamma_err = point[gamma_err_column_title]
            if (omega_val is not None) and (gamma_val is not None):
                ratio = gamma_val / omega_val
                ratio_err = ratio*numpy.sqrt((omega_err/omega_val)**2 + (gamma_err/gamma_val)**2)
                ax.errorbar(temp, ratio,
                            yerr=ratio_err,
                            label= r'$\gamma/\Omega$', marker=marker,
                            color=ratio_edge_color,
                            markerfacecolor=ratio_face_color,
                            linestyle='None', ms=ms, lw=lw)

    # %% Legend

    leg1 = None

    if plot_type in ['rates', 'residuals']:
        omega_patch = patches.Patch(label=r'$\Omega$',
                        facecolor=omega_face_color, edgecolor=omega_edge_color)
        gamma_patch = patches.Patch(label=r'$\gamma$',
                        facecolor=gamma_face_color, edgecolor=gamma_edge_color)
        leg1 = ax.legend(handles=[omega_patch, gamma_patch], loc='upper left',
                          title='Rates')

    elif plot_type == 'ratios':
        ratio_patch = patches.Patch(label=r'$\gamma/\Omega$',
                        facecolor=ratio_face_color, edgecolor=ratio_edge_color)
        leg1 = ax.legend(handles=[ratio_patch], loc='upper left')

    # Samples
    if plot_type in ['rates', 'ratios', 'residuals']:
        sample_patches = []
        for ind in range(len(samples)):
            patch = mlines.Line2D([], [], color='black', marker=markers[ind],
                              linestyle='None', markersize=ms, label=samples[ind])
            sample_patches.append(patch)
        x_loc = 0.16
        ax.legend(handles=sample_patches, loc='upper left', title='Samples',
                  bbox_to_anchor=(x_loc, 1.0))

    if leg1 is not None:
        ax.add_artist(leg1)


# %% Run the file


if __name__ == '__main__':

    plot_type = 'rates'
    # plot_type = 'ratios'
    # plot_type = 'ratio_fits'
    # plot_type = 'residuals'

    rates_to_plot = 'both'
    # rates_to_plot = 'Omega'
    # rates_to_plot = 'gamma'
    
    temp_range = [140, 310]
    xscale = 'linear'
    yscale = 'log'

    file_name = 'compiled_data'
    # file_name = 'compiled_data-test'
    path = 'E:/Shared drives/Kolkowitz Lab Group/nvdata/paper_materials/relaxation_temp_dependence/'
    
    main(file_name, path, plot_type, rates_to_plot,
         temp_range, xscale, yscale)
