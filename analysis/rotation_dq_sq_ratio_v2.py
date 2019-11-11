# -*- coding: utf-8 -*-
"""Calculate the ratio of dq/sq allowabilities (matrix elements squared) for a
set of resonances with a rotated B field and the ratio at an aligned B field
with the same splitting

Created on Sun Jun 16 11:22:40 2019

@author: mccambria
"""


# %% Imports

import analysis.extract_hamiltonian as extract_hamiltonian
from scipy.optimize import minimize_scalar
import numpy
from numpy import pi
import matplotlib.pyplot as plt
from scipy import integrate


# %% Constants


h_bar = 1.055e-34  # J s
gyromagnetic = 2.8e6  # Hz / G
g_mu_B = gyromagnetic * h_bar  # J / G


# %% Functions


def find_mag_B_splitting_objective(x, splitting,
                                   theta_B, par_Pi, perp_Pi, phi_B, phi_Pi):
    calculated_res_pair = extract_hamiltonian.calc_res_pair(x,
                                    theta_B, par_Pi, perp_Pi, phi_B, phi_Pi)
    calculated_splitting = calculated_res_pair[1] - calculated_res_pair[0]
    return (splitting - calculated_splitting)**2


def calc_dq_factor(theta_B, phi_B, mag_B, popt):

    # mag_B cancels, but if it's too small there are significant rounding errors
    noise_params = (5.0, theta_B, phi_B)
    noise_hamiltonian = extract_hamiltonian.calc_B_hamiltonian(*noise_params)

    mat_els = extract_hamiltonian.calc_matrix_elements(noise_hamiltonian,
                                                         mag_B, *popt)

    mat_factors = [numpy.abs(el)**2 for el in mat_els]
    return mat_factors[2]


def calc_dq_factor_surface(noise_theta_B, noise_phi_B, mag_B, popt):

    val = calc_dq_factor(noise_theta_B, noise_phi_B, mag_B, popt)
    val *= numpy.sin(noise_theta_B)
    return val


def calc_rate_factor_surface(noise_theta_B, noise_phi_B, mag_B, popt, ind,
                             noise_power = 100.0):
    """el: 0 for zero to low, 1 for zero to high, 2 for low to high
    """

    noise_params = (noise_power, noise_theta_B, noise_phi_B)
    noise_hamiltonian = extract_hamiltonian.calc_B_hamiltonian(*noise_params)

    mat_els = extract_hamiltonian.calc_matrix_elements(noise_hamiltonian,
                                                       mag_B, *popt)

    mat_factors = [numpy.abs(el)**2 for el in mat_els]

    val = mat_factors[ind]
    val *= numpy.sin(noise_theta_B)
    return val


def calc_Pi_factor_surface(noise_phi, noise_theta, mag_B, popt, ind,
                           noise_power = 1.0):
    """el: 0 for zero to low, 1 for zero to high, 2 for low to high
    """

    noise_params = (noise_power, noise_phi, noise_theta)
    noise_hamiltonian = extract_hamiltonian.calc_Pi_hamiltonian(*noise_params)

    mat_els = extract_hamiltonian.calc_matrix_elements(noise_hamiltonian,
                                                       mag_B, *popt)

    mat_factors = [numpy.abs(el)**2 for el in mat_els]

    val = mat_factors[ind]
    return val


# %% Main


def mag_B_for_rate(name, res_descs, compare_res_desc, meas_rate):
    popt = extract_hamiltonian.main(name, res_descs)
    dummy_mag_B = 10.0
    # MHz**2
    dq_mat_factor, _ = integrate.dblquad(calc_rate_factor_surface,
                                         0, 2*pi, lambda x: 0, lambda x: pi,
                                         args=(1.000, popt, 2, dummy_mag_B))
    dq_mat_factor /= (dummy_mag_B**2)  # dimensionless

    # meas_rate = (g_mu_B * mag_B)**2 (2 * pi / h_bar) * sq_mat_factor * delta
    mag_B = numpy.sqrt(meas_rate / (2 * pi * h_bar * dq_mat_factor)) / gyromagnetic
    mag_B = numpy.sqrt(meas_rate / (2 * pi * dq_mat_factor)) / gyromagnetic
    print(mag_B)


def rate_factor_plot_func_B(name, res_descs):
    mag_Bs = numpy.linspace(0, 1.000, 100)
    noise_mag_Bs = numpy.linspace(1, 100, 100)
    # popt = (0.0, 0.0, 0.0, 0.0, 0.0)
    popt = extract_hamiltonian.main(name, res_descs)

    rates = []
    for val in noise_mag_Bs:
        rate, _ = integrate.dblquad(calc_rate_factor_surface,
                                    0, 2*pi, lambda x: 0, lambda x: pi,
                                    args=(1.000, popt, 0, val))
        rates.append(rate)

    fig, ax = plt.subplots()
    fig.set_tight_layout(True)
    ax.plot(mag_Bs, rates)


def dq_vs_sq_rates(name, res_descs, compare_res_desc):
    popt = extract_hamiltonian.main(name, res_descs)
    # popt = (0, 0, 0.1, 0, 0)
    # mag_B = extract_hamiltonian.find_mag_B(compare_res_desc, *popt)
    mag_B = 0.0
    # print(mag_B)
    # print(popt)
    # return

    # noise_power_amp = 100

    # noise_power = noise_power_amp/(compare_res_desc[1]**2)
    # print(noise_power)
    zero_to_low_integral, zero_to_low_err = integrate.dblquad(
                                        calc_rate_factor_surface,
                                        0, 2*pi, lambda x: 0, lambda x: pi,
                                        args=(mag_B, popt, 0))
    # zero_to_low_integral *= noise_power

    # noise_power = noise_power_amp/(compare_res_desc[2]**2)
    # print(noise_power)
    zero_to_high_integral, zero_to_high_err = integrate.dblquad(
                                        calc_rate_factor_surface,
                                        0, 2*pi, lambda x: 0, lambda x: pi,
                                        args=(mag_B, popt, 1))

    # noise_power = noise_power_amp/((compare_res_desc[2]-compare_res_desc[1])**2)
    # print(noise_power)
    low_to_high_integral, low_to_high_err = integrate.dblquad(
                                        calc_rate_factor_surface,
                                        0, 2*pi, lambda x: 0, lambda x: pi,
                                        args=(mag_B, popt, 2))
    # low_to_high_integral *= noise_power

    print('zero_to_low_integral: {}'.format(zero_to_low_integral))
    print('zero_to_high_integral: {}'.format(zero_to_high_integral))
    print('low_to_high_integral: {}'.format(low_to_high_integral))
    ratio = zero_to_low_integral / low_to_high_integral
    print('ratio: {}'.format(ratio))


def main(name, res_descs, aligned_res_desc, rotated_res_desc):
    """When you run the file, we'll call into main, which should contain the
    body of the script.
    """

    # Get the aligned Hamiltonian parameters
    # popt = [theta_B, par_Pi, perp_Pi, phi_B, phi_Pi]
    aligned_popt = extract_hamiltonian.main(name, res_descs)
    print(aligned_popt)

    # Find mag_B at the point we misaligned the field
    rotated_mag_B = extract_hamiltonian.find_mag_B(aligned_res_desc,
                                                   *aligned_popt)


    # Get the rotated Hamiltonian parameters
    theta_B, phi_B = extract_hamiltonian.find_B_orientation(rotated_res_desc,
                                            rotated_mag_B, aligned_popt[1],
                                            aligned_popt[2], aligned_popt[4])
#    rotated_popt = (theta_B, aligned_popt[1], aligned_popt[2],
#                    phi_B, aligned_popt[4])
    rotated_popt = (theta_B, aligned_popt[1], aligned_popt[2],
                    aligned_popt[3], aligned_popt[4])
    print(rotated_popt)


    # Find the mag_B for an equivalent splitting of the aligned Hamiltonian
    rotated_splitting = rotated_res_desc[2] - rotated_res_desc[1]
    args = (rotated_splitting, *aligned_popt)
    result = minimize_scalar(find_mag_B_splitting_objective, bounds=(0, 1.0),
                             args=args, method='bounded')
    aligned_mag_B = result.x

    aligned_args = (aligned_mag_B, aligned_popt)
    aligned_integral, al_err = integrate.dblquad(calc_dq_factor_surface,
                                         0, 2*pi, lambda x: 0, lambda x: pi,
                                         args=aligned_args)

    rotated_args = (rotated_mag_B, rotated_popt)
    rotated_integral, rot_err = integrate.dblquad(calc_dq_factor_surface,
                                          0, 2*pi, lambda x: 0, lambda x: pi,
                                          args=rotated_args)

    ratio = aligned_integral / rotated_integral
    print('Expected ratio: {}'.format(ratio))


def main_plot(name, res_descs, aligned_res_desc):
    """When you run the file, we'll call into main, which should contain the
    body of the script.
    """

    # Get the aligned Hamiltonian parameters
    # popt = [theta_B, par_Pi, perp_Pi, phi_B, phi_Pi]
    aligned_popt = extract_hamiltonian.main(name, res_descs)

    # Find mag_B at the point we misaligned the field
    rotated_mag_B = extract_hamiltonian.find_mag_B(aligned_res_desc,
                                                   *aligned_popt)

    rotated_popt = numpy.copy(aligned_popt)
    mag_Bs = numpy.linspace(0,1, 100)
    angles = numpy.linspace(0, pi/2, 100)
    ratios = []
    for angle in angles:
        rotated_popt[0] = angle
        res_pair = extract_hamiltonian.calc_res_pair(rotated_mag_B, *rotated_popt)
        rotated_res_desc = [rotated_mag_B, res_pair[0], res_pair[1]]

        rotated_splitting = rotated_res_desc[2] - rotated_res_desc[1]
        zero_field_splitting = res_descs[0][2] - res_descs[0][1]
        if rotated_splitting < zero_field_splitting:
            # There is no mag_B that will give us the same splitting so stop
            ratios.append(numpy.nan)
            continue

        # Find the mag_B for an equivalent splitting of the aligned Hamiltonian
        args = (rotated_splitting, *aligned_popt)
        result = minimize_scalar(find_mag_B_splitting_objective, bounds=(0, 1.0),
                                 args=args, method='bounded')
        aligned_mag_B = result.x

        aligned_args = (aligned_mag_B, aligned_popt)
        aligned_integral, al_err = integrate.dblquad(calc_dq_factor_surface,
                                             0, 2*pi, lambda x: 0, lambda x: pi,
                                             args=aligned_args)

        rotated_args = (rotated_mag_B, rotated_popt)
        rotated_integral, rot_err = integrate.dblquad(calc_dq_factor_surface,
                                              0, 2*pi, lambda x: 0, lambda x: pi,
                                              args=rotated_args)

        ratios.append(aligned_integral / rotated_integral)

    fig, ax = plt.subplots()
    ax.plot(angles, ratios)
    ax.set_ylim(0, 1)


def main_plot_rot(name, res_descs):
    """When you run the file, we'll call into main, which should contain the
    body of the script.
    """

    # [theta_B, par_Pi, perp_Pi, phi_B, phi_Pi]
    popt = extract_hamiltonian.main(name, res_descs)
    popt = list(popt)
    popt[0] = 0.0
    popt[2] = 0.0
    print(popt)
    mag_B = 0.1
    # print(extract_hamiltonian.calc_eigenvectors(mag_B, *popt))
    # return

    gammas = []
    phi_Pis = numpy.linspace(0, 2*pi, 100)
    for ind in range(len(phi_Pis)):
        phi_Pi = phi_Pis[ind]
        popt[4] = phi_Pi
        aligned_integral, al_err = integrate.quad(calc_Pi_factor_circle,
                                              0, 2*pi, args=(mag_B, popt, 2))
        gammas.append(aligned_integral)

    fig, ax = plt.subplots()
    fig.set_tight_layout(True)
    ax.plot(phi_Pis, gammas)


def main_plot_paper(name, res_descs,
                    meas_splittings=None, meas_gammas=None):
    """When you run the file, we'll call into main, which should contain the
    body of the script.
    """

    # [theta_B, par_Pi, perp_Pi, phi_B, phi_Pi]
    popt = extract_hamiltonian.main(name, res_descs)
    popt = list(popt)
    # popt[0] = 0.0
    # popt[2] = 0.0
    print(popt)
    # return
    # popt[0] = 0
    # popt[2] = 0.02
    # popt[4] = pi/4
    # print(extract_hamiltonian.calc_eigenvectors(0.0, *popt))
    # return
    
    meas_splittings /= 1000
    noise_func = calc_Pi_factor_surface
    # noise_func = calc_B_factor_surface

    popt = numpy.copy(popt)
    gamma_bs = []
    splittings = []
    empiricals = []
    empirical_scaling = -2.0
    # mag_Bs = numpy.linspace(0.001, 1.0, 100)
    mag_Bs = numpy.logspace(-3, 0.0, 100)
    for ind in range(len(mag_Bs)):
        mag_B = mag_Bs[ind]
        splitting = extract_hamiltonian.calc_splitting(mag_B, *popt)
        splittings.append(splitting)
        noise_mag = 3.5 + splitting**-2
        aligned_integral, al_err = integrate.dblquad(noise_func, 0, 2*pi,
                                                 lambda x: 0, lambda x: pi,
                                                 args=(mag_B, popt, 2))
        if ind == 0:
            # scaling = numpy.average(meas_gammas[0:2]) / aligned_integral
            # coeff = (10 * meas_gammas[0] + meas_gammas[1]) / 5
            coeff = meas_gammas[1]
            scaling = coeff / aligned_integral
            empirical_coeff = 40 / (splitting**empirical_scaling)
        gamma_b = noise_mag * aligned_integral * 10**-5
        gamma_bs.append(gamma_b)
        # gamma_bs.append(aligned_integral)
        empirical = empirical_coeff * (splitting**empirical_scaling)
        empiricals.append(empirical + 0.5)
        # print(extract_hamiltonian.calc_eigenvectors(mag_B, *popt))

    fig, ax = plt.subplots()
    fig.set_tight_layout(True)
    # ax.plot(splittings, gamma_bs)
    ax.loglog(splittings, gamma_bs)
    ax.loglog(splittings, empiricals)
    # ax.set_ylim(0, 10)
    ax.scatter(meas_splittings, meas_gammas)


# %% Run the file


# The __name__ variable will only be '__main__' if you run this file directly.
# This allows a file's functions, classes, etc to be imported without running
# the script that you set up here.
if __name__ == '__main__':

    # Set up your parameters to be passed to main here
    # name = 'nv1_2019_05_10'
    # res_descs = [[0.0, 2.8537, 2.8751],
    #               [None, 2.8554, 2.8752],
    #               [None, 2.8512, 2.8790],
    #               [None, 2.8520, 2.8800],
    #               [None, 2.8536, 2.8841],
    #               [None, 2.8496, 2.8823],
    #               [None, 2.8396, 2.8917],
    #               [None, 2.8198, 2.9106],  # Reference for misaligned T1
    #               [None, 2.8166, 2.9144],
    #               [None, 2.8080, 2.9240],
    #               [None, 2.7357, 3.0037],
    #               [None, 2.6310, 3.1547],  # Reference for misaligned T1
    #               [None, 2.6061, 3.1678],
    #               [None, 2.6055, 3.1691],
    #               [None, 2.4381, 3.4531],  # 0,-1 and 0,+1 omegas
    #               [None, 2.4371, 3.4539],
    #               ]
    # meas_splittings = numpy.array([19.5, 19.8, 27.7, 28.9, 41.9, 32.7,
    #                                 51.8, 97.8, 116, 268, 561.7, 1016.8])
    # meas_gammas = numpy.array([58.3, 117, 64.5, 56.4, 23.5, 42.6, 13.1,
    #                             3.91, 4.67, 1.98, 0.70, 0.41])

    name = 'NV0_2019_06_06'
    res_descs = [[0.0, 2.8547, 2.8793],
                  [None, 2.8532, 2.8795],
                  [None, 2.8494, 2.8839],
                  [None, 2.8430, 2.8911],
                  [None, 2.8361, 2.8998],
                  [None, 2.8209, 2.9132],
                  [None, 2.7915, 2.9423],
                  [None, 2.7006, 3.0302],
                  [None, 2.4244, 3.3093],
                  [None, 2.2990, 3.4474],  # Aligned
                  ]
    meas_splittings = numpy.array([23.4, 26.2, 36.2, 48.1, 60.5, 92.3, 150.8,
                                    329.6, 884.9, 1080.5, 1148.4])
    meas_gammas = numpy.array([34.5, 29.0, 20.4, 15.8, 9.1, 6.4, 4.08,
                                1.23, 0.45, 0.69, 0.35])

    # aligned_res_desc = [None, 2.6310, 3.1547]
    # rotated_res_desc = [None, 2.7366, 3.0873]

    # sq_compare_res_desc = [0.0, 2.8537, 2.8751]
    # sq_compare_res_desc = [None, 2.4381, 3.4531]
    # sq_compare_res_desc = [None, 2.8520, 2.8800]

    # mag_B_calc_res_desc = [None, 2.4381, 3.4531]
    # mag_B_calc_meas_rate = 1.57e3  # Hz

    # Run the script
#    main(name, res_descs, aligned_res_desc, rotated_res_desc)
    # main_plot(name, res_descs, aligned_res_desc)
    main_plot_paper(name, res_descs, meas_splittings, meas_gammas)
    # main_plot_rot(name, res_descs)
    # dq_vs_sq_rates(name, res_descs, sq_compare_res_desc)
    # rate_factor_plot_func_B(name, res_descs)
    # mag_B_for_rate(name, res_descs,
    #                mag_B_calc_res_desc, mag_B_calc_meas_rate)
