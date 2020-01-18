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


def calc_B_factor_surface(noise_theta_B, noise_phi_B, mag_B, popt, ind,
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


# def mag_B_for_rate(name, res_descs, compare_res_desc, meas_rate):
#     popt = extract_hamiltonian.main(name, res_descs)
#     dummy_mag_B = 10.0
#     # MHz**2
#     dq_mat_factor, _ = integrate.dblquad(calc_rate_factor_surface,
#                                          0, 2*pi, lambda x: 0, lambda x: pi,
#                                          args=(1.000, popt, 2, dummy_mag_B))
#     dq_mat_factor /= (dummy_mag_B**2)  # dimensionless

#     # meas_rate = (g_mu_B * mag_B)**2 (2 * pi / h_bar) * sq_mat_factor * delta
#     mag_B = numpy.sqrt(meas_rate / (2 * pi * h_bar * dq_mat_factor)) / gyromagnetic
#     mag_B = numpy.sqrt(meas_rate / (2 * pi * dq_mat_factor)) / gyromagnetic
#     print(mag_B)


# def rate_factor_plot_func_B(name, res_descs):
#     mag_Bs = numpy.linspace(0, 1.000, 100)
#     noise_mag_Bs = numpy.linspace(1, 100, 100)
#     # popt = (0.0, 0.0, 0.0, 0.0, 0.0)
#     popt = extract_hamiltonian.main(name, res_descs)

#     rates = []
#     for val in noise_mag_Bs:
#         rate, _ = integrate.dblquad(calc_rate_factor_surface,
#                                     0, 2*pi, lambda x: 0, lambda x: pi,
#                                     args=(1.000, popt, 0, val))
#         rates.append(rate)

#     fig, ax = plt.subplots()
#     fig.set_tight_layout(True)
#     ax.plot(mag_Bs, rates)


def dq_vs_sq_rates(name, res_descs):
    # name = 'nv2'
    # popt = [0.6398153129728315, -0.0044880947609542005, 0.0070490732314452695, 0.0, 0.0]
    name = 'nv2_take2'
    # popt = [1.1162003323335492, -0.0031494625116033634, 0.007006402029975579, 0.0, 0.0]
    popt = [0.0, -0.0031494625116033634, 0.007006402029975579, 0.0, 0.0]

    # noise_power_amp = 100

    # noise_power = noise_power_amp/(compare_res_desc[1]**2)
    # print(noise_power)
    ratios = []
    dq_factors = []
    high_vecs = []
    smooth_mag_Bs = numpy.linspace(0, 1.0, 100)
    for mag_B in smooth_mag_Bs:
        # vecs = extract_hamiltonian.calc_eigenvectors(mag_B, *popt)
        # high_vecs.append(list(vecs[1]))
        
        zero_to_low_integral, zero_to_low_err = integrate.dblquad(
                                            calc_B_factor_surface,
                                            0, 2*pi, lambda x: 0, lambda x: pi,
                                            args=(mag_B, popt, 0))
        # zero_to_low_integral *= noise_power
    
        # noise_power = noise_power_amp/(compare_res_desc[2]**2)
        # print(noise_power)
        # zero_to_high_integral, zero_to_high_err = integrate.dblquad(
        #                                     calc_B_factor_surface,
        #                                     0, 2*pi, lambda x: 0, lambda x: pi,
        #                                     args=(mag_B, popt, 1))
    
        # noise_power = noise_power_amp/((compare_res_desc[2]-compare_res_desc[1])**2)
        # print(noise_power)
        low_to_high_integral, low_to_high_err = integrate.dblquad(
                                            calc_B_factor_surface,
                                            0, 2*pi, lambda x: 0, lambda x: pi,
                                            args=(mag_B, popt, 2))
        # low_to_high_integral *= noise_power
        ratio = zero_to_low_integral / low_to_high_integral
        # ratio = low_to_high_integral / zero_to_low_integral
        dq_factors.append(low_to_high_integral)
        # dq_factors.append(calc_B_factor_surface(pi/2, 0, mag_B, popt, 2))
        ratios.append(ratio)
        
        # print('zero_to_low_integral: {}'.format(zero_to_low_integral))
        # print('zero_to_high_integral: {}'.format(zero_to_high_integral))
        # print('low_to_high_integral: {}'.format(low_to_high_integral))
        # ratio = zero_to_low_integral / low_to_high_integral
        # print('ratio: {}'.format(ratio))
    
    splittings = extract_hamiltonian.calc_splitting(smooth_mag_Bs, *popt)
    fig, ax = plt.subplots(figsize=(8.5, 8.5))
    fig.set_tight_layout(True)
    ax.set_title('Generating fit vector: {}'.format(name))
    ax.set_xscale('log')
    ax.set_yscale('log')
    # high_vecs = numpy.array(high_vecs)
    # ax.plot(splittings, numpy.abs(high_vecs[:, 0])**2, label='+1')
    # ax.plot(splittings, numpy.abs(high_vecs[:, 1])**2, label='0')
    # ax.plot(splittings, numpy.abs(high_vecs[:, 2])**2, label='-1')
    # ax.legend()
    ax.plot(splittings, ratios)
    # dq_factors = numpy.array(dq_factors)
    # ax.plot(splittings, dq_factors / dq_factors[0])
    # ax.plot(splittings, splittings**-2 / splittings[0]**-2)


def main(name, aligned_popt, aligned_res_desc, rotated_res_desc):
    """When you run the file, we'll call into main, which should contain the
    body of the script.
    """

    # Get the aligned Hamiltonian parameters
    # popt = [theta_B, par_Pi, perp_Pi, phi_B, phi_Pi]
    # aligned_popt = extract_hamiltonian.main(name, res_descs)
    # print(aligned_popt)

    # Find mag_B at the point we misaligned the field
    rotated_mag_B = extract_hamiltonian.find_mag_B(aligned_res_desc,
                                                   *aligned_popt)


    # Get the rotated Hamiltonian parameters
    theta_B, phi_B = extract_hamiltonian.find_B_orientation(rotated_res_desc,
                                            rotated_mag_B, aligned_popt[1],
                                            aligned_popt[2], aligned_popt[4])
    # theta_B = 70 * numpy.pi / 180
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

    ratio = rotated_integral / aligned_integral
    print('Expected ratio: {}'.format(ratio))


def main_plot(name, aligned_popt, aligned_res_desc):
    """When you run the file, we'll call into main, which should contain the
    body of the script.
    """

    # Get the aligned Hamiltonian parameters
    # popt = [theta_B, par_Pi, perp_Pi, phi_B, phi_Pi]
    # aligned_popt = extract_hamiltonian.main(name, res_descs)

    # Find mag_B at the point we misaligned the field
    rotated_mag_B = extract_hamiltonian.find_mag_B(aligned_res_desc,
                                                   *aligned_popt)

    rotated_popt = numpy.copy(aligned_popt)
    mag_Bs = numpy.linspace(0,1, 100)
    angles = numpy.linspace(0, pi/2, 50)
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

        ratio = aligned_integral / rotated_integral
        if 9.5 < ratio < 10.5:
            print(rotated_splitting)
        ratios.append(ratio)

    fig, ax = plt.subplots()
    ax.plot(angles, ratios)
    # ax.set_ylim(0, 2)


# def main_plot_rot(name, res_descs):
#     """When you run the file, we'll call into main, which should contain the
#     body of the script.
#     """

#     # [theta_B, par_Pi, perp_Pi, phi_B, phi_Pi]
#     popt = extract_hamiltonian.main(name, res_descs)
#     popt = list(popt)
#     popt[0] = 0.0
#     popt[2] = 0.0
#     print(popt)
#     mag_B = 0.1
#     # print(extract_hamiltonian.calc_eigenvectors(mag_B, *popt))
#     # return

#     gammas = []
#     phi_Pis = numpy.linspace(0, 2*pi, 100)
#     for ind in range(len(phi_Pis)):
#         phi_Pi = phi_Pis[ind]
#         popt[4] = phi_Pi
#         aligned_integral, al_err = integrate.quad(calc_Pi_factor_circle,
#                                               0, 2*pi, args=(mag_B, popt, 2))
#         gammas.append(aligned_integral)

#     fig, ax = plt.subplots()
#     fig.set_tight_layout(True)
#     ax.plot(phi_Pis, gammas)


def main_plot_paper(name, res_descs, meas_splittings, meas_gammas):
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
    
    # noise_func = calc_Pi_factor_surface
    noise_func = calc_B_factor_surface

    popt = numpy.copy(popt)
    gamma_bs = []
    splittings = []
    empiricals = []
    # empirical_scaling = -2.0
    # mag_Bs = numpy.linspace(0.001, 1.0, 100)
    min_splitting = min(meas_splittings)
    max_splitting = max(meas_splittings)
    min_mag_B = extract_hamiltonian.find_mag_B_splitting(min_splitting, *popt)
    max_mag_B = extract_hamiltonian.find_mag_B_splitting(max_splitting, *popt)
    mag_Bs = numpy.logspace(numpy.log10(min_mag_B), numpy.log10(max_mag_B), 100)
    for ind in range(len(mag_Bs)):
        mag_B = mag_Bs[ind]
        splitting = extract_hamiltonian.calc_splitting(mag_B, *popt)
        splittings.append(splitting)
        # noise_mag = 15.0 + splitting**-2
        aligned_integral, al_err = integrate.dblquad(noise_func, 0, 2*pi,
                                                 lambda x: 0, lambda x: pi,
                                                 args=(mag_B, popt, 2))
        if ind == 0:
            scaling = meas_gammas[0] / aligned_integral
            # coeff = (10 * meas_gammas[0] + meas_gammas[1]) / 5
            # coeff = meas_gammas[1]
            # scaling = coeff / aligned_integral
            # empirical_coeff = 40 / (splitting**empirical_scaling)
        gamma_b = scaling * aligned_integral
        gamma_bs.append(gamma_b)
        # gamma_bs.append(aligned_integral)
        # empirical = empirical_coeff * (splitting**empirical_scaling)
        # empiricals.append(empirical + 0.5)
        # print(extract_hamiltonian.calc_eigenvectors(mag_B, *popt))

    fig, ax = plt.subplots()
    fig.set_tight_layout(True)
    # ax.plot(splittings, gamma_bs)
    ax.loglog(splittings, gamma_bs)
    # ax.loglog(splittings, empiricals)
    # ax.set_ylim(0, 10)
    ax.scatter(meas_splittings, meas_gammas)


# %% Run the file


# The __name__ variable will only be '__main__' if you run this file directly.
# This allows a file's functions, classes, etc to be imported without running
# the script that you set up here.
if __name__ == '__main__':

    # name = 'fake'
    # res_descs = [[0.0, 2.845, 2.895], 
    #             [None, 2.838, 2.902], 
    #             [None, 2.805, 2.935], 
    #             [None, 2.767, 2.973], 
    #             [None, 2.619, 3.121], 
    #             [None, 2.369, 3.371], 
    #             ]
    
    
    # name = 'nv1_2019_05_10'  # NV1
    # popt = [0.6474219686681678, -0.005159086817872651, 0.009754609612326834, 0.0, 0.0]
    # # popt = [0.6500999024309339, -0.005162723325066839, 0.009743779800841193, 1.0471975511965976, 1.0471975511965976]
    # res_descs = [[0.0, 2.8544, 2.8739],
    #               [None, 2.8554, 2.8752],
    #               [None, 2.8512, 2.8790],
    #               # [None, 2.8520, 2.8800],  # not used in paper
    #               [None, 2.8503, 2.8792],
    #               # [None, 2.8536, 2.8841],  # not used in paper
    #               [None, 2.8396, 2.8917],
    #               [None, 2.8496, 2.8823],
    #               # [None, 2.8198, 2.9106],  # misaligned ref
    #               [None, 2.8166, 2.9144],
    #               [None, 2.8080, 2.9240],
    #               [None, 2.7357, 3.0037],
    #               # [None, 2.7374, 3.0874],  # misaligned, theta_B = 1.014639916147641
    #               # [None, 2.6310, 3.1547],  # misaligned ref for prev
    #               [None, 2.6061, 3.1678],
    #               # [None, 2.6055, 3.1691],  # repeat of previous
    #               [None, 2.4371, 3.4539],  # 0,-1 and 0,+1 omegas
    #               # [None, 2.4381, 3.4531],   # retake 0,-1 and 0,+1 omegas
    #               ]
    
    # # aligned_res_desc = [None, 2.6310, 3.1547]
    # rotated_res_desc = [None, 2.7374, 3.0874]
    
    
    # name = 'NV0_2019_06_06'  # NV4
    # res_descs = [
    #               # [0.0, 2.8547, 2.8793],  # old zero field
    #               [0.0, 2.8556, 2.8790],
    #               [None, 2.8532, 2.8795],
    #               [None, 2.8494, 2.8839],
    #               [None, 2.8430, 2.8911],
    #               [None, 2.8361, 2.8998],
    #               [None, 2.8209, 2.9132],
    #               [None, 2.7915, 2.9423],
    #               [None, 2.7006, 3.0302],
    #               [None, 2.4244, 3.3093],
    #               # [None, 2.4993, 3.5798],  # misaligned
    #               [None, 2.2990, 3.4474],
    #               ]
    # aligned_res_desc = [None, 2.4244, 3.3093]
    
    
    # name = 'nv13_2019_06_10'  # NV5
    # res_descs = [[0.0, 2.8365, 2.8446],  # no T1
    #               [None, 2.8363, 2.8472],
    #               [None, 2.8289, 2.8520],
    #               # [None, 2.8266, 2.8546],  # not used in paper
    #               # [None, 2.8262, 2.8556],  # not used in paper
    #               [None, 2.8247, 2.8545],
    #               [None, 2.8174, 2.8693],
    #               [None, 2.8082, 2.8806],
    #               [None, 2.7948, 2.9077],
    #               [None, 2.7857, 2.9498],
    #               [None, 2.7822, 3.0384],
    #               ]
    # aligned_res_desc = [None, 2.8082, 2.8806]
    

    # Run the script
    main(name, popt, aligned_res_desc, rotated_res_desc)
    # main_plot(name, popt, aligned_res_desc)
    # main_plot_paper(name, res_descs, meas_splittings, meas_gammas)
    # main_plot_rot(name, res_descs)
    # dq_vs_sq_rates(name, res_descs)
    # rate_factor_plot_func_B(name, res_descs)
    # mag_B_for_rate(name, res_descs,
    #                 mag_B_calc_res_desc, mag_B_calc_meas_rate)
