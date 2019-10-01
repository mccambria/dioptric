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


hbar = 1.055e-34  # m^2 kg / s


# %% Functions


def herm(matrix):
    return numpy.transpose(numpy.conj(matrix))


def find_mag_B_splitting_objective(x, splitting,
                                   theta_B, par_Pi, perp_Pi, phi_B, phi_Pi):
    calculated_res_pair = extract_hamiltonian.calc_res_pair(x,
                                    theta_B, par_Pi, perp_Pi, phi_B, phi_Pi)
    calculated_splitting = calculated_res_pair[1] - calculated_res_pair[0]
    return (splitting - calculated_splitting)**2


def calc_dq_factor(theta_B, phi_B,
                   par_Pi, perp_Pi, phi_Pi, mag_B, popt):

    noise_params = (theta_B, par_Pi, perp_Pi, phi_B, phi_Pi)
    noise_hamiltonian = extract_hamiltonian.calc_hamiltonian(*noise_params)

    mat_els = extract_hamiltonian.calc_b_matrix_elements(noise_hamiltonian,
                                                         mag_B, *popt)

    mat_factors = [numpy.abs(el)**2 for el in mat_els]
    return mat_factors[2]


# %% Main


def main(name, res_descs, aligned_res_desc, rotated_res_desc):
    """When you run the file, we'll call into main, which should contain the
    body of the script.
    """

    # Get the aligned Hamiltonian parameters
    # popt = [theta_B, par_Pi, perp_Pi, phi_B, phi_Pi]
    aligned_popt = extract_hamiltonian.main(name, res_descs)
    # aligned_popt = [0.0, 0.0, 0.0, 0.0, 0.0]
    print(aligned_popt)

    # Find mag_B at the point we misaligned the field
    rotated_mag_B = extract_hamiltonian.find_mag_B(aligned_res_desc,
                                                   *aligned_popt)


    # Get the rotated Hamiltonian parameters
    theta_B, phi_B = extract_hamiltonian.find_B_orientation(rotated_res_desc,
                                            rotated_mag_B, aligned_popt[1],
                                            aligned_popt[2], aligned_popt[4])

    rotated_popt = (theta_B, aligned_popt[1], aligned_popt[2],
                    phi_B, aligned_popt[4])
    # rotated_popt = [1.24, 0.0, 0.0, 0.0, 0.0]
    print(rotated_popt)

    rotated_splitting = rotated_res_desc[2] - rotated_res_desc[1]

    # Find the mag_B for an equivalent splitting of the aligned Hamiltonian
    args = (rotated_splitting, *aligned_popt)
    result = minimize_scalar(find_mag_B_splitting_objective, bounds=(0, 1.0),
                             args=args, method='bounded')
    aligned_mag_B = result.x

    aligned_vecs = extract_hamiltonian.calc_eigenvectors(aligned_mag_B, *aligned_popt)
    aligned_ideal_noise = numpy.outer(numpy.conj(aligned_vecs[2]), aligned_vecs[1])
    aligned_ideal_noise += numpy.outer(numpy.conj(aligned_vecs[1]), aligned_vecs[2])
    aligned_ideal_noise = numpy.real(aligned_ideal_noise)
    print(aligned_ideal_noise)

    rotated_vecs = extract_hamiltonian.calc_eigenvectors(rotated_mag_B, *rotated_popt)
    rotated_ideal_noise = numpy.outer(numpy.conj(rotated_vecs[2]), rotated_vecs[1])
    rotated_ideal_noise += numpy.outer(numpy.conj(rotated_vecs[1]), rotated_vecs[2])
    aligned_ideal_noise = numpy.real(rotated_ideal_noise)
    print(rotated_ideal_noise)

    print('\n- vec\n')
    print(aligned_vecs[1])
    print(rotated_vecs[1])
    print('+ vec\n')
    print(aligned_vecs[2])
    print(rotated_vecs[2])
    # return

    # mag_B cancels, but if it's too small there are significant rounding errors
    noise_params = [5.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    noise_hamiltonian = extract_hamiltonian.calc_hamiltonian(*noise_params)
    aligned_mat_els = extract_hamiltonian.calc_b_matrix_elements(noise_hamiltonian,
                                                  aligned_mag_B, *aligned_popt)
    rotated_mat_els = extract_hamiltonian.calc_b_matrix_elements(noise_hamiltonian,
                                                  rotated_mag_B, *rotated_popt)

    aligned_factors = [numpy.abs(el)**2 for el in aligned_mat_els]
    rotated_factors = [numpy.abs(el)**2 for el in rotated_mat_els]

    ratio = aligned_factors[2] / rotated_factors[2]
    print(aligned_noise_theta)
    print(rotated_noise_theta)
    print(ratio)
    # return

    integral = integrate.dblquad()

    ################# Plotting... #################

    # Now let's get the matrix elements of the aligned and rotated cases form
    # at the same splitting. The matrix elements are ordered:
    # zero_to_low_el, zero_to_high_el, low_to_high_el
    aligned_rotated_dq_ratios = []
    aligned_factor_list = []
    rotated_factor_list = []
    noise_mag_B = 5.0  # Computer does its best arithmetic with values ~1
    # noise_mag_Bs = numpy.linspace(0,200,1000)
    # phis = numpy.linspace(0, 2*pi, 1000)
    thetas = numpy.linspace(0, pi/2, 1000)

    # for noise_mag_B in noise_mag_Bs:
    # for phi in phis:
    for theta in thetas:

        # noise_params = [noise_mag_B, 0.0, 0.0, 0.0, 0.0, 0.0]
        # noise_params = [noise_mag_B, pi/2, 0.0, 0.0, phi, 0.0]
        noise_params = [noise_mag_B, theta, 0.0, 0.0, 0, 0.0]
        noise_hamiltonian = extract_hamiltonian.calc_hamiltonian(*noise_params)
        aligned_mat_els = extract_hamiltonian.calc_b_matrix_elements(noise_hamiltonian,
                                                     aligned_mag_B, *aligned_popt)
        rotated_mat_els = extract_hamiltonian.calc_b_matrix_elements(noise_hamiltonian,
                                                     rotated_mag_B, *rotated_popt)

        aligned_factors = [numpy.abs(el)**2 for el in aligned_mat_els]
        rotated_factors = [numpy.abs(el)**2 for el in rotated_mat_els]

        ratio = aligned_factors[2] / rotated_factors[2]
        aligned_rotated_dq_ratios.append(ratio)
        aligned_factor_list.append(aligned_factors[2] * numpy.sin(theta))
        rotated_factor_list.append(rotated_factors[2] * numpy.sin(theta))
        # aligned_rotated_dq_ratios.append(numpy.abs(aligned_mat_els[2]))
        # aligned_rotated_dq_ratios.append(numpy.abs(rotated_mat_els[2]))

    aligned_rotated_dq_ratios = numpy.array(aligned_rotated_dq_ratios)

    fig, ax = plt.subplots(1,1, figsize=(8.5, 8.5))
    fig.set_tight_layout(True)

    # ax.plot(noise_mag_Bs, aligned_rotated_dq_ratios)
    # ax.plot(phis, aligned_rotated_dq_ratios)
    # ax.plot(thetas, aligned_rotated_dq_ratios)

    ax.plot(thetas, aligned_factor_list)
    ax.plot(thetas, rotated_factor_list)


# %% Run the file


# The __name__ variable will only be '__main__' if you run this file directly.
# This allows a file's functions, classes, etc to be imported without running
# the script that you set up here.
if __name__ == '__main__':

    # Set up your parameters to be passed to main here
    name = 'nv1_2019_05_10'
    res_descs = [[0.0, 2.8537, 2.8751],
                  [None, 2.8554, 2.8752],
                  [None, 2.8512, 2.8790],
                  [None, 2.8520, 2.8800],
                  [None, 2.8536, 2.8841],
                  [None, 2.8496, 2.8823],
                  [None, 2.8396, 2.8917],
                  [None, 2.8198, 2.9106],  # Reference for rotated T1
                  [None, 2.8166, 2.9144],
                  [None, 2.8080, 2.9240],
                  [None, 2.7357, 3.0037],
                  [None, 2.6061, 3.1678],
                  [None, 2.6055, 3.1691],
                  [None, 2.4371, 3.4539]]

    aligned_res_desc = [None, 2.8198, 2.9106]
    rotated_res_desc = [None, 2.8454, 2.8873]

    # Run the script
    main(name, res_descs, aligned_res_desc, rotated_res_desc)
