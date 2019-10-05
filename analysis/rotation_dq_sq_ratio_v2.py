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


# %% Functions


def find_mag_B_splitting_objective(x, splitting,
                                   theta_B, par_Pi, perp_Pi, phi_B, phi_Pi):
    calculated_res_pair = extract_hamiltonian.calc_res_pair(x,
                                    theta_B, par_Pi, perp_Pi, phi_B, phi_Pi)
    calculated_splitting = calculated_res_pair[1] - calculated_res_pair[0]
    return (splitting - calculated_splitting)**2


def calc_dq_factor(theta_B, phi_B, mag_B, popt):

    # mag_B cancels, but if it's too small there are significant rounding errors
    noise_params = (5.0, theta_B, 0.0, 0.0, phi_B, 0.0)
    noise_hamiltonian = extract_hamiltonian.calc_hamiltonian(*noise_params)

    mat_els = extract_hamiltonian.calc_b_matrix_elements(noise_hamiltonian,
                                                         mag_B, *popt)

    mat_factors = [numpy.abs(el)**2 for el in mat_els]
    return mat_factors[2]


def calc_dq_factor_surface(noise_theta_B, noise_phi_B, mag_B, popt):

    val = calc_dq_factor(noise_theta_B, noise_phi_B, mag_B, popt)
    val *= numpy.sin(noise_theta_B)
    return val


# %% Main


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
#    theta_B, phi_B = extract_hamiltonian.find_B_orientation(rotated_res_desc,
#                                            rotated_mag_B, aligned_popt[1],
#                                            aligned_popt[2], aligned_popt[4])
#    rotated_popt = (theta_B, aligned_popt[1], aligned_popt[2],
#                    phi_B, aligned_popt[4])
#    print(rotated_popt)


    ######### TEST #########

    rotated_popt = numpy.copy(aligned_popt)
    angles = numpy.linspace(0, pi/2, 100)
    ratios = []
    for angle in angles:
        rotated_popt[0] = angle
#        print(rotated_popt)
        res_pair = extract_hamiltonian.calc_res_pair(rotated_mag_B, *rotated_popt)
        rotated_res_desc = [rotated_mag_B, res_pair[0], res_pair[1]]

        ######### FIN #########

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

#        print(al_err)
#        print(rot_err)
#        print(aligned_integral / (4*pi))
#        print(rotated_integral / (4*pi))
#        print(aligned_integral / rotated_integral)
        ratios.append(aligned_integral / rotated_integral)

    fig, ax = plt.subplots()
    ax.plot(angles, ratios)


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

    # aligned_res_desc = [None, 2.8198, 2.9106]
    # rotated_res_desc = [None, 2.8454, 2.8873]

    # Test
    aligned_res_desc = [None, 2.6055, 3.1691]
    rotated_res_desc = [None, 2.800, 3.200]

    # Run the script
    main(name, res_descs, aligned_res_desc, rotated_res_desc)
