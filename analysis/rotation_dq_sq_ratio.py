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
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


# %% Constants


# %% Functions


def find_mag_B_splitting_objective(x, splitting,
                                   theta_B, par_Pi, perp_Pi, phi_B, phi_Pi):
    calculated_res_pair = extract_hamiltonian.calc_res_pair(x,
                                    theta_B, par_Pi, perp_Pi, phi_B, phi_Pi)
    calculated_splitting = calculated_res_pair[1] - calculated_res_pair[0]
    return (splitting - calculated_splitting)**2


# %% Main


def main(name, res_descs, aligned_res_desc, rotated_res_desc):
    """When you run the file, we'll call into main, which should contain the
    body of the script.
    """

    # Get the aligned Hamiltonian parameters
    # popt = [theta_B, par_Pi, perp_Pi, phi_B, phi_Pi]
    aligned_popt = extract_hamiltonian.main(name, res_descs)
    # aligned_popt = (0,-0.006,0.011,0,0)

    # Find mag_B at the point we misaligned the field
    rotated_mag_B = extract_hamiltonian.find_mag_B(aligned_res_desc,
                                                   *aligned_popt)

    # Get the rotated Hamiltonian parameters
    theta_B, phi_B = extract_hamiltonian.find_B_orientation(rotated_res_desc,
                                            rotated_mag_B, aligned_popt[1],
                                            aligned_popt[2], aligned_popt[4])

    rotated_popt = (theta_B, aligned_popt[1], aligned_popt[2],
                    phi_B, aligned_popt[4])
    # rotated_popt = (pi/2,-0.006,0.011,0,0)

    rotated_splitting = rotated_res_desc[2] - rotated_res_desc[1]

    # Find the mag_B for an equivalent splitting of the aligned Hamiltonian
    args = (rotated_splitting, *aligned_popt)
    result = minimize_scalar(find_mag_B_splitting_objective, bounds=(0, 1.0),
                             args=args, method='bounded')
    aligned_mag_B = result.x

    # Now let's get the matrix elements of the aligned and rotated cases form
    # at the same splitting. The matrix elements are ordered:
    # zero_to_low_el, zero_to_high_el, low_to_high_el
    aligned_rotated_dq_ratios = []
    noise_mag_Bs = numpy.linspace(0.0, 0.100, 100)
    phis = numpy.linspace(0, 2*pi, 100)

    x_vals = [[mag_B*numpy.cos(phi) for phi in phis] for mag_B in noise_mag_Bs]
    x_vals = numpy.array(x_vals)
    y_vals = [[mag_B*numpy.sin(phi) for phi in phis] for mag_B in noise_mag_Bs]
    y_vals = numpy.array(y_vals)

    for noise_mag_B in noise_mag_Bs:

        row = []

        for phi in phis:

            noise_params = [noise_mag_B, pi/2, 0.0, 0.0, phi, 0.0]
            noise_hamiltonian = extract_hamiltonian.calc_hamiltonian(*noise_params)
            aligned_mat_els = extract_hamiltonian.calc_b_matrix_elements(noise_hamiltonian,
                                                         aligned_mag_B, *aligned_popt)
            rotated_mat_els = extract_hamiltonian.calc_b_matrix_elements(noise_hamiltonian,
                                                         rotated_mag_B, *rotated_popt)
        
            aligned_factors = [numpy.abs(el)**2 for el in aligned_mat_els]
            rotated_factors = [numpy.abs(el)**2 for el in rotated_mat_els]
    
            ratio = aligned_factors[2] / rotated_factors[2]
            row.append(rotated_factors[2])

        aligned_rotated_dq_ratios.append(row)

    aligned_rotated_dq_ratios = numpy.array(aligned_rotated_dq_ratios)

    fig = plt.figure(figsize=(8.5, 8.5))
    ax = fig.add_subplot(111, projection='3d')
    fig.set_tight_layout(True)
    ax.set_title('Aligned / rotated DQ ratios for varying B noise magnitude')
    # ax.plot(noise_mag_Bs, aligned_rotated_dq_ratios)
    ax.plot_surface(x_vals, y_vals, aligned_rotated_dq_ratios,
                    cmap=cm.coolwarm)
    ax.set_xlabel('B noise magnitude (GHz)')
    ax.set_ylabel('Aligned / rotated DQ matrix element ratio')


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
