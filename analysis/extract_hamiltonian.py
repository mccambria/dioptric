# -*- coding: utf-8 -*-
"""See document on the wiki.

Created on Sun Jun 16 11:22:40 2019

@author: mccambria
"""


# %% Imports


import numpy
from numpy.linalg import eigvals
from numpy.linalg import eig
from numpy import pi
from scipy.optimize import minimize_scalar
from scipy.optimize import minimize
from scipy.optimize import brute
from numpy import exp
import matplotlib.pyplot as plt


# %% Constants


# GHz
d_gs = 2.87

# numbers
inv_sqrt_2 = 1/numpy.sqrt(2)


# %% Functions


def calc_matrix_elements(noise_hamiltonian,
                         mag_B, theta_B, par_Pi, perp_Pi, phi_B, phi_Pi):

    popt_full = [theta_B, par_Pi, perp_Pi, phi_B, phi_Pi]
    vecs = calc_eigenvectors(mag_B, *popt_full)  # zero, low, high

    zero_to_low_el = numpy.matmul(noise_hamiltonian, vecs[1])
    zero_to_low_el = numpy.matmul(numpy.transpose(vecs[0]), zero_to_low_el)

    zero_to_high_el = numpy.matmul(noise_hamiltonian, vecs[2])
    zero_to_high_el = numpy.matmul(numpy.transpose(vecs[0]), zero_to_high_el)

    low_to_high_el = numpy.matmul(noise_hamiltonian, vecs[2])
    low_to_high_el = numpy.matmul(numpy.transpose(vecs[1]), low_to_high_el)

    return zero_to_low_el, zero_to_high_el, low_to_high_el


def b_matrix_elements(name, res_descs):

    sq_allow_factors = []  # zero to low
    dq_allow_factors = []  # low to high

    zero_zero_comps = []
    low_zero_comps = []
    high_zero_comps = []

    popt = main(name, res_descs)  # Excluding phis
    popt_full = numpy.append(popt, [0.0, 0.0])  # phis = 0

    smooth_mag_Bs = numpy.linspace(0.050, 1.0, 1000)
    noise_params = [0.20, pi/4, 0.0, 0.0, 0.0, 0.0]
    noise_hamiltonian = calc_hamiltonian(*noise_params)

    for mag_B in smooth_mag_Bs:
        vecs = calc_eigenvectors(mag_B, *popt_full)  # zero, low, high
        zero_zero_comps.append(numpy.abs(vecs[0,1])**2)
        low_zero_comps.append(numpy.abs(vecs[1,1])**2)
        high_zero_comps.append(numpy.abs(vecs[2,1])**2)

        ret_vals = calc_matrix_elements(noise_hamiltonian,
                                          mag_B, *popt_full)
        zero_to_low_el, zero_to_high_el, low_to_high_el = ret_vals

        sq_allow_factors.append(numpy.abs(zero_to_low_el)**2)
        dq_allow_factors.append(numpy.abs(low_to_high_el)**2)

    sq_allow_factors = numpy.array(sq_allow_factors)
    dq_allow_factors = numpy.array(dq_allow_factors)

    fig, ax = plt.subplots(figsize=(8.5, 8.5))
    fig.set_tight_layout(True)
    ax.set_title('Generating fit vector: {}'.format(name))
    ax.semilogy(smooth_mag_Bs, dq_allow_factors/sq_allow_factors)
    ax.set_xlabel('B magnitude (GHz)')
    ax.set_ylabel('DQ/SQ rate ratio')

    # fig, ax = plt.subplots(figsize=(8.5, 8.5))
    # fig.set_tight_layout(True)
    # ax.set_title('Generating fit vector: {}'.format(name))
    # ax.plot(smooth_mag_Bs, sq_mat_els, label='SQ')
    # ax.plot(smooth_mag_Bs, dq_mat_els, label='DQ')
    # ax.set_xlabel('B magnitude (GHz)')
    # ax.set_ylabel('Matrix elements magnitude squared')
    # ax.legend()

    # fig, ax = plt.subplots(figsize=(8.5, 8.5))
    # fig.set_tight_layout(True)
    # ax.set_title('Generating fit vector: {}'.format(name))
    # ax.plot(smooth_mag_Bs, zero_zero_comps, label='0, 0')
    # ax.plot(smooth_mag_Bs, low_zero_comps, label='low, 0')
    # ax.plot(smooth_mag_Bs, high_zero_comps, label='high, 0')
    # ax.set_xlabel('B magnitude (GHz)')
    # ax.set_ylabel('|<0|psi>|^2')
    # ax.legend()


def find_B_orientation(rotated_res_desc, mag_B, par_Pi, perp_Pi, phi_Pi):

    # fit_vec = [theta_B, phi_B]
    param_bounds = ((0, pi/2), (0, 2*pi/3))
    guess_params = (pi/3, 0)

    args = (rotated_res_desc, mag_B, par_Pi, perp_Pi, phi_Pi)
    res = minimize(find_B_orientation_objective, guess_params,
                   args=args, bounds=param_bounds, method='SLSQP')

    return res.x


def find_B_orientation_objective(fit_vec, rotated_res_desc,
                                 mag_B, par_Pi, perp_Pi, phi_Pi):
    calculated_res_pair = calc_res_pair(mag_B, fit_vec[0], par_Pi, perp_Pi,
                                        fit_vec[1], phi_Pi)
    diffs = numpy.array(calculated_res_pair) - numpy.array(rotated_res_desc[1:3])
    sum_squared_differences = numpy.sum(diffs**2)
    return sum_squared_differences


def extract_rotated_hamiltonian(name, res_descs,
                                aligned_res_desc, rotated_res_desc):

    # [theta_B, par_Pi, perp_Pi, phi_B, phi_Pi]
    popt = main(name, res_descs)  # Excluding phis

    mag_B = find_mag_B(aligned_res_desc, *popt)
    theta_B, phi_B = find_B_orientation(rotated_res_desc, mag_B,
                                popt[1], popt[2], popt[4])

    return theta_B, phi_B


def generate_fake_data(theta_B, par_Pi, perp_Pi, phi_B, phi_Pi):

    fit_vec = [theta_B, par_Pi, perp_Pi, phi_B, phi_Pi]
    mag_Bs = [0.0, 0.020, 0.060, 0.100, 0.250, 0.500, 1.000]
    res_descs = []
    for mag_B in mag_Bs:
        if mag_B == 0.0:
            res_desc = [mag_B]
        else:
            res_desc = [None]
        res_pair = calc_res_pair(mag_B, *fit_vec)
        rounded_res_pair = [round(el, 3) for el in res_pair]
        res_desc.extend(rounded_res_pair)
        res_descs.append(res_desc)
        print('{}, '.format(res_desc))


def calc_single_hamiltonian(mag_B, theta_B, par_Pi, perp_Pi, phi_B, phi_Pi):
    par_B = mag_B * numpy.cos(theta_B)
    perp_B = mag_B * numpy.sin(theta_B)
    hamiltonian = numpy.array([[d_gs + par_Pi + par_B,
                                inv_sqrt_2 * perp_B * exp(-1j * phi_B),
                                -perp_Pi * exp(1j * phi_Pi)],
                               [inv_sqrt_2 * perp_B * exp(1j * phi_B),
                                0,
                                inv_sqrt_2 * perp_B * exp(-1j * phi_B)],
                               [-perp_Pi * exp(-1j * phi_Pi),
                                inv_sqrt_2 * perp_B * exp(1j * phi_B),
                                d_gs + par_Pi - par_B]])
    return hamiltonian


def calc_hamiltonian(mag_B, theta_B, par_Pi, perp_Pi, phi_B, phi_Pi):
    fit_vec = [theta_B, par_Pi, perp_Pi, phi_B, phi_Pi]
    if (type(mag_B) is list) or (type(mag_B) is numpy.ndarray):
        hamiltonian_list = [calc_single_hamiltonian(val, *fit_vec)
                            for val in mag_B]
        return hamiltonian_list
    else:
        return calc_single_hamiltonian(mag_B, *fit_vec)


def calc_single_B_hamiltonian(mag_B, theta_B, phi_B):
    par_B = mag_B * numpy.cos(theta_B)
    perp_B = mag_B * numpy.sin(theta_B)
    hamiltonian = numpy.array([[par_B,
                                inv_sqrt_2 * perp_B * exp(-1j * phi_B),
                                0],
                               [inv_sqrt_2 * perp_B * exp(1j * phi_B),
                                0,
                                inv_sqrt_2 * perp_B * exp(-1j * phi_B)],
                               [0,
                                inv_sqrt_2 * perp_B * exp(1j * phi_B),
                                -par_B]])
    return hamiltonian


def calc_B_hamiltonian(mag_B, theta_B, phi_B):
    args = (mag_B, theta_B, phi_B)
    if (type(mag_B) is list) or (type(mag_B) is numpy.ndarray):
        hamiltonian_list = [calc_single_B_hamiltonian(*args)
                            for val in mag_B]
        return hamiltonian_list
    else:
        return calc_single_B_hamiltonian(*args)


def calc_single_Pi_hamiltonian(mag_Pi, theta_Pi, phi_Pi):
    d_par = 0.3
    d_perp = 17.0
    par_Pi = d_par * mag_Pi * numpy.cos(theta_Pi)
    perp_Pi = d_perp * mag_Pi * numpy.sin(theta_Pi)
    hamiltonian = numpy.array([[par_Pi, 0, perp_Pi * exp(1j * phi_Pi)],
                               [0, 0, 0],
                               [perp_Pi * exp(-1j * phi_Pi), 0, par_Pi]])
    return hamiltonian


def calc_Pi_hamiltonian(perp_Pi, theta_Pi, phi_Pi):
    args = (perp_Pi, theta_Pi, phi_Pi)
    if (type(perp_Pi) is list) or (type(perp_Pi) is numpy.ndarray):
        hamiltonian_list = [calc_single_Pi_hamiltonian(*args)
                            for val in perp_Pi]
        return hamiltonian_list
    else:
        return calc_single_Pi_hamiltonian(*args)


def calc_res_pair(mag_B, theta_B, par_Pi, perp_Pi, phi_B, phi_Pi):
    hamiltonian = calc_hamiltonian(mag_B, theta_B, par_Pi, perp_Pi,
                                   phi_B, phi_Pi)
    if (type(mag_B) is list) or (type(mag_B) is numpy.ndarray):
        vals = numpy.sort(eigvals(hamiltonian), axis=1)
        resonance_low = numpy.real(vals[:,1] - vals[:,0])
        resonance_high = numpy.real(vals[:,2] - vals[:,0])
    else:
        vals = numpy.sort(eigvals(hamiltonian))
        resonance_low = numpy.real(vals[1] - vals[0])
        resonance_high = numpy.real(vals[2] - vals[0])
    return resonance_low, resonance_high


def calc_splitting(mag_B, theta_B, par_Pi, perp_Pi, phi_B, phi_Pi):
    args = (mag_B, theta_B, par_Pi, perp_Pi, phi_B, phi_Pi)
    res_pair = calc_res_pair(*args)
    return res_pair[1] - res_pair[0]


def calc_eigenvectors(mag_B, theta_B, par_Pi, perp_Pi, phi_B, phi_Pi):
    """Return the normalized eigenvectors, sorted by ascending eigenvalue
    """
    hamiltonian = calc_hamiltonian(mag_B, theta_B, par_Pi, perp_Pi,
                                   phi_B, phi_Pi)
    eigvals, eigvecs = eig(hamiltonian)
    sorted_indices = numpy.argsort(eigvals)
    # sorted_eigvecs = [numpy.round(eigvecs[:,ind], 3) for ind in sorted_indices]
    sorted_eigvecs = [eigvecs[:,ind] for ind in sorted_indices]
    sorted_eigvecs = numpy.array(sorted_eigvecs)
    return sorted_eigvecs


def find_mag_B(res_desc, theta_B, par_Pi, perp_Pi, phi_B, phi_Pi):
    # Just return the given mag_B if it's known
    if res_desc[0] is not None:
        return res_desc[0]
    # Otherwise we'll determine the most likely mag_B for this fit_vec by
    # finding the mag_B that minimizes the distance between the measured
    # resonances and the calculated resonances for a given fit_vec
    args = (res_desc, theta_B, par_Pi, perp_Pi, phi_B, phi_Pi)
    result = minimize_scalar(find_mag_B_objective, bounds=(0, 1.0), args=args,
                             method='bounded')
    if result.success:
        mag_B = result.x
    else:
        # If we didn't find an optimal value, return something that will blow
        # up chisq and push us away from this fit_vec
        mag_B = 0.0
    return mag_B


def find_mag_B_objective(x, res_desc, theta_B, par_Pi, perp_Pi, phi_B, phi_Pi):
    calculated_res_pair = calc_res_pair(x, theta_B, par_Pi, perp_Pi,
                                        phi_B, phi_Pi)
    diffs = numpy.array(calculated_res_pair) - numpy.array(res_desc[1:3])
    sum_squared_differences = numpy.sum(diffs**2)
    return sum_squared_differences


def find_mag_B_splitting(splitting, theta_B, par_Pi, perp_Pi, phi_B, phi_Pi):
    # Otherwise we'll determine the most likely mag_B for this fit_vec by
    # finding the mag_B that minimizes the distance between the measured
    # resonances and the calculated resonances for a given fit_vec
    args = (splitting, theta_B, par_Pi, perp_Pi, phi_B, phi_Pi)
    result = minimize_scalar(find_mag_B_splitting_objective,
                             bounds=(0, 1.5), args=args, method='bounded')
    if result.success:
        mag_B = result.x
    else:
        mag_B = 0.0
    return mag_B


def find_mag_B_splitting_objective(x, splitting,
                                   theta_B, par_Pi, perp_Pi, phi_B, phi_Pi):
    calculated_splitting = calc_splitting(x, theta_B, par_Pi, perp_Pi,
                                          phi_B, phi_Pi)
    return numpy.abs(calculated_splitting - splitting)


def plot_resonances(mag_B_range, theta_B, par_Pi, perp_Pi, phi_B, phi_Pi,
                    name='untitled'):

    smooth_mag_B = numpy.linspace(mag_B_range[0], mag_B_range[1], 1000)
    res_pairs = calc_res_pair(smooth_mag_B, theta_B, par_Pi, perp_Pi,
                              phi_B, phi_Pi)

    fig, ax = plt.subplots(figsize=(8.5, 8.5))
    fig.set_tight_layout(True)
    ax.set_title('Generating fit vector: {}'.format(name))
    ax.plot(smooth_mag_B, res_pairs[0])
    ax.plot(smooth_mag_B, res_pairs[1])
    ax.set_xlabel('B magnitude (GHz)')
    ax.set_ylabel('Resonance (GHz)')

    textstr = '\n'.join((
        r'$\theta_{B}=%.3f \ rad$' % (theta_B, ),
        r'$\Pi_{\parallel}=%.3f \ GHz$' % (par_Pi, ),
        r'$\Pi_{\perp}=%.3f \ GHz$' % (perp_Pi, )))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, textstr, fontsize=14, transform=ax.transAxes,
            verticalalignment='top', bbox=props)

    return fig, ax


def chisq_func_reduced(fit_vec, par_Pi, perp_Pi, phi_B, phi_Pi, res_descs):

    theta_B = fit_vec[0]
    fit_vec = [theta_B, par_Pi, perp_Pi]
    return chisq_func(fit_vec, phi_B, phi_Pi, res_descs)


def chisq_func(fit_vec, phi_B, phi_Pi, res_descs):

    num_resonance_descs = len(res_descs)
    mag_Bs = [find_mag_B(desc, *fit_vec, phi_B, phi_Pi) for desc in res_descs]

    # find_mag_B_objective returns the sum of squared residuals for a single
    # pair of resonances. We want to sum this over all pairs.
    squared_residuals = [find_mag_B_objective(mag_Bs[ind], res_descs[ind],
                         *fit_vec, phi_B, phi_Pi) for ind
                         in range(num_resonance_descs)]
    sum_squared_residuals = numpy.sum(squared_residuals)

    estimated_st_dev = 0.0001
    estimated_var = numpy.sqrt(estimated_st_dev)
    chisq = sum_squared_residuals / estimated_var

    return chisq


# %% Main


def main(name, res_descs):

    ############ Setup ############

    phi_B = 0
    phi_Pi = 0

    # fit_vec = [theta_B, par_Pi, perp_Pi,]
    param_bounds = ((0, pi/2), (-0.050, 0.050), (0, 0.050))

    res_descs = numpy.array(res_descs)
    for desc in res_descs:
        # Set degenerate resonances to the same value
        if desc[2] is None:
            desc[2] = desc[1]
        # Make sure resonances are sorted
        desc[1:3] = numpy.sort(desc[1:3])

    ############ Guess par_Pi and perp_Pi by zero field ############

    zero_field_res_desc = None
    par_Pi = None
    perp_Pi = None
    # See if we have zero-field resonances
    for desc in res_descs:
        if desc[0] == 0.0:
            zero_field_res_desc = desc
            break

    if zero_field_res_desc is not None:
        # Get the splitting and center_freq from the resonances
        zero_field_low = zero_field_res_desc[1]
        zero_field_high = zero_field_res_desc[2]

        # At B = 0 the Hamiltonian has the form
        # [     d_gs + par_Pi,           0,     - perp_Pi * exp(i phi_Pi)   ]
        # [            0,               0,                0              ]
        # [-perp_Pi * exp(-i phi_Pi),     0,           d_gs + par_Pi        ]

        # The eigenvalues are simple in this case
        # [0, d_gs + par_Pi - perp_Pi, d_gs + par_Pi + perp_Pi]
        # The resonances are thus
        # [d_gs + par_Pi - perp_Pi, d_gs + par_Pi + perp_Pi]]
        # and so
        # zero_field_center = (d_gs + par_Pi - perp_Pi + d_gs + par_Pi + perp_Pi) / 2
        # zero_field_center = d_gs + par_Pi
        zero_field_center = (zero_field_high + zero_field_low) / 2
        par_Pi = zero_field_center - d_gs

        # Similarly
        # zero_field_splitting = (d_gs + par_Pi + perp_Pi) - (d_gs + par_Pi - perp_Pi)
        # zero_field_splitting = 2 * perp_Pi
        zero_field_splitting = zero_field_high - zero_field_low
        perp_Pi = zero_field_splitting / 2

    ############ Guess remaining parameters with brute force ############

    if (par_Pi is not None) and (perp_Pi is not None):
        # Just fit for theta_B
        param_ranges = param_bounds[0:1]
        args = (par_Pi, perp_Pi, phi_B, phi_Pi, res_descs)
        x0 = brute(chisq_func_reduced, param_ranges, args=args, Ns=20)
        guess_params = list(x0)
        guess_params.extend([par_Pi, perp_Pi])
    else:
        param_ranges = param_bounds
        args = (phi_B, phi_Pi, res_descs)
        x0 = brute(chisq_func, param_ranges, args=args, Ns=10)
        guess_params = list(x0)

    ############ Fine tuning with minimize ############

    args = (phi_B, phi_Pi, res_descs)
    res = minimize(chisq_func, guess_params, args=args,
                   bounds=param_bounds, method='SLSQP')
    if not res.success:
        print(res.message)
        return

    popt = res.x
    print('popt: {}'.format(popt))

    chisq = res.fun
    print('Chi squared: {:.4g}'.format(chisq))
    degrees_of_freedom = len(res_descs) - len(x0)
    reduced_chisq = res.fun / degrees_of_freedom
    print('Reduced chi squared: {:.4g}'.format(reduced_chisq))

    ############ Plot the result ############

    # Get the mag_B for each pair of resonances with this fit_vec
    mag_Bs = [find_mag_B(desc, *popt, phi_B, phi_Pi) for desc in res_descs]

    # Plot the calculated resonances up to the max mag_B
    # popt[0] = 32 * (numpy.pi / 180)
    fig, ax = plot_resonances([0, max(mag_Bs)], *popt, phi_B, phi_Pi, name)

    # Plot the resonances
    ax.scatter(mag_Bs, res_descs[:,1])
    ax.scatter(mag_Bs, res_descs[:,2])

    # Return the full 5 parameters
    return numpy.append(popt, [0.0, 0.0])


# %% Run the file


# The __name__ variable will only be '__main__' if you run this file directly.
# This allows a file's functions, classes, etc to be imported without running
# the script that you set up here.
if __name__ == '__main__':

    # Each pair of resonances should be a 2-list. So zero_field_resonances
    # is a single 2-list and non_zero_field_resonances is a list of 2-lists.

    ############ Nice ############
    
    name = 'goeppert_mayer-nv7_2019_11_27'
    res_descs = [[0.0, 2.8696, 2.8696],
                  [None, 2.8508, 2.8914],  # 41 MHz
                  [None, 2.7893, 2.9564],  # 167 MHz
                  [None, 2.6719, 3.0846],  # 413 MHz
                  [None, 2.5760, 3.1998],  # 624 MHz
                  [None, 2.4857, 3.3173],  # 832 MHz
                  [None, 2.3298, 3.5369],  # 1207 MHz
                  [None, 2.7243, 3.0243],
                  [None, 2.7113, 3.0402],
                  [None, 2.6270, 3.1365],
                  [None, 2.4991, 3.2991],
                  [None, 2.4290, 3.3976],
                  [None, 2.3844, 3.4575],
                  [None, 2.3293, 3.5355],
                  [None, 2.8028, 2.9413],
                  [None, 2.8286, 2.9116],
                  [None, 2.8466, 2.8937],
                  [None, 2.8302, 2.9098],
                  [None, 2.7706, 2.9749],
                  [None, 2.7706, 2.9749],
                  ] 

    # Run the script
    main(name, res_descs)

    # Fake data
    # args: theta_B, par_Pi, perp_Pi, phi_B, phi_Pi
    # bounds: ((0, pi/2), (-0.050, 0.050), (0, 0.050), (0, pi/3), (0, 2*pi/3))
    # generate_fake_data(1.563, 0.000, 0.025, 0.000, 0.000)
