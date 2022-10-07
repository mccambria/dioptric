# -*- coding: utf-8 -*-
"""See document on the wiki.

Created on Sun Jun 16 11:22:40 2019

@author: mccambria
"""


# %% Imports


import numpy as np
from numpy.linalg import eigvals
from numpy.linalg import eig
from numpy import pi
from scipy.optimize import minimize_scalar
from scipy.optimize import minimize
from scipy.optimize import brute
from numpy import exp
import matplotlib.pyplot as plt


# %% Constants


d_gs = 2.87  # ground state zfs in GHz
# d_gs = 1.42  # excited state zfs in GHz
gmuB = 2.8  # gyromagnetic ratio in MHz / G
gmuB_GHz = gmuB / 1000  # gyromagnetic ratio in GHz / G

# numbers
inv_sqrt_2 = 1 / np.sqrt(2)
im = 0 + 1j


# %% Functions


def conj_trans(matrix):
    return np.conj(np.transpose(matrix))


def calc_matrix_elements(
    noise_hamiltonian, mag_B, theta_B, par_Pi, perp_Pi, phi_B, phi_Pi
):

    popt_full = [theta_B, par_Pi, perp_Pi, phi_B, phi_Pi]
    vecs = calc_eigenvectors(mag_B, *popt_full)  # zero, low, high

    zero_to_low_el = np.matmul(noise_hamiltonian, vecs[1])
    zero_to_low_el = np.matmul(conj_trans(vecs[0]), zero_to_low_el)

    zero_to_high_el = np.matmul(noise_hamiltonian, vecs[2])
    zero_to_high_el = np.matmul(conj_trans(vecs[0]), zero_to_high_el)

    low_to_high_el = np.matmul(noise_hamiltonian, vecs[2])
    low_to_high_el = np.matmul(conj_trans(vecs[1]), low_to_high_el)

    return zero_to_low_el, zero_to_high_el, low_to_high_el


def plot_components(mag_B, popt):

    # mode = 'theta_B'
    mode = "mag_B"

    fig, axes_pack = plt.subplots(1, 3, figsize=(15, 5))
    # fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    fig.set_tight_layout(True)

    if mode == "theta_B":
        x_data = np.linspace(0, pi / 2, 1000)
        plot_x_data = x_data * (180 / pi)
        x_label = "B field angle (deg)"
    elif mode == "mag_B":
        # Matching the Tetienne photodynamics paper
        # 0-50 mT
        plot_x_data = np.linspace(0.001, 50.0, 1000)  # mT
        x_data = plot_x_data * 10  # G
        x_data = x_data * gmuB_GHz  # GHz
        x_label = "\(B\) (mT)"

        # x_data = np.linspace(0.001, 25.0, 1000)  # GHz
        # plot_x_data = x_data
        # x_label = 'B field magnitude (GHz)'
        # plot_x_data = x_data*1000/gmuB  # G
        # x_label = 'B field magnitude (MHz)'

    # theta_Bs = [6, 54.1]
    # line_styles = [None, '--']
    labels = [[r"$\ket{H;0}$", r"$\ket{H;-1}$", r"$\ket{H;+1}$"]]
    for ind in range(1):
        # theta_B = theta_Bs[ind] * (pi/180)
        # line_style = line_styles[ind]

        zero_zero_comps = []
        low_zero_comps = []
        high_zero_comps = []

        zero_plus_comps = []
        low_plus_comps = []
        high_plus_comps = []

        zero_minus_comps = []
        low_minus_comps = []
        high_minus_comps = []

        for val in x_data:

            if mode == "theta_B":
                popt[0] = val
            elif mode == "mag_B":
                mag_B = val

            # vecs are returned zero, low, high
            # components are ordered +1, 0, -1
            vecs = calc_eigenvectors(mag_B, *popt)

            zero_zero_comps.append(np.abs(vecs[0, 1]) ** 2)
            low_zero_comps.append(np.abs(vecs[1, 1]) ** 2)
            high_zero_comps.append(np.abs(vecs[2, 1]) ** 2)

            zero_plus_comps.append(np.abs(vecs[0, 0]) ** 2)
            low_plus_comps.append(np.abs(vecs[1, 0]) ** 2)
            high_plus_comps.append(np.abs(vecs[2, 0]) ** 2)

            zero_minus_comps.append(np.abs(vecs[0, 2]) ** 2)
            low_minus_comps.append(np.abs(vecs[1, 2]) ** 2)
            high_minus_comps.append(np.abs(vecs[2, 2]) ** 2)

        # |Sz;+1> projections
        ax = axes_pack[0]
        ax.set_title(r"$\ket{S_{z};+1}$ projections")
        ax.plot(plot_x_data, zero_plus_comps, label=labels[ind][0])
        ax.plot(plot_x_data, low_plus_comps, label=labels[ind][1])
        ax.plot(plot_x_data, high_plus_comps, label=labels[ind][2])
        ax.set_xlabel(x_label)
        ax.set_ylabel(r"$\abs{\bra{S_{z};+1}\ket{\Psi}}^2$")
        ax.legend()

        # |Sz;0> projections
        ax = axes_pack[1]
        ax.set_title(r"$\ket{S_{z};0}$ projections")
        ax.plot(plot_x_data, zero_zero_comps, label=labels[ind][0])
        ax.plot(plot_x_data, low_zero_comps, label=labels[ind][1])
        ax.plot(plot_x_data, high_zero_comps, label=labels[ind][2])
        ax.set_xlabel(x_label)
        ax.set_ylabel(r"$\abs{\bra{S_{z};0}\ket{\Psi}}^2$")
        ax.legend()

        # |Sz;-1> projections
        ax = axes_pack[2]
        ax.set_title(r"$\ket{S_{z};-1}$ projections")
        ax.plot(plot_x_data, zero_minus_comps, label=labels[ind][0])
        ax.plot(plot_x_data, low_minus_comps, label=labels[ind][1])
        ax.plot(plot_x_data, high_minus_comps, label=labels[ind][2])
        ax.set_xlabel(x_label)
        ax.set_ylabel(r"$\abs{\bra{S_{z};-1}\ket{\Psi}}^2$")
        ax.legend()


def b_matrix_elements(name, res_descs):

    sq_allow_factors = []  # zero to low
    dq_allow_factors = []  # low to high

    zero_zero_comps = []
    low_zero_comps = []
    high_zero_comps = []

    popt = main(name, res_descs)

    noise_params = [0.20, pi / 4, 0.0, 0.0, 0.0, 0.0]
    noise_hamiltonian = calc_hamiltonian(*noise_params)

    smooth_mag_Bs = np.linspace(0, 1.0, 1000)
    for mag_B in smooth_mag_Bs:
        vecs = calc_eigenvectors(mag_B, *popt)  # zero, low, high
        zero_zero_comps.append(np.abs(vecs[0, 1]) ** 2)
        low_zero_comps.append(np.abs(vecs[1, 1]) ** 2)
        high_zero_comps.append(np.abs(vecs[2, 1]) ** 2)

        ret_vals = calc_matrix_elements(noise_hamiltonian, mag_B, *popt)
        zero_to_low_el, zero_to_high_el, low_to_high_el = ret_vals

        sq_allow_factors.append(np.abs(zero_to_low_el) ** 2)
        dq_allow_factors.append(np.abs(low_to_high_el) ** 2)

    sq_allow_factors = np.array(sq_allow_factors)
    dq_allow_factors = np.array(dq_allow_factors)

    fig, ax = plt.subplots(figsize=(8.5, 8.5))
    fig.set_tight_layout(True)
    ax.set_title("Generating fit vector: {}".format(name))
    ax.set_xscale("log")
    ax.set_yscale("log")
    splittings = calc_splitting(smooth_mag_Bs, *popt)

    ax.plot(splittings, dq_allow_factors / sq_allow_factors)
    ax.plot(splittings, splittings ** -2 / 500)
    ax.set_xlabel("B magnitude (GHz)")
    ax.set_ylabel("DQ/SQ rate ratio")

    # ax.plot(splittings, dq_allow_factors)
    # ax.plot(splittings, splittings**-2 / 1000)
    # ax.set_xlabel('Splitting (GHz)')
    # ax.set_ylabel('DQ Matrix Element')

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
    param_bounds = ((0, pi / 2), (0, 2 * pi / 3))
    guess_params = (pi / 3, 0)

    args = (rotated_res_desc, mag_B, par_Pi, perp_Pi, phi_Pi)
    res = minimize(
        find_B_orientation_objective,
        guess_params,
        args=args,
        bounds=param_bounds,
        method="SLSQP",
    )

    return res.x


def find_B_orientation_objective(
    fit_vec, rotated_res_desc, mag_B, par_Pi, perp_Pi, phi_Pi
):
    calculated_res_pair = calc_res_pair(
        mag_B, fit_vec[0], par_Pi, perp_Pi, fit_vec[1], phi_Pi
    )
    diffs = np.array(calculated_res_pair) - np.array(rotated_res_desc[1:3])
    sum_squared_differences = np.sum(diffs ** 2)
    return sum_squared_differences


def extract_rotated_hamiltonian(
    name, res_descs, aligned_res_desc, rotated_res_desc
):

    # [theta_B, par_Pi, perp_Pi, phi_B, phi_Pi]
    popt = main(name, res_descs)  # Excluding phis

    mag_B = find_mag_B(aligned_res_desc, *popt)
    theta_B, phi_B = find_B_orientation(
        rotated_res_desc, mag_B, popt[1], popt[2], popt[4]
    )

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
        print("{}, ".format(res_desc))


def calc_single_hamiltonian(mag_B, theta_B, par_Pi, perp_Pi, phi_B, phi_Pi):
    par_B = mag_B * np.cos(theta_B)
    perp_B = mag_B * np.sin(theta_B)
    hamiltonian = np.array(
        [
            [
                d_gs + par_Pi + par_B,
                inv_sqrt_2 * perp_B * exp(-1j * phi_B),
                -perp_Pi * exp(1j * phi_Pi),
            ],
            [
                inv_sqrt_2 * perp_B * exp(1j * phi_B),
                0,
                inv_sqrt_2 * perp_B * exp(-1j * phi_B),
            ],
            [
                -perp_Pi * exp(-1j * phi_Pi),
                inv_sqrt_2 * perp_B * exp(1j * phi_B),
                d_gs + par_Pi - par_B,
            ],
        ]
    )
    return hamiltonian


def calc_hamiltonian(mag_B, theta_B, par_Pi, perp_Pi, phi_B, phi_Pi):
    fit_vec = [theta_B, par_Pi, perp_Pi, phi_B, phi_Pi]
    if (type(mag_B) is list) or (type(mag_B) is np.ndarray):
        hamiltonian_list = [
            calc_single_hamiltonian(val, *fit_vec) for val in mag_B
        ]
        return hamiltonian_list
    else:
        return calc_single_hamiltonian(mag_B, *fit_vec)


def calc_single_B_hamiltonian(mag_B, theta_B, phi_B):
    par_B = mag_B * np.cos(theta_B)
    perp_B = mag_B * np.sin(theta_B)
    hamiltonian = np.array(
        [
            [par_B, inv_sqrt_2 * perp_B * exp(-1j * phi_B), 0],
            [
                inv_sqrt_2 * perp_B * exp(1j * phi_B),
                0,
                inv_sqrt_2 * perp_B * exp(-1j * phi_B),
            ],
            [0, inv_sqrt_2 * perp_B * exp(1j * phi_B), -par_B],
        ]
    )
    return hamiltonian


def calc_B_hamiltonian(mag_B, theta_B, phi_B):
    args = (mag_B, theta_B, phi_B)
    if (type(mag_B) is list) or (type(mag_B) is np.ndarray):
        hamiltonian_list = [calc_single_B_hamiltonian(*args) for val in mag_B]
        return hamiltonian_list
    else:
        return calc_single_B_hamiltonian(*args)


def calc_single_static_cartesian_B_hamiltonian(B_x, B_y, B_z):
    hamiltonian = np.array(
        [
            [d_gs + B_z, inv_sqrt_2 * (B_x - im * B_y), 0],
            [inv_sqrt_2 * (B_x + im * B_y), 0, inv_sqrt_2 * (B_x - im * B_y)],
            [0, inv_sqrt_2 * (B_x + im * B_y), d_gs - B_z],
        ]
    )
    return hamiltonian


def calc_static_cartesian_B_hamiltonian(B_x, B_y, B_z):
    args = (B_x, B_y, B_z)
    if (type(B_x) is list) or (type(B_x) is np.ndarray):
        hamiltonian_list = [
            calc_single_static_cartesian_B_hamiltonian(*args) for val in B_x
        ]
        return hamiltonian_list
    else:
        return calc_single_static_cartesian_B_hamiltonian(*args)


def calc_single_Pi_hamiltonian(mag_Pi, theta_Pi, phi_Pi):
    d_par = 0.3
    d_perp = 17.0
    par_Pi = d_par * mag_Pi * np.cos(theta_Pi)
    perp_Pi = d_perp * mag_Pi * np.sin(theta_Pi)
    hamiltonian = np.array(
        [
            [par_Pi, 0, perp_Pi * exp(1j * phi_Pi)],
            [0, 0, 0],
            [perp_Pi * exp(-1j * phi_Pi), 0, par_Pi],
        ]
    )
    return hamiltonian


def calc_Pi_hamiltonian(perp_Pi, theta_Pi, phi_Pi):
    args = (perp_Pi, theta_Pi, phi_Pi)
    if (type(perp_Pi) is list) or (type(perp_Pi) is np.ndarray):
        hamiltonian_list = [
            calc_single_Pi_hamiltonian(*args) for val in perp_Pi
        ]
        return hamiltonian_list
    else:
        return calc_single_Pi_hamiltonian(*args)


def calc_res_pair(mag_B, theta_B, par_Pi, perp_Pi, phi_B, phi_Pi):
    hamiltonian = calc_hamiltonian(
        mag_B, theta_B, par_Pi, perp_Pi, phi_B, phi_Pi
    )
    if (type(mag_B) is list) or (type(mag_B) is np.ndarray):
        vals = np.sort(eigvals(hamiltonian), axis=1)
        resonance_low = np.real(vals[:, 1] - vals[:, 0])
        resonance_high = np.real(vals[:, 2] - vals[:, 0])
    else:
        vals = np.sort(eigvals(hamiltonian))
        resonance_low = np.real(vals[1] - vals[0])
        resonance_high = np.real(vals[2] - vals[0])
    return resonance_low, resonance_high


def calc_splitting(mag_B, theta_B, par_Pi, perp_Pi, phi_B, phi_Pi):
    args = (mag_B, theta_B, par_Pi, perp_Pi, phi_B, phi_Pi)
    res_pair = calc_res_pair(*args)
    return res_pair[1] - res_pair[0]


def calc_eigenvectors(mag_B, theta_B, par_Pi, perp_Pi, phi_B, phi_Pi):
    """Return the normalized eigenvectors, sorted by ascending eigenvalue"""
    hamiltonian = calc_hamiltonian(
        mag_B, theta_B, par_Pi, perp_Pi, phi_B, phi_Pi
    )
    eigvals, eigvecs = eig(hamiltonian)
    sorted_indices = np.argsort(eigvals)
    # sorted_eigvecs = [np.round(eigvecs[:,ind], 3) for ind in sorted_indices]
    sorted_eigvecs = [eigvecs[:, ind] for ind in sorted_indices]
    sorted_eigvecs = np.array(sorted_eigvecs)
    # for vec in sorted_eigvecs:
    #     print(np.matmul(hamiltonian, vec) / vec)
    return sorted_eigvecs


def calc_eig_static_cartesian_B(B_x, B_y, B_z):
    """
    Return the normalized eigenvectors and eigenvalues,
    sorted by ascending eigenvalue.
    """
    hamiltonian = calc_static_cartesian_B_hamiltonian(B_x, B_y, B_z)
    eigvals, eigvecs = eig(hamiltonian)
    eigvals = np.real(eigvals)  # ditch the complex part
    sorted_eigvals = np.sort(eigvals)
    sorted_indices = np.argsort(eigvals)
    # sorted_eigvecs = [np.round(eigvecs[:,ind], 3) for ind in sorted_indices]
    sorted_eigvecs = [eigvecs[:, ind] for ind in sorted_indices]
    sorted_eigvecs = np.array(sorted_eigvecs)
    # for vec in sorted_eigvecs:
    #     print(np.matmul(hamiltonian, vec) / vec)
    return sorted_eigvecs, sorted_eigvals


def find_mag_B(res_desc, theta_B, par_Pi, perp_Pi, phi_B, phi_Pi):
    # Just return the given mag_B if it's known
    if res_desc[0] is not None:
        return res_desc[0]
    # Otherwise we'll determine the most likely mag_B for this fit_vec by
    # finding the mag_B that minimizes the distance between the measured
    # resonances and the calculated resonances for a given fit_vec
    args = (res_desc, theta_B, par_Pi, perp_Pi, phi_B, phi_Pi)
    result = minimize_scalar(
        find_mag_B_objective, bounds=(0, 1.0), args=args, method="bounded"
    )
    if result.success:
        mag_B = result.x
    else:
        # If we didn't find an optimal value, return something that will blow
        # up chisq and push us away from this fit_vec
        mag_B = 0.0
    return mag_B


def find_mag_B_objective(x, res_desc, theta_B, par_Pi, perp_Pi, phi_B, phi_Pi):
    calculated_res_pair = calc_res_pair(
        x, theta_B, par_Pi, perp_Pi, phi_B, phi_Pi
    )
    diffs = np.array(calculated_res_pair) - np.array(res_desc[1:3])
    sum_squared_differences = np.sum(diffs ** 2)
    return sum_squared_differences


def find_mag_B_splitting(splitting, theta_B, par_Pi, perp_Pi, phi_B, phi_Pi):
    # Otherwise we'll determine the most likely mag_B for this fit_vec by
    # finding the mag_B that minimizes the distance between the measured
    # resonances and the calculated resonances for a given fit_vec
    args = (splitting, theta_B, par_Pi, perp_Pi, phi_B, phi_Pi)
    result = minimize_scalar(
        find_mag_B_splitting_objective,
        bounds=(0, 1.5),
        args=args,
        method="bounded",
    )
    if result.success:
        mag_B = result.x
    else:
        mag_B = 0.0
    return mag_B


def find_mag_B_splitting_objective(
    x, splitting, theta_B, par_Pi, perp_Pi, phi_B, phi_Pi
):
    calculated_splitting = calc_splitting(
        x, theta_B, par_Pi, perp_Pi, phi_B, phi_Pi
    )
    return np.abs(calculated_splitting - splitting)


def plot_resonances(
    mag_B_range, theta_B, par_Pi, perp_Pi, phi_B, phi_Pi, name="untitled"
):

    smooth_mag_B = np.linspace(mag_B_range[0], mag_B_range[1], 1000)
    res_pairs = calc_res_pair(
        smooth_mag_B, theta_B, par_Pi, perp_Pi, phi_B, phi_Pi
    )

    fig, ax = plt.subplots(figsize=(8.5, 8.5))
    fig.set_tight_layout(True)
    ax.set_title("Generating fit vector: {}".format(name))
    ax.plot(smooth_mag_B, res_pairs[0])
    ax.plot(smooth_mag_B, res_pairs[1])
    ax.set_xlabel("B magnitude (GHz)")
    ax.set_ylabel("Resonance (GHz)")

    textstr = "\n".join(
        (
            r"$\theta_{B}=%.3f \ rad$" % (theta_B,),
            r"$\Pi_{\parallel}=%.3f \ GHz$" % (par_Pi,),
            r"$\Pi_{\perp}=%.3f \ GHz$" % (perp_Pi,),
        )
    )
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    ax.text(
        0.05,
        0.95,
        textstr,
        fontsize=14,
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=props,
    )

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
    squared_residuals = [
        find_mag_B_objective(
            mag_Bs[ind], res_descs[ind], *fit_vec, phi_B, phi_Pi
        )
        for ind in range(num_resonance_descs)
    ]
    sum_squared_residuals = np.sum(squared_residuals)

    estimated_st_dev = 0.0001
    estimated_var = np.sqrt(estimated_st_dev)
    chisq = sum_squared_residuals / estimated_var

    return chisq


# %% Main


def main(name, res_descs):

    ############ Setup ############

    phi_B = 0.0
    phi_Pi = 0.0
    # phi_B = pi/3
    # phi_Pi = pi/3

    # fit_vec = [theta_B, par_Pi, perp_Pi,]
    param_bounds = ((0, pi / 2), (-0.050, 0.050), (0, 0.050))

    res_descs = np.array(res_descs)
    for desc in res_descs:
        # Set degenerate resonances to the same value
        if desc[2] is None:
            desc[2] = desc[1]
        # Make sure resonances are sorted
        desc[1:3] = np.sort(desc[1:3])

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
    res = minimize(
        chisq_func,
        guess_params,
        args=args,
        bounds=param_bounds,
        method="SLSQP",
    )
    if not res.success:
        print(res.message)
        return

    popt = res.x
    popt_full = np.append(popt, [phi_B, phi_Pi])
    # popt_full[0] = 0.5

    chisq = res.fun
    print("Chi squared: {:.4g}".format(chisq))
    degrees_of_freedom = len(res_descs) - len(x0)
    reduced_chisq = res.fun / degrees_of_freedom
    print("Reduced chi squared: {:.4g}".format(reduced_chisq))

    ############ Plot the result ############

    # Get the mag_B for each pair of resonances with this fit_vec
    mag_Bs = [find_mag_B(desc, *popt_full) for desc in res_descs]

    # Plot the calculated resonances up to the max mag_B
    fig, ax = plot_resonances([0, max(mag_Bs)], *popt_full, name)

    # Plot the resonances
    ax.scatter(mag_Bs, res_descs[:, 1])
    ax.scatter(mag_Bs, res_descs[:, 2])

    # Return the full 5 parameters
    print(list(popt_full))
    return popt_full


# %% Run the file


# The __name__ variable will only be '__main__' if you run this file directly.
# This allows a file's functions, classes, etc to be imported without running
# the script that you set up here.
if __name__ == "__main__":

    # Plots with LaTeX
    plt.rcParams["text.latex.preamble"] = [
        r"\usepackage{physics}",
        r"\usepackage{sfmath}",
        r"\usepackage{upgreek}",
        r"\usepackage{helvet}",
    ]
    plt.rcParams.update({"font.size": 12})
    # plt.rcParams.update({'font.family': 'sans-serif'})
    # plt.rcParams.update({'font.sans-serif': ['Helvetica']})
    plt.rc("text", usetex=True)

    ############ Fake ############

    # name = 'fake'
    # res_descs = [[0.0, 2.845, 2.895],
    #             [None, 2.838, 2.902],
    #             [None, 2.805, 2.935],
    #             [None, 2.767, 2.973],
    #             [None, 2.619, 3.121],
    #             [None, 2.369, 3.371],
    #             ]

    ############ ND ############

    # name = 'nv2_2019_04_30'
    # popt = [0.6398153129728315, -0.0044880947609542005, 0.0070490732314452695, 0.0, 0.0]
    # res_descs = [[0.0, 2.8584, 2.8725],
    #                 [None, 2.8507, 2.8798],
    #                 [None, 2.8434, 2.8882],
    #                 [None, 2.8380, 2.8942],
    #                 [None, 2.8379, 2.8948],
    #                 [None, 2.8308, 2.9006],
    #                 [None, 2.8228, 2.9079],
    #                 [None, 2.8155, 2.9171],
    #                 ]

    name = "nv2_2019_04_30_take2"
    popt = [
        1.1162003323335492,
        -0.0031494625116033634,
        0.007006402029975579,
        0.0,
        0.0,
    ]
    res_descs = [
        [0.0, 2.8584, 2.8725],
        [None, 2.8512, 2.8804],
        [None, 2.8435, 2.8990],
        [None, 2.8265, 2.9117],
        [None, 2.7726, 3.0530],
        [None, 2.7738, 3.4712],
    ]

    # name = 'nv1_2019_05_10'  # NV1
    # popt = [0.6474219686681678, -0.005159086817872651, 0.009754609612326834, 0.0, 0.0]
    # popt = [0.6500999024309339, -0.005162723325066839, 0.009743779800841193, 1.0471975511965976, 1.0471975511965976]
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
    #               # [None, 2.7374, 3.0874],  # misaligned
    #               # [None, 2.6310, 3.1547],  # misaligned ref for prev
    #               [None, 2.6061, 3.1678],
    #               # [None, 2.6055, 3.1691],  # repeat of previous
    #               [None, 2.4371, 3.4539],  # 0,-1 and 0,+1 omegas
    #               # [None, 2.4381, 3.4531],   # retake 0,-1 and 0,+1 omegas
    #               ]

    ############ Bulk ############

    # name = 'goeppert_mayer-nv7_2019_11_27'
    # res_descs = [[0.0, 2.8703, None],
    #               [None, 2.8508, 2.8914],  # 41 MHz
    #               [None, 2.7893, 2.9564],  # 167 MHz
    #               [None, 2.6719, 3.0846],  # 413 MHz
    #               [None, 2.5760, 3.1998],  # 624 MHz
    #               [None, 2.4857, 3.3173],  # 832 MHz
    #               [None, 2.3298, 3.5369],  # 1207 MHz
    #               [None, 2.7775, 2.9687],  # spin_echo
    #               [None, 2.7243, 3.0243],
    #               [None, 2.7113, 3.0402],
    #               [None, 2.6270, 3.1365],
    #               [None, 2.4991, 3.2991],
    #               [None, 2.4290, 3.3976],
    #               [None, 2.3844, 3.4575],
    #               [None, 2.3293, 3.5355],
    #               [None, 2.8028, 2.9413],
    #               [None, 2.8286, 2.9116],
    #               [None, 2.8466, 2.8937],
    #               [None, 2.8302, 2.9098],
    #               [None, 2.7706, 2.9749],
    #               ]

    # Run the script
    # main(name, res_descs)

    # popt: theta_B, par_Pi, perp_Pi, phi_B, phi_Pi

    mag_B = 0.5 * gmuB_GHz
    # popt = [55*(pi/180), 0, 0, 0, 0]
    popt = [np.pi / 2, 0, 0, 0, 0]
    # print(calc_eigenvectors(mag_B, *popt))
    # vecs = calc_eigenvectors(mag_B, *popt)
    # for vec in vecs:
    #     vec = np.array(vec)
    #     print(np.abs(vec)**2)
    # b_matrix_elements(name, res_descs)
    # plot_components(mag_B, popt)
    # plot_resonances([0,0.2], *popt)
    print(calc_res_pair(mag_B, *popt))

    # Fake data
    # bounds: ((0, pi/2), (-0.050, 0.050), (0, 0.050), (0, pi/3), (0, 2*pi/3))
    # generate_fake_data(pi/4, 0.000, 0.000, 0.000, 0.000)
