# -*- coding: utf-8 -*-
"""
Modification of extract_hamiltonian used to to determine the field parameters
from the resonances for a set of 4 NVs of different orientations

Created on February 21st, 2023

@author: mccambria
"""


# region Imports and constants

import numpy as np
from numpy.linalg import eigvals
from numpy import pi
from scipy.optimize import minimize_scalar
from scipy.optimize import minimize
from scipy.optimize import brute
from numpy import exp
import matplotlib.pyplot as plt
import utils.kplotlib as kpl

d_gs = 2.870  # ground state zfs in GHz
gmuB = 2.8  # gyromagnetic ratio in MHz / G
gmuB_GHz = gmuB / 1000  # gyromagnetic ratio in GHz / G
inv_sqrt_2 = 1 / np.sqrt(2)
im = 0 + 1j

# endregion
# region Functions


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
        hamiltonian_list = [calc_single_hamiltonian(val, *fit_vec) for val in mag_B]
        return hamiltonian_list
    else:
        return calc_single_hamiltonian(mag_B, *fit_vec)


def calc_res_pair(mag_B, theta_B, par_Pi, perp_Pi, phi_B, phi_Pi):
    hamiltonian = calc_hamiltonian(mag_B, theta_B, par_Pi, perp_Pi, phi_B, phi_Pi)
    if (type(mag_B) is list) or (type(mag_B) is np.ndarray):
        vals = np.sort(eigvals(hamiltonian), axis=1)
        resonance_low = np.real(vals[:, 1] - vals[:, 0])
        resonance_high = np.real(vals[:, 2] - vals[:, 0])
    else:
        vals = np.sort(eigvals(hamiltonian))
        resonance_low = np.real(vals[1] - vals[0])
        resonance_high = np.real(vals[2] - vals[0])
    return resonance_low, resonance_high


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
    calculated_res_pair = calc_res_pair(x, theta_B, par_Pi, perp_Pi, phi_B, phi_Pi)
    diffs = np.array(calculated_res_pair) - np.array(res_desc[1:3])
    sum_squared_differences = np.sum(diffs**2)
    return sum_squared_differences


def plot_resonances(
    mag_B_range, theta_B, par_Pi, perp_Pi, phi_B, phi_Pi, name="untitled"
):

    smooth_mag_B = np.linspace(mag_B_range[0], mag_B_range[1], 1000)
    res_pairs = calc_res_pair(smooth_mag_B, theta_B, par_Pi, perp_Pi, phi_B, phi_Pi)

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


def plot_resonances_custom():
    """Plot the resonances as a function of something"""

    smooth_Pi = np.linspace(0, 0.025, 100)
    res_pairs = [calc_res_pair(0.0, 0.0, 0.0, val, 0.0, 0.0) for val in smooth_Pi]
    res_pairs = np.array(res_pairs)

    fig, ax = plt.subplots()
    kpl.plot_line(ax, smooth_Pi, res_pairs[:, 0])
    kpl.plot_line(ax, smooth_Pi, res_pairs[:, 1])
    ax.set_xlabel("Pi magnitude (GHz)")
    ax.set_ylabel("Resonance (GHz)")

    return fig, ax


def cost_func(e_field, meas_zfs_list, meas_splitting_list):

    test_zfs_list, test_splitting_list = calc_zfs_and_splitting(e_field)

    residuals = []
    for ind in range(4):
        residuals.append(test_zfs_list[ind] - meas_zfs_list[ind])
        residuals.append(test_splitting_list[ind] - meas_splitting_list[ind])
    squared_residuals = [el**2 for el in residuals]
    cost = np.sum(squared_residuals)
    # cost = np.sqrt(np.std(residuals[0::2]) ** 2 + np.std(residuals[1::2]))

    return cost


def optimize_e_field(meas_zfs_list, meas_splitting_list):

    ### Setup

    param_bounds = [(-1, 1), (-1, 1), (-1, 1)]
    args = (meas_zfs_list, meas_splitting_list)

    ### Brute force first pass

    param_ranges = param_bounds
    x0 = brute(cost_func, param_ranges, args=args, Ns=100, workers=-1)
    guess_e_field = list(x0)

    ### Fine tuning with minimize

    res = minimize(
        cost_func,
        guess_e_field,
        args=args,
        bounds=param_bounds,
        method="SLSQP",
    )

    ### Return

    if not res.success:
        print(res.message)
    opti_e_field = res.x
    print(f"Optimized SSR: {res.fun}")
    print(f"Optimized E field: {opti_e_field}")
    return opti_e_field


def calc_zfs_and_splitting(e_field):

    e_field = [el / 1000 for el in e_field]
    e_field_x, e_field_y, e_field_z = e_field

    # For a given E field, plot the ZFS deviation and splitting for 4 orientations
    # NV unit vectors - viewed from above with a and d sticking out of the page (+z)
    # ....a....
    # ....|....
    # b---+---c
    # ....|....
    # ....d....
    bond_angle = np.arccos(-1 / 3)
    half_bond_angle = bond_angle / 2
    nv_orientations = [
        [0, np.sin(half_bond_angle), np.cos(half_bond_angle)],  # a
        [-np.sin(half_bond_angle), 0, -np.cos(half_bond_angle)],  # b
        [np.sin(half_bond_angle), 0, -np.cos(half_bond_angle)],  # c
        [0, -np.sin(half_bond_angle), np.cos(half_bond_angle)],  # d
    ]

    e_field = np.array([e_field_x, e_field_y, e_field_z])

    e_field_components = [
        [np.dot(nv, e_field), np.linalg.norm(np.cross(nv, e_field))]
        for nv in nv_orientations
    ]
    Phi_components = [[0.35 * comp[0], 17 * comp[1]] for comp in e_field_components]
    # Phi_components = [[comp[0], comp[1]] for comp in e_field_components]

    calc_res_pair_Phi = lambda comps: calc_res_pair(0, 0, comps[0], comps[1], 0, 0)
    calc_res_pair_B = lambda comps: calc_res_pair(
        np.sqrt(comps[0] ** 2 + comps[1] ** 2),
        np.arctan(comps[1] / comps[0]),
        0,
        0,
        0,
        0,
    )
    resonances = [calc_res_pair_Phi(comps) for comps in Phi_components]
    # resonances = [calc_res_pair_B(comps) for comps in Phi_components]

    zfs_list = [np.mean(pair) for pair in resonances]
    splitting_list = [pair[1] - pair[0] for pair in resonances]

    # Convert to MHz
    zfs_list = [1000 * el for el in zfs_list]
    splitting_list = [1000 * el for el in splitting_list]

    return zfs_list, splitting_list


# endregion


def main(meas_zfs_list, meas_splitting_list):

    bond_angle = np.arccos(-1 / 3)
    half_bond_angle = (0.7) * bond_angle / 2
    orientation = [0.2, np.sin(half_bond_angle), np.cos(half_bond_angle)]
    e_field = 1 * np.array(orientation)
    opti_zfs_list, opti_splitting_list = calc_zfs_and_splitting(e_field)

    opti_e_field = optimize_e_field(meas_zfs_list, meas_splitting_list)
    opti_zfs_list, opti_splitting_list = calc_zfs_and_splitting(opti_e_field)

    fig, ax = plt.subplots()
    # plot_splitting_list = [el * 1000 for el in splitting_list]
    # plot_zfs_list = [el * 1000 - 2870 for el in zfs_list]
    # kpl.plot_points(ax, plot_splitting_list, plot_zfs_list, label="Optimized")
    kpl.plot_points(ax, opti_splitting_list, opti_zfs_list, label="Optimized")
    kpl.plot_points(ax, meas_splitting_list, meas_zfs_list, label="Measured")
    ax.set_xlabel("Splitting (MHz)")
    # ax.set_ylabel("Deviation from 2.87 GHz (MHz)")
    ax.set_ylabel("ZFS (MHz)")
    ax.legend()


if __name__ == "__main__":

    kpl.init_kplotlib()

    # plot_resonances_custom()

    # Name for the NV, sample, whatever
    name = "Wu-region5"

    # All MHz
    # meas_zfs_list = [2870.5, 2870, 2871, 2870.5]
    # meas_zfs_list = [2870.5, 2870.5, 2870.5, 2870.5]
    meas_zfs_list = [2870, 2870, 2870, 2870]
    meas_splitting_list = [0.0, 6, 6, 12]

    # e_field = np.array([0.1, -0.001, -0.5])
    # meas_zfs_list, meas_splitting_list = calc_zfs_and_splitting(e_field)

    main(meas_zfs_list, meas_splitting_list)

    plt.show(block=True)
