# -*- coding: utf-8 -*-
"""
Deconvolve a spectral function into density of states and average coupling strength

Created on March 9th, 2023

@author: mccambria
"""


import utils.tool_belt as tool_belt
import utils.common as common
import utils.kplotlib as kpl
import csv
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.integrate import quad
from utils.tool_belt import bose

meV_to_K = 11.6045250061657  # 1 meV = 11.6045250061657 K


def einstein_term(e, T):
    return (((e / T) ** 2) * np.exp(e / T)) / ((np.exp(e / T) - 1) ** 2)


# def cambria_test(T, a0, coeff1, coeff2):
#     energies = [68, 150]  # meV
def cambria_test(T, a0, energy1, energy2, coeff1, coeff2):
    energies = [energy1, energy2]  # meV
    coeffs = [coeff1, coeff2]
    total = None
    for ind in range(2):
        coeff = coeffs[ind]
        energy = energies[ind]
        term = coeff * bose(energy, T)
        if total is None:
            total = term
        else:
            total += term
    return a0 + total


def double_occupation(T, a0, energy1, energy2, coeff1, coeff2):
    energies = [energy1, energy2]  # meV
    coeffs = [coeff1, coeff2]
    total = None
    for ind in range(2):
        coeff = coeffs[ind]
        energy = energies[ind]
        term = coeff * bose(energy, T)
        if total is None:
            total = term
        else:
            total += term
    return a0 + total


# def cambria_test_single(T, energy, coeff):
def cambria_test_single(T, coeff):
    energy = 160
    return coeff * einstein_term(energy * meV_to_K, T)


def jacobson(T):
    # X1[10−6/K]  0.0096  0.0210
    # X2[10−6/K]  0.2656  0.3897
    # X3[10−6/K]  2.6799  3.4447
    # X4[10−6/K]  2.3303  2.2796
    # Θ1 [K]      159.3   225.2
    # Θ2 [K]      548.5   634.0
    # Θ3 [K]      1237.9  1365.5
    # Θ4 [K]      2117.8  3068.8

    coeffs = [0.0096, 0.2656, 2.6799, 2.3303]
    energies = [159.3, 548.5, 1237.9, 2117.8]  # K
    # coeffs = [0.0210, 0.3897, 3.4447, 2.2796]
    # energies = [225.2, 634.0, 1365.5, 3068.8]  # K
    jacobson_total = None
    for ind in range(4):
        energy = energies[ind]
        coeff = coeffs[ind] * 10**-6
        sub_lambda = lambda t: coeff * einstein_term(energy, t)
        if jacobson_total is None:
            jacobson_total = sub_lambda(T)
        else:
            jacobson_total += sub_lambda(T)
        # kpl.plot_line(ax, temp_linspace, sub_lambda(temp_linspace), label=ind)
    return jacobson_total


def fig():

    fit_linspace = np.linspace(1, 1000, 1000)
    temp_linspace = np.linspace(1, 1000, 1000)
    a0 = 3.566503
    int_jacobson = []
    for temp in temp_linspace:
        int_jacobson.append(quad(jacobson, 10, temp)[0])
    jacobson_lattice_full = a0 * np.exp(int_jacobson)

    kpl_figsize = kpl.figsize
    adj_figsize = (kpl_figsize[0], 1.2 * kpl_figsize[1])
    fig, axes_pack = plt.subplots(2, 1, sharex=True, figsize=adj_figsize)
    ax1, ax2 = axes_pack

    guess_params = [a0, 68, 167, 0.3, 3.0]
    popt, pcov = curve_fit(
        double_occupation,
        fit_linspace,
        jacobson_lattice_constant(fit_linspace),
        p0=guess_params,
    )
    print(popt)
    double_occupation_lambda = lambda T: double_occupation(T, *popt)

    # Absolute
    kpl.plot_line(
        ax1,
        temp_linspace,
        jacobson_lattice_full,
        label="Jacobson",
        color=kpl.KplColors.RED,
    )
    # kpl.plot_line(
    #     ax1, temp_linspace, double_occupation_lambda(temp_linspace), label="Two-mode"
    # )
    # ax1.legend()
    ax1.set_ylabel(r"Lattice constant ($\si{\angstrom}$)")
    # ax1.set_xlabel("Temperature (K)")

    # Difference
    diff = 1e6 * (jacobson_lattice_full - double_occupation_lambda(temp_linspace))
    kpl.plot_line(ax2, temp_linspace, diff)
    ax2.set_ylabel(r"Model diffs. ($\si{\micro\angstrom}$)")
    ax2.set_xlabel("Temperature (K)")

    # ax1.xaxis.set_tick_params(labelbottom=False)
    # ax1.set_xticks([])
    # fig.get_layout_engine().set(hspace=0)

    fig.text(0, 0.965, "(a)")
    fig.text(0, 0.51, "(b)")


def jacobson_lattice_constant(T):
    # X1[10−6/K]  0.0096  0.0210
    # X2[10−6/K]  0.2656  0.3897
    # X3[10−6/K]  2.6799  3.4447
    # X4[10−6/K]  2.3303  2.2796
    # Θ1 [K]      159.3   225.2
    # Θ2 [K]      548.5   634.0
    # Θ3 [K]      1237.9  1365.5
    # Θ4 [K]      2117.8  3068.8

    coeffs = [0.0096, 0.2656, 2.6799, 2.3303]
    energies = [159.3, 548.5, 1237.9, 2117.8]  # K
    # coeffs = [0.0210, 0.3897, 3.4447, 2.2796]
    # energies = [225.2, 634.0, 1365.5, 3068.8]  # K
    jacobson_total = None
    for ind in range(4):
        energy = energies[ind]
        coeff = coeffs[ind] * 10**-6
        sub_lambda = lambda t: coeff * energy * bose(energy / meV_to_K, t)
        if jacobson_total is None:
            jacobson_total = sub_lambda(T)
        else:
            jacobson_total += sub_lambda(T)
        # kpl.plot_line(ax, temp_linspace, sub_lambda(temp_linspace), label=ind)
    return 3.566503 * np.exp(jacobson_total)
    # return 3.566503 * (1 + jacobson_total)  # 0K from Sato 2002


def main():

    # occupation = lambda e, t: 1 / (np.exp(e / t) - 1)

    fit_linspace = np.linspace(10, 500, 1000)
    temp_linspace = np.linspace(100, 500, 1000)

    a0 = 3.566503

    fig, ax = plt.subplots()
    # kpl.plot_line(
    #     ax,
    #     temp_linspace,
    #     jacobson_lattice_constant(temp_linspace),
    #     label="Jacobson_closed",
    # )
    # return
    # kpl.plot_line(ax, norm_energy, einstein_term(norm_energy), label="Einstein")
    # kpl.plot_line(ax, norm_energy, 40 * occupation(norm_energy), label="Occupation")

    int_jacobson = []
    for temp in temp_linspace:
        int_jacobson.append(quad(jacobson, 10, temp)[0])
    jacobson_lattice_full = a0 * np.exp(int_jacobson)
    # kpl.plot_line(ax, temp_linspace, 3.566503 * np.exp(int_jacobson), label="Jacobson")

    # guess_params = [3.0]
    # guess_params = [3.566503, 0.3, 3.0]
    guess_params = [3.566503, 68, 167, 0.3, 3.0]
    # # guess_params = [167, 3.0]
    fn = cambria_test
    # fn = cambria_test_single
    popt, pcov = curve_fit(
        fn, fit_linspace, jacobson_lattice_constant(fit_linspace), p0=guess_params
    )
    print(popt)
    fn_lambda = lambda T: fn(T, *popt)
    # int_cambria = []
    # for temp in temp_linspace:
    #     int_cambria.append(quad(cambria, 5, temp)[0])
    # kpl.plot_line(ax, temp_linspace, fn_lambda(temp_linspace), label="Cambria")
    kpl.plot_line(
        ax,
        temp_linspace,
        (jacobson_lattice_full - fn_lambda(temp_linspace))
        / (jacobson_lattice_full - a0),
        label="Jacobson - Cambria",
    )
    # print(popt)

    ax.legend()
    ax.set_ylabel("Relative difference")
    ax.set_xlabel("Temperature (K)")


if __name__ == "__main__":

    kpl.init_kplotlib(latex=True)
    # main()
    fig()
    plt.show(block=True)
