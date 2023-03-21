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

meV_to_K = 11.6045250061657  # 1 meV = 11.6045250061657 K


def einstein_term(e, T):
    return (((e / T) ** 2) * np.exp(e / T)) / ((np.exp(e / T) - 1) ** 2)


def cambria_test(T, coeff1, coeff2):
    energies = [60, 153]  # meV
    # def cambria_test(T, energy1, energy2, coeff1, coeff2):
    #     energies = [energy1, energy2]  # meV
    coeffs = [coeff1, coeff2]
    total = None
    for ind in range(2):
        coeff = coeffs[ind]
        energy_K = energies[ind] * meV_to_K
        term = coeff * einstein_term(energy_K, T)
        if total is None:
            total = term
        else:
            total += term
    return total


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

    # coeffs = [0.0096, 0.2656, 2.6799, 2.3303]
    # energies = [159.3, 548.5, 1237.9, 2117.8]  # K
    coeffs = [0.0210, 0.3897, 3.4447, 2.2796]
    energies = [225.2, 634.0, 1365.5, 3068.8]  # K
    jacobson_total = None
    for ind in range(4):
        energy = energies[ind]
        coeff = coeffs[ind] * 10**-6 * 10**-10
        sub_lambda = lambda t: coeff * einstein_term(energy, t)
        if jacobson_total is None:
            jacobson_total = sub_lambda(T)
        else:
            jacobson_total += sub_lambda(T)
        # kpl.plot_line(ax, temp_linspace, sub_lambda(temp_linspace), label=ind)
    return jacobson_total


def main():

    # occupation = lambda e, t: 1 / (np.exp(e / t) - 1)

    temp_linspace = np.linspace(10, 500, 100)

    fig, ax = plt.subplots()
    # kpl.plot_line(ax, norm_energy, einstein_term(norm_energy), label="Einstein")
    # kpl.plot_line(ax, norm_energy, 40 * occupation(norm_energy), label="Occupation")

    int_jacobson = []
    for temp in temp_linspace:
        int_jacobson.append(quad(jacobson, 5, temp)[0])
    kpl.plot_line(ax, temp_linspace, int_jacobson, label="Jacobson")

    guess_params = [3.0]
    # guess_params = [0.3, 3.0]
    # guess_params = [68, 167, 0.3, 3.0]
    # guess_params = [167, 3.0]
    # fn = cambria_test
    fn = cambria_test_single
    popt, pcov = curve_fit(fn, temp_linspace, jacobson(temp_linspace), p0=guess_params)
    cambria = lambda T: fn(T, *popt)
    int_cambria = []
    for temp in temp_linspace:
        int_cambria.append(quad(cambria, 5, temp)[0])
    kpl.plot_line(ax, temp_linspace, int_cambria, label="Cambria")
    print(popt)

    ax.legend()
    ax.set_ylabel("Term")
    ax.set_xlabel("Temp")


if __name__ == "__main__":

    kpl.init_kplotlib()
    main()
    plt.show(block=True)
