# -*- coding: utf-8 -*-
"""
Test why fitting to ab initio relaxation rates underestimates energies
Created on February 5th, 2023

@author: mccambria
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import utils.tool_belt as tool_belt
from utils.tool_belt import gaussian
import utils.common as common
import utils.kplotlib as kpl
from scipy.integrate import quad

Boltzmann = 8.617e-2  # meV / K
hbar = 6.582e-13  #  meV⋅s
from numpy import pi


def bose(energy, temp):
    # For very low temps we can get divide by zero and overflow warnings.
    # Fortunately, numpy is smart enough to know what we mean when this
    # happens, so let's let numpy figure it out and suppress the warnings.
    old_settings = np.seterr(divide="ignore", over="ignore")
    # print(energy / (Boltzmann * temp))
    val = 1 / (np.exp(energy / (Boltzmann * temp)) - 1)
    # val = temp / energy
    # Return error handling to default state for other functions
    np.seterr(**old_settings)
    return val


def double_orbach(temp, coeff1, delta1, coeff2, delta2):
    n1 = bose(delta1, temp)
    n2 = bose(delta2, temp)
    return (coeff1 * n1 * (n1 + 1)) + (coeff2 * n2 * (n2 + 1))


def single_orbach(temp, coeff1, delta1):
    n1 = bose(delta1, temp)
    return coeff1 * n1 * (n1 + 1)


def spectral_function(energy, coeff1, delta1, coeff2, delta2):

    func = lambda energy: spectral_function_sub(energy, coeff1, delta1, coeff2, delta2)

    if type(energy) in [list, np.ndarray]:
        ret_vals = [func(val) for val in energy]
        return np.array(ret_vals)
    else:
        return func(energy)


def spectral_function_single(energy, coeff1, delta1):

    func = lambda energy: spectral_function_sub_single(energy, coeff1, delta1)

    if type(energy) in [list, np.ndarray]:
        ret_vals = [func(val) for val in energy]
        return np.array(ret_vals)
    else:
        return func(energy)


def spectral_function_sub(energy, coeff1, delta1, coeff2, delta2):

    # mode = "box"
    mode = "gaussian"
    # mode = "comb"

    half_width = 10
    # half_width = 1

    if mode == "box":
        count_as_orbach1 = delta1 - half_width < energy < delta1 + half_width
        count_as_orbach2 = delta2 - half_width < energy < delta2 + half_width
        if count_as_orbach1:
            return coeff1
        elif count_as_orbach2:
            return coeff2
        else:
            return 0
    if mode == "gaussian":
        gaussian1 = gaussian(energy, coeff1, delta1, half_width, 0)
        gaussian2 = gaussian(energy, coeff2, delta2, half_width, 0)
        return gaussian1 + gaussian2
    if mode == "comb":
        gaussian1 = gaussian(energy, coeff1, delta1, half_width, 0)
        gaussian2 = gaussian(energy, coeff2, delta2, half_width, 0)
        gaussian3 = gaussian(energy, coeff2 / 3, 100, half_width, 0)
        return gaussian1 + gaussian2 + gaussian3


def spectral_function_sub_single(energy, coeff1, delta1):

    # mode = "box"
    mode = "gaussian"
    # mode = "comb"

    half_width = 10
    # half_width = 1

    # if mode == "box":
    #     count_as_orbach1 = delta1 - half_width < energy < delta1 + half_width
    #     count_as_orbach2 = delta2 - half_width < energy < delta2 + half_width
    #     if count_as_orbach1:
    #         return coeff1
    #     elif count_as_orbach2:
    #         return coeff2
    #     else:
    #         return 0
    if mode == "gaussian":
        return gaussian(energy, coeff1, delta1, half_width, 0)
    # if mode == "comb":
    #     gaussian1 = gaussian(energy, coeff1, delta1, half_width, 0)
    #     gaussian2 = gaussian(energy, coeff2, delta2, half_width, 0)
    #     gaussian3 = gaussian(energy, coeff2 / 3, 100, half_width, 0)
    #     return gaussian1 + gaussian2 + gaussian3


def relaxation_rate(temp, spectral_lambda):

    integrand = (
        lambda energy: spectral_lambda(energy)
        * bose(energy, temp)
        * (bose(energy, temp) + 1)
    )
    min_energy = 0
    max_energy = 200
    num_points = 1000
    energy_linspace = np.linspace(min_energy, max_energy, num_points)
    energy_linspace = energy_linspace[1:]
    width = (max_energy - min_energy) / (num_points - 1)
    integrand_vals = [width * integrand(val) for val in energy_linspace]
    integral = np.sum(integrand_vals)

    return (8 * pi**2) * integral


def main():
    """Generate a spectral function simulacrum and fit a double orbach to it"""

    delta1 = 68
    delta2 = 150

    ### Spectral function

    # spectral_lambda = lambda energy: spectral_function(energy, 0.2, delta1, 0.2, delta2)
    spectral_lambda = lambda energy: spectral_function_single(energy, 0.2, delta2)

    fig, ax = plt.subplots()
    spectral_linspace = np.linspace(0, 200, 1000)
    kpl.plot_line(ax, spectral_linspace, spectral_lambda(spectral_linspace))
    ax.set_xlabel("Energy (meV)")
    ax.set_ylabel("Spectral function (MHz / meV)")

    ### Relaxation rates calculation

    temp_linspace = np.linspace(125, 500, 50)
    ab_initio_rates = [relaxation_rate(temp, spectral_lambda) for temp in temp_linspace]

    ### Fit double orbach

    # fit_func = double_orbach
    # p0 = (1000, delta1, 1000, delta2)

    fit_func = single_orbach
    p0 = (1000, delta2)

    # fixed version
    # fit_func = lambda temp, coeff1, coeff2: double_orbach(
    #     temp, coeff1, delta1, coeff2, delta2
    # )
    # p0 = (1000, 1000)

    errs = [0.1 * rate for rate in ab_initio_rates]
    popt, pcov = curve_fit(fit_func, temp_linspace, ab_initio_rates, sigma=errs, p0=p0)
    double_orbach_rates = fit_func(temp_linspace, *popt)
    print(popt)

    ### Rates plot

    fig, ax = plt.subplots()
    kpl.plot_line(
        ax, temp_linspace, ab_initio_rates, label="Ab initio", color=kpl.KplColors.BLUE
    )
    kpl.plot_line(
        ax,
        temp_linspace,
        double_orbach_rates,
        label="Double Orbach",
        linestyle="dotted",
        color=kpl.KplColors.BLACK,
    )
    ax.legend()
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("Relaxation rate (1 / s)")
    ax.set_yscale("log")


if __name__ == "__main__":

    kpl.init_kplotlib()

    main()

    plt.show(block=True)
