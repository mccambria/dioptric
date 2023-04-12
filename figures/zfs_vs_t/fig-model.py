# -*- coding: utf-8 -*-
"""
Model decsription figure for zfs vs t paper

Created on March 28th, 2023

@author: mccambria
"""


# region Import and constants

import numpy as np
from utils import common
from majorroutines.pulsed_resonance import return_res_with_error
import majorroutines.pulsed_resonance as pesr
import utils.tool_belt as tool_belt
from utils.tool_belt import bose
import matplotlib.pyplot as plt
from utils import kplotlib as kpl
from utils.kplotlib import KplColors
from scipy.optimize import curve_fit
import csv
import pandas as pd
import sys
from analysis import three_level_rabi
from figures.zfs_vs_t.zfs_vs_t_main import get_data_points
from figures.zfs_vs_t.thermal_expansion import (
    fit_double_occupation,
    jacobson_lattice_constant,
)
from figures.zfs_vs_t.deconvolve_spectral_function import deconvolve


# endregion


def fig():
    temp_linspace = np.linspace(1, 1000, 1000)
    min_energy = -5
    max_energy = 175
    energy_linspace = np.linspace(0, max_energy, 1000)
    kpl_figsize = kpl.figsize
    adj_figsize = (kpl_figsize[0], 1.2 * kpl_figsize[1])
    fig, axes_pack = plt.subplots(2, 1, figsize=adj_figsize)
    ax1, ax2 = axes_pack
    ax3 = ax2.twinx()

    double_occupation_lambda = fit_double_occupation()

    # First order effects (lattice constant)
    kpl.plot_line(
        ax1,
        temp_linspace,
        double_occupation_lambda(temp_linspace),
        label="This work",
        color=kpl.KplColors.RED,
    )
    ax1.set_ylabel(r"Lattice constant ($\si{\angstrom}$)")
    ax1.set_xlabel("Temperature (K)")

    # Second order effects
    sigma = np.sqrt(7.5)
    density_of_states, spectral_functions, mean_couplings = deconvolve(
        energy_linspace, sigma
    )
    # labels = [r"$\mathit{S_{z}}$", r"$\mathit{S_{+}}$", r"$\mathit{S_{+}^{2}}$"]
    color = KplColors.ORANGE
    plot_vals = np.array(spectral_functions[0])
    kpl.plot_line(ax2, energy_linspace, plot_vals, color=color)
    ax2.set_ylabel("Spectral function \n(MHz / meV)", color=color)
    ax2.tick_params(axis="y", color=color, labelcolor=color)
    # ax2.spines["left"].set_color(color)
    ax3.spines["left"].set_color(color)  # ax3 vs 2 because 3 is written on top of 2
    ax2.set_xlabel("Energy $\hbar\omega$ (meV)")
    ax2.set_xlim(min_energy, max_energy)

    # DOS
    color = KplColors.GREEN
    plot_vals = density_of_states
    kpl.plot_line(ax3, energy_linspace, plot_vals, color=color)
    ax3.set_ylabel("DOS (1 / meV)", color=color)
    ax3.tick_params(axis="y", color=color, labelcolor=color)
    ax3.spines["right"].set_color(color)

    ### Wrap up

    fig.text(0.07, 0.965, "(a)")
    fig.text(0.07, 0.465, "(b)")


def fig_three_panel():
    temp_linspace = np.linspace(1, 1000, 1000)
    min_energy = -5
    max_energy = 175
    energy_linspace = np.linspace(0, max_energy, 1000)
    kpl_figsize = kpl.figsize
    adj_figsize = (kpl_figsize[0], 1.8 * kpl_figsize[1])
    fig, axes_pack = plt.subplots(3, 1, figsize=adj_figsize)
    ax1, ax2, ax3 = axes_pack

    double_occupation_lambda = fit_double_occupation()

    # First order effects (lattice constant)
    kpl.plot_line(
        ax1,
        temp_linspace,
        double_occupation_lambda(temp_linspace),
        label="This work",
        color=kpl.KplColors.RED,
    )
    ax1.set_ylabel(r"Lattice constant ($\si{\angstrom}$)")
    ax1.set_xlabel("Temperature (K)")

    # Second order effects
    sigma = np.sqrt(7.5)
    density_of_states, spectral_functions, mean_couplings = deconvolve(
        energy_linspace, sigma
    )
    # labels = [r"$\mathit{S_{z}}$", r"$\mathit{S_{+}}$", r"$\mathit{S_{+}^{2}}$"]
    plot_vals = np.array(spectral_functions[0])
    kpl.plot_line(ax2, energy_linspace, plot_vals)
    ax2.set_ylabel("Spectral function \n(MHz / meV)")
    ax2.set_xlabel("Energy $\hbar\omega$ (meV)")
    ax2.set_xlim(min_energy, max_energy)

    # DOS
    plot_vals = density_of_states
    kpl.plot_line(ax3, energy_linspace, plot_vals)
    ax3.set_ylabel("DOS (1 / meV)")
    ax3.set_xlabel("Energy $\hbar\omega$ (meV)")
    ax3.set_xlim(min_energy, max_energy)

    ### Wrap up

    fig.text(0.07, 0.965, "(a)")
    fig.text(0.07, 0.465, "(b)")


if __name__ == "__main__":
    kpl.init_kplotlib(latex=True)

    fig()
    # fig_three_panel()

    plt.show(block=True)
