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
from scipy.integrate import quad
import re
import pandas as pd


def get_data():
    # Get the spectral function data
    nvdata_dir = common.get_nvdata_dir()

    # Not consistent with spin phonon relaxation paper
    # from_nvdata = "paper_materials/relaxation_temp_dependence/2023_02_06-spectral.csv"
    # spectral_file_path = nvdata_dir / from_nvdata

    # Consistent with spin phonon relaxation paper
    file_name = "2023_06_07-spectral"
    file_path = nvdata_dir / "paper_materials/relaxation_temp_dependence"
    xl_file_path = file_path / f"{file_name}.xlsx"
    spectral_file_path = file_path / f"{file_name}.csv"
    compiled_data_file = pd.read_excel(xl_file_path, engine="openpyxl")
    compiled_data_file.to_csv(spectral_file_path, index=None, header=True)

    modes = []
    with open(spectral_file_path, newline="") as f:
        reader = csv.reader(f)
        header = True
        for row in reader:
            # Create columns from the header (first row)
            if header:
                columns = row
                header = False
                continue
            point = {}
            for ind in range(len(columns)):
                column = columns[ind]
                raw_val = row[ind]
                if raw_val == "TRUE":
                    val = True
                else:
                    try:
                        val = eval(raw_val)
                    except Exception:
                        val = raw_val
                point[column] = val
            modes.append(point)

    return modes


def smearing(x, mu, sigma):
    return (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(
        -(1 / 2) * ((x - mu) / sigma) ** 2
    )


def deconvolve(energy_linspace, sigma):
    """Calculate density of states, spectral functions, and mean couplings"""

    modes = get_data()

    density_of_states = []
    spectral_functions = [[], [], []]
    mean_couplings = [[], [], []]
    for energy in energy_linspace:
        dos = 0
        sf = [0] * 3
        for mode in modes:
            smeared_mode = smearing(energy, mode["Energy (meV)"], sigma)
            dos += smeared_mode
            # sf[0] += np.abs(mode["V(2)00 (MHz)"]) * smeared_mode
            # sf[1] += np.abs(mode["V(2)+0 (MHz)"]) * smeared_mode
            # sf[2] += np.abs(mode["V(2)+- (MHz)"]) * smeared_mode
            sf[0] += 2 * mode["V(2)00 (MHz)"] * smeared_mode
            sf[1] += mode["V(2)+0 (MHz)"] * smeared_mode
            sf[2] += mode["V(2)+- (MHz)"] * smeared_mode
        density_of_states.append(dos)
        for ind in range(3):
            spectral_functions[ind].append(sf[ind])
            mean_couplings[ind].append(sf[ind] / dos)

    return density_of_states, spectral_functions, mean_couplings


def main():
    # plot_mode = "dos"
    # plot_mode = "spectral"
    plot_mode = "mean_coupling"

    energy_linspace = np.linspace(0, 200, 1000)
    sigma = 7.5
    sigma = np.sqrt(sigma)
    density_of_states, spectral_functions, mean_couplings = deconvolve(
        energy_linspace, sigma
    )

    # Plots
    labels = [r"$\mathit{S_{z}}$", r"$\mathit{S_{+}}$", r"$\mathit{S_{+}^{2}}$"]
    fig, ax = plt.subplots()
    if plot_mode == "dos":
        kpl.plot_line(ax, energy_linspace, density_of_states)
        ax.set_ylabel("Density of states (1 / meV)")
    elif plot_mode == "spectral":
        for ind in range(3):
            kpl.plot_line(
                ax, energy_linspace, spectral_functions[ind], label=labels[ind]
            )
        ax.set_ylabel("Spectral function (MHz / meV)")
        ax.legend()
    elif plot_mode == "mean_coupling":
        plot_inds = np.where(energy_linspace < 170)
        for ind in range(3):
            plot_couplings = np.array(mean_couplings[ind])[plot_inds]
            kpl.plot_line(
                ax, energy_linspace[plot_inds], plot_couplings, label=labels[ind]
            )
        ax.set_ylabel("Mean coupling (MHz)")
        ax.legend()
    ax.set_xlabel("Energy (meV)")


def fig():
    min_energy = -5
    max_energy = 175
    energy_linspace = np.linspace(0, max_energy, 1000)
    sigma = np.sqrt(5)
    density_of_states, spectral_functions, mean_couplings = deconvolve(
        energy_linspace, sigma
    )

    kpl_figsize = kpl.figsize
    adj_figsize = (kpl_figsize[0], 1.2 * kpl_figsize[1])
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=adj_figsize)

    labels = [r"$\mathit{S_{z}}$", r"$\mathit{S_{+}}$", r"$\mathit{S_{+}^{2}}$"]

    # DOS plot
    kpl.plot_line(ax1, energy_linspace, density_of_states)
    # ax1.set_ylabel("Density of states (1 / meV)")
    ax1.set_ylabel("DOS (meV$^{-1}$)")

    # Mean couplings plot
    # plot_inds = np.where(energy_linspace < 170)
    for ind in range(3):
        plot_couplings = np.array(mean_couplings[ind])
        kpl.plot_line(ax2, energy_linspace, plot_couplings, label=labels[ind])
    ax2.set_ylabel("Mean coupling (MHz)")
    ax2.legend()
    ax2.set_xlabel("Energy $\hbar\omega$ (meV)")
    ax2.set_xlim(min_energy, max_energy)

    fig.text(0, 0.965, "(a)")
    fig.text(0, 0.53, "(b)")


if __name__ == "__main__":
    # norm = quad(smearing, -np.inf, np.inf, args=(0, 5))
    # print(norm)

    nvdata_dir = common.get_nvdata_dir()
    from_nvdata = (
        "paper_materials/relaxation_temp_dependence/2023_06_07-512_atom-spin_phonon.dat"
    )
    spectral_file_path = nvdata_dir / from_nvdata

    total = ""
    with open(spectral_file_path) as f:
        line = f.readline()
        while line:
            total += re.sub("\s+", ",", line.strip())
            line = f.readline()
            total += "\n"
    print(total)

    # kpl.init_kplotlib(latex=True)

    # # main()
    # fig()

    # plt.show(block=True)
