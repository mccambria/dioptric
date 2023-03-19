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


def main():

    # plot_mode = "dos"
    # plot_mode = "spectral"
    plot_mode = "mean_coupling"

    # Get the spectral function data
    nvdata_dir = common.get_nvdata_dir()
    from_nvdata = "paper_materials/relaxation_temp_dependence/2023_02_06-spectral.csv"
    spectral_file_path = nvdata_dir / from_nvdata
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

    energy_linspace = np.linspace(0, 200, 1000)
    sigma = 7.5
    # sigma = np.sqrt(sigma)
    smearing = lambda x, mu: (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(
        -(1 / 2) * ((x - mu) / sigma) ** 2
    )

    # Calculate density of states, spectral functions, and mean couplings
    density_of_states = []
    spectral_functions = [[], [], []]
    mean_couplings = [[], [], []]
    for energy in energy_linspace:
        dos = 0
        sf = [0] * 3
        for mode in modes:
            smeared_mode = smearing(energy, mode["Energy (meV)"])
            dos += smeared_mode
            sf[0] += np.abs(mode["V(2)00 (MHz)"]) * smeared_mode
            sf[1] += np.abs(mode["V(2)+0 (MHz)"]) * smeared_mode
            sf[2] += np.abs(mode["V(2)+- (MHz)"]) * smeared_mode
        density_of_states.append(dos)
        for ind in range(3):
            spectral_functions[ind].append(sf[ind])
            mean_couplings[ind].append(sf[ind] / (max(dos, 1.0)))
            # mean_couplings[ind].append(sf[ind] / dos)

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
        for ind in range(3):
            kpl.plot_line(ax, energy_linspace, mean_couplings[ind], label=labels[ind])
        ax.set_ylabel("Mean coupling (MHz)")
        ax.legend()
    ax.set_xlabel("Energy (meV)")


if __name__ == "__main__":

    kpl.init_kplotlib(latex=True)

    main()

    plt.show(block=True)
