import errno
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import csv
import matplotlib.patches as patches
import matplotlib.lines as mlines
from scipy.optimize import curve_fit
import pandas as pd
import utils.tool_belt as tool_belt
from utils.kplotlib import KplColors, color_mpl_to_color_hex, lighten_color_hex
import utils.common as common
from scipy.odr import ODR, Model, RealData
import sys
from pathlib import Path
import math
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.gridspec import GridSpec
import temp_dependence_fitting
from temp_dependence_fitting import (
    gamma_face_color,
    gamma_edge_color,
    omega_face_color,
    omega_edge_color,
    ratio_face_color,
    ratio_edge_color,
)
import csv
import utils.kplotlib as kpl
from utils.kplotlib import (
    marker_size,
    line_width,
    marker_size_inset,
    line_width_inset,
)

marker_edge_width = line_width
marker_edge_width_inset = line_width_inset


def main():

    ### Params

    # rates_y_range = [1e-2, 1000]
    # rates_yscale = "log"
    # ratio_y_range = [0, 1]
    # ratio_yscale = "linear"
    # temp_range = [125, 480]
    # xscale = "linear"

    rates_y_range = [0.5e-2, 800]
    rates_yscale = "log"
    ratio_y_range = [0, 0.78]
    ratio_yscale = "linear"
    temp_range = [-5, 480]
    xscale = "linear"

    ### Setup

    home = common.get_nvdata_dir()
    path = home / "paper_materials/relaxation_temp_dependence"

    # Fit to Omega and gamma simultaneously
    data_file_name = "compiled_data"
    data_points = temp_dependence_fitting.get_data_points(
        path, data_file_name, temp_range
    )
    (
        popt,
        pvar,
        beta_desc,
        omega_hopper_fit_func,
        omega_wu_fit_func,
        gamma_hopper_fit_func,
        gamma_wu_fit_func,
    ) = temp_dependence_fitting.fit_simultaneous(data_points, "double_orbach")
    omega_hopper_lambda = lambda temp: omega_hopper_fit_func(temp, popt)
    omega_wu_lambda = lambda temp: omega_wu_fit_func(temp, popt)
    gamma_hopper_lambda = lambda temp: gamma_hopper_fit_func(temp, popt)
    gamma_wu_lambda = lambda temp: gamma_wu_fit_func(temp, popt)

    sim_file_name = "Tdep_512_PBE.dat"

    min_temp = temp_range[0]
    max_temp = temp_range[1]
    linspace_min_temp = max(0, min_temp)
    temp_linspace = np.linspace(linspace_min_temp, max_temp, 1000)

    ### Get temps/rates from simulation dat file

    sim_data = np.genfromtxt(
        path / sim_file_name,
        skip_header=1,
        skip_footer=1,
        names=True,
        dtype=None,
        delimiter=" ",
    )

    sim_temps = []
    sim_omega = []
    sim_gamma = []
    for el in sim_data:
        sim_temps.append(el[0])
        sim_omega.append(el[1])
        sim_gamma.append(el[2])
    sim_temps = np.array(sim_temps)
    sim_omega = np.array(sim_omega)
    sim_gamma = np.array(sim_gamma)

    ### Figure prep

    fig, axes_pack = plt.subplots(1, 2, figsize=kpl.double_figsize)
    ax_rates, ax_ratio = axes_pack

    for ax in [ax_rates, ax_ratio]:
        ax.set_xlabel(r"Temperature $\mathit{T}$ (K)")
        ax.set_xscale(xscale)
        ax.set_xlim(min_temp, max_temp)
        ax.axvline(x=125, color="silver", zorder=-10, lw=line_width)

    ax_rates.set_yscale(rates_yscale)
    ax_rates.set_ylim(rates_y_range[0], rates_y_range[1])
    ax_rates.set_ylabel(r"Relaxation rates (s$^{-1}$)")

    ax_ratio.set_yscale(ratio_yscale)
    ax_ratio.set_ylim(ratio_y_range[0], ratio_y_range[1])
    ax_ratio.set_ylabel(r"$\textit{Ab initio}$ rates / model rates")

    ### Plot

    # gamma model
    ax_rates.plot(
        temp_linspace,
        gamma_hopper_lambda(temp_linspace),
        label=r"$\mathrm{\gamma}$ model",
        color=gamma_edge_color,
        linewidth=line_width,
        linestyle="dashed",
    )
    # gamma ab initio
    ax_rates.plot(
        sim_temps,
        sim_gamma,
        label=r"$\mathrm{\gamma}$ $\textit{ab initio}$",
        color=gamma_edge_color,
        linewidth=line_width,
        linestyle="dotted",
    )
    # Omega model
    ax_rates.plot(
        temp_linspace,
        omega_hopper_lambda(temp_linspace),
        label=r"$\mathrm{\Omega}$ model",
        color=omega_edge_color,
        linewidth=line_width,
        linestyle="dashed",
    )
    # Omega ab initio
    ax_rates.plot(
        sim_temps,
        sim_omega,
        label=r"$\mathrm{\Omega}$ $\textit{ab initio}$",
        color=omega_edge_color,
        linewidth=line_width,
        linestyle="dotted",
    )

    # Ratio
    ax_ratio.plot(
        sim_temps,
        sim_gamma / gamma_hopper_lambda(sim_temps),
        label=r"$\mathrm{\gamma}$",
        color=gamma_edge_color,
        linewidth=line_width,
    )
    ax_ratio.plot(
        sim_temps,
        sim_omega / omega_hopper_lambda(sim_temps),
        label=r"$\mathrm{\Omega}$",
        color=omega_edge_color,
        linewidth=line_width,
    )

    ### Wrap up

    fig.text(
        -0.16,
        0.96,
        "(a)",
        transform=ax_rates.transAxes,
        color="black",
        fontsize=18,
    )
    fig.text(
        -0.138,
        0.96,
        "(b)",
        transform=ax_ratio.transAxes,
        color="black",
        fontsize=18,
    )

    ax_rates.legend()
    ax_ratio.legend()
    fig.tight_layout(pad=0.3)
    fig.subplots_adjust(wspace=0.16)


if __name__ == "__main__":

    kpl.init_kplotlib()
    main()
    plt.show(block=True)
