# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 16:00:15 2019

@author: matth
"""


# %% Imports


import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import utils.tool_belt as tool_belt
import majorroutines.rabi as rabi
import utils.common as common
import json
from mpl_toolkits.axes_grid1.anchored_artists import (
    AnchoredSizeBar as scale_bar,
)
from scipy.optimize import curve_fit
from colorutils import Color
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
from figures.relaxation_temp_dependence.temp_dependence_fitting import (
    omega_calc,
    gamma_calc,
)
from analysis import relaxation_rate_analysis

ms = 7
lw = 1.75


# %% Functions


def zero_to_one_threshold(val):
    if val < 0:
        return 0
    elif val > 1:
        return 1
    else:
        return val


# %% Main


def main(data_sets, dosave=False, draft_version=True):

    nvdata_dir = common.get_nvdata_dir()

    # fig, axes_pack = plt.subplots(1,2, figsize=(10,5))
    # fig = plt.figure(figsize=(6.5, 7.5))
    fig = plt.figure(figsize=(4.5, 5.0))
    grid_columns = 30
    half_grid_columns = grid_columns // 2
    gs = gridspec.GridSpec(2, grid_columns, height_ratios=(1.1, 1))

    first_row_sep_ind = 15

    # %% Level structure

    # Add a new axes, make it invisible, steal its rect
    ax = fig.add_subplot(gs[0, 0:first_row_sep_ind])
    ax.set_axis_off()
    ax.text(
        0,
        0.95,
        "(a)",
        transform=ax.transAxes,
        color="black",
        fontsize=18,
    )

    draft_version = True
    # draft_version = False
    if draft_version:
        ax = plt.Axes(fig, [-0.05, 0.5, 0.5, 0.43])
        ax.set_axis_off()
        fig.add_axes(ax)
        level_structure_file = (
            nvdata_dir
            / "paper_materials/relaxation_temp_dependence/figures/level_structure.png"
        )
        img = mpimg.imread(level_structure_file)
        _ = ax.imshow(img)

    # %% Gamma subtraction curve plots

    temps = [round(el["temp"]) for el in data_sets]
    len_data_sets = len(data_sets)

    continuous_colormap = False
    if continuous_colormap:
        min_temp = min(temps)
        max_temp = max(temps)
        temp_range = max_temp - min_temp
        normalized_temps = [(val - min_temp) / temp_range for val in temps]
        # adjusted_temps = [normalized_temps[0], ]
        cmap = matplotlib.cm.get_cmap("coolwarm")
        colors_cmap = [cmap(val) for val in normalized_temps]
    else:
        set1 = matplotlib.cm.get_cmap("Set1").colors
        set2 = matplotlib.cm.get_cmap("Dark2").colors
        if len_data_sets == 5:
            colors_cmap = [set1[6], set1[0], set2[5], set1[2], set1[1]]
        elif len_data_sets:
            colors_cmap = [set1[0], set2[5], set1[2], set1[1]]
        elif len_data_sets == 3:
            colors_cmap = [set1[0], set2[5], set1[1]]

    # Trim the alpha value and convert from 0:1 to 0:255
    colors_rgb = [[255 * val for val in el[0:3]] for el in colors_cmap]
    colors_Color = [Color(tuple(el)) for el in colors_rgb]
    colors_hex = [val.hex for val in colors_Color]
    colors_hsv = [val.hsv for val in colors_Color]
    facecolors_hsv = [(el[0], 0.3 * el[1], 1.2 * el[2]) for el in colors_hsv]
    # Threshold to make sure these are valid colors
    facecolors_hsv = [
        (el[0], zero_to_one_threshold(el[1]), zero_to_one_threshold(el[2]))
        for el in facecolors_hsv
    ]
    facecolors_Color = [Color(hsv=el) for el in facecolors_hsv]
    facecolors_hex = [val.hex for val in facecolors_Color]

    ax = fig.add_subplot(gs[0, first_row_sep_ind:])
    # ax = axes_pack[1]
    l, b, w, h = ax.get_position().bounds
    shift = 0.02
    ax.set_position([l + shift, b, w - shift, h])

    ax.set_xlabel(r"Wait time $\tau$ (ms)")
    # ax.set_ylabel(r"$P_{+1,+1}(\tau) - P_{+1,-1}(\tau)$")
    ax.set_ylabel(r"$\mathrm{\ket{-1}}$, $\mathrm{\ket{+1}}$ population difference")

    min_time = 0.0
    max_time = 18
    # max_time = 15.5
    # max_time = 11.5
    # max_time = 12.5
    # max_time = 9
    xtick_step = 5
    # xtick_step = 4
    # xtick_step = 2
    times = [min_time, max_time]
    ax.set_xticks(np.arange(min_time, max_time + xtick_step, xtick_step))

    # Plot decay curves
    for ind in range(len(data_sets)):

        data_set = data_sets[ind]
        color = colors_hex[ind]
        facecolor = facecolors_hex[ind]
        temp = round(data_set["temp"])
        gamma = data_set["gamma"]
        Omega = data_set["Omega"]

        # Plot the fit/predicted curves
        if (gamma is None) and (Omega is None):
            # MCC make sure these values are up to date
            gamma = gamma_calc(temp)
            Omega = omega_calc(temp)
            smooth_t = np.linspace(times[0], 1.1 * times[-1], 1000)
            fit_decay = np.exp(-(1 / 1000) * (2 * gamma + Omega) * smooth_t)
            ax.plot(smooth_t, fit_decay, color=color, linewidth=lw)

        if data_set["skip"]:
            continue

        path = data_set["path"]
        folder = data_set["folder"]
        data_decay, ste_decay, times_decay = relaxation_rate_analysis.main(
            path, folder, return_gamma_data=True
        )

        # Clip anything beyond the max time
        try:
            times_clip = np.where(times_decay > max_time)[0][0]
        except:
            times_clip = None
        times_decay = times_decay[:times_clip]
        data_decay = data_decay[:times_clip]
        ste_decay = ste_decay[:times_clip]

        plot_errors = False
        if plot_errors:
            ax.errorbar(
                times_decay,
                data_decay,
                yerr=np.array(ste_decay),
                label="{} K".format(temp),
                zorder=5,
                marker="o",
                color=color,
                markerfacecolor=facecolor,
                ms=ms,
                linestyle="",
            )
        else:
            ax.scatter(
                times_decay,
                data_decay,
                label="{} K".format(temp),
                zorder=5,
                marker="o",
                color=color,
                facecolor=facecolor,
                s=ms ** 2,
            )

    ax.legend(handlelength=5)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1])
    fig.text(
        -0.21, 0.95, "(b)", transform=ax.transAxes, color="black", fontsize=18
    )
    x_buffer = 0.02 * max_time
    ax.set_xlim([-x_buffer, max_time + x_buffer])
    ax.set_ylim([-0.05, 1.05])
    # ax.set_ylim([0.05, 1.1])
    # ax.set_yscale("log")

    # %% Experimental layout

    # Add a new axes, make it invisible, steal its rect
    ax = fig.add_subplot(gs[1, :])
    ax.set_axis_off()
    fig.text(
        # 0,
        -0.003,
        0.95,
        "(c)",
        transform=ax.transAxes,
        color="black",
        fontsize=18,
    )

    if draft_version:
        ax.set_axis_off()
        fig.add_axes(ax)
        layout_file = "experimental_layout_simplified.png"
        level_structure_file = (
            nvdata_dir
            / "paper_materials/relaxation_temp_dependence/figures/{}".format(
                layout_file
            )
        )
        img = mpimg.imread(level_structure_file)
        _ = ax.imshow(img)

    # %% Wrap up

    shift = 0.103
    gs.tight_layout(fig, pad=0.3, w_pad=-2.50)
    # gs.tight_layout(fig, pad=0.3, w_pad=0)
    # gs.tight_layout(fig, pad=0.4, h_pad=0.5, w_pad=0.5, rect=[0, 0, 1, 1])
    # fig.tight_layout(pad=0.5)
    # fig.tight_layout()
    # plt.margins(0, 0)
    # fig.subplots_adjust(hspace=0.5, wspace=0.5)

    if dosave:
        ext = "png"
        file_path = str(
            nvdata_dir
            / "paper_materials/relaxation_temp_dependence/figures/main1.{}".format(
                ext
            )
        )
        fig.savefig(file_path, dpi=500)


# %% Run


if __name__ == "__main__":

    tool_belt.init_matplotlib()
    # plt.rcParams.update({'font.size': 18})  # Increase font size
    matplotlib.rcParams["axes.linewidth"] = 1.5

    decay_data_sets = [
        # {
        #     "temp": 415.555,
        #     "skip": False,
        #     "path": "pc_hahn/branch_time-tagger-speedup/t1_interleave_knill/data_collections/",
        #     "folder": "hopper-search-425K",
        #     "Omega": None,
        #     "gamma": None,
        # },
        {
            "temp": 401.590,
            "skip": False,
            "path": "pc_hahn/branch_time-tagger-speedup/t1_interleave_knill/data_collections/",
            "folder": "hopper-search-412.5K",
            "Omega": None,
            "gamma": None,
        },
        # {
        #     "temp": 380.168,
        #     "skip": False,
        #     "path": "pc_hahn/branch_master/t1_interleave_knill/data_collections/",
        #     "folder": "hopper-search-400K",
        #     "Omega": None,
        #     "gamma": None,
        # },
        # {
        #     "temp": 337.584,
        #     "skip": False,
        #     "path": "pc_hahn/branch_time-tagger-speedup/t1_interleave_knill/data_collections/",
        #     "folder": "hopper-search-350K",
        #     "Omega": None,
        #     "gamma": None,
        # },
        {
            "temp": 295,
            "skip": False,
            "path": "pc_hahn/branch_cryo-setup/t1_interleave_knill/data_collections/",
            "folder": "hopper-nv1_2021_03_16-300K",
            "Omega": None,
            "gamma": None,
        },
        # {
        #     "temp": 250,
        #     "skip": False,
        #     "path": "pc_hahn/branch_cryo-setup/t1_interleave_knill/data_collections/",
        #     "folder": "hopper-nv1_2021_03_16-250K",
        #     "Omega": None,
        #     "gamma": None,
        # },
        # {
        #     "temp": 262.5,
        #     "skip": False,
        #     "path": "pc_hahn/branch_cryo-setup/t1_interleave_knill/data_collections/",
        #     "folder": "hopper-nv1_2021_03_16-262.5K",
        #     "Omega": None,
        #     "gamma": None,
        # },
        {
            "temp": 234,  # 237.5 nominal
            "skip": False,
            "path": "pc_hahn/branch_cryo-setup/t1_interleave_knill/data_collections/",
            "folder": "hopper-nv1_2021_03_16-237.5K",
            "Omega": None,
            "gamma": None,
        },
        # {
        #     "temp": 225,
        #     "skip": False,
        #     "path": "pc_hahn/branch_cryo-setup/t1_interleave_knill/data_collections/",
        #     "folder": "hopper-nv1_2021_03_16-225K",
        #     "Omega": None,
        #     "gamma": None,
        # },
        # {
        #     "temp": 200,
        #     "skip": False,
        #     "path": "pc_hahn/branch_cryo-setup/t1_interleave_knill/data_collections/",
        #     "folder": "hopper-nv1_2021_03_16-200K-gamma_minus_1",
        #     "Omega": None,
        #     "gamma": None,
        # },
        {
            "temp": 185,  # 187.5 nominal
            "skip": False,
            "path": (
                "pc_hahn/branch_master/t1_interleave_knill/data_collections/"
            ),
            "folder": "hopper-search-187.5K",
            "Omega": None,
            "gamma": None,
        },
    ]

    dosave = True
    # dosave = False
    main(decay_data_sets, dosave=dosave, draft_version=True)

    plt.show(block=True)
