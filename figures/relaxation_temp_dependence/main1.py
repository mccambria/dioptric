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
from colorutils import Color
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
from figures.relaxation_temp_dependence.temp_dependence_fitting import (
    omega_calc,
    gamma_calc,
)

ms = 7
lw = 1.75


# %% Functions


def process_raw_data(data, ref_range=None):
    """Pull the relaxation signal and ste out of the raw data."""

    num_runs = data["num_runs"]
    num_steps = data["num_steps"]
    sig_counts = np.array(data["sig_counts"])
    ref_counts = np.array(data["ref_counts"])
    time_range = np.array(data["relaxation_time_range"])

    # _, ax = plt.subplots()
    # counts_flatten = sig_counts[:, -3]
    # # counts_flatten = ref_counts.flatten()
    # bins = np.arange(0, max(counts_flatten) + 1, 1)
    # ax.hist(counts_flatten, bins, density=True)

    # Calculate time arrays in ms
    min_time, max_time = time_range / 10 ** 6
    times = np.linspace(min_time, max_time, num=num_steps)

    # Calculate the average signal counts over the runs, and ste
    start_run = None
    stop_run = None
    avg_sig_counts = np.average(sig_counts[start_run:stop_run, :], axis=0)
    std_sig_counts = np.std(sig_counts[start_run:stop_run, :], axis=0, ddof=1)
    # std_sig_counts = np.sqrt(avg_sig_counts)
    ste_sig_counts = std_sig_counts / np.sqrt(num_runs)
    # print(ste_sig_counts)

    # Assume reference is constant and can be approximated to one value
    avg_ref = np.average(ref_counts[start_run:stop_run, :])

    # Divide signal by reference to get normalized counts and st error
    norm_avg_sig = avg_sig_counts / avg_ref
    norm_avg_sig_ste = ste_sig_counts / avg_ref

    # Normalize to the reference range
    if ref_range is not None:
        diff = ref_range[1] - ref_range[0]
        norm_avg_sig = (norm_avg_sig - ref_range[0]) / diff
        norm_avg_sig_ste /= diff

    return norm_avg_sig, norm_avg_sig_ste, times


def relaxation_zero_func(t, gamma, omega):

    # Times are in ms, but rates are in s^-1
    gamma /= 1000
    omega /= 1000

    return (1 / 3) + (2 / 3) * np.exp(-3 * omega * t)


def relaxation_high_func(t, gamma, omega):

    # Times are in ms, but rates are in s^-1
    gamma /= 1000
    omega /= 1000

    first_term = 1 / 3
    second_term = (1 / 2) * np.exp(-(2 * gamma + omega) * t)
    third_term = (-1 / 2) * (-1 / 3) * np.exp(-3 * omega * t)
    return first_term + second_term + third_term


def get_ref_range(rabi_file):

    # Take the low reference range to be the value after 1 perfect pi pulse as calculated from the fit.
    # Assume the pi pulses are nice enough for the high reference range to be 1 with no infidelity.
    data = tool_belt.get_raw_data(rabi_file)
    norm_avg_sig = data["norm_avg_sig"]
    uwave_time_range = data["uwave_time_range"]
    num_steps = data["num_steps"]
    fit_func, popt = rabi.fit_data(uwave_time_range, num_steps, norm_avg_sig)
    rabi_period = 1 / popt[1]
    pi_pulse = rabi_period / 2
    ref_range = [fit_func(pi_pulse, *popt), 1.0]
    # print(ref_range)
    return ref_range


def exp_eq(t, rate, amp):
    return amp * np.exp(-rate * t)


def exp_eq_offset(t, rate, amp, offset):
    return amp * np.exp(-rate * t) + offset


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
    fig = plt.figure(figsize=(6.5, 7.5))
    grid_columns = 30
    half_grid_columns = grid_columns // 2
    gs = gridspec.GridSpec(2, grid_columns, height_ratios=(1, 1))

    first_row_sep_ind = 14

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

    # %% Relaxation out of plots

    temps = [el["temp"] for el in data_sets]

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
        colors_cmap = [set1[6], set1[0], set2[5], set1[2], set1[1]]

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

    ax.set_xlabel(r"Wait time $\tau$ (ms)")
    # ax.set_ylabel("Normalized fluorescence")
    ax.set_ylabel(r"$\ket{-1}$ population")

    min_time = 0.0
    max_time = 15.0
    times = [min_time, max_time]
    xtick_step = 5
    ax.set_xticks(np.arange(min_time, max_time + xtick_step, xtick_step))

    # Plot decay curves
    for ind in range(len(data_sets)):

        data_set = data_sets[ind]
        color = colors_hex[ind]
        facecolor = facecolors_hex[ind]
        temp = data_set["temp"]
        if not data_set["skip"]:
            raw_decay = tool_belt.get_raw_data(data_set["decay_file"])
            ref_range = get_ref_range(data_set["rabi_file"])
            # print(ref_range)
            # MCC remove this after single NV data
            ref_range = [0.68, 1.0]
            # ref_range = [0.68, 0.90]
            # ref_range = None

            signal_decay, ste_decay, times_decay = process_raw_data(
                raw_decay, ref_range
            )
            # Clip anything beyond 15 ms
            try:
                times_clip = np.where(times_decay > max_time)[0][0]
            except:
                times_clip = None
            times_decay = times_decay[:times_clip]
            signal_decay = signal_decay[:times_clip]
        else:
            times_decay = [0]
            signal_decay = [1.0]
            ste_decay = [0]
        # ax.scatter(
        #     times_decay,
        #     signal_decay,
        #     label="{} K".format(temp),
        #     zorder=5,
        #     marker="o",
        #     color=color,
        #     facecolor=facecolor,
        #     s=ms ** 2,
        # )
        ax.errorbar(
            times_decay,
            signal_decay,
            yerr=ste_decay,
            label="{} K".format(temp),
            zorder=5,
            marker="o",
            color=color,
            markerfacecolor=facecolor,
            ms=ms,
            linestyle="",
        )

        smooth_t = np.linspace(times[0], times[-1], 1000)
        gamma = data_set["gamma"]
        Omega = data_set["Omega"]
        if (gamma is None) and (Omega is None):
            # MCC make sure these values are up to date
            gamma = gamma_calc(temp)
            Omega = omega_calc(temp)
        print(temp, gamma, Omega)
        # fit_decay = relaxation_high_func(smooth_t, gamma, Omega)
        fit_decay = relaxation_zero_func(smooth_t, gamma, Omega)
        ax.plot(smooth_t, fit_decay, color=color, linewidth=lw)

    ax.legend()
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1])
    fig.text(
        -0.19, 0.95, "(b)", transform=ax.transAxes, color="black", fontsize=18
    )

    # %% Experimental layout

    # Add a new axes, make it invisible, steal its rect
    ax = fig.add_subplot(gs[1, :])
    ax.set_axis_off()
    fig.text(
        0,
        0.95,
        "(c)",
        transform=ax.transAxes,
        color="black",
        fontsize=18,
    )

    if draft_version:
        ax.set_axis_off()
        fig.add_axes(ax)
        level_structure_file = (
            nvdata_dir
            / "paper_materials/relaxation_temp_dependence/figures/experimental_layout_simplified.png"
        )
        img = mpimg.imread(level_structure_file)
        _ = ax.imshow(img)

    # %% Wrap up

    shift = 0.103
    gs.tight_layout(fig, pad=0.3, w_pad=-2.50)
    # gs.tight_layout(fig, pad=0.4, h_pad=0.5, w_pad=0.5, rect=[0, 0, 1, 1])
    # fig.tight_layout(pad=0.5)
    # fig.tight_layout()
    # plt.margins(0, 0)
    # fig.subplots_adjust(hspace=0.5, wspace=0.5)

    if dosave:
        file_path = str(
            nvdata_dir
            / "paper_materials/relaxation_temp_dependence/figures/main1.eps"
        )
        fig.savefig(file_path, dpi=500)


# %% Run


if __name__ == "__main__":

    tool_belt.init_matplotlib()
    # plt.rcParams.update({'font.size': 18})  # Increase font size
    matplotlib.rcParams["axes.linewidth"] = 1.0

    # -1 decay curves
    decay_data_sets = [
        {
            "temp": 400,
            "skip": True,
            "decay_file": None,
            "rabi_file": None,
            "Omega": None,
            "gamma": None,
        },
        {
            "temp": 350,
            "skip": True,
            "decay_file": None,
            "unity_ref_file": None,
            "zero_ref_file": None,
            "Omega": None,
            "gamma": None,
        },
        {
            "temp": 300,
            "skip": True,
            "decay_file": None,
            "rabi_file": None,
            "Omega": None,
            "gamma": None,
            # "Omega": 59.87,
            # "gamma": 131.57,
        },
        {
            "temp": 250,
            "skip": True,
            "decay_file": None,
            "rabi_file": None,
            "Omega": None,
            "gamma": None,
            # "Omega": 28.53,
            # "gamma": 71.51,
        },
        {
            "temp": 200,
            "skip": False,
            # 1e5 polarization
            # "decay_file": "2022_01_21-23_25_57-wu-nv6_2021_12_25",
            # "rabi_file": "2022_01_21-16_46_16-wu-nv6_2021_12_25",
            # 1e6 polarization
            # "decay_file": "2022_01_23-06_45_24-wu-nv6_2021_12_25",
            # "rabi_file": "2022_01_22-19_23_40-wu-nv6_2021_12_25",
            # -1,-1 off resonance
            # "decay_file": "2022_01_24-11_55_03-wu-nv6_2021_12_25",
            # "rabi_file": "2022_01_22-19_23_40-wu-nv6_2021_12_25",
            # 0,0
            # "decay_file": "2022_01_24-16_25_23-wu-nv6_2021_12_25",
            # "rabi_file": "2022_01_22-19_23_40-wu-nv6_2021_12_25",
            # 1e5 polarization, start at 200 us
            "decay_file": "2022_01_25-06_44_38-wu-nv6_2021_12_25",
            "rabi_file": "2022_01_22-19_23_40-wu-nv6_2021_12_25",
            "Omega": None,
            "gamma": None,
            # "Omega": 11,
            # "gamma": 40,
        },
    ]

    main(decay_data_sets, dosave=False, draft_version=True)

    plt.show(block=True)
