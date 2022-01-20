# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 16:00:15 2019

@author: matth
"""


# %% Imports


import numpy
import matplotlib
import matplotlib.pyplot as plt
import utils.tool_belt as tool_belt
import utils.common as common
import json
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec

ms = 7
lw = 1.75


# %% Functions


def process_raw_data(data, ref_range):
    """Pull the relaxation signal and ste out of the raw data."""

    num_runs = data["num_runs"]
    num_steps = data["num_steps"]
    sig_counts = numpy.array(data["sig_counts"])
    ref_counts = numpy.array(data["ref_counts"])
    time_range = numpy.array(data["relaxation_time_range"])

    # Calculate time arrays in ms
    min_time, max_time = time_range / 10 ** 6
    times = numpy.linspace(min_time, max_time, num=num_steps)

    # Calculate the average signal counts over the runs, and ste
    avg_sig_counts = numpy.average(sig_counts[::], axis=0)
    ste_sig_counts = numpy.std(sig_counts[::], axis=0, ddof=1) / numpy.sqrt(
        num_runs
    )

    # Assume reference is constant and can be approximated to one value
    avg_ref = numpy.average(ref_counts[::])

    # Divide signal by reference to get normalized counts and st error
    norm_avg_sig = avg_sig_counts / avg_ref
    norm_avg_sig_ste = ste_sig_counts / avg_ref

    # Normalize to the reference range
    diff = ref_range[1] - ref_range[0]
    norm_avg_sig = (norm_avg_sig - ref_range[0]) / diff

    return norm_avg_sig, norm_avg_sig_ste, times


def relaxation_zero_func(t, gamma, omega, infid):

    return (1 / 3) + (2 / 3) * numpy.exp(-3 * omega * t)


def relaxation_high_func(t, gamma, omega, infid):

    # Times are in ms, but rates are in s^-1
    gamma /= 1000
    omega /= 1000

    first_term = (1 / 3) + (1 / 2) * ((1 - infid) ** 2) * numpy.exp(
        -(2 * gamma + omega) * t
    )
    second_term = (
        (-1 / 2) * (infid - (1 / 3)) * numpy.exp(-3 * omega * t) * (1 - infid)
    )
    third_term = (infid - (1 / 3)) * numpy.exp(-3 * omega * t) * infid
    return first_term + second_term + third_term


def get_ref_range(data):

    sig_counts = numpy.array(data["sig_counts"])
    ref_counts = numpy.array(data["ref_counts"])
    avg_ref = numpy.average(ref_counts[::])
    avg_sig_counts = numpy.average(sig_counts[::], axis=0)
    return [avg_sig_counts[0] / avg_ref, avg_sig_counts[-1] / avg_ref]


def exp_eq(t, rate, amp):
    return amp * numpy.exp(-rate * t)


def exp_eq_offset(t, rate, amp, offset):
    return amp * numpy.exp(-rate * t) + offset


def subtraction_plot(ax, analysis_file_path):
    """
    This is adapted from Aedan's function of the same name in
    analysis/Paper Figures\Magnetically Forbidden Rate/supplemental_figures.py
    """

    with open(analysis_file_path) as file:
        data = json.load(file)

        zero_relaxation_counts = data["zero_relaxation_counts"]
        zero_relaxation_ste = numpy.array(data["zero_relaxation_ste"])
        zero_zero_time = data["zero_zero_time"]

        plus_relaxation_counts = data["plus_relaxation_counts"]
        plus_relaxation_ste = numpy.array(data["plus_relaxation_ste"])
        plus_plus_time = data["plus_plus_time"]

        omega_opti_params = data["omega_opti_params"]
        gamma_opti_params = data["gamma_opti_params"]
        manual_offset_gamma = data["manual_offset_gamma"]

    zero_zero_time = numpy.array(zero_zero_time)
    try:
        times_15 = numpy.where(zero_zero_time > 15.0)[0][0]
    except:
        times_15 = None
    color = "#FF9933"
    facecolor = "#FFCC33"
    ax.scatter(
        zero_zero_time[:times_15],
        zero_relaxation_counts[:times_15],
        label=r"$F_{\Omega}$",
        zorder=5,
        marker="^",
        s=ms ** 2,
        color=color,
        facecolor=facecolor,
    )
    zero_time_linspace = numpy.linspace(0, 15.0, num=1000)
    ax.plot(
        zero_time_linspace,
        exp_eq(zero_time_linspace, *omega_opti_params),
        color=color,
        linewidth=lw,
    )

    omega_patch = mlines.Line2D(
        [],
        [],
        label=r"$F_{\Omega}$",
        linewidth=lw,
        marker="^",
        markersize=ms,
        color=color,
        markerfacecolor=facecolor,
    )

    plus_plus_time = numpy.array(plus_plus_time)
    try:
        times_15 = numpy.where(plus_plus_time > 15.0)[0][0]
    except:
        times_15 = None
    x_clip = numpy.array(plus_plus_time[:times_15])
    y_clip = numpy.array(plus_relaxation_counts[:times_15])
    # mask = numpy.array([el.is_integer() for el in x_clip])
    # ax.scatter(x_clip[mask], y_clip[mask])
    color = "#993399"
    facecolor = "#CC99CC"
    ax.scatter(
        x_clip,
        y_clip,
        label=r"$F_{\gamma}$",
        zorder=5,
        marker="o",
        s=ms ** 2,
        color=color,
        facecolor=facecolor,
    )
    plus_time_linspace = numpy.linspace(0, 15.0, num=1000)
    gamma_rate = gamma_opti_params[0]
    gamma_opti_params[0] = gamma_rate
    gamma_opti_params_offset = gamma_opti_params + [manual_offset_gamma]
    ax.plot(
        plus_time_linspace,
        exp_eq_offset(plus_time_linspace, *gamma_opti_params_offset),
        color=color,
        linewidth=lw,
    )

    # ax.tick_params(which = 'both', length=8, width=2, colors='k',
    #             direction='in',grid_alpha=0.7)
    ax.set_xlabel(r"Wait time $\tau$ (ms)")
    ax.set_ylabel("Subtraction curve (arb. units)")
    ax.set_xlim(-0.5, 15.5)
    ax.set_yscale("log")

    gamma_patch = mlines.Line2D(
        [],
        [],
        label=r"$F_{\gamma}$",
        linewidth=lw,
        marker="o",
        markersize=ms,
        color=color,
        markerfacecolor=facecolor,
    )
    # ax.legend(handles=[omega_patch, gamma_patch], handlelength=lw)
    ax.legend(handleheight=1.6, handlelength=0.6)

    trans = ax.transAxes
    # trans = ax.get_figure().transFigure  # 0.030, 0.46
    ax.text(-0.15, 1.05, "(c)", transform=trans, color="black", fontsize=18)


# %% Main


def main(data_sets):

    # fig, axes_pack = plt.subplots(1,2, figsize=(10,5))
    fig = plt.figure(figsize=(6.75, 7.5))
    gs = gridspec.GridSpec(2, 2, width_ratios=[2, 3])

    # %% Level structure

    # Add a new axes, make it invisible, steal its rect
    ax = fig.add_subplot(gs[0, 0])
    ax.set_axis_off()
    ax.text(
        -0.43, 1.05, "(a)", transform=ax.transAxes, color="black", fontsize=18
    )

    ax = plt.Axes(fig, [-0.03, 0.51, 0.5, 0.43])
    ax.set_axis_off()
    fig.add_axes(ax)
    # print(gs)
    # fig.add_axes(gs[0, 0])
    # ax = fig.add_subplot(gs[0, 0])

    # ax = axes_pack[0]
    # ax.set_axis_off()
    file = "/run/media/mccambria/Barracuda/acer2-2019-2020/lab/bulk_dq_relaxation/figures_revision2/main1/level_structure.png"
    img = mpimg.imread(file)
    img_plot = ax.imshow(img)

    # l, b, w, h = ax.get_position().bounds
    # ax.set_position([l, b -1.0, w, h])
    # ax.set_axis_off()
    # ax.axis('off')

    # %% Relaxation out of plots

    ax = fig.add_subplot(gs[0, 1])
    # ax = axes_pack[1]

    ax.set_xlabel(r"Wait time $\tau$ (ms)")
    ax.set_ylabel("Fluorescence (arb. units)")
    ax.set_xticks([0, 5, 10, 15])

    times = [0.0, 15.0]

    # Plot decay curves
    for ind in range(len(data_sets)):

        data_set = data_sets[ind]
        path_from_nvdata = data_set["path_from_nvdata"]
        raw_decay = tool_belt.get_raw_data(
            data_set["decay_file"], path_from_nvdata
        )
        # ref_range = get_ref_range(data_set["zero_ref_file"], data_set["unity_ref_file"])
        ref_range = [0.7, 1.0]

        signal_decay, ste_decay, times_decay = process_raw_data(
            raw_decay, ref_range
        )
        smooth_t = numpy.linspace(times[0], times[-1], 1000)
        fit_decay = relaxation_high_func(
            smooth_t, data_set["gamma"], data_set["Omega"], 0.0
        )

        color = "#0D83C5"
        facecolor = "#56B4E9"
        label = "Relaxation \nout of {}".format(r"$\ket{0}$")
        patch = mlines.Line2D(
            [],
            [],
            label=label,
            linewidth=lw,
            marker="o",
            color=color,
            markerfacecolor=facecolor,
            markersize=ms,
        )
        ax.plot(smooth_t, fit_decay, color=color, linewidth=lw)
        try:
            times_15 = numpy.where(times_decay > 15.0)[0][0]
        except:
            times_15 = None
        ax.scatter(
            times_decay[:times_15],
            signal_decay[:times_15],
            label=data_set["temp"],
            zorder=5,
            marker="o",
            color=color,
            facecolor=facecolor,
            s=ms ** 2,
        )

    ax.text(
        -0.25, 1.05, "(b)", transform=ax.transAxes, color="black", fontsize=18
    )

    # %% Sample plots

    # %% Wrap up

    fig.tight_layout(pad=0.5)
    # fig.tight_layout()


# %% Run


if __name__ == "__main__":

    tool_belt.init_matplotlib()
    # plt.rcParams.update({'font.size': 18})  # Increase font size
    matplotlib.rcParams["axes.linewidth"] = 1.0

    # nvdata_dir = common.get_nvdata_dir()

    # Data set a: room temp (300 K)
    data_a = {
        "temp": 300,
        "path_from_nvdata": "pc_hahn/branch_cryo-setup/t1_interleave_knill/data_collections/hopper-nv1_2021_03_16-300K/",
        "decay_file": "2021_05_11-21_11_55-hopper-nv1_2021_03_16",  # high to high
        "unity_ref_file": "2021_05_11-21_11_47-hopper-nv1_2021_03_16",  # zero to zero
        "zero_ref_file": "2021_05_11-21_11_53-hopper-nv1_2021_03_16",  # zero to high
        "Omega": 59.87,
        "gamma": 131.57,
    }

    # Data set b: 250 K data
    data_b = {
        "temp": 250,
        "path_from_nvdata": "pc_hahn/branch_cryo-setup/t1_interleave_knill/data_collections/hopper-nv1_2021_03_16-250K/",
        "decay_file": "2021_05_12-18_41_17-hopper-nv1_2021_03_16",  # high to high
        "unity_ref_file": "2021_05_12-18_41_10-hopper-nv1_2021_03_16",  # zero to zero
        "zero_ref_file": "2021_05_12-18_41_12-hopper-nv1_2021_03_16",  # zero to high
        "Omega": 28.53,
        "gamma": 71.51,
    }

    data_sets = [data_a, data_b]

    main(data_sets)

    plt.show(block=True)
