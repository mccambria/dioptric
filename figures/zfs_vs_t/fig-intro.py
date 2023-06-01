# -*- coding: utf-8 -*-
"""
Intro figure for zfs vs t paper

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
import matplotlib as mpl
from matplotlib.collections import PolyCollection
from scipy.optimize import curve_fit
import csv
import pandas as pd
import sys
from analysis import three_level_rabi
from figures.zfs_vs_t.zfs_vs_t_main import (
    get_data_points,
    get_fitted_model,
    data_points_to_lists,
)
from figures.zfs_vs_t.thermal_expansion import fit_double_occupation


# endregion


def main():
    # setpoint_temps = ["", 350, 400, 450, 500]  # In increasing temp order
    # setpoint_temps = [10, 150, "", 400, 500]  # In increasing temp order
    setpoint_temps = [10, 140, 250, 370, 500]  # In increasing temp order
    # nvs = ["nv7_zfs_vs_t"]
    nvs = ["nv3_zfs_vs_t", "nv7_zfs_vs_t"]
    skip_lambda = (
        lambda point: point["Skip"]
        or point["Sample"] != "Wu"
        or point["Setpoint temp (K)"] not in setpoint_temps
        or point["NV"] not in nvs
    )
    data_points = get_data_points(skip_lambda)

    min_freq = 2.833
    # max_freq = 2.882
    max_freq = 2.889
    freq_center = (min_freq + max_freq) / 2
    freq_range = max_freq - min_freq
    smooth_freqs = pesr.calculate_freqs(freq_center, freq_range, 100)

    edgecolors = []
    # edgecolors = [KplColors.GREEN]
    # Yellow to red
    edgecolors.extend(["#baa309", "#cc771d", "#d8572a", "#c32f27", "#87081f"])
    facecolors = [kpl.lighten_color_hex(el) for el in edgecolors]

    narrow_figsize = (0.65 * kpl.figsize[0], kpl.figsize[1])
    fig, ax = plt.subplots(figsize=narrow_figsize)
    ax2 = ax.twinx()
    # ax3 = ax.twiny()

    min_fluor = 0
    max_fluor = 1.7
    min_temp = 0
    max_temp = 595
    # min_temp = -70
    # max_temp = 510

    num_sets = len(setpoint_temps)
    for ind in range(num_sets):
        # Reverse
        ind = num_sets - 1 - ind

        data_point = data_points[ind]

        fig_file = data_point["ZFS file"]
        edgecolor = edgecolors[ind]
        facecolor = facecolors[ind]
        temp = data_point["Monitor temp (K)"]

        data = tool_belt.get_raw_data(fig_file)
        freq_center = data["freq_center"]
        freq_range = data["freq_range"]
        num_steps = data["num_steps"]
        freqs = pesr.calculate_freqs(freq_center, freq_range, num_steps)
        # smooth_freqs = pesr.calculate_freqs(freq_center, freq_range, 100)

        ref_counts = data["ref_counts"]
        sig_counts = data["sig_counts"]
        num_reps = data["num_reps"]
        nv_sig = data["nv_sig"]
        sample = nv_sig["name"].split("-")[0]
        readout = nv_sig["spin_readout_dur"]
        uwave_pulse_dur = data["uwave_pulse_dur"]

        try:
            norm_style = tool_belt.NormStyle[str.upper(nv_sig["norm_style"])]
        except Exception as exc:
            # norm_style = NormStyle.POINT_TO_POINT
            norm_style = tool_belt.NormStyle.SINGLE_VALUED

        ret_vals = tool_belt.process_counts(
            sig_counts, ref_counts, num_reps, readout, norm_style
        )
        (
            sig_counts_avg_kcps,
            ref_counts_avg_kcps,
            norm_avg_sig,
            norm_avg_sig_ste,
        ) = ret_vals

        # popt = (
        #     data_point["Contrast"],
        #     data_point["Width (MHz)"],
        #     data_point["ZFS (GHz)"],
        #     data_point["Splitting (MHz)"],
        # )
        # fit_func = lambda f: 1 - three_level_rabi.coherent_line(
        #     f, *popt, uwave_pulse_dur
        # )

        popt = (
            data_point["Contrast"],
            data_point["Width (MHz)"],
            data_point["ZFS (GHz)"],
            data_point["Splitting (MHz)"],
        )
        fit_func = lambda f: 1 - pesr.lorentzian_split(f, *popt)

        label = f"{int(temp)} K"
        fit_vals = fit_func(smooth_freqs)
        # y_offset = 0.25 * (temp - 296) / 100
        # Set the base of the fit to the proper temp
        y_min = min(fit_vals)
        normed_height = (temp - min_temp) / (max_temp - min_temp)
        # y_offset = (normed_height * (max_fluor - min_fluor)) - y_min
        y_offset = normed_height * (max_fluor - min_fluor) + min_fluor - y_min
        # y_offset = normed_height * (max_fluor - min_fluor) + min_fluor - 1
        # kpl.plot_line(
        kpl.plot_points(
            ax,
            freqs,
            y_offset + norm_avg_sig,
            color=edgecolor,
            size=kpl.Size.TINY,
            # markerfacecolor=facecolor,
            # label=label,
            # size=kpl.Size.SMALL,
            # marker="o",
            # markersize=kpl.Size.TINY,
            zorder=3,
        )
        kpl.plot_line(
            ax,
            smooth_freqs,
            y_offset + fit_vals,
            # color=KplColors.DARK_GRAY,
            color=edgecolor,
            zorder=2,
        )

        if ind == num_sets - 1:
            y_pos = y_offset + 0.86
        else:
            y_pos = y_offset + 0.92
        ax.text(min_freq + 0.0005, y_pos, label, color=edgecolor, fontsize=15)

    # ax.legend(
    #     handlelength=0.5,
    #     borderpad=0.3,
    #     borderaxespad=0.3,
    #     handletextpad=0.6,
    #     loc=kpl.Loc.LOWER_LEFT,
    # )
    ax.tick_params(left=False, labelleft=False)
    # ax.get_yaxis().set_visible(False)
    ax.set(
        # xlabel="ODMR freq. / ZFS (GHz)",
        # xlabel="Frequency (GHz)",
        xlabel="Frequency (MHz)",
        xticks=[2.84, 2.86, 2.88],
        xticklabels=[2840, 2860, 2880],
        # ylabel="Fluorescence",
        ylabel="Normalized fluorescence",
        xlim=(min_freq, max_freq),
        # ylim=(None, 2.1),
        ylim=(min_fluor, max_fluor),
    )

    ### Plot D(T)
    skip_lambda = lambda point: (
        point["Skip"]
        or point["Sample"] != "Wu"
        # or point["Sample"] != "15micro"
        # or point["ZFS file"] == ""
        # or point["Monitor temp (K)"] >= 296
    )
    data_points = get_data_points(
        skip_lambda,
        condense_all=False,
        condense_samples=True,
    )
    (
        zfs_list,
        zfs_err_list,
        temp_list,
        label_list,
        color_list,
        group_list,
    ) = data_points_to_lists(data_points)
    cambria_lambda = get_fitted_model(temp_list, zfs_list, zfs_err_list)
    temp_linspace = np.linspace(0, 600, 1000)
    d_of_t_color = KplColors.BLUE
    kpl.plot_line(ax2, cambria_lambda(temp_linspace), temp_linspace, color=d_of_t_color)
    ax2.set(
        # xlabel="Zero-field splitting (GHz)",
        # ylabel="Temperature (K)",
        # xlim=(min_freq, max_freq),
        # ylim=(284, 547),
        ylim=(min_temp, max_temp),
        zorder=+1,
    )
    ax2.set_ylabel("Temperature (K)", color=d_of_t_color)
    ax2.tick_params(axis="y", color=d_of_t_color, labelcolor=d_of_t_color)
    ax2.xaxis.label.set_color(d_of_t_color)
    ax2.spines["right"].set_color(d_of_t_color)
    # ax2.spines["top"].set_color(d_of_t_color)
    # ax3.set_xlabel("Zero-field splitting (GHz)", color=d_of_t_color)
    # ax3.tick_params(axis="x", which="both", top=False, labeltop=False)


def waterfall():
    width = 1.0 * kpl.figsize[0]
    height = 0.8 * width
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_subplot(projection="3d")

    setpoint_temps = np.arange(310, 500, 10)
    setpoint_temps = setpoint_temps.tolist()
    setpoint_temps.insert(0, "")
    min_temp = 296
    max_temp = setpoint_temps[-1]

    min_freq = 2.84
    max_freq = 2.881
    freq_center = (min_freq + max_freq) / 2
    freq_range = max_freq - min_freq
    smooth_freqs = pesr.calculate_freqs(freq_center, freq_range, 100)

    skip_lambda = (
        lambda point: point["Skip"]
        or point["Sample"] != "Wu"
        or point["Setpoint temp (K)"] not in setpoint_temps
        or point["NV"] != "nv7_zfs_vs_t"
    )
    data_points = get_data_points(skip_lambda)

    # cmap_name = "coolwarm"
    # cmap_name = "autumn_r"
    # cmap_name = "magma_r"
    cmap_name = "plasma"
    cmap = mpl.colormaps[cmap_name]
    cmap_offset = 0

    poly_zero = 0.75
    poly = lambda x, y: [(x[0], 1.0), *zip(x, y), (x[-1], 1.0)]
    # verts[i] is a list of (x, y) pairs defining polygon i.
    verts = []
    colors = []
    temps = []

    num_sets = len(setpoint_temps)
    for ind in range(num_sets):
        # Reverse
        ind = num_sets - 1 - ind

        data_point = data_points[ind]
        fig_file = data_point["ZFS file"]
        temp = data_point["Monitor temp (K)"]
        temps.append(temp)
        norm_temp = (temp - min_temp + cmap_offset) / (
            max_temp - min_temp + cmap_offset + 25
        )
        color = cmap(norm_temp)

        popt = (
            data_point["Contrast"],
            data_point["Width (MHz)"],
            data_point["ZFS (GHz)"],
            data_point["Splitting (MHz)"],
        )

        data = tool_belt.get_raw_data(fig_file)
        freq_center = data["freq_center"]
        freq_range = data["freq_range"]
        num_steps = data["num_steps"]
        freqs = pesr.calculate_freqs(freq_center, freq_range, num_steps)

        ref_counts = data["ref_counts"]
        sig_counts = data["sig_counts"]
        num_reps = data["num_reps"]
        nv_sig = data["nv_sig"]
        sample = nv_sig["name"].split("-")[0]
        readout = nv_sig["spin_readout_dur"]
        uwave_pulse_dur = data["uwave_pulse_dur"]

        try:
            norm_style = tool_belt.NormStyle[str.upper(nv_sig["norm_style"])]
        except Exception as exc:
            # norm_style = NormStyle.POINT_TO_POINT
            norm_style = tool_belt.NormStyle.SINGLE_VALUED

        ret_vals = tool_belt.process_counts(
            sig_counts, ref_counts, num_reps, readout, norm_style
        )
        norm_avg_sig = ret_vals[2]

        ax.plot(
            freqs,
            norm_avg_sig,
            zs=temp,
            zdir="x",
            color=color,
            linestyle="None",
            marker="o",
            markersize=2,
            # alpha=0.5,
        )

        fit_func = lambda f: 1 - three_level_rabi.coherent_line(
            f, *popt, uwave_pulse_dur
        )
        verts.append(poly(smooth_freqs, fit_func(smooth_freqs)))
        colors.append(color)
        ax.plot(
            smooth_freqs,
            fit_func(smooth_freqs),
            zs=temp,
            # color=KplColors.DARK_GRAY,
            zdir="x",
            color=color,
            # alpha=0.5,
        )

    # poly = PolyCollection(verts, facecolors=colors, alpha=0.7)
    # ax.add_collection3d(poly, zs=temps, zdir="x")

    ax.set(
        xlim=(510, 290),
        ylim=(min_freq, max_freq),
        zlim=(poly_zero, 1.03),
        xlabel="\n$T$ (K)",
        # ylabel="\n$f$ (GHz)",
        ylabel="\n$f$ (MHz)",
        zlabel="$C$",
        yticks=[2.84, 2.86, 2.88],
        yticklabels=[2840, 2860, 2880],
        zticks=[0.8, 0.9, 1.0],
    )
    ax.view_init(elev=38, azim=-22, roll=0)
    # ax.tick_params(left=False, labelleft=False)
    # ax.get_yaxis().set_visible(False)

    fig.tight_layout()
    # fig.tight_layout(rect=(-0.05, 0, 0.95, 1))


def false3d():
    setpoint_temps = ["", 350, 400, 450, 500]  # In increasing temp order
    skip_lambda = (
        lambda point: point["Skip"]
        or point["Sample"] != "Wu"
        or point["Setpoint temp (K)"] not in setpoint_temps
        or point["NV"] != "nv7_zfs_vs_t"
    )
    data_points = get_data_points(skip_lambda)

    min_freq = 2.837
    max_freq = 2.882
    smooth_freq_center = (min_freq + max_freq) / 2
    smooth_freq_range = max_freq - min_freq
    smooth_freqs = pesr.calculate_freqs(smooth_freq_center, smooth_freq_range, 100)

    # Blue, green, yellow, red, brown
    edgecolors = ["#2e83c0", "#4db449", "#f1aa30", "#fb2e18", "#8c564b"]
    facecolors = [kpl.lighten_color_hex(el) for el in edgecolors]

    narrow_figsize = (0.55 * kpl.figsize[0], kpl.figsize[1])
    fig, ax = plt.subplots(figsize=narrow_figsize)
    x_offset_step = 0.005
    y_offset_step = 0.25

    num_sets = len(setpoint_temps)
    for ind in range(num_sets):
        # Reverse
        # ind = num_sets - 1 - ind

        data_point = data_points[ind]

        fig_file = data_point["ZFS file"]
        edgecolor = edgecolors[ind]
        facecolor = facecolors[ind]
        temp = data_point["Monitor temp (K)"]

        popt = (
            data_point["Contrast"],
            data_point["Width (MHz)"],
            data_point["ZFS (GHz)"],
            data_point["Splitting (MHz)"],
        )

        data = tool_belt.get_raw_data(fig_file)
        freq_center = data["freq_center"]
        freq_range = data["freq_range"]
        num_steps = data["num_steps"]
        freqs = pesr.calculate_freqs(freq_center, freq_range, num_steps)
        # smooth_freqs = pesr.calculate_freqs(freq_center, freq_range, 100)

        ref_counts = data["ref_counts"]
        sig_counts = data["sig_counts"]
        num_reps = data["num_reps"]
        nv_sig = data["nv_sig"]
        sample = nv_sig["name"].split("-")[0]
        readout = nv_sig["spin_readout_dur"]
        uwave_pulse_dur = data["uwave_pulse_dur"]

        try:
            norm_style = tool_belt.NormStyle[str.upper(nv_sig["norm_style"])]
        except Exception as exc:
            # norm_style = NormStyle.POINT_TO_POINT
            norm_style = tool_belt.NormStyle.SINGLE_VALUED

        ret_vals = tool_belt.process_counts(
            sig_counts, ref_counts, num_reps, readout, norm_style
        )
        (
            sig_counts_avg_kcps,
            ref_counts_avg_kcps,
            norm_avg_sig,
            norm_avg_sig_ste,
        ) = ret_vals

        fit_func = lambda f: 1 - three_level_rabi.coherent_line(
            f, *popt, uwave_pulse_dur
        )

        x_offset = x_offset_step * ind
        y_offset = y_offset_step * ind
        # offset = 0
        # kpl.plot_line(
        kpl.plot_points(
            ax,
            freqs + x_offset,
            y_offset + norm_avg_sig,
            color=edgecolor,
            label=f"{int(temp)} K",
            markersize=4,
            # markersize=4 - 0.25 * ind,
        )
        adj_smooth_freqs = smooth_freqs + x_offset
        kpl.plot_line(
            ax,
            adj_smooth_freqs,
            y_offset + fit_func(smooth_freqs),
            # color=KplColors.DARK_GRAY,
            color=edgecolor,
            # linewidth=1.5 - 0.1 * ind,
        )
        last_smooth_freqs = adj_smooth_freqs[-1]

    # ax.legend(
    #     handlelength=0.5,
    #     borderpad=0.3,
    #     borderaxespad=0.3,
    #     handletextpad=0.6,
    #     loc=kpl.Loc.LOWER_LEFT,
    # )
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("Normalized fluorescence")
    ax.tick_params(left=False, labelleft=False)
    # ax.get_yaxis().set_visible(False)

    fake_axis_linspace = np.array(
        [min_freq - x_offset_step * 0.0, min_freq + x_offset_step * (num_sets + 0.5)]
    )
    xlim = [fake_axis_linspace[0], last_smooth_freqs]
    y_offset = 0.75 - (y_offset_step / x_offset_step) * xlim[0]
    fake_axis_lambda = lambda x: y_offset + (y_offset_step / x_offset_step) * x
    fake_axis_vals = fake_axis_lambda(fake_axis_linspace)
    # kpl.plot_line(
    #     ax,
    #     fake_axis_linspace,
    #     fake_axis_vals,
    #     color=KplColors.BLACK,
    #     linewidth=1.0,
    # )
    ax.arrow(
        fake_axis_linspace[0],
        fake_axis_vals[0],
        fake_axis_linspace[1] - fake_axis_linspace[0],
        fake_axis_vals[1] - fake_axis_vals[0],
        color=KplColors.BLACK,
        width=0.00001,
        head_width=0.001,
        head_length=0.03,
    )
    ax.set_xlim(xlim)
    ax.set_ylim(fake_axis_vals[0], fake_axis_vals[1] + 0.08)


def quasiharmonic_sketch():
    kpl_figsize = kpl.figsize
    adj_figsize = (kpl_figsize[0], 0.8 * kpl_figsize[1])
    fig, ax = plt.subplots(figsize=adj_figsize)

    min_temp = 170
    max_temp = 230
    temp_linspace = np.linspace(min_temp, max_temp, 1000)

    lattice_constant = fit_double_occupation()

    parabola_points = np.linspace(180, 220, 5)

    for point in parabola_points:
        parabola_linspace = np.linspace(point - 5, point + 5, 100)
        parabola_lambda = lambda t: 1.25e-6 * (t - point) ** 2 + lattice_constant(point)
        if point == 200:
            color = KplColors.DARK_GRAY
        else:
            color = KplColors.LIGHT_GRAY
        kpl.plot_line(
            ax, parabola_linspace, parabola_lambda(parabola_linspace), color=color
        )

    kpl.plot_line(
        ax, temp_linspace, lattice_constant(temp_linspace), color=KplColors.RED
    )

    ax.set_xlim(min_temp, max_temp)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("Temperature T (K)")
    ax.set_ylabel(r"Lattice constant ($\si{\angstrom}$)")


if __name__ == "__main__":
    kpl.init_kplotlib()

    main()
    # waterfall()
    # false3d()
    # quasiharmonic_sketch()

    plt.show(block=True)
