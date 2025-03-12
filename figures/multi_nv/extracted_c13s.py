# -*- coding: utf-8 -*-
"""
Widefield Rabi experiment

Created on November 29th, 2023

@author: mccambria
"""

import sys
import time
import traceback

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from numba import njit
from scipy.optimize import brute

from figures.multi_nv.spin_echo.spin_echo_mcc import quartic_decay
from majorroutines.widefield import base_routine
from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import tool_belt as tb
from utils import widefield as widefield
from utils.kplotlib import KplColors
from utils.tool_belt import curve_fit


def main(
    hfs_res,
    hfs_err_res,
    hfs_echo,
    hfs_err_echo,
    spin_echo_exp_data,
    spin_echo_fit_data,
    circle_or_square_echo,
    circle_or_square_res,
):
    # plot_loop_inds = [16, 45, 99]  # 0.9 MHz
    # plot_loop_inds = [0, 34, 83]  # 0.37 MHz
    # plot_loop_inds = [0, 34]  # 0.37 MHz
    plot_loop_inds = [34, 0]  # 0.37 MHz
    # plot_loop_inds = [65, 92]  # 0.102 MHz
    popts = np.array(spin_echo_fit_data["popts"])
    pcovs = np.array(spin_echo_fit_data["pcovs"])
    pstes = np.array([np.sqrt(np.diag(pcovs[ind])) for ind in range(len(pcovs))])
    mod_freqs = popts[:, -2]
    plot_loop_freqs = [mod_freqs[ind] for ind in plot_loop_inds]
    colors = [KplColors.RED, KplColors.BROWN, KplColors.GRAY]

    ### Data work

    res_order = np.argsort(hfs_res)
    hfs_res = 1000 * np.array(hfs_res)[res_order]
    hfs_err_res = 1000 * np.array(hfs_err_res)[res_order]
    circle_or_square_res = np.array(circle_or_square_res)[res_order]
    echo_order = np.argsort(hfs_echo)
    hfs_echo = np.array(hfs_echo)[echo_order]
    hfs_err_echo = np.array(hfs_err_echo)[echo_order]
    circle_or_square_echo_t2 = np.array(circle_or_square_echo)
    circle_or_square_echo = np.array(circle_or_square_echo)[echo_order]
    sorted_plot_loop_inds = [np.where(echo_order == val)[0] for val in plot_loop_inds]

    ### Fig setup

    figsize = kpl.figsize
    adj_figsize = (figsize[0], 1.5 * figsize[1])
    fig, axes_pack = plt.subplots(2, 1, figsize=adj_figsize)

    ### Mod freq plot

    ax = axes_pack[0]

    num_nvs = len(hfs_res) + len(hfs_echo)

    first_ind = None
    nv_ind = 0
    for hfs_list, hfs_err_list, color, label, circle_or_square in zip(
        (hfs_echo, hfs_res),
        (hfs_err_echo, hfs_err_res),
        (kpl.KplColors.BLUE, kpl.KplColors.RED),
        ("Spin echo", "ESR"),
        (circle_or_square_echo, circle_or_square_res),
    ):
        for ind in range(len(hfs_list)):
            # if ind == len(hfs_list) - 5:
            #     break
            hfs_val = hfs_list[ind]
            hfs_err = hfs_err_list[ind]
            # plot_ax = lower_ax if hfs_val == 0 else upper_ax
            # if color == kpl.KplColors.BLUE and hfs_val in plot_loop_freqs:
            #     sub_ind = plot_loop_freqs.index(hfs_val)
            #     kpl.plot_points(
            #         plot_ax,
            #         nv_ind,
            #         hfs_val,
            #         hfs_err,
            #         color=colors[sub_ind],
            #         marker="D",
            #         size=kpl.Size.SMALL,
            #     )
            # else:
            if hfs_val > 0:
                marker = "o" if circle_or_square[ind] else "s"
                kpl.plot_points(
                    ax,
                    nv_ind,
                    hfs_val,
                    hfs_err,
                    color=color,
                    label=label,
                    marker=marker,
                )
                if first_ind is None:
                    first_ind = nv_ind
                label = None
            nv_ind += 1

    # From Smeltzer 2011
    for theory_val, theory_err in zip(
        [14.8, 13.9, 7.5, 5.7, 4.6, 4.67, 2.63, 2.27],
        [0.1, 0.1, 0.1, 0.2, 0.1, 0.04, 0.07, 0.04],
    ):
        # Experiment values from same paper
        # for theory_val, theory_err in zip(
        #     [13.72, 12.78, 8.92, 6.52, 4.2, 2.4],
        #     [0.03, 0.01, 0.03, 0.04, 0.1, 0.3],
        # ):
        # ax.axhline(theory_val, color=kpl.KplColors.LIGHT_GRAY, zorder=-50)
        ax.fill_between(
            [-1, num_nvs + 2],
            theory_val - theory_err,
            theory_val + theory_err,
            color=kpl.KplColors.LIGHT_GRAY,
            zorder=-50,
        )

    ### Fig labels etc

    # ax.set_xlabel("NV index (ascending order)")
    ax.set_xlabel(r"NV index ($A_{\text{hfs},i}<A_{\text{hfs},i+1}$)")
    num_nvs = nv_ind - 1
    margin = (num_nvs - first_ind) / 70
    ax.set_xlim(first_ind - margin, num_nvs + margin)
    ax.set_xticks([45, 60, 75, 90])
    # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylabel("Coupling strength (MHz)")
    # ax.set_ylabel("$^{13}$C hyperfine coupling (MHz)")
    ax.set_yscale("log")
    ax.set_yticks([0.1, 1, 10], [0.1, 1, 10])
    ax.legend(loc=kpl.Loc.UPPER_LEFT)

    ### T2

    bad_inds = [47, 55, 61, 62, 68, 97]

    num_vals = len(popts[:, 4])
    T2_echo = np.array(
        [popts[ind, 4] for ind in range(num_vals) if ind not in bad_inds]
    )
    T2_echo_err = np.array(
        [pstes[ind, 4] for ind in range(num_vals) if ind not in bad_inds]
    )
    echo_order = np.argsort(T2_echo)
    T2_echo = T2_echo[echo_order]
    T2_echo_err = T2_echo_err[echo_order]
    circle_or_square_echo_t2 = circle_or_square_echo_t2[echo_order]
    keep_inds = T2_echo < 0.3
    num_clipped = len(T2_echo) - np.count_nonzero(keep_inds)
    T2_echo = T2_echo[keep_inds]
    T2_echo_err = T2_echo_err[keep_inds]
    circle_or_square_echo_t2 = circle_or_square_echo_t2[keep_inds]

    num_vals = len(T2_echo)
    x_vals = np.array(range(num_vals)) + num_clipped

    ax = axes_pack[1]

    for ind in range(len(x_vals)):
        marker = "o" if circle_or_square_echo_t2[ind] else "s"
        kpl.plot_points(
            ax,
            x_vals[ind],
            T2_echo[ind] * 1000,
            T2_echo_err[ind] * 1000,
            color=kpl.KplColors.GREEN,
            marker=marker,
        )
    # ax.set_xlabel("NV index (ascending order)")
    ax.set_xlabel(r"NV index ($T_{2,i}<T_{2,i+1}$)")
    ax.set_ylabel("$T_{2}$ time (µs)")
    ax.set_yscale("log")
    ax.set_yticks([30, 100, 300], [30, 100, 300])
    ax.set_ylim(30, 600)
    margin = (np.max(x_vals) - np.min(x_vals)) / 70
    ax.set_xlim(np.min(x_vals) - margin, np.max(x_vals) + margin)


def main_v1(
    hfs_res,
    hfs_err_res,
    hfs_echo,
    hfs_err_echo,
    spin_echo_exp_data,
    spin_echo_fit_data,
):
    # plot_loop_inds = [16, 45, 99]  # 0.9 MHz
    # plot_loop_inds = [0, 34, 83]  # 0.37 MHz
    # plot_loop_inds = [0, 34]  # 0.37 MHz
    plot_loop_inds = [34, 0]  # 0.37 MHz
    # plot_loop_inds = [65, 92]  # 0.102 MHz
    popts = np.array(spin_echo_fit_data["popts"])
    mod_freqs = popts[:, -2]
    plot_loop_freqs = [mod_freqs[ind] for ind in plot_loop_inds]
    colors = [KplColors.RED, KplColors.BROWN, KplColors.GRAY]

    ### Data work

    res_order = np.argsort(hfs_res)
    hfs_res = 1000 * np.array(hfs_res)[res_order]
    hfs_err_res = 1000 * np.array(hfs_err_res)[res_order]
    echo_order = np.argsort(hfs_echo)
    hfs_echo = np.array(hfs_echo)[echo_order]
    hfs_err_echo = np.array(hfs_err_echo)[echo_order]
    sorted_plot_loop_inds = [np.where(echo_order == val)[0] for val in plot_loop_inds]

    ### Fig setup

    figsize = kpl.figsize
    adj_figsize = (figsize[0], 1.5 * figsize[1])
    main_fig = plt.figure(figsize=adj_figsize)
    main_fig.get_layout_engine().set(rect=[0.002, 0.002, 0.995, 0.996])
    main_fig.get_layout_engine().set(h_pad=0)
    main_fig.get_layout_engine().set(w_pad=0)
    fig_a, fig_b = main_fig.subfigures(nrows=2, height_ratios=(1, 0.6))
    upper_ax, lower_ax = fig_a.subplots(
        2,
        1,
        sharex=True,
        height_ratios=[10, 1],
        gridspec_kw={"hspace": 0},
    )

    ### Main plot

    num_nvs = len(hfs_res) + len(hfs_echo)

    nv_ind = 1
    for hfs_list, hfs_err_list, color, label in zip(
        (hfs_echo, hfs_res),
        (hfs_err_echo, hfs_err_res),
        (kpl.KplColors.BLUE, kpl.KplColors.GREEN),
        ("Spin echo", "ESR"),
    ):
        for ind in range(len(hfs_list)):
            # if ind == len(hfs_list) - 5:
            #     break
            hfs_val = hfs_list[ind]
            hfs_err = hfs_err_list[ind]
            plot_ax = lower_ax if hfs_val == 0 else upper_ax
            if color == kpl.KplColors.BLUE and hfs_val in plot_loop_freqs:
                sub_ind = plot_loop_freqs.index(hfs_val)
                kpl.plot_points(
                    plot_ax,
                    nv_ind,
                    hfs_val,
                    hfs_err,
                    color=colors[sub_ind],
                    marker="D",
                    size=kpl.Size.SMALL,
                )
            else:
                kpl.plot_points(
                    plot_ax, nv_ind, hfs_val, hfs_err, color=color, label=label
                )
            nv_ind += 1
            if hfs_val > 0:
                label = None

    # From Smeltzer 2011
    for theory_val, theory_err in zip(
        [14.8, 13.9, 7.5, 5.7, 4.6, 4.67, 2.63, 2.27],
        [0.1, 0.1, 0.1, 0.2, 0.1, 0.04, 0.07, 0.04],
    ):
        # Experiment values from same paper
        # for theory_val, theory_err in zip(
        #     [13.72, 12.78, 8.92, 6.52, 4.2, 2.4],
        #     [0.03, 0.01, 0.03, 0.04, 0.1, 0.3],
        # ):
        # ax.axhline(theory_val, color=kpl.KplColors.LIGHT_GRAY, zorder=-50)
        upper_ax.fill_between(
            [-1, num_nvs + 2],
            theory_val - theory_err,
            theory_val + theory_err,
            color=kpl.KplColors.LIGHT_GRAY,
            zorder=-50,
        )

    ### Fig labels etc

    lower_ax.set_xlabel("NV index")
    margin = 0.8
    lower_ax.set_xlim(-margin, num_nvs + 1 + margin)
    lower_ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    lower_ax.set_yticks([0])
    upper_ax.set_ylabel("$^{13}$C hyperfine coupling (MHz)")
    # upper_ax.set_ylabel("Envelope frequency (MHz)")
    # upper_ax.set_ylabel("T2 (ms)")
    upper_ax.set_yscale("log")
    upper_ax.legend(loc=kpl.Loc.UPPER_LEFT)
    upper_ax.tick_params(axis="x", bottom=False)

    ### Lower quad plots showing individual traces

    axes_pack = fig_b.subplots(
        1,
        len(plot_loop_inds),
        sharey=True,
        gridspec_kw={"wspace": 0},
    )

    norm_counts = np.array(spin_echo_exp_data["norm_counts"])
    norm_counts_ste = np.array(spin_echo_exp_data["norm_counts_ste"])
    red_chi_sqs = spin_echo_fit_data["red_chi_sq_list"]
    taus = np.array(spin_echo_exp_data["taus"])
    total_evolution_times = 2 * np.array(taus) / 1e3
    fit_fn = quartic_decay

    # Skip indices with bad pi pulses etc
    split_esr = [12, 13, 14, 61, 116]
    broad_esr = [52, 11]
    weak_esr = [72, 64, 55, 96, 112, 87, 12, 58, 36]
    skip_inds = list(set(split_esr + broad_esr + weak_esr))
    nv_inds = [ind for ind in range(117) if ind not in skip_inds]

    ax_ind = 0
    for loop_ind, nv_ind in enumerate(nv_inds):
        if loop_ind not in plot_loop_inds:
            continue
        nv_counts = norm_counts[nv_ind]
        nv_counts_ste = norm_counts_ste[nv_ind]
        ax = axes_pack[ax_ind]
        color = colors[ax_ind]
        kpl.plot_points(
            ax,
            total_evolution_times,
            nv_counts,
            nv_counts_ste,
            color=color,
            size=kpl.Size.SMALL,
        )
        linspace_taus = np.linspace(0, np.max(total_evolution_times), 10000)
        linspace_taus = linspace_taus[1:]  # Exclude tau=0 which can diverge
        popt = popts[loop_ind]
        kpl.plot_line(
            ax,
            linspace_taus,
            fit_fn(linspace_taus, *popt),
            color=color,
        )
        ax_ind += 1
        ax.set_xlim(41, 62)

    xticks = [42, 50, 58]
    ax = axes_pack[0]
    ax.set_ylabel("NV$^{-}$ population (arb. units)")
    # ax.tick_params(axis="y", right=True)
    ax.set_xticks(xticks)
    ax.set_ylim(0.12, 0.71)
    ax = axes_pack[1]
    ax.tick_params(axis="y", left=False)
    # ax.set_xlabel("Total evolution time (µs)")
    kpl.set_shared_ax_xlabel(ax, "Total evolution time (µs)")
    ax.set_xticks(xticks)


def simple():
    pass


if __name__ == "__main__":
    kpl.init_kplotlib()

    # For spin echo
    # fmt: off
    nva_inds = [3, 4, 5, 7, 15, 16, 17, 18, 21, 22, 24, 26, 27, 29, 30, 34, 37, 40, 41, 45, 47, 49, 51, 53, 54, 59, 60, 65, 66, 70, 71, 73, 74, 76, 78, 79, 83, 84, 89, 93, 94, 97, 98, 104, 105, 109, 110, 111, 115]
    nvb_inds = [0, 1, 2, 6, 8, 9, 10, 19, 20, 23, 25, 28, 31, 32, 33, 35, 38, 39, 42, 43, 44, 46, 48, 50, 56, 57, 62, 63, 67, 68, 69, 75, 77, 80, 81, 82, 85, 86, 88, 90, 91, 92, 95, 99, 100, 101, 102, 103, 106, 107, 108, 113, 114]
    # fmt: on

    # w/ ionization, dmw None
    spin_echo_exp_data = dm.get_raw_data(file_id=1797877478132)
    # spin_echo_fit_file_id = 1798006231161  # Complicated
    spin_echo_fit_file_id = 1800359001443  # Otsu
    spin_echo_fit_data = dm.get_raw_data(file_id=spin_echo_fit_file_id)
    spin_echo_fit_no_osc_file_id = 1798052675001
    spin_echo_fit_no_osc_data = dm.get_raw_data(file_id=spin_echo_fit_no_osc_file_id)

    ### ESR
    # fmt: off
    # From ./resonance.py, in GHz
    # hfs_res = [0.008270982638238914, 0.015881063467104776, 0.014010042750685282, 0.015391657472928187, 0.012955566280101407, 0.016983227280784243]
    # hfs_err_res = [0.0016409452584717822, 0.0010983852602745553, 0.0004848082620682548, 0.0007214312144406817, 0.0006541485039380769, 0.0012675922107645444]
    hfs_res = [0.01594705781409969, 0.008112202302337768, 0.01434832304717241, 0.015095061294166667, 0.012955566278621947, 0.01714009128506917]
    hfs_err_res = [0.0010531962934879878, 0.0015198744553920284, 0.000490856368946772, 0.0007013394418482069, 0.0006541485041257045, 0.0013231812930978237]
    circle_or_square_res = [True, True, True, False, False, False]
    # fmt: on

    ### ESR
    # From ./spin_echo/spin_echo-mcc.py, in MHz

    # Extract from data set with oscillations allowed
    osc_data = dm.get_raw_data(file_id=spin_echo_fit_file_id)
    osc_popts = np.array(osc_data["popts"])
    num_nvs_echo = len(osc_popts)
    osc_pcovs = np.array(osc_data["pcovs"])
    osc_red_chi_sqs = np.array(osc_data["red_chi_sq_list"])

    # Extract from data set with oscillations not allowed
    no_osc_red_chi_sqs = np.array(spin_echo_fit_no_osc_data["red_chi_sq_list"])

    bad_inds = []
    hfs_echo = []
    hfs_err_echo = []
    no_osc_inds = []
    circle_or_square_echo = []
    for nv_ind in range(num_nvs_echo):
        osc_red_chi_sq = osc_red_chi_sqs[nv_ind]
        no_osc_red_chi_sq = no_osc_red_chi_sqs[nv_ind]
        # Neither fit was good
        if osc_red_chi_sq > 3 and no_osc_red_chi_sq > 3:
            bad_inds.append(nv_ind)
        # Osc fit is significantly better than no osc fit
        # elif True:
        elif (
            osc_red_chi_sq < no_osc_red_chi_sq - 0.5
            # and not osc_red_chi_sq < no_osc_red_chi_sq - 0.7
        ):
            # no_osc_inds.append(nv_ind)
            osc_contrast = osc_popts[nv_ind, -3]
            mod_freq_ind = -2
            mod_freq = osc_popts[nv_ind, mod_freq_ind]
            mod_freq_err = np.sqrt(np.diag(osc_pcovs[nv_ind]))[mod_freq_ind]
            hfs_echo.append(mod_freq)
            hfs_err_echo.append(mod_freq_err)
        # Otherwise assume no legitimate oscillations
        else:
            hfs_echo.append(0)
            hfs_err_echo.append(0)
            no_osc_inds.append(nv_ind)
        circle_or_square_echo.append(nv_ind in nva_inds)

    print(bad_inds)
    print(no_osc_inds)

    # hfs_res = []
    # hfs_err_res = []

    main(
        hfs_res,
        hfs_err_res,
        hfs_echo,
        hfs_err_echo,
        spin_echo_exp_data,
        spin_echo_fit_data,
        circle_or_square_echo,
        circle_or_square_res,
    )

    plt.show(block=True)
