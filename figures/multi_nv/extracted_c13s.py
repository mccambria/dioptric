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
):
    # plot_loop_inds = [16, 45, 99]  # 0.9 MHz
    # plot_loop_inds = [0, 34, 83]  # 0.37 MHz
    plot_loop_inds = [0, 34]  # 0.37 MHz
    # plot_loop_inds = [65, 92]  # 0.102 MHz
    popts = np.array(spin_echo_fit_data["popts"])
    mod_freqs = popts[:, -2]
    plot_loop_freqs = [mod_freqs[ind] for ind in plot_loop_inds]
    colors = [KplColors.BROWN, KplColors.RED, KplColors.GRAY]

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
    fig_a, fig_b = main_fig.subfigures(nrows=2, height_ratios=(1, 0.6))
    upper_ax, lower_ax = fig_a.subplots(
        2,
        1,
        sharex=True,
        height_ratios=[10, 1],
        gridspec_kw={"hspace": 0},
    )
    main_fig.set_constrained_layout_pads(h_pad=0, hspace=0, w_pad=0, wspace=0)

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
    ax = axes_pack[1]
    ax.tick_params(axis="y", left=False)
    # ax.set_xlabel("Total evolution time (µs)")
    kpl.set_shared_ax_xlabel(ax, "Total evolution time (µs)")
    ax.set_xticks(xticks)


def simple():
    pass


if __name__ == "__main__":
    kpl.init_kplotlib()

    spin_echo_exp_data = dm.get_raw_data(
        file_id=1795168199914
    )  # w/o ionization, dmw None
    spin_echo_fit_file_id = 1796557235526  # T2_exp variable
    spin_echo_fit_data = dm.get_raw_data(file_id=spin_echo_fit_file_id)

    # fmt: off
    # From ./resonance.py, in GHz
    hfs_res = [0.008270982638238914, 0.015881063467104776, 0.014010042750685282, 0.015391657472928187, 0.012955566280101407, 0.016983227280784243]
    hfs_err_res = [0.0016409452584717822, 0.0010983852602745553, 0.0004848082620682548, 0.0007214312144406817, 0.0006541485039380769, 0.0012675922107645444]
    # fmt: on
    # From ./spin_echo/spin_echo-mcc.py, in MHz
    data = dm.get_raw_data(file_id=spin_echo_fit_file_id)
    popts = np.array(data["popts"])
    num_nvs_echo = len(popts)
    pcovs = np.array(data["pcovs"])
    red_chi_sqs = np.array(data["red_chi_sq_list"])
    osc_contrasts = popts[:, -3]
    mod_freqs = popts[:, -2]
    # criteria = [red_chi_sqs < 2.0, osc_contrasts > 0.5, mod_freqs > 0.05]
    criteria = [red_chi_sqs < 2.0, osc_contrasts > 0.1, mod_freqs > 0.05]
    # criteria = [red_chi_sqs < 2.0, osc_contrasts > 0.1, mod_freqs > 0.5]
    good_inds = list(range(num_nvs_echo))
    for el in criteria:
        good_inds = np.intersect1d(good_inds, np.where(el)[0])
    pstes = np.array([np.sqrt(np.diag(pcovs[ind])) for ind in range(len(pcovs))])
    hfs_echo = popts[good_inds, -2]
    hfs_err_echo = pstes[good_inds, -2]

    # MCC test
    hfs_echo = np.append(hfs_echo, [0] * 10)
    hfs_err_echo = np.append(hfs_err_echo, [0] * 10)

    main(
        hfs_res,
        hfs_err_res,
        hfs_echo,
        hfs_err_echo,
        spin_echo_exp_data,
        spin_echo_fit_data,
    )

    plt.show(block=True)
