# -*- coding: utf-8 -*-
"""
Main text fig 2

Created on June 5th, 2024

@author: mccambria
"""

import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from majorroutines.widefield import resonance, spin_echo
from utils import common
from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import tool_belt as tb
from utils import widefield as widefield
from utils.constants import NVSig


def main(esr_data, spin_echo_data):
    ### Setup

    # w_factor = 0.4
    # h_factor = 0.5
    # figsize = kpl.double_figsize
    # figsize[1] *= 1.3
    # main_fig = plt.figure(figsize=figsize)
    # seq_esr_fig, spin_echo_double_fig = main_fig.subfigures(
    #     ncols=2, width_ratios=(w_factor, 1 - w_factor)
    # )
    # seq_fig, esr_fig = seq_esr_fig.subfigures(
    #     nrows=2, height_ratios=(h_factor, 1 - h_factor)
    # )
    # spin_echo_fig, spin_echo_zoom_fig = spin_echo_double_fig.subfigures(
    #     nrows=2, height_ratios=(1, 1)
    # )

    figsize = kpl.double_figsize
    figsize[1] *= 1.6
    main_fig = plt.figure(figsize=figsize)
    seq_esr_fig, spin_echo_figs = main_fig.subfigures(nrows=2, height_ratios=(1, 2))
    spin_echo_fig, spin_echo_zoom_fig = spin_echo_figs.subfigures(
        ncols=2, width_ratios=(1, 1), wspace=0.01
    )
    seq_fig, esr_fig = seq_esr_fig.subfigures(
        ncols=2, width_ratios=(0.6, 0.4), wspace=0.01
    )

    num_nvs = 10
    mosaic_layout = kpl.calc_mosaic_layout(num_nvs, num_rows=2)

    ### Sequence

    # NV-specific axes
    nrows = 6
    seq_axes_pack = seq_fig.subplots(
        nrows=nrows, sharex=True, sharey=True, height_ratios=[1, 1, 1, 0.25, 1, 1]
    )
    global_ax = seq_axes_pack[-1]

    # Global pulse axis
    seq_ax = seq_fig.add_subplot(111)
    seq_ax.set_ylabel(" ", rotation="horizontal", labelpad=40, loc="bottom")
    seq_ax.sharex(seq_axes_pack[0])
    # seq_ax.sharey(seq_axes_pack[0])

    for ax in [*seq_axes_pack, seq_ax]:
        ax.tick_params(
            which="both",
            top=False,
            bottom=False,
            left=False,
            right=False,
            labelbottom=False,
            labelleft=False,
        )
        ax.spines[["left", "right", "top"]].set_visible(False)

    seq_axes_pack[-3].spines[["bottom"]].set_visible(False)
    seq_ax.spines[["bottom"]].set_visible(False)
    seq_ax.patch.set_alpha(0)

    labels = [*[f"NV {ind}" for ind in range(3)], "...", "NV $\it{n}$", "Global"]
    for ind in range(nrows):
        ax = seq_axes_pack[ind]
        if ind == nrows - 3:
            ax.set_ylabel(labels[ind])
        else:
            ax.set_ylabel(labels[ind], rotation="horizontal", labelpad=50, loc="bottom")

    ax = seq_axes_pack[0]
    ax.set_xlim([0, 80])
    ax.set_ylim([0.1, 1.03])
    seq_ax.set_ylim([0.1, 1.01])

    # Annotations
    seq_axes_pack[0].set_title(" ")
    seq_ax.set_title(" ")
    seq_axes_pack[-1].set_xlabel(" ")
    seq_ax.set_xlabel(" ")
    seq_fig.text(0.1, 0.9, "Charge pol.")
    # seq_fig.text(0.4, 0.5, "Spin pol.", rotation=90)
    seq_fig.text(0.4, 0.1, "Spin pol.")
    # seq_fig.text(0.6, 0.3, "RF seq.", rotation=90)
    seq_fig.text(0.6, 0.1, "RF seq.")
    seq_fig.text(0.7, 0.9, "SCC")
    # seq_fig.text(0.9, 0.5, "Charge state\nreadout", horizontalalignment="center")
    seq_fig.text(0.9, 0.1, "Readout", horizontalalignment="center")

    row_skip_inds = [nrows - 3, nrows - 1]

    # Charge polarization
    start = 0
    stop = 0
    for ind in range(nrows):
        if ind in row_skip_inds:
            start += 2
            continue
        ax = seq_axes_pack[ind]
        start = stop + 4
        stop = start + 2
        kpl.plot_sequence(ax, [0, start, stop, 0], [0, 1, 0], color=kpl.KplColors.GREEN)

    # Spin polarization
    start = stop + 1
    stop = start + 11
    kpl.plot_sequence(global_ax, [0, start, stop, 0], [0, 1, 0], color="#d9d900")

    # Microwaves A
    # start = stop + 2
    # stop = start + 1
    # # kpl.plot_sequence(
    # # seq_ax, [0, start, stop, 0], [0, 1, 0], color=kpl.KplColors.BROWN
    # # )
    # start = stop + 1
    # stop = start + 1
    # # kpl.plot_sequence(
    # # seq_ax, [0, start, stop, 0], [0, 1, 0], color=kpl.KplColors.BROWN
    # # )
    start = stop + 1
    stop = start + 9
    kpl.plot_sequence(
        global_ax, [0, start, stop, 0], [0, 1, 0], color=kpl.KplColors.DARK_GRAY
    )

    # SCC
    for ind in range(nrows):
        if ind in row_skip_inds:
            start += 2
            continue
        ax = seq_axes_pack[ind]
        start = stop + 4
        stop = start + 1
        kpl.plot_sequence(ax, [0, start, stop, 0], [0, 1, 0], color=kpl.KplColors.RED)

    # Charge state readout
    start = stop + 1
    stop = 200
    kpl.plot_sequence(global_ax, [0, start, stop, 0], [0, 1, 0], color="#f5f556")

    ### ESR

    esr_axes_pack = esr_fig.subplot_mosaic(mosaic_layout, sharex=True, sharey=True)

    nv_list = esr_data["nv_list"]
    freqs = esr_data["freqs"]

    counts = np.array(esr_data["states"])
    sig_counts = counts[0]
    ref_counts = counts[1]

    avg_counts, avg_counts_ste, norms = widefield.process_counts(
        nv_list, sig_counts, ref_counts, threshold=False
    )
    norms = norms[0]

    resonance.create_fit_figure(
        nv_list,
        freqs,
        avg_counts,
        avg_counts_ste,
        norms,
        esr_axes_pack,
        mosaic_layout,
        no_legend=True,
    )

    ### Spin echo

    spin_echo_axes_pack = spin_echo_fig.subplot_mosaic(
        mosaic_layout, sharex=True, sharey=True
    )
    spin_echo.create_fit_figure(
        spin_echo_data, spin_echo_axes_pack, mosaic_layout, no_legend=True
    )

    zoom_range = [65, 87]
    # ax = spin_echo_axes_pack[mosaic_layout[0][0]]
    # ax.axvspan(*zoom_range, color=kpl.KplColors.LIGHT_GRAY, zorder=-11)
    for ax in spin_echo_axes_pack.values():
        ax.axvspan(*zoom_range, color=kpl.KplColors.LIGHT_GRAY, zorder=-11)

    spin_echo_zoom_axes_pack = spin_echo_zoom_fig.subplot_mosaic(
        mosaic_layout, sharex=True, sharey=True
    )
    spin_echo.create_fit_figure(
        spin_echo_data, spin_echo_zoom_axes_pack, mosaic_layout, no_legend=True
    )
    ax = spin_echo_zoom_axes_pack[mosaic_layout[0][0]]
    ax.set_xlim(zoom_range)

    # ### Adjustments

    # main_fig.get_layout_engine().set(w_pad=0, h_pad=0, hspace=0, wspace=0)

    # main_fig.text(0, 0.96, "(a)")
    # main_fig.text(w_factor - 0.04, 0.96, "(b)")
    # main_fig.text(w_factor - 0.04, 1 - h_factor + 0.05, "(c)")


if __name__ == "__main__":
    kpl.init_kplotlib()

    esr_data = dm.get_raw_data(file_id=1543601415736)
    spin_echo_data = dm.get_raw_data(file_id=1548381879624)

    main(esr_data, spin_echo_data)

    plt.show(block=True)
