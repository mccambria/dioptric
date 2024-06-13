# -*- coding: utf-8 -*-
"""
Main text fig 3

Created on June 5th, 2024

@author: mccambria
"""

import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from majorroutines.widefield import base_routine
from utils import common
from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import tool_belt as tb
from utils import widefield as widefield
from utils.constants import NVSig

def main(data):
    
    figsize = kpl.figsize
    figsize[2] = 1.2*figsize[1]
    main_fig = plt.figure(figsize=figsize)
    
    seq_fig, data_fig = main_fig.subfigures(
        nrows=2, height_ratios=(0.2, 0.8), hspace=0.01
    )
    
    ### Seq
    
    seq_ax = axes_pack[0,0]

    global_alpha = 0.7

    # NV-specific axes
    nrows = 6
    seq_axes_pack = seq_fig.subplots(
        nrows=nrows,
        sharex=True,
        sharey=True,
        height_ratios=[1, 1, 1, 0.25, 1, 1],
        # hspace=0.005,
    )
    global_ax = seq_axes_pack[-1]

    # Global pulse axis
    seq_ax = seq_fig.add_subplot(111)
    seq_ax.set_ylabel(" ", rotation="horizontal", labelpad=40, loc="bottom")
    seq_ax.sharex(seq_axes_pack[0])
    # seq_ax.sharey(seq_axes_pack[0])
    global_ax = seq_ax

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
    global_ax.set_ylabel(" ", labelpad=50, loc="bottom")

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
    seq_fig.text(0.4, 0.3, "Spin pol.", horizontalalignment="center", rotation=90)
    # seq_fig.text(0.4, 0.1, "Spin pol.")
    seq_fig.text(0.6, 0.3, "RF seq.", horizontalalignment="center", rotation=90)
    # seq_fig.text(0.6, 0.1, "RF seq.")
    seq_fig.text(0.7, 0.9, "SCC")
    seq_fig.text(
        0.9, 0.3, "Charge state\nreadout", horizontalalignment="center", rotation=90
    )
    # seq_fig.text(0.9, 0.1, "Readout", horizontalalignment="center")

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
    kpl.plot_sequence(
        global_ax, [0, start, stop, 0], [0, 1, 0], color="#d9d900", alpha=global_alpha
    )

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
        global_ax,
        [0, start, stop, 0],
        [0, 1, 0],
        color=kpl.KplColors.DARK_GRAY,
        alpha=global_alpha,
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
    
    
    ### Data
    
    data_axes_pack = data_fig.subplots(2,2)


if __name__ == "__main__":
    kpl.init_kplotlib()

    data = dm.get_raw_data(file_id=)

    main(data)

    plt.show(block=True)