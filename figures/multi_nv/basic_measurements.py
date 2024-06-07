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

    w_factor = 0.4
    h_factor = 0.5
    figsize = kpl.double_figsize
    figsize[1] *= 1.1
    main_fig = plt.figure(figsize=figsize)
    seq_esr_fig, spin_echo_double_fig = main_fig.subfigures(
        ncols=2, width_ratios=(w_factor, 1 - w_factor)
    )
    seq_fig, esr_fig = seq_esr_fig.subfigures(
        nrows=2, height_ratios=(h_factor, 1 - h_factor)
    )
    spin_echo_fig, spin_echo_zoom_fig = spin_echo_double_fig.subfigures(
        nrows=2, height_ratios=(1, 1)
    )
    num_nvs = 10
    mosaic_layout = kpl.calc_mosaic_layout(num_nvs, num_rows=2)

    ### Sequence

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
    ax = spin_echo_axes_pack[mosaic_layout[0][0]]
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
