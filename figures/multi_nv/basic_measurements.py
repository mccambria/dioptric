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

from majorroutines.widefield import resonance
from utils import common
from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import tool_belt as tb
from utils import widefield as widefield
from utils.constants import NVSig


def main(esr_data, spin_echo_data):
    ### Setup

    num_nvs = 10
    sub_layout = kpl.calc_mosaic_layout(num_nvs, num_rows=2)
    layout = [
        ["sequence", "spin_echo"],
        [sub_layout, "spin_echo"],
    ]
    w_factor = 0.3
    h_factor = 0.4
    figsize = kpl.double_figsize
    # figsize[1] *= 1.1
    main_fig, axes_pack = plt.subplot_mosaic(
        layout,
        figsize=figsize,
        width_ratios=(w_factor, 1 - w_factor),
        height_ratios=(h_factor, 1 - h_factor),
    )
    esr_axes_pack = {}
    esr_ax = axes_pack[sub_layout[0][0]]
    for val in np.array(sub_layout).flatten():
        ax = axes_pack[val]
        esr_axes_pack[val] = ax
        if ax != esr_ax:
            ax.sharex(esr_ax)
            ax.sharey(esr_ax)

    ### Sequence

    ### ESR

    nv_list = esr_data["nv_list"]
    freqs = esr_data["freqs"]

    counts = np.array(esr_data["states"])
    sig_counts = counts[0]
    ref_counts = counts[1]

    avg_counts, avg_counts_ste, norms = widefield.process_counts(
        nv_list, sig_counts, ref_counts, threshold=False
    )
    norms = norms[0]

    fit_fig = resonance.create_fit_figure(
        nv_list, freqs, avg_counts, avg_counts_ste, norms, esr_axes_pack
    )

    ### Spin echo

    ax = axes_pack["spin_echo"]

    ### Adjustments

    main_fig.get_layout_engine().set(w_pad=0, h_pad=0, hspace=0, wspace=0)

    main_fig.text(0, 0.96, "(a)")
    main_fig.text(w_factor - 0.04, 0.96, "(b)")
    main_fig.text(w_factor - 0.04, 1 - h_factor + 0.05, "(c)")


if __name__ == "__main__":
    kpl.init_kplotlib()

    esr_data = dm.get_raw_data(file_id=1543601415736)
    spin_echo_data = dm.get_raw_data(file_id=1548381879624)

    main(esr_data, spin_echo_data)

    plt.show(block=True)
