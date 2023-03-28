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
from scipy.optimize import curve_fit
import csv
import pandas as pd
import sys
from analysis import three_level_rabi
from figures.zfs_vs_t.zfs_vs_t_main import get_data_points


# endregion


def main():

    fig_files = []  # In increasing temp order
    skip_lambda = lambda point: point["ZFS file"] not in fig_files
    data_points = get_data_points(skip_lambda)

    # Blue, green, yellow, red
    edgecolors = ["#3285c1", "#4db449", "#f1aa30", "#f2e18"]
    facecolors = [kpl.lighten_color_hex(el) for el in edgecolors]

    fig, ax = plt.subplots()

    for ind in range(4):

        fig_file = fig_files[ind]
        edgecolor = edgecolors[ind]
        facecolor = facecolors[ind]

        popt = []

        data = tool_belt.get_raw_data(fig_file)
        freq_center = data["freq_center"]
        freq_range = data["freq_range"]
        num_steps = data["num_steps"]
        freqs = pesr.calculate_freqs(freq_center, freq_range, num_steps)
        smooth_freqs = pesr.calculate_freqs(freq_center, freq_range, 100)

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

        fit_func = lambda f: three_level_rabi.coherent_line(f, *popt, uwave_pulse_dur)

        kpl.plot_line(ax, freqs, (0.5 * ind) + norm_avg_sig, color=edgecolor)
        kpl.plot_line(
            ax,
            smooth_freqs,
            (0.5 * ind) + fit_func(smooth_freqs),
            color=KplColors.MEDIUM_GRAY,
        )


if __name__ == "__main__":

    kpl.init_kplotlib()

    main()

    plt.show(block=True)
