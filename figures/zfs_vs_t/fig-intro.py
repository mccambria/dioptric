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

    setpoint_temps = ["", 350, 400, 450]  # In increasing temp order
    skip_lambda = (
        lambda point: point["Skip"]
        or point["Sample"] != "Wu"
        or point["Setpoint temp (K)"] not in setpoint_temps
        or point["NV"] != "nv11_zfs_vs_t"
    )
    data_points = get_data_points(skip_lambda)

    # Blue, green, yellow, red
    edgecolors = ["#4db449", "#f1aa30", "#fb2e18", "#8c564b"]
    facecolors = [kpl.lighten_color_hex(el) for el in edgecolors]

    narrow_figsize = (0.55 * kpl.figsize[0], kpl.figsize[1])
    fig, ax = plt.subplots(figsize=narrow_figsize)

    for ind in [3, 2, 1, 0]:

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

        fit_func = lambda f: 1 - three_level_rabi.coherent_line(
            f, *popt, uwave_pulse_dur
        )

        offset = 0.25 * ind
        # offset = 0
        kpl.plot_line(
            ax,
            freqs,
            offset + norm_avg_sig,
            color=edgecolor,
            # markerfacecolor=facecolor,
            label=f"{int(temp)} K",
            # size=kpl.Size.SMALL,
        )
        kpl.plot_line(
            ax,
            smooth_freqs,
            offset + fit_func(smooth_freqs),
            color=KplColors.DARK_GRAY,
            # color=facecolor,
        )
        # ax.legend(loc="upper right")
        ax.legend(handlelength=1.5, borderpad=0.3, borderaxespad=0.3, handletextpad=0.6)
        ax.set_xlabel("Frequency (GHz)")
        ax.set_ylabel("Normalized fluorescence")
        ax.tick_params(left=False, labelleft=False)
        # ax.get_yaxis().set_visible(False)


if __name__ == "__main__":

    kpl.init_kplotlib()

    main()

    plt.show(block=True)
