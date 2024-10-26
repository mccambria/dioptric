# -*- coding: utf-8 -*-
"""
Main text fig 1

Created on June 5th, 2024

@author: mccambria
"""

import io
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.special import comb

from majorroutines.widefield.charge_monitor import process_check_readout_fidelity
from majorroutines.widefield.charge_state_histograms import create_histogram
from utils import common
from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import tool_belt as tb
from utils import widefield as widefield
from utils.constants import NVSig


def time_to_snr_1_inde(num_nvs, serial_or_parallel, interrogation_time, params):
    snr, tos, top = params
    if serial_or_parallel:
        tis = interrogation_time
        tip = 0
    else:
        tis = 0
        tip = interrogation_time
    num_shots = (1 / snr) ** 2
    return num_shots * (num_nvs * (tos + tis) + top + tip)


def time_to_snr_1_corr(num_nvs, serial_or_parallel, interrogation_time, params):
    snr, tos, top = params
    num_shots = (1 / snr) ** 4
    if serial_or_parallel:
        tis = interrogation_time
        tip = 0
        return num_shots * comb(num_nvs, 2) * (tos + tis)
    else:
        tis = 0
        tip = interrogation_time
        return num_shots * (num_nvs * (tos + tis) + top + tip)


def main():
    params = [
        [0.03, 0.3e-6, 0],
        [0.02, 0, 0.3e-6],
        [0.25, 5e-3, 0],
        [0.25, 21e-6, 62e-3],
        [0.25, 21e-6, 17e-3],
    ]
    labels = [
        "Conv., serial",
        "Conv., parallel",
        "SCC, serial",
        "SCC, parallel",
        "SCC (proj.), parallel",
    ]
    serial_or_parallels = [
        True,
        False,
        True,
        False,
        False,
    ]

    fns = (time_to_snr_1_inde, time_to_snr_1_corr)
    min_num_nvs = [1, 2]
    for ind in range(2):
        fn = fns[ind]

        figsize = kpl.double_figsize
        fig, axes_pack = plt.subplots(1, 2, figsize=figsize)

        # Vs number of NVs

        ax = axes_pack[0]

        num_methods = len(labels)
        max_num_nvs = 1000
        integration_time = 100e-6
        nv_linspace = np.arange(min_num_nvs[ind], max_num_nvs + 1, 1)
        for ind in range(num_methods):
            plot_vals = fn(
                nv_linspace, serial_or_parallels[ind], integration_time, params[ind]
            )
            kpl.plot_line(ax, nv_linspace, plot_vals, label=labels[ind])
        # ax.legend()
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Number of NVs")
        ax.set_ylabel("Time to unit SNR (s)")

        ax.legend(loc=kpl.Loc.UPPER_LEFT, prop={"size": 16})

        # Vs integration time

        ax = axes_pack[1]

        num_methods = len(labels)
        num_nvs = 100
        integration_times_linspace = np.logspace(-8, -1, 1000)
        for ind in range(num_methods):
            plot_vals = fn(
                num_nvs,
                serial_or_parallels[ind],
                integration_times_linspace,
                params[ind],
            )
            kpl.plot_line(ax, integration_times_linspace, plot_vals, label=labels[ind])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Interrogation time (s)")
        ax.set_ylabel("Time to unit SNR (s)")
        ax.set_xticks(np.logspace(-8, -1, 8))


if __name__ == "__main__":
    kpl.init_kplotlib()

    main()

    plt.show(block=True)
