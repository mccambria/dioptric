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

from majorroutines.widefield.charge_monitor import process_check_readout_fidelity
from majorroutines.widefield.charge_state_histograms import create_histogram
from utils import common
from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import tool_belt as tb
from utils import widefield as widefield
from utils.constants import NVSig


def time_to_snr_1(num_nvs, serial_or_parallel, interrogation_time, params):
    snr, tos, top = params
    if serial_or_parallel:
        tis = interrogation_time
        tip = 0
    else:
        tis = 0
        tip = interrogation_time
    num_shots = (1 / snr) ** 2
    return num_shots * (num_nvs * (tos + tis) + top + tip)


def main():
    params = [
        [0.03, 0.3e-6, 0],
        [0.02, 0, 0.3e-6],
        [0.25, 62e-3, 0],
        [0.25, 21e-6, 62e-3],
        [0.25, 21e-6, 17e-3],
    ]
    labels = [
        "Standard, serial",
        "Standard, parallel",
        "SCC, serial",
        "SCC, parallel",
        "SCC, parallel, projected",
    ]
    serial_or_parallels = [
        True,
        False,
        True,
        False,
        False,
    ]

    fig, ax = plt.subplots()
    num_methods = len(labels)
    max_num_nvs = 1000
    nv_linspace = np.arange(1, max_num_nvs + 1, 1)
    for ind in range(num_methods):
        plot_vals = time_to_snr_1(
            nv_linspace, serial_or_parallels[ind], 100e-6, params[ind]
        )
        kpl.plot_line(ax, nv_linspace, plot_vals, label=labels[ind])
    ax.legend()
    ax.set_yscale("log")
    ax.set_xlabel("Number of NVs")
    ax.set_ylabel("Time to unity SNR (s)")


if __name__ == "__main__":
    kpl.init_kplotlib()

    main()

    plt.show(block=True)
