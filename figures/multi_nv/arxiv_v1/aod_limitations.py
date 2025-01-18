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

from utils import common
from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import tool_belt as tb
from utils import widefield as widefield
from utils.constants import NVSig


def main():
    fig, ax = plt.subplots()
    beam_size_linspace = np.linspace(0, 1, 1000)

    # plot_vals = 0.59 * ((beam_size_linspace * (8 / 2.4) * 40) ** 2)
    plot_vals = 0.59 * ((beam_size_linspace * (8 / 2.9) * 45) ** 2)
    kpl.plot_line(ax, beam_size_linspace, plot_vals, label="AOD bandwidth")

    lifetimes = [5e-3, 3e-3, 1e-3]
    labels = [
        "Spin relaxation (bulk)",
        "Spin relaxation (shallow)",
        "Spin relaxation (nanodiamond)",
    ]
    for lifetime, label in zip(lifetimes, labels):
        plot_vals = lifetime / (beam_size_linspace * 10e-6)
        kpl.plot_line(ax, beam_size_linspace, plot_vals, label=label)

    ax.legend()
    ax.set_xlabel("Normalized beam size")
    ax.set_ylabel("Max number of NV centers")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 10000])


if __name__ == "__main__":
    kpl.init_kplotlib()

    main()

    plt.show(block=True)
