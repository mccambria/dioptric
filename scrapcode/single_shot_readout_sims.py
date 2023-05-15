# -*- coding: utf-8 -*-
"""
Tests demonstrating ideas for single-shot readout

Created on Fri Mar  5 12:42:32 2021

@author: matth
"""


# region Import and constants

import numpy as np
from math import factorial
from utils import common
from majorroutines.pulsed_resonance import return_res_with_error
import majorroutines.pulsed_resonance as pesr
import utils.tool_belt as tool_belt
from utils.tool_belt import bose
import matplotlib.pyplot as plt
from utils import kplotlib as kpl
from pathos.multiprocessing import ProcessingPool
from utils.kplotlib import KplColors
from scipy.optimize import curve_fit
from scipy.optimize import minimize
import csv
import pandas as pd
import sys
from analysis import three_level_rabi
import figures.zfs_vs_t.thermal_expansion as thermal_expansion
import csv
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib as mpl
import matplotlib.legend_handler
from scipy.special import binom

count_rate = 1e6
lookback = 150e-9
collection = 0.01
P_ms_0 = 19 / 20
P_ms_1 = 1 / 2
# num_excs = round(count_rate * lookback / collection)
num_excs = 5


# endregion
# region Functions


def poisson(val, param):
    return (param**val) * np.exp(-param) / factorial(val)


def poisson_cum(val, param):
    # +1 in range to include the passed val
    return sum([poisson(ind, param) for ind in range(val + 1)])


def discrete_single(i, m, n, C, P):
    """_summary_

    Parameters
    ----------
    i : int
        Num photons actually collected
    m : int
        Total num photons emitted
    n : int
        Maximum possible number of photons emitted
    C : float
        Collection efficiency
    P : float
        Probability of radiative decay (vs ISC)

    Returns
    -------
    _type_
        _description_
    """
    exp = 1 if m < n else 0
    return binom(m, i) * C**i * (1 - C) ** (m - i) * P**m * (1 - P) ** exp


def discrete(n, m, C, P):
    summand = 0
    for i in np.linspace(n, m, (m - n + 1)):
        summand += discrete_single(i, n, m, C, P)
    return summand


def discrete_cum(n, m, C, P):
    return sum([discrete(ind, m, C, P) for ind in range(n + 1)])


def main():
    param_bright = 4
    param_dark = 0.7 * param_bright

    fig, ax = plt.subplots()
    max_photon_counts = 20
    photon_counts = np.linspace(0, max_photon_counts, max_photon_counts + 1, dtype=int)
    probs_bright = [poisson(val, param_bright) for val in photon_counts]
    probs_dark = [poisson(val, param_dark) for val in photon_counts]
    kpl.plot_points(ax, photon_counts, probs_bright, label=r"$m_{s}=0$")
    kpl.plot_points(ax, photon_counts, probs_dark, label=r"$m_{s}=\pm 1$")
    ax.set_xlabel("Photon count")
    ax.set_ylabel("Probability")
    ax.legend()

    fig, ax = plt.subplots()
    tp_rate = np.array([1 - poisson_cum(val, param_bright) for val in photon_counts])
    fp_rate = np.array([1 - poisson_cum(val, param_dark) for val in photon_counts])
    kpl.plot_points(ax, photon_counts, fp_rate / tp_rate)
    ax.set_xlabel(r"$n$ photons", usetex=True)
    ax.set_ylabel(r"False positive rate / true positive rate", usetex=True)


def main2():
    photon_counts = np.linspace(0, num_excs, num_excs + 1, dtype=int)

    # fig, ax = plt.subplots()
    # probs_ms_0 = [discrete(n, m, C, P_ms_0) for n in photon_counts]
    # probs_ms_1 = [discrete(n, m, C, P_ms_1) for n in photon_counts]
    # kpl.plot_points(ax, photon_counts, probs_ms_0, label=r"$m_{s}=0$")
    # kpl.plot_points(ax, photon_counts, probs_ms_1, label=r"$m_{s}=\pm 1$")
    # ax.set_xlabel("Photon count")
    # ax.set_ylabel("Probability")
    # ax.legend()

    # True positive / false positive rate
    fig, ax = plt.subplots()
    # Positive rate, that we get >= j photons
    tp_rates = []  # True positive
    fp_rates = []  # False positive
    # Thresholds
    for j in photon_counts:
        tp_rate = 0
        fp_rate = 0

        # Neutral charge state
        if j == 0:
            fp_rate += 0.3
            # fp_rate += 0.3 + 0.7 * 0.1
            # tp_rate = 0.7 * 0.9
        # Num emitted photons
        for k in np.linspace(j, num_excs, num_excs - j + 1, dtype=int):
            # Num collected photons
            for i in np.linspace(j, k, k - j + 1, dtype=int):
                # Start in ms=0
                val = 0.7 * 0.9 * discrete_single(i, k, num_excs, collection, P_ms_0)
                if k < num_excs:  # ISC
                    tp_rate += 0.5 * val
                    fp_rate += 0.5 * val
                else:  # No ISC
                    tp_rate += val
                # Start in ms=1
                val = 0.7 * 0.1 * discrete_single(i, k, num_excs, collection, P_ms_1)
                if k < num_excs:  # ISC
                    tp_rate += 0.5 * val
                    fp_rate += 0.5 * val
                else:  # No ISC
                    fp_rate += val

        tp_rates.append(tp_rate)
        fp_rates.append(fp_rate)

    p_rates = [tp_rates[ind] + fp_rates[ind] for ind in photon_counts]
    ratios = [tp_rates[ind] / p_rates[ind] for ind in photon_counts]
    kpl.plot_points(ax, photon_counts, ratios)
    ax.set_xlabel("Threshold (>= # photons)")
    ax.set_ylabel("Probability of true positive")

    # Time to positive
    fig, ax = plt.subplots()
    p_times = [lookback / p_rates[ind] for ind in photon_counts]
    photon_counts = np.linspace(0, num_excs, num_excs + 1, dtype=int)
    kpl.plot_points(ax, photon_counts, p_times)
    ax.set_xlabel("Threshold (>= # photons)")
    ax.set_ylabel("Time to threshold (s)")
    ax.set_yscale("log")


# endregion

if __name__ == "__main__":
    # # print(150e-9 / (1 - discrete_cum(2, 15, 0.005, 19 / 20)))
    # vals = [
    #     discrete_single(i, num_excs, num_excs, collection, P_ms_0)
    #     for i in np.linspace(3, num_excs, num_excs + 1)
    # ]
    # prob = 0.7 * 0.9 * sum(vals)
    # print(prob)
    # print(lookback / prob)
    # sys.exit()

    kpl.init_kplotlib()
    # main()
    main2()
    plt.show(block=True)
