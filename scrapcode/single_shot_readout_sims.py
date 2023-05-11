# -*- coding: utf-8 -*-
"""
Get the NV temp based on the ZFS, using numbers from: 'Temperature dependent 
energy level shifts of nitrogen-vacancy centers in diamond'

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


# endregion
# region Functions


def poisson(val, param):
    return (param**val) * np.exp(-param) / factorial(val)


def poisson_cum(val, param):
    # +1 in range to include the passed val
    return sum([poisson(ind, param) for ind in range(val + 1)])


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


# endregion

if __name__ == "__main__":
    print(1 - poisson_cum(15, 4))
    sys.exit()

    kpl.init_kplotlib()
    main()
    plt.show(block=True)
