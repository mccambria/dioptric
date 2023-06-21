# -*- coding: utf-8 -*-
"""
Model decsription figure for zfs vs t paper

Created on March 28th, 2023

@author: mccambria
"""


# region Import and constants

import numpy as np
from utils import common
import utils.tool_belt as tool_belt
from utils.tool_belt import bose
import matplotlib.pyplot as plt
import matplotlib as mpl
from utils import kplotlib as kpl
from utils.kplotlib import KplColors
from scipy.optimize import curve_fit
import csv
import pandas as pd
import sys


# endregion


def bose(x):
    # return x**2
    return 1 / (np.exp(1 / x) - 1)


def taylor(x, *args):
    ret_val = 0
    for ind in range(len(args)):
        ret_val += args[ind] * x**ind
    return ret_val


def main():
    fig, ax = plt.subplots()

    points = np.linspace(0, 2, 1000)
    points = points[1:]
    popt, pcov = curve_fit(
        taylor, points, bose(points), p0=(0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
    )
    print(popt)

    kpl.plot_line(ax, points, bose(points), label="Bose")
    kpl.plot_line(ax, points, taylor(points, *popt), label="Taylor")
    ax.legend()


if __name__ == "__main__":
    kpl.init_kplotlib()

    main()

    plt.show(block=True)
