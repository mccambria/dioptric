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

a = np.pi / (500e-9)
z_tot = 5
w_a = 1e-3


# endregion


def waist(z_a):
    return np.sqrt(w_a**2 + np.sqrt(w_a**4 - (4 * z_a**2) / a**2)) / np.sqrt(2)


def width(z_a):
    z_b = z_tot - z_a
    w = waist(z_a)
    z_R = a * w**2
    return w * np.sqrt(1 + (z_b / z_R) ** 2)


def width2(z, z_a=3):
    w = waist(z_a)
    z_R = a * w**2
    return w * np.sqrt(1 + ((z - z_a) / z_R) ** 2)


# def taylor():
#     fixed_width = 3e-3
#     waist =


def main():
    fig, ax = plt.subplots()

    points = np.linspace(-10, 10, 1000)

    # kpl.plot_line(ax, points, waist(points))
    # kpl.plot_line(ax, points, width(points))
    kpl.plot_line(ax, points, width2(points, z_a=3))
    kpl.plot_line(ax, points, width2(points, z_a=2.5))
    kpl.plot_line(ax, points, width2(points, z_a=2))
    kpl.plot_line(ax, points, width2(points, z_a=0))
    # kpl.plot_line(ax, width(points), a * waist(points) ** 2)
    # ax.legend()


if __name__ == "__main__":
    kpl.init_kplotlib()

    main()

    plt.show(block=True)
