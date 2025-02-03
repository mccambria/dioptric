# -*- coding: utf-8 -*-
"""
Model decsription figure for zfs vs t paper

Created on March 28th, 2023

@author: mccambria
"""


# region Import and constants

import csv
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

import utils.tool_belt as tool_belt
from utils import common
from utils import kplotlib as kpl
from utils.kplotlib import KplColors
from utils.tool_belt import bose

# endregion


def fano_lorentzian(x, b):
    term1 = 1 / (x**2 + (1 / 2) ** 2)
    term2 = b * x / (x**2 + (1 / 2) ** 2)
    return term1 + term2


def main():
    fig, ax = plt.subplots()

    # x_linspace = np.linspace(-0.1, 0.1, 1000)
    x_linspace = np.linspace(-5, 5, 1000)
    for b in [0, 0.1, 0.2]:
        vals = fano_lorentzian(x_linspace, b)
        kpl.plot_line(ax, x_linspace, vals, label=b)

    ax.legend(title=r"$b/\Gamma$")  # , loc="lower right")
    ax.set_xlabel(r"$(\omega-\omega_{0}) / \Gamma$")
    ax.set_ylabel(r"$P(\omega)$")
    ax.set_title(r"Fano-Lorentzian lineshape")
    text = r"$C/\Gamma^{2}=1$"
    kpl.anchored_text(ax, text, loc=kpl.Loc.UPPER_LEFT)


if __name__ == "__main__":
    kpl.init_kplotlib()

    main()

    plt.show(block=True)
