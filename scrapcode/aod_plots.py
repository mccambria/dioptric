# -*- coding: utf-8 -*-
"""
AOD spec plots

Created on July 14th, 2023

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


def flatness():
    # Define the data
    data = np.array(
        [
            [26.7, 40.4, 40.8, 30.0, 15.8],
            [21.9, 38.9, 47.8, 41.0, 23.1],
            [22.2, 39.0, 48.0, 40.5, 23.0],
            [25.7, 44.0, 51.3, 40.2, 21.4],
            [21.8, 32.6, 32.0, 21.7, 11.7],
        ]
    )

    # Define the x-axis and y-axis values
    x = np.array([55, 65, 75, 85, 95])
    y = np.array([55, 65, 75, 85, 95])

    # Create a figure and axes
    fig, ax = plt.subplots()

    # Plot the data
    im = kpl.imshow(ax, data, cbar_label="Optical power (uW)")

    # Set the tick labels
    ax.set_xticks(np.arange(len(x)))
    ax.set_yticks(np.arange(len(y)))
    ax.set_xticklabels(x)
    ax.set_yticklabels(y)

    # Rotate the tick labels and set their alignment
    # plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
    #         rotation_mode="anchor")

    # Add colorbar
    # cbar = ax.figure.colorbar(im, ax=ax)

    ax.set_xlabel("X frequency (MHz)")
    ax.set_ylabel("Y frequency (MHz)")


def optical_vs_rf_power():
    # Data
    rf_power = [
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,
        1.1,
        1.2,
        1.3,
        1.4,
        1.5,
        1.6,
        1.7,
        1.8,
        1.9,
        2.0,
    ]
    optical_power = [
        14.5,
        18.7,
        22.6,
        26.4,
        29.8,
        33.2,
        35.6,
        38.4,
        40.5,
        42.1,
        42.9,
        43.3,
        43.1,
        43.0,
        42.7,
        42.2,
    ]
    diffraction_efficiency = [
        0.078,
        0.101,
        0.122,
        0.143,
        0.161,
        0.179,
        0.192,
        0.208,
        0.219,
        0.228,
        0.232,
        0.234,
        0.233,
        0.232,
        0.231,
        0.228,
    ]

    # Create figure and axes
    fig, ax1 = plt.subplots()

    # Plot left y-axis (optical power)
    ax1.plot(rf_power, optical_power)
    ax1.set_xlabel("RF Power (W)")
    ax1.set_ylabel("Optical Power (uW)")
    ax1.tick_params("y")

    # Create twin axes for right y-axis (diffraction efficiency)
    ax2 = ax1.twinx()
    ax2.plot(rf_power, diffraction_efficiency)
    ax2.set_ylabel("Diffraction Efficiency")
    ax2.tick_params("y")
    ax2.set_yticks([0.08, 0.12, 0.16, 0.20, 0.24])

    # Set grid and title
    # ax1.grid(True)
    # plt.title('Optical Power and Diffraction Efficiency vs RF Power')


if __name__ == "__main__":
    kpl.init_kplotlib()
    # optical_vs_rf_power()
    flatness()
    plt.show(block=True)
