# -*- coding: utf-8 -*-
"""
EMCCD vs qCMOS

Created on May 11th, 2023

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
from scipy.special import gammaln, xlogy

import utils.tool_belt as tool_belt
from utils import common
from utils import kplotlib as kpl
from utils.kplotlib import KplColors
from utils.tool_belt import bose

target_sensitivity = 1e-15
delta_ms = 2
gyro = 1.76e11
density = 1e24
T2_star = 30e-6
# T2_star = 150e-6
# T2_star = 10e-3
interrogation_time = 0.5 * T2_star
# interrogation_time = 0.6 * T2_star
contrast = 0.3
counts_per_shot_per_nv = 100e3 * 100e-9
overhead_time = 1e-6
# overhead_time = 1e-3
# overhead_time = 10e-6
total_time = interrogation_time + overhead_time

# endregion


def sensitivity(volume):
    fundamental = (1 / delta_ms) * (1 / gyro)
    num_nvs = density * volume
    dephasing = 1 / np.exp(-interrogation_time / T2_star)
    readout_noise = np.sqrt(1 + (1 / (contrast**2 * counts_per_shot_per_nv)))
    return (
        fundamental
        * (1 / np.sqrt(num_nvs))
        * dephasing
        * readout_noise
        * (np.sqrt(total_time) / interrogation_time)
    )


def volume():
    fundamental = (1 / delta_ms) * (1 / gyro)
    dephasing = 1 / np.exp(-interrogation_time / T2_star)
    readout_noise = np.sqrt(1 + (1 / (contrast**2 * counts_per_shot_per_nv)))
    readout_noise = 4
    return (
        fundamental
        * (1 / np.sqrt(density))
        # * (1 / density)
        * dephasing
        * readout_noise
        * (np.sqrt(total_time) / interrogation_time)
        / target_sensitivity
    ) ** 2
    # )


if __name__ == "__main__":
    # volume = 0.017
    # print(sensitivity(volume))

    print(volume() * 1e9)
    # print(volume() * 1e18)
    # print(volume())
