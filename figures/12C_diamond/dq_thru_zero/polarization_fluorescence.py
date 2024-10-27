# -*- coding: utf-8 -*-
"""
Created on Jan 3 2023

@author: agardill
"""


import time
from random import shuffle

import labrad
import matplotlib.pyplot as plt
import numpy
from numpy import pi
from numpy.linalg import eigvals
from scipy.optimize import curve_fit

import majorroutines.targeting as targeting
import utils.kplotlib as kpl
import utils.tool_belt as tool_belt
from utils.kplotlib import KplColors
from utils.tool_belt import NormStyle, States

pi = numpy.pi


def plot_count_rate(angles, count_rate, title, do_fit=True):
    kpl.init_kplotlib()

    angles_rad = numpy.array(angles) * numpy.pi / 180

    angles_lin = numpy.linspace(angles[0], angles[-1], 100)
    angles_lin_rad = angles_lin * numpy.pi / 180

    if do_fit:
        fit_func = lambda t, amp, offset, freq, phase: tool_belt.sin_phase(
            t, amp, offset, freq, phase
        )
        init_params = [5, 24, 4, 1]
        popt, pcov = curve_fit(
            fit_func,
            angles_rad,
            count_rate,
            p0=init_params,
        )

        max_angle = (pi / 2 - popt[3]) / popt[2]
        if (pi / 2 - popt[3]) / popt[2] < 0:
            max_angle = (pi / 2 - popt[3]) / popt[2] + 2 * pi / popt[2]

    print(popt)
    print(max_angle * 180 / pi)
    # Plot setup
    fig, ax = plt.subplots(1, 1)
    ax.set_xlabel("Waveplate angle (deg)")
    ax.set_ylabel("Count rate (kcps)")
    ax.set_title(title)

    # Plotting
    kpl.plot_points(ax, angles, count_rate, color=KplColors.BLUE)

    if do_fit:
        kpl.plot_line(
            ax,
            angles_lin,
            fit_func(angles_lin_rad, *popt),
            label="fit",
            color=KplColors.RED,
        )

        base_text = "Max counts at {:.1f} deg"
        size = kpl.Size.SMALL
        text = base_text.format(max_angle * 180 / pi)
        kpl.anchored_text(ax, text, kpl.Loc.LOWER_RIGHT, size=size)

    ax.legend()


angles = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]


count_rate = [43.4, 39.4, 34.2, 30.3, 29.3, 38.2, 45, 48.8, 49, 48.5, 42.5, 33.1, 28.7]
title = "NV1"
plot_count_rate(angles, count_rate, title)


count_rate = [
    40.8,
    37.8,
    30.8,
    24.8,
    29.3,
    34.2,
    42.4,
    45.8,
    45.5,
    42.9,
    40.7,
    32.9,
    27,
]
title = "NV4"
plot_count_rate(angles, count_rate, title)

count_rate = [37.7, 35.1, 28.7, 30.2, 32.4, 33.2, 36, 36, 37.7, 38.2, 35.3, 32.9, 28.1]
title = "NV5"
plot_count_rate(angles, count_rate, title)

count_rate = [45.7, 38.6, 30.4, 29.8, 30, 35.6, 43.5, 45.5, 48.4, 45, 40.6, 33.4, 26.7]
title = "NV6"
plot_count_rate(angles, count_rate, title)

count_rate = [
    25.4,
    29.5,
    30.6,
    30.7,
    30.2,
    30.4,
    26.4,
    23.7,
    22.9,
    25.7,
    29.4,
    31.4,
    32.8,
]
title = "NV7"
plot_count_rate(angles, count_rate, title)

count_rate = [
    20.3,
    25.2,
    28.5,
    28,
    27.6,
    26.7,
    22.9,
    18,
    19.8,
    22.8,
    26.1,
    28.8,
    29,
]
title = "NV8"
plot_count_rate(angles, count_rate, title)

count_rate = [
    43.7,
    40.8,
    33.3,
    27.6,
    29.9,
    39.9,
    42.8,
    50.6,
    48.5,
    47.8,
    40.5,
    35.3,
    28.1,
]
title = "NV9"
plot_count_rate(angles, count_rate, title)
