# -*- coding: utf-8 -*-
"""
Histogram plots for widefield charge state ssr

Created on November 14th, 2023

@author: mccambria
"""


import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

import majorroutines.optimize as optimize
from majorroutines.widefield.optimize import optimize_pixel_with_img_array
from utils import common
from utils import kplotlib as kpl
from utils import positioning as pos
from utils import tool_belt as tb
from utils import widefield as widefield_utils
from utils.constants import LaserKey
from utils.kplotlib import HistType


def many_cond_init_cdf(num_attempts, init_prob, num_nvs):
    return (1 - (1 - init_prob) ** (num_attempts)) ** num_nvs


def many_cond_init_pdf(num_attempts, init_prob, num_nvs):
    return many_cond_init_cdf(num_attempts, init_prob, num_nvs) - many_cond_init_cdf(
        num_attempts - 1, init_prob, num_nvs
    )


def many_cond_init_expected(init_prob, num_nvs):
    max_num_attempts = 1000
    test_vals = np.linspace(1, max_num_attempts, max_num_attempts)
    # many_cond_init_pdf(test_vals, init_prob, num_nvs)
    # return np.dot(test_vals, many_cond_init_pdf(test_vals, init_prob, num_nvs))
    return 1 + np.sum(1 - (1 - (1 - init_prob) ** (test_vals)) ** (num_nvs))


def many_cond_init_plot():
    init_prob = 0.65
    max_num_nvs = 1000
    fig, ax = plt.subplots()

    plot_x_vals = np.linspace(1, max_num_nvs, max_num_nvs)
    plot_y_vals = [
        many_cond_init_expected(init_prob, num_nvs) for num_nvs in plot_x_vals
    ]
    kpl.plot_line(ax, plot_x_vals, plot_y_vals)
    ax.set_xlabel("Number of NVs")
    ax.set_ylabel("Expected shots until success")
    kpl.plot_line(ax, plot_x_vals, 0.88 * np.log(plot_x_vals) + plot_y_vals[0])

    # plot_x_vals = np.linspace(1, 20, 20)
    # plot_y_vals = many_cond_init_cdf(1000, init_prob, plot_x_vals)
    # kpl.plot_line(ax, plot_x_vals, plot_y_vals)


if __name__ == "__main__":
    kpl.init_kplotlib()

    many_cond_init_plot()

    plt.show(block=True)


# sum_m m [s(m) - s(m-1)]
# 1 (s(1) -s(0)) + 2 (s(2) -s(1))+ 3 (s(3) -s(2)) + ...
# -s(0) +ms(m) - sum_m s(m)
