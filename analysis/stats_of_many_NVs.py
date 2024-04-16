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
from scipy.optimize import curve_fit
from scipy.special import binom

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


def generic_log(x, x_offset, y_offset, scaling):
    return scaling * np.log(x + x_offset) + y_offset


# def generic_log(x, y_offset, scaling):
#     return scaling * np.log(x) + y_offset


# def generic_log(x, x_offset, y_offset):
#     scaling = 1 / 0.5
#     return scaling * np.log(x + x_offset) + y_offset


def many_cond_init_plot():
    init_prob = 0.65
    q = 1 - init_prob
    max_num_nvs = 100
    fig, ax = plt.subplots()

    plot_x_vals = np.linspace(1, max_num_nvs, max_num_nvs)
    plot_y_vals = [
        many_cond_init_expected(init_prob, num_nvs) for num_nvs in plot_x_vals
    ]
    kpl.plot_line(ax, plot_x_vals, plot_y_vals)
    ax.set_xlabel("Number of NVs")
    ax.set_ylabel("Expected shots until success")
    # popt, pcov = curve_fit(
    #     generic_log, plot_x_vals, plot_y_vals, p0=(0, 1 / init_prob, 1)
    # )
    # print(popt)
    # kpl.plot_line(ax, plot_x_vals, generic_log(plot_x_vals, *popt))

    plot_y_vals = []
    for num_nvs in np.linspace(1, max_num_nvs, max_num_nvs, dtype=int):
        k_vals = np.linspace(1, num_nvs, num_nvs)
        y_vals = 1 - np.sum(
            binom(num_nvs, k_vals) * (-1) ** k_vals * (q**k_vals) / (1 - q**k_vals)
        )
        plot_y_vals.append(y_vals)
    kpl.plot_line(ax, plot_x_vals, plot_y_vals)

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
