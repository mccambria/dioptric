# -*- coding: utf-8 -*-
"""
EMCCD vs qCMOS

Created on May 11th, 2023

@author: mccambria
"""


# region Import and constants

import csv
import sys
from functools import partial
from multiprocessing import Pool

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

inv_root_2_pi = 1 / np.sqrt(2 * np.pi)
area = 5
# area = 10
emccd_readout_time = 6e-3 * np.sqrt(area / 5)
# emccd_readout_time = 1e-3
qcmos_readout_time = 0.5e-3 * np.sqrt(area / 5)
# qcmos_readout_time = 5e-3
w_star = 1 / 2
p0p = 0.1
p1p = 0.6
qubit_rate_1 = 40 / (0.75 * 0.05)
qubit_rate_0 = qubit_rate_1 / 4

# endregion


def normal(x, mean, sigma):
    return (inv_root_2_pi / sigma) * np.exp(-(((x - mean) / sigma) ** 2) / 2)


def poisson(x, mean):
    return np.exp(xlogy(x, mean) - mean - gammaln(x + 1))


def neg_bin(x, mean):
    return np.exp(
        gammaln(x + mean) - gammaln(x + 1) - gammaln(mean) + xlogy(x + mean, 0.5)
    )


def emccd(x, qubit_rate, exposure_time):
    dark_rate = 0.0015
    clock_induced_charges = 0.0009
    quantum_efficiency = 0.75
    qubit_counts = quantum_efficiency * qubit_rate * exposure_time
    dark_counts = area * dark_rate * exposure_time
    clock_induced_counts = area * clock_induced_charges
    mean = qubit_counts + dark_counts + clock_induced_counts
    return neg_bin(x, mean)

    # Accounting for readout noise
    # sigma = np.sqrt(2 * mean)
    # start = np.round(mean - 4 * sigma)
    # stop = np.round(mean + 4 * sigma)
    # integral_range = np.arange(start, stop)
    # return np.sum(
    #     normal(x[:, np.newaxis], integral_range[np.newaxis, :], readout_noise / em_gain)
    #     * neg_bin(integral_range[np.newaxis, :], mean),
    #     axis=1,
    # )


def qcmos(x, qubit_rate, exposure_time):
    dark_rate = 0.006
    readout_noise = 0.43 * area
    # readout_noise = 0.3 * area
    quantum_efficiency = 0.55
    qubit_counts = quantum_efficiency * qubit_rate * exposure_time
    dark_counts = area * dark_rate * exposure_time
    mean = qubit_counts + dark_counts
    # return neg_bin(x, mean)

    # Accounting for readout noise
    sigma = np.sqrt(mean)
    stop = np.round(mean + 4 * sigma) + 10
    integral_range = np.arange(0, stop)
    return np.sum(
        normal(x[:, np.newaxis], integral_range[np.newaxis, :], readout_noise)
        * poisson(integral_range[np.newaxis, :], mean),
        axis=1,
    )


def measurement_noise(
    dist, qubit_rate_0, qubit_rate_1, exposure_time, ret_thresh=False
):
    mean_0 = qubit_rate_0 * exposure_time
    mean_1 = qubit_rate_1 * exposure_time
    start = 0 if dist == emccd else -15
    integral_vals, step = np.linspace(
        start, mean_1 + 4 * np.sqrt(mean_1) + 10, 10000, retstep=True
    )

    pdf_0 = dist(integral_vals, qubit_rate_0, exposure_time)
    pdf_1 = dist(integral_vals, qubit_rate_1, exposure_time)
    cdf_0 = np.cumsum(pdf_0) * step
    cdf_1 = np.cumsum(pdf_1) * step
    # Renormalize
    cdf_0 /= cdf_0[-1]
    cdf_1 /= cdf_1[-1]

    r0 = 1 - cdf_0
    r1 = cdf_1

    def calc_y(w):
        yp = p0p * (1 - w) + (1 - p1p) * w
        y = r0 * (1 - yp) + (1 - r1) * yp
        return y

    contrast = 1 - (1 - calc_y(1)) - calc_y(0)
    y = calc_y(w_star)
    meas_noises = np.sqrt(y * (1 - y)) / contrast
    meas_noises[contrast < 0.001] = np.nan  # Prevent 0/0

    opti_ind = np.nanargmin(meas_noises)
    threshold = integral_vals[opti_ind]
    meas_noise = meas_noises[opti_ind]

    if ret_thresh:
        return threshold
    return meas_noise


def calc_char_avg_time(inte_time, dist, qubit_rate_0, qubit_rate_1, exposure_time):
    readout_time = emccd_readout_time if dist == emccd else qcmos_readout_time
    shot_time = inte_time + exposure_time + readout_time
    meas_noise = measurement_noise(dist, qubit_rate_0, qubit_rate_1, exposure_time)
    return (meas_noise**2) * shot_time


def optimize(inte_time, dist, qubit_rate_0, qubit_rate_1):
    exposure_times = np.linspace(0.0001, 0.1, 1000)
    # exposure_times = np.linspace(0.015, 0.025, 1000)
    # char_avg_times = []
    # for exposure_time in exposure_times:
    #     char_avg_time = calc_char_avg_time(
    #         inte_time, dist, qubit_rate_0, qubit_rate_1, exposure_time
    #     )
    #     char_avg_times.append(char_avg_time)

    calc_char_avg_time_sub = partial(
        calc_char_avg_time, inte_time, dist, qubit_rate_0, qubit_rate_1
    )
    # val = calc_char_avg_time_sub(0.05)
    with Pool() as p:
        char_avg_times = p.map(calc_char_avg_time_sub, exposure_times)

    char_avg_times = np.array(char_avg_times)
    min_ind = np.argmin(char_avg_times)
    exposure_time = exposure_times[min_ind]
    char_avg_time = char_avg_times[min_ind]

    # Plot characteristic averaging time as a function of exposure time
    ax = plt.gca()
    if ax is None:
        fig, ax = plt.subplots()
    dist_name = "EMCCD" if dist is emccd else "qCMOS"
    kpl.plot_line(ax, exposure_times * 1000, char_avg_times * 1000, label=dist_name)
    ax.set_xlabel(r"Exposure time $t_{\mathrm{e}}$ (ms)", usetex=True)
    ax.set_ylabel(r"Char. averaging time $T^{*}$ (ms)", usetex=True)
    # kpl.show(block=True)

    return (char_avg_time, exposure_time)


def main():
    num_inte_times = 100
    # num_inte_times = 10
    # num_inte_times = 3
    inte_times = np.logspace(-7, 0, num_inte_times)
    data = np.empty((num_inte_times, 2, 2))
    for ind, inte_time in enumerate(inte_times):
        for jnd in [0, 1]:
            dist = emccd if jnd == 0 else qcmos
            # data[ind, jnd, :] = (0, 1)
            opti_vals = optimize(inte_time, dist, qubit_rate_0, qubit_rate_1)
            data[ind, jnd, :] = opti_vals

    figsize = kpl.figsize
    double_figsize = figsize.copy()
    double_figsize[0] *= 2
    fig, axes_pack = plt.subplots(1, 2, figsize=double_figsize)
    ax = axes_pack[0]
    kpl.plot_line(ax, inte_times * 1000, data[:, 0, 0] * 1000, label="EMCCD")
    kpl.plot_line(ax, inte_times * 1000, data[:, 1, 0] * 1000, label="qCMOS")
    ax.set_xlabel(r"Integration time $t_{\mathrm{i}}$ (ms)", usetex=True)
    ax.set_ylabel(r"Char. averaging time $T^{*}$ (ms)", usetex=True)
    ax.set_xscale("log")
    ax.set_yscale("log")
    # ax.set_ylim(10, None)
    ax.legend()

    # fig, ax = plt.subplots(figsize=figsize)
    ax = axes_pack[1]
    kpl.plot_line(ax, inte_times * 1000, data[:, 0, 1] * 1000, label="EMCCD")
    kpl.plot_line(ax, inte_times * 1000, data[:, 1, 1] * 1000, label="qCMOS")
    ax.set_xlabel(r"Integration time $t_{\mathrm{i}}$ (ms)", usetex=True)
    ax.set_xscale("log")
    ax.set_ylabel(r"Optimal exposure time $t_{\mathrm{e}}$ (ms)", usetex=True)
    ax.legend()

    fig.text(0.002, 0.95, "(a)")
    fig.text(0.502, 0.95, "(b)")


def plot_dists():
    figsize = kpl.figsize
    figsize[0] *= 2
    fig, axes_pack = plt.subplots(1, 2, figsize=figsize)

    dists = [emccd, qcmos]
    dist_names = ["EMCCD", "qCMOS"]
    state_names = [r"$\ket{0}$", r"$\ket{1}$"]
    qubit_rates = [qubit_rate_0, qubit_rate_1]
    exp_times = [13.7, 16.4]  # ms
    # exp_times = [40, 40]  # ms
    # starts = [0, -14]
    starts = [0, -8]

    for ind in range(2):
        ax = axes_pack[ind]
        dist = dists[ind]
        exp_time = exp_times[ind] / 1000
        dist_name = dist_names[ind]
        start = starts[ind]

        threshold = measurement_noise(
            dist, qubit_rate_0, qubit_rate_1, exp_time, ret_thresh=True
        )

        x_vals, step = np.linspace(start, 30, 1000, retstep=True)
        # x_vals, step = np.linspace(start, 65, 1000, retstep=True)
        for jnd in range(2):
            qubit_rate = qubit_rates[jnd]
            state_name = state_names[jnd]
            dist_vals = dist(x_vals, qubit_rate, exp_time)
            dist_vals /= np.cumsum(dist_vals)[-1] * step
            kpl.plot_line(ax, x_vals, dist_vals, label=f"{dist_name}, {state_name}")
            print(np.cumsum(dist_vals[x_vals < threshold])[-1] * step)

        ax.axvline(threshold, color=kpl.KplColors.DARK_GRAY)

        ax.set_xlabel("Readout value")
        ax.set_ylabel("Probability density")

        plt.rcParams["text.usetex"] = True
        ax.legend()
        plt.rcParams["text.usetex"] = False

    fig.text(0.002, 0.95, "(a)")
    fig.text(0.502, 0.95, "(b)")


if __name__ == "__main__":
    kpl.init_kplotlib()
    # main()
    optimize(10e-6, emccd, qubit_rate_0, qubit_rate_1)
    optimize(10e-6, qcmos, qubit_rate_0, qubit_rate_1)
    ax = plt.gca()
    ax.legend()
    # plot_dists()
    kpl.show(block=True)
