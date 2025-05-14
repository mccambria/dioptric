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
area = np.pi * 3**2
emccd_readout_time = 12e-3
# emccd_readout_time = 1e-3
qcmos_readout_time = 1e-3

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
    readout_noise = 100
    quantum_efficiency = 0.75
    em_gain = 5000
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
    readout_noise = 0.43
    quantum_efficiency = 0.55
    qubit_counts = quantum_efficiency * qubit_rate * exposure_time
    dark_counts = area * dark_rate * exposure_time
    mean = qubit_counts + dark_counts
    # return neg_bin(x, mean)

    # Accounting for readout noise
    sigma = np.sqrt(2 * mean)
    stop = np.round(mean + 4 * sigma) + 10
    integral_range = np.arange(0, stop)
    return np.sum(
        normal(x[:, np.newaxis], integral_range[np.newaxis, :], readout_noise)
        * poisson(integral_range[np.newaxis, :], mean),
        axis=1,
    )


def single_shot_snr(dist, qubit_rate_0, qubit_rate_1, exposure_time):
    w = 1 / 2
    mean_0 = qubit_rate_0 * exposure_time
    mean_1 = qubit_rate_1 * exposure_time
    start = 0 if dist == emccd else -5
    integral_vals = np.linspace(start, mean_1 + 4 * np.sqrt(mean_1) + 10, 10000)

    pdf_0 = dist(integral_vals, qubit_rate_0, exposure_time)
    pdf_1 = dist(integral_vals, qubit_rate_1, exposure_time)
    cdf_0 = np.cumsum(pdf_0)
    cdf_1 = np.cumsum(pdf_1)
    # Renormalize
    cdf_0 /= cdf_0[-1]
    cdf_1 /= cdf_1[-1]

    q0 = cdf_0
    p0 = 1 - q0
    q1 = cdf_1
    p1 = 1 - q1
    signal = w * (p1 - p0)
    noise = np.sqrt(
        w * p1 * (q1 + p1) + (1 - w) * p0 * (q0 + p0) - (w * p1 + (1 - w) * p0) ** 2
    )
    snrs = signal / noise
    snrs = np.nan_to_num(snrs, nan=np.nan, posinf=np.nan, neginf=np.nan)
    threshold = integral_vals[np.nanargmax(snrs)]
    snr = np.nanmax(snrs)

    # return threshold
    return snr


def calc_char_avg_time(meas_time, dist, qubit_rate_0, qubit_rate_1, exposure_time):
    readout_time = emccd_readout_time if dist == emccd else qcmos_readout_time
    snr = single_shot_snr(dist, qubit_rate_0, qubit_rate_1, exposure_time)
    # return snr
    return (1 / (snr**2)) * (meas_time + exposure_time + readout_time)


def optimize(meas_time, dist, qubit_rate_0, qubit_rate_1):
    exposure_times = np.linspace(0.0001, 0.1, 1000)
    # char_avg_times = []
    # for exposure_time in exposure_times:
    #     char_avg_time = calc_char_avg_time(
    #         meas_time, dist, qubit_rate_0, qubit_rate_1, exposure_time
    #     )
    #     char_avg_times.append(char_avg_time)

    calc_char_avg_time_sub = partial(
        calc_char_avg_time, meas_time, dist, qubit_rate_0, qubit_rate_1
    )
    # calc_char_avg_time_sub(0.01)
    with Pool() as p:
        char_avg_times = p.map(calc_char_avg_time_sub, exposure_times)

    exposure_time = exposure_times[np.argmin(char_avg_times)]
    char_avg_time = np.min(char_avg_times)

    # plt.plot(exposure_times, char_avg_times)
    # kpl.show(block=True)

    return (char_avg_time, exposure_time)


def main():
    qubit_rate_0 = 1.0e3 / 4
    qubit_rate_1 = 1.0e3

    num_meas_times = 100
    # num_meas_times = 3
    meas_times = np.logspace(-7, 0, num_meas_times)
    data = np.empty((num_meas_times, 2, 2))
    for ind, meas_time in enumerate(meas_times):
        for jnd in [0, 1]:
            dist = emccd if jnd == 0 else qcmos
            opti_vals = optimize(meas_time, dist, qubit_rate_0, qubit_rate_1)
            data[ind, jnd, :] = opti_vals

    fig, ax = plt.subplots()
    kpl.plot_line(ax, meas_times, data[:, 0, 0], label="EMCCD")
    kpl.plot_line(ax, meas_times, data[:, 1, 0], label="qCMOS")
    ax.set_xlabel("Integration time (s)")
    ax.set_xscale("log")
    ax.set_ylabel("Characteristic averaging time (s)")
    ax.legend()

    fig, ax = plt.subplots()
    kpl.plot_line(ax, meas_times, data[:, 0, 1], label="EMCCD")
    kpl.plot_line(ax, meas_times, data[:, 1, 1], label="qCMOS")
    ax.set_xlabel("Integration time (s)")
    ax.set_xscale("log")
    ax.set_ylabel("Optimal exposure time (s)")
    ax.legend()


def test():
    qubit_rate_0 = 1.0e3 / 4
    qubit_rate_1 = 1.0e3
    exposure_time = 20e-3

    fig, ax = plt.subplots()
    x_vals = np.linspace(0, 35, 10000)
    # x_vals = np.linspace(-2, 35, 10000)
    # # kpl.plot_line(ax, x_vals, poisson(x_vals, 10))
    # # kpl.plot_line(ax, x_vals, neg_bin(x_vals, 37.5))
    fn = emccd
    # fn = qcmos
    kpl.plot_line(
        ax, x_vals, fn(x_vals, qubit_rate_0, exposure_time), label="placeholder"
    )
    kpl.plot_line(
        ax, x_vals, fn(x_vals, qubit_rate_1, exposure_time), label="placeholder"
    )
    ax.set_xlabel(r"Readout value $X_{i}$", usetex=True)
    ax.set_ylabel("Probability density")
    legend = ax.legend()

    labels = [r"$\ket{0}$", r"$\ket{1}$"]
    for ind, text in enumerate(legend.get_texts()):
        text.set_text(labels[ind])
        text.set_usetex(True)
    ax.set_xlim(-2, 35)
    # plt.draw()

    # # x_vals = np.linspace(0.05, 0.2, 1000)
    # qubit_rate_0 = 1.0e3 / 4
    # qubit_rate_1 = 1.0e3
    # exposure_time = 0.02
    # # single_shot_snr(qcmos, qubit_rate_0, qubit_rate_1, exposure_time)
    # optimize(1e-6, emccd, qubit_rate_0, qubit_rate_1)
    # # calc_char_avg_time(1e-6, qcmos, qubit_rate_0, qubit_rate_1, 0.0002)

    # y_vals = [
    #     single_shot_snr(qcmos, qubit_rate_0, qubit_rate_1, x_val) for x_val in x_vals
    # ]
    # kpl.plot_line(ax, x_vals, y_vals)
    # y_vals = [
    #     single_shot_snr(emccd, qubit_rate_0, qubit_rate_1, x_val) for x_val in x_vals
    # ]
    # kpl.plot_line(ax, x_vals, y_vals)


if __name__ == "__main__":
    kpl.init_kplotlib()
    test()
    # main()
    kpl.show(block=True)
