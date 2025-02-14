# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 13:43:15 2023

@author: kolkowitz
"""


import copy
import json
import os
import time
from random import shuffle

import labrad
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

import majorroutines.targeting as targeting
import utils.kplotlib as kpl
import utils.tool_belt as tool_belt
from utils.kplotlib import KplColors, Size
from utils.tool_belt import States


def gaussian(freq, constrast, sigma, center):
    return constrast * np.exp(-((freq - center) ** 2) / (2 * (sigma**2)))


def tri_gaus(freqs, center, hyperfine, a1, s1, a2, s2, a3, s3):
    c2 = center + hyperfine
    c3 = center - hyperfine
    gauss1 = gaussian(freqs, a1, s1, center)
    gauss2 = gaussian(freqs, a2, s2, c2)
    gauss3 = gaussian(freqs, a3, s3, c3)
    return 1.0 - gauss1 - gauss2 - gauss3


def calculate_freqs(freq_range, freq_center, num_steps):
    half_freq_range = freq_range / 2
    freq_low = freq_center - half_freq_range
    freq_high = freq_center + half_freq_range
    return np.linspace(freq_low, freq_high, num_steps)


def fit_resonance(
    freq_range,
    freq_center,
    num_steps,
    norm_avg_sig,
    narrow_fit,
    norm_avg_sig_ste=None,
    ref_counts=None,
):
    hyperfine = 2.189288 * 1e-3
    fit_func = lambda freqs, a1, s1, a2, s2, a3, s3: tri_gaus(
        freqs, freq_center, hyperfine, a1, s1, a2, s2, a3, s3
    )

    # hyperfine = 2.189288*1e-3
    # fit_func = lambda freqs, center1, center2,a1, s1, a2, s2, a3, s3,a4, s4, a5, s5, a6, s6: six_gaus(freqs,center1, center2, hyperfine, a1, s1, a2, s2, a3, s3,
    #                                                                    a4, s4, a5, s5, a6, s6)

    if narrow_fit == True:
        guess_params = [
            0.1,
            0.00005,
            0.1,
            0.00005,
            0.1,
            0.00003,
        ]
    else:
        guess_params = [
            0.1,
            0.0001,
            0.1,
            0.0001,
            0.1,
            0.0001,
        ]

    freqs = calculate_freqs(freq_range, freq_center, num_steps)

    popt, pcov = curve_fit(
        fit_func,
        freqs,
        norm_avg_sig,
        p0=guess_params,
    )
    # popt=guess_params
    # print(popt)
    # print('{:.5f}'.format(popt[0]))
    # print(popt[1])
    # print('{:.5f}'.format(np.sqrt(pcov[0][0])))

    return fit_func, popt, pcov


def do_plot(rabi_file, pesr_file_list, act_freq_center):
    rabi_file_name = rabi_file + ".txt"
    with open(rabi_file_name) as f:
        rabi_data = json.load(f)

    uwave_time_range = rabi_data["uwave_time_range"]
    num_steps = rabi_data["num_steps"]
    norm_avg_sig = rabi_data["norm_avg_sig"]
    norm_avg_sig_ste = rabi_data["norm_avg_sig_ste"]

    taus = np.linspace(uwave_time_range[0], uwave_time_range[-1], num_steps)

    smooth_taus = np.linspace(uwave_time_range[0], uwave_time_range[-1], num=1000)
    # Fitting
    offset = 0.9
    frequency = 0.012
    decay = 1000
    init_params = [offset, frequency, decay]
    fit_func = tool_belt.cosexp_1_at_0
    popt, pcov = curve_fit(
        fit_func,
        taus,
        norm_avg_sig,
        p0=init_params,
        sigma=norm_avg_sig_ste,
        absolute_sigma=True,
    )
    # popt =init_params
    # print(popt)
    kpl.init_kplotlib()

    rabi_fig, ax = plt.subplots(1, 1, figsize=(3.2, 3.2))

    kpl.plot_points(ax, taus, norm_avg_sig, color=KplColors.BLUE)
    # ax.plot(taus, norm_avg_sig,  '.', color = KplColors.BLUE)
    kpl.plot_line(ax, smooth_taus, fit_func(smooth_taus, *popt), color=KplColors.RED)
    ax.set_ylabel("Normalized fluorescence")
    ax.set_xlabel(r"Microwave duration, $\tau$ (ns)")
    ax.set_xticks([0, 25, 50, 75, 100])
    # ax.legend()

    esr_fig, ax = plt.subplots(1, 1, figsize=(2.5 * 1.1, 3.5 * 1.1))

    ni = 0
    for file in pesr_file_list:
        narrow_fit = False
        if file == "2023_04_08-09_17_50-johnson-nv0_2023_04_06":
            narrow_fit = True
        file_name = file + ".txt"
        with open(file_name) as f:
            data = json.load(f)

        freq_center = data["freq_center"]
        freq_range = data["freq_range"]
        num_steps = data["num_steps"]
        norm_avg_sig = np.array(data["norm_avg_sig"])
        norm_avg_sig_ste = np.array(data["norm_avg_sig_ste"])

        half_freq_range = freq_range / 2
        freq_low = freq_center - half_freq_range
        freq_high = freq_center + half_freq_range
        freqs = np.linspace(freq_low, freq_high, num_steps)

        smooth_freqs = np.linspace(freq_low, freq_high, 1000)

        fit_func, popt, _ = fit_resonance(
            freq_range,
            act_freq_center,
            num_steps,
            norm_avg_sig,
            narrow_fit,
            norm_avg_sig_ste,
        )

        freq_MHz = (freqs - act_freq_center) * 1e3
        smooth_freqs_MHz = (smooth_freqs - act_freq_center) * 1e3
        kpl.plot_line(
            ax,
            smooth_freqs_MHz,
            fit_func(smooth_freqs, *popt) + ni,
            color=KplColors.RED,
        )
        # kpl.plot_points(ax, freq_MHz, norm_avg_sig + ni,  color=KplColors.BLUE, size=Size.TINY)
        ax.plot(freq_MHz, norm_avg_sig + ni, ".", color=KplColors.BLUE)

        ni = ni + 0.15

    ax.set_ylabel("Normalized fluorescence")
    ax.set_xlabel(r"Detuning, $\Delta \nu$ (MHz)")
    ax.set_xticks([-6, -4, -2, 0, 2, 4, 6])


act_freq_center = 2.91932
rabi_file = "2023_04_10-16_20_23-johnson-nv0_2023_04_06"

file_11 = "2023_04_08-00_25_43-johnson-nv0_2023_04_06"
file_14 = "2023_04_08-01_48_07-johnson-nv0_2023_04_06"
file_17 = "2023_04_08-03_15_36-johnson-nv0_2023_04_06"
file_185 = "2023_04_08-12_24_54-johnson-nv0_2023_04_06"
file_20 = "2023_04_08-04_45_05-johnson-nv0_2023_04_06"
file_23 = "2023_04_08-06_15_17-johnson-nv0_2023_04_06"
file_26 = "2023_04_08-07_45_43-johnson-nv0_2023_04_06"
file_29 = "2023_04_08-09_17_50-johnson-nv0_2023_04_06"
file_31 = "2023_04_08-13_56_52-johnson-nv0_2023_04_06"

pesr_file_list = [file_14, file_185, file_20, file_29]
# pesr_file_list = [file_11, file_14,file_17,  file_185,file_20,file_23,  file_26,file_29  ,file_31]
do_plot(rabi_file, pesr_file_list, act_freq_center)
