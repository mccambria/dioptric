# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 20:05:07 2023

@author: kolkowitz
"""

import copy
import json
import os
import time
from random import shuffle

import labrad
import matplotlib.pyplot as plt
import numpy as numpy
from scipy.optimize import curve_fit

import majorroutines.targeting as targeting
import utils.kplotlib as kpl
import utils.tool_belt as tool_belt
from utils.kplotlib import KplColors, Size
from utils.tool_belt import NormStyle, States


def do_plot(data):
    kpl.init_kplotlib()
    # nv_name = data['nv_sig']["name"]

    #### T2 time

    norm_avg_sig = data["norm_avg_sig"]
    sig_counts = data["sig_counts"]
    ref_counts = data["ref_counts"]
    precession_time_range = data["precession_time_range"]
    num_steps = data["num_steps"]
    num_reps = data["num_reps"]
    # do_dq = data['do_dq']
    nv_sig = data["nv_sig"]
    readout = nv_sig["spin_readout_dur"]
    norm_style = NormStyle.SINGLE_VALUED

    ret_vals = tool_belt.process_counts(
        sig_counts, ref_counts, num_reps, readout, norm_style
    )
    (
        sig_counts_avg_kcps,
        ref_counts_avg_kcps,
        norm_avg_sig,
        norm_avg_sig_ste,
    ) = ret_vals

    min_precession_time = int(precession_time_range[0])
    max_precession_time = int(precession_time_range[1])

    taus = numpy.linspace(
        min_precession_time,
        max_precession_time,
        num=num_steps,
    )

    taus_ms = numpy.array(taus) * 2 / 1e6

    #               amp, decay, offset
    guess_params = [0.05, 0.5, 1.2]
    fit_func = lambda x, a, b, d: tool_belt.exp_t2(x, a, b, d)

    popt, pcov = curve_fit(
        fit_func,
        taus_ms,
        norm_avg_sig,
        sigma=norm_avg_sig_ste,
        absolute_sigma=True,
        p0=guess_params,
    )
    print(popt)
    print(numpy.sqrt(numpy.diag(pcov)))

    taus_ms_linspace = numpy.linspace(taus_ms[0], taus_ms[-1], num=1000)

    fig_fit, ax = plt.subplots(1, 1, figsize=(4, 3))
    kpl.plot_points(
        ax, taus_ms, norm_avg_sig, yerr=norm_avg_sig_ste, color=KplColors.BLUE
    )
    kpl.plot_line(
        ax, taus_ms_linspace, fit_func(taus_ms_linspace, *popt), color=KplColors.RED
    )
    ax.set_xlabel(r"Free precesion time, $T = 2\tau$ (ms)")
    ax.set_ylabel("Normalized fluorescence")
    # ax.legend()


file = "2022_12_31-08_44_22-siena-nv6_2022_12_22"
file_name = file + ".txt"
with open(file_name) as f:
    data = json.load(f)

do_plot(data)
