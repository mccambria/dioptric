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


def T2_scale_alt(N, n, T2_0):  # https://www.nature.com/articles/s41467-019-11776-8
    return (N) ** (n) * T2_0


def plot(N_dq, N_sq, t2_sq, t2_sq_unc, t2_dq, t2_dq_unc, title, do_fit=True):
    kpl.init_kplotlib()

    N_lin = numpy.linspace(N_sq[0], N_sq[-1], 100)

    if do_fit:
        fit_func = T2_scale_alt
        init_params = [0.05, 1.5]
        popt_sq, pcov_sq = curve_fit(
            fit_func,
            N_sq,
            t2_sq,
            sigma=t2_sq_unc,
            absolute_sigma=True,
            p0=init_params,
        )

        # popt_dq, pcov_dq = curve_fit(
        #     fit_func,
        #     N_dq,
        #     t2_dq,
        #     sigma=t2_dq_unc,
        #     absolute_sigma=True,
        #     p0=init_params,
        # )

        # print('T2_0 = {} +/- {} us'.format(popt[0], numpy.sqrt(pcov[0][0])))
        # print(popt_dq)

    # Plot setup
    fig, ax = plt.subplots(1, 1)
    ax.set_xlabel("CPMG-N")
    ax.set_ylabel("T2 time (us)")
    ax.set_title(title)

    # Plotting
    kpl.plot_points(ax, N_sq, t2_sq, yerr=t2_sq_unc, label="SQ", color=KplColors.RED)
    kpl.plot_points(ax, N_dq, t2_dq, yerr=t2_dq_unc, label="DQ", color=KplColors.GREEN)

    if do_fit:
        kpl.plot_line(
            ax, N_lin, fit_func(N_lin, *popt_sq), label="SQ fit", color=KplColors.RED
        )
        # kpl.plot_line(ax, N_lin, fit_func(N_lin,*popt_dq ), label = 'DQ fit', color=KplColors.GREEN)
    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.legend()


# NV4

N_sq = [2, 4, 8, 16, 32, 64, 128, 256, 512]

# t2_sq = [1.76,  # fixed decay value
#          2.42,
#          2.38,
#          3.20,
#          2.67,
#          2.74,
#          3.21,
#          2.40,
#          3.06
#          ]
t2_sq = [
    2.06,  # free decay value
    2.78,
    2.31,
    3.40,
    2.60,
    2.43,
    1.97,
    3.09,
    3.26,
]
t2_sq_unc = [0.08, 0.12, 0.07, 0.13, 0.08, 0.11, 0.08, 0.12, 0.28]

N_dq = [
    # 2, 4, 8 ,16, 32,
    # 64,
    # 128,
    # 256,
    # 512
]
t2_dq = [
    # 1.19, 1.24, 1.42, 1.96,1.96,
    # 1.11,
    # 2.65, #1.76,
    # 1.94,
    # 2.94
]
t2_dq_unc = [
    # 0.14, 0.15, 0.20, 0.37, 0.36,
    # 0.13,
    # 0.27,
    # 0.3,
    # 0.47
]

title = "NV4-2023_01_16"
plot(N_dq, N_sq, t2_sq, t2_sq_unc, t2_dq, t2_dq_unc, title, do_fit=False)
