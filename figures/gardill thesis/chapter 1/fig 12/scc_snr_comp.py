# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 13:53:56 2023

@author: gardill
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
from scipy.signal import find_peaks

import majorroutines.targeting as targeting
import utils.kplotlib as kpl
import utils.tool_belt as tool_belt
from utils.kplotlib import KplColors, Size
from utils.tool_belt import States


def snr(tE, tR, SNR_ss, T):
    return SNR_ss * numpy.sqrt(T / (tE + tR))


T = 10
tR_scc = 50
SNR_scc = 0.1
tR_c = 0.0003
SNR_c = 0.05

kpl.init_kplotlib()
fig, ax = plt.subplots()
ax.set_xlabel(r"$t_R$")
ax.set_ylabel("SNR")

t_list = numpy.linspace(1, 100, 1000)
kpl.plot_line(
    ax, t_list, snr(t_list, tR_scc, SNR_scc, T), label="SCC", color=KplColors.GREEN
)
kpl.plot_line(ax, t_list, snr(t_list, tR_c, SNR_c, T), label="C", color=KplColors.RED)
ax.legend()
