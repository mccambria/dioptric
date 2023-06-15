# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 13:53:56 2023

@author: gardill
"""

import utils.tool_belt as tool_belt
import numpy as numpy
import os
import time
import labrad
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from random import shuffle
from utils.tool_belt import States
import utils.kplotlib as kpl
from utils.kplotlib import KplColors
from utils.kplotlib import Size
import copy
import matplotlib.pyplot as plt
import majorroutines.optimize as optimize
import json
from scipy.signal import find_peaks


def snr(tE, tR, SNR_ss, T):
    return SNR_ss * numpy.sqrt(T/(tE + tR))

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
kpl.plot_line(
    ax, t_list, snr(t_list, tR_c, SNR_c, T), label="C", color=KplColors.RED
)
ax.legend()

