# -*- coding: utf-8 -*-
"""
Calculations for "To threshold or not to threshold" note

Created on March 29th, 2024

@author: mccambria
"""

import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from scipy.optimize import curve_fit
from scipy.special import factorial

from majorroutines.widefield import optimize
from utils import common, widefield
from utils import data_manager as dm
from utils import kplotlib as kpl


def optimum_threshold(l0, l1):
    return np.floor((l0 - l1) / np.log(l0 / l1))


def poisson_pdf(k, l):
    return l**k * pn.exp(-l) / factorial(k)


def poisson_cdf(k, l):
    vals = [poisson_pdf(i, l) for i in range(i + 1)]
    return np.sum(vals)


def fidelity(T, l0, l1):
    return (1 / 2) * (poisson_cdf(T, l0) + (1 - poisson_cdf(T, l1)))


def optimum_fidelity(l0, l1):
    T = optimum_threshold(l0, l1)
    return fidelity(T, l0, l1)


def snr_counting(l0, l1):
    return (l1 - l0) / np.sqrt(l1 + l0)


def snr_thresholding(l0, l1):
    T = optimum_threshold(l0, l1)
    p1 = 1 - poisson_cdf(T, l1)
    p0 = 1 - poisson_cdf(T, l0)
    return (p1 - p0) / np.sqrt(p1(1 - p1) + p0(1 - p0))


def snr_gain(l0, l1):
    if l1 > l0:
        return (snr_thresholding(l0, l1) - snr_counting(l0, l1)) / snr_counting(l0, l1)
    else:
        return 0


if __name__ == "__main__":
    kpl.init_kplotlib()

    kpl.show(block=True)
