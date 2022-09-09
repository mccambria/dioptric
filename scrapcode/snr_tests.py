# -*- coding: utf-8 -*-
"""
This file contains standardized functions intended to simplify the 
creation of plots for publications in a consistent style.

Created on June 22nd, 2022

@author: mccambria
"""

import numpy as np
from math import factorial as fact
import matplotlib.pyplot as plt


def calc_SNR0(p, l0, l1, Ti, Te):

    q = 1 - p

    ret_val = 1 / (1 + (q / p))
    ret_val *= np.sqrt(1 / (Ti + Te))
    ret_val *= (l0 - l1) / np.sqrt(l0 + l1)

    print(ret_val)


def plot_SNR1(p, l0, l1, Ti, Te):

    q = 1 - p

    t_vals = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    t_vals = [round(l0 + el) for el in t_vals]
    SNR_vals = []

    for t in t_vals:
        SNR = 1 / (
            1 + (q / p) * (np.exp(-l1) / np.exp(-l0)) * ((l1 / l0) ** t)
        )
        SNR *= np.sqrt(
            1 / (Te + (fact(t) / (p * (l0 ** t) * np.exp(-l0))) * Ti)
        )
        SNR *= (l0 - l1) / np.sqrt(l0 + l1)
        SNR_vals.append(SNR)

    print(SNR_vals)


if __name__ == "__main__":

    p = 0.05
    l0 = 10
    l1 = 4
    Ti = 350e-9
    Te = 1e-6

    calc_SNR0(p, l0, l1, Ti, Te)
    plot_SNR1(p, l0, l1, Ti, Te)
