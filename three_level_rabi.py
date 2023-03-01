# -*- coding: utf-8 -*-
"""
Rabi flopping in a three level system

Created on February 28th, 2023

@author: mccambria
"""

# region Import and constants


import numpy as np
import utils.tool_belt as tool_belt
import matplotlib.pyplot as plt
from utils import common
from utils import kplotlib as kpl
from utils.kplotlib import KplColors
from scipy.optimize import curve_fit
import csv
import pandas as pd
import sys


def gen_hamiltonian(dp, Omega, dm):
    return np.array(
        [
            [dp, Omega / 2, 0],
            [Omega / 2, 0, Omega / 2],
            [0, Omega / 2, dm],
        ]
    )


def main():

    Omega = 1
    half_splitting = 0.2
    # half_splitting = 10
    center_freq = 2870
    wp = center_freq + half_splitting
    wm = center_freq - half_splitting

    drive_freqs = np.linspace(2860, 2880, 1000)
    ts = np.linspace(0, 10 * np.pi / Omega, 1000)
    x_vals = drive_freqs
    x_vals = ts

    plot_mags = [[], [], []]

    # t = np.pi / Omega
    # for drive_freq in drive_freqs:
    drive_freq = center_freq
    for t in ts:

        dp = wp - drive_freq
        dm = wm - drive_freq

        hamiltonian = gen_hamiltonian(dp, Omega, dm)
        eigvals, eigvecs = np.linalg.eig(hamiltonian)
        idx = eigvals.argsort()[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        intial_vec = np.array([0, 1, 0])
        initial_comps = [np.dot(intial_vec, eigvecs[ind]) for ind in range(3)]
        final_comps = [
            initial_comps[ind] * np.exp((0 + 1j) * eigvals[ind] * t) for ind in range(3)
        ]
        final_vec = np.array([0, 0, 0], dtype=np.complex128)
        for ind in range(3):
            final_vec += eigvecs[ind] * final_comps[ind]

        plot_mags[0].append(np.absolute(final_vec[0]) ** 2)
        plot_mags[1].append(np.absolute(final_vec[1]) ** 2)
        plot_mags[2].append(np.absolute(final_vec[2]) ** 2)

    fig, ax = plt.subplots()

    kpl.plot_line(ax, x_vals, plot_mags[0], label="+")
    kpl.plot_line(ax, x_vals, plot_mags[1], label="0")
    kpl.plot_line(ax, x_vals, plot_mags[2], label="-")

    ax.legend()


if __name__ == "__main__":

    kpl.init_kplotlib()

    main()

    plt.show(block=True)
