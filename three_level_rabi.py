# -*- coding: utf-8 -*-
"""
Rabi flopping in a three level system

Created on February 28th, 2023

@author: mccambria
"""

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
from scipy.integrate import odeint


def gen_hamiltonian(dp, Omega, dm):
    return np.array(
        [
            [dp, Omega / 2, 0],
            [Omega / 2, 0, Omega / 2],
            [0, Omega / 2, dm],
        ]
    )


def gen_relaxation_rates(dp, Omega, dm):
    rp = Omega**2 / np.sqrt(dp**2 + Omega**2)
    rm = Omega**2 / np.sqrt(dm**2 + Omega**2)
    return np.array(
        [
            [-rp, rp, 0],
            [rp, -rp - rm, rm],
            [0, rm, -rm],
        ]
    )


def coherent_line(freq, contrast, rabi_freq, center, splitting, pulse_dur):
    if type(freq) in [list, np.ndarray]:
        line = [
            coherent_line_single(f, rabi_freq, center, splitting, pulse_dur)
            for f in freq
        ]
        line = np.array(line)
        return line
    else:
        return coherent_line_single(freq, rabi_freq, center, splitting, pulse_dur)


def coherent_line_single(freq, contrast, rabi_freq, center, splitting, pulse_dur):

    dp = (center + splitting / 2) - freq
    dm = (center - splitting / 2) - freq

    hamiltonian = gen_hamiltonian(dp, rabi_freq, dm)
    eigvals, eigvecs = np.linalg.eig(hamiltonian)
    idx = eigvals.argsort()[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    intial_vec = np.array([0, 1, 0])
    initial_comps = [np.dot(intial_vec, eigvecs[:, ind]) for ind in range(3)]
    final_comps = [
        initial_comps[ind] * np.exp((0 + 1j) * eigvals[ind] * pulse_dur)
        for ind in range(3)
    ]
    final_vec = np.array([0, 0, 0], dtype=np.complex128)
    for ind in range(3):
        final_vec += eigvecs[:, ind] * final_comps[ind]

    line = 1 - np.absolute(final_vec[1]) ** 2
    return contrast * line


def calc_dy_dt(y, t, relaxation_rates):
    return np.matmul(relaxation_rates, y)


def plot_mat_els():

    Omega = 1
    # half_splitting = 0.5
    half_splitting = 10
    center_freq = 2870
    wp = center_freq + half_splitting
    wm = center_freq - half_splitting

    Omegas = np.linspace(-2, 2, 1000)

    x_vals = Omegas

    plot_mags = [[], [], []]

    drive_freq = wp
    for Omega in Omegas:

        dp = wp - drive_freq
        dm = wm - drive_freq

        hamiltonian = gen_hamiltonian(dp, Omega, dm)
        eigvals, eigvecs = np.linalg.eig(hamiltonian)
        idx = eigvals.argsort()[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        # plot_mags[0].append(eigvals[0])
        # plot_mags[1].append(eigvals[1])
        # plot_mags[2].append(eigvals[2])
        plot_mags[0].append(np.abs(eigvecs[0][1]) ** 2)
        plot_mags[1].append(np.abs(eigvecs[1][1]) ** 2)
        plot_mags[2].append(np.abs(eigvecs[2][1]) ** 2)

    fig, ax = plt.subplots()

    kpl.plot_line(ax, x_vals, plot_mags[0], label="+")
    kpl.plot_line(ax, x_vals, plot_mags[1], label="0")
    kpl.plot_line(ax, x_vals, plot_mags[2], label="-")

    ax.legend()


def incoherent():

    Omega = 1
    half_splitting = 3
    # half_splitting = 5
    center_freq = 2870
    wp = center_freq + half_splitting
    wm = center_freq - half_splitting

    drive_freqs = np.linspace(2850, 2890, 1000)
    ts = np.linspace(0, 1 / Omega, 1000)
    x_vals = drive_freqs
    # x_vals = ts

    plot_mags = [[], [], []]

    # t = 10 / Omega
    t = max(ts)
    for drive_freq in drive_freqs:
        # drive_freq = wp
        # for t in ts:

        dp = wp - drive_freq
        dm = wm - drive_freq

        relaxation_rates = gen_relaxation_rates(dp, Omega, dm)
        intial_vec = np.array([0, 1, 0])

        eigvals, eigvecs = np.linalg.eig(relaxation_rates)
        initial_comps = [np.dot(intial_vec, eigvecs[:, ind]) for ind in range(3)]
        final_comps = [
            initial_comps[ind] * np.exp(eigvals[ind] * t) for ind in range(3)
        ]
        final_vec = np.array([0, 0, 0], dtype=np.float128)
        for ind in range(3):
            final_vec += eigvecs[:, ind] * final_comps[ind]

        plot_mags[0].append(final_vec[0])
        plot_mags[1].append(final_vec[1])
        plot_mags[2].append(final_vec[2])

    fig, ax = plt.subplots()

    kpl.plot_line(ax, x_vals, plot_mags[0], label="+")
    kpl.plot_line(ax, x_vals, plot_mags[1], label="0")
    kpl.plot_line(ax, x_vals, plot_mags[2], label="-")

    ax.legend()


def main():

    Omega = 10
    half_splitting = 0.5
    # half_splitting = 5
    center_freq = 2870
    wp = center_freq + half_splitting
    wm = center_freq - half_splitting

    drive_freqs = np.linspace(2860, 2880, 1000)
    ts = np.linspace(0, 100 * np.pi / Omega, 100000)
    # x_vals = drive_freqs
    x_vals = ts

    plot_mags = [[], [], []]

    # t = np.pi / Omega
    # t = 2.8
    # for drive_freq in drive_freqs:
    drive_freq = wp
    for t in ts:

        dp = wp - drive_freq
        dm = wm - drive_freq

        hamiltonian = gen_hamiltonian(dp, Omega, dm)
        eigvals, eigvecs = np.linalg.eig(hamiltonian)

        intial_vec = np.array([0, 1, 0])
        initial_comps = [np.dot(intial_vec, eigvecs[:, ind]) for ind in range(3)]
        final_comps = [
            initial_comps[ind] * np.exp((0 + 1j) * eigvals[ind] * t) for ind in range(3)
        ]
        final_vec = np.array([0, 0, 0], dtype=np.complex128)
        for ind in range(3):
            final_vec += eigvecs[:, ind] * final_comps[ind]

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

    # main()
    incoherent()
    # plot_mat_els()

    plt.show(block=True)
