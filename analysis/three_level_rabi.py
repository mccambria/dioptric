# -*- coding: utf-8 -*-
"""
Rabi flopping in a three level system

Created on February 28th, 2023

@author: mccambria
"""


from mpmath import mp  # Arbitrary-precision math - necessary for matrix diagonalization
import numpy as np
from pathos.multiprocessing import ProcessingPool
import utils.tool_belt as tool_belt
import matplotlib.pyplot as plt
from utils import common
from utils import kplotlib as kpl
from utils.kplotlib import KplColors
from scipy.optimize import curve_fit
import csv
import pandas as pd
import sys
import majorroutines.pulsed_resonance as pesr
from scipy.integrate import odeint


def gen_hamiltonian_v1(dp, Omega, dm):
    return mp.matrix(
        [[dp, Omega / 2, 0], [Omega / 2, 0, Omega / 2], [0, Omega / 2, dm]]
    )


def gen_hamiltonian(detuning, rabi_freq, rabi_phase, splitting_freq, splitting_phase):
    rabi_coeff = np.exp(1j * rabi_phase)
    splitting_coeff = np.exp(1j * splitting_phase)
    return mp.matrix(
        [
            [
                detuning,
                rabi_coeff * rabi_freq / np.sqrt(2),
                splitting_coeff * splitting_freq / 2,
            ],
            [
                np.conjugate(rabi_coeff) * rabi_freq / np.sqrt(2),
                0,
                rabi_coeff * rabi_freq / np.sqrt(2),
            ],
            [
                np.conjugate(splitting_coeff) * splitting_freq / 2,
                np.conjugate(rabi_coeff) * rabi_freq / np.sqrt(2),
                detuning,
            ],
        ]
    )


def gen_relaxation_rates(dp, Omega, dm):
    rp = Omega**2 / np.sqrt(dp**2 + Omega**2)
    rm = Omega**2 / np.sqrt(dm**2 + Omega**2)
    return mp.matrix(
        [
            [-rp, rp, 0],
            [rp, -rp - rm, rm],
            [0, rm, -rm],
        ]
    )


def coherent_line_v1(freq, contrast, rabi_freq, center, splitting, pulse_dur):
    # Average over the hyperfine splittings
    line = None
    for adj_splitting in (splitting - 4.4, splitting, splitting + 4.4):
        args = [contrast, rabi_freq, center, adj_splitting, pulse_dur]
        if line is None:
            line = single_conversion(coherent_line_single, freq, *args)
        else:
            line += single_conversion(coherent_line_single, freq, *args)
    line /= 3

    # Set the contrast to be the max of the line
    # line *= contrast / (np.max(line))
    # Set the contrast to be what we'd observe for perfect population transfer
    line *= contrast
    return line

    # args = [contrast, rabi_freq, center, splitting, pulse_dur]
    # return single_conversion(coherent_line_single, freq, *args)


def coherent_line(
    freq,
    contrast,
    center,
    rabi_freq,
    rabi_phase,
    splitting_freq,
    splitting_phase,
    pulse_dur,
):
    # Average over the hyperfine splittings
    # line = None
    # for adj_splitting in (splitting - 4.4, splitting, splitting + 4.4):
    #     args = [contrast, rabi_freq, center, adj_splitting, pulse_dur]
    #     if line is None:
    #         line = single_conversion(coherent_line_single, freq, *args)
    #     else:
    #         line += single_conversion(coherent_line_single, freq, *args)
    # line /= 3

    args = [
        contrast,
        center,
        rabi_freq,
        rabi_phase,
        splitting_freq,
        splitting_phase,
        pulse_dur,
    ]
    line = single_conversion(coherent_line_single, freq, *args)

    # Set the contrast to be the max of the line
    # line *= contrast / (np.max(line))
    # Set the contrast to be what we'd observe for perfect population transfer
    line *= contrast
    return line

    # args = [contrast, rabi_freq, center, splitting, pulse_dur]
    # return single_conversion(coherent_line_single, freq, *args)


def incoherent_line(freq, contrast, rabi_freq, center, splitting, offset, pulse_dur):
    args = [contrast, rabi_freq, center, splitting, offset, pulse_dur]
    return single_conversion(incoherent_line_single, freq, *args)


def single_conversion(single_func, freq, *args):
    if type(freq) in [list, np.ndarray]:
        single_func_lambda = lambda freq: single_func(freq, *args)
        # with ProcessingPool() as p:
        #     line = p.map(single_func_lambda, freq)
        line = np.array([single_func_lambda(f) for f in freq])
        return line
    else:
        return single_func(freq, *args)


def coherent_line_single_v1(freq, contrast, rabi_freq, center, splitting, pulse_dur):
    dp = (center * 1000 + splitting / 2) - (freq * 1000)
    dm = (center * 1000 - splitting / 2) - (freq * 1000)

    if pulse_dur == None:
        pulse_dur = 1 / (2 * rabi_freq)
    else:
        pulse_dur /= 1000

    # coupling = rabi_freq / mp.sqrt(2)  # Account for sqrt(2) factor at splitting=0
    coupling = rabi_freq
    hamiltonian = gen_hamiltonian(dp, coupling, dm)
    eigvals, eigvecs = mp.eighe(hamiltonian)

    intial_vec = mp.matrix([[0], [1], [0]])
    initial_comps = [mp.fdot(intial_vec, eigvecs[:, ind]) for ind in range(3)]
    final_comps = [
        initial_comps[ind] * mp.exp(2 * mp.pi * (0 + 1j) * eigvals[ind] * pulse_dur)
        for ind in range(3)
    ]
    final_vec = mp.matrix([[0], [0], [0]])
    for ind in range(3):
        final_vec += eigvecs[:, ind] * final_comps[ind]

    line = 1 - mp.fabs(final_vec[1]) ** 2
    ret_val = line
    # ret_val = contrast * line

    return np.float64(ret_val)


def coherent_line_single(
    freq,
    contrast,
    center,
    rabi_freq,
    rabi_phase,
    splitting_freq,
    splitting_phase,
    pulse_dur,
):
    detuning = (center - freq) * 1000

    if pulse_dur == None:
        pulse_dur = 1 / (2 * rabi_freq)
    else:
        pulse_dur /= 1000

    # coupling = rabi_freq / mp.sqrt(2)  # Account for sqrt(2) factor at splitting=0
    # coupling = rabi_freq
    hamiltonian = gen_hamiltonian(
        detuning, rabi_freq, rabi_phase, splitting_freq, splitting_phase
    )
    eigvals, eigvecs = mp.eighe(hamiltonian)

    intial_vec = mp.matrix([[0], [1], [0]])
    initial_comps = [mp.fdot(intial_vec, eigvecs[:, ind]) for ind in range(3)]
    final_comps = [
        initial_comps[ind] * mp.exp(2 * mp.pi * (0 + 1j) * eigvals[ind] * pulse_dur)
        for ind in range(3)
    ]
    final_vec = mp.matrix([[0], [0], [0]])
    for ind in range(3):
        final_vec += eigvecs[:, ind] * final_comps[ind]

    line = 1 - mp.fabs(final_vec[1]) ** 2
    ret_val = line
    # ret_val = contrast * line

    return np.float64(ret_val)


def incoherent_line_single(
    freq, contrast, rabi_freq, center, splitting, offset, pulse_dur
):
    dp = (center * 1000 + splitting / 2) - (freq * 1000)
    dm = (center * 1000 - splitting / 2) - (freq * 1000)

    if pulse_dur == None:
        pulse_dur = 1 / (2 * rabi_freq)
    else:
        pulse_dur /= 1000

    relaxation_rates = gen_relaxation_rates(dp, rabi_freq, dm)
    eigvals, eigvecs = mp.eigsy(relaxation_rates)

    intial_vec = mp.matrix([[0], [1], [0]])
    initial_comps = [mp.fdot(intial_vec, eigvecs[:, ind]) for ind in range(3)]
    final_comps = [
        initial_comps[ind] * mp.exp(eigvals[ind] * pulse_dur) for ind in range(3)
    ]
    final_vec = mp.matrix([[0], [0], [0]])
    for ind in range(3):
        final_vec += eigvecs[:, ind] * final_comps[ind]

    line = 1 - final_vec[1]
    ret_val = offset + contrast * line
    return np.float64(ret_val)


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
    Omega = 4
    half_splitting = 3
    # half_splitting = 5
    center_freq = 2.87
    wp = center_freq + half_splitting
    wm = center_freq - half_splitting

    drive_freqs = np.linspace(2.80, 2.890, 1000)
    t = 500

    fig, ax = plt.subplots()

    fl = coherent_line(drive_freqs, 1, Omega, center_freq, half_splitting * 2, t)
    kpl.plot_line(ax, drive_freqs, fl)


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
    # incoherent()
    # plot_mat_els()

    freqs = np.linspace(2.85, 2.89, 100)
    fig, ax = plt.subplots()

    # contrast, center, rabi_freq, rabi_phase, splitting_freq, splitting_phase, pulse_dur
    kpl.plot_line(
        ax,
        freqs,
        coherent_line(freqs, 0.2, 2.87, 5.2, 0.0 * np.pi, 5, 0.0 * np.pi, 50),
    )
    kpl.plot_line(
        ax,
        freqs,
        coherent_line(freqs, 0.2, 2.87, 5.2, 0.25 * np.pi, 5, 0.5 * np.pi, 50),
    )
    # kpl.plot_line(ax, freqs, coherent_line(freqs, 0.2, 2.87, 5.2, 0, 4, 1.0, 200))
    # kpl.plot_line(ax, freqs, coherent_line(freqs, 0.2, 2.87, 5.2, 1.0, 4, 1.0, 200))
    # kpl.plot_line(ax, freqs, coherent_line(freqs, 0.2, 2.87, 5.2, -1.0, 4, 1.0, 200))

    plt.show(block=True)
