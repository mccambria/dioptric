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


def gen_hamiltonian(dp, dm, rabi_freq, splitting_freq, phase):
    # def gen_hamiltonian(dp, dm, rabi_freq):
    # splitting_freq = 0
    # phase = 0
    phase_factor = mp.expj(phase)
    conj_phase_factor = mp.conj(phase_factor)
    # Include spin matrix factors
    normed_rabi = rabi_freq / mp.sqrt(2)
    normed_splitting = splitting_freq / 2
    return mp.matrix(
        [
            [dp, phase_factor * normed_rabi, normed_splitting],
            [conj_phase_factor * normed_rabi, 0, phase_factor * normed_rabi],
            [normed_splitting, conj_phase_factor * normed_rabi, dm],
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


def coherent_line(freq, contrast, center, rabi_freq, splitting_freq, phase, pulse_dur):
    # def coherent_line(freq, contrast, center, rabi_freq, pulse_dur):
    # Average over the hyperfine splittings
    # line = None
    # for Bz in (-2.2, 0, 2.2):
    #     args = [contrast, center, rabi_freq, splitting_freq, phase, pulse_dur, Bz]
    #     # args = [contrast, center, rabi_freq, pulse_dur, Bz]
    #     if line is None:
    #         line = single_conversion(coherent_line_single, freq, *args)
    #     else:
    #         line += single_conversion(coherent_line_single, freq, *args)
    # line /= 3

    args = [contrast, center, rabi_freq, splitting_freq, phase, pulse_dur]
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
    freq, contrast, center, rabi_freq, splitting_freq, phase, pulse_dur, Bz=0
):
    # def coherent_line_single(freq, contrast, center, rabi_freq, pulse_dur, Bz=0):

    detuning = (center - freq) * 1000
    dp = detuning + Bz
    dm = detuning - Bz

    if pulse_dur == None:
        pulse_dur = 1 / (2 * rabi_freq)
    else:
        pulse_dur /= 1000

    # coupling = rabi_freq / mp.sqrt(2)  # Account for sqrt(2) factor at splitting=0
    # coupling = rabi_freq
    hamiltonian = gen_hamiltonian(dp, dm, rabi_freq, splitting_freq, phase)
    # hamiltonian = gen_hamiltonian(dp, dm, rabi_freq)
    eigvals, eigvecs = mp.eighe(hamiltonian)

    # Check degeneracy
    diffs = [eigvals[0] - eigvals[1], eigvals[0] - eigvals[2], eigvals[1] - eigvals[2]]
    diffs = [np.abs(val) for val in diffs]
    degeneracies = [val < 0.01 for val in diffs]
    # if True in degeneracies:
    #     test = 1
    # if 2.87488 < freq < 2.87489:
    #     test = 1

    intial_vec = mp.matrix([[0], [1], [0]])
    initial_comps = [mp.fdot(intial_vec, eigvecs[:, ind]) for ind in range(3)]
    final_comps = [
        initial_comps[ind] * mp.expj(2 * mp.pi * eigvals[ind] * pulse_dur)
        for ind in range(3)
    ]
    final_vec = mp.matrix([[0], [0], [0]])
    for ind in range(3):
        final_vec += eigvecs[:, ind] * final_comps[ind]

    line = 1 - mp.fabs(final_vec[1]) ** 2
    ret_val = line
    # ret_val = contrast * line

    return np.float64(ret_val)
    # return np.float64(min(diffs))


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

    freqs = np.linspace(2.85, 2.89, 1000)
    # freqs = np.linspace(2.8745, 2.8755, 1000)
    fig, ax = plt.subplots()

    # contrast, center, rabi_freq, rabi_phase, splitting_freq, splitting_phase, pulse_dur
    # kpl.plot_line(ax, freqs, coherent_line(freqs, 0.2, 2.87, 5.2, 0, 0.0 * np.pi, 50))
    # kpl.plot_line(ax, freqs, coherent_line(freqs, 0.2, 2.87, 5.2, -10, 0.0 * np.pi, 50))
    # kpl.plot_line(ax, freqs, coherent_line(freqs, 0.2, 2.87, 5.2, +10, 0.0 * np.pi, 50))
    # kpl.plot_line(ax, freqs, coherent_line(freqs, 0.2, 2.875, 5.2, 0, 0.0 * np.pi, 50))

    # kpl.plot_line(ax, freqs, coherent_line(freqs, 0.30, 2.87, 1.717, 1.0, 0, 200))
    # kpl.plot_line(ax, freqs, coherent_line(freqs, 0.30, 2.871, 1.717, 0.0, 0, 200))
    # kpl.plot_line(
    #     ax, freqs, coherent_line(freqs, 0.30, 2.87, 1.717, 1.133, np.pi / 4, 200)
    # )
    # kpl.plot_line(
    #     ax, freqs, coherent_line(freqs, 0.30, 2.877, 1.717, 1.133, np.pi / 2, 200)
    # )
    params = [
        2.71769061e-01,
        2.86946093e00,
        2.04020876e00,
        7.19232927e-01,
        -1.71050225e-03,
        # np.pi / 4,
    ]
    # kpl.plot_line(ax, freqs, coherent_line(freqs, *params, 200))
    f = 2.875
    detuning = (params[1] - f) * 1000
    dp = detuning
    dm = detuning
    hamiltonian = gen_hamiltonian(dp, dm, params[2], params[3], 0 * params[4])
    eigvals, eigvecs = mp.eighe(hamiltonian)
    print(eigvals)

    plt.show(block=True)
