# -*- coding: utf-8 -*-
"""
Widefield Rabi experiment

Created on November 29th, 2023

@author: mccambria
@author: sbchand
"""

import sys
import time
import traceback

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from majorroutines.widefield import base_routine
from utils import common
from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import tool_belt as tb
from utils import widefield as widefield
from utils.constants import NVSig


def create_fit_figure(nv_list, phis, norm_counts, norm_counts_ste):
    def cos_func(phi, amp, phase_offset):
        return 0.5 * amp * np.cos(phi - phase_offset) + 0.5

    fit_fns = []
    popts = []

    for nv_ind in range(num_nvs):
        nv_counts = norm_counts[nv_ind]
        nv_counts_ste = norm_counts_ste[nv_ind]

        guess_params = [1.0, 0.0]

        try:
            popt, _ = curve_fit(
                cos_func,
                phis,
                nv_counts,
                p0=guess_params,
                sigma=nv_counts_ste,
                absolute_sigma=True,
            )
        except Exception:
            popt = None

        fit_fns.append(cos_func if popt is not None else None)
        popts.append(popt)
        # Create new figure for this NV
        fig, ax = plt.subplots(figsize=(6, 5))

        # Plot data points
        ax.errorbar(
            phis,
            nv_counts,
            yerr=abs(nv_counts_ste),
            fmt="o",
            label=f"NV {nv_ind}",
            capsize=3,
        )

        # Plot fit if successful
        if popt is not None:
            phi_fit = np.linspace(min(phis), max(phis), 200)
            fit_vals = cos_func(phi_fit, *popt)
            ax.plot(phi_fit, fit_vals, "-", label="Fit")
            residuals = cos_func(phis, *popt) - nv_counts
            chi_sq = np.sum((residuals / nv_counts_ste) ** 2)
            red_chi_sq = chi_sq / (len(nv_counts) - len(popt))
            print(f"NV {nv_ind} - Reduced chi²: {red_chi_sq:.3f}")

        ax.set_xlabel("Phase (rad)")
        ax.set_ylabel("Normalized Counts")
        ax.set_title(f"Cosine Fit for NV {nv_ind}")
        ax.legend()
        ax.grid(True)

        # Beautify
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        plt.tight_layout()
        plt.show(block=True)


# Helper functions for rotations
def R_x(theta):
    return np.array(
        [
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)],
        ]
    )


def R_y(theta):
    return np.array(
        [
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)],
        ]
    )


def R_z(theta):
    return np.array(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ]
    )


# Simulate spin echo sequence
def spin_echo_signal(phase_array_deg):
    signal = []
    for phi_deg in phase_array_deg:
        phi_rad = np.deg2rad(phi_deg)

        # Initial state: spin up along Z
        bloch_vector = np.array([0, 0, 1])

        # π/2 pulse along X → brings spin to Y
        bloch_vector = R_x(-np.pi / 2) @ bloch_vector

        # Free evolution → identity for ideal case (skip)

        # π pulse along X → refocus
        bloch_vector = R_x(-np.pi) @ bloch_vector

        # Free evolution again → identity

        # Final π/2 pulse along axis with variable phase
        # This is equivalent to a π/2 pulse around an axis in XY plane
        final_rotation = R_z(phi_rad) @ R_x(-np.pi / 2) @ R_z(-phi_rad)
        bloch_vector = final_rotation @ bloch_vector

        # Project onto Z (measurement axis)
        signal.append(bloch_vector[2])  # This is what you measure

    return np.array(signal)


# Simulate and plot
def simulate_plot():
    phase_deg = np.linspace(0, 360, 200)
    signal = spin_echo_signal(phase_deg)

    plt.plot(phase_deg, signal, label="Spin Echo Signal")
    plt.xlabel("Final π/2 Phase (degrees)", fontsize=15)
    plt.ylabel("Z Projection (~ 1 / Signal)", fontsize=15)
    plt.title("Phase-Sensitive Spin Echo Simulation", fontsize=15)
    plt.grid(True)
    plt.legend()
    plt.show()


# Bloch vector rotation functions
def R_axis(theta, n):
    """Return a rotation matrix for angle theta around axis n (unit vector)."""
    n = n / np.linalg.norm(n)
    nx, ny, nz = n
    ct, st = np.cos(theta), np.sin(theta)
    return np.array(
        [
            [
                ct + nx**2 * (1 - ct),
                nx * ny * (1 - ct) - nz * st,
                nx * nz * (1 - ct) + ny * st,
            ],
            [
                ny * nx * (1 - ct) + nz * st,
                ct + ny**2 * (1 - ct),
                ny * nz * (1 - ct) - nx * st,
            ],
            [
                nz * nx * (1 - ct) - ny * st,
                nz * ny * (1 - ct) + nx * st,
                ct + nz**2 * (1 - ct),
            ],
        ]
    )


def apply_sequence(num_pulses, overrotation=0.0, axis_error_deg=0.0):
    """Simulate an XY sequence with pulse imperfections."""
    bloch = np.array([0, 0, 1])  # Start in |0> (Z+)

    # Initial π/2 pulse around X
    bloch = R_axis(-np.pi / 2, np.array([1, 0, 0])) @ bloch

    # Imperfect π pulses
    for i in range(num_pulses):
        # Alternate between X and Y pulses (XY sequence)
        ideal_axis = np.array([1, 0, 0]) if i % 2 == 0 else np.array([0, 1, 0])

        # Apply small axis error
        angle_offset = np.deg2rad(axis_error_deg)
        axis = ideal_axis + angle_offset * np.random.randn(3)

        # π pulse with small overrotation
        pulse_angle = np.pi * (1 + overrotation)
        bloch = R_axis(-pulse_angle, axis) @ bloch

    # Final π/2 pulse (back to Z)
    bloch = R_axis(-np.pi / 2, np.array([1, 0, 0])) @ bloch

    return bloch[2]  # Z projection (fluorescence ~ ms=0)


def simulate_pulse_errors():
    # Sweep over number of pulses
    pulse_counts = [0, 2, 4, 8, 16, 32]
    overrotation = 0.01  # 1% overrotation error
    axis_error_deg = 2.0  # 2° axis error
    signals = []
    for n in pulse_counts:
        signal = apply_sequence(
            n, overrotation=overrotation, axis_error_deg=axis_error_deg
        )
        signals.append(signal)

    # Plot
    plt.figure(figsize=(6, 4))
    plt.plot(pulse_counts, signals, "o-", label="With pulse errors")
    plt.xlabel("Number of π pulses")
    plt.ylabel("Signal (Z projection)")
    plt.title("Decay of coherence due to pulse errors")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    kpl.init_kplotlib()
    file_id = 1817334208399
    data = dm.get_raw_data(file_id=file_id, load_npz=False, use_cache=True)
    nv_list = data["nv_list"]
    num_nvs = len(nv_list)
    num_steps = data["num_steps"]
    num_runs = data["num_runs"]
    phis = data["phis"]

    counts = np.array(data["counts"])
    sig_counts = counts[0]
    ref_counts = counts[1]

    norm_counts, norm_counts_ste = widefield.process_counts(
        nv_list, sig_counts, ref_counts, threshold=True
    )
    file_name = dm.get_file_name(file_id=file_id)
    print(f"{file_name}_{file_id}")
    num_nvs = len(nv_list)
    phi_step = phis[1] - phis[0]
    num_steps = len(phis)
    fit_fig = create_fit_figure(nv_list, phis, norm_counts, norm_counts_ste)
    # simulate_plot()
    # simulate_pulse_errors()
    kpl.show(block=True)
