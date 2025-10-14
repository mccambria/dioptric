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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def cos_func(phi_deg, amp, phase_offset_deg, offset):
    return amp * np.cos(np.radians(phi_deg) - np.radians(phase_offset_deg)) + offset

def fit_phase_fringe(phis_deg, norm_counts, norm_counts_ste):
    """
    Fit all NV phase fringes to cos_func and return a DataFrame with per-NV params.
    Returns:
      fit_df: DataFrame with columns [amplitude, phase_offset, offset, chi_sq, contrast]
      median_fit: dict with popt, phis_fit, vals_fit for the median NV trace
      med_counts, med_counts_ste: median normalized counts (+ STE) over NVs
    """
    num_nvs = norm_counts.shape[0]
    results = {"amplitude": [], "phase_offset": [], "offset": [], "chi_sq": []}

    for i in range(num_nvs):
        y = norm_counts[i]
        yerr = np.asarray(norm_counts_ste[i])
        # Safe uncertainties (avoid zeros)
        yerr = np.where(np.asarray(yerr) <= 0, np.nanmedian(np.abs(y - np.nanmedian(y))) or 1.0, yerr)

        try:
            popt, pcov = curve_fit(
                cos_func, phis_deg, y, p0=[0.25, 0.0, 0.5], sigma=yerr, absolute_sigma=True, maxfev=20000
            )
            residuals = cos_func(phis_deg, *popt) - y
            dof = max(len(y) - len(popt), 1)
            chi_sq_red = np.nansum((residuals / yerr) ** 2) / dof
        except Exception:
            popt = [np.nan, np.nan, np.nan]
            chi_sq_red = np.nan

        results["amplitude"].append(popt[0])
        results["phase_offset"].append(popt[1])
        results["offset"].append(popt[2])
        results["chi_sq"].append(chi_sq_red)

    fit_df = pd.DataFrame(results)
    fit_df["contrast"] = 2.0 * fit_df["amplitude"]  # peak-to-trough

    # Median trace across NVs
    med_counts = np.nanmedian(norm_counts, axis=0)
    med_counts_ste = np.nanmedian(norm_counts_ste, axis=0)

    try:
        popt_med, _ = curve_fit(
            cos_func, phis_deg, med_counts, p0=[0.25, 0.0, 0.5],
            sigma=np.where(med_counts_ste<=0, np.nanmedian(np.abs(med_counts - np.nanmedian(med_counts))) or 1.0, med_counts_ste),
            absolute_sigma=True, maxfev=20000
        )
        phi_fit = np.linspace(np.min(phis_deg), np.max(phis_deg), 300)
        vals_fit = cos_func(phi_fit, *popt_med)
        median_fit = {"popt": popt_med, "phi_fit": phi_fit, "vals_fit": vals_fit}
    except Exception:
        median_fit = {"popt": [np.nan, np.nan, np.nan], "phi_fit": None, "vals_fit": None}

    return fit_df, median_fit, med_counts, med_counts_ste

def load_and_process(file_stem):
    """
    Uses your dm.get_raw_data + widefield.process_counts to produce phis (deg),
    norm_counts [n_nv x n_phi], and norm_counts_ste with thresholding.
    """
    data = dm.get_raw_data(file_stem=file_stem, load_npz=True, use_cache=True)
    nv_list = data["nv_list"]
    counts = np.array(data["counts"])
    sig_counts = counts[0]
    ref_counts = counts[1]

    norm_counts, norm_counts_ste = widefield.process_counts(
        nv_list, sig_counts, ref_counts, threshold=True
    )
    phis = np.array(data["phis"])
    # Your fitting uses degrees:
    phis_deg = phis  # if already in degrees; otherwise: np.degrees(phis)
    return phis_deg, np.array(norm_counts), np.array(norm_counts_ste)

def compare_two_runs(file_stem_A, label_A, file_stem_B, label_B, show_individual=False):
    # Load
    phis_A, norm_A, ste_A = load_and_process(file_stem_A)
    phis_B, norm_B, ste_B = load_and_process(file_stem_B)

    # Fit
    fit_A, med_A, medc_A, medste_A = fit_phase_fringe(phis_A, norm_A, ste_A)
    fit_B, med_B, medc_B, medste_B = fit_phase_fringe(phis_B, norm_B, ste_B)

    # --- Print summary stats ---
    def stats(name, df):
        c = df["contrast"].dropna()
        if len(c) == 0:
            print(f"{name}: no valid fits")
            return
        print(f"{name}  (N={len(c)} NVs)")
        print(f"  median contrast: {np.nanmedian(c):.3f}")
        print(f"  mean ± std:      {np.nanmean(c):.3f} ± {np.nanstd(c):.3f}")
        q10, q90 = np.nanpercentile(c, [10, 90])
        print(f"  10–90% range:    [{q10:.3f}, {q90:.3f}]")
        print()

    print("\n=== Per-NV fringe contrast (peak–to–trough = 2*amp) ===")
    stats(label_A, fit_A)
    stats(label_B, fit_B)

    # Median contrast from fitted medians (2*amp_med)
    def median_contrast_from_fit(med):
        amp_med = med["popt"][0] if med["popt"] is not None else np.nan
        return 2.0 * amp_med

    Cmed_A = median_contrast_from_fit(med_A)
    Cmed_B = median_contrast_from_fit(med_B)
    print(f"Median-fit contrast: {label_A}: {Cmed_A:.3f} | {label_B}: {Cmed_B:.3f}")
    print(f"Δ contrast ({label_A} − {label_B}): {Cmed_A - Cmed_B:.3f}\n")

    # --- Plots ---
    # (1) Overlay median fringes
    fig1, ax1 = plt.subplots(figsize=(6, 5))
    ax1.errorbar(phis_A, medc_A, yerr=medste_A, fmt="o", alpha=0.6, label=f"{label_A} median")
    if med_A["phi_fit"] is not None:
        ax1.plot(med_A["phi_fit"], med_A["vals_fit"], "--", alpha=0.9)

    ax1.errorbar(phis_B, medc_B, yerr=medste_B, fmt="s", alpha=0.6, label=f"{label_B} median")
    if med_B["phi_fit"] is not None:
        ax1.plot(med_B["phi_fit"], med_B["vals_fit"], "-.", alpha=0.9)

    ax1.set_xlabel("Phase, φ (degrees)")
    ax1.set_ylabel("Median normalized counts")
    ax1.set_title("Median fringes: overlay")
    ax1.grid(True); ax1.spines["right"].set_visible(False); ax1.spines["top"].set_visible(False)
    ax1.legend()

    # (2) Side-by-side contrast distributions
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    data_to_plot = [fit_A["contrast"].dropna().values, fit_B["contrast"].dropna().values]
    ax2.violinplot(data_to_plot, showmeans=True, showextrema=True)
    ax2.set_xticks([1, 2]); ax2.set_xticklabels([label_A, label_B])
    ax2.set_ylabel("Per-NV fringe contrast (peak–to–trough)")
    ax2.set_title("Per-NV contrast distributions")
    ax2.grid(True); ax2.spines["right"].set_visible(False); ax2.spines["top"].set_visible(False)

    # (3) Optional: plot individual NV fits for each run
    if show_individual:
        def plot_individual(phis, norm, ste, label):
            fig, ax = plt.subplots(figsize=(6, 5))
            for i in range(norm.shape[0]):
                ax.errorbar(phis, norm[i], yerr=np.abs(ste[i]), fmt="o", alpha=0.25)
            ax.set_xlabel("Phase (deg)"); ax.set_ylabel("Normalized counts")
            ax.set_title(f"All NVs: {label}"); ax.grid(True)
            ax.spines["right"].set_visible(False); ax.spines["top"].set_visible(False)
        plot_individual(phis_A, norm_A, ste_A, label_A)
        plot_individual(phis_B, norm_B, ste_B, label_B)

    plt.show()

    return fit_A, fit_B, Cmed_A, Cmed_B

def plot_contrast_scatter(fit_A, label_A, fit_B=None, label_B=None):
    """
    Scatter plot of per-NV contrast magnitudes (|2*amp|).
    If fit_B is provided, overlays both runs.
    """
    C_A = np.abs(np.array(fit_A["amplitude"], dtype=float))
    x_A = np.arange(len(C_A))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(x_A, C_A, marker="o", alpha=0.8, label=label_A)

    if fit_B is not None:
        C_B = np.abs(np.array(fit_B["amplitude"], dtype=float))
        # Align x by index; if lengths differ, trim to min length
        n = min(len(C_A), len(C_B))
        x_B = np.arange(n) + 0.15  # tiny offset so points don’t fully overlap
        ax.scatter(x_B, C_B[:n], marker="s", alpha=0.8, label=label_B)

    ax.set_xlabel("NV index")
    ax.set_ylabel("Per-NV contrast")
    ax.set_title("Per-NV Fringe Contrast")
    ax.grid(True)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.legend()
    plt.tight_layout()
    plt.show()
# ------------------ EXAMPLE CALL ------------------
if __name__ == "__main__":
    kpl.init_kplotlib()

    fit_spin, fit_xy8, Cmed_spin, Cmed_xy8 = compare_two_runs(
        file_stem_A="2025_10_11-00_03_47-rubin-nv0_2025_09_08",  # spin echo
        label_A="Spin Echo",
        file_stem_B="2025_10_13-14_00_31-rubin-nv0_2025_09_08",   # XY8
        label_B="XY8",
        show_individual=False,  # set True if you want the per-NV scatter panels
    )
    plot_contrast_scatter(fit_spin, "Spin Echo", fit_xy8, "XY8")
    kpl.show(block=True)

    # If you want CSVs of per-NV contrasts:
    # fit_spin.to_csv("spin_echo_perNV_contrast.csv", index=False)
    # fit_xy8.to_csv("xy8_perNV_contrast.csv", index=False)

# if __name__ == "__main__":
#     kpl.init_kplotlib()
#     # file_id = 1817334208399
#     file_id = "2025_03_28-12_53_58-rubin-nv0_2025_02_26"
#     data = dm.get_raw_data(file_stem=file_id, load_npz=True, use_cache=True)
#     nv_list = data["nv_list"]
#     num_nvs = len(nv_list)
#     num_steps = data["num_steps"]
#     num_runs = data["num_runs"]
#     phis = data["phis"]

#     counts = np.array(data["counts"])
#     sig_counts = counts[0]
#     ref_counts = counts[1]

#     norm_counts, norm_counts_ste = widefield.process_counts(
#         nv_list, sig_counts, ref_counts, threshold=True
#     )
#     # file_name = dm.get_file_name(file_id=file_id)
#     # print(f"{file_name}_{file_id}")
#     num_nvs = len(nv_list)
#     phi_step = phis[1] - phis[0]
#     num_steps = len(phis)
#     fit_fig = create_fit_figure(nv_list, phis, norm_counts, norm_counts_ste)
#     # simulate_plot()
#     # simulate_pulse_errors()
#     kpl.show(block=True)
