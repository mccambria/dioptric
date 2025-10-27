# -*- coding: utf-8 -*-
"""
Spin Echo Analysis and Visualization

Created on December 22nd, 2024

@author: Saroj Chand
"""

import sys
import time
import traceback
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from joblib import Parallel, delayed
from matplotlib.ticker import FormatStrFormatter
from scipy.optimize import curve_fit
from numpy.linalg import lstsq
from scipy.optimize import least_squares

from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import widefield as widefield


# -----------------------------
# Models & helpers
# -----------------------------
def c2_block_model(T_us, C0_i, s_i, A_i, f0_kHz, Tc_us, phi0_rad):
    """
    Single-NV two-block Hahn correlation model:
        C_i(T) = C0_i + s_i * A_i * exp(-T/Tc) * cos(2π f0 T + phi0)
    Units:
      - T_us: microseconds
      - f0_kHz: kHz  (so f0_kHz * 1e-3 gives µs^-1)
      - Tc_us: microseconds
      - phi0_rad: radians
    """
    omega_us = 2 * np.pi * (f0_kHz * 1e-3)  # convert kHz -> µs^-1
    return C0_i + s_i * A_i * np.exp(-T_us / Tc_us) * np.cos(omega_us * T_us + phi0_rad)


def estimate_phi_per_nv(T_ns, Ci):
    """
    Quick per-NV phase estimate using FFT freq guess (ns/MHz version).
    Returns (f0_MHz, phi_i).
    """
    x = Ci - np.mean(Ci)
    dt_s = (T_ns[1] - T_ns[0]) * 1e-9
    freqs_Hz = np.fft.rfftfreq(len(T_ns), d=dt_s)
    amp = np.abs(np.fft.rfft(x))
    if len(amp) <= 1:
        f0_Hz = 1e6  # fallback 1 MHz
    else:
        f0_Hz = freqs_Hz[np.argmax(amp[1:]) + 1]
    f0_MHz = f0_Hz / 1e6

    # Linear LS on cos/sin at that frequency
    w_ns = 2 * np.pi * (f0_MHz * 1e-3)  # rad per ns
    X = np.column_stack([np.cos(w_ns * T_ns), np.sin(w_ns * T_ns), np.ones_like(T_ns)])
    a, b, _ = lstsq(X, Ci, rcond=None)[0]
    phi = np.arctan2(-b, a)  # a cos + b sin = R cos(wT + phi)
    return f0_MHz, phi


def phase_cluster_signs(T_ns, C):
    """
    Returns:
      s (N,) in {+1,-1}, f0_guess_MHz (median of per-NV), phis (per-NV)
    """
    f0s, phis = [], []
    for i in range(C.shape[0]):
        f0_i, phi_i = estimate_phi_per_nv(T_ns, C[i])
        f0s.append(f0_i)
        phis.append(phi_i)
    f0_guess = float(np.median(f0s))
    phis = np.unwrap(np.array(phis))
    s = np.sign(np.cos(phis))
    s[s == 0] = 1
    return s.astype(int), f0_guess, phis


def pack_params(C0, A, f0_MHz, Tc_ns, phi0_rad):
    return np.concatenate([C0, A, np.array([f0_MHz, Tc_ns, phi0_rad])])


def unpack_params(p, N):
    C0 = p[:N]
    A = p[N : 2 * N]
    f0_MHz, Tc_ns, phi0_rad = p[-3:]
    return C0, A, f0_MHz, Tc_ns, phi0_rad


def residuals_joint_nsMHz(p, T_ns, C, s):
    N, M = C.shape
    C0, A, f0_MHz, Tc_ns, phi0_rad = unpack_params(p, N)

    # Guard rails inside iterations
    Tc_ns = max(Tc_ns, 1e-3)
    f0_MHz = max(f0_MHz, 0.01)

    exp_env = np.exp(-T_ns / Tc_ns)  # (M,)
    cos_term = np.cos(2 * np.pi * (f0_MHz * 1e-3) * T_ns + phi0_rad)  # (M,)

    model = (
        C0[:, None] + (s[:, None] * A[:, None]) * exp_env[None, :] * cos_term[None, :]
    )
    return (model - C).ravel()


def joint_fit_two_block(T_ns, C, s, f0_guess_MHz=None, Tc_guess_ns=None):
    N, M = C.shape

    C0_0 = C.mean(axis=1)
    A_0 = 0.6 * np.maximum(1e-12, (C.max(axis=1) - C.min(axis=1)))

    if f0_guess_MHz is None:
        # global FFT on median-detrended
        X = C - C.mean(axis=1, keepdims=True)
        x_med = np.median(X, axis=0)
        dt_s = (T_ns[1] - T_ns[0]) * 1e-9
        freqs_Hz = np.fft.rfftfreq(M, d=dt_s)
        amp = np.abs(np.fft.rfft(x_med))
        f0_guess_MHz = (freqs_Hz[np.argmax(amp[1:]) + 1] / 1e6) if len(amp) > 1 else 1.0

    if Tc_guess_ns is None:
        Tc_guess_ns = max(0.5 * (T_ns.max() - T_ns.min()), 1.0)

    phi0_guess = 0.0
    p0 = pack_params(C0_0, A_0, f0_guess_MHz, Tc_guess_ns, phi0_guess)

    lb = np.concatenate(
        [
            C.min(axis=1) - 0.2 * np.ptp(C, axis=1),  # C0_i lower
            0.0 * np.ones(N),  # A_i >= 0
            np.array([0.01, 1.0, -2 * np.pi]),  # f0>=0.01 MHz, Tc>=1 ns
        ]
    )
    ub = np.concatenate(
        [
            C.max(axis=1) + 0.2 * np.ptp(C, axis=1),  # C0_i upper
            10.0 * np.ptp(C, axis=1),  # generous A_i
            np.array([500.0, 1e9, 2 * np.pi]),  # f0<=500 MHz, Tc up to 1e9 ns
        ]
    )

    res = least_squares(
        residuals_joint_nsMHz,
        p0,
        bounds=(lb, ub),
        args=(T_ns, C, s),
        max_nfev=20000,
        verbose=0,
    )

    C0, A, f0_MHz, Tc_ns, phi0_rad = unpack_params(res.x, N)
    out = {
        "success": res.success,
        "cost": res.cost,
        "C0": C0,
        "A": A,
        "s": s,
        "f0_MHz": f0_MHz,
        "Tc_ns": Tc_ns,
        "phi0_rad": (phi0_rad + np.pi) % (2 * np.pi) - np.pi,
        "residual_rms": float(np.sqrt(np.mean(res.fun**2))),
        "nfev": res.nfev,
    }
    return out


def fit_two_block_pipeline(T_ns, C):
    s, f0_guess_MHz, phis = phase_cluster_signs(T_ns, C)
    fit = joint_fit_two_block(T_ns, C, s, f0_guess_MHz, Tc_guess_ns=None)
    fit["phi_i_est_rad"] = phis
    return fit


# -----------------------------
# Minimal plotting/QA
# -----------------------------


def plot_two_block_overlays(T_us, C, fit):
    C0, A, s = fit["C0"], fit["A"], fit["s"]
    f0, Tc, phi0 = fit["f0_kHz"], fit["Tc_us"], fit["phi0_rad"]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(T_us, np.median(C, axis=0), "o", ms=3, label="Median data")
    model_med = np.median(
        [
            c2_block_model(T_us, C0[i], s[i], A[i], f0, Tc, phi0)
            for i in range(C.shape[0])
        ],
        axis=0,
    )
    ax.plot(T_us, model_med, "-", lw=2, label="Median model")
    ax.set_xlabel("T (µs)")
    ax.set_ylabel("Correlation C(T)")
    ax.set_title(f"Joint fit: f0={f0:.2f} kHz, Tc={Tc:.1f} µs, φ0={phi0:.2f} rad")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()


def plot_phase_hist(phis):
    phis = (np.array(phis) + np.pi) % (2 * np.pi) - np.pi
    plt.figure(figsize=(6, 5))
    plt.hist(phis, bins=24)
    plt.xlabel("φ_i (rad)")
    plt.ylabel("count")
    plt.title("Quick per-NV phase (two peaks near 0 and π)")
    plt.tight_layout()


def plot_each_nv_fit(T_ns, C, C_ste, fit, pause=0.0, save_dir=None):
    """
    Loop over NVs and show a fit overlay per plot with dense tau.
    """
    C0, A, s = fit["C0"], fit["A"], fit["s"]
    f0, Tc, phi0 = fit["f0_MHz"], fit["Tc_ns"], fit["phi0_rad"]

    for i in range(C.shape[0]):
        Ci = C[i]

        # Make tau dense for smooth curve
        tau_dense = np.linspace(T_ns.min(), T_ns.max(), 500)  # e.g. 500 points
        model_i = c2_block_model(tau_dense, C0[i], s[i], A[i], f0, Tc, phi0)

        fig, ax = plt.subplots(figsize=(7, 5))

        # Plot experimental data
        if C_ste is not None:
            ax.errorbar(
                T_ns,
                Ci,
                yerr=np.abs(C_ste[i]),
                fmt="o",
                ms=4,
                lw=1,
                label=f"NV {i} data",
            )
        else:
            ax.plot(T_ns, Ci, "o", ms=4, label=f"NV {i} data")

        # Plot smooth fit curve
        ax.plot(tau_dense, model_i, "-", lw=2, label="Fit model")

        ax.set_xlabel("T (ns)")
        ax.set_ylabel("Correlation C(T)")
        ax.set_title(
            f"NV {i} | f0={f0:.3f} MHz, Tc={Tc:.1f} ns, φ0={phi0:.2f} rad, s={s[i]:+d}",
            fontsize=15,
        )
        ax.grid(True)
        ax.legend()

        plt.show(block=True)  # or pause if you want interactive stepping


# -----------------------------
# Main
# -----------------------------


def _auto_to_us(T_axis):
    """
    Convert provided time axis to microseconds.
    Heuristic: if max(T) > 1e4 assume ns; if < 1e3 likely already µs.
    """
    T_axis = np.asarray(T_axis, dtype=float)
    if T_axis.max() > 1e4:  # looks like ns
        return T_axis / 1e3  # ns -> µs
    return T_axis  # assume already µs


def plot_spin_echo_all(nv_list, taus, norm_counts, norm_counts_ste):
    fig, ax = plt.subplots()
    # Scatter plot with error bars
    # print(norm_counts.shape)
    median_counts = np.median(norm_counts, axis=0)
    median_counts_ste = np.median(norm_counts_ste, axis=0)
    ax.errorbar(
        taus,
        median_counts,
        yerr=np.abs(median_counts_ste),
        fmt="o",
    )
    # Plot the fitted curve if available
    title = f"Median across {len(nv_list)} NVs"
    ax.set_title(title)
    ax.set_xlabel("Total Evolution time (us)")
    ax.set_ylabel("Norm. NV- Population")
    ax.grid(True)
    # plt.show(block=True)
    ### Indivudual NV plots
    # for nv_idx in range(len(nv_list)):
    #     nv_tau = taus  # Convert to µs
    #     nv_counts = norm_counts[nv_idx]
    #     nv_counts_ste = norm_counts_ste[nv_idx]
    #     # Plot data and fit on full plot
    #     fig, ax = plt.subplots()
    #     ax.errorbar(
    #         taus,
    #         median_counts,
    #         yerr=np.abs(nv_counts_ste),
    #         fmt="o",
    #     )
    #     title = f"NV {nv_idx}"
    #     ax.set_title(title)
    #     ax.set_xlabel("Total Evolution time (us)")
    #     ax.set_ylabel("Norm. NV- Population")
    #     ax.grid(True)
    #     plt.show(block=True)
    # return

    sns.set(style="whitegrid", palette="muted")
    num_nvs = len(nv_list)
    colors = sns.color_palette("deep", num_nvs)
    num_cols = 7
    num_rows = int(np.ceil(len(nv_list) / num_cols))

    # Full plot
    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=(num_cols * 2, num_rows * 3),
        sharex=True,
        sharey=False,
        constrained_layout=True,
        gridspec_kw={"wspace": 0.0, "hspace": 0.0},
    )
    axes = axes.flatten()

    for nv_idx, ax in enumerate(axes):
        if nv_idx >= len(nv_list):
            ax.axis("off")
            continue

        nv_tau = taus  # Convert to µs
        nv_counts = norm_counts[nv_idx]
        # Plot data and fit on full plot
        sns.lineplot(
            x=nv_tau,
            y=nv_counts,
            ax=ax,
            color=colors[nv_idx % len(colors)],
            lw=0,
            marker="o",
            markersize=3,
            # label=f"NV {nv_idx}",
        )
        ax.errorbar(
            nv_tau,
            norm_counts[nv_idx],
            yerr=abs(norm_counts_ste[nv_idx]),
            fmt="none",
            lw=1.5,
            ecolor=colors[nv_idx % len(colors)],
            alpha=0.9,
        )
        # ax.legend(fontsize="xx-small")
        ax.grid(True, which="both", linestyle="--", linewidth=0.5)
        # ax.tick_params(labelleft=False)
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        ax.tick_params(axis="y", labelsize=8, direction="in", pad=-10)
        for label in ax.get_yticklabels():
            label.set_horizontalalignment("right")
            label.set_x(0.02)  # Fine-tune this as needed
            label.set_zorder(100)

    # Set xticks only for bottom row
    for col in range(num_cols):
        bottom_row_idx = num_rows * num_cols - num_cols + col
        if bottom_row_idx < len(nv_list):
            ax = axes[bottom_row_idx]
            tick_positions = np.linspace(min(taus) + 2, max(taus) - 2, 6)
            ax.set_xticks(tick_positions)
            # ax.set_xscale("log")
            ax.set_xticklabels(
                [f"{tick:.2f}" for tick in tick_positions], rotation=45, fontsize=9
            )
            ax.set_xlabel("Time (µs)")

    fig.text(
        0.000,
        0.5,
        "NV$^{-}$ Population",
        va="center",
        rotation="vertical",
        fontsize=12,
    )
    # fig.suptitle(f"XY8 {all_file_ids_str}", fontsize=12, y=0.99)
    fig.tight_layout(pad=0.2, rect=[0.02, 0.01, 0.99, 0.99])


if __name__ == "__main__":
    kpl.init_kplotlib()

    # Process and analyze data from multiple files
    file_stems = [
        "2025_10_15-11_06_09-rubin-nv0_2025_09_08",
        "2025_10_15-05_35_19-rubin-nv0_2025_09_08",
    ]

    try:
        data = widefield.process_multiple_files(file_stems, load_npz=True)

        nv_list = data["nv_list"]
        taus_raw = data["lag_taus"]  # could be ns or µs
        T_us = _auto_to_us(taus_raw)  # ensure µs for the model

        counts = np.array(data["counts"])
        sig_counts, ref_counts = counts[0], counts[1]

        norm_counts, norm_counts_ste = widefield.process_counts(
            nv_list, sig_counts, ref_counts, threshold=True
        )
        norm_counts = np.asarray(norm_counts)  # shape (N, M)
        norm_counts_ste = np.asarray(norm_counts_ste)  # shape (N, M)

        # --- Optional: select a subset of NVs (ensure indices exist) ---
        # fmt:off
        # indices_113_MHz = [1, 3, 6, 10, 14, 16, 17, 19, 23, 24, 25, 26, 27, 32, 33, 34, 35, 37, 38, 41, 49, 50, 51, 53, 54, 55, 60, 62, 63, 64, 66, 67, 68, 70, 72, 73, 74, 75, 76, 78, 80, 81, 82, 83, 84, 86, 88, 90, 92, 93, 95, 96, 99, 100, 101, 102, 103, 105, 108, 109, 111, 113, 114]
        # indices_217_MHz = [2, 4, 5, 7, 8, 9, 11, 12, 13, 15, 18, 20, 21, 22, 28, 29, 30, 31, 36, 39, 40, 42, 43, 44, 45, 46, 47, 48, 52, 56, 57, 58, 59, 61, 65, 69, 71, 77, 79, 85, 87, 89, 91, 94, 97, 98, 104, 106, 107, 110, 112, 115, 116, 117]
        # fmt:on

        # Keep only in-range indices
        # N_all = len(nv_list)
        # sel = [i for i in indices_217_MHz if 0 <= i < N_all]
        # if len(sel) > 0:
        #     nv_list = [nv_list[i] for i in sel]
        #     norm_counts = norm_counts[sel, :]
        #     norm_counts_ste = norm_counts_ste[sel, :]

        # --- Two-block joint fit ---

        # fit = fit_two_block_pipeline(T_us, norm_counts)
        # print(
        #     f"[Two-block fit ns/MHz] success={fit['success']}, "
        #     f"f0={fit['f0_MHz']:.3f} MHz, Tc={fit['Tc_ns']:.1f} ns, "
        #     f"φ0={fit['phi0_rad']:.2f} rad, RMS={fit['residual_rms']:.4g}"
        # )

        # Per-NV plots (step through one by one)
        # plot_each_nv_fit(
        #     T_us, norm_counts, norm_counts_ste, fit, pause=0.0, save_dir=None
        # )

        # --- Plots ---
        # plot_phase_hist(fit["phi_i_est_rad"])
        # plot_each_nv_fit(T_us, norm_counts, norm_counts_ste, fit)
        # plot_two_block_overlays(T_us, C, fit)
        plot_spin_echo_all(nv_list, T_us, norm_counts, norm_counts_ste)
    except Exception as e:
        print(f"Error occurred: {e}")
        print(traceback.format_exc())

    kpl.show(block=True)
