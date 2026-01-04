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


def plot_spin_echo_all(nv_list, taus, norm_counts, norm_counts_ste, ori = "All", params ="" ):
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
            ax.set_xscale("log")
            ax.set_xticklabels(
                [f"{tick:.2f}" for tick in tick_positions], rotation=45, fontsize=9
            )
            ax.set_xlabel("Time (ns)")

    fig.text(
        0.000,
        0.5,
        "NV$^{-}$ Population",
        va="center",
        rotation="vertical",
        fontsize=12,
    )
    seq_name = "Two-block Hahn Correlation"
    pulse_seq = "π/2 – τ – π – τ – π/2 – T_lag – π/2 – τ – π – τ – π/2"
    seq_variant = "Quadrature readout"
    # params = "τ = 44ns, T_lag ~ [16 ns – 2 µs]"

    fig.suptitle(
        f"{seq_name}({pulse_seq})\n"
        f"{seq_variant} – {ori}({params})",
        fontsize=14,
        y=0.995
    )
    # fig.suptitle(f"XY8 {all_file_ids_str}", fontsize=12, y=0.99)
    fig.tight_layout(pad=0.2, rect=[0.02, 0.01, 0.99, 0.99])



import numpy as np
import matplotlib.pyplot as plt

def is_uniform_axis(x, rtol=1e-2):
    x = np.asarray(x, float)
    dx = np.diff(x)
    return np.std(dx) / np.mean(dx) < rtol

def detrend_rows(Y):
    Y = np.asarray(Y, float)
    return Y - np.mean(Y, axis=1, keepdims=True)

def fft_psd_uniform(T_us, Y):
    """
    T_us: (M,) uniform spacing
    Y: (N,M)
    Returns: f_Hz (K,), psd_mean (K,), psd_all (N,K)
    """
    T_s = T_us * 1e-6
    dt = np.median(np.diff(T_s))
    Yd = detrend_rows(Y)
    w = np.hanning(Yd.shape[1])[None, :]          # window
    Yw = Yd * w

    F = np.fft.rfft(Yw, axis=1)
    f_Hz = np.fft.rfftfreq(Yw.shape[1], d=dt)

    # window-normalized "power" (good enough for peak finding)
    psd = (np.abs(F) ** 2) / np.sum(w**2)
    psd_mean = np.median(psd, axis=0)
    return f_Hz, psd_mean, psd

def lomb_scargle_psd(T_us, Y, fmin_Hz=1.0, fmax_Hz=5e6, nfreq=4000):
    """
    Lomb–Scargle for uneven sampling. Uses a simple implementation.
    Returns: f_Hz (nfreq,), p_mean (nfreq,), p_all (N,nfreq)
    """
    from scipy.signal import lombscargle

    t = np.asarray(T_us) * 1e-6
    Yd = detrend_rows(Y)

    f_Hz = np.linspace(fmin_Hz, fmax_Hz, nfreq)
    w = 2*np.pi*f_Hz

    p_all = np.zeros((Yd.shape[0], nfreq))
    for i in range(Yd.shape[0]):
        p_all[i] = lombscargle(t, Yd[i], w, precenter=False, normalize=True)

    p_mean = np.median(p_all, axis=0)
    return f_Hz, p_mean, p_all

def plot_spectrum(f_Hz, p_mean, title="Spectrum"):
    fig, ax = plt.subplots()
    ax.plot(f_Hz/1e3, p_mean)
    ax.set_xlabel("Frequency (kHz)")
    ax.set_ylabel("Power (arb.)")
    ax.set_title(title)
    # ax.set_xscale("log")
    ax.grid(True, which="both", ls="--", lw=0.5)
    return fig

def demodulate_at_f0(T_us, Y, f0_Hz):
    """
    Returns per-NV complex amplitude z_i = <y_i * exp(-i 2π f0 t)>
    """
    t = np.asarray(T_us) * 1e-6
    Yd = detrend_rows(Y)

    ref = np.exp(-1j * 2*np.pi*f0_Hz * t)[None, :]
    z = np.mean(Yd * ref, axis=1)   # (N,)
    amp = np.abs(z)
    phase = np.angle(z)            # [-pi, pi]
    return z, amp, phase


def pairwise_phase_corr_vs_r(xy_um, phase, nbins=25):
    xy_um = np.asarray(xy_um, float)   # (N,2)
    N = len(phase)

    # pairwise distances + phase correlation
    d_list, c_list = [], []
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(xy_um[i]-xy_um[j])
            c = np.cos(phase[i]-phase[j])
            d_list.append(d); c_list.append(c)

    d_list = np.asarray(d_list); c_list = np.asarray(c_list)
    bins = np.linspace(d_list.min(), d_list.max(), nbins+1)
    centers = 0.5*(bins[:-1]+bins[1:])
    g = np.full(nbins, np.nan)
    for k in range(nbins):
        m = (d_list>=bins[k]) & (d_list<bins[k+1])
        if np.any(m):
            g[k] = np.mean(c_list[m])
    return centers, g

def plot_g_r(r_um, g):
    fig, ax = plt.subplots()
    ax.plot(r_um, g, "o-")
    ax.set_xlabel("Distance r (µm)")
    ax.set_ylabel(r"$\langle \cos(\Delta\phi)\rangle$")
    ax.set_title("Phase-correlation vs distance at f0")
    ax.grid(True, ls="--", lw=0.5)
    return fig


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.signal import lombscargle

# -----------------------------
# Spectrum-based init (works for uneven sampling)
# -----------------------------
def lomb_init_freqs(T_us, y, K=1, fmin_Hz=1e3, fmax_Hz=None, nfreq=8000):
    """
    Return K peak frequencies (Hz) from Lomb–Scargle of a 1D trace y(T).
    Handles non-uniform T.
    """
    t = np.asarray(T_us, float) * 1e-6
    yy = np.asarray(y, float) - np.mean(y)

    if fmax_Hz is None:
        ts = np.sort(t)
        dt_min = np.min(np.diff(ts))
        fmax_Hz = 0.5 / dt_min  # rough upper cap

    f = np.linspace(fmin_Hz, fmax_Hz, nfreq)
    w = 2 * np.pi * f
    p = lombscargle(t, yy, w, precenter=False, normalize=True)

    # pick separated top-K peaks
    idx = np.argsort(p)[::-1]
    picked = []
    for j in idx:
        fj = f[j]
        if all(abs(fj - fk) > 0.05 * fk for fk in picked):  # 5% separation
            picked.append(fj)
        if len(picked) == K:
            break
    if len(picked) < K:
        picked += [picked[-1] if picked else (fmin_Hz + 1.0)] * (K - len(picked))
    return np.array(picked, float)


# -----------------------------
# Per-NV decaying cosine model fits
# -----------------------------
def fit_one_nv(T_us, y, yerr=None, K=1, init_freqs_Hz=None, fmin_Hz=1e3, fmax_Hz=None):
    """
    Fit one NV trace with K=1 or K=2 decaying cosines sharing Tc.
    Uses seconds+Hz internally (avoids ns/us mixups).
    Returns dict with fitted params + diagnostics.
    """
    T_us = np.asarray(T_us, float)
    t = T_us * 1e-6
    y = np.asarray(y, float)

    # weights
    if yerr is None:
        sigma = np.ones_like(y)
    else:
        sigma = np.asarray(yerr, float)
        sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, np.nanmedian(sigma[sigma > 0]))
        sigma = np.maximum(sigma, 1e-12)

    # initial guesses
    C0_0 = float(np.mean(y))
    ptp = float(np.max(y) - np.min(y) + 1e-12)
    A0 = 0.5 * ptp
    Tc0 = 0.5 * (t.max() - t.min())
    Tc0 = max(Tc0, 1e-6)

    if init_freqs_Hz is None:
        init_freqs_Hz = lomb_init_freqs(T_us, y, K=K, fmin_Hz=fmin_Hz, fmax_Hz=fmax_Hz)

    init_freqs_Hz = np.asarray(init_freqs_Hz, float)

    # frequency bounds: keep it reasonably tight around init to prevent nonsense
    def f_bounds(f0):
        # allow ±30% or ±2 MHz, whichever is larger
        span = max(0.30 * f0, 2e6)
        lo = max(fmin_Hz, f0 - span)
        hi = (fmax_Hz if fmax_Hz is not None else (f0 + span))
        return lo, hi

    if K == 1:
        f0 = init_freqs_Hz[0]
        flo, fhi = f_bounds(f0)

        # p = [C0, A, f, Tc, phi]
        p0 = np.array([C0_0, A0, f0, Tc0, 0.0], float)

        lb = np.array([C0_0 - 3*ptp, -10*ptp, flo, 1e-9, -2*np.pi], float)
        ub = np.array([C0_0 + 3*ptp,  10*ptp, fhi, 1e2,   2*np.pi], float)

        def model(p):
            C0, A, f, Tc, phi = p
            env = np.exp(-t / Tc)
            return C0 + A * env * np.cos(2*np.pi*f*t + phi)

        def resid(p):
            return (model(p) - y) / sigma

        res = least_squares(resid, p0, bounds=(lb, ub), loss="soft_l1", f_scale=1.0, max_nfev=20000)
        yhat = model(res.x)

        C0, A, f, Tc, phi = res.x
        out = dict(
            success=bool(res.success),
            C0=C0, A1=A, f1_Hz=f, Tc_s=Tc, phi1=phi,
            rms=float(np.sqrt(np.mean((yhat - y)**2))),
            wrms=float(np.sqrt(np.mean(((yhat - y)/sigma)**2))),
            nfev=int(res.nfev),
            yhat=yhat,
        )
        return out

    else:
        f1, f2 = init_freqs_Hz[:2]
        f1lo, f1hi = f_bounds(f1)
        f2lo, f2hi = f_bounds(f2)

        # p = [C0, A1, f1, phi1, A2, f2, phi2, Tc]
        p0 = np.array([C0_0, 0.7*A0, f1, 0.0, 0.3*A0, f2, 0.0, Tc0], float)

        lb = np.array([C0_0 - 3*ptp, -10*ptp, f1lo, -2*np.pi,
                       -10*ptp, f2lo, -2*np.pi, 1e-9], float)
        ub = np.array([C0_0 + 3*ptp,  10*ptp, f1hi,  2*np.pi,
                        10*ptp, f2hi,  2*np.pi, 1e2], float)

        def model(p):
            C0, A1, f1, phi1, A2, f2, phi2, Tc = p
            env = np.exp(-t / Tc)
            return C0 + env*(A1*np.cos(2*np.pi*f1*t + phi1) + A2*np.cos(2*np.pi*f2*t + phi2))

        def resid(p):
            return (model(p) - y) / sigma

        res = least_squares(resid, p0, bounds=(lb, ub), loss="soft_l1", f_scale=1.0, max_nfev=30000)
        yhat = model(res.x)

        C0, A1, f1, phi1, A2, f2, phi2, Tc = res.x

        # enforce ordering (swap if needed)
        if f2 < f1:
            A1, A2 = A2, A1
            f1, f2 = f2, f1
            phi1, phi2 = phi2, phi1

        out = dict(
            success=bool(res.success),
            C0=C0, A1=A1, f1_Hz=f1, phi1=phi1,
            A2=A2, f2_Hz=f2, phi2=phi2,
            Tc_s=Tc,
            rms=float(np.sqrt(np.mean((yhat - y)**2))),
            wrms=float(np.sqrt(np.mean(((yhat - y)/sigma)**2))),
            nfev=int(res.nfev),
            yhat=yhat,
        )
        return out


def fit_all_nvs(T_us, Y, Yerr=None, K=1, init_freqs_Hz=None, fmin_Hz=1e3, fmax_Hz=None):
    """
    Fit all NVs independently.
    Y: (N,M)
    Returns dict of arrays.
    """
    Y = np.asarray(Y, float)
    N, M = Y.shape
    if Yerr is not None:
        Yerr = np.asarray(Yerr, float)

    # global init from median if not provided
    if init_freqs_Hz is None:
        y_med = np.median(Y, axis=0)
        init_freqs_Hz = lomb_init_freqs(T_us, y_med, K=K, fmin_Hz=fmin_Hz, fmax_Hz=fmax_Hz)

    results = []
    for i in range(N):
        yi = Y[i]
        ei = (Yerr[i] if Yerr is not None else None)
        ri = fit_one_nv(T_us, yi, yerr=ei, K=K, init_freqs_Hz=init_freqs_Hz, fmin_Hz=fmin_Hz, fmax_Hz=fmax_Hz)
        ri["nv_index"] = i
        results.append(ri)

    # pack
    out = {
        "K": K,
        "init_freqs_Hz": np.array(init_freqs_Hz, float),
        "success": np.array([r["success"] for r in results], bool),
        "rms": np.array([r["rms"] for r in results], float),
        "wrms": np.array([r["wrms"] for r in results], float),
        "Tc_us": np.array([r["Tc_s"]*1e6 for r in results], float),
        "C0": np.array([r["C0"] for r in results], float),
        "A1": np.array([r["A1"] for r in results], float),
        "f1_MHz": np.array([r["f1_Hz"]*1e-6 for r in results], float),
        "phi1": np.array([r["phi1"] for r in results], float),
        "raw": results,
    }
    if K == 2:
        out["A2"] = np.array([r["A2"] for r in results], float)
        out["f2_MHz"] = np.array([r["f2_Hz"]*1e-6 for r in results], float)
        out["phi2"] = np.array([r["phi2"] for r in results], float)
    return out


# -----------------------------
# Scatter plots
# -----------------------------
def plot_fit_scatters(res, title_prefix="Per-NV fits", wrms_cut=None):
    idx = np.arange(len(res["success"]))
    good = res["success"].copy()
    if wrms_cut is not None:
        good &= (res["wrms"] < wrms_cut)

    # 1) frequency vs NV index
    plt.figure(figsize=(8,4))
    plt.scatter(idx[good], res["f1_MHz"][good], s=15)
    plt.xlabel("NV index")
    plt.ylabel("f1 (MHz)")
    plt.title(f"{title_prefix}: f1")
    plt.grid(True, ls="--", lw=0.5)

    # 2) amplitude vs NV index
    plt.figure(figsize=(8,4))
    plt.scatter(idx[good], res["A1"][good], s=15)
    plt.xlabel("NV index")
    plt.ylabel("A1 (arb.)")
    plt.title(f"{title_prefix}: A1")
    plt.grid(True, ls="--", lw=0.5)

    # 3) Tc vs NV index
    plt.figure(figsize=(8,4))
    plt.scatter(idx[good], res["Tc_us"][good], s=15)
    plt.xlabel("NV index")
    plt.ylabel("Tc (µs)")
    plt.title(f"{title_prefix}: Tc")
    plt.grid(True, ls="--", lw=0.5)

    # 4) RMS / wRMS diagnostic
    plt.figure(figsize=(8,4))
    plt.scatter(idx, res["wrms"], s=15)
    plt.xlabel("NV index")
    plt.ylabel("wRMS (dimensionless)")
    plt.title(f"{title_prefix}: fit quality (wRMS)")
    plt.grid(True, ls="--", lw=0.5)

    # 5) freq vs amplitude (color by phase)
    plt.figure(figsize=(6,5))
    sc = plt.scatter(res["f1_MHz"][good], res["A1"][good], c=res["phi1"][good], s=25)
    plt.xlabel("f1 (MHz)")
    plt.ylabel("A1 (arb.)")
    plt.title(f"{title_prefix}: f1 vs A1 (color=phi1)")
    plt.grid(True, ls="--", lw=0.5)
    plt.colorbar(sc, label="phi1 (rad)")

    if res["K"] == 2:
        # show second frequency too
        plt.figure(figsize=(8,4))
        plt.scatter(idx[good], res["f2_MHz"][good], s=15)
        plt.xlabel("NV index")
        plt.ylabel("f2 (MHz)")
        plt.title(f"{title_prefix}: f2")
        plt.grid(True, ls="--", lw=0.5)

        plt.figure(figsize=(6,5))
        sc2 = plt.scatter(res["f2_MHz"][good], res["A2"][good], c=res["phi2"][good], s=25)
        plt.xlabel("f2 (MHz)")
        plt.ylabel("A2 (arb.)")
        plt.title(f"{title_prefix}: f2 vs A2 (color=phi2)")
        plt.grid(True, ls="--", lw=0.5)
        plt.colorbar(sc2, label="phi2 (rad)")

import numpy as np
import matplotlib.pyplot as plt

def predict_dense(T_us_dense, fit):
    """Return model prediction at dense times for either K=1 or K=2 fit dict."""
    t = np.asarray(T_us_dense, float) * 1e-6

    C0 = fit["C0"]
    Tc = fit["Tc_s"]
    env = np.exp(-t / Tc)

    if "f2_Hz" in fit:  # K=2
        A1, f1, phi1 = fit["A1"], fit["f1_Hz"], fit["phi1"]
        A2, f2, phi2 = fit["A2"], fit["f2_Hz"], fit["phi2"]
        yhat = C0 + env * (
            A1 * np.cos(2*np.pi*f1*t + phi1) +
            A2 * np.cos(2*np.pi*f2*t + phi2)
        )
    else:  # K=1
        A1, f1, phi1 = fit["A1"], fit["f1_Hz"], fit["phi1"]
        yhat = C0 + A1 * env * np.cos(2*np.pi*f1*t + phi1)

    return yhat

def plot_one_nv_fit(T_us, y, yerr, res, i, dense=1200, xscale="log"):
    """
    res: output of fit_all_nvs(...)
    i: NV index
    """
    fit = res["raw"][i]

    T_us = np.asarray(T_us, float)
    y = np.asarray(y, float)
    if yerr is not None:
        yerr = np.asarray(yerr, float)

    # dense time axis
    if xscale == "log":
        Tmin = np.min(T_us[T_us > 0])
        Tmax = np.max(T_us)
        T_dense = np.geomspace(Tmin, Tmax, dense)
    else:
        T_dense = np.linspace(np.min(T_us), np.max(T_us), dense)

    y_dense = predict_dense(T_dense, fit)

    # title string
    if "f2_Hz" in fit:
        title = (f"NV {i} | f1={fit['f1_Hz']*1e-6:.3f} MHz, "
                 f"f2={fit['f2_Hz']*1e-6:.3f} MHz | "
                 f"Tc={fit['Tc_s']*1e6:.2f} µs | wrms={fit['wrms']:.2f}")
    else:
        title = (f"NV {i} | f={fit['f1_Hz']*1e-6:.3f} MHz | "
                 f"Tc={fit['Tc_s']*1e6:.2f} µs | wrms={fit['wrms']:.2f}")

    fig, ax = plt.subplots(figsize=(7, 5))
    if yerr is not None:
        ax.errorbar(T_us, y, yerr=np.abs(yerr), fmt="o", ms=4, lw=1, capsize=2, label="data")
    else:
        ax.plot(T_us, y, "o", ms=4, label="data")

    ax.plot(T_dense, y_dense, "-", lw=2, label="fit")
    ax.set_title(title)
    ax.set_xlabel("T_lag (µs)")
    ax.set_ylabel("Signal (arb.)")
    ax.grid(True, ls="--", lw=0.5)
    if xscale is not None:
        ax.set_xscale(xscale)
    ax.legend()
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    kpl.init_kplotlib()
    # Process and analyze data from multiple files
    # file_stems = [
    #     "2025_10_15-11_06_09-rubin-nv0_2025_09_08",
    #     "2025_10_15-05_35_19-rubin-nv0_2025_09_08",
    # ]
    ###interpulses gap 200ns
    # file_stems = [
    #     "2025_10_16-00_40_47-rubin-nv0_2025_09_08",
    #     "2025_10_16-05_56_38-rubin-nv0_2025_09_08",
    # ]
    ## interpulses gap 44ns
    file_stems = [
        "2025_10_17-06_30_22-rubin-nv0_2025_09_08",
        "2025_10_17-01_15_46-rubin-nv0_2025_09_08",
    ]
    
    ### interpulses gap 15us
    # file_stems = [
    #     "2026_01_02-00_24_34-johnson-nv0_2025_10_21",
    # ]
    file_stems = [
        "2026_01_03-18_26_05-johnson-nv0_2025_10_21",
    ]

    try:
        data = widefield.process_multiple_files(file_stems, load_npz=True)

        nv_list = data["nv_list"]
        taus_raw = data["lag_taus"]  # could be ns or µs
        # T_us = _auto_to_us(taus_raw)  # ensure µs for the model
        T_us = np.array(taus_raw) / 1e3  # ensure µs for the model
        tau = data["tau"] / 1e3 ##
        
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

        #fmt:off
        # ORI_11m1 = [0, 1, 3, 5, 6, 7, 9, 10, 13, 18, 19, 21, 24, 25, 27, 28, 30, 32, 34, 36, 40, 41, 43, 44, 46, 48, 49, 51, 52, 53, 56, 57, 64, 65, 66, 67, 68, 69, 73, 75, 77, 80, 82, 84, 86, 88, 91, 98, 100, 101, 102, 103, 106, 107, 109, 110, 111, 113, 115, 116, 118, 119, 120, 121, 123, 124, 127, 129, 130, 131, 132, 133, 134, 135, 141, 142, 146, 149, 150, 152, 153, 156, 157, 158, 162, 163, 165, 167, 168, 171, 174, 177, 179, 184, 185, 186, 187, 189, 190, 191, 192, 193, 195, 198, 201, 203]
        # ORI_m111 = [2, 4, 8, 11, 12, 14, 15, 16, 17, 20, 22, 23, 26, 29, 31, 33, 35, 37, 38, 39, 42, 45, 47, 50, 54, 55, 58, 59, 60, 61, 62, 63, 70, 71, 72, 74, 76, 78, 79, 81, 83, 85, 87, 89, 90, 92, 93, 94, 95, 96, 97, 99, 104, 105, 108, 112, 114, 117, 122, 125, 126, 128, 136, 137, 138, 139, 140, 143, 144, 145, 147, 148, 151, 154, 155, 159, 160, 161, 164, 166, 169, 170, 172, 173, 175, 176, 178, 180, 181, 182, 183, 188, 194, 196, 197, 199, 200, 202]
        #fmt:on
        # Keep only in-range indices
        # N_all = len(nv_list)
        # sel = [i for i in indices_217_MHz if 0 <= i < N_all]
        # if len(sel) > 0:
        #     nv_list = [nv_list[i] for i in sel]
        #     norm_counts = norm_counts[sel, :]
        #     norm_counts_ste = norm_counts_ste[sel, :]


        # Ensure increasing T and consistent ordering
        order = np.argsort(T_us)
        T_us_sorted = np.asarray(T_us)[order]
        Y = np.asarray(norm_counts)[:, order]
        T = T_us_sorted
        Yerr = np.asarray(norm_counts_ste)[:, order]

        # after you fit:
        # res = fit_all_nvs(T, Y, Yerr=Yerr, K=1, ...)


        # Decide K:
        # Start with K=1; if you KNOW you have two real peaks (e.g. ~5 and ~9 MHz), set K=2.
        K = 1  # or 2

        # Fit all NVs
        res = fit_all_nvs(T, Y, Yerr=Yerr, K=K, fmin_Hz=1e3)
        i = 0  # any NV index
        for i in range(len(nv_list)):
            plot_one_nv_fit(T_us_sorted, Y[i], Yerr[i], res, i, xscale=None)
            plt.show()
            
        # print("Global init freqs (MHz):", res["init_freqs_Hz"]*1e-6)
        # print("Success rate:", np.mean(res["success"]))

        # Scatter plots
        # plot_fit_scatters(res, title_prefix=f"K={K} decaying-cos fit", wrms_cut=5.0)


        # --- Two-block joint fit ---

        # fit = fit_two_block_pipeline(T_us, norm_counts)
        # print(
        #     f"[Two-block fit ns/MHz] success={fit['success']}, "
        #     f"f0={fit['f0_MHz']:.3f} MHz, Tc={fit['Tc_ns']:.1f} ns, "
        #     f"φ0={fit['phi0_rad']:.2f} rad, RMS={fit['residual_rms']:.4g}"
        # )
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
        # plot_each_nv_fit(
        #     T_us, norm_counts, norm_counts_ste, fit
        # )

        # --- Plots ---
        # plot_phase_hist(fit["phi_i_est_rad"])
        # plot_phase_hist(fit["phi_i_est_rad"])
        # plot_each_nv_fit(T_us, norm_counts, norm_counts_ste, fit)
        # plot_two_block_overlays(T_us, C, fit)
        params = f"Interpulse gap = {tau}µs, T_lag = {min(T_us)-max(T_us)} µs"
        # plot_spin_echo_all(nv_list, T_us, norm_counts, norm_counts_ste, ori= "ORI_11m1", params=params)
        # plot_spin_echo_all(nv_list, T_us, norm_counts, norm_counts_ste)
    except Exception as e:
        print(f"Error occurred: {e}")
        print(traceback.format_exc())

    kpl.show(block=True)
