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

from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import widefield as widefield

import numpy as np
from scipy.optimize import curve_fit
# =========================
# Two-block correlation: frequency guess via periodogram
# Uses Lomb–Scargle for uneven T; FFT if uniform.
# =========================

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter
from dataclasses import dataclass

try:
    from scipy.signal import lombscargle
except ImportError:
    lombscargle = None  # will raise a helpful error later

# ---------- helpers ----------
def to_seconds(T_axis, time_unit="auto"):
    T = np.asarray(T_axis, dtype=float)
    if time_unit == "s":
        return T, "s"
    if time_unit == "us":
        return T * 1e-6, "µs"
    if time_unit == "ns":
        return T * 1e-9, "ns"
    # auto guess ...
    if np.nanmax(T) > 1.0:
        return T, "s"
    return T, "s"


def is_uniform_grid(Ts, tol=0.05):
    Ts = np.asarray(Ts)
    d = np.diff(np.sort(Ts))
    if len(d) == 0:
        return True
    return (np.std(d) / np.mean(d)) < tol

def next_pow2(n):
    return 1 << (int(np.ceil(np.log2(max(n,1)))))

def detrend(y, w=None):
    y = np.asarray(y, dtype=float)
    if w is None:
        return y - np.nanmean(y)
    w = np.asarray(w, dtype=float)
    m = np.isfinite(y) & np.isfinite(w) & (w>0)
    if not np.any(m):
        return y - np.nanmean(y)
    yw = y[m]; ww = w[m]
    mu = np.sum(yw*ww) / np.sum(ww)
    out = y.copy()
    out[m] = y[m] - mu
    return out

@dataclass
class PeakSpec:
    nv_index: int
    f_peak: float | None   # Hz
    P_peak: float | None
    f_top: list            # top-k freqs (Hz)
    P_top: list            # top-k powers
    method: str            # 'LS' or 'FFT'

# ---------- core spectrum per NV ----------
def spectrum_for_nv(T_s, C, C_err=None, fmin=None, fmax=None, oversample=5, topk=3):
    """
    T_s: seconds (1D)
    C: contrast array (1D)
    C_err: optional errors (used only to weight detrend)
    returns: freqs (Hz), power, peak info (PeakSpec)
    """
    m = np.isfinite(T_s) & np.isfinite(C)
    T = np.asarray(T_s)[m]; y = np.asarray(C)[m]
    if C_err is not None:
        w = 1.0 / (np.asarray(C_err)[m]**2 + 1e-30)
    else:
        w = None

    if T.size < 3:
        return np.array([]), np.array([]), PeakSpec(-1, None, None, [], [], "NA")

    # detrend/center
    y0 = detrend(y, w=w)

    # frequency grid bounds
    Tspan = T.max() - T.min()
    if Tspan <= 0:
        return np.array([]), np.array([]), PeakSpec(-1, None, None, [], [], "NA")

    # min dt for Nyquist-ish limit
    dT_min = np.min(np.diff(np.sort(T)))
    # defaults if not provided
    fmin = 0.0 if fmin is None else max(0.0, float(fmin))
    # slight guard band below Nyquist
    fmax_default = 0.5 / dT_min if dT_min > 0 else 1.0 / Tspan
    fmax = float(fmax_default if fmax is None else fmax)

    # at least a few cycles across span
    Nf = int(max(256, oversample * len(T)))

    if is_uniform_grid(T):
        # --- FFT route ---
        method = "FFT"
        # sort by time; uniform assumed
        order = np.argsort(T); T = T[order]; y0 = y0[order]
        dt = np.median(np.diff(T))
        # window
        win = np.hanning(len(y0))
        y_win = y0 * win
        nfft = next_pow2(len(y_win) * 4)
        Y = np.fft.rfft(y_win, n=nfft)
        freqs = np.fft.rfftfreq(nfft, d=dt)
        P = (np.abs(Y)**2) / np.sum(win**2)
        # clip to requested band
        mband = (freqs >= fmin) & (freqs <= fmax)
        freqs = freqs[mband]; P = P[mband]
    else:
        # --- Lomb–Scargle for uneven T ---
        if lombscargle is None:
            raise ImportError("scipy.signal.lombscargle not available. Please install SciPy.")
        method = "LS"
        freqs = np.linspace(max(fmin, 1.0/Tspan), fmax, Nf)
        # scipy’s lombscargle takes angular frequency (rad/s)
        omega = 2*np.pi*freqs
        # normalize by variance to make power comparable
        yvar = np.var(y0) if np.var(y0) > 0 else 1.0
        P = lombscargle(T, y0, omega, precenter=True, normalize=True) * yvar

    # pick peaks
    if len(P) == 0:
        return freqs, P, PeakSpec(-1, None, None, [], [], method)

    # top-k indices
    idx_sorted = np.argsort(P)[::-1][:max(topk,1)]
    f_top = freqs[idx_sorted].tolist()
    P_top = P[idx_sorted].tolist()
    fpk = f_top[0]; Ppk = P_top[0]
    return freqs, P, PeakSpec(-1, fpk, Ppk, f_top, P_top, method)

# ---------- batch & plotting ----------
def run_two_block_spectrum(nv_list, T_axis, norm_counts, norm_counts_ste=None,
                           time_unit="us", fmin=None, fmax=None,
                           logx=True, ncols=7, figsize_scale=(2.0,3.0), topk=3):
    """
    Computes per-NV spectra and plots a grid of power vs frequency.
    Returns (peaks_df, all_freqs, all_powers).
    """
    T_s, tlabel = to_seconds(T_axis, time_unit=time_unit)
    nNV = len(nv_list)
    sns.set(style="whitegrid", palette="deep")

    # collect spectra/peaks
    all_freqs, all_P, peaks = [], [], []
    for i in range(nv_list if isinstance(nv_list, int) else len(nv_list)):
        freqs, P, pk = spectrum_for_nv(T_s, norm_counts[i],
                                       None if norm_counts_ste is None else norm_counts_ste[i],
                                       fmin=fmin, fmax=fmax, topk=topk)
        peaks.append(pk)
        all_freqs.append(freqs)
        all_P.append(P)

    # grid plot
    ncols = int(ncols)
    nrows = int(np.ceil(nNV / ncols))
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols*figsize_scale[0], nrows*figsize_scale[1]),
        sharex=False, sharey=False, constrained_layout=True,
        gridspec_kw={"wspace": 0.05, "hspace": 0.05}
    )
    axes = axes.flatten() if nrows*ncols>1 else [axes]
    colors = sns.color_palette("deep", nNV)

    for i, ax in enumerate(axes):
        if i >= nNV:
            ax.axis("off"); continue
        f = all_freqs[i]; P = all_P[i]
        if len(f) == 0:
            ax.text(0.5,0.5,"no data", ha="center", va="center"); continue
        ax.plot(f/1e6, P, lw=1.2, color=colors[i % len(colors)])
        if peaks[i].f_peak is not None:
            ax.axvline(peaks[i].f_peak/1e6, linestyle="--", alpha=0.5, color=colors[i % len(colors)])
        if logx:
            ax.set_xscale("log")
        ax.grid(True, which="both", ls="--", lw=0.5, alpha=0.6)
        ax.set_title(f"NV {i}", fontsize=9)
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2g"))
        # label bottom row x-axis in MHz
        row = i // ncols
        if row == (nrows - 1):
            ax.set_xlabel("Frequency (MHz)", fontsize=9)
        ax.tick_params(axis="x", labelsize=8)
        ax.tick_params(axis="y", labelsize=8)

    fig.text(0.005, 0.5, "Power (arb.)", va="center", rotation="vertical", fontsize=12)
    fig.suptitle("Two-block correlation spectrum (per NV)", y=1.02)
    plt.show()

    # summary table
    import pandas as pd
    rows = []
    for i, pk in enumerate(peaks):
        rows.append({
            "NV": i,
            "Method": pk.method,
            "f_peak (Hz)": pk.f_peak,
            "f_peak (MHz)": None if pk.f_peak is None else pk.f_peak/1e6,
            "P_peak": pk.P_peak,
            "Top_k_freqs_MHz": [x/1e6 for x in pk.f_top],
            "Top_k_powers": pk.P_top
        })
    df = pd.DataFrame(rows).sort_values("NV").reset_index(drop=True)
    with pd.option_context("display.max_columns", None, "display.width", 120):
        print(df)
    return df, all_freqs, all_P



def plot_spin_echo_all(nv_list, taus, norm_counts, norm_counts_ste):
    fig, ax = plt.subplots()
    # Scatter plot with error bars
    print(norm_counts.shape)
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
    fig.tight_layout()
    # plt.show(block=True)

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
    file_stems = ["2025_10_15-11_06_09-rubin-nv0_2025_09_08",
                  "2025_10_15-05_35_19-rubin-nv0_2025_09_08",
                  ]
    try:
        data = widefield.process_multiple_files(file_stems, load_npz=True)
        # data = dm.get_raw_data(file_stem=file_stem, load_npz=False, use_cache=False)
        # nv_list = data["nv_list"]
        # taus = data["lag_taus"]
        # rabi_feq = data["config"]["Microwaves"]["VirtualSigGens"][str(1)]["uwave_power"]
        # rabi_period = data["config"]["Microwaves"]["VirtualSigGens"][str(1)][
        #     "rabi_period"
        # ]
        # print(f"rabi freq:{rabi_feq}, rabi period: {rabi_period}")
        # counts = np.array(data["counts"])
        # sig_counts, ref_counts = counts[0], counts[1]
        # norm_counts, norm_counts_ste = widefield.process_counts(
        #     nv_list, sig_counts, ref_counts, threshold=True
        # )
        # norm_counts, norm_counts_ste = widefield.process_counts(
        #     nv_list, sig_counts, ref_counts, threshold=True
        # )
        # fit_fns, popts = fit_spin_echo(
        #     nv_list, total_evolution_times, norm_counts, norm_counts_ste
        # )
        # plot_spin_echo_fits(
        #     nv_list, total_evolution_times, norm_counts, norm_counts_ste
        # )
        # Assuming your loaded structures from widefield.*:
        nv_list = data["nv_list"]
        taus = data["lag_taus"]             # if these are µs, pass time_unit='us' below
        counts = np.array(data["counts"])
        sig_counts, ref_counts = counts[0], counts[1]

        norm_counts, norm_counts_ste = widefield.process_counts(
            nv_list, sig_counts, ref_counts, threshold=True
        )

        # Run end-to-end (set time_unit='us' if taus are in microseconds)
        # df, fit_results = run_two_block_hahn_analysis(
        #     nv_list=nv_list,
        #     T_axis=taus,
        #     norm_counts=norm_counts,
        #     norm_counts_ste=norm_counts_ste,
        #     time_unit='us',          # <-- change to 's' if your taus are already in seconds
        #     logx=False,
        #     ncols=7,
        #     title="Two-block Hahn correlation per NV",
        #     n_jobs=-1,               # parallel
        # )

        # after you build nv_list, taus (T), norm_counts, norm_counts_ste:
        df_spec, freqs_all, P_all = run_two_block_spectrum(
            nv_list=nv_list,
            T_axis=taus,
            norm_counts=norm_counts,
            norm_counts_ste=norm_counts_ste,
            time_unit='ns', 
            fmin=1e5, fmax=2e7,
            logx=False,
            ncols=7,
            topk=3
        )
        # plot_spin_echo_all(nv_list, taus, norm_counts, norm_counts_ste)
    except Exception as e:
        print(f"Error occurred: {e}")
        print(traceback.format_exc())

    kpl.show(block=True)
