# -*- coding: utf-8 -*-
"""
Spin-echo: finer fit + fitted-figure + parameter panels

- Physics-y comb with quartic lobes, amplitude taper, width growth, chirp
- Optional two-frequency sin^2 beating with phases
- Smoothly plugs into your plotting + data pipeline

Author: @saroj chand
"""

import os
import sys
import traceback
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import inspect
import re, hashlib, datetime as dt
import itertools, random
import pandas as pd
import matplotlib.pyplot as plt, matplotlib.ticker as mticker


# --- Optional numba (falls back gracefully) ----------------------------------
try:
    from numba import njit
except Exception:
    def njit(*_args, **_kwargs):
        def wrap(fn):
            return fn
        return wrap
# --- Your utilities (assumed in PYTHONPATH) ----------------------------------
from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import widefield as widefield
from utils.tool_belt import curve_fit
from scipy.optimize import least_squares
from analysis.spin_echo_work.echo_plot_helpers import extract_T2_freqs_and_errors, params_to_dict, plot_individual_fits

# =============================================================================
# Finer model
#    - COMB has NO overall comb_contrast
#    - MOD carries the overall amplitude (and optional beating)
# =============================================================================

from analysis.spin_echo_work.echo_fit_models import fine_decay, fine_decay_fixed_revival

# ==========================================
#  helper: decide which NVs to keep based on T2_ms
# ==========================================

def _t2_keep_mask(t2_ms,
                  method="iqr",          # "iqr" | "mad" | "z" | None (combine with abs_range if you want)
                  iqr_k=1.5,             # IQR multiplier (1.5 classic, 3.0 stricter)
                  mad_k=3.5,             # MAD multiplier (≈3–4 is common)
                  z_thresh=4.0,          # |z| threshold
                  abs_range=None,        # e.g. (0.01, 50.0) ms   -> keep only inside this range
                  finite_only=True):
    """
    Returns a boolean mask of same length as t2_ms where True = keep.
    Combines: finite filter, optional abs_range, and one robust method.
    """
    t2 = np.asarray(t2_ms, float)
    keep = np.ones_like(t2, dtype=bool)

    if finite_only:
        keep &= np.isfinite(t2)

    if abs_range is not None:
        lo, hi = abs_range
        keep &= (t2 >= lo) & (t2 <= hi)

    # Robust method (computed only on currently "keep" values)
    x = t2[keep]
    if x.size == 0:
        return keep  # nothing left anyway

    if method == "iqr":
        q1, q3 = np.nanpercentile(x, [25, 75])
        iqr = q3 - q1
        lo = q1 - iqr_k * iqr
        hi = q3 + iqr_k * iqr
        keep_subset = (x >= lo) & (x <= hi)

    elif method == "mad":
        med = np.nanmedian(x)
        mad = np.nanmedian(np.abs(x - med)) + 1e-12
        # robust z ~ 0.6745*(x-med)/MAD   -> use |robust z| <= mad_k
        rz = 0.6745 * (x - med) / mad
        keep_subset = np.abs(rz) <= mad_k

    elif method == "z":
        mu = np.nanmean(x)
        sd = np.nanstd(x) + 1e-12
        z = (x - mu) / sd
        keep_subset = np.abs(z) <= z_thresh

    else:
        # No robust method; rely only on finite/abs_range
        return keep

    # write back into full mask at the positions that were "keep" so far
    idx = np.where(keep)[0]
    keep[idx] = keep_subset
    return keep


_UNIFIED_KEYS = [
    "baseline", "comb_contrast", "revival_time_us", "width0_us", "T2_ms", "T2_exp",
    "amp_taper_alpha", "width_slope", "revival_chirp",
    "osc_contrast", "osc_f0", "osc_f1", "osc_phi0", "osc_phi1"
]

def _normalize_popt_to_unified(p):
    q = np.full(14, np.nan, float)
    L = len(p); p = np.asarray(p, float)
    if L == 6:      # variable core, no extras
        q[0:6] = p[0:6]
    elif L == 5:    # fixed core, no extras
        q[0] = p[0]; q[1] = p[1]; q[2] = np.nan; q[3] = p[2]; q[4] = p[3]; q[5] = p[4]
    elif L == 14:   # variable + extras
        q[0:6] = p[0:6]; q[6:] = p[6:14]
    elif L == 13:   # fixed + extras
        q[0] = p[0]; q[1] = p[1]; q[2] = np.nan; q[3] = p[2]; q[4] = p[3]; q[5] = p[4]
        q[6:] = p[5:13]
    else:
        if L >= 6:
            q[0:6] = p[0:6]
            if L > 6:
                m = min(8, L-6); q[6:6+m] = p[6:6+m]
        elif L >= 5:
            q[0] = p[0]; q[1] = p[1]; q[2] = np.nan; q[3] = p[2]; q[4] = p[3]; q[5] = p[4]
    return q

# ==========================================
# 2) Parameter panels with full-length mask
#    (T2-based outlier rejection)
# ==========================================

def plot_each_param_separately(popts, chi2_list,
                               fit_nv_labels,
                               save_prefix=None,
                               include_trend=True,
                               bins=30,
                               t2_policy=dict(method="iqr", iqr_k=1.5,
                                              abs_range=None, mad_k=3.5,
                                              z_thresh=4.0, finite_only=True)):
    valid = [(i, p, chi2_list[i] if i < len(chi2_list) else np.nan)
             for i, p in enumerate(popts) if p is not None]
    if not valid:
        print("No successful fits.")
        return [], np.zeros(len(popts), bool), np.array([], int)

    uni_rows, x_labels, chi2_ok, positions = [], [], [], []
    for i, p, chi in valid:
        uni_rows.append(_normalize_popt_to_unified(p))
        x_labels.append(fit_nv_labels[i])   # <— use the provided labels!
        chi2_ok.append(chi)
        positions.append(i)

    arr      = np.vstack(uni_rows)
    x_labels = np.asarray(x_labels)
    chi2_ok  = np.asarray(chi2_ok, float)
    positions= np.asarray(positions, int)

    # T2 filter on the valid subset
    t2 = arr[:, 4]
    keep_valid = _t2_keep_mask(
        t2,
        method=t2_policy.get("method", "iqr"),
        iqr_k=t2_policy.get("iqr_k", 1.5),
        mad_k=t2_policy.get("mad_k", 3.5),
        z_thresh=t2_policy.get("z_thresh", 4.0),
        abs_range=t2_policy.get("abs_range", None),
        finite_only=t2_policy.get("finite_only", True)
    )

    full_mask = np.zeros(len(popts), dtype=bool)
    full_mask[positions] = keep_valid

    arr_f      = arr[keep_valid]
    labels_f   = x_labels[keep_valid]
    chi2_f     = chi2_ok[keep_valid]

    def _one(vec, name, ylabel):
        fig, axes = plt.subplots(1, 2 if include_trend else 1, figsize=(10 if include_trend else 5, 4))
        if include_trend: 
            axh, axt = axes
        else:             
            axh = axes
        axh.hist(vec[np.isfinite(vec)], bins=bins)
        axh.set_title(f"{name} histogram") 
        axh.set_xlabel(ylabel) 
        axh.set_ylabel("count")
        if include_trend:
            axt.plot(labels_f, vec, ".", ms=4)
            axt.set_title(f"{name} vs NV label") 
            axt.set_xlabel("NV label") 
            axt.set_ylabel(ylabel)
        if save_prefix: 
            file_path = dm.get_file_path(__file__, timestamp, f"{save_prefix}_{name}.png")
            dm.save_figure(fig, file_path)
        return fig

    figs = []
    units = ["arb.","arb.","µs","µs","ms","–",
             "–","– per revival","fraction","arb.","1/µs","1/µs","rad","rad"]
    for col, (name, unit) in enumerate(zip(_UNIFIED_KEYS, units)):
        figs.append((name, _one(arr_f[:, col], name, unit)))

    fig_chi, axes = plt.subplots(1, 2 if include_trend else 1, figsize=(10 if include_trend else 5, 4))
    if include_trend: 
        axh, axt = axes
    else:             
        axh = axes
    axh.hist(chi2_f[np.isfinite(chi2_f)], bins=bins)
    axh.set_title("reduced χ² histogram") 
    axh.set_xlabel("χ²_red")
    axh.set_ylabel("count")
    if include_trend:
        axt.plot(labels_f, chi2_f, ".", ms=4)
        axt.set_title("reduced χ² vs NV label") 
        axt.set_xlabel("NV label") 
        axt.set_ylabel("χ²_red")
    fig_chi.tight_layout()
    if save_prefix: 
        fig_chi.savefig(f"{save_prefix}-chi2_red.png", dpi=220)
        file_path = dm.get_file_path(__file__, timestamp, f"{save_prefix}-chi2_red.png")
        dm.save_figure(fig_chi, file_path)
        
    figs.append(("chi2_red", fig_chi))

    kept_labels = labels_f.astype(int)
    return figs, full_mask, kept_labels


# ==========================================
# 3) Individual NV fit plots
#    - keep passed nv_inds
#    - safely handle shorter popts (fixed-revival)
# ==========================================



# --- keep your existing extract_T2_freqs_and_errors(...) ---

def _mask_huge_errors(values, sigmas, *, rel_cap=None, pct_cap=None):
    """
    Returns a copy of 'sigmas' with too-large bars set to NaN (so matplotlib won't draw them).
      rel_cap:   max allowed sigma/value (e.g., 0.75). If value<=0 or NaN, rel test is skipped.
      pct_cap:   clip absolute sigma above this percentile to NaN (e.g., 95).
    """
    if sigmas is None: return None
    v = np.asarray(values, float)
    s = np.asarray(sigmas, float).copy()

    # relative cap
    if rel_cap is not None:
        with np.errstate(divide='ignore', invalid='ignore'):
            rel = s / v
        bad_rel = ~np.isfinite(rel) | (rel > float(rel_cap))
        s[bad_rel] = np.nan

    # percentile cap (absolute)
    if pct_cap is not None:
        finite = np.isfinite(s) & (s > 0)
        if np.any(finite):
            thresh = np.percentile(s[finite], float(pct_cap))
            s[(s > thresh)] = np.nan

    return s

def plot_sorted_panels_with_err(
    nv_labels, T2_us, sT2_us, A_pick_kHz, sA_pick_kHz, *,
    mask_no_decay=None, mask_fit_fail=None,
    title_prefix="Spin-Echo", t2_units="µs",
    # error-bar pruning controls (tune as you like)
    t2_rel_cap=1.0,      # hide T2 bars with σ > 100% of value
    t2_pct_cap=99,       # and hide top 5% largest absolute T2 sigmas
    A_rel_cap=0.75,      # hide A bars with σ > 75% of value
    A_pct_cap=99         # and hide top 5% largest absolute A sigmas
):
    N = len(nv_labels)
    mask_no_decay = np.zeros(N, bool) if mask_no_decay is None else mask_no_decay
    mask_fit_fail = np.zeros(N, bool) if mask_fit_fail is None else mask_fit_fail

    # ---- (a) T2 sorted with pruned error bars ----
    valid_t2 = np.isfinite(T2_us) & (~mask_no_decay) & (~mask_fit_fail)
    if np.any(valid_t2):
        idx = np.where(valid_t2)[0]
        order = idx[np.argsort(T2_us[idx])]
        x = np.arange(1, order.size+1)

        if t2_units.lower().startswith("ms"):
            y    = T2_us[order] / 1000.0
            yerr_raw = (sT2_us[order] / 1000.0) if sT2_us is not None else None
        else:
            y    = T2_us[order]
            yerr_raw = sT2_us[order] if sT2_us is not None else None

        yerr = _mask_huge_errors(y, yerr_raw, rel_cap=t2_rel_cap, pct_cap=t2_pct_cap)

        plt.figure(figsize=(10,5))
        plt.errorbar(x, y, yerr=yerr, fmt="o", ms=3, lw=0.8, capsize=2, elinewidth=0.8, alpha=0.95)
        plt.grid(alpha=0.3)
        plt.xlabel("NV index (sorted)")
        plt.ylabel(r"$T_2$ (" + ("ms" if t2_units.lower().startswith("ms") else "µs") + ")")
        plt.yscale("log")
        plt.title(f"{title_prefix}: $T_2$ (sorted)")
        note = f"Excluded: no-decay={mask_no_decay.sum()}, fit-fail={mask_fit_fail.sum()}; Used={order.size}/{N}"
        plt.text(0.01, 0.98, note, transform=plt.gca().transAxes, ha="left", va="top", fontsize=8)
    else:
        print("[plot] No valid T2 to plot.")

    # ---- (b) Ahfs sorted with pruned error bars ----
    valid_A = np.isfinite(A_pick_kHz) & (A_pick_kHz > 0)
    if np.any(valid_A):
        idx = np.where(valid_A)[0]
        order = idx[np.argsort(A_pick_kHz[idx])]
        x = np.arange(1, order.size+1)
        y = A_pick_kHz[order]
        yerr_raw = sA_pick_kHz[order] if sA_pick_kHz is not None else None

        yerr = _mask_huge_errors(y, yerr_raw, rel_cap=A_rel_cap, pct_cap=A_pct_cap)

        fig, ax = plt.subplots(figsize=(8,6))
        ax.errorbar(
            x, y, yerr=yerr,
            fmt="o", ms=3, lw=0.8,
            capsize=2, elinewidth=0.8,
            alpha=0.95,
            label="Spin-echo derived (picked)",
        )
        # ax.grid(alpha=0.3)

        # Axis labels / title
        ax.set_xlabel("NV index (sorted)")
        ax.set_ylabel(r"$A_{\mathrm{hfs}}$ (kHz)")
        ax.set_title(f"{title_prefix}: $A_{{\\rm hfs}}$ (sorted)")

        # Log y-scale
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(mticker.LogLocator(base=10))
        ax.yaxis.set_minor_locator(mticker.LogLocator(base=10, subs=np.arange(2, 10)))
        ax.yaxis.set_minor_formatter(mticker.NullFormatter())

        ax.grid(True, which="both", alpha=0.3)
        # Note about excluded points
        note = f"Excluded here (no valid freq): {(~valid_A).sum()}"

        # Put note in axes coordinates (top-left of plot)
        ax.text(
            0.01, 0.98, note,
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=8,
        )

        fig.tight_layout()
    else:
        print("[plot] No hyperfine points to plot.")




# ---------- small helper used by your panel plot ----------
def _mask_huge_errors(y, yerr, *, rel_cap=1.0, pct_cap=99):
    if yerr is None: return None
    yerr = np.asarray(yerr, float).copy()
    bad = ~np.isfinite(yerr)
    if np.any(np.isfinite(y) & np.isfinite(yerr)):
        # cap relative blow-ups
        bad |= (yerr > rel_cap * np.maximum(1e-12, np.abs(y)))
        # cap extreme tail
        thr = np.nanpercentile(yerr[~bad], pct_cap) if np.any(~bad) else np.inf
        bad |= (yerr > thr)
    yerr[bad] = 0.0
    return yerr

# ================= Experimental spectrum builders =============================

def build_exp_lines(
    f0_kHz, f1_kHz,
    *,
    fmin_kHz=1.0, fmax_kHz=20000.0,
    weight_mode="unit",          # {"unit","invvar","inv_chi2","custom"}
    sA_pick_kHz=None,            # needed for invvar
    chis=None,                   # needed for inv_chi2
    custom_weights=None,         # array same length as combined freqs if weight_mode="custom"
    per_line_scale=1.0,
):
    """
    Turn the experimental fitted frequencies into a list of sticks (freq, weight).
    Includes both columns (f0,f1), filters to [fmin,fmax], and applies weights.

    weight_mode:
      - "unit"      → each line weight = 1
      - "invvar"    → weight = 1 / σ_A^2   (uses sA_pick_kHz; broadcasts to both f0,f1)
      - "inv_chi2"  → weight = 1 / χ²      (uses chis; broadcasts to both f0,f1)
      - "custom"    → use `custom_weights` (must match length of resulting lines)
    """
    f0 = np.asarray(f0_kHz, float)
    f1 = np.asarray(f1_kHz, float)

    # collect valid frequencies
    F = []
    idx_src = []  # (row_index, which) for building weights later
    for which, arr in enumerate((f0, f1)):   # 0→f0, 1→f1
        m = np.isfinite(arr) & (arr >= fmin_kHz) & (arr <= fmax_kHz)
        if np.any(m):
            F.append(arr[m])
            # remember where these came from
            idxs = np.nonzero(m)[0]
            idx_src.extend([(int(i), which) for i in idxs])

    if not F:
        return np.array([]), np.array([])

    freqs = np.concatenate(F)
    order = np.argsort(freqs)
    freqs = freqs[order]
    idx_src = [idx_src[i] for i in order]

    # weights
    if weight_mode == "unit":
        w = np.ones_like(freqs, float)

    elif weight_mode == "invvar":
        if sA_pick_kHz is None:
            raise ValueError("invvar weighting needs sA_pick_kHz.")
        sA = np.asarray(sA_pick_kHz, float)
        w = np.zeros_like(freqs, float)
        for k, (i, _which) in enumerate(idx_src):
            si = sA[i]
            w[k] = 0.0 if (not np.isfinite(si) or si <= 0) else 1.0 / (si * si)

    elif weight_mode == "inv_chi2":
        if chis is None:
            raise ValueError("inv_chi2 weighting needs chis.")
        c = np.asarray(chis, float)
        w = np.zeros_like(freqs, float)
        for k, (i, _which) in enumerate(idx_src):
            ci = c[i]
            w[k] = 0.0 if (not np.isfinite(ci) or ci <= 0) else 1.0 / ci

    elif weight_mode == "custom":
        if custom_weights is None:
            raise ValueError("custom weighting needs custom_weights.")
        cw = np.asarray(custom_weights, float)
        if cw.shape != freqs.shape:
            raise ValueError("custom_weights must match combined frequency array length.")
        w = cw.copy()

    else:
        raise ValueError(f"Unknown weight_mode='{weight_mode}'")

    return freqs, per_line_scale * w


def plot_exp_sticks(
    freqs_kHz, weights=None, *,
    title="Experimental ESEEM sticks (sorted)",
    weight_caption="Weight (arb.)",
    min_weight=0.0
):
    """Vertical-stick plot for experimental lines."""
    if freqs_kHz.size == 0:
        print("[plot_exp_sticks] No lines to plot.")
        return
    w = np.ones_like(freqs_kHz, float) if (weights is None) else np.asarray(weights, float)
    m = (w >= float(min_weight)) & np.isfinite(freqs_kHz) & np.isfinite(w)
    f = freqs_kHz[m]; w = w[m]
    order = np.argsort(f); f = f[order]; w = w[order]

    plt.figure(figsize=(10, 5))
    for fk, wk in zip(f, w):
        plt.vlines(fk, 0.0, wk, linewidth=1.5)
    plt.xlabel("Frequency (kHz)")
    plt.ylabel(weight_caption)
    plt.xscale("log")
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.show()



def convolved_exp_spectrum(
    freqs_kHz,
    weights=None,
    *,
    f_range_kHz=(1.0, 20000.0),
    npts=4000,
    shape="gauss",          # {"gauss","lorentz"}
    width_kHz=8.0,          # σ for gauss, HWHM γ for lorentz
    width_is_fwhm=False,    # if True, converts FWHM -> (σ or γ)
    log_x=True,             # plotting scale only
    merge_tol_Hz=0.0,       # >0 to merge near-duplicate sticks before convolving
    clean_neg_weights=True, # clip negatives to 0
    normalize="max",        # {"max","area",None}
    title_prefix="Experimental ESEEM spectrum",
    weight_caption="unit weight",
    ax=None
):
    """
    Convolve discrete experimental lines with an analytic kernel (area-normalized).
    Returns (f_kHz, spec), where spec may be normalized.

    Notes:
      - If you care about low-f detail, set log_x=True for plotting but keep evaluation
        dense (increase npts) or use log_x=False with a fine linear grid.
      - width_kHz is σ (gauss) or γ (lorentz). Set width_is_fwhm=True to pass FWHM.
    """
    f_in = np.asarray(freqs_kHz, float)
    if f_in.size == 0:
        raise ValueError("No experimental lines provided.")
    w_in = np.ones_like(f_in) if weights is None else np.asarray(weights, float)

    # Clean/clip weights
    w_in = np.where(np.isfinite(w_in), w_in, 0.0)
    if clean_neg_weights:
        w_in = np.clip(w_in, 0.0, None)

    # Band-limit lines
    lo, hi = map(float, f_range_kHz)
    if lo <= 0 and log_x:
        raise ValueError("f_range_kHz[0] must be > 0 when log_x=True.")
    m = np.isfinite(f_in) & (f_in >= lo) & (f_in <= hi)
    f0 = f_in[m]; a0 = w_in[m]
    if f0.size == 0:
        raise ValueError("No experimental lines in requested range.")

    # Optional: merge near-duplicates (before convolution)
    if merge_tol_Hz and merge_tol_Hz > 0:
        order = np.argsort(f0)
        f0 = f0[order]; a0 = a0[order]
        f_merged = [f0[0]]; a_merged = [a0[0]]
        tol = float(merge_tol_Hz) * 1e-3  # convert Hz -> kHz for our arrays
        for fk, ak in zip(f0[1:], a0[1:]):
            if abs(fk - f_merged[-1]) <= tol:
                # accumulate weight into previous line
                a_merged[-1] += ak
            else:
                f_merged.append(fk); a_merged.append(ak)
        f0 = np.asarray(f_merged, float)
        a0 = np.asarray(a_merged, float)

    # Evaluation grid (dense linear; we can still plot in log-x)
    f = (np.logspace(np.log10(lo), np.log10(hi), int(npts))
         if log_x else np.linspace(lo, hi, int(npts)))
    f = f.astype(float)

    # Kernel width handling
    if shape.lower().startswith("gauss"):
        # Use σ; if FWHM given, convert: FWHM = 2√(2ln2) σ
        sigma = float(width_kHz)
        if width_is_fwhm:
            sigma = sigma / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        # Vectorized Gaussian: sum_k a_k * N(f; f_k, σ)
        # Area-normalized kernel: 1/(σ√(2π)) * exp(-0.5 ((f-fk)/σ)^2)
        df = f[:, None] - f0[None, :]
        spec = np.exp(-0.5 * (df / sigma) ** 2) / (sigma * np.sqrt(2.0 * np.pi))
        spec = (spec * a0[None, :]).sum(axis=1)
        kern_label = f"Gaussian ({'FWHM' if width_is_fwhm else 'σ'}={width_kHz:.2f} kHz)"
    else:
        # Lorentzian uses HWHM γ; convert FWHM -> γ if requested (FWHM = 2γ)
        gamma = float(width_kHz)
        if width_is_fwhm:
            gamma = gamma / 2.0
        df2 = (f[:, None] - f0[None, :]) ** 2
        spec = (gamma / np.pi) / (df2 + gamma ** 2)
        spec = (spec * a0[None, :]).sum(axis=1)
        kern_label = f"Lorentzian ({'FWHM' if width_is_fwhm else 'γ'}={width_kHz:.2f} kHz)"

    # Optional normalization
    if normalize == "max":
        mval = np.max(spec) if spec.size else 1.0
        if mval > 0:
            spec = spec / mval
    elif normalize == "area":
        # trapezoid area in kHz
        area = np.trapz(spec, f)
        if area > 0:
            spec = spec / area

    # Plot
    # ax = ax or plt.gca()
    # ax, fig = plt.figure()
    fig, ax = plt.subplots()
    if log_x:
        ax.set_xscale("log")
    ax.plot(f, spec, lw=1.6)
    ax.set_xlim(lo, hi)
    ax.set_xlabel("Frequency (kHz)")
    ylab = "Intensity (arb.)"
    if normalize == "max":
        ylab += " (unit max)"
    elif normalize == "area":
        ylab += " (unit area)"
    ax.set_ylabel(ylab + f"\nweights = {weight_caption}")
    ax.set_title(f"{title_prefix} • {kern_label}")
    ax.grid(True, which="both", alpha=0.25)
    plt.tight_layout()

    return f, spec


def plot_sorted_exp_branches(
    f0_kHz,
    f1_kHz,
    sf0_kHz=None,
    sf1_kHz=None,
    title_prefix="Spin-echo",
    f_range_kHz=(15, 15000),
    A_rel_cap=0.7,
    A_pct_cap=200.0,
):
    """
    Plot experimental spin-echo-derived frequencies for f0 and f1 separately,
    sorted by magnitude, on a log-y axis.

    Parameters
    ----------
    f0_kHz, f1_kHz : array-like
        Arrays of picked frequencies (kHz) for each NV (one per NV per branch).
    sf0_kHz, sf1_kHz : array-like or None
        1σ errors on f0 and f1 (kHz). If None, no error bars for that branch.
    title_prefix : str
        Prefix for the plot title.
    f_range_kHz : (float, float)
        Y-axis range in kHz.
    A_rel_cap, A_pct_cap : float
        Parameters passed to _mask_huge_errors for clipping crazy error bars.
    """
    f0_kHz = np.asarray(f0_kHz, float)
    f1_kHz = np.asarray(f1_kHz, float)
    sf0_kHz = None if sf0_kHz is None else np.asarray(sf0_kHz, float)
    sf1_kHz = None if sf1_kHz is None else np.asarray(sf1_kHz, float)

    # --- Branch f0 ---
    valid0 = np.isfinite(f0_kHz) & (f0_kHz > 0)
    if np.any(valid0):
        idx0 = np.where(valid0)[0]
        order0 = idx0[np.argsort(f0_kHz[idx0])]
        x0 = np.arange(1, order0.size + 1)
        y0 = f0_kHz[order0]
        y0err_raw = sf0_kHz[order0] if sf0_kHz is not None else None
        y0err = _mask_huge_errors(y0, y0err_raw,
                                  rel_cap=A_rel_cap, pct_cap=A_pct_cap)
    else:
        order0 = np.array([], int)
        x0 = y0 = y0err = None

    # --- Branch f1 ---
    valid1 = np.isfinite(f1_kHz) & (f1_kHz > 0)
    if np.any(valid1):
        idx1 = np.where(valid1)[0]
        order1 = idx1[np.argsort(f1_kHz[idx1])]
        x1 = np.arange(1, order1.size + 1)
        y1 = f1_kHz[order1]
        y1err_raw = sf1_kHz[order1] if sf1_kHz is not None else None
        y1err = _mask_huge_errors(y1, y1err_raw,
                                  rel_cap=A_rel_cap, pct_cap=A_pct_cap)
    else:
        order1 = np.array([], int)
        x1 = y1 = y1err = None

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(10, 6))

    if x0 is not None:
        ax.errorbar(
            x0, y0, yerr=y0err,
            fmt="o", ms=3, lw=0.8,
            capsize=2, elinewidth=0.8,
            alpha=0.95,
            label=r"Exp $f_0$ (lower branch)",
        )

    if x1 is not None:
        ax.errorbar(
            x1, y1, yerr=y1err,
            fmt="s", ms=3, lw=0.8,
            capsize=2, elinewidth=0.8,
            alpha=0.95,
            label=r"Exp $f_1$ (upper branch)",
        )

    ax.set_yscale("log")
    ax.set_ylim(*f_range_kHz)
    ax.set_xlabel("NV index (sorted within each branch)")
    ax.set_ylabel(r"Frequency (kHz)")
    ax.set_title(f"{title_prefix}: $f_0$ and $f_1$ (sorted)")
    ax.set_ylim(ymin=10, ymax=10000)
    ax.yaxis.set_major_locator(mticker.LogLocator(base=10))
    ax.yaxis.set_minor_locator(mticker.LogLocator(base=10, subs=np.arange(2, 10)))
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax.grid(True, which="both", alpha=0.3)

    # Notes about excluded NVs
    note0 = f"Excluded f0: {(~valid0).sum()}"
    note1 = f"Excluded f1: {(~valid1).sum()}"
    ax.text(
        0.01, 0.97, note0,
        transform=ax.transAxes,
        ha="left", va="top", fontsize=8,
    )
    ax.text(
        0.01, 0.92, note1,
        transform=ax.transAxes,
        ha="left", va="top", fontsize=8,
    )

    ax.legend(framealpha=0.85)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    kpl.init_kplotlib()
    # --- Load your data------------------------------------
    file_stems = ["2025_10_10-11_29_40-rubin-nv0_2025_09_08",
                  "2025_10_10-08_55_59-rubin-nv0_2025_09_08",
                  "2025_10_10-06_28_12-rubin-nv0_2025_09_08",
                  "2025_10_10-03_59_48-rubin-nv0_2025_09_08",
                  "2025_10_10-01_31_59-rubin-nv0_2025_09_08",
                  "2025_10_09-23_03_41-rubin-nv0_2025_09_08",
                  "2025_10_10-14_23_58-rubin-nv0_2025_09_08",
                  "2025_10_10-17_04_27-rubin-nv0_2025_09_08"]
    
    # file_stems = ["2025_10_29-10_33_01-johnson-nv0_2025_10_21",
    #             "2025_10_29-02_21_07-johnson-nv0_2025_10_21",
    #             ]
    
    ###204NVs
    file_stems = ["2025_10_31-23_53_21-johnson-nv0_2025_10_21",
                  "2025_10_31-15_40_56-johnson-nv0_2025_10_21",
                  "2025_10_31-07_42_45-johnson-nv0_2025_10_21",
                ]
    
    ###204NVs dataset 2
    file_stems_1 = ["2025_11_03-01_47_09-johnson-nv0_2025_10_21",
                  "2025_11_02-14_49_57-johnson-nv0_2025_10_21",
                  "2025_11_02-04_46_56-johnson-nv0_2025_10_21",
                ]
    ###204NVs
    file_stems_2 = ["2025_11_11-06_02_04-johnson-nv0_2025_10_21",
                    "2025_11_10-20_58_00-johnson-nv0_2025_10_21",
                  "2025_11_10-11_36_39-johnson-nv0_2025_10_21",
                  "2025_11_10-03_06_14-johnson-nv0_2025_10_21",
                ]
    
    # file_stems = file_stems_1 + file_stems_2
    
    ### get proceeded data data
    # file_stem = "2025_11_10-16_17_03-johnson_204nv_s3-003c56" #dataset 1 
    # file_stem = "2025_11_11-01_05_17-johnson_204nv_s3-0e14ae" #dataset 3
    file_stem = "2025_11_11-01_15_45-johnson_204nv_s6-6d8f5c" #dataset2 + dataset3
    data = dm.get_raw_data(file_stem=file_stem)
    nv_list = data["nv_list"]
    norm_counts = np.array(data["norm_counts"])
    norm_counts_ste = np.array(data["norm_counts_ste"])
    total_evolution_times = np.array(data["total_evolution_times"])

    # ## laod analysed data
    timestamp = dm.get_time_stamp()
    # # file_stem= "2025_11_01-16_57_48-rubin-nv0_2025_09_08"
    # file_stem= "2025_11_10-19_33_17-johnson_204nv_s3-003c56" 
    # file_stem= "2025_11_10-21_38_55-johnson_204nv_s3-003c56" 
    # file_stem= "2025_11_11-01_46_41-johnson_204nv_s3-003c56" 
    # file_stem= "2025_11_11-06_23_14-johnson_204nv_s6-6d8f5c" 
    # file_stem= "2025_11_12-08_51_17-johnson_204nv_s7-aab2d0"
    file_stem= "2025_11_13-06_28_22-sample_204nv_s1-e85aa7"
    
    
    data = dm.get_raw_data(file_stem=file_stem)
    popts = data["popts"]
    chis = data["red_chi2"]
    fit_nv_labels = data ["nv_labels"]
    fit_fn_names = data["fit_fn_names"]
    repr_nv_sig = widefield.get_repr_nv_sig(nv_list)
    repr_nv_name = repr_nv_sig.name
    
    # 2) PARAM PANELS (T2 outlier filter)
    # figs, keep_mask, kept_labels = plot_each_param_separately(
    #     popts, chis, fit_nv_labels, 
    #     save_prefix= "rubin-spin_echo-2025_09_08",
    #     t2_policy=dict(method="iqr", iqr_k=5, abs_range=(0.00, 1.0))
    # )

    
    fit_nv_labels  = list(map(int, data["nv_labels"]))
    fit_fn_names   = data["fit_fn_names"]

    # 1) Map stored names -> real callables
    _fn_map = {
        "fine_decay": fine_decay,
        "fine_decay_fixed_revival": fine_decay_fixed_revival,
    }
    fit_fns = []
    for name in fit_fn_names:
        if name is None:
            fit_fns.append(None)
        else:
            fn = _fn_map.get(name)
            if fn is None:
                fn = fine_decay
            fit_fns.append(fn)
        
    # 3) INDIVIDUAL FITS — PASS THE SAME LABELS + PER-NV FIT FUNCTIONS
    _ = plot_individual_fits(
        norm_counts, norm_counts_ste, total_evolution_times,
        popts,
        nv_inds=fit_nv_labels,
        fit_fn_per_nv=fit_fns,
        # keep_mask=keep_mask,
        show_residuals=True,
        block=False
    )
     
    # # --------------------------
    # # Example usage
    # # --------------------------
    # ---- Extract once ----
    (nv, T2_us, f0_kHz, f1_kHz, A_pick_kHz, chis, fit_fail,
    sT2_us, sf0_kHz, sf1_kHz, sA_pick_kHz) = extract_T2_freqs_and_errors(
        data, pick_freq="max", chi2_fail_thresh=3.0
    )

    THRESH_US = 600.0
    valid = np.isfinite(T2_us) & (~fit_fail)
    mask  = valid & (T2_us <= THRESH_US)

    df = pd.DataFrame({
        "nv": nv,
        "T2_us": T2_us,
        "T2_ms": T2_us/1000.0,
        "sT2_us": sT2_us,
        "f0_kHz": f0_kHz,   "sf0_kHz": sf0_kHz,
        "f1_kHz": f1_kHz,   "sf1_kHz": sf1_kHz,
        "A_pick_kHz": A_pick_kHz, "sA_pick_kHz": sA_pick_kHz,
        "red_chi2": chis,
        "fit_fail": fit_fail,
    })
    sel = df.loc[mask].sort_values("T2_us", ascending=False).reset_index(drop=True)
    print(f"NVs with T2 ≥ {THRESH_US:.0f} µs: {len(sel)}")
    print(sel[["nv","T2_us","sT2_us","A_pick_kHz","sA_pick_kHz",
            "f0_kHz","f1_kHz","red_chi2"]].to_string(index=False))

    CAP_US = 600.0
    mask_cap_exact = np.isfinite(T2_us) & np.isclose(T2_us, CAP_US, atol=1e-6)
    cap_indices = np.where(mask_cap_exact)[0]
    cap_labels  = np.asarray(nv)[mask_cap_exact]

    # ---- Apply mask to ALL arrays before plotting ----
    nv_m          = np.asarray(nv)[mask]
    T2_us_m       = np.asarray(T2_us)[mask]
    sT2_us_m      = np.asarray(sT2_us)[mask]
    A_pick_kHz_m  = np.asarray(A_pick_kHz)[mask]
    sA_pick_kHz_m = np.asarray(sA_pick_kHz)[mask]
    f0_kHz_m      = np.asarray(f0_kHz)[mask]
    f1_kHz_m      = np.asarray(f1_kHz)[mask]
    sf0_kHz_m     = np.asarray(sf0_kHz)[mask]
    sf1_kHz_m     = np.asarray(sf1_kHz)[mask]
    chis_m        = np.asarray(chis)[mask]
    fit_fail_m    = np.asarray(fit_fail)[mask]   # should be all False


    # ---- Plot with masked arrays only ----
    # (1) Your existing panels (unchanged)
    # plot_sorted_panels_with_err(
    #     nv_m, T2_us_m, sT2_us_m, A_pick_kHz_m, sA_pick_kHz_m,
    #     mask_fit_fail=np.zeros_like(fit_fail_m, dtype=bool),
    #     t2_rel_cap=1.0, t2_pct_cap=95, A_rel_cap=0.75, A_pct_cap=95
    # )

    # (2) Experimental sticks: choose a weighting
    #   a) unit weights
    F_kHz, W = build_exp_lines(f0_kHz_m, f1_kHz_m, fmin_kHz=1, fmax_kHz=20000, weight_mode="unit")

    #   b) inverse-variance (needs sA_pick_kHz_m)
    # F_kHz, W = build_exp_lines(f0_kHz_m, f1_kHz_m, fmin_kHz=1, fmax_kHz=20000,
    #                            weight_mode="invvar", sA_pick_kHz=sA_pick_kHz_m)

    #   c) inverse chi^2 (needs chis_m)
    F_kHz, W = build_exp_lines(f0_kHz_m, f1_kHz_m, fmin_kHz=1, fmax_kHz=20000,
                               weight_mode="inv_chi2", chis=chis_m)

    # plot_exp_sticks(
    #     F_kHz, W,
    #     title="Experimental ESEEM sticks (sorted)",
    #     weight_caption="unit weight"  # or "1/σ_A^2", or "1/χ²"
    # )

    # # (3) Convolved spectrum (Gaussian)
    # _fff, _SSS = convolved_exp_spectrum(
    #     F_kHz, W,
    #     f_range_kHz=(1, 20000),
    #     npts=2400,
    #     shape="gauss",          # "lorentz" also available
    #     width_kHz=8.0,
    #     title_prefix="Experimental ESEEM spectrum",
    #     weight_caption="unit weight"  # match what you chose above
    # )
    
    # plot_sorted_exp_branches(
    # f0_kHz=f0_kHz_m,
    # f1_kHz=f1_kHz_m,
    # sf0_kHz=sf0_kHz_m,
    # sf1_kHz=sf1_kHz_m,
    # title_prefix="Spin-echo fit with free oscillation frequencies",
    # f_range_kHz=(10, 6000),)


    kpl.show(block=True)
