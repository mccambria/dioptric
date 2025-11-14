# -*- coding: utf-8 -*-
"""
Spin-echo: finer fit + fitted-figure + parameter panels

- Physics-y comb with quartic lobes, amplitude taper, width growth, chirp
- Optional two-frequency sin^2 beating with phases
- Smoothly plugs into your plotting + data pipeline

Author: you + chatgpt teammate
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
# =============================================================================
# Finer model
#    - COMB has NO overall comb_contrast
#    - MOD carries the overall amplitude (and optional beating)
# =============================================================================

def fine_decay(
    tau_us,
    baseline=1.0,
    comb_contrast=0.6,
    revival_time=37.0,
    width0_us=6.0,
    T2_ms=0.08,
    T2_exp=1.0,
    amp_taper_alpha=0.0,
    width_slope=0.0,
    revival_chirp=0.0,
    # NEW (additive, signed; zero-mean cos carrier(s))
    osc_amp=0.0,
    osc_f0=0.0,
    osc_phi0=0.0,
    osc_f1=0.0,
    osc_phi1=0.0,
):
    """
    signal(τ) = baseline
                - comb_contrast * envelope(τ) * COMB(τ)               [dip]
                +                 envelope(τ) * COMB(τ) * OSC(τ)      [signed oscillation]

    envelope(τ) = exp[-(τ / (1000*T2_ms))^T2_exp]
    COMB(τ)     = Σ_k [ 1/(1+k)^amp_taper_alpha ] * exp(-((τ-μ_k)/w_k)^4)
                    μ_k = k * revival_time * (1 + k*revival_chirp)
                    w_k = width0_us * (1 + k*width_slope)

    OSC(τ)      = osc_amp * [ cos(2π f0 τ + φ0) + cos(2π f1 τ + φ1) ]   # zero mean
    τ in microseconds, f in cycles/μs, phases in rad.
    """
    amp_taper_alpha = 0.0 if amp_taper_alpha is None else float(amp_taper_alpha)
    width_slope     = 0.0 if width_slope     is None else float(width_slope)
    revival_chirp   = 0.0 if revival_chirp   is None else float(revival_chirp)
    osc_amp         = 0.0 if osc_amp         is None else float(osc_amp)
    osc_f0          = 0.0 if osc_f0          is None else float(osc_f0)
    osc_f1          = 0.0 if osc_f1          is None else float(osc_f1)
    osc_phi0        = 0.0 if osc_phi0        is None else float(osc_phi0)
    osc_phi1        = 0.0 if osc_phi1        is None else float(osc_phi1)

    tau = np.asarray(tau_us, dtype=float).ravel()
    width0_us    = max(1e-9, float(width0_us))
    revival_time = max(1e-9, float(revival_time))
    T2_us        = max(1e-9, 1000.0 * float(T2_ms))
    T2_exp       = float(T2_exp)

    # envelope
    envelope = np.exp(-((tau / T2_us) ** T2_exp))

    # number of revivals to include
    tau_max = float(np.nanmax(tau)) if tau.size else 0.0
    n_guess = max(1, min(64, int(np.ceil(1.2 * tau_max / revival_time)) + 1))

    comb = _comb_quartic_powerlaw(
        tau,
        revival_time,
        width0_us,
        amp_taper_alpha,
        width_slope,
        revival_chirp,
        n_guess
    )

    carrier = envelope * comb
    # baseline minus revival dip
    dip = comb_contrast * carrier
    # additive, zero-mean oscillation (can push above baseline)
    osc = 0.0
    if osc_amp != 0.0:
        if osc_f0 != 0.0:
            osc += np.cos(2*np.pi*osc_f0 * tau + osc_phi0)
        if osc_f1 != 0.0:
            osc += np.cos(2*np.pi*osc_f1 * tau + osc_phi1)

    return baseline - dip + carrier * (osc_amp * osc)

def fine_decay_fixed_revival(
    tau,
    baseline,
    comb_contrast,
    width0_us,
    T2_ms,
    T2_exp,
    amp_taper_alpha=None,
    width_slope=None,
    revival_chirp=None,
    osc_amp=None,
    osc_f0=None,
    osc_f1=None,
    osc_phi0=None,
    osc_phi1=None,
    _fixed_rev_time_us=37.0
):
    return fine_decay(
        tau,
        baseline,
        comb_contrast,
        _fixed_rev_time_us,
        width0_us,
        T2_ms,
        T2_exp,
        amp_taper_alpha,
        width_slope,
        revival_chirp,
        osc_amp,   # was osc_contrast
        osc_f0,
        osc_phi0,  # <-- swap order
        osc_f1,    # <-- swap order
        osc_phi1,
    )


@njit
def _comb_quartic_powerlaw(
    tau,
    revival_time,
    width0_us,
    amp_taper_alpha,
    width_slope,
    revival_chirp,
    n_guess
):
    """
    NOTE: no overall amplitude factor here (no comb_contrast).
    """
    n = tau.shape[0]
    out = np.zeros(n, dtype=np.float64)
    tmax = 0.0
    for i in range(n):
        if tau[i] > tmax:
            tmax = tau[i]

    for k in range(n_guess):
        mu_k = k * revival_time * (1.0 + k * revival_chirp)
        w_k  = width0_us * (1.0 + k * width_slope)
        if w_k <= 0.0:
            continue
        if mu_k > tmax + 5.0 * w_k:
            break

        amp_k = 1.0 / ((1.0 + k) ** amp_taper_alpha)  # <- amplitude taper only
        inv_w4 = 1.0 / (w_k ** 4)

        for i in range(n):
            x = tau[i] - mu_k
            out[i] += amp_k * np.exp(- (x * x) * (x * x) * inv_w4)

    return out


# ==========================================
#  helper: decide which NVs to keep based on T2_ms
# ==========================================

# --- helper: decide which NVs to keep based on T2_ms ---
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

def _coerce_to_core6(p, default_rev=39.2):
    p = np.asarray(p, float)
    if len(p) == 5:  # fixed-revival core -> inject revival_time for plotting with fine_decay
        b, cc, w0, t2, exp = p
        return np.array([b, cc, default_rev, w0, t2, exp], float)
    return p

def _safe_call_fit_fn(fit_fn, t, p, default_rev=39.2):
    try:
        return fit_fn(t, *p)
    except TypeError:
        return fit_fn(t, *_coerce_to_core6(p, default_rev=default_rev))

# --- helpers you already have ---
def _coerce_to_core6(p, default_rev=39.2):
    p = np.asarray(p, float)
    if len(p) == 5:  # fixed-revival core -> inject revival_time for plotting with fine_decay
        b, cc, w0, t2, exp = p
        return np.array([b, cc, default_rev, w0, t2, exp], float)
    return p

def _safe_call_fit_fn(fit_fn, t, p, default_rev=39.2):
    try:
        return fit_fn(t, *p)
    except TypeError:
        return fit_fn(t, *_coerce_to_core6(p, default_rev=default_rev))

# --- NEW: map p -> dict using the fit function signature (best effort) ---
def _params_to_dict(fit_fn, p, default_rev=39.2):
    """
    Turn a parameter vector 'p' into a name->value dict using the fit_fn signature.
    If lengths don't match (e.g., coercion paths), fall back to a sensible mapping.
    """
    p = np.asarray(p, float).tolist()
    try:
        sig = inspect.signature(fit_fn)
        names = [k for k in sig.parameters.keys()][1:]  # drop 'tau'
    except Exception:
        names = []

    d = {}
    if names and len(p) <= len(names):
        for name, val in zip(names, p):
            d[name] = float(val)
        # In case of fixed-revival core (no 'revival_time'):
        if ('revival_time' not in d) and ('width0_us' in d) and ('T2_ms' in d):
            d['revival_time'] = float(default_rev)
        # Back-compat: normalize osc names
        if 'osc_contrast' in d and 'osc_amp' not in d:
            d['osc_amp'] = d['osc_contrast']
    else:
        # Fallback by length heuristics
        # Core-6: [baseline, comb_contrast, revival_time, width0_us, T2_ms, T2_exp]
        if len(p) >= 6:
            d.update(dict(
                baseline=p[0], comb_contrast=p[1], revival_time=p[2],
                width0_us=p[3], T2_ms=p[4], T2_exp=p[5]
            ))
        elif len(p) == 5:
            d.update(dict(
                baseline=p[0], comb_contrast=p[1], revival_time=default_rev,
                width0_us=p[2], T2_ms=p[3], T2_exp=p[4]
            ))
        # Try to place extras in a common order if present beyond core-6
        # [amp_taper_alpha, width_slope, revival_chirp, osc_amp (or contrast),
        #  osc_f0, osc_f1, osc_phi0, osc_phi1, mu0_us]
        extras = p[6:] if len(p) > 6 else []
        keys_extras = [
            "amp_taper_alpha", "width_slope", "revival_chirp",
            "osc_amp", "osc_f0", "osc_f1", "osc_phi0", "osc_phi1"
        ]
        for k, v in zip(keys_extras, extras):
            d[k] = float(v)

    # Final tidy: ensure consistent fields exist (even if missing)
    for k in ["baseline","comb_contrast","revival_time","width0_us","T2_ms","T2_exp",
              "amp_taper_alpha","width_slope","revival_chirp",
              "osc_amp","osc_f0", "osc_f1","osc_phi0","osc_phi1"]:
        d.setdefault(k, None)
    return d

def _echo_summary_lines(t_us, y):
    if len(y) == 0:
        return []
    arr = np.asarray(y, float)
    n = max(3, len(arr)//6)
    early = float(np.nanmean(arr[:n])); late = float(np.nanmean(arr[-n:]))
    return [f"range: {arr.min():.3f}…{arr.max():.3f}",
            f"⟨early⟩→⟨late⟩: {early:.3f}→{late:.3f}"]

def _format_param_box(pdct):
    """Make a compact, readable box for the most relevant parameters."""
    def fmt(v, nd=3):
        return ("—" if v is None else (f"{v:.{nd}g}" if isinstance(v, float) else str(v)))
    lines = []
    lines.append(f"baseline: {fmt(pdct['baseline'])}, comb_contrast: {fmt(pdct['comb_contrast'])}")
    lines.append(f"Trev (μs): {fmt(pdct['revival_time'])}, rev_width (μs): {fmt(pdct['width0_us'])}")
    lines.append(f"T2 (ms): {fmt(pdct['T2_ms'])}, T2_exp (n): {fmt(pdct['T2_exp'])}")
    # Oscillation terms (show only if present / non-zero)
    if (pdct.get("osc_amp") is not None) and (abs(pdct.get("osc_amp",0.0)) > 1e-6):
        lines.append(f"osc_amp: {fmt(pdct['osc_amp'])}")
        if pdct.get("osc_f0", None) is not None:
            lines.append(f"f0 (cyc/μs): {fmt(pdct['osc_f0'])}, f1 (cyc/μs): {fmt(pdct['osc_f1'])}")
        if pdct.get("osc_phi0", None) is not None:
            lines.append(f"φ0 (rad): {fmt(pdct['osc_phi0'])}, φ1 (rad): {fmt(pdct['osc_phi1'])}")
    # Comb shaping
    if any(pdct.get(k, None) not in (None, 0.0) for k in ("amp_taper_alpha","width_slope","revival_chirp")):
        lines.append(f"α: {fmt(pdct['amp_taper_alpha'])}, slope: {fmt(pdct['width_slope'])}, chirp: {fmt(pdct['revival_chirp'])}")
    return lines

# --- UPDATED: now annotates each subplot with a fit-parameter box (and optional χ²_red) ---
def plot_individual_fits(
    norm_counts, 
    norm_counts_ste,
    total_evolution_times,
    popts,
    nv_inds,              # labels same order as popts
    fit_fn_per_nv,        # per-NV fit function
    keep_mask=None,
    show_residuals=True,
    n_fit_points=1000,
    save_prefix=None,
    block=False,
    default_rev_for_plot=39.2,
    red_chi2_list=None,          # OPTIONAL: pass list of reduced-χ² (same order as popts)
    show_param_box=True,         # toggle the on-plot parameter box
):
    N = len(popts)
    assert len(nv_inds) == N, "nv_inds must be same length/order as popts"
    t_all = np.asarray(total_evolution_times, float)

    positions = np.arange(N)
    if keep_mask is not None:
        positions = positions[np.asarray(keep_mask, bool)]

    figs = []
    for pos in positions:
        lbl = nv_inds[pos]
        p   = popts[pos]
        if p is None:
            continue

        fit_fn = fit_fn_per_nv[pos] or fine_decay

        y = np.asarray(norm_counts[lbl], float)
        e = np.asarray(norm_counts_ste[lbl], float)

        if show_residuals:
            fig, (ax, axr) = plt.subplots(2, 1, figsize=(7, 6), sharex=True,
                                          gridspec_kw=dict(height_ratios=[3, 1], hspace=0.06))
        else:
            fig, ax = plt.subplots(1, 1, figsize=(7, 4.6))

        # data
        ax.errorbar(t_all, y, yerr=e, fmt="o", ms=3.5, lw=0.8, alpha=0.9, capsize=2)
        ax.set_ylabel("Normalized NV$^{-}$ population")

        # model curve (dense grid)
        t_fit = np.linspace(np.nanmin(t_all), np.nanmax(t_all), n_fit_points)
        y_fit = _safe_call_fit_fn(fit_fn, t_fit, p, default_rev=default_rev_for_plot)
        ax.plot(t_fit, y_fit, "-", lw=2)

        ax.set_title(f"NV {lbl}")
        ymin = min(np.nanmin(y)-0.1, -0.1)
        ymax = max(np.nanmax(y)+0.1, 1.2)
        ax.set_ylim(ymin, ymax)
        ax.grid(True, alpha=0.25)

        # residuals
        if show_residuals:
            y_model = _safe_call_fit_fn(fit_fn, t_all, p, default_rev=default_rev_for_plot)
            res = y - y_model
            axr.axhline(0.0, ls="--", lw=1.0)
            axr.plot(t_all, res, ".", ms=3.5)
            axr.set_xlabel("Total evolution time (µs)")
            axr.set_ylabel("res.")
            axr.grid(True, alpha=0.25)
        else:
            ax.set_xlabel("Total evolution time (µs)")

        # --- NEW: add a fit-parameter box + echo summary + optional χ²_red ---
        if show_param_box:
            pdict = _params_to_dict(fit_fn, p, default_rev=default_rev_for_plot)
            box_lines = _format_param_box(pdict)
            # top-right: parameter box
            ax.text(
                0.99, 0.98, "\n".join(box_lines), transform=ax.transAxes,
                ha="right", va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.6, lw=0.6)
            )
            # top-left: quick data summary + χ²_red (if provided)
            # left_lines = _echo_summary_lines(t_all, y)
            left_lines = []
            if red_chi2_list is not None and np.isfinite(red_chi2_list[pos]):
                left_lines.append(f"χ²_red: {red_chi2_list[pos]:.3g}")
            if left_lines:
                ax.text(
                    0.01, 0.98, "\n".join(left_lines), transform=ax.transAxes,
                    ha="left", va="top", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.6, lw=0.6)
                )

        # optional save (uses your dm/timestamp if present)
        if save_prefix:
            try:
                file_path = dm.get_file_path(__file__, timestamp, f"{save_prefix}-nv{int(lbl):03d}")
                dm.save_figure(fig, file_path, f"nv{int(lbl):03d}")
            except Exception:
                pass

        figs.append((lbl, fig))

    if figs:
        plt.show(block=block)
    return figs

# =============================================================================
# CLI / Example
# =============================================================================
# ---- helpers ----
def _index_map(unified_keys):
    return {k:i for i,k in enumerate(unified_keys)}

def _safe_sigma(pcov, idx):
    try:
        if pcov is None: return np.nan
        pcov = np.asarray(pcov, float)
        if idx is None or idx >= pcov.shape[0]: return np.nan
        v = pcov[idx, idx]
        return np.sqrt(v) if (np.isfinite(v) and v >= 0) else np.nan
    except Exception:
        return np.nan

def extract_T2_freqs_and_errors(fit_dict, *, pick_freq="max", chi2_fail_thresh=None):
    """
    Returns
    -------
    nv_labels, T2_us, f0_kHz, f1_kHz, A_pick_kHz, chis, fit_fail,
    sT2_us, sf0_kHz, sf1_kHz, sA_pick_kHz
    """
    keys   = fit_dict["unified_keys"]
    kmap   = _index_map(keys)
    popts  = fit_dict["popts"]
    pcovs  = fit_dict.get("pcovs", [None]*len(popts))
    chis   = np.array(fit_dict.get("red_chi2", [np.nan]*len(popts)), float)
    nvlbl  = np.asarray(fit_dict["nv_labels"], int)

    idx_T2 = kmap.get("T2_ms", None)
    idx_f0 = kmap.get("osc_f0", None)
    idx_f1 = kmap.get("osc_f1", None)

    N = len(popts)
    T2_us      = np.full(N, np.nan)
    f0_kHz     = np.full(N, np.nan)
    f1_kHz     = np.full(N, np.nan)
    A_pick_kHz = np.full(N, np.nan)
    sT2_us     = np.full(N, np.nan)
    sf0_kHz    = np.full(N, np.nan)
    sf1_kHz    = np.full(N, np.nan)
    sA_pick_kHz= np.full(N, np.nan)
    fit_fail   = np.zeros(N, bool)

    for i, (p, C) in enumerate(zip(popts, pcovs)):
        if not isinstance(p, (list, tuple)):
            fit_fail[i] = True
            continue

        # chi2 filter (optional)
        if chi2_fail_thresh is not None:
            try:
                if float(chis[i]) > float(chi2_fail_thresh):
                    fit_fail[i] = True
            except Exception:
                pass

        # T2 (ms -> µs) + sigma
        if idx_T2 is not None and idx_T2 < len(p):
            try:
                T2_us[i]  = float(p[idx_T2]) * 1000.0
                sT2_ms    = _safe_sigma(C, idx_T2)
                sT2_us[i] = (sT2_ms * 1000.0) if np.isfinite(sT2_ms) else np.nan
            except Exception:
                pass

        # f0, f1 (cycles/µs = MHz) -> kHz + sigmas
        cand = []
        tags = []  # keep which index produced which
        if idx_f0 is not None and idx_f0 < len(p):
            try:
                f0 = float(p[idx_f0])
                if np.isfinite(f0) and f0 > 0:
                    f0_kHz[i]  = 1000.0 * f0
                    s0         = _safe_sigma(C, idx_f0)
                    sf0_kHz[i] = (1000.0 * s0) if np.isfinite(s0) else np.nan
                    cand.append(f0); tags.append("f0")
            except Exception:
                pass
        if idx_f1 is not None and idx_f1 < len(p):
            try:
                f1 = float(p[idx_f1])
                if np.isfinite(f1) and f1 > 0:
                    f1_kHz[i]  = 1000.0 * f1
                    s1         = _safe_sigma(C, idx_f1)
                    sf1_kHz[i] = (1000.0 * s1) if np.isfinite(s1) else np.nan
                    cand.append(f1); tags.append("f1")
            except Exception:
                pass

        if cand:
            if pick_freq == "min":
                j = int(np.argmin(cand))
            elif pick_freq == "nonzero_first":
                j = 0
            else:  # "max"
                j = int(np.argmax(cand))
            f_pick = cand[j]
            tag    = tags[j]
            A_pick_kHz[i] = 1000.0 * f_pick
            if tag == "f0":
                sA_pick_kHz[i] = sf0_kHz[i]
            else:
                sA_pick_kHz[i] = sf1_kHz[i]

    return (nvlbl, T2_us, f0_kHz, f1_kHz, A_pick_kHz, chis, fit_fail,
            sT2_us, sf0_kHz, sf1_kHz, sA_pick_kHz)

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
    file_stem= "2025_11_12-08_51_17-johnson_204nv_s7-aab2d0"
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
    # _ = plot_individual_fits(
    #     norm_counts, norm_counts_ste, total_evolution_times,
    #     popts,
    #     nv_inds=fit_nv_labels,
    #     fit_fn_per_nv=fit_fns,
    #     # keep_mask=keep_mask,
    #     show_residuals=True,
    #     block=False
    # )
     
    # # --------------------------
    # # Example usage
    # # --------------------------
    # ---- Extract once ----
    (nv, T2_us, f0_kHz, f1_kHz, A_pick_kHz, chis, fit_fail,
    sT2_us, sf0_kHz, sf1_kHz, sA_pick_kHz) = extract_T2_freqs_and_errors(
        data, pick_freq="max", chi2_fail_thresh=3.0
    )

    # ---- Base validity + T2 threshold mask ----
    THRESH_US = 600.0
    valid = np.isfinite(T2_us) & (~fit_fail)
    mask  = valid & (T2_us <= THRESH_US)

    # ---- Build a table (pre-mask) just for printing/sanity checks ----
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
    print(sel[["nv","T2_us","sT2_us","A_pick_kHz","sA_pick_kHz","f0_kHz","f1_kHz","red_chi2"]].to_string(index=False))

    # ---- Optional: detect exact cap hits (e.g., when T2 pinned at a limit) ----
    CAP_US = 600.0
    mask_cap_exact = np.isfinite(T2_us) & np.isclose(T2_us, CAP_US, atol=1e-6)
    cap_indices = np.where(mask_cap_exact)[0]
    cap_labels  = np.asarray(nv)[mask_cap_exact]
    # print(f"[cap=={CAP_US:.0f} µs] Count: {mask_cap_exact.sum()}")
    # print("Indices:", cap_indices.tolist())
    # print("NV labels:", cap_labels.tolist())

    # ---- Apply mask to ALL arrays before plotting ----
    nv_m          = np.asarray(nv)[mask]
    T2_us_m       = np.asarray(T2_us)[mask]
    sT2_us_m      = np.asarray(sT2_us)[mask]
    A_pick_kHz_m  = np.asarray(A_pick_kHz)[mask]
    sA_pick_kHz_m = np.asarray(sA_pick_kHz)[mask]
    # If your plotting uses these too:
    f0_kHz_m      = np.asarray(f0_kHz)[mask]
    f1_kHz_m      = np.asarray(f1_kHz)[mask]
    chis_m        = np.asarray(chis)[mask]
    fit_fail_m    = np.asarray(fit_fail)[mask]   # should now all be False by construction

    # ---- Plot with masked arrays only ----
    # (1) Your existing panels (unchanged)
    plot_sorted_panels_with_err(
        nv_m, T2_us_m, sT2_us_m, A_pick_kHz_m, sA_pick_kHz_m,
        mask_fit_fail=np.zeros_like(fit_fail_m, dtype=bool),
        t2_rel_cap=1.0, t2_pct_cap=95, A_rel_cap=0.75, A_pct_cap=95
    )

    # (2) Experimental sticks: choose a weighting
    #   a) unit weights
    F_kHz, W = build_exp_lines(f0_kHz_m, f1_kHz_m, fmin_kHz=1, fmax_kHz=20000, weight_mode="unit")

    #   b) inverse-variance (needs sA_pick_kHz_m)
    # F_kHz, W = build_exp_lines(f0_kHz_m, f1_kHz_m, fmin_kHz=1, fmax_kHz=20000,
    #                            weight_mode="invvar", sA_pick_kHz=sA_pick_kHz_m)

    #   c) inverse chi^2 (needs chis_m)
    F_kHz, W = build_exp_lines(f0_kHz_m, f1_kHz_m, fmin_kHz=1, fmax_kHz=20000,
                               weight_mode="inv_chi2", chis=chis_m)

    plot_exp_sticks(
        F_kHz, W,
        title="Experimental ESEEM sticks (sorted)",
        weight_caption="unit weight"  # or "1/σ_A^2", or "1/χ²"
    )

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

    kpl.show(block=True)
