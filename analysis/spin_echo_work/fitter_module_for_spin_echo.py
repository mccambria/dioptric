# -*- coding: utf-8 -*-
"""
Spin-echo: finer fit + fitted-figure + parameter panels

- Physics-y comb with quartic lobes, amplitude taper, width growth, chirp
- Optional two-frequency sin^2 beating with phases
- Smoothly plugs into your plotting + data pipeline

Author: Sarohj Chand
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
import numbers

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

# =============================================================================
# Fitting helpers
# =============================================================================

def _smart_percentiles(y, p_low=5, p_high=90):
    y = np.asarray(y, float)
    if y.size == 0:
        return 0.0, 1.0
    lo = float(np.nanpercentile(y, p_low))
    hi = float(np.nanpercentile(y, p_high))
    return lo, hi

def _uniformize(times_us, y):
    """Resample to a uniform grid if needed (for FFT seeding). Returns (tu, yu)."""
    t = np.asarray(times_us, float)
    y = np.asarray(y, float)
    if t.size < 8:
        return t, y
    dt = np.diff(t)
    if np.allclose(dt, dt.mean(), rtol=1e-2, atol=1e-3):
        return t, y
    n = max(256, int(2**np.ceil(np.log2(len(t)))))
    tu = np.linspace(t.min(), t.max(), n)
    yu = np.interp(tu, t, y)
    return tu, yu

def _initial_guess_and_bounds(times_us, y, enable_extras=True, fixed_rev_time=None):
    """
    Smart p0 and bounds for fine_decay that (a) respect your 0→0.6→1 behavior
    and (b) allow both slow and fast beats without violating Nyquist.
    """
    times_us = np.asarray(times_us, float)
    y = np.asarray(y, float)

    # ------- robust baseline & contrast -------
    y_lo, y_hi = _smart_percentiles(y, 5, 90)
    baseline_guess = float(y_hi) if np.isfinite(y_hi) else (y[0] if y.size else 0.6)
    # we want ~0.4 contrast if min near 0 and baseline ~0.6
    comb_contrast_guess = max(0.05, min(baseline_guess - y_lo, baseline_guess - 0.02, 0.9))
    # enforce min >= 0 later via bounds tying comb_contrast to baseline

    # ------- envelope rough seed -------
    # take a late window
    if times_us.size:
        j0 = max(0, len(times_us) - max(7, len(times_us)//10))
        y_late = float(np.nanmean(y[j0:])) if j0 < len(y) else float(y[-1])
        ratio = (baseline_guess - y_late) / max(1e-9, comb_contrast_guess)
        ratio = min(max(ratio, 1e-6), 0.999999)
        T2_exp_guess = 2.0  # slightly softer than 3
        # T2_ms guess scaled by total span; keep conservative
        tspan_us = max(1.0, float(times_us.max() - times_us.min())) if times_us.size else 100.0
        T2_ms_guess = max(0.01, 0.25 * (tspan_us/1000.0) / max(1e-6, (-np.log(ratio))**(1.0/T2_exp_guess)))
    else:
        T2_exp_guess, T2_ms_guess = 2.0, 0.1

    width0_guess = 6.0
    revival_guess = 38.0 if fixed_rev_time is None else fixed_rev_time

    # ------- base vector & bounds -------
    if fixed_rev_time is None:
        p0 = [baseline_guess, comb_contrast_guess, revival_guess, width0_guess, T2_ms_guess, T2_exp_guess]
        # tie comb_contrast to baseline: ub later adjusted below
        lb = [0.0,  0.00, 25.0, 1.0,  0.001, 0.6]
        ub = [1.05, 0.95, 55.0, 20.0, 0.6, 4.0]
    else:
        p0 = [baseline_guess, comb_contrast_guess, width0_guess, T2_ms_guess, T2_exp_guess]
        lb = [0.0,  0.00, 1.0,  0.001, 0.6]
        ub = [1.05, 0.95, 20.0, 0.6, 4.0]

    # tighten comb_contrast upper bound so min >= 0:
    # comb_contrast <= baseline - eps
    eps_min = 0.01
    if fixed_rev_time is None:
        ub[1] = min(ub[1], max(0.05, p0[0] - eps_min))
    else:
        ub[1] = min(ub[1], max(0.05, p0[0] - eps_min))

    if not enable_extras:
        return np.array(p0, float), np.array(lb, float), np.array(ub, float)

    
    # extra_p0 = [0.3, 0.02, 0.0,  0.30, 0.10, 0.01, 0.0, 0.0]
    # extra_lb = [0.0, 0.00, -0.01, -1.00, 0.00, 0.00, -np.pi, -np.pi]
    # extra_ub = [2.0, 0.20,  0.01,  1.00, 0.40, 0.20,  np.pi,  np.pi]
    
    extra_p0 = [0.3, 0.02, 0.0,  0.30, 0.10, 0.01, 0.0, 0.0]
    extra_lb = [0.0, 0.00, -0.06, -1.00, 0.00, 0.00, -np.pi, -np.pi]
    extra_ub = [4.0, 0.80,  0.06,  1.00, 6.00, 6.00,  np.pi,  np.pi]

    p0.extend(extra_p0); lb.extend(extra_lb); ub.extend(extra_ub)
    return np.array(p0, float), np.array(lb, float), np.array(ub, float)

def core_only(tau, baseline, comb_contrast, revival_time, width0_us, T2_ms, T2_exp,
              amp_taper_alpha=0.0, width_slope=0.0, revival_chirp=0.0):
    tau = np.asarray(tau, float)
    env = np.exp(-((tau / (1000.0*T2_ms)) ** T2_exp))
    tau_max = float(np.nanmax(tau)) if tau.size else 0.0
    n_guess = max(1, min(64, int(np.ceil(1.2 * tau_max / max(1e-9, revival_time))) + 1))
    comb = _comb_quartic_powerlaw(tau, revival_time, width0_us,
                                  amp_taper_alpha or 0.0,
                                  width_slope or 0.0,
                                  revival_chirp or 0.0,
                                  n_guess)
    return baseline - comb_contrast * env * comb


def _chi2_red(y, yerr, yfit, npar):
    resid = (y - yfit) / np.maximum(1e-12, yerr)
    chi2 = float(np.sum(resid**2))
    dof  = max(1, len(y) - npar)
    return chi2 / dof

def _sanitize_trace(times_us, y, yerr, err_floor=1e-3):
    """Remove non-finite rows; enforce an error floor."""
    m = np.isfinite(times_us) & np.isfinite(y) & np.isfinite(yerr)
    t = times_us[m].astype(float)
    yy = y[m].astype(float)
    ee = np.maximum(err_floor, np.abs(yerr[m].astype(float)))
    # drop duplicates in time (keep first)
    if t.size:
        _, uniq_idx = np.unique(t, return_index=True)
        t, yy, ee = t[uniq_idx], yy[uniq_idx], ee[uniq_idx]
    return t, yy, ee


def _revival_weights(t, Trev, w0, width_slope=0.0, nrev=8, boost=2.0):
    w = np.ones_like(t, float)
    for k in range(nrev):
        mu_k = k * Trev
        wk = w0 * (1.0 + k*width_slope)
        w *= 1.0 + (boost-1.0) * np.exp(-((t - mu_k)/(1.5*wk+1e-9))**2)
    return w

def _fit_curve_fit(fit_fn, times_us, y, yerr, p0, lb, ub, maxfev):
    popt, pcov, _ = curve_fit(
        fit_fn,
        times_us,
        y,
        p0,
        yerr,
        bounds=[lb, ub],
        ftol=1e-7, xtol=1e-7, gtol=1e-7,
        maxfev=maxfev,              # <- IMPORTANT
    )
    yfit = fit_fn(times_us, *popt)
    red  = _chi2_red(y, yerr, yfit, len(popt))
    return popt, pcov, red

def _fit_least_squares(fit_fn, times_us, y, yerr, p0, lb, ub, max_nfev, weights=None):
    """Robust fallback using soft-L1; optional per-point weights."""
    t = np.asarray(times_us, float)
    yy = np.asarray(y, float)
    ee = np.maximum(1e-12, np.asarray(yerr, float))
    ww = None if weights is None else np.asarray(weights, float)

    def resid(p):
        r = (yy - fit_fn(t, *p)) / ee
        return r if ww is None else r * ww

    res = least_squares(
        resid, x0=np.asarray(p0, float), bounds=(lb, ub),
        loss="soft_l1", f_scale=1.0,
        max_nfev=max_nfev, ftol=1e-8, xtol=1e-8, gtol=1e-8,
    )
    popt = res.x
    yfit = fit_fn(t, *popt)
    red  = _chi2_red(yy, ee, yfit, len(popt))
    return popt, None, red

def _pick_best(cands):
    """cands is list of tuples (name, popt, pcov, red, fit_fn, note)"""
    good = [(i,c) for i,c in enumerate(cands) if np.isfinite(c[3])]
    if not good:
        return None
    # pick smallest reduced chi^2
    i_best = min(good, key=lambda ic: ic[1][3])[0]
    return cands[i_best]


def _core_len_for_fn(fit_fn):
    """Number of core params before 'extras' start."""
    # fine_decay: [baseline, comb_contrast, revival_time, width0_us, T2_ms, T2_exp] -> 6
    # fine_decay_fixed_revival: [baseline, comb_contrast, width0_us, T2_ms, T2_exp] -> 5
    return 6 if fit_fn is fine_decay else 5

def _set_osc_amp_bounds(lb, ub, fit_fn, new_min, new_max):
    """In-place change of osc_amp bounds in [amp_taper_alpha, width_slope, revival_chirp, osc_amp, ...]."""
    k0 = _core_len_for_fn(fit_fn)  # start of extras
    idx_amp = k0 + 3               # extras: α, slope, chirp, osc_amp, ...
    if idx_amp >= len(lb):
        # extras disabled; nothing to set
        return False
    lb[idx_amp] = float(new_min)
    ub[idx_amp] = float(new_max)
    return True

# === NEW: parameter indexing helpers =========================================
def _param_index_map(fit_fn):
    """
    Return a dict mapping parameter names -> indices for the chosen fit_fn.
    Order (fine_decay):
      [baseline, comb_contrast, revival_time, width0_us, T2_ms, T2_exp,
       amp_taper_alpha, width_slope, revival_chirp, osc_contrast,
       osc_f0, osc_f1, osc_phi0, osc_phi1]

    For fine_decay_fixed_revival, revival_time is absent from the core.
    """
    if fit_fn is fine_decay:
        names = [
            "baseline", "comb_contrast", "revival_time", "width0_us", "T2_ms", "T2_exp",
            "amp_taper_alpha", "width_slope", "revival_chirp", "osc_contrast",
            "osc_f0", "osc_f1", "osc_phi0", "osc_phi1"
        ]
    else:
        names = [
            "baseline", "comb_contrast", "width0_us", "T2_ms", "T2_exp",
            "amp_taper_alpha", "width_slope", "revival_chirp", "osc_contrast",
            "osc_f0", "osc_f1", "osc_phi0", "osc_phi1"
        ]
    return {n:i for i,n in enumerate(names)}

def _set_bounds(lb, ub, idx, vmin, vmax):
    lb[idx] = float(vmin)
    ub[idx] = float(vmax)

def _set_initial(p0, idx, val):
    p0[idx] = float(val)

def _get_val(vec, idx, default=None):
    return (float(vec[idx]) if 0 <= idx < len(vec) else default)

# === NEW: frequency band inference from sampling =============================
def _freq_band_from_times(times_us, margin=0.05, fmax_cap=None):
    """
    Infer a sensible [fmin, fmax] band (cycles/µs) from time span and min spacing.
    fmin ~ 1/(span), fmax ~ 1/(2*dt_min). Shrink/expand slightly via 'margin'.
    """
    t = np.asarray(times_us, float)
    if t.size < 2:
        return 0.0, 1.0
    span = max(1e-9, float(t.max() - t.min()))
    dt_min = max(1e-9, float(np.diff(np.unique(t)).min()))
    fmin = 1.0 / span
    fmax = 0.5 / dt_min
    if fmax_cap is not None:
        fmax = min(fmax, float(fmax_cap))
    # apply margin
    fmin = max(0.0, (1.0 - margin) * fmin)
    fmax = (1.0 + margin) * fmax
    return fmin, fmax

def _normalize_freq_band(times_us, band=None, fmax_cap=None):
    fmin_samp, fmax_samp = _freq_band_from_times(times_us, margin=0.0, fmax_cap=fmax_cap)
    if band is None:
        fmin, fmax = fmin_samp, fmax_samp
    else:
        fmin, fmax = map(float, band)
        if fmin > fmax:  # swap if inverted
            fmin, fmax = fmax, fmin
        # intersect with sampling band (Nyquist-safe)
        fmin = max(0.0, max(fmin, fmin_samp))
        fmax = min(fmax, fmax_samp)
    # ensure non-empty, add tiny slack
    eps = 1e-9
    if not np.isfinite(fmin) or not np.isfinite(fmax) or fmax <= fmin + eps:
        # fallback to safe band from sampling
        fmin, fmax = fmin_samp, fmax_samp
        if fmax <= fmin + eps:
            # last-ditch: pick a tiny positive window
            fmin, fmax = 0.0, max(1e-3, fmax_samp)
    return float(fmin), float(fmax)

def _clone_vecs(*vecs):
    return [np.array(v, float) if v is not None else None for v in vecs]

def _grid_product(dict_of_lists):
    """
    dict_of_lists: {"osc_f0":[...], "osc_f1":[...], "osc_phi0":[...], ...}
    Yields dicts with one selection per key. Empty dict if input empty.
    """
    if not dict_of_lists:
        yield {}
        return
    keys = list(dict_of_lists.keys())
    lists = [dict_of_lists[k] for k in keys]
    sizes = [len(L) for L in lists]
    if any(s == 0 for s in sizes):
        return
    import itertools
    for combo in itertools.product(*lists):
        yield {k:v for k,v in zip(keys, combo)}

def _apply_param_overrides(p0, lb, ub, fit_fn, overrides=None, bound_boxes=None):
    """
    overrides: {"osc_f0": value, "revival_time": value, ...} -> set initial p0
    bound_boxes: {"osc_f0": (min,max), ...} -> set bounds
    Returns updated (p0, lb, ub).
    """
    pmap = _param_index_map(fit_fn)
    p0, lb, ub = _clone_vecs(p0, lb, ub)
    overrides = overrides or {}
    bound_boxes = bound_boxes or {}
    for name, val in overrides.items():
        if name in pmap:
            _set_initial(p0, pmap[name], val)
    for name, box in bound_boxes.items():
        if (name in pmap) and (box is not None):
            _set_bounds(lb, ub, pmap[name], box[0], box[1])
    return p0, lb, ub

def _sanitize_bound_boxes(bound_boxes, band):
    if not isinstance(bound_boxes, dict): 
        return {}
    bmin, bmax = band
    out = {}
    for k, box in bound_boxes.items():
        if box is None: 
            continue
        lo, hi = box
        if lo > hi: 
            lo, hi = hi, lo
        lo = max(0.0, max(lo, bmin))
        hi = min(hi, bmax)
        if hi <= lo + 1e-9:
            # collapse to a tiny valid window around the midpoint
            mid = 0.5*(bmin + bmax)
            lo, hi = max(0.0, mid*0.98), mid*1.02
        out[k] = (float(lo), float(hi))
    return out

def build_freq_pairs(freqs, band, max_pairs=40, min_sep=0.01):
    """Return a small set of (f0,f1) with f1 < f0, well separated, in-band."""
    lo, hi = band
    fs = [
        f for f in sorted(set(round(float(x), 9) for x in freqs))
        if np.isfinite(f) and lo <= f <= hi
    ]
    # Prefer the top-K by FFT magnitude if you have that; otherwise take spaced picks:
    if len(fs) > max_pairs:
        step = max(1, len(fs)//max_pairs)
        fs = fs[::step]
    out = []
    for i, f0 in enumerate(fs):
        for f1 in fs[:i]:              # enforce f1 < f0 (no permutations, no equals)
            if abs(f0 - f1) < min_sep: # avoid nearly identical pairs
                continue
            out.append((f0, f1))
    return out[:max_pairs]

# ---- parameter indexing helpers --------------------------------------------

def _param_index_map(fit_fn):
    """
    Map parameter names to indices for the chosen fit function.

    For fine_decay (free revival_time), the expected order is:
      [baseline, comb_contrast, revival_time, width0_us, T2_ms, T2_exp,
       amp_taper_alpha, width_slope, revival_chirp,
       osc_amp, osc_f0, osc_phi0, osc_f1, osc_phi1]

    For fine_decay_fixed_revival (no revival_time in core), the order is:
      [baseline, comb_contrast, width0_us, T2_ms, T2_exp,
       amp_taper_alpha, width_slope, revival_chirp,
       osc_amp, osc_f0, osc_phi0, osc_f1, osc_phi1]
    """
    if fit_fn is fine_decay:
        names = [
            "baseline","comb_contrast","revival_time","width0_us","T2_ms","T2_exp",
            "amp_taper_alpha","width_slope","revival_chirp",
            "osc_amp","osc_f0","osc_phi0","osc_f1","osc_phi1"
        ]
    else:  # fine_decay_fixed_revival
        names = [
            "baseline","comb_contrast","width0_us","T2_ms","T2_exp",
            "amp_taper_alpha","width_slope","revival_chirp",
            "osc_amp","osc_f0","osc_phi0","osc_f1","osc_phi1"
        ]
    return {n: i for i, n in enumerate(names)}

def _set_bounds(lb, ub, idx, vmin, vmax):
    idx = int(idx)
    lb[idx] = float(vmin)
    ub[idx] = float(vmax)

def _set_initial(p0, idx, val):
    idx = int(idx)
    p0[idx] = float(val)


# ================= Bound-hit repair utilities ================================

def _param_index_map(fit_fn):
    """Parameter order map matching your fine_decay / fine_decay_fixed_revival."""
    if fit_fn is fine_decay:
        names = [
            "baseline","comb_contrast","revival_time","width0_us","T2_ms","T2_exp",
            "amp_taper_alpha","width_slope","revival_chirp",
            "osc_amp","osc_f0","osc_phi0","osc_f1","osc_phi1"
        ]
    else:  # fine_decay_fixed_revival
        names = [
            "baseline","comb_contrast","width0_us","T2_ms","T2_exp",
            "amp_taper_alpha","width_slope","revival_chirp",
            "osc_amp","osc_f0","osc_phi0","osc_f1","osc_phi1"
        ]
    return {n:i for i,n in enumerate(names)}

def _set_bounds(lb, ub, idx, vmin, vmax):
    idx = int(idx); lb[idx] = float(vmin); ub[idx] = float(vmax)

def _set_initial(p0, idx, val):
    idx = int(idx); p0[idx] = float(val)


def _bound_hits(popt, lb, ub, frac_tol=0.01, abs_tol=1e-9):
    """
    Return {idx: 'low'|'high'} for params within tol of bounds.
    """
    hits = {}
    popt = np.asarray(popt, float)
    lb   = np.asarray(lb,   float)
    ub   = np.asarray(ub,   float)
    span = np.maximum(ub - lb, 0.0)
    low_tol  = np.maximum(frac_tol*span, abs_tol)
    high_tol = np.maximum(frac_tol*span, abs_tol)
    for i, (p, lo, hi, ltol, htol) in enumerate(zip(popt, lb, ub, low_tol, high_tol)):
        if not np.isfinite(p): 
            continue
        if (p - lo) <= ltol:
            hits[i] = "low"
        elif (hi - p) <= htol:
            hits[i] = "high"
    return hits

def _repair_bounds_for_hits(
    lb, ub, hits, pmap, *,
    physics_caps=None,          # optional dict: name -> (hard_lo, hard_hi)
    freq_band=None,             # optional (flo, fhi) in cycles/µs for osc_f*
    scale=1.8,
    min_span=1e-6,
):
    """
    Widen only the parameters that hit bounds by 'scale', respecting physics caps.
    Guarantees non-negative frequencies for osc_f0/osc_f1 and keeps them within freq_band if given.
    Returns (lb2, ub2).
    """
    import numpy as np

    lb2 = np.array(lb, float)
    ub2 = np.array(ub, float)
    name_by_idx = {j: i for i, j in pmap.items()}  # idx -> name

    # ---- sensible defaults if not provided ----
    # NB: you can tweak these to your experiment
    if physics_caps is None:
        physics_caps = {}

    flo, fhi = (None, None)
    if freq_band is not None and len(freq_band) == 2:
        flo, fhi = float(freq_band[0]), float(freq_band[1])

    def _caps_for(name, cur_lo, cur_hi):
        # start from provided caps (if any)
        hard_lo, hard_hi = physics_caps.get(name, (-np.inf, np.inf))

        # non-negativity for frequencies, and keep within band if known
        if name in ("osc_f0", "osc_f1"):
            if np.isfinite(hard_lo):
                hard_lo = max(hard_lo, 0.0)
            else:
                hard_lo = 0.0  # enforce non-negative freq
            if flo is not None:
                hard_lo = max(hard_lo, 0.0)  # keep ≥0 even if band is tiny
            if fhi is not None and np.isfinite(hard_hi):
                hard_hi = min(hard_hi, fhi)
            elif fhi is not None:
                hard_hi = fhi

        # a few gentle, safe caps (only if not already provided)
        if name == "T2_ms" and not np.isfinite(physics_caps.get("T2_ms", (np.nan,np.nan))[0]):
            hard_lo = max(hard_lo, 1e-6)    # >0
        if name == "width0_us" and not np.isfinite(physics_caps.get("width0_us", (np.nan,np.nan))[0]):
            hard_lo = max(hard_lo, 1e-6)    # >0
        if name == "T2_exp" and not np.isfinite(physics_caps.get("T2_exp", (np.nan,np.nan))[0]):
            # typical range; adjust to your prior
            hard_lo = max(hard_lo, 0.25)
            hard_hi = min(hard_hi, 4.0) if np.isfinite(hard_hi) else 4.0

        return hard_lo, hard_hi

    for idx, side in hits.items():
        name = name_by_idx.get(idx, None)
        if name is None:
            continue

        lo, hi = float(lb2[idx]), float(ub2[idx])
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            continue

        span = max(min_span, hi - lo)
        hard_lo, hard_hi = _caps_for(name, lo, hi)

        if side == "low":
            lo_new = lo - (scale - 1.0) * span
            hi_new = hi + 0.25 * span
        else:  # side == "high"
            lo_new = lo - 0.25 * span
            hi_new = hi + (scale - 1.0) * span

        # clamp to hard caps
        lo_new = max(hard_lo, lo_new)
        hi_new = min(hard_hi, hi_new)

        # enforce non-negative for frequencies explicitly (belt & suspenders)
        if name in ("osc_f0", "osc_f1"):
            lo_new = max(0.0, lo_new)

        # keep min span
        if not np.isfinite(lo_new) or not np.isfinite(hi_new) or hi_new <= lo_new:
            mid = 0.5 * (lo + hi)
            lo_new, hi_new = mid - 0.5 * min_span, mid + 0.5 * min_span
            # re-apply caps
            lo_new = max(hard_lo, lo_new)
            hi_new = min(hard_hi, hi_new)
            if hi_new - lo_new < min_span:
                # final guarantee
                hi_new = lo_new + min_span

        lb2[idx], ub2[idx] = float(lo_new), float(hi_new)

    return lb2, ub2


def _retie_contrast_to_baseline(p0, lb, ub, pmap, eps=0.01):
    """Ensure comb_contrast ≤ baseline - eps by adjusting its UB."""
    if "baseline" in pmap and "comb_contrast" in pmap:
        ib = pmap["baseline"]; ic = pmap["comb_contrast"]
        ub[ic] = min(ub[ic], max(0.0, p0[ib] - eps))
    return p0, lb, ub

def _reseed_to_center(p0, lb, ub, only_idxs=None):
    """Set p0 to centers (optionally only for indices in only_idxs)."""
    p0 = np.array(p0, float)
    lb = np.array(lb, float)
    ub = np.array(ub, float)
    if only_idxs is None:
        mask = np.ones_like(p0, dtype=bool)
    else:
        mask = np.zeros_like(p0, dtype=bool)
        mask[list(only_idxs)] = True
    centers = 0.5*(lb + ub)
    p0[mask] = centers[mask]
    return p0

# Physics-aware hard caps (tune to your platform)
_PHYS_CAPS = {
    "baseline": (0.0, 1.1),
    "comb_contrast": (0.0, 1.0),
    "revival_time": (20.0, 70.0),     # µs; adjust to your B-field
    "width0_us": (0.2, 50.0),
    "T2_ms": (0.003, 5.0),
    "T2_exp": (0.6, 3.0),
    "amp_taper_alpha": (0.0, 3.0),
    "width_slope": (-1.0, 1.0),
    "revival_chirp": (-0.01, 0.01),
    "osc_amp": (-10.0, 10.0),         # amp windows still govern effective range
    "osc_f0": (0.0, 5.0),
    "osc_f1": (0.0, 5.0),
    "osc_phi0": (-np.pi, np.pi),
    "osc_phi1": (-np.pi, np.pi),
}


def _uniq_in_band(freqs, lo, hi, r=9):
    s, out = set(), []
    for f in freqs:
        try: f = float(f)
        except: continue
        if np.isfinite(f) and lo <= f <= hi:
            k = round(f, r)
            if k not in s: s.add(k); out.append(f)
    return sorted(out)

def _maxmin(vals, k):
    if not vals: return []
    if k >= len(vals): return list(vals)
    picked = [vals[0], vals[-1]]; cand = vals[1:-1]
    while len(picked) < k and cand:
        best = max(cand, key=lambda v: min(abs(v-c) for c in picked))
        picked.append(best); cand.remove(best)
    return sorted(picked)

def diversified_prior_overrides(
    freq_seeds, band, *, min_sep=0.01, nyq_guard=0.9,
    n_single=8, n_pairs=20, nbuckets=5, jitter=0.02, seed=1234
):
    random.seed(seed)
    lo, hi = band; hi *= nyq_guard
    seeds = _uniq_in_band(freq_seeds, lo, hi)
    if not seeds: return []

    # ---- singles (f1=0): bucketed + max-min + light jitter ----
    edges = np.linspace(seeds[0], seeds[-1], nbuckets+1)
    buckets = [[] for _ in range(nbuckets)]
    bi = 0
    for v in seeds:
        while bi < nbuckets-1 and v > edges[bi+1]: bi += 1
        buckets[bi].append(v)

    singles = []
    for B in buckets:
        singles += _maxmin(B, min(2, len(B)))
    if len(singles) < n_single:
        singles = _uniq_in_band(singles + _maxmin(seeds, n_single), lo, hi)

    def apart(x, S): return all(abs(x-s) >= min_sep for s in S)
    for v in list(singles)[: max(1, n_single//3)]:
        j = v*(1+random.uniform(-jitter, jitter))
        if lo <= j <= hi and apart(j, singles): singles.append(j)
    singles = _uniq_in_band(singles, lo, hi)[:n_single]

    # ---- pairs: low↔high + harmonics + max-min fallback ----
    ovs, seen = [], set()
    def push(a,b):
        a,b = (a,b) if a<=b else (b,a)
        if abs(a-b)<min_sep: return False
        k=(round(a,6),round(b,6))
        if k in seen: return False
        seen.add(k); ovs.append({"osc_f0":a,"osc_f1":b}); return True

    for f0 in singles: push(f0, 0.0)
    need = max(0, n_pairs)

    # low↔high
    mid = nbuckets//2 or 1
    lows  = [b for b in buckets[:mid] if b]
    highs = [b for b in buckets[mid:] if b]
    for BL in lows:
        for BH in highs:
            for a in _maxmin(BL, min(2,len(BL))):
                for b in _maxmin(BH, min(2,len(BH))):
                    if need and push(a,b): need -= 1
                    if need==0: break
                if need==0: break
        if need==0: break

    # harmonics (½, ⅔, ¾)
    if need:
        for f0 in _maxmin(seeds, min(8,len(seeds))):
            for r in (0.5, 2/3, 0.75):
                f1 = f0*r
                if lo<=f1<=hi and need and push(f0,f1): need -= 1
            if need==0: break

    # max-min fallback by separation
    if need:
        cand = sorted(
            ((abs(a-b),a,b) for a,b in itertools.combinations(seeds,2) if abs(a-b)>=min_sep),
            reverse=True
        )
        for _,a,b in cand:
            if need and push(a,b): need -= 1
            if need==0: break

    return ovs

def _score_tuple(popt, redchi, lb, ub, pmap, *, amp_weight=1e-3, bound_weight=0.1):
    """
    Return a tuple for min()-selection:
      ( redchi + bound_penalty, amp_norm )
    so we first pick good χ² that are NOT at bounds; then prefer smaller osc_amp.

    - bound_penalty adds 'bound_weight' if a key param (T2_ms, T2_exp, osc_f*) is near its bound.
    - amp_norm is |osc_amp| (or 0 if absent), scaled by amp_weight so it is a tie-breaker only.
    """
    pen = 0.0
    if pmap:  # protect if pmap is None
        def _near_wall(idx):
            if idx is None or idx >= len(popt): return False
            lo, hi, x = lb[idx], ub[idx], popt[idx]
            if not (np.isfinite(lo) and np.isfinite(hi)): return False
            span = max(1e-12, hi - lo)
            return (x - lo) <= 0.01*span or (hi - x) <= 0.01*span

        iT2   = pmap.get("T2_ms")
        inT2  = pmap.get("T2_exp")
        if iT2 is not None and _near_wall(iT2): pen += bound_weight * 2.0  # extra-punish T2 at wall
        if inT2 is not None and _near_wall(inT2): pen += bound_weight

        for k in ("osc_f0","osc_f1"):
            ik = pmap.get(k)
            if ik is not None and _near_wall(ik): pen += 0.5*bound_weight

    iA = pmap.get("osc_amp") if pmap else None
    amp_norm = abs(float(popt[iA])) if (iA is not None and iA < len(popt)) else 0.0
    return (float(redchi) + pen, amp_weight * amp_norm)


# ================= Parameter indexing / bounds helpers =======================

def _param_index_map(fit_fn):
    """
    Map parameter names to indices for the chosen fit function.

    For fine_decay (free revival_time):
      [baseline, comb_contrast, revival_time, width0_us, T2_ms, T2_exp,
       amp_taper_alpha, width_slope, revival_chirp,
       osc_amp, osc_f0, osc_phi0, osc_f1, osc_phi1]

    For fine_decay_fixed_revival (no revival_time in core):
      [baseline, comb_contrast, width0_us, T2_ms, T2_exp,
       amp_taper_alpha, width_slope, revival_chirp,
       osc_amp, osc_f0, osc_phi0, osc_f1, osc_phi1]
    """
    if fit_fn is fine_decay:
        names = [
            "baseline","comb_contrast","revival_time","width0_us","T2_ms","T2_exp",
            "amp_taper_alpha","width_slope","revival_chirp",
            "osc_amp","osc_f0","osc_phi0","osc_f1","osc_phi1",
        ]
    else:  # fine_decay_fixed_revival
        names = [
            "baseline","comb_contrast","width0_us","T2_ms","T2_exp",
            "amp_taper_alpha","width_slope","revival_chirp",
            "osc_amp","osc_f0","osc_phi0","osc_f1","osc_phi1",
        ]
    return {n: i for i, n in enumerate(names)}

def _core_len_for_fn(fit_fn):
    """Number of 'core' params before oscillation extras start."""
    return 6 if fit_fn is fine_decay else 5

def _set_bounds(lb, ub, idx, vmin, vmax):
    idx = int(idx)
    lb[idx] = float(vmin)
    ub[idx] = float(vmax)

def _set_initial(p0, idx, val):
    idx = int(idx)
    p0[idx] = float(val)

def _get_val(vec, idx, default=None):
    return float(vec[idx]) if (0 <= idx < len(vec)) else default

def _set_osc_amp_bounds(lb, ub, fit_fn, new_min, new_max):
    """
    In-place change of osc_amp bounds in
    [amp_taper_alpha, width_slope, revival_chirp, osc_amp, ...].
    """
    k0 = _core_len_for_fn(fit_fn)  # start of extras
    idx_amp = k0 + 3               # extras: α, slope, chirp, osc_amp, ...
    if idx_amp >= len(lb):
        return False
    lb[idx_amp] = float(new_min)
    ub[idx_amp] = float(new_max)
    return True

def _clone_vecs(*vecs):
    return [np.array(v, float) if v is not None else None for v in vecs]

# Small osc_amp window by default – more physics-motivated and numerically stable
def _amp_windows_and_seeds():
    # windows in osc_amp, and a few seeds inside each
    return [
        ((-0.6, 0.6), [0.15, 0.35]),
        ((-1.0, 1.0), [0.35, 0.75]),
    ]


def _is_num(x):
    return isinstance(x, numbers.Number) and np.isfinite(x)

def _sanitize_overrides(ov: dict, pmap: dict) -> dict:
    clean = {}
    for k, v in (ov or {}).items():
        if k not in pmap:
            # skip unknown params silently (or warn)
            continue
        # unwrap 1-element containers
        if isinstance(v, (list, tuple, np.ndarray)):
            if len(v) != 1:
                raise TypeError(f"override '{k}' must be scalar, got {type(v)} with len={len(v)}")
            v = v[0]
        if isinstance(v, dict):
            raise TypeError(f"override '{k}' must be scalar, got dict")
        if not _is_num(v):
            raise TypeError(f"override '{k}' must be numeric, got {v!r} ({type(v)})")
        clean[k] = float(v)
    return clean

def extract_catalog_pairs_cyc(records, orientations, band_kHz, weight_mode):
    """
    Extract (f0,f1) pairs from catalog in cycles/µs (MHz):

      f0 = max(f_plus, f_minus)
      f1 = min(f_plus, f_minus)

    Both must fall inside [kmin, kmax] in kHz.
    """
    pairs = []
    weights = []
    kmin, kmax = band_kHz
    pair_indeces = []

    for r in records:
        # optional orientation filter
        if orientations is not None:
            ori = tuple(r.get("orientation", ()))
            if ori not in orientations:
                continue

        fplus_Hz  = r.get("f_plus_Hz",  None)
        fminus_Hz = r.get("f_minus_Hz", None)
        if fplus_Hz is None or fminus_Hz is None:
            continue

        fplus_kHz  = float(fplus_Hz)  / 1e3
        fminus_kHz = float(fminus_Hz) / 1e3

        # both must lie within the kHz band
        if not (kmin <= fplus_kHz <= kmax and kmin <= fminus_kHz <= kmax):
            continue

        # convert to cycles/µs (== MHz)
        fplus_cyc  = fplus_kHz  / 1000.0
        fminus_cyc = fminus_kHz / 1000.0

        f0 = max(fplus_cyc, fminus_cyc)
        f1 = min(fplus_cyc, fminus_cyc)
        if f0 <= 0.0 or f1 < 0.0:
            continue

        pairs.append((f0, f1))
        if weight_mode == "kappa":
            w = float(r.get("kappa", 1.0))
        elif weight_mode == "per_line":
            w = 1.0
        else:
            w = 1.0
        weights.append(w)
        
        # 
        site_index = r.get("site_index", None)
        pair_indeces.append(site_index)

    pairs = np.asarray(pairs, float)
    weights = np.asarray(weights, float) if weights else None
    return pairs, weights, pair_indeces

# ================= Updated fitter with bound-repair ===========================

def fit_one_nv_with_freq_sweeps(
    times_us, y, yerr,
    amp_bound_grid=((-0.5, 0.5), (-1, 1), (-2, 2)),
    *,
    # Frequency handling
    freq_bound_boxes=None,              # e.g. {"osc_f0": (0.04, 0.36), "osc_f1": (0.0, 0.36)}
    freq_seed_band=None,                # (fmin, fmax) in cycles/µs; if None -> inferred
    freq_seed_n_peaks=2,                # keep small for speed
    seed_include_harmonics=False,       # enable only if residuals remain high
    # Extra multi-starts (any param in your model)
    extra_overrides_grid=None,          # e.g. {"osc_phi0":[0, np.pi/2], "osc_phi1":[0]}
    # Model choice
    use_fixed_revival=False,
    enable_extras=True,
    fixed_rev_time=37.6,
    # Prior harmonic pass (fast pre-fit using simple (f0,f1) ratios)
    prior_enable=True,
    prior_pairs_topK=24,                  # small, fast
    prior_min_sep=0.01,
    early_stop_redchi=None,              # stop after prior if already excellent
    # Optimizer budgets (progressive)
    small_maxfev=40000,
    small_max_nfev=60000,
    big_maxfev=120000,
    big_max_nfev=180000,
    refine_target_red=1.05,             # if best > this, refine the winner with big budgets
    # Coarse screening
    coarse_K=8,                         # keep top-K seeds per amp window
    coarse_max_nfev=200000,
    # Data cleaning
    err_floor=1e-3,
    # NEW:
    allowed_records=None,            # list of dicts (your JSON)
    allowed_orientations=None,       # e.g. [(1,1,1)]
    allowed_tol_kHz=8.0,             # half-width window for bounds around each allowed line
    allowed_weight_mode="none",      # "none" | "kappa" | "per_line"
    p_occ=0.011,
    # Logging
    verbose=True,
):
    """
    Robust & fast single-NV fitter with:
      • FFT-derived frequency seeds (band-limited)
      • Optional harmonic prior pass (quick (f0,f1) trials)
      • Coarse→fine screening: keep only top-K survivors for heavy fits
      • Early stop if a great χ² is reached
      • Progressive budgets: cheap pass first, refine the winner only if needed
      • NEW: bound-hit repair — widen only the wall-hitting params, reseed, refit

    Returns
    -------
    popt_best, pcov_best, redchi_best, fn_best, amp_window_best, overrides_used

    Requires (already in your code base):
      _sanitize_trace, _initial_guess_and_bounds, _core_len_for_fn,
      _normalize_freq_band, _sanitize_bound_boxes, _seed_freqs_fft,
      _clone_vecs, _set_osc_amp_bounds, _apply_param_overrides, _grid_product,
      _fit_curve_fit, _fit_least_squares, _pick_best, _chi2_red,
      fine_decay, fine_decay_fixed_revival, harmonic_candidates
    """

    # ---------- local cheap coarse scorer (core-only LSQ) ----------
    def _coarse_redchi(fit_fn, t, yy, ee, p0, lb, ub, kcore):
        try:
            popt, _, _ = _fit_least_squares(
                fit_fn, t, yy, ee,
                p0[:kcore], lb[:kcore], ub[:kcore],
                max_nfev=coarse_max_nfev
            )
            yfit = fit_fn(t, *popt)
            return float(_chi2_red(yy, ee, yfit, len(popt)))
        except Exception:
            return float('inf')

    # -----------------------------
    # 1) Sanitize input trace
    # -----------------------------
    t, yy, ee = _sanitize_trace(times_us, y, yerr, err_floor=err_floor)
    if len(t) < 8:
        raise RuntimeError("Too few valid points after sanitization.")

    # -----------------------------
    # 2) Base model & initial bounds
    # -----------------------------
    fit_fn_base = fine_decay if not use_fixed_revival else fine_decay_fixed_revival
    p0_base, lb_base, ub_base = _initial_guess_and_bounds(
        t, yy, enable_extras,
        fixed_rev_time=(None if fit_fn_base is fine_decay else fixed_rev_time),
    )
    pmap = _param_index_map(fit_fn_base)

    # Core-only vectors (for FFT detrending / quick trials)
    if enable_extras:
        kcore = _core_len_for_fn(fit_fn_base)
    else:
        kcore = len(p0_base)


    # -----------------------------
    # 3) Frequency band + bound boxes
    # -----------------------------
    # band = _normalize_freq_band(t, freq_seed_band, fmax_cap=None)  # (lo, hi)
    band = freq_seed_band  # (lo, hi)
    lo, hi = band

    bound_boxes = {}
    if enable_extras and "osc_f0" in pmap:
        bound_boxes["osc_f0"] = (max(0.0, lo * 0.8), hi * 1.1)
    if enable_extras and "osc_f1" in pmap:
        bound_boxes["osc_f1"] = (0.0, hi * 1.1)

    if isinstance(freq_bound_boxes, dict):
        for k, v in freq_bound_boxes.items():
            if k in pmap and v is not None:
                bound_boxes[k] = tuple(v)

    bound_boxes = _sanitize_bound_boxes(bound_boxes, band)

    # -----------------------------
    # 4) Allowed-line PAIRS ONLY (no FFT, no cross-pair mixing)
    # -----------------------------
    if allowed_records is None or len(allowed_records) == 0:
        raise RuntimeError("allowed_records must be provided to restrict f to f+/− pairs.")

    band_kHz = (lo * 1000.0, hi * 1000.0)
    print(band_kHz)
    # Now also get cite/site IDs
    pairs_cyc, pair_wts, pair_ids = extract_catalog_pairs_cyc(
        allowed_records, allowed_orientations, band_kHz, allowed_weight_mode
    )

    if pairs_cyc.size == 0:
        raise RuntimeError("No (f+, f−) pairs fall within the sampling/Nyquist band.")

    # Make pair_ids indexable like the arrays
    pair_ids = np.asarray(pair_ids, dtype=object)

    nyq_guard = 0.99
    hi_guard = nyq_guard * hi

    # mask out-of-band pairs
    m = (
        (pairs_cyc[:, 0] >= lo) & (pairs_cyc[:, 0] <= hi_guard) &
        (pairs_cyc[:, 1] >= 0.0) & (pairs_cyc[:, 1] <= hi_guard)
    )
    pairs_cyc = pairs_cyc[m]
    pair_ids  = pair_ids[m]
    if pair_wts is not None:
        pair_wts = pair_wts[m]

    if pairs_cyc.size == 0:
        raise RuntimeError("No pairs remain after Nyquist guard.")

    # Optional: sort pairs by weight
    order = np.arange(len(pairs_cyc))
    if (allowed_weight_mode != "none") and (pair_wts is not None):
        order = np.argsort(-pair_wts)

    pairs_cyc = pairs_cyc[order]
    pair_ids  = pair_ids[order]
    if pair_wts is not None:
        pair_wts = pair_wts[order]

    # De-duplicate pairs (round inside cyc/µs space)
    key = np.round(pairs_cyc, 6)
    _, uniq_idx = np.unique(key, axis=0, return_index=True)
    uniq_idx = np.sort(uniq_idx)

    pairs_cyc = pairs_cyc[uniq_idx]
    pair_ids  = pair_ids[uniq_idx]
    if pair_wts is not None:
        pair_wts = pair_wts[uniq_idx]

    # Top-K subset used for the PRIOR stage
    if len(pairs_cyc) > prior_pairs_topK:
        pairs_cyc_topK = pairs_cyc[:prior_pairs_topK]
        pair_ids_topK  = pair_ids[:prior_pairs_topK]
        pair_wts_topK  = pair_wts[:prior_pairs_topK] if pair_wts is not None else None
    else:
        pairs_cyc_topK = pairs_cyc
        pair_ids_topK  = pair_ids
        pair_wts_topK  = pair_wts

    # (optional) keep both the full set and top-K set around
    # pairs_cyc, pair_ids, pair_wts hold ALL;
    # pairs_cyc_topK, pair_ids_topK, pair_wts_topK hold PRIOR subset


    # ---------------------------------------------
    # 5) PRIOR (keep catalog pairs; amplitude free)
    # ---------------------------------------------
    all_candidates = []
    if prior_enable and enable_extras and ("osc_f0" in pmap) and pairs_cyc_topK.size > 0:
        # exact pairs from catalog (no singles), keep site IDs
        prior_ovrs = []
        for (f0, f1), site_id in zip(pairs_cyc_topK, pair_ids_topK):
            prior_ovrs.append(
                {
                    "osc_f0": float(f0),
                    "osc_f1": float(f1),
                    "site_id": int(site_id),   # <-- carries catalog index / site id
                }
            )

        AMP_WINDS = _amp_windows_and_seeds()   # we still use its windows, but amp is FREE
        n_amp_windows = len(AMP_WINDS)
        if verbose:
            print(
                f"[PRIOR] starting: {len(prior_ovrs)} override(s) × "
                f"{n_amp_windows} amp-window(s) (amp=free) "
                f"= {len(prior_ovrs)*n_amp_windows} trial(s) in band {lo:.5g}–{hi:.5g} cyc/µs",
                flush=True
            )

        start_best = np.inf
        best_prior = None

        # Optional: lock freqs very tightly (set small eps>0 to hard-freeze)
        eps = 1e-6  # e.g., 1e-6 to pin, 0.0 to leave free

        for ov in prior_ovrs:
            sid = int(ov.get("site_id", -1))  # <-- extract site_id for this prior

            # Build local frequency boxes if locking is requested
            box_local = {}
            if eps > 0.0:
                if "osc_f0" in pmap and "osc_f0" in ov:
                    f0 = float(ov["osc_f0"])
                    box_local["osc_f0"] = (max(0.0, f0 - eps), f0 + eps)
                if "osc_f1" in pmap and "osc_f1" in ov:
                    f1 = float(ov["osc_f1"])
                    box_local["osc_f1"] = (max(0.0, f1 - eps), f1 + eps)

            for (amin, amax), _a_seeds in AMP_WINDS:
                # amplitude is FREE within (amin, amax): set bounds only, no osc_amp in overrides
                base_p0, base_lb, base_ub = _clone_vecs(p0_base, lb_base, ub_base)
                _set_osc_amp_bounds(base_lb, base_ub, fit_fn_base, amin, amax)

                # DO NOT add osc_amp to overrides -> keep it free
                ov2 = dict(ov)
                ov2 = _sanitize_overrides(ov2, _param_index_map(fit_fn_base))

                p0_try, lb_try, ub_try = _apply_param_overrides(
                    base_p0, base_lb, base_ub, fit_fn_base,
                    overrides=ov2, bound_boxes=(box_local or bound_boxes)
                )

                try:
                    popt, pcov, red = _fit_least_squares(
                        fit_fn_base, t, yy, ee, p0_try, lb_try, ub_try,
                        max_nfev=small_max_nfev // 4
                    )
                except Exception:
                    # fallback: core-only
                    try:
                        popt, pcov, red = _fit_least_squares(
                            fit_fn_base, t, yy, ee,
                            p0_try[:kcore], lb_try[:kcore], ub_try[:kcore],
                            max_nfev=small_max_nfev // 4
                        )
                    except Exception:
                        continue

                sc = _score_tuple(popt, red, lb_try, ub_try, _param_index_map(fit_fn_base))
                if sc[0] < start_best:
                    start_best = sc[0]
                    # store sid in best_prior
                    best_prior = ("lsq_prior", (amin, amax), ov2, sid, popt, pcov, red, fit_fn_base)

        best_prior_candidate = None
        if best_prior is not None:
            name, ab, overrides, sid, popt, pcov, red, used_fn = best_prior
            # match layout of all_candidates: (ab, overrides, sid, mode, popt, pcov, red, used_fn)
            best_prior_candidate = (ab, overrides, sid, name, popt, pcov, red, used_fn)
            all_candidates.append(best_prior_candidate)
            if verbose:
                print(f"[PRIOR-BEST] amp={ab}, mode={name}, redχ²={red:.4g}, "
                      f"overrides={overrides}, site_id={sid}")
            if isinstance(early_stop_redchi, (int, float)) and red <= early_stop_redchi:
                if verbose:
                    print(f"[EARLY-STOP] Using prior candidate with redχ²={red:.4g}, site_id={sid}")
                # include sid in the early-return signature
                return popt, pcov, red, used_fn, ab, overrides, sid

    # ---------------------------------------------------
    # 6) Build PAIRED overrides (no mixing) + optional singles
    # ---------------------------------------------------
    # lock_pair_freqs_eps = 1e-6 
    # lock_eps = float(globals().get("lock_pair_freqs_eps", 0.0))
    lock_eps = 1e-6 
    # lock_eps = float(globals().get("lock_pair_freqs_eps", 0.0))
    def _local_boxes_for_pair(ov):
        if lock_eps <= 0.0:
            return None
        return {
            "osc_f0": (max(0.0, ov["osc_f0"] - lock_eps), ov["osc_f0"] + lock_eps),
            "osc_f1": (max(0.0, ov["osc_f1"] - lock_eps), ov["osc_f1"] + lock_eps),
        }

    def _product_with_extras(base_ov, extras_dict):
        if not extras_dict:
            return [dict(base_ov)]
        keys = list(extras_dict.keys())
        vals = [list(v) for v in extras_dict.values()]
        out = []
        def rec(i, cur):
            if i == len(keys):
                out.append(cur.copy()); return
            k = keys[i]
            for x in vals[i]:
                if isinstance(x, (list, tuple, dict, np.ndarray)):
                    raise TypeError(f"extra override '{k}' must be scalar, got {type(x)}")
                cur[k] = float(x)
                rec(i+1, cur)
            cur.pop(k, None)
        rec(0, dict(base_ov))
        return out

    # Final list of (override_dict, local_bound_boxes_or_None)
    # paired_overrides = [{"osc_f0": float(f0), "osc_f1": float(f1)} for (f0, f1) in pairs_cyc]
    paired_overrides = []
    for (f0, f1), site_id in zip(pairs_cyc, pair_ids):
        base_ov = {
            "osc_f0": float(f0),
            "osc_f1": float(f1),
            # no site_id here: this dict is only for fit parameters
        }
        paired_overrides.append((base_ov, int(site_id)))

    override_trials = []
    for (base_ov, sid) in paired_overrides:
        for ov_ex in _product_with_extras(base_ov, extra_overrides_grid):
            ov_fit = _sanitize_overrides(ov_ex, _param_index_map(fit_fn_base))
            override_trials.append((ov_fit, _local_boxes_for_pair(base_ov), sid))

        # ---------------------------------------
    # 7) Full sweep over amp windows & seeds
    #    (coarse→fine survivor screening)
    # ---------------------------------------
    def _run_attempts_with_budget(p0_try, lb_try, ub_try, overrides, maxfev, max_nfev):
        # bounds sanity
        if np.any(~np.isfinite(lb_try)) or np.any(~np.isfinite(ub_try)):
            if verbose: print("  [bounds] non-finite lb/ub; skip")
            return None

        # Ensure strict lb < ub everywhere (SciPy requirement)
        # If any interval is degenerate (lb == ub), widen it slightly around that value.
        eps_bounds = 1e-6
        bad_eq = (ub_try <= lb_try)
        if np.any(bad_eq):
            if verbose:
                idx = np.where(bad_eq)[0].tolist()
                print(f"  [bounds] lb>=ub at idx={idx}; widening by ±{eps_bounds:g}")
            mid = 0.5 * (ub_try[bad_eq] + lb_try[bad_eq])
            lb_try[bad_eq] = mid - eps_bounds
            ub_try[bad_eq] = mid + eps_bounds

        # If anything is still inverted (numeric weirdness), bail
        if np.any(ub_try < lb_try):
            if verbose:
                bad = np.where(ub_try < lb_try)[0].tolist()
                print(f"  [bounds] ub<lb at idx={bad}; skip")
            return None

        # probe model at p0 to catch domain errors (nan/overflow)
        try:
            y0 = fit_fn_base(t, *p0_try)
            if (not np.all(np.isfinite(y0))) or (y0.shape != yy.shape):
                if verbose: print("  [probe] model(p0) non-finite or wrong shape; skip")
                return None
        except Exception as e:
            if verbose: print(f"  [probe] model(p0) raised: {e!r}")
            return None
        
        attempts = []
        def _log(tag):
            if verbose:
                keys = ", ".join([f"{k}:{overrides[k]:.6g}" for k in overrides.keys()]) if overrides else ""
                print(f"  [overrides={{ {keys} }}] {tag}")

        def _record(tag, fn, ok_tuple=None, err=None):
            if ok_tuple is not None:
                popt, pcov, red = ok_tuple
                attempts.append((tag, popt, pcov, red, fn, "ok"))
            else:
                attempts.append((tag, None, None, np.inf, fn, f"fail: {err}"))
                if verbose: print(f"    -> {tag} failed: {err}")

        # least_squares (core)
        try:
            _log("least_squares no-extras (soft_l1)")
            ok = _fit_least_squares(fit_fn_base, t, yy, ee, p0_try[:kcore], lb_try[:kcore], ub_try[:kcore], max_nfev=max_nfev)
            _record("lsq_noextras", fit_fn_base, ok_tuple=ok)
        except Exception as e:
            _record("lsq_noextras", fit_fn_base, err=e)

        # least_squares (+extras)
        if enable_extras:
            try:
                _log("least_squares + extras (soft_l1)")
                ok = _fit_least_squares(fit_fn_base, t, yy, ee, p0_try, lb_try, ub_try, max_nfev=max_nfev)
                _record("lsq_extras", fit_fn_base, ok_tuple=ok)
            except Exception as e:
                _record("lsq_extras", fit_fn_base, err=e)

        # curve_fit (core)
        try:
            _log("curve_fit no-extras")
            ok = _fit_curve_fit(fit_fn_base, t, yy, ee, p0_try[:kcore], lb_try[:kcore], ub_try[:kcore], maxfev=maxfev)
            _record("curve_fit_noextras", fit_fn_base, ok_tuple=ok)
        except Exception as e:
            _record("curve_fit_noextras", fit_fn_base, err=e)

        # curve_fit (+extras)
        if enable_extras:
            try:
                _log("curve_fit + extras")
                ok = _fit_curve_fit(fit_fn_base, t, yy, ee, p0_try, lb_try, ub_try, maxfev=maxfev)
                _record("curve_fit_extras", fit_fn_base, ok_tuple=ok)
            except Exception as e:
                _record("curve_fit_extras", fit_fn_base, err=e)

        # fixed-revival (optional)
        if fit_fn_base is fine_decay:
            try:
                _log("curve_fit fixed-revival")
                p0_C, lb_C, ub_C = _initial_guess_and_bounds(t, yy, enable_extras=False, fixed_rev_time=fixed_rev_time)
                ok = _fit_curve_fit(fine_decay_fixed_revival, t, yy, ee, p0_C, lb_C, ub_C, maxfev=maxfev)
                _record("curve_fit_fixedrev", fine_decay_fixed_revival, ok_tuple=ok)
            except Exception as e:
                _record("curve_fit_fixedrev", fine_decay_fixed_revival, err=e)

        # choose best; if ALL failed, report and return None
        best_here = _pick_best(attempts)
        if best_here is None:
            if verbose:
                why = [a[0]+": "+str(a[5]) for a in attempts]
                print("  [attempts] all failed → no candidate\n    " + "\n    ".join(why))
            return None
        
        mode, popt, pcov, red, used_fn, _note = best_here
        # Recompute χ² consistently on sanitized data
        yfit = used_fn(t, *popt)
        red  = _chi2_red(yy, ee, yfit, len(popt))
        return (mode, popt, pcov, red, used_fn)


    # main sweep
    for ab in amp_bound_grid:
        a_min, a_max = float(ab[0]), float(ab[1])

        base_p0, base_lb, base_ub = _clone_vecs(p0_base, lb_base, ub_base)
        _set_osc_amp_bounds(base_lb, base_ub, fit_fn_base, a_min, a_max)
        base_p0, base_lb, base_ub = _apply_param_overrides(
            base_p0, base_lb, base_ub, fit_fn_base, overrides=None, bound_boxes=bound_boxes,
        )

        # ---- coarse screening over *paired* overrides for this amp window ----
        coarse_pool = []  # (coarse_red, (p0_try, lb_try, ub_try, overrides, site_id))
        for (overrides, local_boxes, sid) in override_trials:
            bboxes = local_boxes or None
            p0_try, lb_try, ub_try = _apply_param_overrides(
                base_p0, base_lb, base_ub, fit_fn_base,
                overrides=overrides, bound_boxes=bboxes,
            )
            cscore = _coarse_redchi(fit_fn_base, t, yy, ee, p0_try, lb_try, ub_try, kcore)
            coarse_pool.append((cscore, (p0_try, lb_try, ub_try, overrides, sid)))

        coarse_pool.sort(key=lambda x: x[0])
        survivors = [x[1] for x in coarse_pool[:max(1, int(coarse_K))]]

        if not survivors:
            if verbose:
                print("[COARSE] no survivors; taking best coarse candidate anyway")
            # pick the lowest coarse score if exists
            if coarse_pool:
                survivors = [sorted(coarse_pool, key=lambda x: (np.inf if not np.isfinite(x[0]) else x[0]))[0][1]]

        # ---- run heavy attempts on survivors with SMALL budgets ----
        for (p0_try, lb_try, ub_try, overrides, sid) in survivors:
            res = _run_attempts_with_budget(
                p0_try, lb_try, ub_try, overrides,
                maxfev=small_maxfev, max_nfev=small_max_nfev
            )
            if res is None:
                continue
            mode, popt, pcov, red, used_fn = res
            all_candidates.append((ab, overrides, sid, mode, popt, pcov, red, used_fn))

            if isinstance(early_stop_redchi, (int, float)) and red <= early_stop_redchi:
                if verbose:
                    print(f"[EARLY-STOP] Sweep reached redχ²={red:.3g}, site_id={sid}")
                return popt, pcov, red, used_fn, ab, overrides, sid

            
    if (len(all_candidates) == 0) and (best_prior_candidate is not None):
        if verbose:
            print("[SWEEP-EMPTY] Falling back to prior winner.")
        ab, ov, sid, mode, popt, pcov, red, used_fn = best_prior_candidate
        return popt, pcov, red, used_fn, ab, ov, sid

    
    # -----------------------------
    # 8) Select best (and refine if needed)
    # -----------------------------
    if not all_candidates:
        raise RuntimeError("All frequency/amp sweep attempts failed.")
    
    # each c: (ab, overrides, sid, mode, popt, pcov, red, used_fn)
    def _pick_best_with_penalty(cands, lb_ref, ub_ref, pmap_ref):
        scored = []
        for c in cands:
            ab, ov, sid, mode, popt, pcov, red, used_fn = c
            sc = _score_tuple(popt, red, lb_ref, ub_ref, pmap_ref)
            scored.append((sc, c))
        scored.sort(key=lambda z: z[0])  # lexicographic: (chi+pen, amp_tie)
        return scored[0][1]


    abest, overrides_best, site_id_best, mode_best, popt_best, pcov_best, red_best, fn_best = \
        _pick_best_with_penalty(all_candidates, lb_base, ub_base, _param_index_map(fit_fn_base))

            
    if verbose:
        print(f"[BEST] amp={abest}, mode={mode_best}, redχ²={red_best:.4g}, "
          f"overrides={overrides_best}, site_id={site_id_best}")

    # --- Early exit if already good enough ---
    if np.isfinite(red_best) and (early_stop_redchi is not None) and (red_best <= early_stop_redchi):
        if verbose:
            print(f"[EARLY-STOP] Keeping prior/sweep best (redχ²={red_best:.4g} ≤ {early_stop_redchi})")
        return popt_best, pcov_best, red_best, fn_best, abest, overrides_best, site_id_best


    if not np.isfinite(red_best):
        return popt_best, pcov_best, red_best, fn_best, abest, overrides_best, site_id_best


    if verbose:
        print("[REFINE] Re-running best candidate with bigger budgets")

    # Rebuild bounds around the winning amp window
    base_p0, base_lb, base_ub = _clone_vecs(p0_base, lb_base, ub_base)
    _set_osc_amp_bounds(base_lb, base_ub, fn_best, float(abest[0]), float(abest[1]))
    base_p0, base_lb, base_ub = _apply_param_overrides(
        base_p0, base_lb, base_ub, fn_best, overrides=None, bound_boxes=bound_boxes,
    )
    p0_try, lb_try, ub_try = _apply_param_overrides(
        base_p0, base_lb, base_ub, fn_best, overrides=overrides_best, bound_boxes=None,
    )

    # Warm start from previous best params
    p0_try = np.array(popt_best, dtype=float)

    # --- SNAP TO ALLOWED LINES (robust) ---
    pidx = _param_index_map(fn_best)

    allowed_set = np.asarray(pairs_cyc, float)  # or np.asarray(ordered, float)
    allowed_set = allowed_set[np.isfinite(allowed_set) & (allowed_set > 0)]
    _tol_kHz = float(allowed_tol_kHz) if 'allowed_tol_kHz' in globals() else 2.0
    
    # --- Freeze to the selected pair (strict) ---
    pidx = _param_index_map(fn_best)

    def _has(name): 
        return (name in pidx) and (pidx[name] < len(p0_try))

    # Use exactly the pair that won during the sweep:
    if _has("osc_f0") and ("osc_f0" in overrides_best):
        p0_try[pidx["osc_f0"]] = float(overrides_best["osc_f0"])
        lb_try[pidx["osc_f0"]] = ub_try[pidx["osc_f0"]] = p0_try[pidx["osc_f0"]]

    use_f1 = (overrides_best.get("osc_f1", 0.0) != 0.0)
    if _has("osc_f1") and use_f1:
        p0_try[pidx["osc_f1"]] = float(overrides_best["osc_f1"])
        lb_try[pidx["osc_f1"]] = ub_try[pidx["osc_f1"]] = p0_try[pidx["osc_f1"]]
    
    # Run big budgets
    res = _run_attempts_with_budget(
        p0_try, lb_try, ub_try, overrides_best,
        maxfev=big_maxfev, max_nfev=big_max_nfev
    )

    if res is not None:
        mode_r, popt_r, pcov_r, red_r, fn_r = res
        if np.isfinite(red_r) and (red_r < red_best):
            if verbose:
                print(f"[REPAIR-BEST] mode={mode_r}, redχ²={red_r:.4g} < {red_best:.4g}, "
                    f"site_id={site_id_best}")
            popt_best, pcov_best, red_best, fn_best = popt_r, pcov_r, red_r, fn_r

    # --- Bound-hit repair pass ------------------------------------------------
    pmap_win = _param_index_map(fn_best)
    hits = _bound_hits(popt_best, lb_try, ub_try, frac_tol=0.01, abs_tol=1e-10)

    # If osc_amp is pressed against window edge, escalate window once (e.g. (-1,1)→(-3,3))
    if "osc_amp" in pmap_win:
        iA = pmap_win["osc_amp"]
        if iA in hits:
            a0, a1 = float(abest[0]), float(abest[1])
            if (a1 - a0) <= 2.01:
                if verbose:
                    print("[REPAIR] Escalating osc_amp window to (-2,2)")
                abest = (-2.0, 2.0)

    if hits:
        # Build set of indices we must NOT touch (frequencies frozen by overrides_best)
        frozen_idxs = set()
        if "osc_f0" in pmap_win and ("osc_f0" in overrides_best):
            frozen_idxs.add(pmap_win["osc_f0"])
        use_f1 = (overrides_best.get("osc_f1", 0.0) != 0.0)
        if use_f1 and ("osc_f1" in pmap_win) and ("osc_f1" in overrides_best):
            frozen_idxs.add(pmap_win["osc_f1"])

        # Remove frozen indices from hits so they won't be widened or reseeded
        hits = {k: v for k, v in hits.items() if k not in frozen_idxs}

        if hits:
            if verbose:
                inv = {j: i for i, j in pmap_win.items()}
                hit_names = [inv.get(i, f"param[{i}]") for i in hits.keys()]
                print(f"[REPAIR] Params at bounds (non-frozen): {hit_names} → widening boxes & reseeding")

            # 1) widen only those hit params (use the correct map and freq band)
            lb_rep, ub_rep = _repair_bounds_for_hits(lb_try, ub_try, hits, pmap_win, freq_band=band)

            # 2) keep frequencies frozen in the repaired boxes
            if "osc_f0" in pmap_win and (pmap_win["osc_f0"] in frozen_idxs):
                i0 = pmap_win["osc_f0"]; f0 = float(popt_best[i0])
                lb_rep[i0] = ub_rep[i0] = f0
            if use_f1 and "osc_f1" in pmap_win and (pmap_win["osc_f1"] in frozen_idxs):
                i1 = pmap_win["osc_f1"]; f1 = float(popt_best[i1])
                lb_rep[i1] = ub_rep[i1] = f1

            # 3) re-tie contrast to baseline
            p0_rep = np.array(popt_best, float)
            p0_rep, lb_rep, ub_rep = _retie_contrast_to_baseline(p0_rep, lb_rep, ub_rep, pmap_win, eps=0.01)

            # 4) reseed hit indices at centers (exclude frozen)
            reseed_idxs = [i for i in hits.keys() if i not in frozen_idxs]
            p0_rep = _reseed_to_center(p0_rep, lb_rep, ub_rep, only_idxs=reseed_idxs)

            # 5) run heavy attempt
            res_rep = _run_attempts_with_budget(
                p0_rep, lb_rep, ub_rep, overrides_best,
                maxfev=big_maxfev, max_nfev=big_max_nfev
            )
            # if res_rep is not None:
            #     mode_r, popt_r, pcov_r, red_r, fn_r = res_rep
            #     if np.isfinite(red_r) and (red_r < red_best):
            #         if verbose:
            #             print(f"[REPAIR-BEST] mode={mode_r}, redχ²={red_r:.4g} < {red_best:.4g}")
            #         popt_best, pcov_best, red_best, fn_best = popt_r, pcov_r, red_r, fn_r
            #     else:
            #         if verbose:
            #             print(f"[REPAIR] did not improve (keeping redχ²={red_best:.4g})")
            if res_rep  is not None:
                mode_r, popt_r, pcov_r, red_r, fn_r = res_rep
                if np.isfinite(red_r) and (red_r < red_best):
                    if verbose:
                        print(f"[REPAIR-BEST] mode={mode_r}, redχ²={red_r:.4g} < {red_best:.4g}, "
                            f"site_id={site_id_best}")
                    popt_best, pcov_best, red_best, fn_best = popt_r, pcov_r, red_r, fn_r
                else:
                    if verbose:
                        print(f"[REPAIR] did not improve (keeping redχ²={red_best:.4g}, "
                            f"site_id={site_id_best})")


    return popt_best, pcov_best, red_best, fn_best, abest, overrides_best, site_id_best



def run_with_amp_and_freq_sweeps(
    nv_list,
    norm_counts,
    norm_counts_ste,
    times_us,
    nv_inds=None,
    *,
    # -------- amplitude sweep --------
    amp_bound_grid=((-0.5, 0.5), (-1, 1), (-2, 2)),
    
    # -------- frequency handling --------
    freq_bound_boxes=None,              # {"osc_f0":(lo,hi),"osc_f1":(lo,hi)} in cyc/µs
    freq_seed_band=None,                # (fmin,fmax) cyc/µs
    freq_seed_n_peaks=3,
    seed_include_harmonics=True,

    # -------- extra multi-start overrides --------
    extra_overrides_grid=None,          # dict of param->[values]

    # -------- model toggles --------
    use_fixed_revival=False,
    enable_extras=True,
    fixed_rev_time=37.6,

    # -------- prior / early-stop --------
    prior_enable=True,
    prior_pairs_topK=24,
    prior_min_sep=0.01,
    early_stop_redchi=None,

    # -------- optimizer budgets --------
    small_maxfev=40_000,
    small_max_nfev=60_000,
    big_maxfev=120_000,
    big_max_nfev=180_000,
    refine_target_red=1.05,

    # -------- coarse screening --------
    coarse_K=8,
    coarse_max_nfev=200_000,

    # -------- data cleaning --------
    err_floor=1e-3,

    # -------- allowed-lines (catalog) controls --------
    allowed_records=None,               # list[dict] from catalog (all orientations)
    allowed_orientations=None,          # global fallback (e.g. [(1,1,1)])
    allowed_tol_kHz=8.0,                # ± window for snapping/bounds
    allowed_weight_mode="none",         # "none"|"kappa"|"per_line"
    p_occ=0.011,

    # -------- NEW: per-NV orientations --------
    nv_orientations=None,               # shape (N_NV, 3), ints ±1

    # -------- logging --------
    verbose=True,
):
    """
    Run single-NV fits across a set of NV indices with amplitude-bound sweeps,
    frequency constraints, and optional physics-guided prior seeds.

    If nv_orientations is provided, for each NV we restrict the catalog to
    that NV's orientation only when generating allowed lines.
    """
    # Normalize NV indices
    if nv_inds is None:
        nv_inds = list(range(len(nv_list)))

    # Basic array guards
    times_us = np.asarray(times_us, float).ravel()
    norm_counts = np.asarray(norm_counts, float)
    norm_counts_ste = np.asarray(norm_counts_ste, float)

    if norm_counts.shape != norm_counts_ste.shape:
        raise ValueError("norm_counts and norm_counts_ste must have the same shape.")
    if norm_counts.shape[0] < max(nv_inds) + 1:
        raise ValueError("nv_inds references an index beyond norm_counts rows.")

    # Optional sanity check on nv_orientations
    if nv_orientations is not None:
        nv_orientations = np.asarray(nv_orientations, int)
        if nv_orientations.shape[0] < max(nv_inds) + 1 or nv_orientations.shape[1] != 3:
            raise ValueError(
                f"nv_orientations must have shape (N_NV, 3); got {nv_orientations.shape}"
            )

    popts, pcovs, chis, fit_fns = [], [], [], []
    chosen_amp_bounds, chosen_overrides = [], []
    best_site_ids = []

    for idx, lbl in enumerate(nv_inds, 1):
        if verbose:
            print(f"Fitting (amp+freq sweeps) for NV {lbl}  [{idx}/{len(nv_inds)}]")

        # Pull/shape this NV's trace
        y  = np.asarray(norm_counts[lbl], float).ravel()
        ye = np.asarray(norm_counts_ste[lbl], float).ravel()

        if (y.size != times_us.size) or (ye.size != times_us.size):
            print(f"[WARN] NV {lbl}: length mismatch (t={times_us.size}, y={y.size}, e={ye.size}); skipping.")
            popts.append(None); pcovs.append(None); chis.append(np.nan); fit_fns.append(None)
            chosen_amp_bounds.append(None); chosen_overrides.append({})
            continue

        # Gentle error floor
        ye = np.maximum(ye, err_floor)

        # ---------- per-NV orientation filter for catalog ----------
        if nv_orientations is not None:
            ori_vec = nv_orientations[lbl]          # e.g. [-1, 1, 1]
            ori_tuple = tuple(int(x) for x in ori_vec)
            local_allowed_orientations = [ori_tuple]
        else:
            # fallback to whatever was passed globally
            local_allowed_orientations = allowed_orientations

        try:
            pi, cov, chi, fn, ab, ov, sid = fit_one_nv_with_freq_sweeps(
                times_us, y, ye,
                amp_bound_grid=amp_bound_grid,

                # Frequency handling
                freq_bound_boxes=freq_bound_boxes,
                freq_seed_band=freq_seed_band,
                freq_seed_n_peaks=freq_seed_n_peaks,
                seed_include_harmonics=seed_include_harmonics,

                # Extra multi-starts
                extra_overrides_grid=extra_overrides_grid,

                # Model choice
                use_fixed_revival=use_fixed_revival,
                enable_extras=enable_extras,
                fixed_rev_time=fixed_rev_time,

                # Prior / early-stop
                prior_enable=prior_enable,
                prior_pairs_topK=prior_pairs_topK,
                prior_min_sep=prior_min_sep,
                early_stop_redchi=early_stop_redchi,

                # Optimizer budgets
                small_maxfev=small_maxfev,
                small_max_nfev=small_max_nfev,
                big_maxfev=big_maxfev,
                big_max_nfev=big_max_nfev,
                refine_target_red=refine_target_red,

                # Coarse screening
                coarse_K=coarse_K,
                coarse_max_nfev=coarse_max_nfev,

                # Data cleaning
                err_floor=err_floor,

                # Allowed-lines (catalog)
                allowed_records=allowed_records,                # full catalog
                allowed_orientations=local_allowed_orientations, # per-NV orientation
                allowed_tol_kHz=allowed_tol_kHz,
                allowed_weight_mode=allowed_weight_mode,
                p_occ=p_occ,

                # Logging
                verbose=verbose,
            )

        except Exception as e:
            if verbose:
                print(f"[WARN] Fit failed for NV {lbl}: {e}")
            pi = cov = fn = None
            chi = np.nan
            ab = None
            sid = -1
            ov = {}

        popts.append(pi); pcovs.append(cov); chis.append(chi); fit_fns.append(fn)
        chosen_amp_bounds.append(ab); chosen_overrides.append(ov)
        best_site_ids.append(sid)

        if verbose:
            if np.isfinite(chi):
                print(f"[DONE] NV {lbl}: redχ²={chi:.4g}, amp_bounds={ab}, overrides={ov}, site_id={sid}")
            else:
                print(f"[DONE] NV {lbl}: redχ²=nan (failed), site_id={sid}")

    return popts, pcovs, chis, fit_fns, nv_inds, chosen_amp_bounds, chosen_overrides,  best_site_ids
