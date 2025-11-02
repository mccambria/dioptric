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
# def fine_decay(
#     tau,
#     baseline,
#     comb_contrast,
#     revival_time,
#     width0_us,
#     T2_ms,
#     T2_exp,
#     amp_taper_alpha=None,
#     width_slope=None,
#     revival_chirp=None,
#     osc_contrast=None,
#     osc_f0=None,
#     osc_f1=None,
#     osc_phi0=None,
#     osc_phi1=None,
# ):
#     """
#     signal(τ) = baseline - envelope(τ) * MOD(τ) * COMB(τ)

#     envelope(τ) = exp[-((τ / (1000*T2_ms)) ** T2_exp)]

#     COMB(τ) = sum_k  [ 1/(1+k)^amp_taper_alpha ] * exp(-((τ - μ_k)/w_k)^4)
#         μ_k = k * revival_time * (1 + k*revival_chirp)
#         w_k = width0_us * (1 + k*width_slope)

#     MOD(τ) = comb_contrast - osc_contrast * sin^2(π f0 τ + φ0) * sin^2(π f1 τ + φ1)
#     """
#     # defaults
#     if amp_taper_alpha is None: amp_taper_alpha = 0.0
#     if width_slope     is None: width_slope     = 0.0
#     if revival_chirp   is None: revival_chirp   = 0.0
#     if osc_contrast    is None: osc_contrast    = 0.0
#     if osc_f0          is None: osc_f0          = 0.0
#     if osc_f1          is None: osc_f1          = 0.0
#     if osc_phi0        is None: osc_phi0        = 0.0
#     if osc_phi1        is None: osc_phi1        = 0.0

#     tau = np.asarray(tau, dtype=float).ravel()
#     width0_us    = max(1e-9, float(width0_us))
#     revival_time = max(1e-9, float(revival_time))
#     T2_us        = max(1e-9, 1000.0 * float(T2_ms))
#     T2_exp       = float(T2_exp)

#     # envelope
#     envelope = np.exp(-((tau / T2_us) ** T2_exp))

#     # number of revivals to include
#     tau_max = float(np.nanmax(tau)) if tau.size else 0.0
#     n_guess = max(1, min(64, int(np.ceil(1.2 * tau_max / revival_time)) + 1))

#     comb = _comb_quartic_powerlaw(
#         tau,
#         revival_time,
#         width0_us,
#         amp_taper_alpha,
#         width_slope,
#         revival_chirp,
#         n_guess
#     )

#     # beating lives in MOD; comb_contrast is the overall amplitude (once)
#     if (osc_contrast != 0.0) and (osc_f0 != 0.0 or osc_f1 != 0.0):
#         s0 = np.sin(np.pi * osc_f0 * tau + osc_phi0)
#         s1 = np.sin(np.pi * osc_f1 * tau + osc_phi1)
#         beat = (s0 * s0) * (s1 * s1)
#         mod = comb_contrast - osc_contrast * beat
#     else:
#         mod = comb_contrast

#     return baseline - envelope * mod * comb
def fine_decay(
    tau,
    baseline,
    comb_contrast,
    revival_time,
    width0_us,
    T2_ms,
    T2_exp,
    amp_taper_alpha=None,
    width_slope=None,
    revival_chirp=None,
    # --- NEW: signed oscillation amplitude (additive), plus 2 freqs optional
    osc_amp=None,            # signed, can be ±
    osc_f0=None,
    osc_phi0=None,
    osc_f1=None,
    osc_phi1=None,
):
    """
    signal(τ) = baseline
                - envelope(τ) * comb_contrast * COMB(τ)         [revival dip]
                + envelope(τ) * COMB(τ) * OSC(τ)                [signed oscillation]
    where:  envelope = exp[-(τ/T2)^n],  COMB = sum_k ... quartic lobes
            OSC(τ) = osc_amp * [cos(2π f0 τ + φ0) + cos(2π f1 τ + φ1)]   (missing terms ignored)
    """
    # defaults
    if amp_taper_alpha is None: amp_taper_alpha = 0.0
    if width_slope     is None: width_slope     = 0.0
    if revival_chirp   is None: revival_chirp   = 0.0
    if osc_amp         is None: osc_amp         = 0.0
    if osc_f0          is None: osc_f0          = 0.0
    if osc_phi0        is None: osc_phi0        = 0.0
    if osc_f1          is None: osc_f1          = 0.0
    if osc_phi1        is None: osc_phi1        = 0.0

    tau = np.asarray(tau, float).ravel()
    width0_us    = max(1e-9, float(width0_us))
    revival_time = max(1e-9, float(revival_time))
    T2_us        = max(1e-9, 1000.0*float(T2_ms))
    T2_exp       = float(T2_exp)

    envelope = np.exp(-((tau / T2_us) ** T2_exp))

    # how many revivals
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

    carrier = envelope * comb  # use same spatial/temporal gating

    # revival dip (always subtractive)
    dip = comb_contrast * carrier

    # signed oscillation term (zero-mean cosines)
    osc = 0.0
    if osc_amp != 0.0:
        if osc_f0 != 0.0:
            osc += np.cos(2*np.pi*osc_f0 * tau + osc_phi0)
        if osc_f1 != 0.0:
            osc += np.cos(2*np.pi*osc_f1 * tau + osc_phi1)
        # if neither f is set, osc stays 0

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
    osc_contrast=None,
    osc_f0=None,
    osc_f1=None,
    osc_phi0=None,
    osc_phi1=None,
    _fixed_rev_time_us=50.0
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
        osc_contrast,
        osc_f0,
        osc_f1,
        osc_phi0,
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
# def _initial_guess_and_bounds(times_us, y, enable_extras=True, fixed_rev_time=None):
#     """
#     Build p0 and bounds for fine_decay*.
#     p = [baseline, comb_contrast, revival_time, width0_us, T2_ms, T2_exp, ... extras ...]
#     or if fixed revival: [baseline, comb_contrast, width0_us, T2_ms, T2_exp, ...]
#     """
#     # crude baseline from early points; robust min for contrast scale
#     idx_b = min(7, len(y)-1) if len(y) else 0
#     baseline_guess = float(y[idx_b]) if y.size else 0.5
#     quart_min = float(np.nanmin(y)) if y.size else baseline_guess - 0.2
#     comb_contrast_guess = max(1e-3, baseline_guess - quart_min)

#     # Envelope rough T2 using a late point:
#     j = max(0, len(times_us)-7)
#     # avoid invalids
#     ratio = (baseline_guess - y[j]) / max(1e-9, comb_contrast_guess)
#     ratio = min(max(ratio, 1e-9), 0.999999)
#     # exp(-(t/T2)^exp) ~ ratio -> -(t/T2)^3 ~ ln(ratio) -> T2 ~ t / (-ln(ratio))^(1/3)
#     T2_exp_guess = 3.0
#     tlate = max(1e-3, float(times_us[j]))
#     T2_ms_guess = 0.1 * (1.0 / max(1e-9, (-np.log(ratio)) ** (1.0 / T2_exp_guess)))

#     # width ~ a few µs, revival ~ ~50 µs typical in your code
#     width0_guess = 6.0
#     revival_guess = 39.2 if fixed_rev_time is None else fixed_rev_time

#     # Base params and bounds:
#     if fixed_rev_time is None:
#         p0  = [baseline_guess, comb_contrast_guess, revival_guess, width0_guess, T2_ms_guess, T2_exp_guess]
#         lb  = [0.0,           0.0,                30.0,          0.5,          0.0,        0.5]
#         ub  = [1.0,           1.0,                50.0,          20.0,         2000.0,     10.0]
#     else:
#         p0  = [baseline_guess, comb_contrast_guess, width0_guess, T2_ms_guess, T2_exp_guess]
#         lb  = [0.0,           0.0,                0.5,          0.0,          0.5]
#         ub  = [1.0,           1.0,                20.0,         2000.0,       10.0]

#     # add extras?
#     if enable_extras:
#         # [amp_taper_alpha, width_slope, revival_chirp, osc_contrast, osc_f0, osc_f1, osc_phi0, osc_phi1]
#         # extra_p0  = [0.3, 0.02, 0.0,  0.15, 0.30, 0.10, 0.0, 0.0]
#         # extra_lb  = [0.0, 0.00, -0.01, -0.5, 0.00, 0.00, -np.pi, -np.pi]
#         # extra_ub  = [2.0, 0.20,  0.01,  0.5, 5.00, 1.00,  np.pi,  np.pi]
        
#         extra_p0  = [0.3,   0.02,   0.0,   0.05,  0.02,  0.00,  0.0,   0.0]
#         extra_lb  = [0.0,   0.00,  -0.01, -0.60,  0.001, 0.00, -np.pi, -np.pi]
#         extra_ub  = [2.0,   0.20,   0.01,  0.60,  0.20,  0.20,  np.pi,  np.pi]
#         # indices:   α     width_s  chirp  Aosc   f0     f1     φ0      φ1
#         p0.extend(extra_p0)
#         lb.extend(extra_lb)
#         ub.extend(extra_ub)

#     return np.array(p0, float), np.array(lb, float), np.array(ub, float)

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

def _fft_peak_freq(times_us, resid, fmin, fmax):
    """Find dominant frequency (cycles/μs) in [fmin, fmax] from FFT of residual."""
    t, r = _uniformize(times_us, resid)
    if t.size < 16:
        return None
    # detrend & window
    r = r - np.nanmedian(r)
    r = np.nan_to_num(r, nan=0.0)
    r = r * np.hanning(len(r))
    dt = np.diff(t).mean()
    freqs = np.fft.rfftfreq(len(r), d=dt)  # cycles per μs if t is μs
    Y = np.fft.rfft(r)
    band = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(band):
        return None
    k = np.argmax(np.abs(Y[band]))
    f_hat = float(freqs[band][k])
    return f_hat if np.isfinite(f_hat) and f_hat > 0 else None

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
        lb = [0.0,  0.00, 30.0, 1.0,  0.005, 0.6]
        ub = [1.05, 0.95, 60.0, 20.0, 2000.0, 4.0]
    else:
        p0 = [baseline_guess, comb_contrast_guess, width0_guess, T2_ms_guess, T2_exp_guess]
        lb = [0.0,  0.00, 1.0,  0.005, 0.6]
        ub = [1.05, 0.95, 20.0, 2000.0, 4.0]

    # tighten comb_contrast upper bound so min >= 0:
    # comb_contrast <= baseline - eps
    eps_min = 0.01
    if fixed_rev_time is None:
        ub[1] = min(ub[1], max(0.05, p0[0] - eps_min))
    else:
        ub[1] = min(ub[1], max(0.05, p0[0] - eps_min))

    if not enable_extras:
        return np.array(p0, float), np.array(lb, float), np.array(ub, float)

    # ------- oscillation seeds from data (residual FFT) -------
    # quick residual against a smooth proxy (use running median or simple poly)
    # here: subtract high percentile baseline as crude detrend
    # resid = y - baseline_guess

    # sampling-derived frequency bounds
    # if times_us.size >= 2:
    #     dt_min = float(np.diff(np.unique(times_us)).min())
    #     fnyq = 0.5 / max(dt_min, 1e-6)     # cycles/μs
    #     fmax_fast = min(0.9 * fnyq, 10.0)  # cap very high just in case
    #     tspan = float(times_us.max() - times_us.min())
    #     fmin_resolvable = max(1.0 / max(2.0 * tspan, 1e-6), 0.001)
    # else:
    #     fmax_fast, fmin_resolvable = 1.0, 0.001

    # slow band & fast band
    # f0_lo, f0_hi = max(0.002, fmin_resolvable), 0.05
    # f1_lo, f1_hi = max(0.05, 2.0 / max(tspan, 1e-6)), max(0.06, fmax_fast)

    # f0_hat = _fft_peak_freq(times_us, resid, f0_lo, f0_hi)
    # f1_hat = _fft_peak_freq(times_us, resid, f1_lo, f1_hi)

    # osc_amp_seed = min(0.15, 0.5 * comb_contrast_guess)
    # osc_amp_lb   = -min(0.9, comb_contrast_guess)
    # osc_amp_ub   =  min(0.9, comb_contrast_guess)
    # ------- extras: [amp_taper_alpha, width_slope, revival_chirp, osc_contrast, osc_f0, osc_f1, osc_phi0, osc_phi1]
    # extra_p0 = [0.3, 0.02, 0.0,  0.10,  0.02,  0.10,  0.0,  0.0]
    # extra_lb = [0.0, 0.00, -0.01, 0.00,  f0_lo, f1_lo, -np.pi, -np.pi]
    # extra_ub = [2.0, 0.20,  0.01, min(0.9, p0[1]*1.2),  f0_hi, f1_hi,  np.pi,  np.pi]
    
    extra_p0  = [0.3, 0.02, 0.0,  0.15, 0.50, 0.10, 0.0, 0.0]
    extra_lb  = [0.0, 0.00, -0.01, -1.0, 0.00, 0.00, -np.pi, -np.pi]
    extra_ub  = [2.0, 0.20,  0.01,  1.0, 5.00, 5.00,  np.pi,  np.pi]
        
    # [amp_taper_alpha, width_slope, revival_chirp, osc_amp, osc_f0, osc_f1, osc_phi0, osc_phi1]
    # extra_p0 = [0.3, 0.02, 0.0,  osc_amp_seed,  0.10,  0.1,  0.0, 0.0]
    # extra_lb = [0.0, 0.00,-0.01,  osc_amp_lb,    f0_lo,    f1_lo,   -np.pi,-np.pi]
    # extra_ub = [2.0, 0.20, 0.01,  osc_amp_ub,    f0_hi,    f1_hi,    np.pi, np.pi]
    # seed slow/fast from FFT if found
    # if f0_hat is not None:
    #     extra_p0[4] = np.clip(f0_hat, extra_lb[4], extra_ub[4])
    # if f1_hat is not None:
    #     extra_p0[5] = np.clip(f1_hat, extra_lb[5], extra_ub[5])

    # start oscillation contrast small; allow up to ~1.2× comb_contrast but <= 0.9
    # extra_p0[3] = min(0.15, p0[1]*0.5)

    p0.extend(extra_p0); lb.extend(extra_lb); ub.extend(extra_ub)
    return np.array(p0, float), np.array(lb, float), np.array(ub, float)


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

def _fit_least_squares(fit_fn, times_us, y, yerr, p0, lb, ub, max_nfev):
    """Robust fallback using soft-L1 loss; returns popt, pseudo-pcov=None, red."""
    def resid(p):
        return (y - fit_fn(times_us, *p)) / yerr
    res = least_squares(
        resid, x0=p0, bounds=(lb, ub),
        loss="soft_l1", f_scale=1.0,
        max_nfev=max_nfev, ftol=1e-8, xtol=1e-8, gtol=1e-8,
    )
    popt = res.x
    yfit = fit_fn(times_us, *popt)
    red  = _chi2_red(y, yerr, yfit, len(popt))
    return popt, None, red

def _pick_best(cands):
    """cands is list of tuples (name, popt, pcov, red, fit_fn, note)"""
    good = [(i,c) for i,c in enumerate(cands) if np.isfinite(c[3])]
    if not good:
        return None
    # pick smallest reduced chi^2
    i_best = min(good, key=lambda ic: ic[1][3])[0]
    return cands[i_best]

# ---------- replace your fit_one_nv with this ----------
def fit_one_nv(times_us, y, yerr,
               use_fixed_revival=False,
               enable_extras=True,
               fixed_rev_time=39.2,
               maxfev=200000, 
               max_nfev=200000,
               verbose=True):

    # clean data, enforce error floor
    t, yy, ee = _sanitize_trace(times_us, y, yerr, err_floor=1e-3)
    if len(t) < 8:
        raise RuntimeError("Too few valid points after sanitization")

    # choose model + initial guesses / bounds
    if use_fixed_revival:
        fit_fn = fine_decay_fixed_revival
        p0, lb, ub = _initial_guess_and_bounds(t, yy, enable_extras, fixed_rev_time=fixed_rev_time)
    else:
        fit_fn = fine_decay
        p0, lb, ub = _initial_guess_and_bounds(t, yy, enable_extras, fixed_rev_time=None)

    attempts = []

    def log_try(name, fn, *_):
        if verbose:
            print(f"  trying: {name}")

    # --- Strategy A: full model with extras, curve_fit (fast path)
    try:
        log_try("curve_fit + extras", fit_fn)
        popt, pcov, red = _fit_curve_fit(fit_fn, t, yy, ee, p0, lb, ub, maxfev=maxfev)
        attempts.append(("curve_fit_extras", popt, pcov, red, fit_fn, "ok"))
    except Exception as e:
        attempts.append(("curve_fit_extras", None, None, np.inf, fit_fn, f"fail: {e}"))

    # --- Strategy B: disable extras (fewer params), curve_fit
    if enable_extras:
        n_core = 6 if (fit_fn is fine_decay) else 5
        p0_B  = p0[:n_core]; lb_B = lb[:n_core]; ub_B = ub[:n_core]
        try:
            log_try("curve_fit no-extras", fit_fn)
            popt, pcov, red = _fit_curve_fit(fit_fn, t, yy, ee, p0_B, lb_B, ub_B, maxfev=maxfev)
            attempts.append(("curve_fit_noextras", popt, pcov, red, fit_fn, "ok"))
        except Exception as e:
            attempts.append(("curve_fit_noextras", None, None, np.inf, fit_fn, f"fail: {e}"))

    # --- Strategy C: force fixed revival if not already
    if not use_fixed_revival:
        fit_fn_C = fine_decay_fixed_revival
        p0_C, lb_C, ub_C = _initial_guess_and_bounds(t, yy, enable_extras=False, fixed_rev_time=fixed_rev_time)
        try:
            log_try("curve_fit fixed-revival", fit_fn_C)
            popt, pcov, red = _fit_curve_fit(fit_fn_C, t, yy, ee, p0_C, lb_C, ub_C, maxfev=maxfev)
            attempts.append(("curve_fit_fixedrev", popt, pcov, red, fit_fn_C, "ok"))
        except Exception as e:
            attempts.append(("curve_fit_fixedrev", None, None, np.inf, fit_fn_C, f"fail: {e}"))

    # --- Strategy D: robust least_squares soft-L1, with extras if present
    try:
        log_try("least_squares + extras (soft_l1)", fit_fn)
        popt, pcov, red = _fit_least_squares(fit_fn, t, yy, ee, p0, lb, ub, max_nfev=max_nfev)
        attempts.append(("lsq_extras", popt, pcov, red, fit_fn, "ok"))
    except Exception as e:
        attempts.append(("lsq_extras", None, None, np.inf, fit_fn, f"fail: {e}"))

    # --- Strategy E: robust least_squares without extras
    if enable_extras:
        n_core = 6 if (fit_fn is fine_decay) else 5
        p0_E  = p0[:n_core]; lb_E = lb[:n_core]; ub_E = ub[:n_core]
        try:
            log_try("least_squares no-extras (soft_l1)", fit_fn)
            popt, pcov, red = _fit_least_squares(fit_fn, t, yy, ee, p0_E, lb_E, ub_E, max_nfev=max_nfev)
            attempts.append(("lsq_noextras", popt, pcov, red, fit_fn, "ok"))
        except Exception as e:
            attempts.append(("lsq_noextras", None, None, np.inf, fit_fn, f"fail: {e}"))

    best = _pick_best(attempts)
    if best is None:
        # bubble up a readable message with last failure note
        last_note = attempts[-1][5] if attempts else "no attempts"
        raise RuntimeError(f"All fit strategies failed; last={last_note}")

    name, popt, pcov, red, used_fn, note = best
    # recompute χ² with original arrays (times_us, y, yerr) for consistency
    yfit = used_fn(times_us, *popt)
    red  = _chi2_red(y, yerr, yfit, len(popt))
    return popt, pcov, red, used_fn

# =============================================================================
# Orchestration
# =============================================================================
def run(nv_list, norm_counts, norm_counts_ste, times_us,
        nv_inds=None, use_fixed_revival=False, enable_extras=True, fixed_rev_time=39.2):
    if nv_inds is None:
        nv_inds = list(range(len(nv_list)))  # labels you want to fit/plot

    popts, pcovs, chis, fit_fns = [], [], [], []
    for lbl in nv_inds:  # iterate in the SAME label order
        print(f"Fitting for NV {lbl}")
        try:
            pi, cov, chi, fn = fit_one_nv(
                times_us, norm_counts[lbl], norm_counts_ste[lbl],
                use_fixed_revival=use_fixed_revival,
                enable_extras=enable_extras,
                fixed_rev_time=fixed_rev_time,
            )
        except Exception as e:
            print(f"[WARN] Fit failed for NV {lbl}: {e}")
            pi=cov=fn=None; chi=np.nan
        popts.append(pi); pcovs.append(cov); chis.append(chi); fit_fns.append(fn)

    # VERY IMPORTANT: return the label list (same length/order as popts)
    return popts, pcovs, chis, fit_fns, list(nv_inds)

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

def plot_individual_fits(
    norm_counts, 
    norm_counts_ste,
    total_evolution_times,
    popts,
    nv_inds,              # <— SAME labels list you passed to run and got back
    fit_fn_per_nv,        # <— from run; guarantees matching signature
    keep_mask=None,
    show_residuals=True,
    n_fit_points=1000,
    save_prefix=None,
    dpi=220,
    block=False
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

        fit_fn = fit_fn_per_nv[pos]
        if fit_fn is None:
            # last resort: try fine_decay with coercion
            fit_fn = fine_decay

        y = np.asarray(norm_counts[lbl], float)
        e = np.asarray(norm_counts_ste[lbl], float)

        if show_residuals:
            fig, (ax, axr) = plt.subplots(2, 1, figsize=(7, 6), sharex=True,
                                          gridspec_kw=dict(height_ratios=[3, 1], hspace=0.05))
        else:
            fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))

        ax.errorbar(t_all, y, yerr=e, fmt="o", ms=3.5, lw=0.8, alpha=0.85, capsize=2)
        ax.set_ylabel("Normalized NV$^{-}$ population")

        t_fit = np.linspace(np.nanmin(t_all), np.nanmax(t_all), n_fit_points)
        y_fit = _safe_call_fit_fn(fit_fn, t_fit, p)
        ax.plot(t_fit, y_fit, "-", lw=2)

        ax.set_title(f"NV {lbl}")
        ax.set_ylim(min(np.nanmin(y)-0.1, -0.1), max(np.nanmax(y)+0.1, 1.2))
        ax.grid(True, alpha=0.25)

        if show_residuals:
            y_model = _safe_call_fit_fn(fit_fn, t_all, p)
            res = y - y_model
            axr.axhline(0.0, ls="--", lw=1.0)
            axr.plot(t_all, res, ".", ms=3.5)
            axr.set_xlabel("Total evolution time (µs)")
            axr.set_ylabel("res.")
            axr.grid(True, alpha=0.25)
        else:
            ax.set_xlabel("Total evolution time (µs)")

        if save_prefix:
            file_path = dm.get_file_path(__file__, timestamp, f"{save_prefix}-nv{int(lbl):03d}")
            dm.save_figure(fig, file_path,f"nv{int(lbl):03d}")
        figs.append((lbl, fig))

    if figs:
        plt.show(block=block)
    return figs

# =============================================================================
# CLI / Example
# =============================================================================

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
    
    data = widefield.process_multiple_files(file_stems, load_npz=True)
    nv_list = data["nv_list"]
    taus = data["taus"]
    total_evolution_times = 2 * np.array(taus) / 1e3  
    counts = np.array(data["counts"])
    sig = counts[0]
    ref = counts[1]
    norm_counts, norm_counts_ste = widefield.process_counts(nv_list, sig, ref, threshold=True)
    # Example NV filtering
    # split_esr = [12, 13, 14, 61, 116]
    # broad_esr = [52, 11]
    # weak_esr  = [72, 64, 55, 96, 112, 87, 12, 58, 36]
    # skip_inds = list(set(split_esr + broad_esr + weak_esr))
    # skip_inds = []
    # nv_inds = [ind for ind in range(len(data["nv_list"])) if ind not in skip_inds]
    # nv_inds = [2, 10, 22, 25, 55, 112]
    # nv_inds = [22]
    # --- Run and plot ---------------------------------------------------------
    ts = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    repr_nv_sig = widefield.get_repr_nv_sig(data["nv_list"])
    save_prefix = dm.get_file_path(__file__, ts, f"{repr_nv_sig.name}-finefit")

    # Toggle these as you wish:
    USE_FIXED_REVIVAL = False       # True -> uses fine_decay_fixed_revival
    ENABLE_EXTRAS     = True        # enable alpha/width_slope/chirp + beating + phases
    DEFAULT_REV_US = 39.2
    # 1) FIT
    popts, pcovs, chis, fit_fns, fit_nv_labels = run(
        nv_list, norm_counts, norm_counts_ste, total_evolution_times,
        nv_inds=None,
        use_fixed_revival=False, enable_extras=True, fixed_rev_time=38.2
    )
    timestamp = dm.get_time_stamp()
    fit_dict = {
        "timestamp": timestamp,
        "dataset_ids": file_stems,
        "default_rev_us": float(DEFAULT_REV_US),
        "run_flags": {
            "use_fixed_revival": bool(USE_FIXED_REVIVAL),
            "enable_extras": bool(ENABLE_EXTRAS),
        },
        "nv_labels": list(map(int, fit_nv_labels)),
        "times_us": np.asarray(total_evolution_times, float).tolist(),
        "popts": [p.tolist() if p is not None else None for p in popts],
        "pcovs": [c.tolist() if c is not None else None for c in pcovs],
        "red_chi2": [float(c) if c is not None else None for c in chis],
        "fit_fn_names": [
            fn.__name__ if fn is not None else None for fn in fit_fns
        ],
        "unified_keys": [
            "baseline", "comb_contrast", "revival_time_us", "width0_us", "T2_ms", "T2_exp",
            "amp_taper_alpha", "width_slope", "revival_chirp",
            "osc_contrast", "osc_f0", "osc_f1", "osc_phi0", "osc_phi1"
        ]
    }
    repr_nv_sig = widefield.get_repr_nv_sig(nv_list)
    repr_nv_name = repr_nv_sig.name
    file_path = dm.get_file_path(__file__, timestamp, repr_nv_name)
    print(file_path )
    # dm.save_raw_data(fit_dict, file_path)

    ## laod analysed data
    # file_stem= "2025_11_01-16_57_48-rubin-nv0_2025_09_08"
    # data = dm.get_raw_data(file_stem=file_stem)
    # popts = data["popts"]
    # chis = data["red_chi2"]
    # fit_nv_labels = data ["nv_labels"]
    # fit_fns = data["fit_fn_names"]
    # repr_nv_sig = widefield.get_repr_nv_sig(nv_list)
    # repr_nv_name = repr_nv_sig.name
    # 2) PARAM PANELS (T2 outlier filter)
    figs, keep_mask, kept_labels = plot_each_param_separately(
        popts, chis, fit_nv_labels, 
        save_prefix= "rubin-spin_echo-2025_09_08",
        t2_policy=dict(method="iqr", iqr_k=5, abs_range=(0.00, 1.0))
    )

    # 3) INDIVIDUAL FITS — PASS THE SAME LABELS + PER-NV FIT FUNCTIONS
    _ = plot_individual_fits(
        norm_counts, norm_counts_ste, total_evolution_times,
        popts,
        nv_inds=fit_nv_labels,
        fit_fn_per_nv=fit_fns,
        keep_mask=keep_mask,
        show_residuals=True,
        block=False
    )

    
    ## laod analysed data
    kpl.show(block=True)
