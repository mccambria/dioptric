# -*- coding: utf-8 -*-
"""
Spin-echo: finer fit + modulation visualization

- Physics-y comb with quartic lobes, amplitude taper, width growth, chirp
- Optional two-frequency sin^2 beating with phases
- Residual spectrum & sliding Lomb–Scargle to *see* modulation clearly
"""

import json
import traceback
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.signal import lombscargle

# --- Optional numba (falls back gracefully) ----------------------------------
try:
    from numba import njit
except Exception:
    def njit(*_args, **_kwargs):
        def wrap(fn): return fn
        return wrap

# --- Your utilities (assumed in PYTHONPATH) ----------------------------------
from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import widefield as widefield
from utils.tool_belt import curve_fit  # your wrapper around scipy.curve_fit


# =============================================================================
# Physics model
# =============================================================================

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
    osc_contrast=None,
    osc_f0=None,
    osc_f1=None,
    osc_phi0=None,
    osc_phi1=None,
):
    """
    signal(τ) = baseline - envelope(τ) * MOD(τ) * COMB(τ)

    envelope(τ) = exp[-((τ / (1000*T2_ms)) ** T2_exp)]

    COMB(τ) = sum_k  [ 1/(1+k)^amp_taper_alpha ] * exp(-((τ - μ_k)/w_k)^4)
        μ_k = k * revival_time * (1 + k*revival_chirp)
        w_k = width0_us * (1 + k*width_slope)

    MOD(τ) = comb_contrast - osc_contrast * sin^2(π f0 τ + φ0) * sin^2(π f1 τ + φ1)
    """
    # defaults
    if amp_taper_alpha is None: amp_taper_alpha = 0.0
    if width_slope     is None: width_slope     = 0.0
    if revival_chirp   is None: revival_chirp   = 0.0
    if osc_contrast    is None: osc_contrast    = 0.0
    if osc_f0          is None: osc_f0          = 0.0
    if osc_f1          is None: osc_f1          = 0.0
    if osc_phi0        is None: osc_phi0        = 0.0
    if osc_phi1        is None: osc_phi1        = 0.0

    tau = np.asarray(tau, float).ravel()
    width0_us    = max(1e-9, float(width0_us))
    revival_time = max(1e-9, float(revival_time))
    T2_us        = max(1e-9, 1000.0 * float(T2_ms))
    T2_exp       = float(T2_exp)

    # envelope
    envelope = np.exp(-((tau / T2_us) ** T2_exp))

    # how many revivals to include
    tau_max = float(np.nanmax(tau)) if tau.size else 0.0
    n_guess = max(1, min(64, int(np.ceil(1.2 * tau_max / revival_time)) + 1))

    comb = _comb_quartic_powerlaw(
        tau, revival_time, width0_us,
        amp_taper_alpha, width_slope, revival_chirp, n_guess
    )

    # beating (optional) in MOD
    if (osc_contrast != 0.0) and (osc_f0 != 0.0 or osc_f1 != 0.0):
        s0 = np.sin(np.pi * osc_f0 * tau + osc_phi0)
        s1 = np.sin(np.pi * osc_f1 * tau + osc_phi1)
        beat = (s0 * s0) * (s1 * s1)
        mod = comb_contrast - osc_contrast * beat
    else:
        mod = comb_contrast

    return baseline - envelope * mod * comb


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
    Quartic lobes with power-law amplitude taper, width growth, optional chirp.
    No global amplitude here (that lives in MOD via comb_contrast).
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

        amp_k = 1.0 / ((1.0 + k) ** amp_taper_alpha)
        inv_w4 = 1.0 / (w_k ** 4)

        for i in range(n):
            x = tau[i] - mu_k
            out[i] += amp_k * np.exp(- (x * x) * (x * x) * inv_w4)

    return out


# =============================================================================
# Fitting helpers (minimal but robust)
# =============================================================================

def _safe_sigma(yerr, floor=1e-3):
    yerr = np.asarray(yerr, float)
    yerr = np.where(np.isfinite(yerr), yerr, floor)
    return np.maximum(floor, np.abs(yerr))


def _sanitize_trace(times_us, y, yerr, floor=1e-3):
    m = np.isfinite(times_us) & np.isfinite(y) & np.isfinite(yerr)
    t = np.asarray(times_us, float)[m]
    yy = np.asarray(y, float)[m]
    ee = _safe_sigma(np.asarray(yerr, float)[m], floor=floor)
    if t.size:
        _, idx = np.unique(t, return_index=True)
        t, yy, ee = t[idx], yy[idx], ee[idx]
    return t, yy, ee


def _chi2_red(y, yerr, yfit, npar):
    resid = (y - yfit) / _safe_sigma(yerr)
    chi2 = float(np.sum(resid**2))
    dof  = max(1, len(y) - npar)
    return chi2 / dof


def _initial_guess_and_bounds(times_us, y, enable_extras=True, fixed_rev_time=None):
    """
    p (variable revival) = [baseline, comb_contrast, revival_time, width0_us, T2_ms, T2_exp, ...extras...]
    p (fixed revival)    = [baseline, comb_contrast, width0_us, T2_ms, T2_exp, ...extras...]
    """
    # baseline & contrast scale
    idx_b = min(7, len(y)-1) if len(y) else 0
    baseline_guess = float(y[idx_b]) if len(y) else 0.5
    quart_min = float(np.nanmin(y)) if len(y) else baseline_guess - 0.2
    comb_contrast_guess = max(1e-3, baseline_guess - quart_min)

    # envelope rough guess
    T2_exp_guess = 3.0
    if len(times_us):
        j = max(0, len(times_us)-7)
        tlate = max(1e-3, float(times_us[j]))
        ratio = (baseline_guess - y[j]) / max(1e-9, comb_contrast_guess)
        ratio = min(max(ratio, 1e-9), 0.999999)
        T2_ms_guess = 0.1 * (1.0 / max(1e-9, (-np.log(ratio)) ** (1.0 / T2_exp_guess)))
    else:
        T2_ms_guess = 0.3

    width0_guess = 6.0
    revival_guess = 39.2 if fixed_rev_time is None else fixed_rev_time

    if fixed_rev_time is None:
        p0 = [baseline_guess, comb_contrast_guess, revival_guess, width0_guess, T2_ms_guess, T2_exp_guess]
        lb = [0.0,           0.0,                30.0,          0.5,          0.0,        0.5]
        ub = [1.0,           1.0,                50.0,          20.0,         2000.0,     10.0]
    else:
        p0 = [baseline_guess, comb_contrast_guess, width0_guess, T2_ms_guess, T2_exp_guess]
        lb = [0.0,           0.0,                0.5,           0.0,          0.5]
        ub = [1.0,           1.0,                20.0,          2000.0,       10.0]

    if enable_extras:
        extra_p0 = [0.3, 0.02, 0.0,  0.15, 0.30, 0.10, 0.0, 0.0]
        extra_lb = [0.0, 0.00, -0.01, -0.5, 0.00, 0.00, -np.pi, -np.pi]
        extra_ub = [2.0, 0.20,  0.01,  0.5, 5.00, 1.00,  np.pi,  np.pi]
        p0.extend(extra_p0); lb.extend(extra_lb); ub.extend(extra_ub)

    return np.array(p0, float), np.array(lb, float), np.array(ub, float)


def fit_one_nv(times_us, y, yerr,
               use_fixed_revival=False, enable_extras=True,
               fixed_rev_time=39.2, maxfev=200000, max_nfev=200000):
    """
    Tries curve_fit with/without extras; falls back to robust least_squares.
    Returns popt, pcov (or None), chi2_red, fit_fn
    """
    t, yy, ee = _sanitize_trace(times_us, y, yerr, floor=1e-3)
    if len(t) < 8:
        raise RuntimeError("Too few valid points after sanitization")

    if use_fixed_revival:
        fit_fn = fine_decay_fixed_revival
        p0, lb, ub = _initial_guess_and_bounds(t, yy, enable_extras, fixed_rev_time=fixed_rev_time)
        n_core = 5
    else:
        fit_fn = fine_decay
        p0, lb, ub = _initial_guess_and_bounds(t, yy, enable_extras, fixed_rev_time=None)
        n_core = 6

    # Strategy A: full param set
    try:
        popt, pcov, _ = curve_fit(
            fit_fn, t, yy, p0, ee, bounds=[lb, ub],
            ftol=1e-7, xtol=1e-7, gtol=1e-7, maxfev=maxfev
        )
        yfit = fit_fn(t, *popt)
        return popt, pcov, _chi2_red(yy, ee, yfit, len(popt)), fit_fn
    except Exception:
        pass

    # Strategy B: core only
    if enable_extras:
        try:
            popt, pcov, _ = curve_fit(
                fit_fn, t, yy, p0[:n_core], ee, bounds=[lb[:n_core], ub[:n_core]],
                ftol=1e-7, xtol=1e-7, gtol=1e-7, maxfev=maxfev
            )
            yfit = fit_fn(t, *popt)
            return popt, pcov, _chi2_red(yy, ee, yfit, len(popt)), fit_fn
        except Exception:
            pass

    # Strategy C: robust least-squares (soft-L1), full set
    def resid(p): return (yy - fit_fn(t, *p)) / ee
    try:
        res = least_squares(resid, x0=p0, bounds=(lb, ub),
                            loss="soft_l1", f_scale=1.0,
                            max_nfev=max_nfev, ftol=1e-8, xtol=1e-8, gtol=1e-8)
        popt = res.x
        yfit = fit_fn(t, *popt)
        return popt, None, _chi2_red(yy, ee, yfit, len(popt)), fit_fn
    except Exception as e:
        raise RuntimeError(f"All strategies failed: {e}")


# =============================================================================
# Visualization: residual spectrum, sliding LS, MOD overlay, autocorr
# =============================================================================

def _residuals(t, y, yerr, yfit):
    t = np.asarray(t, float); y = np.asarray(y, float); yfit = np.asarray(yfit, float)
    e = _safe_sigma(yerr)
    m = np.isfinite(t) & np.isfinite(y) & np.isfinite(yfit) & np.isfinite(e)
    return t[m], (y[m] - yfit[m]), e[m]


def _lomb_scargle(t, r, fmin=0.01, fmax=2.0, n_f=4000):
    t = np.asarray(t, float)
    r = np.asarray(r, float) - np.nanmean(r)
    freqs = np.linspace(fmin, fmax, n_f)
    w = 2 * np.pi * freqs
    p = lombscargle(t, r, w, precenter=True, normalize=True)
    return freqs, p


def _windowed_lomb_scargle(t, r, win_us=15.0, step_us=4.0,
                           fmin=0.01, fmax=2.0, n_f=1500):
    t = np.asarray(t, float)
    r = np.asarray(r, float) - np.nanmean(r)
    centers, powers = [], []
    freqs = np.linspace(fmin, fmax, n_f)
    w = 2 * np.pi * freqs
    tmin, tmax = np.nanmin(t), np.nanmax(t)
    c = tmin + 0.5 * win_us
    while c <= tmax - 0.5 * win_us:
        lo, hi = c - 0.5 * win_us, c + 0.5 * win_us
        m = (t >= lo) & (t <= hi)
        if m.sum() >= 10:
            pm = lombscargle(t[m], r[m], w, precenter=True, normalize=True)
            powers.append(pm); centers.append(c)
        c += step_us
    if not powers:
        return np.array([]), freqs, np.zeros((0, len(freqs)))
    return np.asarray(centers), freqs, np.vstack(powers)


def reconstruct_MOD(t, comb_contrast, osc_contrast=0.0,
                    f0=0.0, f1=0.0, phi0=0.0, phi1=0.0):
    t = np.asarray(t, float)
    if (osc_contrast == 0.0) or ((f0 == 0.0) and (f1 == 0.0)):
        return np.full_like(t, comb_contrast, float)
    s0 = np.sin(np.pi * f0 * t + phi0)
    s1 = np.sin(np.pi * f1 * t + phi1)
    return comb_contrast - osc_contrast * (s0 * s0) * (s1 * s1)


def autocorr(x):
    x = np.asarray(x, float) - np.nanmean(x)
    ac = np.correlate(x, x, mode="full")
    ac = ac[ac.size // 2:]
    if ac[0] != 0:
        ac = ac / ac[0]
    return ac


def visualize_modulation_suite(
    t_us, y, yerr, yfit,
    popt=None,                # if provided, reconstruct MOD(τ)
    model_key_index=None,     # indices for keys inside popt
    freq_range=(0.01, 1.2),   # in 1/µs
    ls_win_us=20.0, ls_step_us=5.0,
    title="Spin-echo modulation diagnostics"
):
    """
    (A) data+fit + residuals   (B) residual Lomb–Scargle
    (C) sliding LS spectrogram (D) MOD(τ) overlay (if popt given)
    (E) residual autocorrelation
    """
    t, r, _ = _residuals(t_us, y, yerr, yfit)

    fmin, fmax = freq_range
    freqs, power = _lomb_scargle(t, r, fmin=fmin, fmax=fmax, n_f=4000)
    centers, f2, P2 = _windowed_lomb_scargle(
        t, r, win_us=ls_win_us, step_us=ls_step_us, fmin=fmin, fmax=fmax, n_f=1200
    )

    nrows = 3 if popt is None else 4
    fig = plt.figure(figsize=(10, 12 if popt is None else 14))
    gs = fig.add_gridspec(nrows=nrows, ncols=2,
                          height_ratios=[1.2, 1.1, 1.0, 0.9][:nrows],
                          hspace=0.35, wspace=0.25)

    # (A1) data + fit
    axA1 = fig.add_subplot(gs[0, 0])
    axA1.errorbar(t_us, y, yerr=yerr, fmt="o", ms=3.5, lw=0.8, alpha=0.85, capsize=2, label="data")
    axA1.plot(t_us, yfit, "-", lw=2, label="fit")
    axA1.set_ylabel("Norm. NV$^-$ population")
    axA1.set_title(title)
    axA1.grid(alpha=0.25)
    axA1.legend()

    # (A2) residuals
    axA2 = fig.add_subplot(gs[0, 1])
    axA2.axhline(0, ls="--", lw=1)
    axA2.plot(t, r, ".", ms=3.5)
    axA2.set_xlabel("τ (µs)")
    axA2.set_ylabel("residual")
    axA2.set_title("Residuals vs time")
    axA2.grid(alpha=0.25)

    # (B) LS spectrum
    axB = fig.add_subplot(gs[1, :])
    axB.plot(freqs, power, lw=1.6)
    axB.set_xlim(fmin, fmax)
    axB.set_xlabel("frequency (1/µs)")
    axB.set_ylabel("Lomb–Scargle power")
    axB.set_title("Residual spectrum (Lomb–Scargle)")
    axB.grid(alpha=0.25)

    if (popt is not None) and (model_key_index is not None):
        f0 = popt[ model_key_index["osc_f0"] ]
        f1 = popt[ model_key_index["osc_f1"] ]
        for f, lab in [(f0, "f0"), (f1, "f1")]:
            if np.isfinite(f) and f > 0:
                axB.axvline(f, color="k", ls="--", lw=1)
                axB.text(f, 0.95*np.nanmax(power), lab, ha="center", va="top")

    # (C) Spectrogram-like sliding LS
    axC = fig.add_subplot(gs[2, :])
    if P2.size > 0:
        t_edges = np.r_[centers - ls_step_us/2, centers[-1] + ls_step_us/2] if centers.size else centers
        f_edges = np.r_[f2, f2[-1] + (f2[1]-f2[0])]
        Pn = P2 / (np.nanmax(P2, axis=1, keepdims=True) + 1e-12)
        im = axC.pcolormesh(t_edges, f_edges, Pn.T, shading="auto")
        axC.set_ylabel("frequency (1/µs)")
        axC.set_xlabel("τ (µs)")
        axC.set_title(f"Sliding LS (win={ls_win_us} µs, step={ls_step_us} µs)")
        fig.colorbar(im, ax=axC, label="norm. power")
    else:
        axC.text(0.5, 0.5, "Not enough points for sliding LS", ha="center", va="center", transform=axC.transAxes)
        axC.axis("off")

    # (D1) Reconstructed MOD(τ) + (D2) autocorr
    if popt is not None and model_key_index is not None:
        axD1 = fig.add_subplot(gs[3-1, 0])  # index-safe
        comb_contrast = popt[ model_key_index["comb_contrast"] ]
        osc_contrast  = popt[ model_key_index["osc_contrast"] ]
        f0            = popt[ model_key_index["osc_f0"] ]
        f1            = popt[ model_key_index["osc_f1"] ]
        phi0          = popt[ model_key_index["osc_phi0"] ]
        phi1          = popt[ model_key_index["osc_phi1"] ]
        t_dense = np.linspace(np.min(t_us), np.max(t_us), 2000)
        mod = reconstruct_MOD(t_dense, comb_contrast, osc_contrast, f0, f1, phi0, phi1)
        axD1.plot(t_dense, mod, lw=1.8)
        axD1.set_title("Reconstructed MOD(τ) from fit")
        axD1.set_xlabel("τ (µs)"); axD1.set_ylabel("MOD(τ)")
        axD1.grid(alpha=0.25)

    axD2 = fig.add_subplot(gs[3-1, 1])
    ac = autocorr(r)
    if len(t) > 1:
        dt = np.median(np.diff(np.sort(t)))
        lags = np.arange(len(ac)) * dt
    else:
        lags = np.arange(len(ac))
    axD2.plot(lags, ac, lw=1.6)
    axD2.set_title("Residual autocorrelation")
    axD2.set_xlabel("lag (µs)"); axD2.set_ylabel("ACF")
    axD2.grid(alpha=0.25)

    return fig


# =============================================================================
# Batch fit orchestration + labeled record
# =============================================================================

def run(nv_list, norm_counts, norm_counts_ste, total_evolution_times_us,
        nv_inds=None, use_fixed_revival=False, enable_extras=True, fixed_rev_time=39.2):
    if nv_inds is None:
        nv_inds = list(range(len(nv_list)))

    popts, pcovs, chis, fit_fns, labels = [], [], [], [], []

    for i, nv in enumerate(nv_inds):
        try:
            popt, pcov, chi, fn = fit_one_nv(
                total_evolution_times_us, norm_counts[nv], norm_counts_ste[nv],
                use_fixed_revival=use_fixed_revival, enable_extras=enable_extras,
                fixed_rev_time=fixed_rev_time
            )
        except Exception:
            print(f"[WARN] Fit failed for NV {nv}\n" + traceback.format_exc())
            popt = pcov = fn = None; chi = np.nan

        popts.append(popt); pcovs.append(pcov); chis.append(chi); fit_fns.append(fn); labels.append(nv)

    return popts, pcovs, chis, fit_fns, labels


_UNIFIED_KEYS = [
    "baseline", "comb_contrast", "revival_time_us", "width0_us", "T2_ms", "T2_exp",
    "amp_taper_alpha", "width_slope", "revival_chirp",
    "osc_contrast", "osc_f0", "osc_f1", "osc_phi0", "osc_phi1"
]


def _normalize_popt_to_unified(p):
    q = np.full(14, np.nan, float)
    if p is None: return q
    p = np.asarray(p, float); L = len(p)
    if L == 6:      q[0:6] = p[0:6]
    elif L == 5:    q[0]=p[0]; q[1]=p[1]; q[2]=np.nan; q[3]=p[2]; q[4]=p[3]; q[5]=p[4]
    elif L == 14:   q[0:6]=p[0:6]; q[6:]=p[6:14]
    elif L == 13:   q[0]=p[0]; q[1]=p[1]; q[2]=np.nan; q[3]=p[2]; q[4]=p[3]; q[5]=p[4]; q[6:]=p[5:13]
    else:
        if L >= 6:
            q[0:6]=p[0:6]
            if L>6:
                m=min(8,L-6); q[6:6+m]=p[6:6+m]
        elif L >= 5:
            q[0]=p[0]; q[1]=p[1]; q[2]=np.nan; q[3]=p[2]; q[4]=p[3]; q[5]=p[4]
    return q


def build_fitted_record(nv_labels, popts, pcovs, chis,
                        model_name, file_stems, extra_meta=None):
    """
    Returns a JSON-serializable dict with clear labeling.
    """
    records = []
    for lbl, p, cov, chi in zip(nv_labels, popts, pcovs, chis):
        rec = {
            "nv_label": int(lbl),
            "fit_success": (p is not None),
            "reduced_chi2": None if not np.isfinite(chi) else float(chi),
            "params_unified": dict(zip(_UNIFIED_KEYS, map(lambda x: None if np.isnan(x) else float(x),
                                                         _normalize_popt_to_unified(p)))),
        }
        records.append(rec)

    out = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model": model_name,
        "source_files": list(map(str, file_stems)),
        "n_fits": len(records),
        "fits": records,
    }
    if extra_meta:
        out["meta"] = extra_meta
    return out


# =============================================================================
# CLI / Example
# =============================================================================
if __name__ == "__main__":
    kpl.init_kplotlib()

    # --- Load data (adjust your stems as needed) ------------------------------
    file_stems = [
        "2025_10_10-11_29_40-rubin-nv0_2025_09_08",
        "2025_10_10-08_55_59-rubin-nv0_2025_09_08",
        "2025_10_10-06_28_12-rubin-nv0_2025_09_08",
        "2025_10_10-03_59_48-rubin-nv0_2025_09_08",
        "2025_10_10-01_31_59-rubin-nv0_2025_09_08",
        "2025_10_09-23_03_41-rubin-nv0_2025_09_08",
        "2025_10_10-14_23_58-rubin-nv0_2025_09_08",
        "2025_10_10-17_04_27-rubin-nv0_2025_09_08"
    ]

    data = widefield.process_multiple_files(file_stems, load_npz=True)
    nv_list = data["nv_list"]
    taus = np.array(data["taus"])
    total_evolution_times_us = 2 * taus / 1e3  # τ_total in µs

    counts = np.array(data["counts"])
    sig, ref = counts[0], counts[1]
    norm_counts, norm_counts_ste = widefield.process_counts(nv_list, sig, ref, threshold=True)

    # --- Fit (all NVs by default) --------------------------------------------
    USE_FIXED_REVIVAL = False
    ENABLE_EXTRAS = True
    FIXED_REV_TIME = 39.2

    popts, pcovs, chis, fit_fns, fit_nv_labels = run(
        nv_list, norm_counts, norm_counts_ste, total_evolution_times_us,
        nv_inds=None,
        use_fixed_revival=USE_FIXED_REVIVAL,
        enable_extras=ENABLE_EXTRAS,
        fixed_rev_time=FIXED_REV_TIME
    )

    # --- Save labeled fit record ---------------------------------------------
    ts = dm.get_time_stamp()
    repr_nv_sig = widefield.get_repr_nv_sig(nv_list)
    base = dm.get_file_path(__file__, ts, f"{repr_nv_sig.name}-finefit")

    # record = build_fitted_record(
    #     fit_nv_labels, popts, pcovs, chis,
    #     model_name=("fine_decay_fixed_revival" if USE_FIXED_REVIVAL else "fine_decay"),
    #     file_stems=file_stems,
    #     extra_meta={
    #         "use_fixed_revival": USE_FIXED_REVIVAL,
    #         "enable_extras": ENABLE_EXTRAS,
    #         "fixed_rev_time_us": FIXED_REV_TIME
    #     }
    # )
    # with open(base + "-fits.json", "w", encoding="utf-8") as f:
    #     json.dump(record, f, indent=2)
    # print(f"[saved] {base}-fits.json")

    # --- Visualize modulation for a few NVs ----------------------------------
    # pick top-3 by (finite, smallest) chi2
    finite_idx = [i for i, c in enumerate(chis) if np.isfinite(c) and popts[i] is not None]
    if finite_idx:
        order = sorted(finite_idx, key=lambda i: chis[i])[:3]
        for pos in order:
            nv = fit_nv_labels[pos]
            t_us = total_evolution_times_us
            y    = norm_counts[nv]
            yerr = norm_counts_ste[nv]
            p    = popts[pos]
            model = fit_fns[pos] if fit_fns[pos] is not None else fine_decay
            yfit = model(t_us, *p)

            # map indices for the "full" layout (adjust if you used fixed-core)
            if len(p) >= 14:  # full with extras
                key_map = {"comb_contrast": 1, "osc_contrast": 9,
                           "osc_f0": 10, "osc_f1": 11, "osc_phi0": 12, "osc_phi1": 13}
            elif len(p) >= 6:  # core only (no extras) -> skip MOD overlay
                key_map = None
            else:              # fixed revival 5 or 13 -> still okay; extras may exist
                key_map = {"comb_contrast": 1, "osc_contrast": 9,
                           "osc_f0": 10, "osc_f1": 11, "osc_phi0": 12, "osc_phi1": 13} if len(p) >= 13 else None

            fig = visualize_modulation_suite(
                t_us, y, yerr, yfit,
                popt=p if key_map is not None else None,
                model_key_index=key_map,
                freq_range=(0.02, 1.2),
                ls_win_us=20.0, ls_step_us=5.0,
                title=f"NV {nv} – modulation diagnostics"
            )
            # fig.savefig(f"{base}-nv{int(nv):03d}-moddiag.png", dpi=220)
            # plt.close(fig)

    kpl.show(block=True)
