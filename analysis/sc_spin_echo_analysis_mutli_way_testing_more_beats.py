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
# =============================================================================
# =========================
# Flexible MOD: N×sin^2 with optional linear chirp per tone
# =========================


# ---- Packing / Unpacking helpers for variable-N modulation ------------------

def _pack_params_core(baseline, comb_contrast, revival_time, width0_us, T2_ms, T2_exp):
    return [baseline, comb_contrast, revival_time, width0_us, T2_ms, T2_exp]

def _unpack_params_core(p):
    baseline, comb_contrast, revival_time, width0_us, T2_ms, T2_exp = p[:6]
    return baseline, comb_contrast, revival_time, width0_us, T2_ms, T2_exp

def _pack_mod_params(mod_amps, mod_f0, mod_f1, mod_phi, use_chirp):
    """
    Return a flat list for N tones. If use_chirp=False, f1 is ignored.
    Order per tone: [A_i, f0_i, (f1_i), phi_i]
    """
    out = []
    N = len(mod_amps)
    for i in range(N):
        out += [float(mod_amps[i]), float(mod_f0[i])]
        if use_chirp: out += [float(mod_f1[i])]
        out += [float(mod_phi[i])]
    return out

def _unpack_mod_params(p_flat, N, use_chirp):
    """
    Inverse of _pack_mod_params. Returns arrays A, f0, f1, phi (f1 zeros if not used).
    """
    amps, f0s, f1s, phis = [], [], [], []
    idx = 0
    for _ in range(N):
        A = p_flat[idx]; idx += 1
        f0 = p_flat[idx]; idx += 1
        if use_chirp:
            f1 = p_flat[idx]; idx += 1
        else:
            f1 = 0.0
        phi = p_flat[idx]; idx += 1
        amps.append(A); f0s.append(f0); f1s.append(f1); phis.append(phi)
    return (np.asarray(amps, float),
            np.asarray(f0s, float),
            np.asarray(f1s, float),
            np.asarray(phis, float))


# ---- Generic model with variable # of sin² terms and optional chirp ---------

def fine_decay_nf(
    tau,
    # --- core (same as yours) ---
    baseline,
    comb_contrast,
    revival_time,
    width0_us,
    T2_ms,
    T2_exp,
    amp_taper_alpha=0.0,
    width_slope=0.0,
    revival_chirp=0.0,
    # --- modulation bundle (packed) ---
    Nfreq=0,
    use_chirp=False,
    mod_params_flat=(),
):
    """
    signal(τ) = baseline - envelope(τ) * MOD_N(τ) * COMB(τ)

    MOD_N(τ) = comb_contrast - sum_{i=1..N} A_i * sin^2( π * [f0_i + f1_i * τ] * τ + φ_i )
      where f1_i = 0 if use_chirp=False.
    """
    tau = np.asarray(tau, float).ravel()

    # envelope
    T2_us = max(1e-9, 1000.0 * float(T2_ms))
    env = np.exp(-((tau / T2_us) ** float(T2_exp)))

    # comb (no global amplitude)
    tau_max = float(np.nanmax(tau)) if tau.size else 0.0
    n_guess = max(1, min(64, int(np.ceil(1.2 * tau_max / max(1e-9, float(revival_time)))) + 1))
    comb = _comb_quartic_powerlaw(
        tau, float(revival_time), max(1e-9, float(width0_us)),
        float(amp_taper_alpha), float(width_slope), float(revival_chirp), n_guess
    )

    # modulation
    if (Nfreq is None) or (int(Nfreq) <= 0):
        mod = float(comb_contrast)
    else:
        A, f0, f1, phi = _unpack_mod_params(mod_params_flat, int(Nfreq), bool(use_chirp))
        # sum of sin^2 terms
        sterm = 0.0
        if bool(use_chirp):
            # linear chirp: f_i(τ) = f0_i + f1_i τ
            for Ai, f0i, f1i, phii in zip(A, f0, f1, phi):
                arg = np.pi * (f0i + f1i * tau) * tau + phii
                sterm += Ai * (np.sin(arg) ** 2)
        else:
            for Ai, f0i, phii in zip(A, f0, phi):
                arg = np.pi * (f0i * tau) + phii
                sterm += Ai * (np.sin(arg) ** 2)
        mod = float(comb_contrast) - sterm

    return float(baseline) - env * mod * comb


# ---- Parameter vector builder for fine_decay_nf -----------------------------

def make_p0_bounds_nf(times_us, y,
                      Nfreq=2, use_chirp=False,
                      enable_comb_extras=True):
    """
    Build p0, lb, ub for the variable-N model.
    Layout of the flat parameter vector we will fit:
      [ core(6),
        comb_extras(3) = amp_taper_alpha, width_slope, revival_chirp (optional),
        MOD bundle (packed per-tone) ]

    Returns: p0, lb, ub, meta dict describing indices and shapes.
    """
    t = np.asarray(times_us, float)
    y = np.asarray(y, float)
    idx_b = min(7, len(y)-1) if len(y) else 0
    baseline_guess = float(y[idx_b]) if len(y) else 0.5
    quart_min = float(np.nanmin(y)) if len(y) else baseline_guess - 0.2
    comb_contrast_guess = max(1e-3, baseline_guess - quart_min)

    T2_exp_guess = 3.0
    if len(t):
        j = max(0, len(t)-7)
        tlate = max(1e-3, float(t[j]))
        ratio = (baseline_guess - y[j]) / max(1e-9, comb_contrast_guess)
        ratio = min(max(ratio, 1e-9), 0.999999)
        T2_ms_guess = 0.1 * (1.0 / max(1e-9, (-np.log(ratio)) ** (1.0 / T2_exp_guess)))
    else:
        T2_ms_guess = 0.3

    core0 = [baseline_guess, comb_contrast_guess, 39.2, 6.0, T2_ms_guess, 3.0]
    core_lb = [0.0, 0.0, 30.0, 0.5, 0.0, 0.5]
    core_ub = [1.0, 1.0, 50.0, 20.0, 2000.0, 10.0]

    if enable_comb_extras:
        comb0 = [0.3, 0.02, 0.0]     # alpha, width_slope, revival_chirp
        comb_lb = [0.0, 0.00, -0.01]
        comb_ub = [2.0, 0.20,  0.01]
    else:
        comb0 = []
        comb_lb = []
        comb_ub = []

    # modulation tones
    N = int(max(0, Nfreq))
    if N == 0:
        mod0 = []; mod_lb = []; mod_ub = []
    else:
        # coarse defaults: small amplitudes, ~0.1-0.5 1/µs frequencies, phases 0
        A0   = [0.10]*N
        f00  = np.linspace(0.08, 0.45, N).tolist()
        phi0 = [0.0]*N
        if use_chirp:
            f10 = [0.0]*N  # start with no chirp; fit can turn it on
            mod0 = _pack_mod_params(A0, f00, f10, phi0, use_chirp=True)
            # bounds
            A_lb, A_ub   = [0.0]*N, [0.8]*N
            f0_lb, f0_ub = [0.00]*N, [5.00]*N
            f1_lb, f1_ub = [-0.02]*N, [0.02]*N  # conservative linear chirp
            phi_lb, phi_ub = [-np.pi]*N, [np.pi]*N

            mod_lb = []
            for i in range(N): mod_lb += [A_lb[i], f0_lb[i], f1_lb[i], phi_lb[i]]
            mod_ub = []
            for i in range(N): mod_ub += [A_ub[i], f0_ub[i], f1_ub[i], phi_ub[i]]

        else:
            mod0 = _pack_mod_params(A0, f00, None, phi0, use_chirp=False)
            A_lb, A_ub   = [0.0]*N, [0.8]*N
            f0_lb, f0_ub = [0.00]*N, [5.00]*N
            phi_lb, phi_ub = [-np.pi]*N, [np.pi]*N

            mod_lb = []
            for i in range(N): mod_lb += [A_lb[i], f0_lb[i],           phi_lb[i]]
            mod_ub = []
            for i in range(N): mod_ub += [A_ub[i], f0_ub[i],           phi_ub[i]]

    p0 = core0 + comb0 + mod0
    lb = core_lb + comb_lb + mod_lb
    ub = core_ub + comb_ub + mod_ub

    meta = {
        "Nfreq": N,
        "use_chirp": bool(use_chirp),
        "idx_core": slice(0, 6),
        "idx_comb": slice(6, 9) if enable_comb_extras else slice(6, 6),
        "idx_mod":  slice(9, 9 + len(mod0)) if enable_comb_extras else slice(6, 6 + len(mod0)),
        "enable_comb_extras": bool(enable_comb_extras),
    }
    return np.asarray(p0, float), np.asarray(lb, float), np.asarray(ub, float), meta


def _call_fine_decay_nf_from_flat(t, p, meta):
    """
    Call fine_decay_nf(t, ...) with a FLAT parameter vector p and meta that
    defines Nfreq, use_chirp, and slices.
    """
    N, use_chirp = meta["Nfreq"], meta["use_chirp"]
    s_core = meta["idx_core"]; s_comb = meta["idx_comb"]; s_mod = meta["idx_mod"]

    core = p[s_core]
    baseline, comb_contrast, revival_time, width0, T2_ms, T2_exp = _unpack_params_core(core)

    if s_comb.stop - s_comb.start == 3:
        alpha, width_slope, rev_chirp = p[s_comb]
    else:
        alpha = width_slope = rev_chirp = 0.0

    mod_params = p[s_mod] if (N > 0) else ()
    return fine_decay_nf(
        t, baseline, comb_contrast, revival_time, width0, T2_ms, T2_exp,
        amp_taper_alpha=alpha, width_slope=width_slope, revival_chirp=rev_chirp,
        Nfreq=N, use_chirp=use_chirp, mod_params_flat=mod_params
    )


def fit_nf_model(t, y, yerr,
                 Nfreq=2, use_chirp=False,
                 enable_comb_extras=True,
                 maxfev=200000, max_nfev=200000):
    """
    Fit one NV trace with the variable-N model. Returns:
      popt_flat, pcov_or_None, chi2_red, meta, callable predict(t)
    """
    t = np.asarray(t, float); y = np.asarray(y, float); e = np.asarray(yerr, float)
    m = np.isfinite(t) & np.isfinite(y) & np.isfinite(e)
    t, y, e = t[m], y[m], np.maximum(1e-3, np.abs(e[m]))
    if t.size < 8:
        raise RuntimeError("Too few valid points")

    p0, lb, ub, meta = make_p0_bounds_nf(t, y, Nfreq=Nfreq, use_chirp=use_chirp,
                                         enable_comb_extras=enable_comb_extras)

    # Try curve_fit first (fast)
    try:
        popt, pcov, _ = curve_fit(lambda tt, *pp: _call_fine_decay_nf_from_flat(tt, np.asarray(pp, float), meta),
                                  t, y, p0, e, bounds=[lb, ub],
                                  ftol=1e-7, xtol=1e-7, gtol=1e-7, maxfev=maxfev)
        yfit = _call_fine_decay_nf_from_flat(t, popt, meta)
        chi2 = _chi2_red(y, e, yfit, len(popt))
        predict = lambda tt: _call_fine_decay_nf_from_flat(np.asarray(tt, float), popt, meta)
        return popt, pcov, chi2, meta, predict
    except Exception:
        pass

    # Robust least-squares (soft-L1)
    def resid(pp): return (y - _call_fine_decay_nf_from_flat(t, pp, meta)) / e
    res = least_squares(resid, x0=p0, bounds=(lb, ub), loss="soft_l1", f_scale=1.0,
                        max_nfev=max_nfev, ftol=1e-8, xtol=1e-8, gtol=1e-8)
    popt = res.x
    yfit = _call_fine_decay_nf_from_flat(t, popt, meta)
    chi2 = _chi2_red(y, e, yfit, len(popt))
    predict = lambda tt: _call_fine_decay_nf_from_flat(np.asarray(tt, float), popt, meta)
    return popt, None, chi2, meta, predict


# ---- Try a grid of models (Nfreq = 0..Nmax; chirp on/off) and pick best -----

def _aic(chi2, n, k):  # n = #points, k = #params
    return n * np.log(max(1e-12, chi2)) + 2 * k

def _bic(chi2, n, k):
    return n * np.log(max(1e-12, chi2)) + k * np.log(max(1, n))

def fit_models_grid(t, y, yerr,
                    Nmax=3,
                    test_chirp=(False, True),
                    enable_comb_extras=True):
    """
    Fit a family of models and return the best by BIC.
    Returns dict with entries for all tried models and 'best' key.
    """
    t = np.asarray(t, float); y = np.asarray(y, float); e = np.asarray(yerr, float)
    m = np.isfinite(t) & np.isfinite(y) & np.isfinite(e)
    t, y, e = t[m], y[m], np.maximum(1e-3, np.abs(e[m]))

    results = []
    for N in range(0, int(Nmax)+1):
        for use_chirp in test_chirp:
            try:
                popt, pcov, chi2, meta, predict = fit_nf_model(
                    t, y, e, Nfreq=N, use_chirp=use_chirp,
                    enable_comb_extras=enable_comb_extras
                )
                k = len(popt)
                model_tag = f"N={N}, chirp={'on' if use_chirp else 'off'}"
                results.append({
                    "model": model_tag,
                    "Nfreq": N,
                    "use_chirp": bool(use_chirp),
                    "popt": popt,
                    "pcov": pcov,
                    "chi2_red": chi2,
                    "AIC": _aic(chi2, len(t), k),
                    "BIC": _bic(chi2, len(t), k),
                    "meta": meta,
                    "predict": predict
                })
            except Exception as ex:
                results.append({
                    "model": f"N={N}, chirp={'on' if use_chirp else 'off'}",
                    "error": str(ex)
                })

    # pick best by BIC (lower is better)
    ok = [r for r in results if "BIC" in r]
    best = min(ok, key=lambda r: r["BIC"]) if ok else None
    return {"trials": results, "best": best}

# =============================================================================
# CLI / Example
# =============================================================================
# ============================================
# Run variable-N (0..Nmax) model selection for ALL NVs
# - Picks best by BIC for each NV
# - Saves per-NV fit plots
# - Writes a summary CSV + full JSON
# ============================================

import os, json
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Assumes you already defined:
#   fit_models_grid(t, y, yerr, Nmax=3, test_chirp=(False, True), enable_comb_extras=True)
#   _unpack_params_core, _unpack_mod_params, _call_fine_decay_nf_from_flat

def _parse_best_params(best):
    """
    Convert best['popt'] into a readable dict with core, comb_extras, and modulation arrays.
    """
    p = np.asarray(best["popt"], float)
    meta = best["meta"]
    core = p[meta["idx_core"]]
    baseline, comb_contrast, revival_time_us, width0_us, T2_ms, T2_exp = _unpack_params_core(core)

    # comb extras (may be absent)
    if meta["idx_comb"].stop - meta["idx_comb"].start == 3:
        amp_taper_alpha, width_slope, revival_chirp = p[meta["idx_comb"]]
    else:
        amp_taper_alpha = width_slope = revival_chirp = 0.0

    # modulation
    N = int(meta["Nfreq"])
    use_chirp = bool(meta["use_chirp"])
    if N > 0:
        A, f0, f1, phi = _unpack_mod_params(p[meta["idx_mod"]], N, use_chirp)
        A = A.tolist(); f0 = f0.tolist(); f1 = f1.tolist(); phi = phi.tolist()
    else:
        A, f0, f1, phi = [], [], [], []

    return {
        "baseline": baseline,
        "comb_contrast": comb_contrast,
        "revival_time_us": revival_time_us,
        "width0_us": width0_us,
        "T2_ms": T2_ms,
        "T2_exp": T2_exp,
        "amp_taper_alpha": amp_taper_alpha,
        "width_slope": width_slope,
        "revival_chirp": revival_chirp,
        "Nfreq": N,
        "use_chirp": use_chirp,
        "mod_A": A,
        "mod_f0_per_us": f0,
        "mod_f1_per_us2": f1,   # zeros if use_chirp=False
        "mod_phi_rad": phi,
    }

def _plot_one_nv_fit(nv_label, t_us, y, yerr, best, save_path):
    """Data + best fit line; annotate model."""
    predict = best["predict"]
    yfit = predict(t_us)

    plt.figure(figsize=(6.8, 4.4))
    plt.errorbar(t_us, y, yerr=yerr, fmt="o", ms=3.5, lw=0.8, alpha=0.9, capsize=2, label="data")
    plt.plot(t_us, yfit, "-", lw=2.0, label=best["model"])
    plt.xlabel("τ (µs)")
    plt.ylabel("Norm. NV$^-$")
    ttl = f"NV {nv_label}  |  χ²_red={best['chi2_red']:.3f}, BIC={best['BIC']:.1f}"
    plt.title(ttl)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=220)
    plt.close()

def run_model_selection_all(
    total_evolution_times_us,
    norm_counts,
    norm_counts_ste,
    nv_inds=None,                 # which NVs to process (labels); default = range(N)
    Nmax=3,
    test_chirp=(False, True),
    enable_comb_extras=True,
    out_dir="model_select_out",
    prefix="nv_fit"
):
    """
    Loop over all NVs, try N=0..Nmax (with/without chirp), pick best by BIC.
    Saves:
      - PNG per NV (data + best fit)
      - summary.csv (one row per NV)
      - results.json (full trials + best for reproducibility)
    Returns: results dict with per-NV entries.
    """
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    base = f"{prefix}-{ts}"

    N = len(norm_counts)
    if nv_inds is None:
        nv_inds = list(range(N))
    else:
        nv_inds = list(nv_inds)

    results = {
        "timestamp": ts,
        "Nmax": int(Nmax),
        "test_chirp": list(test_chirp),
        "enable_comb_extras": bool(enable_comb_extras),
        "per_nv": []
    }

    rows = []
    for pos, nv_label in enumerate(nv_inds):
        try:
            t_us = np.asarray(total_evolution_times_us, float)
            y    = np.asarray(norm_counts[nv_label], float)
            yerr = np.asarray(norm_counts_ste[nv_label], float)

            sel = fit_models_grid(
                t_us, y, yerr,
                Nmax=Nmax,
                test_chirp=test_chirp,
                enable_comb_extras=enable_comb_extras
            )
            best = sel["best"]
            if best is None:
                raise RuntimeError("No valid model fit")

            # Save per-NV plot
            fig_path = os.path.join(out_dir, f"{base}-nv{int(nv_label):03d}.png")
            _plot_one_nv_fit(nv_label, t_us, y, yerr, best, fig_path)

            # Parse parameters into readable dict
            par = _parse_best_params(best)

            # One tidy summary row
            row = {
                "nv": int(nv_label),
                "model": best["model"],
                "Nfreq": par["Nfreq"],
                "chirp": par["use_chirp"],
                "chi2_red": float(best["chi2_red"]),
                "AIC": float(best["AIC"]),
                "BIC": float(best["BIC"]),
                "n_params": int(len(best["popt"])),
                "baseline": par["baseline"],
                "comb_contrast": par["comb_contrast"],
                "revival_time_us": par["revival_time_us"],
                "width0_us": par["width0_us"],
                "T2_ms": par["T2_ms"],
                "T2_exp": par["T2_exp"],
                "amp_taper_alpha": par["amp_taper_alpha"],
                "width_slope": par["width_slope"],
                "revival_chirp": par["revival_chirp"],
                "fig_path": fig_path,
            }
            # store first three tones explicitly for quick scans (rest kept in JSON)
            for i in range(min(3, par["Nfreq"])):
                row[f"A{i+1}"]   = par["mod_A"][i]
                row[f"f0_{i+1}"] = par["mod_f0_per_us"][i]
                row[f"f1_{i+1}"] = par["mod_f1_per_us2"][i]
                row[f"phi_{i+1}"] = par["mod_phi_rad"][i]

            rows.append(row)

            # Store full structure for this NV
            results["per_nv"].append({
                "nv": int(nv_label),
                "best": {
                    "model": best["model"],
                    "chi2_red": float(best["chi2_red"]),
                    "AIC": float(best["AIC"]),
                    "BIC": float(best["BIC"]),
                    "n_params": int(len(best["popt"])),
                    "params_flat": np.asarray(best["popt"], float).tolist(),
                    "meta": {
                        "Nfreq": par["Nfreq"],
                        "use_chirp": par["use_chirp"],
                        # human-readable param dict:
                        "parsed": par
                    },
                    "figure": fig_path
                },
                "trials": [
                    {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                     for k, v in r.items() if k not in ("predict", "popt", "pcov", "meta")}
                    for r in sel["trials"]
                ],
            })

            print(f"[OK] NV {nv_label}: {best['model']}  χ²_red={best['chi2_red']:.3f}  -> {fig_path}")

        except Exception as ex:
            print(f"[WARN] NV {nv_label} failed: {ex}")
            rows.append({
                "nv": int(nv_label),
                "model": None, "Nfreq": None, "chirp": None,
                "chi2_red": np.nan, "AIC": np.nan, "BIC": np.nan,
                "n_params": 0, "baseline": np.nan, "comb_contrast": np.nan,
                "revival_time_us": np.nan, "width0_us": np.nan,
                "T2_ms": np.nan, "T2_exp": np.nan,
                "amp_taper_alpha": np.nan, "width_slope": np.nan,
                "revival_chirp": np.nan, "fig_path": None
            })
            results["per_nv"].append({
                "nv": int(nv_label),
                "error": str(ex)
            })

    # Save summary table
    df = pd.DataFrame(rows).sort_values("nv")
    csv_path = os.path.join(out_dir, f"{base}-summary.csv")
    df.to_csv(csv_path, index=False)

    # Save full JSON (without the non-serializable callables)
    json_path = os.path.join(out_dir, f"{base}-results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved summary: {csv_path}")
    print(f"Saved results: {json_path}")
    return {"summary_csv": csv_path, "results_json": json_path, "table": df, "results": results}

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
    
    data = widefield.process_multiple_files(file_stems, load_npz=True)
    nv_list = data["nv_list"]
    taus = data["taus"]
    total_evolution_times = 2 * np.array(taus) / 1e3  
    counts = np.array(data["counts"])
    sig = counts[0]
    ref = counts[1]
    norm_counts, norm_counts_ste = widefield.process_counts(nv_list, sig, ref, threshold=True)
    # Example NV filtering
    out = run_model_selection_all(
        total_evolution_times_us=total_evolution_times,
        norm_counts=norm_counts,
        norm_counts_ste=norm_counts_ste,
        nv_inds=None,                 # or a subset like [2,10,22,...]
        Nmax=3,
        test_chirp=(False, True),
        enable_comb_extras=True,
        out_dir="model_select_out",
        prefix="rubin-finefit"
    )

    # Quick peek:
    print(out["summary_csv"])
    print(out["results_json"])
    # display(out["table"].head())


    kpl.show(block=True)
