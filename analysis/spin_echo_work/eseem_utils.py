# === eseem_utils.py (cleaned) ================================================
from __future__ import annotations
import json
from typing import Iterable, Optional, Tuple, List, Dict
import numpy as np
import matplotlib.pyplot as plt

# ---------- I/O ----------
def load_catalog(path_json: str) -> List[Dict]:
    with open(path_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["records"] if isinstance(data, dict) and "records" in data else data

def select_records(
    recs: List[Dict],
    fmin_kHz: float = 150.0,
    fmax_kHz: float = 20000.0,
    orientations: Optional[Iterable[Tuple[int,int,int]]] = None,
) -> List[Dict]:
    lo, hi = float(fmin_kHz), float(fmax_kHz)
    allow = None if orientations is None else {tuple(map(int,o)) for o in orientations}
    out = []
    for r in recs:
        if allow is not None and tuple(r.get("orientation",(0,0,0))) not in allow:
            continue
        fm = float(r.get("f_minus_Hz", np.nan))/1e3
        fp = float(r.get("f_plus_Hz",  np.nan))/1e3
        if (np.isfinite(fm) and lo <= fm <= hi) or (np.isfinite(fp) and lo <= fp <= hi):
            out.append(r)
    return out

# ---------- weighting kernel ----------
def _line_weight_from_record(
    r: Dict,
    tag: str,                       # "minus" or "plus"
    *,
    p_occ: float,
    weight_mode: str = "kappa",     # {"kappa","per_line","amp","unit"}
    per_line_scale: float = 1.0,
) -> float:
    mode = weight_mode.lower()
    if mode == "kappa":
        w = float(r.get("kappa", 0.0))
    elif mode == "per_line":
        key = "line_w_minus" if tag == "minus" else "line_w_plus"
        w = float(r.get(key, 0.0))
    elif mode == "amp":
        # legacy proxy weight from your older builder
        w = float(r.get("amp_weight", 1.0)) * float(p_occ)
    else:
        w = 1.0
    return per_line_scale * max(0.0, w)

# ---------- fundamentals (f±) ----------
def fundamentals_from_recs(
    recs: List[Dict],
    *,
    p_occ: float = 0.011,
    f_range_kHz: Tuple[float,float] = (150.0, 20000.0),
    weight_mode: str = "kappa",
    per_line_scale: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    lo, hi = map(float, f_range_kHz)
    F, W = [], []
    for r in recs:
        fm = float(r.get("f_minus_Hz", np.nan))/1e3
        fp = float(r.get("f_plus_Hz",  np.nan))/1e3
        if np.isfinite(fm) and lo <= fm <= hi:
            F.append(fm); W.append(_line_weight_from_record(r, "minus",
                        p_occ=p_occ, weight_mode=weight_mode, per_line_scale=per_line_scale))
        if np.isfinite(fp) and lo <= fp <= hi:
            F.append(fp); W.append(_line_weight_from_record(r, "plus",
                        p_occ=p_occ, weight_mode=weight_mode, per_line_scale=per_line_scale))
    if not F:
        return np.array([]), np.array([])
    F = np.asarray(F, float); W = np.asarray(W, float)
    m = np.isfinite(F) & np.isfinite(W) & (W > 0)
    return F[m], W[m]

# ---------- dedup/merge ----------
def _dedup_merge_sorted(freqs_kHz: np.ndarray, weights: np.ndarray,
                        min_sep_kHz: float = 6.0, mode: str = "sum"):
    order = np.argsort(freqs_kHz)
    f, w = freqs_kHz[order], weights[order]
    out_f, out_w = [], []
    for fi, wi in zip(f, w):
        if not out_f:
            out_f.append(fi); out_w.append(wi); continue
        if abs(fi - out_f[-1]) < float(min_sep_kHz):
            out_w[-1] = out_w[-1] + wi if mode == "sum" else max(out_w[-1], wi)
        else:
            out_f.append(fi); out_w.append(wi)
    return np.asarray(out_f), np.asarray(out_w)

# ---------- feasible set for priors ----------
def build_feasible_freqs_from_recs(
    recs: List[Dict],
    *,
    p_occ: float = 0.011,
    f_range_kHz: Tuple[float,float] = (150.0, 20000.0),
    include_combos: bool = False,
    min_sep_kHz: float = 8.0,
    top_M_fundamentals: int = 60,
    weight_floor: float = 0.0,
    use_weights: bool = True,              # kept for API compat
    weight_mode: str = "kappa",
    per_line_scale: float = 1.0,
) -> Dict[str, np.ndarray]:
    fF, wF = fundamentals_from_recs(
        recs,
        p_occ=p_occ,
        f_range_kHz=f_range_kHz,
        weight_mode=weight_mode,
        per_line_scale=per_line_scale,
    )
    if fF.size == 0:
        return {"fund_freq_kHz": np.array([]), "fund_weight": np.array([]),
                "all_freq_kHz": np.array([]),  "all_weight": np.array([])}

    # keep strongest by weight, then dedup
    if weight_floor > 0:
        m = wF >= float(weight_floor); fF, wF = fF[m], wF[m]
    if top_M_fundamentals and fF.size > top_M_fundamentals:
        pick = np.argsort(wF)[::-1][:top_M_fundamentals]
        fF, wF = fF[pick], wF[pick]
    fF, wF = _dedup_merge_sorted(fF, wF, min_sep_kHz=min_sep_kHz, mode="sum")

    all_f, all_w = fF.copy(), wF.copy()

    # (optional) simple pairwise combos with product weights (weak)
    if include_combos and fF.size >= 2:
        cand_f, cand_w = [], []
        lo, hi = map(float, f_range_kHz)
        for i in range(len(fF)):
            for j in range(i+1, len(fF)):
                s = fF[i] + fF[j]
                d = abs(fF[i] - fF[j])
                if lo <= s <= hi:
                    cand_f.append(s); cand_w.append(wF[i]*wF[j])
                if lo <= d <= hi:
                    cand_f.append(d); cand_w.append(wF[i]*wF[j])
        if cand_f:
            cand_f = np.asarray(cand_f, float); cand_w = np.asarray(cand_w, float)
            fcat = np.concatenate([all_f, cand_f])
            wcat = np.concatenate([all_w, cand_w])
            all_f, all_w = _dedup_merge_sorted(fcat, wcat, min_sep_kHz=min_sep_kHz, mode="sum")

    order = np.argsort(all_f)
    return {
        "fund_freq_kHz": fF,
        "fund_weight":   wF,
        "all_freq_kHz":  all_f[order],
        "all_weight":    all_w[order],
    }

# ---------- grids/boxes for the fitter ----------
def make_strict_overrides_grid(feasible_kHz: Iterable[float],
                               allow_single: bool = True,
                               max_pairs: int = 40) -> Dict[str, List[float]]:
    kHz = [float(f) for f in feasible_kHz if np.isfinite(f) and f > 0]
    if not kHz: return {}
    feas = sorted(set(f/1000.0 for f in kHz))  # cycles/µs
    singles = [{"osc_f0": f0, "osc_f1": 0.0} for f0 in feas] if allow_single else []
    pairs = []
    for i, f0 in enumerate(feas):
        for f1 in feas[:i]:
            if abs(f0 - f1) > 1e-6:
                pairs.append({"osc_f0": f0, "osc_f1": f1})
            if len(pairs) >= max_pairs: break
        if len(pairs) >= max_pairs: break
    grid = singles + pairs
    if not grid: return {}
    return {"osc_f0": [g["osc_f0"] for g in grid],
            "osc_f1": [g["osc_f1"] for g in grid]}

def make_soft_boxes_from_feasible(feasible_kHz: Iterable[float], eps_kHz: float = 8.0):
    kHz = [float(f) for f in feasible_kHz if np.isfinite(f) and f > 0]
    if not kHz: return None
    lo, hi = min(kHz) - float(eps_kHz), max(kHz) + float(eps_kHz)
    return {"osc_f0": (max(0.0, lo/1000.0), max(0.0, hi/1000.0)),
            "osc_f1": (0.0,                 max(0.0, hi/1000.0))}

# ---------- optional plotting ----------
def convolved_exp_spectrum(
    freqs_kHz: np.ndarray,
    weights: Optional[np.ndarray] = None,
    *,
    f_range_kHz: Tuple[float,float] = (1, 20000),
    npts: int = 2400,
    shape: str = "gauss",   # {"gauss","lorentz"}
    width_kHz: float = 8.0,
    title_prefix: str = "Experimental ESEEM spectrum",
    weight_caption: str = "unit weight",
    log_x: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    if freqs_kHz is None or len(freqs_kHz)==0:
        raise ValueError("No lines provided.")
    w = np.ones_like(freqs_kHz, float) if weights is None else np.asarray(weights, float)
    lo, hi = map(float, f_range_kHz)
    m = np.isfinite(freqs_kHz) & np.isfinite(w) & (freqs_kHz>=lo) & (freqs_kHz<=hi) & (w>0)
    f0, a0 = np.asarray(freqs_kHz)[m], w[m]
    if f0.size == 0:
        raise ValueError("No lines in requested range.")
    f = (np.logspace(np.log10(lo), np.log10(hi), int(npts)) if log_x else
         np.linspace(lo, hi, int(npts)))
    spec = np.zeros_like(f, float)
    if shape.lower().startswith("gauss"):
        s = float(width_kHz); norm = s*np.sqrt(2*np.pi)
        for fk, ak in zip(f0, a0): spec += ak*np.exp(-0.5*((f-fk)/s)**2)/norm
        kern = f"Gaussian (σ={width_kHz:.2f} kHz)"
    else:
        g = float(width_kHz)
        for fk, ak in zip(f0, a0): spec += ak*(g/np.pi)/((f-fk)**2 + g**2)
        kern = f"Lorentzian (γ={width_kHz:.2f} kHz)"
    plt.figure(figsize=(9,5))
    if log_x: plt.xscale("log")
    plt.plot(f, spec, lw=1.6)
    plt.xlim(lo, hi); plt.xlabel("Frequency (kHz)")
    plt.ylabel(f"Intensity (arb.)\nweights = {weight_caption}")
    plt.title(f"{title_prefix} • {kern}"); plt.grid(True, which="both", alpha=0.25)
    plt.tight_layout(); plt.show()
    return f, spec
# ============================================================================ 
