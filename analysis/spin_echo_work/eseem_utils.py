# ===== Expected-spectrum helpers (discrete, experiment-ready) =================
import numpy as np
import matplotlib.pyplot as plt
import json, csv, numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# --- Load your precomputed catalog (from build_essem_catalog) ---
def load_catalog(path_json):
    with open(path_json, "r") as f:
        return json.load(f)

# --- Helpers: window + pick orientations ---
def select_records(recs, fmin_kHz=150.0, fmax_kHz=20000.0, orientations=None):
    out = []
    ori_set = {tuple(o) for o in orientations} if orientations else None
    for r in recs:
        fm_k = r["f_minus_Hz"]/1e3
        fp_k = r["f_plus_Hz"] /1e3
        if not (np.isfinite(fm_k) and np.isfinite(fp_k)): 
            continue
        if not (fmin_kHz <= fm_k <= fmax_kHz and fmin_kHz <= fp_k <= fmax_kHz):
            continue
        if ori_set and tuple(r["orientation"]) not in ori_set:
            continue
        out.append(r)
    return out


def fundamentals_from_recs(recs, p_occ=0.011, use_weights=True,
                           f_range_kHz=(150.0, 20000.0)):
    """Return fundamentals {f- , f+} and weights for all sites inside range."""
    fk, wk = [], []
    fmin, fmax = map(float, f_range_kHz)
    for r in recs:
        w = p_occ * (float(r.get("amp_weight", 1.0)) if use_weights else 1.0)
        f_minus = float(r["f_minus_Hz"]) / 1e3
        f_plus  = float(r["f_plus_Hz"])  / 1e3
        if np.isfinite(f_minus) and fmin <= f_minus <= fmax and w > 0:
            fk.append(f_minus); wk.append(w)
        if np.isfinite(f_plus) and fmin <= f_plus <= fmax and w > 0:
            fk.append(f_plus);  wk.append(w)
    fk = np.asarray(fk, float); wk = np.asarray(wk, float)
    return fk, wk

def _dedup_merge_sorted(freqs_kHz, weights=None, min_sep_kHz=5.0, mode="sum"):
    """
    Collapse near-duplicates: if |fi - fj| < min_sep, merge.
    mode="sum" sums weights of colliding lines; mode="max" keeps the max weight.
    """
    if weights is None: weights = np.ones_like(freqs_kHz, float)
    order = np.argsort(freqs_kHz)
    f = freqs_kHz[order]; w = weights[order]

    out_f, out_w = [], []
    for fi, wi in zip(f, w):
        if not out_f:
            out_f.append(fi); out_w.append(wi); continue
        if abs(fi - out_f[-1]) < float(min_sep_kHz):
            if mode == "sum": out_w[-1] += wi
            else:             out_w[-1]  = max(out_w[-1], wi)
        else:
            out_f.append(fi); out_w.append(wi)
    return np.asarray(out_f), np.asarray(out_w)

def combos_from_fundamentals(freqs_kHz, weights=None, f_range_kHz=(150,20000),
                             include=('sum','diff'), min_sep_kHz=5.0):
    """
    Build pairwise combinations from fundamentals.
    Weight heuristic: w_ij = w_i * w_j (cross terms are weaker).
    include: any of {"sum","diff"}.
    """
    fmin, fmax = map(float, f_range_kHz)
    if weights is None: weights = np.ones_like(freqs_kHz, float)
    f, w = np.asarray(freqs_kHz, float), np.asarray(weights, float)

    cand_f, cand_w = [], []
    n = len(f)
    for i in range(n):
        for j in range(i+1, n):
            if 'sum' in include:
                s = f[i] + f[j]
                if fmin <= s <= fmax:
                    cand_f.append(s); cand_w.append(w[i]*w[j])
            if 'diff' in include:
                d = abs(f[i] - f[j])
                if d >= fmin and d <= fmax:
                    cand_f.append(d); cand_w.append(w[i]*w[j])

    if not cand_f:
        return np.array([],float), np.array([],float)
    cand_f = np.asarray(cand_f, float); cand_w = np.asarray(cand_w, float)
    return _dedup_merge_sorted(cand_f, cand_w, min_sep_kHz=min_sep_kHz, mode="sum")

def build_feasible_freqs_from_recs(recs,
                                   p_occ=0.011,
                                   f_range_kHz=(150,20000),
                                   include_combos=False,
                                   min_sep_kHz=5.0,
                                   top_M_fundamentals=60,
                                   weight_floor=0.0,
                                   use_weights=True):
    """
    Returns a dict:
      {
        "fund_freq_kHz":  array([...]),
        "fund_weight":    array([...]),
        "all_freq_kHz":   array([...]),   # fundamentals (+ combos if requested), deduped+sorted
        "all_weight":     array([...]),
      }
    """
    # Fundamentals
    fF, wF = fundamentals_from_recs(recs, p_occ=p_occ, use_weights=use_weights,
                                    f_range_kHz=f_range_kHz)
    # Filter by weight floor & keep top M by weight to control combinatorics
    if fF.size:
        m = (wF >= float(weight_floor))
        fF, wF = fF[m], wF[m]
        if fF.size > top_M_fundamentals:
            order = np.argsort(wF)[::-1][:top_M_fundamentals]
            fF, wF = fF[order], wF[order]
        fF, wF = _dedup_merge_sorted(fF, wF, min_sep_kHz=min_sep_kHz, mode="sum")

    # Optionally add pairwise combos
    all_f, all_w = fF.copy(), wF.copy()
    if include_combos and fF.size >= 2:
        fC, wC = combos_from_fundamentals(fF, wF, f_range_kHz=f_range_kHz,
                                          include=('sum','diff'),
                                          min_sep_kHz=min_sep_kHz)
        if fC.size:
            # concatenate and dedup again
            fcat = np.concatenate([fF, fC]); wcat = np.concatenate([wF, wC])
            all_f, all_w = _dedup_merge_sorted(fcat, wcat, min_sep_kHz=min_sep_kHz, mode="sum")

    # Final sort
    order = np.argsort(all_f)
    all_f, all_w = all_f[order], all_w[order]

    return {
        "fund_freq_kHz": fF, "fund_weight": wF,
        "all_freq_kHz":  all_f, "all_weight": all_w,
    }


def kHz_to_cycles_per_us(freq_kHz):
    """cycles/µs == MHz; kHz → cycles/µs conversion = kHz / 1000."""
    f = np.asarray(freq_kHz, float)
    return f / 1000.0

def build_override_grid_for_fitter(freq_kHz, allow_single=True):
    """
    Produce an extra_overrides_grid suitable for run_with_amp_and_freq_sweeps(...):
      {"osc_f0": [...], "osc_f1": [0.0] + [...]  (if allow_single)}
    Input in kHz; output in cycles/µs.
    """
    f_cyc = kHz_to_cycles_per_us(freq_kHz)
    grid = {"osc_f0": list(map(float, f_cyc))}
    if allow_single:
        grid["osc_f1"] = [0.0] + list(map(float, f_cyc))
    else:
        grid["osc_f1"] = list(map(float, f_cyc))
    return grid
# ============================================================================