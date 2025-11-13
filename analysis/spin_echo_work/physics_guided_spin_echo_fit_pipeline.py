"""
Spin-echo fitting pipeline (physics-guided priors + FFT seeds)
----------------------------------------------------------------
Drop this file next to your analysis code and run the entry-point
function `fit_spin_echo_dataset(...)`.

It assumes you already have:
  • norm_counts:   shape (N_NV, N_time)
  • norm_counts_ste: same shape (uncertainties)
  • total_evolution_times: 1D array of τ in microseconds
  • nv_list:       list of NV labels (ints or strings)
  • file_stems:    list of dataset source IDs (for provenance)
  • your fitter:   run_with_amp_and_freq_sweeps(...) imported/defined
  • your model:    fine_decay / fine_decay_fixed_revival

Optional: A catalog JSON with per-site expected ESEEM lines (f±, κ).
If provided, we build a small, diversified set of (f0,f1) candidates
(cycles/μs) to pass as priors to the fitter.

This module is careful to:
  • Bound frequency search by sampling-derived Nyquist limits
  • Use conservative amp windows first, then expand as needed
  • Serialize results with a stable schema (JSON)
  • Plot quick sanity visuals per NV (optional)
"""
from __future__ import annotations
import os, re, json, hashlib
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional, Tuple, List, Dict
from utils import data_manager as dm

import numpy as np
import matplotlib.pyplot as plt

# --------- Imports expected to exist in your environment ---------------------
# - fine_decay / fine_decay_fixed_revival
# - run_with_amp_and_freq_sweeps
# - eseem_utils: optional physics-guided candidate builders

try:
    from eseem_utils import (
        load_catalog,
        select_records,
        build_single_freq_candidates,
        build_pair_freq_candidates,
        plot_candidate_pairs,
    )
except Exception:
    # Lightweight fallbacks if eseem_utils isn't present; you can remove this.
    def load_catalog(path):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data['records'] if isinstance(data, dict) and 'records' in data else data
    def select_records(recs, fmin_kHz=150, fmax_kHz=20000, orientations=None):
        def ok_o(r):
            return True if orientations is None else r.get('orientation') in set(map(tuple, orientations))
        out=[]
        for r in recs:
            fp=float(r.get('f_plus_kHz', np.nan)); fm=float(r.get('f_minus_kHz', np.nan))
            if not ok_o(r):
                continue
            if (np.isfinite(fp) and fmin_kHz<=fp<=fmax_kHz) or (np.isfinite(fm) and fmin_kHz<=fm<=fmax_kHz):
                out.append(r)
        return out
    def build_single_freq_candidates(recs, p_occ=0.011, orientations=None, f_range_kHz=(150,20000), n_keep=24, weighted=True):
        # naive top-frequency chooser
        F=[]
        for r in select_records(recs, *f_range_kHz, orientations=orientations):
            for k in ('f_plus_kHz','f_minus_kHz'):
                f=r.get(k, None)
                if f is None: continue
                F.append(float(f))
        if not F: return []
        F=sorted(set([float(f) for f in F if np.isfinite(f) and f>0]))
        return [f*1e-3 for f in F[:n_keep]]
    def build_pair_freq_candidates(recs, p_occ=0.011, orientations=None, f_range_kHz=(150,20000), n_keep_each=16, min_sep_cyc_per_us=0.01, weighted=True):
        singles = build_single_freq_candidates(recs, p_occ, orientations, f_range_kHz, n_keep_each, weighted)
        singles = sorted(set(float(x) for x in singles))
        pairs=[]
        for i,f0 in enumerate(singles):
            for f1 in singles[:i]:
                if abs(f0-f1)>=min_sep_cyc_per_us:
                    pairs.append((f0,f1))
            pairs.append((f0,0.0))
        return pairs[:40]
    def plot_candidate_pairs(pairs: List[Tuple[float,float]], title='Candidate (f0,f1)'):
        if not pairs: return
        arr=np.asarray(pairs,float)
        plt.figure(figsize=(6,5)); plt.plot(arr[:,0],arr[:,1],'.'); plt.xlabel('f0 (cyc/μs)'); plt.ylabel('f1 (cyc/μs)'); plt.title(title); plt.grid(alpha=0.3); plt.tight_layout(); plt.show()

# We also expect your fitter to be importable from your analysis code
try:
    from fitter_module_for_spin_echo import run_with_amp_and_freq_sweeps  # rename to your path
except Exception:
    # If already in the global namespace (e.g., Jupyter), this import may be unnecessary.
    pass

# -------------------------- Utilities ---------------------------------------

def _infer_sampling_band(times_us: np.ndarray, margin: float = 0.05) -> Tuple[float, float]:
    t = np.asarray(times_us, float)
    t = t[np.isfinite(t)]
    if t.size < 2:
        return 0.0, 1.0
    span = float(t.max() - t.min())
    if span <= 0:
        return 0.0, 1.0
    diffs = np.diff(np.unique(t))
    dt_min = float(np.nanmin(diffs)) if diffs.size else span
    fmin = 1.0 / span
    fmax = 0.5 / max(1e-9, dt_min)
    # slack
    return (max(0.0, (1 - margin) * fmin), (1 + margin) * fmax)


def _stable_name(file_stems: List[str], nv_labels: List[int], suffix: str) -> str:
    tokens = []
    for s in file_stems:
        m = re.search(r"-([A-Za-z0-9]+)-nv", s)
        if m:
            tokens.append(m.group(1))
    sample = max(set(tokens), key=tokens.count) if tokens else "sample"
    srcsig = f"s{len(file_stems)}-" + hashlib.sha1("|".join(file_stems).encode()).hexdigest()[:6]
    return f"{sample}_{len(nv_labels)}nv_{srcsig}_{suffix}"


@dataclass
class FitConfig:
    # Model toggles
    use_fixed_revival: bool = False
    enable_extras: bool = True
    fixed_rev_time_us: float = 37.6

    # Amplitude windows to sweep for osc_amp
    amp_bound_grid: Tuple[Tuple[float, float], ...] = ((-0.6, 0.6), (-1.0, 1.0), (-2.0, 2.0))

    # Frequency constraints (cycles/μs)
    freq_seed_band: Optional[Tuple[float, float]] = None   # if None, inferred from sampling
    freq_bound_boxes: Optional[Dict[str, Tuple[float, float]]] = None  # {"osc_f0":(lo,hi), "osc_f1":(lo,hi)}
    freq_seed_n_peaks: int = 6
    seed_include_harmonics: bool = True

    # Physics-guided priors from catalog (optional)
    catalog_path: Optional[str] = None
    orientations: Optional[List[Tuple[int,int,int]]] = None
    p_occ: float = 0.011
    f_range_kHz: Tuple[float, float] = (150.0, 20000.0)
    n_keep_each: int = 16
    min_sep_cyc_per_us: float = 0.01

    # Fitter verbosity
    verbose: bool = True


@dataclass
class FitOutputs:
    timestamp: str
    dataset_ids: List[str]
    default_rev_us: float
    run_flags: Dict[str, bool]
    nv_labels: List[int]
    times_us: List[float]
    popts: List[Optional[List[float]]]
    pcovs: List[Optional[List[List[float]]]]
    red_chi2: List[Optional[float]]
    fit_fn_names: List[Optional[str]]
    unified_keys: List[str]


# ------------------------- Prior builders -----------------------------------

def build_prior_overrides_from_catalog(cfg: FitConfig) -> Optional[Dict[str, List[float]]]:
    """Return an `extra_overrides_grid` dict or None.

    NOTE: run_with_amp_and_freq_sweeps() accepts a grid over parameter lists
    and forms the Cartesian product. To avoid huge grids, we:
      1) build a compact set of pairs
      2) derive a compact set of singles (used when f1=0)
      3) return small lists; yes, it will cross-combine, but lists are short
    """
    if not cfg.catalog_path:
        return None
    try:
        recs = select_records(
            load_catalog(cfg.catalog_path),
            fmin_kHz=cfg.f_range_kHz[0],
            fmax_kHz=cfg.f_range_kHz[1],
            orientations=cfg.orientations,
        )
    except Exception as e:
        print(f"[WARN] Could not load catalog '{cfg.catalog_path}': {e}")
        return None

    pairs = build_pair_freq_candidates(
        recs,
        p_occ=cfg.p_occ,
        orientations=cfg.orientations,
        f_range_kHz=cfg.f_range_kHz,
        n_keep_each=cfg.n_keep_each,
        min_sep_cyc_per_us=cfg.min_sep_cyc_per_us,
        weighted=True,
    )

    # Keep a very small, diverse subset to limit the cross-product explosion
    pairs = pairs[:20]

    singles = sorted({p[0] for p in pairs} | {p[1] for p in pairs if p[1] > 0})
    singles = singles[:12]

    # Optional preview
    if cfg.verbose and pairs:
        plot_candidate_pairs(pairs, title="Physics-guided candidate pairs (cycles/μs)")

    # Grid: these will be cross-combined (small!)
    grid = {
        "osc_f0": singles[:6],  # handful for f0
        "osc_f1": [0.0] + singles[:6],  # allow single-tone (f1=0)
        # You can optionally seed phases:
        # "osc_phi0": [0.0, np.pi/2],
        # "osc_phi1": [0.0],
    }
    return grid


# --------------------------- Main entry-point -------------------------------

def fit_spin_echo_dataset(
    nv_list: List[int],
    norm_counts: np.ndarray,
    norm_counts_ste: np.ndarray,
    total_evolution_times: np.ndarray,
    file_stems: List[str],
    *,
    default_rev_us: float = 37.2,
    nv_inds: Optional[List[int]] = None,
    cfg: Optional[FitConfig] = None,
    save_dir: Optional[str] = None,
    make_plots: bool = False,
) -> Tuple[FitOutputs, str]:
    """Run the full fit (amp+freq sweeps), optionally with catalog priors.

    Returns (FitOutputs dataclass, saved_json_path)
    """
    if cfg is None:
        cfg = FitConfig()

    # Sanity on arrays
    norm_counts = np.asarray(norm_counts, float)
    norm_counts_ste = np.asarray(norm_counts_ste, float)
    total_evolution_times = np.asarray(total_evolution_times, float)

    if nv_inds is None:
        nv_inds = list(range(len(nv_list)))

    # Sampling-informed band if not provided
    band = cfg.freq_seed_band
    if band is None:
        band = _infer_sampling_band(total_evolution_times, margin=0.05)
        if cfg.verbose:
            print(f"[band] inferred from sampling: {band[0]:.6g}–{band[1]:.6g} cycles/μs")

    # Default frequency boxes: respect Nyquist
    freq_boxes = cfg.freq_bound_boxes or {"osc_f0": (max(0.001, band[0]), band[1]),
                                          "osc_f1": (0.0, band[1])}

    # Optional physics-guided prior grid
    prior_grid = build_prior_overrides_from_catalog(cfg)

    # Run the fitter
    print("=== Spin-echo fits starting ===")
    popts, pcovs, chis, fit_fns, fit_nv_labels = run_with_amp_and_freq_sweeps(
        nv_list,
        norm_counts,
        norm_counts_ste,
        total_evolution_times,
        nv_inds=nv_inds,
        amp_bound_grid=cfg.amp_bound_grid,
        freq_bound_boxes=freq_boxes,
        freq_seed_band=band,
        freq_seed_n_peaks=cfg.freq_seed_n_peaks,
        seed_include_harmonics=cfg.seed_include_harmonics,
        extra_overrides_grid=prior_grid,  # None if no catalog provided
        use_fixed_revival=cfg.use_fixed_revival,
        enable_extras=cfg.enable_extras,
        fixed_rev_time=cfg.fixed_rev_time_us,
        verbose=cfg.verbose,
    )

    # Package outputs (stable schema)
    unified_keys = [
        "baseline","comb_contrast","revival_time_us","width0_us","T2_ms","T2_exp",
        "amp_taper_alpha","width_slope","revival_chirp",
        "osc_amp","osc_f0","osc_f1","osc_phi0","osc_phi1"
    ]

    out = FitOutputs(
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        dataset_ids=list(map(str, file_stems)),
        default_rev_us=float(default_rev_us),
        run_flags={
            "use_fixed_revival": bool(cfg.use_fixed_revival),
            "enable_extras": bool(cfg.enable_extras),
        },
        nv_labels=list(map(int, fit_nv_labels)),
        times_us=np.asarray(total_evolution_times, float).tolist(),
        popts=[(p.tolist() if p is not None else None) for p in popts],
        pcovs=[(c.tolist() if c is not None else None) for c in pcovs],
        red_chi2=[(float(x) if x is not None else None) for x in chis],
        fit_fn_names=[(fn.__name__ if fn is not None else None) for fn in fit_fns],
        unified_keys=unified_keys,
    )

    # Save JSON
    tokens = []
    for s in file_stems:
        m = re.search(r"-([A-Za-z0-9]+)-nv", s)  # e.g. "...-johnson-nv0_..."
        if m: tokens.append(m.group(1))
    sample = max(set(tokens), key=tokens.count) if tokens else "sample"
    srcsig = f"s{len(file_stems)}-{hashlib.sha1('|'.join(file_stems).encode()).hexdigest()[:6]}"
    # --- tiny signature of the source list ---
    name   = f"{sample}_{len(fit_nv_labels)}nv_{srcsig}"
    # name   = f"{sample}_{len(fit_nv_labels)}nv_{date}_{rev}_{model}_{sweep}_{srcsig}"
    # print(name)
    timestamp = dm.get_time_stamp()
    file_path = dm.get_file_path(__file__, timestamp, name)
    dm.save_raw_data(out, file_path)
    save_dir = save_dir or str(Path.cwd() / "spin_echo_fits")

    print(f"[SAVE] {file_path}")

    # Optional quick plots: residual snapshots for a few NVs
    if make_plots:
        _quick_residual_plots(
            nv_list, norm_counts, norm_counts_ste, total_evolution_times,
            popts, fit_fns, out.nv_labels, nmax=6
        )

    return out, file_path


# -------------------------- Plot helpers ------------------------------------

def _quick_residual_plots(nv_list, y, ye, t_us, popts, fit_fns, labels, nmax=6):
    idxs = list(range(min(nmax, len(labels))))
    for i in idxs:
        p = popts[i]; fn = fit_fns[i]
        if p is None or fn is None:
            continue
        yy = y[i]; ee = np.maximum(1e-9, ye[i])
        yfit = fn(t_us, *p)
        resid = (yy - yfit) / ee
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
        axes[0].plot(t_us, yy, '.', ms=3, alpha=0.8)
        axes[0].plot(t_us, yfit, '-', lw=1.2)
        axes[0].set_xlabel('τ (μs)'); axes[0].set_ylabel('Norm. counts')
        axes[0].set_title(f'NV {labels[i]}: fit')
        axes[0].grid(alpha=0.3)
        axes[1].plot(t_us, resid, '.', ms=3)
        axes[1].axhline(0, color='k', lw=0.8)
        axes[1].set_xlabel('τ (μs)'); axes[1].set_ylabel('resid / σ')
        axes[1].set_title('Residuals')
        axes[1].grid(alpha=0.3)
        plt.tight_layout(); plt.show()


# --------------------------- Example usage ----------------------------------
if __name__ == "__main__":
    # Example placeholders — replace with your real arrays/objects
    # norm_counts, norm_counts_ste, total_evolution_times, nv_list, file_stems
    file_stems = "2025_11_11-01_15_45-johnson_204nv_s6-6d8f5c" #dataset2 + dataset3
    data = dm.get_raw_data(file_stem=file_stems)
    nv_list = data["nv_list"]
    norm_counts = np.array(data["norm_counts"])
    norm_counts_ste = np.array(data["norm_counts_ste"])
    total_evolution_times = np.array(data["total_evolution_times"])
    
    cfg = FitConfig(
        use_fixed_revival=False,
        enable_extras=True,
        fixed_rev_time_us=37.6,
        # optional physics-guided priors:
        catalog_path="analysis/spin_echo_work/essem_freq_catalog_22A.json",
        orientations=None,          # or [(1,1,1), (1,-1,1), ...]
        p_occ=0.011,                # 1.1% 13C
        f_range_kHz=(150, 20000),
        n_keep_each=16,
        min_sep_cyc_per_us=0.01,
        verbose=True,
    )

    out, path = fit_spin_echo_dataset(
        nv_list,                    # your labels
        norm_counts,                # shape (N_NV, N_t)
        norm_counts_ste,            # same shape
        total_evolution_times,      # τ in μs
        file_stems,                 # provenance list
        default_rev_us=37.2,
        nv_inds=None,               # or a subset list
        cfg=cfg,
        save_dir="spin_echo_fits",
        make_plots=True,
    )
    print("Saved:", path)
    # raise SystemExit(
    #     "Import this module and call fit_spin_echo_dataset(...) from your notebook/script."
    # )
