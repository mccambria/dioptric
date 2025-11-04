# make_spin_echo_dataset.py
# Generate a large synthetic spin-echo dataset in shards (NPZ + JSON metadata).
# - Reproducible via (rng_seed, run_salt)
# - Uses your saved fits to draw data-driven phenomenology per NV
# - Saves per-trace picked 13C site IDs for supervision
#
# Tip: Start with smaller NUM_SHARDS to verify speed/IO and then scale.

import json, math, os, sys, time, hashlib
from pathlib import Path
from dataclasses import dataclass
import numpy as np

# --- Your project utils (adjust if needed) ---
from utils import data_manager as dm
# Import your simulator (the one you pasted earlier)
# If it's in the same file, you can "from <module> import simulate_random_spin_echo_average"
from analysis.sc_c13_hyperfine_sim_data_driven import simulate_random_spin_echo_average  # <-- update path

timestamp = dm.get_time_stamp()
name   = "dataset_spin_echo"
file_path = dm.get_file_path(__file__, timestamp, name)

# ------------- CONFIG -------------
class CONFIG:
    # Fit file (your big 204 NV fit)
    FILE_STEM = "2025_11_02-19_55_17-johnson_204nv_s3-003c56"

    # Dataset size
    NUM_TRACES   = 200_000          # total number of traces to create
    SAMPLES      = 256              # points per trace (taus linspace inside simulator)
    TAU_RANGE_US = (0.0, 100.0)     # microseconds

    # Hyperfine table & physics filters
    HYPERFINE_PATH = "analysis/nv_hyperfine_coupling/nv-2.txt"
    DISTANCE_CUTOFF = 8.0           # Å or nm as in your table (whatever your file uses)
    AK_MIN_KHZ = 0               # e.g., 5.0 to keep |A∥| >= 5 kHz (if Ak_abs=True)
    AK_MAX_KHZ = 600               # e.g., 300.0 to keep |A∥| <= 300 kHz
    AK_ABS     = True               # compare |A∥| if True, signed A∥ if False
    ABUNDANCE  = 0.011              # natural 13C abundance
    NUM_SPINS  = None               # None = all present sites, or int to subsample strongest/top_Apar
    SELECTION  = "uniform"         # "top_Apar" | "uniform" | "distance_weighted"

    # Phenomenology jitter around each NV's fitted params
    JITTER_MIX = 0.30               # fraction of cohort MAD used as Gaussian jitter
    RNG_SEED   = 20251102           # global RNG seed for reproducibility

    # Sharding/IO
    OUTDIR       = file_path
    SHARD_SIZE   = 5_000            # traces per shard (NUM_TRACES must be multiple or last shard shorter)
    DTYPE        = np.float32       # storage dtype for traces

    # Optional noise (applied after simulation)
    ADD_GAUSS_NOISE = True
    NOISE_STD      = 0.004          # ~0.4% RMS on normalized traces
# -----------------------------------

@dataclass
class FitCohort:
    labels: np.ndarray      # [N_nv] int
    keys:   list            # list of unified_keys order
    P:      np.ndarray      # [N_nv, N_keys] float with NaNs for missing
    med:    np.ndarray      # [N_keys]
    mad:    np.ndarray      # [N_keys]
    K:      dict            # name -> index
    bounds: dict            # param bounds (name -> (lo, hi))

def _nanmedian(x): 
    v = np.nanmedian(x)
    return v if np.isfinite(v) else np.nan

def _mad(x):
    med = _nanmedian(x)
    if not np.isfinite(med): 
        return np.nan
    return _nanmedian(np.abs(x - med)) * 1.4826

def load_fit_cohort(file_stem: str) -> FitCohort:
    fit = dm.get_raw_data(file_stem=file_stem)

    keys = fit["unified_keys"]
    labels = np.array(list(map(int, fit["nv_labels"])), int)
    popts  = fit["popts"]

    P = np.full((len(popts), len(keys)), np.nan, float)

    def _asdict(p):
        d = {k: None for k in keys}
        if p is None:
            return d
        for k, v in zip(keys, p + [None]*(len(keys)-len(p))):
            d[k] = v
        return d

    for i, p in enumerate(popts):
        d = _asdict(p)
        for j, k in enumerate(keys):
            v = d[k]
            if v is None: 
                continue
            try:
                P[i, j] = float(v)
            except Exception:
                pass

    med = np.array([_nanmedian(P[:,j]) for j in range(P.shape[1])])
    mad = np.array([_mad(P[:,j])       for j in range(P.shape[1])])

    K = {k: j for j, k in enumerate(keys)}
    bounds = {
        "baseline": (0.0, 1.2),
        "comb_contrast": (0.0, 1.1),
        "revival_time_us": (25.0, 55.0),
        "width0_us": (1.0, 25.0),
        "T2_ms": (1e-3, 2e3),
        "T2_exp": (0.6, 4.0),
        "amp_taper_alpha": (0.0, 2.0),
        "width_slope": (-0.2, 0.2),
        "revival_chirp": (-0.06, 0.06),
        # two-sine^2 variant params (if present in unified_keys)
        "osc_contrast": (0.0, 1.0),
        "osc_f0": (0.0, 0.5),      # cycles/us (MHz)
        "osc_f1": (0.0, 0.5),
        "osc_phi0": (-np.pi, np.pi),
        "osc_phi1": (-np.pi, np.pi),
        # additive-carrier variant (if you use it)
        "osc_amp": (-0.5, 0.5),
    }
    return FitCohort(labels=labels, keys=keys, P=P, med=med, mad=mad, K=K, bounds=bounds)

def _clip_by_key(bounds: dict, k: str, v: float) -> float:
    if k not in bounds:
        return float(v)
    lo, hi = bounds[k]
    return float(np.minimum(hi, np.maximum(lo, v)))

def make_nv_param_drawer(cohort: FitCohort, jitter_mix: float, rng: np.random.Generator):
    P, med, mad, K, bounds = cohort.P, cohort.med, cohort.mad, cohort.K, cohort.bounds

    def _nv_prior_draw(i_nv: int) -> dict:
        out = {}
        for k, j in K.items():
            mu = P[i_nv, j] if np.isfinite(P[i_nv, j]) else med[j]
            if not np.isfinite(mu):
                # last-ditch sane defaults
                defaults = dict(
                    baseline=0.6, comb_contrast=0.45, revival_time_us=38.0, width0_us=7.0,
                    T2_ms=0.08, T2_exp=1.2,
                    amp_taper_alpha=0.0, width_slope=0.0, revival_chirp=0.0,
                    osc_contrast=0.0, osc_f0=0.0, osc_f1=0.0, osc_phi0=0.0, osc_phi1=0.0,
                    osc_amp=0.0,
                )
                mu = defaults.get(k, 0.0)
            sig = mad[j]
            if not np.isfinite(sig) or sig == 0.0:
                sig = 1e-3
            draw = mu + jitter_mix * sig * rng.standard_normal()
            out[k] = _clip_by_key(bounds, k, draw)
        return out
    return _nv_prior_draw

def add_noise(y: np.ndarray, std: float, rng: np.random.Generator) -> np.ndarray:
    if std <= 0:
        return y
    return (y + std * rng.standard_normal(y.shape)).astype(CONFIG.DTYPE, copy=False)

def trace_from_nv(i_nv: int, nv_label: int, rng: np.random.Generator,
                  nv_draw, tau_range_us, samples) -> tuple[np.ndarray, dict]:
    """
    Make a single trace using one phenomenology draw and one stochastic 13C realization.
    Returns (trace[T], meta_dict).
    """
    # Draw phenomenology around this NV’s fit
    fp = nv_draw(i_nv)

    # Map to simulator's fine_params (two-sine^2 "MOD" version)
    fine_params = dict(
        baseline=fp.get("baseline", 0.6),
        comb_contrast=fp.get("comb_contrast", 0.45),
        revival_time=fp.get("revival_time_us", 38.0),
        width0_us=fp.get("width0_us", 7.0),
        T2_ms=fp.get("T2_ms", 0.08),
        T2_exp=fp.get("T2_exp", 1.2),
        amp_taper_alpha=fp.get("amp_taper_alpha", 0.0),
        width_slope=fp.get("width_slope", 0.0),
        revival_chirp=fp.get("revival_chirp", 0.0),
        # Two-sine^2 contrast variant:
        osc_contrast=fp.get("osc_contrast", 0.0),
        osc_f0_MHz=fp.get("osc_f0", 0.0),   # cycles/us == MHz
        osc_f1_MHz=fp.get("osc_f1", 0.0),
        osc_phi0=fp.get("osc_phi0", 0.0),
        osc_phi1=fp.get("osc_phi1", 0.0),
    )

    # Make a per-trace salt (reproducible)
    salt = (int(nv_label) & 0xFFFFFFFF) ^ int(rng.integers(0, 2**31-1))

    taus_us, echo, aux = simulate_random_spin_echo_average(
        hyperfine_path=CONFIG.HYPERFINE_PATH,
        tau_range_us=tau_range_us,
        num_spins=CONFIG.NUM_SPINS,
        num_realizations=1,                  # one realization per trace
        distance_cutoff=CONFIG.DISTANCE_CUTOFF,
        Ak_min_kHz=CONFIG.AK_MIN_KHZ,
        Ak_max_kHz=CONFIG.AK_MAX_KHZ,
        Ak_abs=CONFIG.AK_ABS,
        R_NV=np.eye(3),
        fine_params=fine_params,
        abundance_fraction=CONFIG.ABUNDANCE,
        rng_seed=CONFIG.RNG_SEED,            # global
        run_salt=salt,                       # per-trace salt
        randomize_positions=False,
        selection_mode=CONFIG.SELECTION,
        ensure_unique_across_realizations=False,
        annotate_from_realization=0,
        keep_nv_orientation=True,
        fixed_site_ids=None,
        fixed_presence_mask=None,
        reuse_present_mask=True,
    )

    y = np.asarray(echo, float)
    if CONFIG.ADD_GAUSS_NOISE:
        y = add_noise(y, CONFIG.NOISE_STD, rng)
    else:
        y = y.astype(CONFIG.DTYPE, copy=False)

    picked = aux.get("picked_ids_per_realization", [[]])
    site_ids = list(map(int, picked[0])) if picked and len(picked) else []

    meta = dict(
        nv_label=int(nv_label),
        fine_params={k: float(v) for k, v in fine_params.items()},
        site_ids=site_ids,
    )
    return y, meta, np.asarray(taus_us, float)

def main():
    cfg = CONFIG
    outdir = Path(cfg.OUTDIR)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load cohort fits and build NV param drawer
    cohort = load_fit_cohort(cfg.FILE_STEM)
    rng = np.random.default_rng(cfg.RNG_SEED)
    nv_draw = make_nv_param_drawer(cohort, cfg.JITTER_MIX, rng)

    # Choose which NVs to use (e.g., all that have non-NaN baseline)
    valid = np.isfinite(cohort.P[:, cohort.K.get("baseline", 0)])
    nv_indices = np.where(valid)[0]
    if nv_indices.size == 0:
        raise RuntimeError("No valid NVs found in the fit cohort.")
    nv_labels = cohort.labels[nv_indices]

    # Work out sharding
    N = cfg.NUM_TRACES
    per = cfg.SHARD_SIZE
    num_shards = math.ceil(N / per)
    print(f"Will write {N} traces in {num_shards} shard(s) of ~{per} each to {outdir}")

    # Deterministic NV assignment: round-robin through valid NVs
    rr = np.resize(nv_indices, N)
    rr_labels = np.resize(nv_labels, N)

    # Generate
    tau_cache = None
    t0 = time.time()
    for s in range(num_shards):
        start = s * per
        end   = min(N, (s+1) * per)
        n_this = end - start

        traces = np.zeros((n_this, cfg.SAMPLES), dtype=cfg.DTYPE)
        metas  = []
        taus_us = None

        for i in range(n_this):
            i_nv = int(rr[start + i])
            lbl  = int(rr_labels[start + i])
            y, meta, taus = trace_from_nv(i_nv, lbl, rng, nv_draw, cfg.TAU_RANGE_US, cfg.SAMPLES)

            # resample to desired SAMPLES if simulator used a different default
            if taus.shape[0] != cfg.SAMPLES:
                # linear resample
                x = np.linspace(taus.min(), taus.max(), cfg.SAMPLES, dtype=float)
                y = np.interp(x, taus, y).astype(cfg.DTYPE, copy=False)
                taus_out = x
            else:
                taus_out = taus

            traces[i, :] = y
            metas.append(meta)

            if tau_cache is None:
                tau_cache = np.asarray(taus_out, float)

        # Save shard
        shard_path = outdir / f"shard_{s:04d}.npz"
        np.savez_compressed(
            shard_path,
            traces=traces,
            taus_us=tau_cache.astype(np.float32),
        )

        meta_path = outdir / f"shard_{s:04d}.json"
        with open(meta_path, "w") as f:
            json.dump(
                dict(
                    file_stem=cfg.FILE_STEM,
                    hyperfine_path=cfg.HYPERFINE_PATH,
                    distance_cutoff=cfg.DISTANCE_CUTOFF,
                    ak_min_kHz=cfg.AK_MIN_KHZ,
                    ak_max_kHz=cfg.AK_MAX_KHZ,
                    ak_abs=cfg.AK_ABS,
                    abundance=cfg.ABUNDANCE,
                    num_spins=cfg.NUM_SPINS,
                    selection=cfg.SELECTION,
                    jitter_mix=cfg.JITTER_MIX,
                    rng_seed=cfg.RNG_SEED,
                    add_gauss_noise=cfg.ADD_GAUSS_NOISE,
                    noise_std=cfg.NOISE_STD,
                    metas=metas,
                ),
                f,
                indent=2,
            )

        print(f"[{s+1}/{num_shards}] wrote {shard_path.name} ({n_this} traces)")

    dt = time.time() - t0
    print(f"Done. Wrote {N} traces in {num_shards} shard(s) to {outdir} in {dt:.1f}s.")

if __name__ == "__main__":
    main()
