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
from analysis.sc_c13_hyperfine_sim_data_driven import (
    simulate_random_spin_echo_average,
    make_R_NV,
    B_vec_G,
)

timestamp = dm.get_time_stamp()
name = "dataset_spin_echo"
file_path = dm.get_file_path(__file__, timestamp, name)


# ------------- CONFIG -------------
class CONFIG:
    # Fit file (your big 204 NV fit)
    FILE_STEM = "2025_11_02-19_55_17-johnson_204nv_s3-003c56"

    ORIENTATIONS = [(1, 1, -1), (-1, 1, 1)]
    ORI_MODE = "random"  # "single" | "random" | "both"

    # Dataset size
    NUM_TRACES = 200_000  # total number of traces to create
    SAMPLES = 256  # points per trace (taus linspace inside simulator)
    TAU_RANGE_US = (0.0, 100.0)  # microseconds

    # Hyperfine table & physics filters
    HYPERFINE_PATH = "analysis/nv_hyperfine_coupling/nv-2.txt"
    DISTANCE_CUTOFF = 15.0  # Å or nm as in your table (whatever your file uses)
    AK_MIN_KHZ = 0  # e.g., 5.0 to keep |A∥| >= 5 kHz (if Ak_abs=True)
    AK_MAX_KHZ = 6000  # e.g., 300.0 to keep |A∥| <= 300 kHz
    AK_ABS = True  # compare |A∥| if True, signed A∥ if False
    ABUNDANCE = 0.011  # natural 13C abundance
    NUM_SPINS = None  # None = all present sites, or int to subsample strongest/top_Apar
    SELECTION = "uniform"  # "top_Apar" | "uniform" | "distance_weighted"

    # Phenomenology jitter around each NV's fitted params
    JITTER_MIX = 0.30  # fraction of cohort MAD used as Gaussian jitter
    RNG_SEED = 20251102  # global RNG seed for reproducibility

    # Sharding/IO
    OUTDIR = file_path
    SHARD_SIZE = (
        5_000  # traces per shard (NUM_TRACES must be multiple or last shard shorter)
    )
    DTYPE = np.float32  # storage dtype for traces

    # Optional noise (applied after simulation)
    ADD_GAUSS_NOISE = False
    NOISE_STD = 0.004  # ~0.4% RMS on normalized traces

    # ---- NV filtering knobs (data-driven, but safe defaults) ----
    CONTRAST_KEY = "comb_contrast"
    T2_KEY = "T2_ms"
    CONTRAST_RANGE = (0.2, 0.8)  # keep NVs with usable contrast
    T2_RANGE_MS = (0.001, 1.0)  # 3 µs – 5 ms window; tune to your system
    ROBUST_MAD_K = 4.0  # reject extreme outliers beyond K*MAD


# -----------------------------------


@dataclass
class FitCohort:
    labels: np.ndarray  # [N_nv] int
    keys: list  # list of unified_keys order
    P: np.ndarray  # [N_nv, N_keys] float with NaNs for missing
    med: np.ndarray  # [N_keys]
    mad: np.ndarray  # [N_keys]
    K: dict  # name -> index
    bounds: dict  # param bounds (name -> (lo, hi))


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
    popts = fit["popts"]

    P = np.full((len(popts), len(keys)), np.nan, float)

    def _asdict(p):
        d = {k: None for k in keys}
        if p is None:
            return d
        for k, v in zip(keys, p + [None] * (len(keys) - len(p))):
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

    med = np.array([_nanmedian(P[:, j]) for j in range(P.shape[1])])
    mad = np.array([_mad(P[:, j]) for j in range(P.shape[1])])

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
        "osc_f0": (0.0, 0.5),  # cycles/us (MHz)
        "osc_f1": (0.0, 0.5),
        "osc_phi0": (-np.pi, np.pi),
        "osc_phi1": (-np.pi, np.pi),
        # additive-carrier variant (if you use it)
        "osc_amp": (-0.5, 0.5),
    }
    return FitCohort(
        labels=labels, keys=keys, P=P, med=med, mad=mad, K=K, bounds=bounds
    )


def _clip_by_key(bounds: dict, k: str, v: float) -> float:
    if k not in bounds:
        return float(v)
    lo, hi = bounds[k]
    return float(np.minimum(hi, np.maximum(lo, v)))


def make_nv_param_drawer(
    cohort: FitCohort, jitter_mix: float, rng: np.random.Generator
):
    P, med, mad, K, bounds = cohort.P, cohort.med, cohort.mad, cohort.K, cohort.bounds

    def _nv_prior_draw(i_nv: int) -> dict:
        out = {}
        for k, j in K.items():
            mu = P[i_nv, j] if np.isfinite(P[i_nv, j]) else med[j]
            if not np.isfinite(mu):
                # last-ditch sane defaults
                defaults = dict(
                    baseline=0.6,
                    comb_contrast=0.45,
                    revival_time_us=38.0,
                    width0_us=7.0,
                    T2_ms=0.08,
                    T2_exp=1.2,
                    amp_taper_alpha=0.0,
                    width_slope=0.0,
                    revival_chirp=0.0,
                    osc_contrast=0.0,
                    osc_f0=0.0,
                    osc_f1=0.0,
                    osc_phi0=0.0,
                    osc_phi1=0.0,
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


def safe_json_dump(obj, path: Path, retries: int = 5, backoff: float = 0.1):
    """
    Atomically write JSON with retries (Windows/network-drive friendly).
    Writes to <path>.tmp then os.replace(...) to final path.
    """
    path = Path(str(path).strip())
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    last_err = None
    for a in range(retries):
        try:
            with open(tmp, "w", encoding="utf-8", newline="\n") as f:
                json.dump(obj, f, ensure_ascii=False, indent=2)
                f.flush()
                os.fsync(f.fileno())  # force to disk
            os.replace(tmp, path)  # atomic on Windows too
            return
        except OSError as e:
            last_err = e
            time.sleep(backoff * (a + 1))
    raise last_err


timestamp = dm.get_time_stamp()
name = "dataset_spin_echo"
file_path = dm.get_file_path(__file__, timestamp, name)


def add_noise(y: np.ndarray, std: float, rng: np.random.Generator) -> np.ndarray:
    if std <= 0:
        return y
    return (y + std * rng.standard_normal(y.shape)).astype(CONFIG.DTYPE, copy=False)


def _nanmad(x):
    med = np.nanmedian(x)
    return med, 1.4826 * np.nanmedian(np.abs(x - med))


def make_nv_filter_mask(
    cohort, contrast_key, t2_key, contrast_range, t2_range_ms, mad_k=0.0
):
    K = cohort.K
    P = cohort.P

    # Pull columns (may contain NaNs)
    c_idx = K.get(contrast_key, None)
    t_idx = K.get(t2_key, None)
    if c_idx is None or t_idx is None:
        raise KeyError(f"Missing keys in cohort: need '{contrast_key}' and '{t2_key}'")

    C = P[:, c_idx]  # comb_contrast
    T2 = P[:, t_idx]  # T2_ms
    # Basic finite + range masks
    m_fin = np.isfinite(C) & np.isfinite(T2)
    m_c = (C >= contrast_range[0]) & (C <= contrast_range[1])
    m_t2 = (T2 >= t2_range_ms[0]) & (T2 <= t2_range_ms[1])

    mask = m_fin & m_c & m_t2

    # Optional robust outlier rejection using MAD (separately for C and T2)
    if mad_k and mad_k > 0:
        c_med, c_mad = _nanmad(C)
        t_med, t_mad = _nanmad(T2)
        if np.isfinite(c_mad) and c_mad > 0:
            mask &= np.abs(C - c_med) <= mad_k * c_mad
        if np.isfinite(t_mad) and t_mad > 0:
            mask &= np.abs(T2 - t_med) <= mad_k * t_mad

    return mask


# before the shard loop:


def trace_from_nv(
    i_nv: int, nv_label: int, rng: np.random.Generator, nv_draw, tau_range_us, samples
) -> tuple[np.ndarray, dict]:
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
        osc_f0_MHz=fp.get("osc_f0", 0.0),  # cycles/us == MHz
        osc_f1_MHz=fp.get("osc_f1", 0.0),
        osc_phi0=fp.get("osc_phi0", 0.0),
        osc_phi1=fp.get("osc_phi1", 0.0),
    )

    # decide orientations to generate
    axes = CONFIG.ORIENTATIONS
    if CONFIG.ORI_MODE == "single":
        idxs = [0]
    elif CONFIG.ORI_MODE == "random":
        idxs = [int(rng.integers(0, len(axes)))]
    elif CONFIG.ORI_MODE == "both":
        idxs = list(range(len(axes)))
    else:
        raise ValueError(f"Unknown ORI_MODE: {CONFIG.ORI_MODE}")

    # pass R_BY_ORI down (or store globally); inside trace_from_nv:
    R_BY_ORI = [make_R_NV(ax) for ax in axes]
    # Make a per-trace salt (reproducible)
    salt_base = int(nv_label) & 0xFFFFFFFF
    out = []
    for ori_id in idxs:
        ax = axes[ori_id]
        # R_NV = make_R_NV(ax)
        R_NV = R_BY_ORI[ori_id]
        # orientation-specific salt for reproducibility
        salt = salt_base ^ (ori_id << 20) ^ int(rng.integers(0, 2**19))

        taus_us, echo, aux = simulate_random_spin_echo_average(
            hyperfine_path=CONFIG.HYPERFINE_PATH,
            tau_range_us=tau_range_us,
            num_spins=CONFIG.NUM_SPINS,
            num_realizations=1,
            distance_cutoff=CONFIG.DISTANCE_CUTOFF,
            Ak_min_kHz=CONFIG.AK_MIN_KHZ,
            Ak_max_kHz=CONFIG.AK_MAX_KHZ,
            Ak_abs=CONFIG.AK_ABS,
            R_NV=R_NV,
            fine_params=fine_params,
            abundance_fraction=CONFIG.ABUNDANCE,
            rng_seed=CONFIG.RNG_SEED,
            run_salt=salt,
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

        present_counts = aux.get("stats", {}).get("present_counts", [])
        chosen_counts = aux.get("stats", {}).get("chosen_counts", [])
        present_count = int(present_counts[0]) if present_counts else None
        chosen_count = int(chosen_counts[0]) if chosen_counts else None

        picked = aux.get("picked_ids_per_realization", [[]])
        site_ids = list(map(int, picked[0])) if picked and len(picked) else []

        meta = dict(
            nv_label=int(nv_label),
            orientation_id=int(ori_id),  # 0/1 depending on which axis we used
            nv_axis=tuple(int(v) for v in ax),  # the ⟨±1,±1,±1⟩ vector used
            B_vec_G=list(map(float, B_vec_G)),  # provenance: lab B (Gauss)
            site_ids=site_ids,  # per-trace supervision labels
            present_count=present_count,  # how many 13C were present after Bernoulli
            chosen_count=chosen_count,  # how many were chosen for the product
            fine_params={k: float(v) for k, v in fine_params.items()},
        )
        out.append((y, meta, np.asarray(taus_us, float)))

    return out


def main():
    cfg = CONFIG
    outdir = Path(cfg.OUTDIR)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load cohort fits and build NV param drawer
    cohort = load_fit_cohort(cfg.FILE_STEM)
    rng = np.random.default_rng(cfg.RNG_SEED)
    nv_draw = make_nv_param_drawer(cohort, cfg.JITTER_MIX, rng)

    # Choose which NVs to use (e.g., all that have non-NaN baseline)
    # valid = np.isfinite(cohort.P[:, cohort.K.get("baseline", 0)])
    fmask = make_nv_filter_mask(
        cohort,
        contrast_key=CONFIG.CONTRAST_KEY,
        t2_key=CONFIG.T2_KEY,
        contrast_range=CONFIG.CONTRAST_RANGE,
        t2_range_ms=CONFIG.T2_RANGE_MS,
        mad_k=CONFIG.ROBUST_MAD_K,
    )
    nv_indices = np.where(fmask)[0]
    if nv_indices.size == 0:
        raise RuntimeError(
            "No NVs passed the contrast/T2 filters — relax thresholds or inspect fits."
        )

    nv_labels = cohort.labels[nv_indices]

    # (Optional) quick stats
    print(
        f"[filter] kept {nv_indices.size}/{cohort.P.shape[0]} NVs "
        f"({100.0*nv_indices.size/cohort.P.shape[0]:.1f}%) "
        f"with {CONFIG.CONTRAST_RANGE[0]}≤{CONFIG.CONTRAST_KEY}≤{CONFIG.CONTRAST_RANGE[1]} "
        f"and {CONFIG.T2_RANGE_MS[0]}≤{CONFIG.T2_KEY}≤{CONFIG.T2_RANGE_MS[1]} ms"
    )
    nv_filter = {
        "contrast_key": CONFIG.CONTRAST_KEY,
        "contrast_range": CONFIG.CONTRAST_RANGE,
        "T2_key": CONFIG.T2_KEY,
        "T2_range_ms": CONFIG.T2_RANGE_MS,
        "robust_MAD_K": CONFIG.ROBUST_MAD_K,
        "n_kept": int(nv_indices.size),
        "n_total": int(cohort.P.shape[0]),
    }
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
        end = min(N, (s + 1) * per)
        n_this = end - start

        traces = np.zeros((n_this, cfg.SAMPLES), dtype=cfg.DTYPE)
        metas = []
        taus_us = None
        # allocate enough rows for worst-case emissions
        max_rows = n_this if CONFIG.ORI_MODE != "both" else 2 * n_this
        traces = np.empty((max_rows, cfg.SAMPLES), dtype=cfg.DTYPE)
        metas = [None] * max_rows  # pre-size list to avoid reallocation
        w = 0

        for i in range(n_this):
            i_nv = int(rr[start + i])
            lbl = int(rr_labels[start + i])

            triples = trace_from_nv(
                i_nv, lbl, rng, nv_draw, cfg.TAU_RANGE_US, cfg.SAMPLES
            )

            for y, meta, taus in triples:
                if taus.shape[0] != cfg.SAMPLES:
                    # only interpolate when necessary
                    x = np.linspace(taus.min(), taus.max(), cfg.SAMPLES, dtype=float)
                    y = np.interp(x, taus, y).astype(cfg.DTYPE, copy=False)
                    if tau_cache is None:
                        tau_cache = x.astype(np.float32)
                else:
                    if tau_cache is None:
                        tau_cache = taus.astype(np.float32)

                traces[w, :] = y
                metas[w] = meta
                w += 1

        # trim to actual number of written rows
        traces = traces[:w, :]
        metas = metas[:w]

        # for i in range(n_this):
        #     i_nv = int(rr[start + i])
        #     lbl = int(rr_labels[start + i])

        #     triples = trace_from_nv(
        #         i_nv, lbl, rng, nv_draw, cfg.TAU_RANGE_US, cfg.SAMPLES
        #     )

        #     # emit one (single/random) or two (both) traces
        #     for y, meta, taus in triples:
        #         if taus.shape[0] != cfg.SAMPLES:
        #             x = np.linspace(taus.min(), taus.max(), cfg.SAMPLES, dtype=float)
        #             y = np.interp(x, taus, y).astype(cfg.DTYPE, copy=False)
        #             taus_out = x
        #         else:
        #             taus_out = taus

        #         # grow arrays if ORI_MODE == "both"
        #         if traces.shape[0] == i:
        #             traces[i, :] = y
        #         else:
        #             # append row (rare path; if you want fixed shard size, keep ORI_MODE != "both")
        #             traces = np.vstack([traces, y[None, :]])
        #             metas.append(meta)
        #             if tau_cache is None:
        #                 tau_cache = np.asarray(taus_out, float)

        #     # only append the meta for the first emission if single/random
        #     if CONFIG.ORI_MODE != "both":
        #         metas.append(triples[0][1])
        #     if tau_cache is None:
        #         tau_cache = np.asarray(triples[0][2], float)

        # Save shard
        shard_path = outdir / f"shard_{s:04d}.npz"
        np.savez_compressed(
            shard_path,
            traces=traces,
            taus_us=tau_cache.astype(np.float32),
        )
        meta_path = outdir / f"shard_{s:04d}.json"
        safe_json_dump(
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
                nv_filter=nv_filter,
            ),
            meta_path,
        )
        print(
            f"[{s+1}/{num_shards}] wrote {shard_path.name} ({n_this} traces) & {meta_path.name}"
        )
    dt = time.time() - t0
    print(f"Done. Wrote {N} traces in {num_shards} shard(s) to {outdir} in {dt:.1f}s.")


if __name__ == "__main__":
    main()
