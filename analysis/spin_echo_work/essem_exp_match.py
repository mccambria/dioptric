import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from utils import data_manager as dm
from utils import kplotlib as kpl
from analysis.sc_c13_hyperfine_sim_data_driven import (
    read_hyperfine_table_safe,
    B_vec_T,   # your lab field (Tesla)
)
from analysis.spin_echo_work.echo_fit_models import fine_decay, fine_decay_fixed_revival
from analysis.spin_echo_work.echo_plot_helpers import (
    extract_T2_freqs_and_errors, 
    params_to_dict,
    plot_echo_with_sites,
    ) 

# ---------------------------------------------------------------------
# CONFIG / PATHS
# ---------------------------------------------------------------------

HYPERFINE_PATH = "analysis/nv_hyperfine_coupling/nv-2.txt"
CATALOG_JSON   = "analysis/spin_echo_work/essem_freq_kappa_catalog_22A.json"

# Optional: which orientations to consider when comparing theory/exp
DEFAULT_ORIENTATIONS = [
    (1, 1, 1),
    (1, 1, -1),
    (1, -1, 1),
    (-1, 1, 1),
]

# ---------------------------------------------------------------------
# CATALOG + HYPERFINE LOADING
# ---------------------------------------------------------------------

def load_catalog(path_json: str = CATALOG_JSON):
    with open(path_json, "r") as f:
        return json.load(f)


def load_hyperfine_table(
    path_txt: str = HYPERFINE_PATH,
    distance_cutoff: float | None = None,
) -> pd.DataFrame:
    """
    Load the NV=2 hyperfine table and attach a `site_index` column that
    matches the indices used in the ESEEM catalog construction.

    Parameters
    ----------
    path_txt : str
        Path to the hyperfine tensor text file (NV-(111) frame).
    distance_cutoff : float or None
        If not None, keep only sites with `distance < distance_cutoff` (Å)
        *before* assigning `site_index`.

    Returns
    -------
    df : pandas.DataFrame
        Hyperfine table with a `site_index` column (0..N_filtered-1).
    """
    df = read_hyperfine_table_safe(path_txt).copy()

    if distance_cutoff is not None:
        df = df[df["distance"] < float(distance_cutoff)]

    df = df.reset_index(drop=True)
    df["site_index"] = df.index

    return df



def select_records(
    recs,
    fmin_kHz: float = 50.0,
    fmax_kHz: float = 6000.0,
    orientations=None,
):
    """
    Filter catalog records to a frequency window and (optionally) orientation set.
    """
    if orientations is not None:
        ori_set = {tuple(o) for o in orientations}
    else:
        ori_set = None

    out = []
    for r in recs:
        fm_k = r["f_minus_Hz"] / 1e3
        fp_k = r["f_plus_Hz"] / 1e3
        if not (np.isfinite(fm_k) and np.isfinite(fp_k)):
            continue
        if not (fmin_kHz <= fm_k <= fmax_kHz and fmin_kHz <= fp_k <= fmax_kHz):
            continue
        if ori_set and tuple(r["orientation"]) not in ori_set:
            continue
        out.append(r)
    return out


# ---------------------------------------------------------------------
# EXPECTED SPECTRUM (THEORY)
# ---------------------------------------------------------------------

def lines_from_recs(
    recs,
    orientations=None,
    fmin_kHz: float = 50.0,
    fmax_kHz: float = 6000.0,
):
    """
    Return arrays: freqs_kHz (2 per site), weights (same length), and site_idx (pairs).
    Weights try to use physically motivated fields if present:
       - line_w_minus / line_w_plus
       - or amp_weight
       - or kappa
       - otherwise 1.
    """
    if orientations is not None:
        ori_set = {tuple(o) for o in orientations}
    else:
        ori_set = None

    freqs = []
    weights = []
    site_idx = []

    for i, r in enumerate(recs):
        if ori_set and tuple(r["orientation"]) not in ori_set:
            continue

        fm = r["f_minus_Hz"] / 1e3
        fp = r["f_plus_Hz"] / 1e3
        if not (np.isfinite(fm) and np.isfinite(fp)):
            continue
        if not (fmin_kHz <= fm <= fmax_kHz and fmin_kHz <= fp <= fmax_kHz):
            continue

        # Pick weight: prefer explicit line weights if present
        if "line_w_minus" in r and "line_w_plus" in r:
            w_minus = float(r["line_w_minus"])
            w_plus  = float(r["line_w_plus"])
        else:
            base = float(r.get("amp_weight", r.get("kappa", 1.0)))
            w_minus = w_plus = base

        freqs.extend([fm, fp])
        weights.extend([w_minus, w_plus])
        site_idx.extend([r["site_index"], r["site_index"]])

    return np.array(freqs, float), np.array(weights, float), np.array(site_idx, int)


def expected_stick_spectrum_from_recs(
    recs,
    p_occ: float = 0.011,
    orientations=None,
    f_range_kHz=(50, 6000),
    use_weights: bool = True,
    merge_tol_kHz: float = 2.0,
    normalize: bool = False,
):
    """
    Build a discrete expected spectrum (stick plot) from catalog records.
    Returns: (f_stick_kHz, a_stick, fig, ax)
    """
    freqs, w, _ = lines_from_recs(
        recs,
        orientations=orientations,
        fmin_kHz=f_range_kHz[0],
        fmax_kHz=f_range_kHz[1],
    )
    if freqs.size == 0:
        raise ValueError("No catalog lines in the requested frequency range.")

    amps = p_occ * (w if use_weights else np.ones_like(w))

    order = np.argsort(freqs)
    f = freqs[order]
    a = amps[order]

    # merge nearby lines
    f_merged = []
    a_merged = []
    acc_f = f[0]
    acc_a = a[0]
    for f0, a0 in zip(f[1:], a[1:]):
        if abs(f0 - acc_f) <= merge_tol_kHz:
            new_a = acc_a + a0
            acc_f = (acc_f * acc_a + f0 * a0) / (new_a + 1e-30)
            acc_a = new_a
        else:
            f_merged.append(acc_f)
            a_merged.append(acc_a)
            acc_f, acc_a = f0, a0
    f_merged.append(acc_f)
    a_merged.append(acc_a)

    f_stick = np.asarray(f_merged, float)
    a_stick = np.asarray(a_merged, float)

    if normalize and a_stick.sum() > 0:
        a_stick = a_stick / a_stick.sum()

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_xscale("log")
    ax.vlines(f_stick, 0.0, a_stick, linewidth=1.0, alpha=0.9, label="Expected (catalog)")

    ax.set_xlim(*f_range_kHz)
    ax.set_xlabel("Frequency (kHz)")
    ax.set_ylabel("Expected intensity (arb.)" + (" (normalized)" if normalize else ""))
    ax.set_title(f"Expected ESEEM stick spectrum (p_occ={p_occ:.3f})")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(framealpha=0.85)
    fig.tight_layout()

    return f_stick, a_stick, fig, ax


# ---------------------------------------------------------------------
# EXPERIMENTAL FREQUENCIES FROM FITS
# ---------------------------------------------------------------------

def extract_experimental_frequencies(
    file_stem: str,
    chi2_fail_thresh: float = 3.0,
    T2_thresh_us: float = 600.0,
):
    """
    Load fitted spin-echo data for a widefield dataset and return masked arrays
    for NV labels, T2, f0, f1, etc. using your existing extractor.
    """
    data = dm.get_raw_data(file_stem=file_stem)

    (nv, T2_us, f0_kHz, f1_kHz, A_pick_kHz, chis, fit_fail,
     sT2_us, sf0_kHz, sf1_kHz, sA_pick_kHz) = extract_T2_freqs_and_errors(
        data, pick_freq="max", chi2_fail_thresh=chi2_fail_thresh
    )

    nv = np.asarray(nv)

    # base validity
    valid = np.isfinite(T2_us) & (~fit_fail)
    # NOTE: ≥ threshold (long-lived spins), not ≤
    mask = valid & (T2_us <= T2_thresh_us)

    # masked arrays
    out = {
        "nv":          nv[mask],
        "T2_us":       np.asarray(T2_us)[mask],
        "sT2_us":      np.asarray(sT2_us)[mask],
        "A_pick_kHz":  np.asarray(A_pick_kHz)[mask],
        "sA_pick_kHz": np.asarray(sA_pick_kHz)[mask],
        "f0_kHz":      np.asarray(f0_kHz)[mask],
        "f1_kHz":      np.asarray(f1_kHz)[mask],
        "sf0_kHz":     np.asarray(sf0_kHz)[mask],
        "sf1_kHz":     np.asarray(sf1_kHz)[mask],
        "chis":        np.asarray(chis)[mask],
    }
    return out


def plot_sorted_exp_branches(
    f0_kHz,
    f1_kHz,
    sf0_kHz=None,
    sf1_kHz=None,
    title_prefix="Experimental",
    f_range_kHz=(50, 6000),
):
    """
    Plot experimental f0 and f1 separately, sorted by frequency.
    """
    f0 = np.asarray(f0_kHz, float)
    f1 = np.asarray(f1_kHz, float)

    # branch 0
    mask0 = np.isfinite(f0) & (f0 >= f_range_kHz[0]) & (f0 <= f_range_kHz[1])
    order0 = np.argsort(f0[mask0])
    x0 = np.arange(1, order0.size + 1)
    y0 = f0[mask0][order0]

    # branch 1
    mask1 = np.isfinite(f1) & (f1 >= f_range_kHz[0]) & (f1 <= f_range_kHz[1])
    order1 = np.argsort(f1[mask1])
    x1 = np.arange(1, order1.size + 1)
    y1 = f1[mask1][order1]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x0, y0, "o", ms=3, alpha=0.9, label="f0 (exp)")
    ax.plot(x1, y1, "o", ms=3, alpha=0.9, label="f1 (exp)")

    ax.set_yscale("log")
    ax.set_ylim(*f_range_kHz)
    ax.set_xlabel("NV index (sorted within branch)")
    ax.set_ylabel("Frequency (kHz)")
    ax.set_title(f"{title_prefix}: f0 / f1 (sorted)")
    ax.yaxis.set_major_locator(mticker.LogLocator(base=10))
    ax.yaxis.set_minor_locator(mticker.LogLocator(base=10, subs=np.arange(2, 10)))
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(framealpha=0.85)
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------
# MATCHING EXPERIMENTAL (f0,f1) TO CATALOG SITES
# ---------------------------------------------------------------------

def match_exp_pairs_to_catalog(
    recs,
    nv_labels,
    f0_kHz,
    f1_kHz,
    tol_kHz: float = 8.0,
    orientations=None,
    f_range_kHz=(50, 6000),
):
    """
    For each NV (f0,f1) pair, find the best-matching catalog site by minimizing
    a simple distance in (f_minus,f_plus) space.

    Metric: min over permutations of (|f0-f_minus| + |f1-f_plus| , |f0-f_plus| + |f1-f_minus|)

    Only accept a match if both frequencies are within tol_kHz.
    Returns a pandas DataFrame with one row per NV.
    """
    # pre-filter recs
    recs_f = select_records(recs, fmin_kHz=f_range_kHz[0],
                            fmax_kHz=f_range_kHz[1],
                            orientations=orientations)

    if not recs_f:
        raise ValueError("No catalog records in desired range/orientations.")

    # precompute arrays for speed
    fm_all = np.array([r["f_minus_Hz"] / 1e3 for r in recs_f])
    fp_all = np.array([r["f_plus_Hz"] / 1e3  for r in recs_f])

    rows = []
    for nv_label, f0, f1 in zip(nv_labels, f0_kHz, f1_kHz):
        if not (np.isfinite(f0) and np.isfinite(f1)):
            rows.append({"nv": nv_label, "has_match": False})
            continue

        # two possible assignments: (f0->fm, f1->fp) or swapped
        d1 = np.abs(f0 - fm_all) + np.abs(f1 - fp_all)
        d2 = np.abs(f0 - fp_all) + np.abs(f1 - fm_all)

        use_swap = d2 < d1
        best_idx  = np.argmin(np.minimum(d1, d2))
        best_swap = bool(use_swap[best_idx])

        fm = fm_all[best_idx]
        fp = fp_all[best_idx]
        r  = recs_f[best_idx]

        if best_swap:
            match_f0 = fp
            match_f1 = fm
        else:
            match_f0 = fm
            match_f1 = fp

        err0 = np.abs(f0 - match_f0)
        err1 = np.abs(f1 - match_f1)
        max_err = max(err0, err1)

        has_match = (err0 <= tol_kHz) and (err1 <= tol_kHz)

        row = {
            "nv": int(nv_label),
            "f0_exp_kHz": float(f0),
            "f1_exp_kHz": float(f1),
            "f0_theory_kHz": float(match_f0),
            "f1_theory_kHz": float(match_f1),
            "err0_kHz": float(err0),
            "err1_kHz": float(err1),
            "max_err_kHz": float(max_err),
            "has_match": bool(has_match),
            "orientation": tuple(r["orientation"]),
            "site_index": int(r["site_index"]),
            "distance_A": float(r.get("distance_A", np.nan)),
            "kappa": float(r.get("kappa", np.nan)),
            "theta_deg": float(r.get("theta_deg", np.nan)),
        }
        rows.append(row)

    matches_df = pd.DataFrame(rows)
    return matches_df


def enrich_matches_with_hyperfine(matches_df, hf_df):
    """
    Merge in hyperfine-tensor info from the original NV-2 table.
    """
    out = matches_df.merge(
        hf_df,
        on="site_index",
        how="left",
        suffixes=("", "_hf"),
    )
    return out


# ---------------------------------------------------------------------
# COMBINED PLOT: EXPECTED SPECTRUM + EXP FREQUENCIES
# ---------------------------------------------------------------------

def plot_expected_with_exp_overlay(
    recs,
    exp_f0_kHz,
    exp_f1_kHz,
    matches_df=None,
    p_occ: float = 0.011,
    f_range_kHz=(50, 6000),
    orientations=None,
    merge_tol_kHz: float = 2.0,
):
    """
    - Plot expected stick spectrum from catalog (log-x).
    - Overlay experimental f0/f1 as vertical lines:
        * one color for matched
        * another for unmatched (if matches_df provided)
    """
    f_stick, a_stick, fig, ax = expected_stick_spectrum_from_recs(
        recs,
        p_occ=p_occ,
        orientations=orientations,
        f_range_kHz=f_range_kHz,
        merge_tol_kHz=merge_tol_kHz,
        normalize=False,
    )

    exp_f0 = np.asarray(exp_f0_kHz, float)
    exp_f1 = np.asarray(exp_f1_kHz, float)

    if matches_df is None:
        # simple overlay: all experimental lines same style
        for f in np.concatenate([exp_f0, exp_f1]):
            if f_range_kHz[0] <= f <= f_range_kHz[1]:
                ax.axvline(f, ymax=0.8, linestyle="--", alpha=0.5, color="C3")
        ax.plot([], [], "--", color="C3", alpha=0.8, label="Experimental f0,f1")
        ax.legend(framealpha=0.85)
        fig.tight_layout()
        return fig, ax

    # If we have matches_df, use it to separate matched vs unmatched NVs
    match_map = {int(row.nv): bool(row.has_match) for _, row in matches_df.iterrows()}

    matched_f = []
    unmatched_f = []
    for nv_label, f0, f1 in zip(matches_df["nv"], exp_f0, exp_f1):
        if not (np.isfinite(f0) and np.isfinite(f1)):
            continue
        if match_map.get(int(nv_label), False):
            matched_f.extend([f0, f1])
        else:
            unmatched_f.extend([f0, f1])

    for f in matched_f:
        if f_range_kHz[0] <= f <= f_range_kHz[1]:
            ax.axvline(f, ymax=0.8, linestyle="--", alpha=0.8, color="C3")
    for f in unmatched_f:
        if f_range_kHz[0] <= f <= f_range_kHz[1]:
            ax.axvline(f, ymax=0.8, linestyle=":", alpha=0.6, color="C1")

    ax.plot([], [], "--", color="C3", alpha=0.9, label="Exp (matched NVs)")
    ax.plot([], [], ":",  color="C1", alpha=0.9, label="Exp (unmatched NVs)")
    ax.legend(framealpha=0.85)
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------
# MAIN DRIVER EXAMPLE
# ---------------------------------------------------------------------

def run_essem_match_analysis(
    file_stem: str,
    chi2_fail_thresh: float = 3.0,
    T2_thresh_us: float = 600.0,
    match_tol_kHz: float = 8.0,
    f_range_kHz=(50, 6000),
):
    """
    High-level convenience function that:
      - loads catalog + hyperfine table
      - loads experimental fits and extracts (f0,f1)
      - matches each NV to a catalog site
      - returns enriched matches_df
      - and makes a combined spectrum plot + an exp-only plot.
    """
    print(f"[INFO] Loading catalog from: {CATALOG_JSON}")
    recs_all = load_catalog(CATALOG_JSON)
    hf_df = load_hyperfine_table(HYPERFINE_PATH)

    print(f"[INFO] Loading experimental fit data from file_stem='{file_stem}'")
    exp = extract_experimental_frequencies(
        file_stem=file_stem,
        chi2_fail_thresh=chi2_fail_thresh,
        T2_thresh_us=T2_thresh_us,
    )

    print("[INFO] Matching experimental (f0,f1) to catalog sites...")
    matches_df = match_exp_pairs_to_catalog(
        recs_all,
        nv_labels=exp["nv"],
        f0_kHz=exp["f0_kHz"],
        f1_kHz=exp["f1_kHz"],
        tol_kHz=match_tol_kHz,
        orientations=DEFAULT_ORIENTATIONS,
        f_range_kHz=f_range_kHz,
    )
    matches_enriched = enrich_matches_with_hyperfine(matches_df, hf_df)

    print("\n[SUMMARY] Match statistics (using tol = "
          f"{match_tol_kHz:.1f} kHz):")
    print(matches_enriched["has_match"].value_counts(dropna=False))

    # Combined theory+exp spectrum
    print("[INFO] Making combined spectrum plot (theory + f0,f1 overlays)...")
    plot_expected_with_exp_overlay(
        recs_all,
        exp_f0_kHz=exp["f0_kHz"],
        exp_f1_kHz=exp["f1_kHz"],
        matches_df=matches_enriched,
        f_range_kHz=f_range_kHz,
        orientations=DEFAULT_ORIENTATIONS,
    )

    # Experimental only, sorted by branch
    print("[INFO] Making experimental-only sorted f0/f1 plot...")
    plot_sorted_exp_branches(
        f0_kHz=exp["f0_kHz"],
        f1_kHz=exp["f1_kHz"],
        title_prefix=f"Exp (T2 ≥ {T2_thresh_us:.0f} µs)",
        f_range_kHz=f_range_kHz,
    )

    plt.show()
    return matches_enriched


# -------------------------------------------------------------------
# 1) σ-distance between an experimental pair (f0,f1) and catalog (fm,fp)
# -------------------------------------------------------------------
def pair_distance_sigma_vec(
    f0_kHz: float,
    f1_kHz: float,
    sf0_kHz: float,
    sf1_kHz: float,
    fm_kHz: np.ndarray,
    fp_kHz: np.ndarray,
) -> np.ndarray:
    """
    Return the per-site σ-distance between experimental (f0,f1) and
    catalog (fm,fp), taking into account both labelings:
      (f0 ↔ fm, f1 ↔ fp) and (f0 ↔ fp, f1 ↔ fm)

    D_j = min( sqrt(((f0-fm_j)/sf0)^2 + ((f1-fp_j)/sf1)^2),
               sqrt(((f0-fp_j)/sf0)^2 + ((f1-fm_j)/sf1)^2) )
    """
    fm_kHz = np.asarray(fm_kHz, float)
    fp_kHz = np.asarray(fp_kHz, float)

    d1 = np.sqrt(((f0_kHz - fm_kHz) / sf0_kHz) ** 2 +
                 ((f1_kHz - fp_kHz) / sf1_kHz) ** 2)
    d2 = np.sqrt(((f0_kHz - fp_kHz) / sf0_kHz) ** 2 +
                 ((f1_kHz - fm_kHz) / sf1_kHz) ** 2)
    return np.minimum(d1, d2)


def pair_distance_plain_vec(
    f0_kHz: float,
    f1_kHz: float,
    fm_kHz: np.ndarray,
    fp_kHz: np.ndarray,
) -> np.ndarray:
    """
    Plain Euclidean distance in kHz between (f0,f1) and (fm,fp),
    again minimizing over the two possible assignments.
    """
    fm_kHz = np.asarray(fm_kHz, float)
    fp_kHz = np.asarray(fp_kHz, float)

    d1 = np.sqrt((f0_kHz - fm_kHz) ** 2 + (f1_kHz - fp_kHz) ** 2)
    d2 = np.sqrt((f0_kHz - fp_kHz) ** 2 + (f1_kHz - fm_kHz) ** 2)
    return np.minimum(d1, d2)


# -------------------------------------------------------------------
# 2) Main analysis: NV ↔ 13C site matching with confidence metrics
# -------------------------------------------------------------------
def run_full_essem_match_analysis(
    file_stem: str,
    chi2_fail_thresh: float = 3.0,
    T2_thresh_us: float = 600.0,
    match_tol_kHz: float = 8.0,
    f_range_kHz=(50.0, 6000.0),
    catalog_json: str = "analysis/spin_echo_work/essem_freq_catalog_22A.json",
    # Theory / calibration uncertainties
    theory_sigma_kHz: float = 30.0,
    frac_A_theory: float = 0.5,
) -> pd.DataFrame:
    """
    Full NV ↔ 13C-site match analysis with confidence metrics.

    Returns a DataFrame with one row per NV, including:
      - NV info: nv, T2_us, A_pick_kHz, f0/f1, errors, red_chi2, fit_fail
      - Best-match site info: site_index, orientation, distance_A, A_par/A_perp, ...
      - Matching metrics:
          match_sigma   (σ-distance for best site)
          gap_sigma     (σ-distance gap to second-best)
          conf_freq     (frequency-only confidence 0–1)
          Z_amp, conf_amp
          confidence    (combined: conf_freq * conf_amp)
          dist_plain_kHz (plain Euclidean distance in kHz)
          within_tol    (boolean: dist_plain_kHz <= match_tol_kHz)
          has_match     (boolean: within_tol & not NaN)
    """

    # ---------- 1) Load experimental fit summary ----------
    data = dm.get_raw_data(file_stem=file_stem)

    (
        nv,
        T2_us,
        f0_kHz,
        f1_kHz,
        A_pick_kHz,
        chis,
        fit_fail,
        sT2_us,
        sf0_kHz,
        sf1_kHz,
        sA_pick_kHz,
    ) = extract_T2_freqs_and_errors(
        data,
        pick_freq="max",
        chi2_fail_thresh=chi2_fail_thresh,  # only used internally for error logic
    )

    nv         = np.asarray(nv, int)
    T2_us      = np.asarray(T2_us, float)
    f0_kHz     = np.asarray(f0_kHz, float)
    f1_kHz     = np.asarray(f1_kHz, float)
    A_pick_kHz = np.asarray(A_pick_kHz, float)
    chis       = np.asarray(chis, float)
    fit_fail   = np.asarray(fit_fail, bool)
    sT2_us     = np.asarray(sT2_us, float)
    sf0_kHz    = np.asarray(sf0_kHz, float)
    sf1_kHz    = np.asarray(sf1_kHz, float)
    sA_pick_kHz= np.asarray(sA_pick_kHz, float)

    N_nv = nv.size

    # Valid mask (same logic you used before)
    valid = np.isfinite(T2_us) & (~fit_fail) & np.isfinite(f0_kHz) & np.isfinite(f1_kHz)
    t2_ok = T2_us <= T2_thresh_us
    mask  = valid & t2_ok

    # ---------- 2) Load catalog + precompute arrays ----------
    catalog_recs = load_catalog(catalog_json)
    recs = select_records(
        catalog_recs,
        fmin_kHz=f_range_kHz[0],
        fmax_kHz=f_range_kHz[1],
        orientations=None,
    )

    if len(recs) == 0:
        raise ValueError("No catalog records in the requested frequency window.")

    # Arrays for speed
    fm_all_kHz = np.array([r["f_minus_Hz"] / 1e3 for r in recs], float)
    fp_all_kHz = np.array([r["f_plus_Hz"]  / 1e3 for r in recs], float)
    site_all   = np.array([r["site_index"]          for r in recs], int)
    ori_all    = np.array([r["orientation"]         for r in recs], int)
    dist_all   = np.array([r["distance_A"]          for r in recs], float)

    A_par_all_kHz  = np.array([r["A_par_Hz"]  / 1e3 for r in recs], float)
    A_perp_all_kHz = np.array([r["A_perp_Hz"] / 1e3 for r in recs], float)
    theta_all_deg  = np.array([r.get("theta_deg", np.nan) for r in recs], float)

    # Optional weights / kappa
    kappa_all      = np.array([r.get("kappa", np.nan) for r in recs], float)
    wminus_all     = np.array([r.get("line_w_minus", np.nan) for r in recs], float)
    wplus_all      = np.array([r.get("line_w_plus",  np.nan) for r in recs], float)

    # ---------- 3) Loop over NVs and compute best match ----------
    rows = []

    for i in range(N_nv):
        nv_label = int(nv[i])

        # Defaults for "no match" row:
        row_base = dict(
            nv=nv_label,
            T2_us=float(T2_us[i]),
            sT2_us=float(sT2_us[i]),
            A_pick_kHz=float(A_pick_kHz[i]),
            sA_pick_kHz=float(sA_pick_kHz[i]),
            f0_kHz=float(f0_kHz[i]),
            f1_kHz=float(f1_kHz[i]),
            sf0_kHz=float(sf0_kHz[i]),
            sf1_kHz=float(sf1_kHz[i]),
            red_chi2=float(chis[i]),
            fit_fail=bool(fit_fail[i]),
        )

        if not mask[i]:
            # Not eligible for matching (bad fit, T2 too long, NaNs, etc.)
            rows.append(
                dict(
                    **row_base,
                    has_match=False,
                    within_tol=False,
                    site_index=None,
                    orientation_x=None,
                    orientation_y=None,
                    orientation_z=None,
                    distance_A=np.nan,
                    fm_kHz=np.nan,
                    fp_kHz=np.nan,
                    A_par_kHz=np.nan,
                    A_perp_kHz=np.nan,
                    theta_deg=np.nan,
                    kappa=np.nan,
                    line_w_minus=np.nan,
                    line_w_plus=np.nan,
                    match_sigma=np.nan,
                    gap_sigma=np.nan,
                    conf_freq=np.nan,
                    Z_amp=np.nan,
                    conf_amp=np.nan,
                    confidence=np.nan,
                    dist_plain_kHz=np.nan,
                )
            )
            continue

        f0 = float(f0_kHz[i])
        f1 = float(f1_kHz[i])
        sf0 = float(sf0_kHz[i])
        sf1 = float(sf1_kHz[i])

        # If errors are non-finite or tiny, regularize them a bit
        if not np.isfinite(sf0) or sf0 <= 0.0:
            sf0 = max(1.0, theory_sigma_kHz)  # arbitrary floor
        if not np.isfinite(sf1) or sf1 <= 0.0:
            sf1 = max(1.0, theory_sigma_kHz)

        # Combine experimental + theory σ in quadrature
        sf0_eff = np.hypot(sf0, theory_sigma_kHz)
        sf1_eff = np.hypot(sf1, theory_sigma_kHz)

        # σ-distance for all sites
        D_all = pair_distance_sigma_vec(
            f0_kHz=f0,
            f1_kHz=f1,
            sf0_kHz=sf0_eff,
            sf1_kHz=sf1_eff,
            fm_kHz=fm_all_kHz,
            fp_kHz=fp_all_kHz,
        )

        # Plain distance (kHz)
        d_plain_all = pair_distance_plain_vec(
            f0_kHz=f0,
            f1_kHz=f1,
            fm_kHz=fm_all_kHz,
            fp_kHz=fp_all_kHz,
        )

        # Best and second-best
        best_idx = int(np.argmin(D_all))
        D_best = float(D_all[best_idx])
        d_plain_best = float(d_plain_all[best_idx])

        D_sorted = np.sort(D_all)
        if D_sorted.size > 1:
            D_second = float(D_sorted[1])
        else:
            D_second = np.inf

        gap_sigma = D_second - D_best

        # Frequency-only "posterior-like" confidence
        lik_all = np.exp(-0.5 * D_all**2)
        lik_sum = float(lik_all.sum())
        conf_freq = float(lik_all[best_idx] / lik_sum) if lik_sum > 0 else np.nan

        # Extract best site info
        site_idx   = int(site_all[best_idx])
        ori_xyz    = ori_all[best_idx]
        dist_A     = float(dist_all[best_idx])
        fm_best    = float(fm_all_kHz[best_idx])
        fp_best    = float(fp_all_kHz[best_idx])
        A_par_best = float(A_par_all_kHz[best_idx])
        A_perp_best= float(A_perp_all_kHz[best_idx])
        theta_best = float(theta_all_deg[best_idx])
        kappa_best = float(kappa_all[best_idx]) if np.isfinite(kappa_all[best_idx]) else np.nan
        wminus_best= float(wminus_all[best_idx]) if np.isfinite(wminus_all[best_idx]) else np.nan
        wplus_best = float(wplus_all[best_idx]) if np.isfinite(wplus_all[best_idx]) else np.nan

        # Amplitude consistency (A_perp vs A_pick)
        A_exp  = float(A_pick_kHz[i])
        sA_exp = float(sA_pick_kHz[i])

        if not np.isfinite(A_exp) or A_exp <= 0 or not np.isfinite(A_perp_best):
            Z_amp   = np.nan
            conf_amp= 1.0  # neutral
        else:
            sigma_theory = max(frac_A_theory * abs(A_perp_best), 0.1)
            sigma_tot = np.sqrt((sA_exp if np.isfinite(sA_exp) else 0.0) ** 2 +
                                sigma_theory ** 2)
            if sigma_tot <= 0:
                sigma_tot = sigma_theory
            Z_amp = (A_exp - A_perp_best) / sigma_tot
            conf_amp = float(np.exp(-0.5 * Z_amp**2))

        # Combined confidence
        if np.isfinite(conf_freq) and np.isfinite(conf_amp):
            confidence = conf_freq * conf_amp
        else:
            confidence = np.nan

        # Within match tolerance?
        within_tol = d_plain_best <= match_tol_kHz

        rows.append(
            dict(
                **row_base,
                has_match=bool(within_tol),
                within_tol=bool(within_tol),
                site_index=site_idx,
                orientation_x=int(ori_xyz[0]),
                orientation_y=int(ori_xyz[1]),
                orientation_z=int(ori_xyz[2]),
                distance_A=dist_A,
                fm_kHz=fm_best,
                fp_kHz=fp_best,
                A_par_kHz=A_par_best,
                A_perp_kHz=A_perp_best,
                theta_deg=theta_best,
                kappa=kappa_best,
                line_w_minus=wminus_best,
                line_w_plus=wplus_best,
                match_sigma=D_best,
                gap_sigma=gap_sigma,
                conf_freq=conf_freq,
                Z_amp=Z_amp,
                conf_amp=conf_amp,
                confidence=confidence,
                dist_plain_kHz=d_plain_best,
            )
        )

    df_match = pd.DataFrame(rows)

    # Optional: sort by confidence descending for quick browsing
    # df_match = df_match.sort_values(
    #     ["has_match", "confidence"],
    #     ascending=[False, False],
    # ).reset_index(drop=True)

    return df_match

# If _params_to_dict and plot_echo_with_sites live in another module, import from there instead:
# from analysis.spin_echo_work.echo_plot_helpers import _params_to_dict
# from analysis.spin_echo_work.echo_site_plots import plot_echo_with_sites

_fn_map = {
    "fine_decay": fine_decay,
    "fine_decay_fixed_revival": fine_decay_fixed_revival,
}

def _get_coord_cols(hf_df: pd.DataFrame):
    """
    Detect which columns hold NV-frame coordinates in Å.
    Adjust if your nv-2 table uses different names.
    """
    if {"x_A", "y_A", "z_A"}.issubset(hf_df.columns):
        return "x_A", "y_A", "z_A"
    if {"x", "y", "z"}.issubset(hf_df.columns):
        return "x", "y", "z"
    raise KeyError(
        "Could not find coordinate columns in hyperfine table; "
        "modify _get_coord_cols() to match your column names."
    )

def _find_catalog_rec_for_match(row, catalog_recs):
    """
    Find the catalog record corresponding to this match row.

    We match on:
      - site_index
      - orientation  (tuple of 3 ints)
    """
    if catalog_recs is None:
        return None

    # site_index
    site_index = int(row.get("site_index", -1))

    # Orientation: normalize to tuple[int,int,int]
    ori = row.get("orientation", None)
    if isinstance(ori, str):
        try:
            s = ori.strip("()[]")
            parts = [int(x) for x in s.replace(",", " ").split()]
            ori = tuple(parts) if len(parts) == 3 else None
        except Exception:
            ori = None
    elif isinstance(ori, (list, tuple)) and len(ori) == 3:
        ori = tuple(int(v) for v in ori)
    else:
        ori = None

    for rec in catalog_recs:
        if int(rec.get("site_index", -1)) != site_index:
            continue
        rec_ori = rec.get("orientation", None)
        if isinstance(rec_ori, (list, tuple)) and len(rec_ori) == 3:
            rec_ori = tuple(int(v) for v in rec_ori)
        else:
            rec_ori = None

        if ori is None or rec_ori is None:
            # if something is missing, at least match on site_index
            if int(rec.get("site_index", -1)) == site_index:
                return rec
        else:
            if rec_ori == ori:
                return rec

    return None

def _build_site_info_from_match_and_catalog(row, hf_row, catalog_recs):
    """
    row          : one row from matches_enriched (Series for single NV)
    hf_row       : matching hyperfine row (geometry, coords, etc.)
    catalog_recs : list of dicts from essem_freq_catalog_XX.json
    """
    cat_rec = _find_catalog_rec_for_match(row, catalog_recs)

    # ---- Geometry: r ----
    r_val = hf_row.get("r", np.nan)
    if not np.isfinite(r_val):
        r_val = row.get("distance_A", np.nan)

    # ---- Orientation ----
    if cat_rec is not None:
        ori = cat_rec.get("orientation", None)
    else:
        ori = row.get("orientation", None)

    if isinstance(ori, str):
        try:
            s = ori.strip("()[]")
            parts = [int(x) for x in s.replace(",", " ").split()]
            ori = tuple(parts) if len(parts) == 3 else None
        except Exception:
            ori = None
    elif isinstance(ori, (list, tuple)) and len(ori) == 3:
        ori = tuple(int(v) for v in ori)
    else:
        ori = None

    def _cat_float(key, default=np.nan, scale=1.0):
        if cat_rec is None or key not in cat_rec:
            return float(default)
        v = cat_rec[key]
        try:
            return float(v) * scale
        except Exception:
            return float(default)

    # A_par / A_perp from catalog if available, else matches_enriched
    A_par_kHz  = _cat_float("A_par_Hz",  default=np.nan, scale=1e-3)
    A_perp_kHz = _cat_float("A_perp_Hz", default=np.nan, scale=1e-3)

    if np.isnan(A_par_kHz) and "A_par_Hz" in row.index:
        try:
            A_par_kHz = float(row["A_par_Hz"]) * 1e-3
        except Exception:
            pass
    if np.isnan(A_perp_kHz) and "A_perp_Hz" in row.index:
        try:
            A_perp_kHz = float(row["A_perp_Hz"]) * 1e-3
        except Exception:
            pass

    # theta from catalog or row
    theta_deg = _cat_float("theta_deg", default=np.nan, scale=1.0)
    if np.isnan(theta_deg) and "theta_deg" in row.index:
        try:
            theta_deg = float(row["theta_deg"])
        except Exception:
            pass

    # kappa, fI, f-/f+ from catalog
    kappa  = _cat_float("kappa",      default=np.nan)
    fI_kHz = _cat_float("fI_Hz",      default=np.nan, scale=1e-3)
    fm_kHz = _cat_float("f_minus_Hz", default=np.nan, scale=1e-3)
    fp_kHz = _cat_float("f_plus_Hz",  default=np.nan, scale=1e-3)

    site_index = int(row.get("site_index", hf_row.get("site_index", -1)))

    return [{
        "site_id":    site_index,
        "r":          float(r_val),
        "Apar_kHz":   float(A_par_kHz),
        "Aperp_kHz":  float(A_perp_kHz),
        "theta_deg":  float(theta_deg),
        "kappa":      float(kappa),
        "fI_kHz":     float(fI_kHz),
        "fm_kHz":     float(fm_kHz),
        "fp_kHz":     float(fp_kHz),
        "orientation": ori,
    }]




def make_echo_plus_matched_site_plot(
    counts_file_stem: str,
    fit_file_stem: str,
    matches_enriched: pd.DataFrame,
    hf_df: pd.DataFrame,
    nv_label: int,
    use_half_time_as_tau: bool = True,
    units_label: str = "(Norm.)",
    confidence = None,
):
    """
    Make a single figure:
      left: experimental spin-echo trace + fit + envelope
      right: matched 13C site in 3D (NV frame)
    for a chosen NV label.

    Inputs
    ------
    counts_file_stem : str
        File stem for the processed echo counts (norm_counts etc).
    fit_file_stem : str
        File stem for the spin-echo fits (popts, fit_fn_names, nv_labels).
    matches_enriched : DataFrame
        Output of run_full_essem_match_analysis(...).
    hf_df : DataFrame
        Hyperfine table (including site_index + x/y/z columns).
    nv_label : int
        NV label to plot.
    """
  
    # ---------- 1) Match-row + hyperfine row for this NV ----------
    row = matches_enriched.loc[matches_enriched["nv"] == nv_label]
    if row.empty:
        raise ValueError(f"No entry for NV {nv_label} in matches_enriched.")
    row = row.iloc[0]

    if not bool(row["has_match"]):
        print(
            f"[WARN] NV {nv_label} has no good catalog match (has_match=False). "
            "Plot will still show the NV and a 'dummy' site if available."
        )

    site_index = int(row["site_index"])

    # matching hyperfine row (for positions, etc.)
    hf_row = hf_df.loc[hf_df["site_index"] == site_index]
    if hf_row.empty:
        raise ValueError(f"No hyperfine row found for site_index={site_index}.")
    hf_row = hf_row.iloc[0]

    xcol, ycol, zcol = _get_coord_cols(hf_df)

    # ---------- 2) Load counts data (echo) ----------
    data_counts = dm.get_raw_data(file_stem=counts_file_stem)

    norm_counts     = data_counts["norm_counts"]
    norm_counts_ste = data_counts["norm_counts_ste"]
    total_times_us  = np.asarray(data_counts["total_evolution_times"], float)

    # echo trace for this NV (from counts file)
    echo     = np.asarray(norm_counts[int(nv_label)], float)
    echo_ste = np.asarray(norm_counts_ste[int(nv_label)], float)

    # Use τ = total_time/2 or τ = total_time
    if use_half_time_as_tau:
        taus_us = total_times_us / 2.0
    else:
        taus_us = total_times_us

    # ---------- 3) Load fit data (parameters) ----------
    data_fit = dm.get_raw_data(file_stem=fit_file_stem)

    fit_nv_labels = np.array(list(map(int, data_fit["nv_labels"])))
    idx = np.where(fit_nv_labels == int(nv_label))[0]
    if idx.size == 0:
        raise ValueError(f"NV {nv_label} not found in fit_nv_labels for {fit_file_stem}.")
    idx = int(idx[0])

    popts        = data_fit["popts"]
    fit_fn_names = data_fit["fit_fn_names"]

    fit_fn_name = fit_fn_names[idx]
    fit_fn      = _fn_map.get(fit_fn_name, fine_decay)
    p           = np.asarray(popts[idx], float)

    # Convert p -> nice dict for plotting box
    fine_params = params_to_dict(fit_fn, p, default_rev=39.2)
    # Add an explicit T2 in µs if not present
    if "T2_fit_us" not in fine_params or fine_params["T2_fit_us"] is None:
        fine_params["T2_fit_us"] = 1000.0 * fine_params.get("T2_ms", 0.0)

    # ---------- 4) Build aux dict for plot_echo_with_sites ----------
    # All candidate sites (background cloud)
    all_pos = None
    if {xcol, ycol, zcol}.issubset(hf_df.columns):
        all_pos = hf_df[[xcol, ycol, zcol]].to_numpy(float)

    # Foreground: just the matched site
    pos = np.array(
        [[hf_row[xcol], hf_row[ycol], hf_row[zcol]]],
        dtype=float,
    )
    
    catalog_recs = load_catalog(CATALOG_JSON)
    site_info = _build_site_info_from_match_and_catalog(row, hf_row, catalog_recs)

    stats = {
        "N_candidates": len(hf_df),
        "abundance_fraction": None,   # set if you know 13C fraction explicitly
    }

    aux = {
        "positions": pos,
        "site_info": site_info,
        "all_candidate_positions": all_pos,
        "stats": stats,
        "distance_cutoff": float(row.get("distance_A", np.nan)),
        "Ak_min_kHz": None,
        "Ak_max_kHz": None,
        "picked_ids_per_realization": None,
    }

    sim_info = {
        "T2_fit_us": fine_params.get("T2_fit_us", None),
        "distance_cutoff": aux["distance_cutoff"],
    }

    # ---------- 5) Call your existing helper ----------
    fig = plot_echo_with_sites(
        taus_us=taus_us,
        echo=echo,
        aux=aux,
        title="Spin Echo (experimental + matched site)",
        rmax=None,
        fine_params=fine_params,
        units_label=units_label,
        nv_label=nv_label,
        sim_info=sim_info,
        show_env=True,
        show_env_times_comb=False,
        echo_ste=echo_ste,
        fit_fn=fit_fn,
        fit_params=p,
        tau_is_half_time=use_half_time_as_tau,
        default_rev_for_plot=39.2,
    )

    return fig



if __name__ == "__main__":
    kpl.init_kplotlib()

    # ---- 1) file stems ----
    fit_file_stem    = "2025_11_13-06_28_22-sample_204nv_s1-e85aa7"   # where popts & freqs live
    counts_file_stem = "2025_11_11-01_15_45-johnson_204nv_s6-6d8f5c"  # merged dataset2+3 counts

    # ---- 2) global theory-vs-exp matching (use FIT file) ----
    matches_enriched = run_full_essem_match_analysis(
        file_stem=fit_file_stem,
        chi2_fail_thresh=3.0,
        T2_thresh_us=600.0,
        match_tol_kHz=8.0,
        f_range_kHz=(50, 6000),
    )

    hf_df = load_hyperfine_table(distance_cutoff=15.0)   # or 15.0, etc.

    # pick an NV that has a match:
    good = matches_enriched[matches_enriched["has_match"]]
    if good.empty:
        raise RuntimeError("No NV has a good match; check thresholds.")
    nv_example = int(good.iloc[0]["nv"])

    # ---- 3) make the combined echo + site plot ----
    fig = make_echo_plus_matched_site_plot(
        counts_file_stem=counts_file_stem,
        fit_file_stem=fit_file_stem,
        matches_enriched=matches_enriched,
        hf_df=hf_df,
        nv_label=nv_example,
        use_half_time_as_tau=True,
    )

    
    matches_enriched = run_full_essem_match_analysis(
        file_stem=fit_file_stem,
        chi2_fail_thresh=3.0,
        T2_thresh_us=600.0,
        match_tol_kHz=8.0,
        f_range_kHz=(50, 6000),
        catalog_json="analysis/spin_echo_work/essem_freq_catalog_22A.json",
        theory_sigma_kHz=30.0,
        frac_A_theory=0.5,
    )

    # e.g. look at high-confidence matches:
    good = matches_enriched[
        (matches_enriched["has_match"])
        & (matches_enriched["confidence"] > 0.5)
    ]
    print(good[[
        "nv","fm_kHz","fp_kHz","f0_kHz","f1_kHz",
        "match_sigma","gap_sigma","Z_amp","confidence"
    ]].to_string(index=False))
    
    confidence = matches_enriched
    # ---- 3) make the combined echo + site plot ----
    fig = make_echo_plus_matched_site_plot(
        counts_file_stem=counts_file_stem,
        fit_file_stem=fit_file_stem,
        matches_enriched=matches_enriched,
        hf_df=hf_df,
        nv_label=nv_example,
        use_half_time_as_tau=True,
        confidence = confidence,
    )

    kpl.show(block=True)
