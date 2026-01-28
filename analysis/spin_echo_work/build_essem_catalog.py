import json, csv, numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from analysis.sc_c13_hyperfine_sim_data_driven import read_hyperfine_table_safe, B_vec_T

# ----------------------------------------------------------------------
# Spin-1/2 operators (Pauli / 2)
# ----------------------------------------------------------------------
Sx = 0.5 * np.array([[0, 1],
                     [1, 0]], float)
Sy = 0.5 * np.array([[0,-1j],
                     [1j, 0]], complex)
Sz = 0.5 * np.array([[1, 0],
                     [0,-1]], float)


# ----------------------------------------------------------------------
# Geometry + Hamiltonian helpers
# ----------------------------------------------------------------------
def _build_U_from_orientation(orientation, phi_deg: float = 0.0):
    """
    Build a rotation matrix U that sends cubic axes to the NV frame for
    a given orientation (±1, ±1, ±1) and an optional in-plane twist phi_deg.
    """
    ez = np.asarray(orientation, float)
    ez /= np.linalg.norm(ez)

    # pick a trial x-axis not collinear with ez
    trial = np.array([1.0, -1.0, 0.0])
    if abs(np.dot(trial / np.linalg.norm(trial), ez)) > 0.95:
        trial = np.array([0.0, 1.0, -1.0])

    ex = trial - np.dot(trial, ez) * ez
    ex /= np.linalg.norm(ex)
    ey = np.cross(ez, ex)
    ey /= np.linalg.norm(ey)

    U0 = np.column_stack([ex, ey, ez])

    phi = np.deg2rad(phi_deg)
    Rz = np.array([[np.cos(phi), -np.sin(phi), 0.0],
                   [np.sin(phi),  np.cos(phi), 0.0],
                   [0.0,          0.0,        1.0]])
    return U0 @ Rz, ez


def essem_lines_by_diag(
    A_file_Hz: np.ndarray,
    orientation=(1, 1, 1),
    B_lab_vec=None,
    gamma_n_Hz_per_T: float = 10.705e6,
    ms: int = -1,
    phi_deg: float = 0.0,
):
    """
    Diagonalize the nuclear Hamiltonian with and without hyperfine to get
    ESEEM frequencies f_- and f_+ for a single 13C site.

    Parameters
    ----------
    A_file_Hz : (3,3) array
        Hyperfine tensor (Hz) in the NV-(111) frame (as in your table).
    orientation : tuple
        NV orientation in cubic coordinates (e.g. (1,1,1)).
    B_lab_vec : array-like, shape (3,)
        Lab-frame B-field vector. Units must match gamma_n_Hz_per_T.
    gamma_n_Hz_per_T : float
        Nuclear gyromagnetic ratio in Hz/T (13C: 10.705e6 Hz/T).
    ms : int
        Electron spin manifold (+/-1). For NV- ESEEM, typically -1.
    phi_deg : float
        Additional in-plane twist of hyperfine tensor about NV z.

    Returns
    -------
    f_minus, f_plus, fI_split, omega_ms_split, A_cubic, z_nv_cubic
        All in Hz (except A_cubic tensor and unit vector z_nv_cubic).
    """
    if B_lab_vec is None:
        raise ValueError("B_lab_vec must be provided.")

    # rotate A into cubic frame for this NV
    U, z_nv_cubic = _build_U_from_orientation(orientation, phi_deg=phi_deg)
    A_cubic = U @ A_file_Hz @ U.T

    B_lab = np.asarray(B_lab_vec, float)
    Bmag = float(np.linalg.norm(B_lab))
    if Bmag == 0.0:
        raise ValueError("B field magnitude is zero.")
    bx, by, bz = B_lab / Bmag

    # nuclear Zeeman
    fI_Hz = gamma_n_Hz_per_T * Bmag
    HZ = fI_Hz * (bx * Sx + by * Sy + bz * Sz)

    # hyperfine term projected along NV axis (ms-dependent)
    Aeff_vec = A_cubic @ z_nv_cubic
    Hhf = float(ms) * (Aeff_vec[0] * Sx + Aeff_vec[1] * Sy + Aeff_vec[2] * Sz)

    evals0 = np.linalg.eigvalsh(HZ)
    evalsms = np.linalg.eigvalsh(HZ + Hhf)

    fI_split = float(abs(evals0[1] - evals0[0]))
    omega_ms_split = float(abs(evalsms[1] - evalsms[0]))

    f_minus = abs(omega_ms_split - fI_split)
    f_plus = omega_ms_split + fI_split
    return f_minus, f_plus, fI_split, omega_ms_split, A_cubic, z_nv_cubic


# ----------------------------------------------------------------------
# 1) Build and save full ESEEM catalog
# ----------------------------------------------------------------------
def build_essem_catalog(
    hyperfine_path: str,
    B_lab_vec,
    orientations=((1, 1, 1), (1, 1, -1), (1, -1, 1), (-1, 1, 1)),
    distance_max_A: float = 22.0,
    gamma_n_Hz_per_T: float = 10.705e6,
    ms: int = -1,
    phi_deg: float = 0.0,
    out_json: str = "essem_freq_catalog.json",
    out_csv: str = "essem_freq_catalog.csv",
):
    """
    Reads the hyperfine table, computes (f-, f+) for all sites within distance_max_A
    and all NV orientations; saves JSON+CSV with handy fields for later fitting.

    Units:
      - Axx, Ayy, ... in your table are assumed MHz -> multiplied by 1e6 to Hz.
      - B_lab_vec units must match gamma_n_Hz_per_T (default assumes Tesla).
    """
    df = read_hyperfine_table_safe(hyperfine_path).copy()
    df = df[df["distance"] <= float(distance_max_A)].reset_index(drop=True)

    B_lab = np.asarray(B_lab_vec, float)
    Bmag = float(np.linalg.norm(B_lab))
    B_hat = B_lab / Bmag

    records = []
    for ori in orientations:
        ori_tuple = tuple(int(x) for x in ori)

        for i, row in df.iterrows():
            # A_file in Hz (NV-(111) frame)
            A_file_Hz = np.array(
                [
                    [row.Axx, row.Axy, row.Axz],
                    [row.Axy, row.Ayy, row.Ayz],
                    [row.Axz, row.Ayz, row.Azz],
                ],
                float,
            ) * 1e6  # MHz -> Hz

            (
                fm,
                fp,
                fI,
                wms,
                A_cubic,
                z_nv,
            ) = essem_lines_by_diag(
                A_file_Hz=A_file_Hz,
                orientation=ori,
                B_lab_vec=B_lab,
                gamma_n_Hz_per_T=gamma_n_Hz_per_T,
                ms=ms,
                phi_deg=phi_deg,
            )

            # amplitude proxy: ~ sin^2(theta)*(A_perp/omega)^2
            A_par = float(B_hat @ A_cubic @ B_hat)
            A_perp_vec = A_cubic @ B_hat - A_par * B_hat
            A_perp = float(np.linalg.norm(A_perp_vec))
            cos_th = float(np.clip(B_hat @ (z_nv / np.linalg.norm(z_nv)), -1, 1))
            sin2_th = 1.0 - cos_th**2
            amp_wt = (A_perp / max(wms, 1e-30)) ** 2 * sin2_th

            records.append(
                {
                    "orientation": ori_tuple,
                    "site_index": int(i),
                    "distance_A": float(row["distance"]),
                    "f_minus_Hz": float(fm),
                    "f_plus_Hz": float(fp),
                    "fI_Hz": float(fI),
                    "omega_ms_Hz": float(wms),
                    "A_par_Hz": float(A_par),
                    "A_perp_Hz": float(A_perp),
                    "amp_weight": float(amp_wt),
                }
            )

    # Save JSON
    with open(out_json, "w") as f:
        json.dump(records, f, indent=2)

    # Save CSV
    if records:
        keys = list(records[0].keys())
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(records)

    return records


# ----------------------------------------------------------------------
# 2) Load + basic filtering
# ----------------------------------------------------------------------
def load_catalog(path_json: str):
    """Load previously built ESEEM frequency catalog from JSON."""
    with open(path_json, "r") as f:
        return json.load(f)


def _record_in_freq_window(rec, fmin_kHz: float, fmax_kHz: float) -> bool:
    fm_k = rec["f_minus_Hz"] / 1e3
    fp_k = rec["f_plus_Hz"] / 1e3
    okm = fmin_kHz <= fm_k <= fmax_kHz
    okp = fmin_kHz <= fp_k <= fmax_kHz
    return okm and okp


def select_records(
    recs,
    fmin_kHz: float = 150.0,
    fmax_kHz: float = 20000.0,
    orientations=None,
):
    """
    Filter catalog records by frequency window and (optionally) orientations.
    Returns a new list of dicts.
    """
    ori_set = {tuple(o) for o in orientations} if orientations else None
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


# ----------------------------------------------------------------------
# 3) API compatible with old load_candidates (used by fitter)
# ----------------------------------------------------------------------
def load_candidates(
    path_json: str,
    fmin_kHz: float = 150.0,
    fmax_kHz: float = 20000.0,
    top_by_weight=None,
    orientations=None,
):
    """
    Load catalog and select frequency pairs within [fmin, fmax] (kHz).
    Optionally restrict orientations and/or choose top-N by amp_weight.
    Returns a list of dicts ready to seed the fit.
    """
    recs = load_catalog(path_json)
    sel = select_records(recs, fmin_kHz=fmin_kHz, fmax_kHz=fmax_kHz, orientations=orientations)

    if top_by_weight is not None and len(sel) > top_by_weight:
        sel = sorted(sel, key=lambda r: r.get("amp_weight", 1.0), reverse=True)[
            :top_by_weight
        ]
    return sel


# ----------------------------------------------------------------------
# 4) Expected spectra (theory-only)
# ----------------------------------------------------------------------
def lines_from_recs(
    recs,
    orientations=None,
    fmin_kHz: float = 15.0,
    fmax_kHz: float = 15000.0,
):
    """
    Return arrays: freqs_kHz (2 per site), amp_weight (same length),
    and site_idx (one per pair).
    """
    ori_set = {tuple(o) for o in orientations} if orientations else None

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
        if (fmin_kHz <= fm <= fmax_kHz) and (fmin_kHz <= fp <= fmax_kHz):
            w = float(r.get("kappa", 1.0))
            freqs += [fm, fp]
            weights += [w, w]
            site_idx += [i, i]

    return np.array(freqs, float), np.array(weights, float), np.array(site_idx, int)


def expected_spectrum_kHz(recs, p_occ: float = 0.011):
    """
    Simple discrete spectrum: each (f-, f+) gets weight p_occ * amp_weight.
    Returns (freqs_kHz, weights).
    """
    freqs, w, _ = lines_from_recs(recs, fmin_kHz=0.0, fmax_kHz=np.inf)
    weights = p_occ * (w if w.size else np.array([], float))
    return freqs, weights


def plot_sorted_expected_sticks(
    recs,
    p_occ: float = 0.011,
    f_range_kHz=(15, 15000),
):
    """
    Plot frequencies sorted by rank (like a 'spectrum CDF' on log-y).
    """
    fk, wt = expected_spectrum_kHz(recs, p_occ=p_occ)
    m = (
        np.isfinite(fk)
        & (fk >= f_range_kHz[0])
        & (fk <= f_range_kHz[1])
        & (wt > 0)
    )
    fk = fk[m]

    order = np.argsort(fk)
    fk_s = fk[order]
    idx = np.arange(1, fk_s.size + 1)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(idx, fk_s, ".", ms=2, label="Expected sticks")
    ax.set_yscale("log", base=10)
    ax.set_ylim(*f_range_kHz)
    ax.set_xlabel("Rank (sorted by frequency)")
    ax.set_ylabel("Frequency (kHz)")
    ax.set_title("Sorted expected ESEEM sticks")
    ax.yaxis.set_major_locator(mticker.LogLocator(base=10))
    ax.yaxis.set_minor_locator(mticker.LogLocator(base=10, subs=np.arange(2, 10)))
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax.grid(True, which="both", axis="y", alpha=0.3)
    ax.legend(framealpha=0.85)
    plt.tight_layout()
    plt.show()


def expected_spectrum_kHz_split(recs, p_occ: float = 0.011):
    """
    Return expected frequencies/weights for f0 (f_minus) and f1 (f_plus) separately.
    """
    f0_list, w0_list = [], []
    f1_list, w1_list = [], []

    for r in recs:
        w = p_occ * float(r.get("amp_weight", 1.0))

        f0 = r["f_minus_Hz"] / 1e3  # kHz
        f1 = r["f_plus_Hz"]  / 1e3  # kHz

        f0_list.append(f0); w0_list.append(w)
        f1_list.append(f1); w1_list.append(w)

    return (
        np.array(f0_list, float), np.array(w0_list, float),
        np.array(f1_list, float), np.array(w1_list, float),
    )


def plot_sorted_expected_sticks_split(
    recs,
    p_occ: float = 0.011,
    f_range_kHz=(15, 15000),
):
    """
    Plot sorted frequencies for f0 (f_minus) and f1 (f_plus) separately.

    Y-axis: frequency (log scale)
    X-axis: rank within each set (f0, f1).
    """
    f0, w0, f1, w1 = expected_spectrum_kHz_split(recs, p_occ=p_occ)

    # masks + sorting for each branch separately
    m0 = (
        np.isfinite(f0)
        & (f0 >= f_range_kHz[0])
        & (f0 <= f_range_kHz[1])
        & (w0 > 0)
    )
    m1 = (
        np.isfinite(f1)
        & (f1 >= f_range_kHz[0])
        & (f1 <= f_range_kHz[1])
        & (w1 > 0)
    )

    f0_s = np.sort(f0[m0])
    f1_s = np.sort(f1[m1])

    idx0 = np.arange(1, f0_s.size + 1)
    idx1 = np.arange(1, f1_s.size + 1)

    fig, ax = plt.subplots(figsize=(8, 6))

    # f0 (f_minus) as one color
    ax.plot(idx0, f0_s, ".", ms=2, label="Expected f0 (f_minus)")

    # f1 (f_plus) as another color
    ax.plot(idx1, f1_s, ".", ms=2, label="Expected f1 (f_plus)")

    ax.set_yscale("log", base=10)
    ax.set_ylim(*f_range_kHz)
    ax.set_xlabel("Rank (sorted within each branch)")
    ax.set_ylabel("Frequency (kHz)")
    ax.set_title("Sorted expected ESEEM sticks (f0 vs f1)")

    ax.yaxis.set_major_locator(mticker.LogLocator(base=10))
    ax.yaxis.set_minor_locator(mticker.LogLocator(base=10, subs=np.arange(2, 10)))
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax.grid(True, which="both", axis="y", alpha=0.3)
    ax.legend(framealpha=0.85)

    plt.tight_layout()
    plt.show()


def expected_stick_spectrum_from_recs(
    recs,
    p_occ: float = 0.011,
    orientations=None,
    f_range_kHz=(150, 20000),
    use_weights: bool = True,
    merge_tol_kHz: float = 2.0,
    normalize: bool = False,
    overlay_convolved=None,
):
    """
    Discrete 'expected' spectrum:
      intensity at each line = p_occ * amp_weight (if present).
    Lines closer than merge_tol_kHz are merged to avoid double-plotting.

    Returns
    -------
    f_stick, a_stick, fig, ax
      So you can overlay experimental frequencies if desired.
    """
    freqs, w, _ = lines_from_recs(recs, orientations, *f_range_kHz)
    if freqs.size == 0:
        raise ValueError("No lines in range.")

    amps = p_occ * (w if use_weights else np.ones_like(w))

    # sort by frequency
    order = np.argsort(freqs)
    f = freqs[order]
    a = amps[order]

    # merge close-by lines
    f_merged = []
    a_merged = []
    if f.size:
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
    ax.vlines(f_stick, 0.0, a_stick, linewidth=1.0, alpha=0.9, label="Expected sticks")

    if overlay_convolved is not None:
        xf, yf = overlay_convolved()
        ax.plot(xf, yf, lw=1.6, alpha=0.7, label="Convolved (overlay)")

    ax.set_xlim(*f_range_kHz)
    ax.set_xlabel("Frequency (kHz)")
    ax.set_ylabel("Expected intensity (arb.)" + (" (normalized)" if normalize else ""))
    title = f"Discrete expected spectrum (p_occ={p_occ:.3f})"
    if orientations:
        title += f" • orientations={list(map(tuple, orientations))}"
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(framealpha=0.85)
    plt.tight_layout()
    plt.show()

    return f_stick, a_stick, fig, ax


# ----------------------------------------------------------------------
# 5) Matching experiment to catalog
# ----------------------------------------------------------------------
def match_exp_pairs_to_catalog(
    exp_pairs_kHz,
    recs,
    tol_kHz: float = 5.0,
):
    """
    Given experimental (f0, f1) pairs, find the nearest catalog (f-, f+) pair.

    We compare both orderings:
        dist1 = sqrt((f0 - f_minus)^2 + (f1 - f_plus)^2)
        dist2 = sqrt((f0 - f_plus)^2 + (f1 - f_minus)^2)
    and take the smaller.

    Parameters
    ----------
    exp_pairs_kHz : array-like, shape (N,2)
        Experimental frequencies (kHz) from your fits (osc_f0, osc_f1).
    recs : list of dict
        Catalog records (output of select_records or load_candidates).
    tol_kHz : float
        Maximum distance in kHz to consider a 'match'. Larger distances will
        be marked as 'no_match'.

    Returns
    -------
    df : pandas.DataFrame
        One row per experimental pair, with best matching catalog site info.
    """
    exp_pairs = np.asarray(exp_pairs_kHz, float)
    if exp_pairs.ndim == 1:
        exp_pairs = exp_pairs.reshape(-1, 2)

    # precompute catalog frequencies
    cat_fm = np.array([r["f_minus_Hz"] for r in recs]) / 1e3
    cat_fp = np.array([r["f_plus_Hz"] for r in recs]) / 1e3

    rows = []
    for idx, (f0, f1) in enumerate(exp_pairs):
        if not (np.isfinite(f0) and np.isfinite(f1)):
            rows.append(
                {
                    "nv_index": idx,
                    "f0_exp_kHz": f0,
                    "f1_exp_kHz": f1,
                    "match": False,
                    "dist_kHz": np.nan,
                }
            )
            continue

        # distance to each catalog site (two orderings)
        d1 = np.sqrt((f0 - cat_fm) ** 2 + (f1 - cat_fp) ** 2)
        d2 = np.sqrt((f0 - cat_fp) ** 2 + (f1 - cat_fm) ** 2)
        d = np.minimum(d1, d2)
        j = int(np.argmin(d))
        d_best = float(d[j])

        rec = recs[j]
        matched = d_best <= tol_kHz

        row = {
            "nv_index": idx,
            "f0_exp_kHz": float(f0),
            "f1_exp_kHz": float(f1),
            "match": matched,
            "dist_kHz": d_best,
            "f_minus_cat_kHz": float(cat_fm[j]),
            "f_plus_cat_kHz": float(cat_fp[j]),
            "orientation": tuple(rec["orientation"]),
            "site_index": int(rec["site_index"]),
            "distance_A": float(rec["distance_A"]),
            # "A_par_kHz": rec["A_par_Hz"] / 1e3,
            # "A_perp_kHz": rec["A_perp_Hz"] / 1e3,
            # "amp_weight": float(rec.get("amp_weight", 1.0)),
        }
        rows.append(row)

    return pd.DataFrame(rows)


def overlay_exp_on_expected_spectrum(
    recs,
    exp_pairs_kHz,
    p_occ: float = 0.011,
    f_range_kHz=(150, 20000),
    merge_tol_kHz: float = 2.0,
    orientations=None,
    use_weights: bool = True,
):
    """
    Plot expected stick spectrum from catalog and overlay experimental (f0,f1).

    Parameters
    ----------
    recs : list of dict
        Catalog records (from select_records / load_catalog).
    exp_pairs_kHz : array-like, shape (N,2) or (2,)
        Experimental frequencies (kHz), e.g. (osc_f0, osc_f1) per NV.
    p_occ : float
        13C occupancy probability.
    f_range_kHz : (float, float)
        X-range to show on the log axis (kHz).
    merge_tol_kHz : float
        Merge nearby theory lines within this tolerance.
    orientations : list, optional
        If not None, restrict catalog to these NV orientations.
    use_weights : bool
        If True, scale line heights by amp_weight.
    """
    # First: theory sticks (this already does plt.show() in your version),
    # so we call a slightly modified variant that DOESN'T show immediately.
    freqs, w, _ = lines_from_recs(recs, orientations, *f_range_kHz)
    if freqs.size == 0:
        raise ValueError("No theoretical lines in range.")

    amps = p_occ * (w if use_weights else np.ones_like(w))

    # sort
    order = np.argsort(freqs)
    f = freqs[order]
    a = amps[order]

    # merge nearby lines
    f_merged = []
    a_merged = []
    if f.size:
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

    # --- make the plot ---
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_xscale("log")

    # theory: thin blue sticks
    ax.vlines(f_stick, 0.0, a_stick, linewidth=1.0, alpha=0.7, label="Theory (catalog)")

    ax.set_xlim(*f_range_kHz)
    ax.set_xlabel("Frequency (kHz)")
    ax.set_ylabel("Intensity (arb.)")
    ax.set_title(f"Expected vs experimental ESEEM (p_occ={p_occ:.3f})")
    ax.grid(True, which="both", alpha=0.25)

    # --- overlay experiment ---
    exp_pairs = np.asarray(exp_pairs_kHz, float)
    if exp_pairs.ndim == 1:
        exp_pairs = exp_pairs.reshape(-1, 2)

    # flatten to a 1D set of frequencies
    freqs_exp = np.sort(exp_pairs, axis=1).ravel()

    # choose a y-height for experimental lines: e.g. 80% of max theory height
    y_max = float(a_stick.max()) if a_stick.size else 1.0
    exp_height = 0.8 * y_max

    for f in freqs_exp:
        if f_range_kHz[0] <= f <= f_range_kHz[1]:
            ax.vlines(
                f,
                0.0,
                exp_height,
                colors="C3",
                linestyles="--",
                linewidth=2.0,
                alpha=0.9,
            )

    # single legend entry for all experimental lines
    ax.plot([], [], "--", color="C3", linewidth=2.0, label="Experiment (f0, f1)")

    ax.legend(framealpha=0.85)
    fig.tight_layout()
    plt.show()

    return fig, ax

# ----------------------------------------------------------------------
# Example usage
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # 1) (optional) Build catalog once
    # catalog = build_essem_catalog(
    #     hyperfine_path="analysis/nv_hyperfine_coupling/nv-2.txt",
    #     B_lab_vec=B_vec_T,            # ensure units match gamma_n_Hz_per_T
    #     gamma_n_Hz_per_T=10.705e6,
    #     distance_max_A=15.0,
    #     ms=-1,
    #     phi_deg=0.0,
    #     out_json="essem_freq_catalog_22A.json",
    #     out_csv="essem_freq_catalog_22A.csv",
    # )

    # 2) For analysis: load + select band
    recs = select_records(
        load_catalog("analysis/spin_echo_work/essem_freq_kappa_catalog_22A.json"),
        fmin_kHz=15,
        fmax_kHz=15000,
        orientations=None,
    )

    # Theory-only visualizations
    # plot_sorted_expected_sticks(recs, p_occ=0.011, f_range_kHz=(15, 1500))
    
    f_stick, a_stick, fig, ax = expected_stick_spectrum_from_recs(
        recs,
        p_occ=0.011,
        orientations=None,
        f_range_kHz=(15, 15000),
        use_weights=True,
        merge_tol_kHz=2.0,
        normalize=False,
    )

    # 3) Example: compare experiment to catalog
    #    Suppose for this NV, fitter gave osc_f0, osc_f1 in kHz:
    # exp_pairs_kHz = [(232.0, 125.0), (490.0, 384.0), (69.6, 36.76),(753.89, 647.49)]
    # matches_df = match_exp_pairs_to_catalog(exp_pairs_kHz, recs, tol_kHz=5.0)
    # print(matches_df.head())

    # # 4) Overlay experiment on expected spectrum (visual check)
    # overlay_exp_on_expected_spectrum(recs, exp_pairs_kHz, p_occ=0.011,
    #                                  f_range_kHz=(15, 6000), merge_tol_kHz=2.0)

    # plot_sorted_expected_sticks(recs, p_occ=0.011, f_range_kHz=(15, 15000))

    # expected_stick_spectrum_from_recs(
    #     recs, p_occ=0.011, orientations=None,
    #     f_range_kHz=(150, 20000), use_weights=True,
    #     merge_tol_kHz=2.0, normalize=False
    # )
    # plot_sorted_expected_sticks_split(recs, p_occ=0.011, f_range_kHz=(15, 15000))