import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from analysis.sc_c13_hyperfine_sim_data_driven import read_hyperfine_table_safe

# -------------------------------------------------------------------
# 1) NV-axis C3v symmetry operations (in NV frame)
# -------------------------------------------------------------------


def build_c3v_ops():
    """
    Return a list of 3x3 rotation/reflection matrices representing
    an approximate C3v point group about the NV z-axis in the *NV frame*.

    - 3 rotations about z_NV by 0, 120, 240 degrees
    - 3 mirror reflections (mirror in xz-plane and rotated copies)
    """
    ops = []

    # 3-fold rotations about the NV z-axis
    for k in range(3):
        phi = 2.0 * np.pi * k / 3.0
        c = np.cos(phi)
        s = np.sin(phi)
        Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        ops.append(Rz)

    # mirror in xz-plane (y -> -y), then rotate to get other mirror planes
    M = np.diag([1.0, -1.0, 1.0])
    for k in range(3):
        phi = 2.0 * np.pi * k / 3.0
        c = np.cos(phi)
        s = np.sin(phi)
        Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        ops.append(Rz @ M)

    return ops


# -------------------------------------------------------------------
# 2) Group NV-frame positions into symmetry orbits
# -------------------------------------------------------------------


def find_c3v_orbits_from_nv2(
    hyperfine_path: str,
    r_max_A: float = 10.0,
    tol_r_A: float = 1e-2,
    tol_dir: float = 5e-2,
    xcol: str = "x",
    ycol: str = "y",
    zcol: str = "z",
):
    """
    Use only the *geometry* of the NV-2 hyperfine table (x_A,y_A,z_A in NV frame)
    to group sites into symmetry-equivalent shells under an approximate C3v group.

    Parameters
    ----------
    hyperfine_path : str
        Path to nv-2 hyperfine table (same file you use elsewhere).
    r_max_A : float
        Only include sites with distance <= r_max_A.
    tol_r_A : float
        Tolerance in radius (Å) for considering two sites as same shell.
    tol_dir : float
        Tolerance on direction (norm of difference of unit vectors) for
        identifying symmetry-equivalent positions.
    xcol, ycol, zcol : str
        Column names for NV-frame coordinates in Å.

    Returns
    -------
    orbit_df : pandas.DataFrame
        One row per orbit (shell) with columns:
          - orbit_id
          - n_equiv_theory          (multiplicity from symmetry)
          - rep_site_index          (representative nv-2 site index)
          - r_A                     (radius of shell)
          - theta_deg               (angle to NV axis)
          - site_indices            (list of all member indices)
    """
    df = read_hyperfine_table_safe(hyperfine_path).copy()

    # Require coordinates to exist
    for col in (xcol, ycol, zcol):
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in hyperfine table.")

    # Positions in NV frame, select sites within radius cutoff
    pos = df[[xcol, ycol, zcol]].to_numpy(float)
    r = np.linalg.norm(pos, axis=1)

    keep = r <= float(r_max_A)
    pos = pos[keep]
    r = r[keep]
    site_idx = np.arange(df.shape[0])[keep]

    # Precompute unit vectors (direction on sphere)
    unit = np.zeros_like(pos)
    nonzero = r > 0
    unit[nonzero] = (pos[nonzero].T / r[nonzero]).T

    ops = build_c3v_ops()
    used = np.zeros(pos.shape[0], dtype=bool)

    orbits = []

    for i in range(pos.shape[0]):
        if used[i]:
            continue

        # Representative site for this orbit
        r0 = r[i]
        u0 = unit[i]

        members = [i]
        used[i] = True

        # Test all other sites for membership in this orbit
        for j in range(i + 1, pos.shape[0]):
            if used[j]:
                continue

            # Rough radius match
            if abs(r[j] - r0) > tol_r_A:
                continue

            uj = unit[j]
            # Check if any symmetry operation connects u0 -> uj
            is_equiv = False
            for U in ops:
                u_rot = U @ u0
                if np.linalg.norm(u_rot - uj) < tol_dir:
                    is_equiv = True
                    break

            if is_equiv:
                used[j] = True
                members.append(j)

        # Build orbit entry
        members = np.asarray(members, int)
        n_eq = members.size

        # Representative = first member
        rep_idx = members[0]
        r_shell = r[rep_idx]

        # angle between position and NV axis (z_NV)
        zhat = np.array([0.0, 0.0, 1.0])
        cos_th = np.clip(np.dot(unit[rep_idx], zhat), -1.0, 1.0)
        theta_deg = float(np.degrees(np.arccos(cos_th)))

        orbits.append(
            dict(
                orbit_id=len(orbits),
                n_equiv_theory=int(n_eq),
                rep_site_index=int(site_idx[rep_idx]),
                r_A=float(r_shell),
                theta_deg=theta_deg,
                site_indices=[int(site_idx[k]) for k in members],
            )
        )

    orbit_df = pd.DataFrame(orbits).sort_values("r_A").reset_index(drop=True)
    return orbit_df


def build_site_multiplicity_with_theory(
    matches_df: pd.DataFrame,
    orbit_df: pd.DataFrame,
    p13: float = 0.011,
):
    """
    Combine experimental multiplicity (how many NVs matched a site)
    with theoretical multiplicity (how many symmetry-equivalent sites per shell),
    and include:
      - list of symmetry-equivalent sites that are actually occupied
      - total number of NVs across those equivalent sites
      - mean matched frequencies per site (if present in matches_df)

    Parameters
    ----------
    matches_df : DataFrame
        Output of run_full_essem_match_analysis(...), must contain:
          - nv_label
          - site_index
          - orientation
          - x_A, y_A, z_A
          - distance_A
          - kappa
        Optionally (for frequency info):
          - f0_kHz, f1_kHz  (or whatever you named them)
    orbit_df : DataFrame
        Output of your symmetry classification, must contain:
          - orbit_id
          - n_equiv_theory
          - site_indices (list of ints for that orbit)
    p13 : float
        13C fraction (natural ~0.011).

    Returns
    -------
    site_stats : DataFrame
        One row per unique matched site with columns:
          - site_index, orientation, x_A,y_A,z_A,distance_A
          - n_matches (how many NVs matched this site)
          - frac_NV (n_matches / N_NV)
          - kappa_mean
          - (optional) f0_kHz_mean, f1_kHz_mean
          - orbit_id, n_equiv_theory, theta_deg (if available)
          - p_shell (probability that this shell is occupied by ≥1 13C)
          - E_n_matches (expected number of NVs showing this shell)
          - match_ratio (n_matches / E_n_matches)
          - equiv_occupied_sites (list of symmetry-equivalent sites that are occupied)
          - n_equiv_occupied (# of occupied symmetry-equivalent sites)
          - n_matches_equiv_total (total NVs across those occupied equivalent sites)
    """

    # --------- 0) How many NVs are in the dataset? ---------
    N_NV = matches_df["nv_label"].nunique()
    print(f"[INFO] Number of NVs in dataset: N_NV = {N_NV}")

    # --------- 1) Experimental site stats (NV → site multiplicity) ---------
    site_key = [
        "site_index",
        "orientation",
        "x_A",
        "y_A",
        "z_A",
        "distance_A",
    ]

    # Build aggregation dict, adding frequency averages if present
    agg_dict = {
        "n_matches": ("nv_label", "count"),
        "kappa_mean": ("kappa", "mean"),
    }
    if "f0_fit_kHz" in matches_df.columns:
        agg_dict["f0_kHz_mean"] = ("f0_fit_kHz", "mean")
    if "f1_fit_kHz" in matches_df.columns:
        agg_dict["f1_kHz_mean"] = ("f1_fit_kHz", "mean")

    site_stats = matches_df.groupby(site_key, as_index=False).agg(**agg_dict)

    site_stats["frac_NV"] = site_stats["n_matches"] / float(N_NV)
    print(f"[INFO] Experimental unique sites: {len(site_stats)}")

    # --------- 2) Build site_index → (orbit_id, n_equiv_theory, theta_deg) map ---------
    rows = []
    for _, row in orbit_df.iterrows():
        orbit_id = int(row["orbit_id"])
        n_equiv = int(row["n_equiv_theory"])
        theta_deg = float(row.get("theta_deg", np.nan))
        site_list_raw = row["site_indices"]  # list of ints for that orbit

        for sid in site_list_raw:
            rows.append(
                dict(
                    site_index=int(sid),
                    orbit_id=orbit_id,
                    n_equiv_theory=n_equiv,
                    theta_deg=theta_deg,
                )
            )

    orbit_map = pd.DataFrame(rows)
    print(f"[INFO] Orbit map has {len(orbit_map)} site entries.")

    # --------- 3) Join experimental and theory info ---------
    site_stats = site_stats.merge(
        orbit_map,
        on="site_index",
        how="left",
    )

    # --------- 3a) Attach full orbit membership to each row ---------
    # orbit_id -> list of member site indices
    orbit_members = (
        orbit_df[["orbit_id", "site_indices"]]
        .set_index("orbit_id")["site_indices"]
        .to_dict()
    )

    site_stats["orbit_site_indices"] = site_stats["orbit_id"].map(orbit_members)

    # --------- 3b) From experiment: which of those orbit members are 'occupied'? ---------
    # Build a quick lookup: site_index -> n_matches
    nmatch_map = site_stats.groupby("site_index")["n_matches"].max().to_dict()

    def _occupied_members(row):
        members = row["orbit_site_indices"]
        # Some rows have NaN (float) here if there's no orbit match
        if not isinstance(members, (list, tuple, np.ndarray)):
            return []
        return [int(s) for s in members if nmatch_map.get(int(s), 0) > 0]

    site_stats["equiv_occupied_sites"] = site_stats.apply(_occupied_members, axis=1)
    site_stats["n_equiv_occupied"] = site_stats["equiv_occupied_sites"].apply(len)

    # Total number of NVs across those occupied equivalent sites
    def _total_matches_equiv(row):
        members = row["equiv_occupied_sites"]
        if not isinstance(members, (list, tuple, np.ndarray)) or len(members) == 0:
            return 0
        return int(sum(nmatch_map.get(int(s), 0) for s in members))

    site_stats["n_matches_equiv_total"] = site_stats.apply(_total_matches_equiv, axis=1)

    # --------- 4) Simple theory expectation for n_matches ---------
    # For one NV and one shell with n_equiv_theory sites at 13C fraction p13:
    #   p_shell = 1 - (1 - p13) ** n_equiv_theory  ≈ n_equiv_theory * p13 for small p13
    n_eq = site_stats["n_equiv_theory"].fillna(1).astype(float)
    p_shell = 1.0 - (1.0 - p13) ** n_eq

    site_stats["p_shell"] = p_shell
    site_stats["E_n_matches"] = (
        N_NV * p_shell
    )  # expected # of NVs with that shell occupied

    # Ratio of observed to expected
    site_stats["match_ratio"] = site_stats["n_matches"] / site_stats["E_n_matches"]

    return site_stats


#### plot helper
def mutliplicity_plots(site_stats_full):
    ss = site_stats_full.copy()
    plt.figure(figsize=(6, 5))
    sc = plt.scatter(
        ss["distance_A"],
        ss["n_matches"],
        c=ss["kappa_mean"],
        s=20,
    )
    plt.xlabel("Distance to NV (Å)", fontsize=15)
    plt.ylabel("Number of NVs matched to this site", fontsize=15)
    cbar = plt.colorbar(sc)
    cbar.set_label(r"$\langle \kappa \rangle$")
    plt.title("Site multiplicity vs. distance (per site)", fontsize=15)

    ss = site_stats_full.copy()

    plt.figure(figsize=(6, 6))
    plt.scatter(
        ss["E_n_matches"],
        ss["n_matches"],
        s=20,
        alpha=0.7,
    )
    max_val = max(ss["E_n_matches"].max(), ss["n_matches"].max()) * 1.1
    plt.plot([0, max_val], [0, max_val], "k--", linewidth=1)  # identity line

    plt.xlabel("Expected #NVs with this shell occupied (E_n_matches)", fontsize=15)
    plt.ylabel("Observed #NVs matched (n_matches)", fontsize=15)
    plt.xscale("log")
    plt.yscale("log")
    plt.title("Observed vs expected site multiplicity", fontsize=15)

    # Build orbit-level stats
    orbit_stats = site_stats_full.groupby("orbit_id", as_index=False).agg(
        r_A=("distance_A", "mean"),
        theta_deg=("theta_deg", "mean"),
        n_equiv_theory=("n_equiv_theory", "max"),
        n_equiv_occupied=("n_equiv_occupied", "max"),
        n_matches_equiv_total=("n_matches_equiv_total", "max"),
    )

    # Fraction of symmetry-equivalent sites that are actually occupied
    orbit_stats["frac_occupied_sites"] = (
        orbit_stats["n_equiv_occupied"] / orbit_stats["n_equiv_theory"]
    )

    plt.figure(figsize=(6, 5))
    sc = plt.scatter(
        orbit_stats["r_A"],
        orbit_stats["n_matches_equiv_total"],
        c=orbit_stats["frac_occupied_sites"],
        s=20,
    )
    plt.xlabel("Radius of shell r (Å)", fontsize=15)
    plt.ylabel("Total NVs matched in this shell (all equivalent sites)", fontsize=13)
    cbar = plt.colorbar(sc)
    cbar.set_label("Fraction of symmetry-equivalent sites occupied", fontsize=13)
    plt.title("Orbit-level occupancy", fontsize=15)

    plt.figure(figsize=(6, 5))
    sc = plt.scatter(
        orbit_stats["theta_deg"],
        orbit_stats["r_A"],
        c=orbit_stats["n_matches_equiv_total"],
        s=20,
    )
    plt.xlabel(r"Angle to NV axis $\theta$ (deg)", fontsize=15)
    plt.ylabel("Radius r (Å)", fontsize=15)
    cbar = plt.colorbar(sc)
    cbar.set_label("Total NVs matched in orbit", fontsize=15)
    plt.title("Topography of active 13C shells", fontsize=15)
    plt.show()

    plt.figure(figsize=(6, 5))
    plt.hist(
        site_stats_full["n_matches"],
        bins=np.arange(1, site_stats_full["n_matches"].max() + 2) - 0.5,
    )
    plt.xlabel("# NVs matched to site", fontsize=15)
    plt.ylabel("Count of sites", fontsize=15)
    plt.title("Distribution of per-site multiplicities", fontsize=15)
    plt.show()

    plt.figure(figsize=(6, 5))
    plt.hist(site_stats_full["match_ratio"], bins=30)
    plt.xlabel("match_ratio = n_matches / E_n_matches", fontsize=15)
    plt.ylabel("Count of sites", fontsize=15)
    plt.title("Over-/under-representation of sites", fontsize=15)
    plt.show()


if __name__ == "__main__":

    # Suppose you already have:
    #   matches_enriched  (from run_full_essem_match_analysis)
    #   orbit_df          (your symmetry-analysis DF)

    HYPERFINE_PATH = "analysis/nv_hyperfine_coupling/nv-2.txt"
