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
    with theoretical multiplicity (how many symmetry-equivalent sites per shell/orbit),
    and include:
      - list of symmetry-equivalent sites that are actually occupied
      - total number of NVs across those equivalent sites
      - per-site and per-orbit expectations

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
        Optionally:
          - f0_fit_kHz, f1_fit_kHz
    orbit_df : DataFrame
        Output of symmetry classification, must contain:
          - orbit_id
          - n_equiv_theory
          - theta_deg
          - site_indices (list of ints for that orbit)
    p13 : float
        13C fraction (natural ~0.011).

    Returns
    -------
    site_stats : DataFrame
        One row per unique matched site with columns:
          # geometry / identity
          - site_index, orientation, x_A,y_A,z_A,distance_A
          - orbit_id, n_equiv_theory, theta_deg (if matched to an orbit)

          # experimental multiplicity (site-level)
          - n_matches             (# NVs matched to this site)
          - frac_NV               (n_matches / N_NV)
          - kappa_mean
          - optional: f0_kHz_mean, f1_kHz_mean

          # symmetry info (orbit-level, attached to each site)
          - orbit_site_indices
          - equiv_occupied_sites
          - n_equiv_occupied
          - n_matches_equiv_total

          # theory expectations (site-level)
          - p_site
          - E_n_matches_site
          - match_ratio_site
          - E_n_matches     (alias of E_n_matches_site, for backward compatibility)
          - match_ratio     (alias of match_ratio_site)

          # theory expectations (orbit-level; constant within an orbit)
          - p_shell              (prob shell/orbit has ≥ 1 occupied site)
          - E_n_matches_shell    (expected # NVs with that shell occupied)
          - match_ratio_shell    (n_matches_equiv_total / E_n_matches_shell)
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

    # --------- 3) Join experimental and theory info (per-site) ---------
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
    # Quick lookup: site_index -> n_matches
    nmatch_map = site_stats.groupby("site_index")["n_matches"].max().to_dict()

    def _occupied_members(row):
        members = row["orbit_site_indices"]
        if not isinstance(members, (list, tuple, np.ndarray)):
            return []
        return [int(s) for s in members if nmatch_map.get(int(s), 0) > 0]

    site_stats["equiv_occupied_sites"] = site_stats.apply(_occupied_members, axis=1)
    site_stats["n_equiv_occupied"] = site_stats["equiv_occupied_sites"].apply(len)

    def _total_matches_equiv(row):
        members = row["equiv_occupied_sites"]
        if not isinstance(members, (list, tuple, np.ndarray)) or len(members) == 0:
            return 0
        return int(sum(nmatch_map.get(int(s), 0) for s in members))

    site_stats["n_matches_equiv_total"] = site_stats.apply(_total_matches_equiv, axis=1)

    # --------- 4) Theory expectation: per-site ---------
    # Each *site* is occupied with probability p13 for a given NV
    site_stats["p_site"] = p13
    site_stats["E_n_matches_site"] = N_NV * p13
    site_stats["match_ratio_site"] = (
        site_stats["n_matches"] / site_stats["E_n_matches_site"]
    )

    # Backward-compatible names
    site_stats["E_n_matches"] = site_stats["E_n_matches_site"]
    site_stats["match_ratio"] = site_stats["match_ratio_site"]

    # --------- 5) Theory expectation: orbit/shell-level ---------
    # First build an orbit-level table from site_stats itself
    orbit_level = (
        site_stats.dropna(subset=["orbit_id"])  # ignore sites not matched to any orbit
        .groupby("orbit_id", as_index=False)
        .agg(
            n_equiv_theory=("n_equiv_theory", "max"),
            n_matches_equiv_total=("n_matches_equiv_total", "max"),
            n_equiv_occupied=("n_equiv_occupied", "max"),
        )
    )

    # Probability that an orbit/shell has ≥ 1 occupied site
    n_eq = orbit_level["n_equiv_theory"].astype(float)
    orbit_level["p_shell"] = 1.0 - (1.0 - p13) ** n_eq

    # Expected # of NVs for which this shell is occupied at least once
    orbit_level["E_n_matches_shell"] = N_NV * orbit_level["p_shell"]

    # Ratio of observed total matches across equivalent sites to shell expectation
    # (This is orbit-level; we will map it onto each site in that orbit)
    with np.errstate(divide="ignore", invalid="ignore"):
        orbit_level["match_ratio_shell"] = (
            orbit_level["n_matches_equiv_total"] / orbit_level["E_n_matches_shell"]
        )

    # Map orbit-level quantities back to each site row
    shell_p_map = orbit_level.set_index("orbit_id")["p_shell"].to_dict()
    shell_E_map = orbit_level.set_index("orbit_id")["E_n_matches_shell"].to_dict()
    shell_R_map = orbit_level.set_index("orbit_id")["match_ratio_shell"].to_dict()

    site_stats["p_shell"] = site_stats["orbit_id"].map(shell_p_map)
    site_stats["E_n_matches_shell"] = site_stats["orbit_id"].map(shell_E_map)
    site_stats["match_ratio_shell"] = site_stats["orbit_id"].map(shell_R_map)

 
    # ---------- NEW: orbit / shell level summary ----------
    orbit_stats = site_stats.groupby("orbit_id", as_index=False).agg(
        r_A=("distance_A", "mean"),
        theta_deg=("theta_deg", "mean"),
        n_equiv_theory=("n_equiv_theory", "max"),
        n_equiv_occupied=("n_equiv_occupied", "max"),
        n_matches_equiv_total=("n_matches_equiv_total", "max"),
        kappa_mean=("kappa_mean", "mean"),
    )

    orbit_stats["frac_occupied_sites"] = (
        orbit_stats["n_equiv_occupied"] / orbit_stats["n_equiv_theory"]
    )

    # Shell-level expectation (defined once here)
    n_eq = orbit_stats["n_equiv_theory"].astype(float)
    orbit_stats["p_shell"] = 1.0 - (1.0 - p13) ** n_eq
    orbit_stats["E_n_matches_shell"] = N_NV * orbit_stats["p_shell"]
    orbit_stats["match_ratio_shell"] = (
        orbit_stats["n_matches_equiv_total"] / orbit_stats["E_n_matches_shell"]
    )

    return site_stats, orbit_stats
    # return site_stats


def plot_orbit_rings_3d(
    orbit_stats,
    color_key="frac_occupied_sites",  # or "n_matches_equiv_total" or "kappa_mean"
    size_key="n_matches_equiv_total",  # or None
    nv_axis_color="b",
    dataset_label=None,  # NEW: e.g. "Experiment" or "Simulation"
):
    """
    3D visualization of C3v orbits as rings around the NV.
    - NV at origin
    - NV axis along +z
    - Each orbit is a ring at fixed radius r_A and polar angle theta_deg
    - Discrete symmetry-equivalent sites marked on each ring

    dataset_label: optional string appended to the title so that
                   experiment vs simulation plots are distinguishable.
    """

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")

    # ----- 1) Choose what sets color -----
    if color_key in orbit_stats.columns:
        col_vals_orbit = orbit_stats[color_key].values.astype(float)
    elif "frac_occupied_sites" in orbit_stats.columns:
        col_vals_orbit = orbit_stats["frac_occupied_sites"].values.astype(float)
        color_key = "frac_occupied_sites"
    else:
        col_vals_orbit = orbit_stats["n_matches_equiv_total"].values.astype(float)
        color_key = "n_matches_equiv_total"

    cmin, cmax = float(col_vals_orbit.min()), float(col_vals_orbit.max())
    if color_key == "frac_occupied_sites":
        cmin, cmax = 0.0, 1.0
        c_label = "Fraction of equivalent sites occupied"
    elif color_key == "kappa_mean":
        c_label = r"$\langle \kappa \rangle$ (ESEEM misalignment)"
    else:
        c_label = "Total NVs matched in orbit"

    # ----- 2) Marker sizes from size_key -----
    if size_key is not None and size_key in orbit_stats.columns:
        size_vals_orbit = orbit_stats[size_key].values.astype(float)
        if size_vals_orbit.max() > 0:
            size_norm = (size_vals_orbit - size_vals_orbit.min()) / (
                size_vals_orbit.max() - size_vals_orbit.min() + 1e-12
            )
        else:
            size_norm = np.zeros_like(size_vals_orbit)
        size_vals_orbit = 10 + 20 * size_norm
    else:
        size_vals_orbit = np.full(len(orbit_stats), 40.0)

    all_x, all_y, all_z, all_c, all_s = [], [], [], [], []

    # ----- 3) Loop over orbits: draw rings + discrete sites -----
    for idx, row in orbit_stats.iterrows():
        r = row["r_A"]
        theta = np.deg2rad(row["theta_deg"])
        n_equiv = int(row["n_equiv_theory"])
        c_val = (
            float(row[color_key]) if color_key in row else float(col_vals_orbit[idx])
        )
        s_val = float(size_vals_orbit[idx])

        # continuous ring
        phi_ring = np.linspace(0, 2 * np.pi, 300)
        x_ring = r * np.sin(theta) * np.cos(phi_ring)
        y_ring = r * np.sin(theta) * np.sin(phi_ring)
        z_ring = r * np.cos(theta) * np.ones_like(phi_ring)
        ax.plot(
            x_ring,
            y_ring,
            z_ring,
            linewidth=0.4,
            alpha=0.25,
            color="gray",
        )

        # discrete symmetry-equivalent sites
        phi_sites = np.linspace(0, 2 * np.pi, n_equiv, endpoint=False)
        x_sites = r * np.sin(theta) * np.cos(phi_sites)
        y_sites = r * np.sin(theta) * np.sin(phi_sites)
        z_sites = r * np.cos(theta) * np.ones_like(phi_sites)

        all_x.append(x_sites)
        all_y.append(y_sites)
        all_z.append(z_sites)
        all_c.append(np.full_like(x_sites, c_val, dtype=float))
        all_s.append(np.full_like(x_sites, s_val, dtype=float))

    X = np.concatenate(all_x)
    Y = np.concatenate(all_y)
    Z = np.concatenate(all_z)
    C = np.concatenate(all_c)
    S = np.concatenate(all_s)

    sc = ax.scatter(
        X,
        Y,
        Z,
        c=C,
        s=S,
        vmin=cmin,
        vmax=cmax,
        alpha=0.9,
        edgecolors="none",
    )
    cbar = plt.colorbar(sc, ax=ax, pad=0.1)
    cbar.set_label(c_label, fontsize=11)

    # ----- 4) NV position + axis -----
    ax.scatter(
        [0],
        [0],
        [0],
        marker="*",
        s=40,
        color=nv_axis_color,
        edgecolors="k",
        linewidths=0.5,
        zorder=5,
    )

    R_max = orbit_stats["r_A"].max() * 1.1
    ax.quiver(
        0,
        0,
        0,
        0,
        0,
        R_max,
        color=nv_axis_color,
        linewidth=1.0,
        arrow_length_ratio=0.08,
    )
    ax.text(
        0,
        0,
        1.05 * R_max,
        "NV axis",
        fontsize=10,
        ha="center",
        va="bottom",
        color=nv_axis_color,
    )

    # ----- 5) Formatting -----
    ax.set_xlabel("x (Å)")
    ax.set_ylabel("y (Å)")
    ax.set_zlabel("z (Å)")

    base_title = r"C$_{3v}$ orbit rings around the NV"
    if dataset_label:
        ax.set_title(f"{base_title} ({dataset_label})", fontsize=14)
    else:
        ax.set_title(base_title, fontsize=14)

    max_range = R_max
    for axis in "xyz":
        getattr(ax, f"set_{axis}lim")((-max_range, max_range))

    ax.view_init(elev=22, azim=40)
    ax.grid(True, alpha=0.5)

    # plt.tight_layout()
    plt.show()


def multiplicity_plots(site_stats_full, dataset_label=None):
    """
    Make a set of diagnostic plots for site multiplicity, shell occupancy,
    and the geometry of active 13C shells around the NV.

    dataset_label: optional string (e.g. "Experiment", "Simulation") that
                   is appended to plot titles to distinguish datasets.
    """
    title_suffix = f" ({dataset_label})" if dataset_label else ""

    ss = site_stats_full.copy()

    # ---------- 1) Per-site: distance vs n_matches, colored by kappa ----------
    plt.figure(figsize=(6, 5))
    sc = plt.scatter(
        ss["distance_A"],
        ss["n_matches"],
        c=ss["kappa_mean"],
        s=15,
    )
    plt.xlabel("Distance to NV (Å)", fontsize=15)
    plt.ylabel("Number of NVs matched to this site", fontsize=15)
    cbar = plt.colorbar(sc)
    cbar.set_label(r"$\langle \kappa \rangle$", fontsize=13)
    plt.title("Site multiplicity vs. distance (per site)" + title_suffix, fontsize=15)
    plt.tight_layout()
    plt.show()

    # ---------- 2) Observed vs expected multiplicity (per-site) ----------
    plt.figure(figsize=(6, 6))
    plt.scatter(
        ss["E_n_matches"],
        ss["n_matches"],
        s=15,
        alpha=0.7,
    )
    max_val = max(ss["E_n_matches"].max(), ss["n_matches"].max()) * 1.1
    plt.plot([0, max_val], [0, max_val], "k--", linewidth=1)

    plt.xlabel("Expected #NVs with this shell occupied (E_n_matches)", fontsize=15)
    plt.ylabel("Observed #NVs matched (n_matches)", fontsize=15)
    plt.xscale("log")
    plt.yscale("log")
    plt.title("Observed vs expected site multiplicity" + title_suffix, fontsize=15)
    plt.tight_layout()
    plt.show()

    # ---------- 3) Orbit-level stats ----------
    orbit_stats = site_stats_full.groupby("orbit_id", as_index=False).agg(
        r_A=("distance_A", "mean"),
        theta_deg=("theta_deg", "mean"),
        n_equiv_theory=("n_equiv_theory", "max"),
        n_equiv_occupied=("n_equiv_occupied", "max"),
        n_matches_equiv_total=("n_matches_equiv_total", "max"),
        kappa_mean=("kappa_mean", "mean"),
        p_shell=("p_shell", "max"),
        E_n_matches_shell=("E_n_matches_shell", "max"),
        match_ratio_shell=("match_ratio_shell", "max"),
    )

    # ---------- 4) Orbit-level occupancy: r vs total matched, colored by fraction ----------
    plt.figure(figsize=(6, 5))
    sc = plt.scatter(
        orbit_stats["r_A"],
        orbit_stats["n_matches_equiv_total"],
        c=orbit_stats["match_ratio_shell"],  # or orbit_stats["match_ratio_shell"]
        s=15,
    )
    plt.xlabel("Radius of shell r (Å)", fontsize=15)
    plt.ylabel("Total NVs matched in this shell (all equivalent sites)", fontsize=13)
    cbar = plt.colorbar(sc)
    cbar.set_label("Fraction of symmetry-equivalent sites occupied", fontsize=13)
    plt.title("Orbit-level occupancy" + title_suffix, fontsize=15)
    plt.tight_layout()
    plt.show()

    # ---------- 5) Polar orbit map ----------
    theta_deg = orbit_stats["theta_deg"].values
    theta_rad = np.deg2rad(theta_deg)
    r_A = orbit_stats["r_A"].values
    weight = orbit_stats["n_matches_equiv_total"].values  # or frac_occupied_sites

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="polar")

    sc = ax.scatter(
        theta_rad,
        r_A,
        c=weight,
        s=18,
    )
    cbar = plt.colorbar(sc, ax=ax, pad=0.1)
    cbar.set_label("Total NVs matched in orbit", fontsize=12)

    # Light rings for shells
    r_rounded = np.round(r_A, 2)
    for r in np.unique(r_rounded):
        phi = np.linspace(0, 2 * np.pi, 400)
        ax.plot(
            phi,
            np.full_like(phi, r),
            linestyle="--",
            linewidth=0.4,
            alpha=0.25,
        )

    ax.set_title("Topography of active $^{13}$C shells" + title_suffix, fontsize=14)
    ax.set_rlabel_position(135)
    ax.tick_params(labelsize=10)
    plt.tight_layout()
    plt.show()

    # ---------- 6) Shell multiplicity (number of orbits per radius) ----------
    ring_counts = (
        orbit_stats.assign(r_rounded=np.round(orbit_stats["r_A"], 2))
        .groupby("r_rounded", as_index=False)
        .size()
        .rename(columns={"size": "n_orbits"})
        .sort_values("r_rounded")
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    markerline, stemlines, baseline = ax.stem(
        ring_counts["r_rounded"],
        ring_counts["n_orbits"],
        basefmt=" ",
        linefmt="-",
        markerfmt="o",
    )
    plt.setp(stemlines, linewidth=1.0, alpha=0.7)
    plt.setp(markerline, markersize=4)

    ax.set_xlabel("Radius r (Å)", fontsize=12)
    ax.set_ylabel("# distinct orbits at this radius", fontsize=12)
    ax.set_title(
        "Shell multiplicity (number of orbits per radius)" + title_suffix, fontsize=13
    )
    ax.tick_params(labelsize=10)
    plt.tight_layout()
    plt.show()

    # ---------- 7) Polar: radius vs effective misalignment ----------
    theta_eff_rad = np.arcsin(np.sqrt(orbit_stats["kappa_mean"].clip(0, 1)))
    r_A = orbit_stats["r_A"].values
    weight = orbit_stats["n_matches_equiv_total"].values

    fig = plt.figure(figsize=(7, 6))
    # fig.subplots_adjust(top=0.82)  # you can tune 0.80–0.88 as needed
    ax = fig.add_subplot(111, projection="polar")

    sc = ax.scatter(
        theta_eff_rad,
        r_A,
        c=weight,
        s=20,
    )

    cbar = plt.colorbar(sc, ax=ax, pad=0.1)
    cbar.set_label("Total NVs matched in orbit", fontsize=12)

    ax.set_title(
        r"Active $^{13}$C orbits: radius vs ESEEM misalignment" + title_suffix,
        fontsize=14,
        pad=20,
    )
    ax.set_thetamin(0)
    ax.set_thetamax(90)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(135)
    # plt.tight_layout()
    plt.show()

    # ---------- 8) ESEEM misalignment vs radius ----------
    plt.figure(figsize=(6, 5))
    plt.scatter(
        orbit_stats["r_A"],
        orbit_stats["kappa_mean"],
        c=orbit_stats["n_matches_equiv_total"],
        s=30,
    )
    plt.xlabel("Radius r (Å)")
    plt.ylabel(r"Mean ESEEM misalignment $\langle\kappa\rangle$")
    cbar = plt.colorbar()
    cbar.set_label("Total NVs matched in orbit")
    plt.title(r"ESEEM misalignment vs distance" + title_suffix)
    plt.tight_layout()
    plt.show()

    # ---------- 9) Cartesian: r vs theta ----------
    plt.figure(figsize=(6, 5))
    sc = plt.scatter(
        theta_deg,
        r_A,
        c=weight,
        s=15,
    )
    plt.xlabel(r"Angle to NV axis $\theta$ (deg)", fontsize=15)
    plt.ylabel("Radius r (Å)", fontsize=15)
    cbar = plt.colorbar(sc)
    cbar.set_label("Total NVs matched in orbit", fontsize=15)
    plt.title("Topography of active $^{13}$C shells" + title_suffix, fontsize=15)
    plt.tight_layout()
    plt.show()

    # ---------- 10) Which orbits contribute most ESEEM signal? ----------
    plt.figure(figsize=(6, 5))
    plt.scatter(
        orbit_stats["kappa_mean"],
        orbit_stats["n_matches_equiv_total"],
        s=20,
        alpha=0.7,
    )
    plt.xlabel(r"Mean misalignment $\langle\kappa\rangle$")
    plt.ylabel("Total NVs matched in orbit")
    plt.title(r"Which orbits contribute most ESEEM signal?" + title_suffix)
    plt.tight_layout()
    plt.show()

    # ---------- 11) Histogram: per-site multiplicities ----------
    plt.figure(figsize=(6, 5))
    bins = np.arange(1, site_stats_full["n_matches"].max() + 2) - 0.5
    plt.hist(site_stats_full["n_matches"], bins=bins)
    plt.xlabel("# NVs matched to site", fontsize=15)
    plt.ylabel("Count of sites", fontsize=15)
    plt.title("Distribution of per-site multiplicities" + title_suffix, fontsize=15)
    plt.tight_layout()
    plt.show()

    # ---------- 12) Histogram: match_ratio ----------
    plt.figure(figsize=(6, 5))
    plt.hist(site_stats_full["match_ratio"], bins=30)
    plt.xlabel("match_ratio = n_matches / E_n_matches", fontsize=15)
    plt.ylabel("Count of sites", fontsize=15)
    plt.title("Over-/under-representation of sites" + title_suffix, fontsize=15)
    plt.tight_layout()
    plt.show()

    # ---------- 13) 3D orbit rings ----------
    plot_orbit_rings_3d(
        orbit_stats,
        color_key="frac_occupied_sites",
        dataset_label=dataset_label,
    )

    plot_orbit_rings_3d(
        orbit_stats,
        color_key="kappa_mean",
        size_key="n_matches_equiv_total",
        dataset_label=dataset_label,
    )


def make_a_table(site_stats_full, topN=15, dataset_label=None):
    # --- 1) Choose which columns & rows to show ---
    cols_to_show = [
        "site_index",
        "orientation",
        "distance_A",
        "n_matches",
        "n_equiv_theory",
        "p_shell",
        "E_n_matches",
        "match_ratio",
        "kappa_mean",
        "f0_kHz_mean",
        "f1_kHz_mean",
        "equiv_occupied_sites",
        "n_equiv_occupied",
        "n_matches_equiv_total",
    ]
    table_df = (
        site_stats_full.sort_values("n_matches", ascending=False)[cols_to_show]
        .head(topN)
        .copy()
    )

    # --- 2) Format numeric columns nicely ---
    float_cols_3 = ["distance_A", "p_shell", "E_n_matches", "match_ratio", "kappa_mean"]
    float_cols_1 = ["f0_kHz_mean", "f1_kHz_mean"]

    for col in float_cols_3:
        if col in table_df.columns:
            table_df[col] = table_df[col].map(lambda x: f"{x:.3f}")

    for col in float_cols_1:
        if col in table_df.columns:
            table_df[col] = table_df[col].map(lambda x: f"{x:.1f}")

    def _fmt_equiv_sites(val):
        if isinstance(val, (list, tuple)):
            return ",".join(str(int(v)) for v in val)
        return (
            ""
            if (val is None or (isinstance(val, float) and np.isnan(val)))
            else str(val)
        )

    if "equiv_occupied_sites" in table_df.columns:
        table_df["equiv_occupied_sites"] = table_df["equiv_occupied_sites"].apply(
            _fmt_equiv_sites
        )

    # --- 3) Pretty column labels ---
    col_labels = [
        "site\nindex",
        "orientation",
        r"distance\n($\mathrm{\AA}$)",
        r"$n_{\mathrm{matches}}$",
        r"$n_{\mathrm{eq}}$\n(theory)",
        r"$p_{\mathrm{shell}}$",
        r"$\mathbb{E}[n_{\mathrm{matches}}]$",
        r"match\nratio",
        r"$\bar{\kappa}$",
        r"$\bar f_0$\n(kHz)",
        r"$\bar f_1$\n(kHz)",
        "equiv.\noccupied\nsites",
        r"$n_{\mathrm{occ}}^{\mathrm{equiv}}$",
        r"$n_{\mathrm{matches}}^{\mathrm{equiv}}$",
    ]
    assert len(col_labels) == len(cols_to_show)

    # --- 4) Build table figure ---
    n_rows, n_cols = table_df.shape
    fig_w = max(10, 0.85 * n_cols)
    fig_h = max(4.0, 0.45 * (n_rows + 2))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    table = ax.table(
        cellText=table_df.values,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.scale(1.0, 1.25)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#e0e0e0")

    base_title = (
        "Top sites by experimental multiplicity and symmetry-adjusted expectation"
    )
    if dataset_label:
        title = f"{base_title} ({dataset_label})"
    else:
        title = base_title

    equations = (
        r"$p_{\mathrm{shell}} = 1 - (1 - p_{13})^{n_{\mathrm{eq}}},\quad "
        r"\mathbb{E}[n_{\mathrm{matches}}] = N_{\mathrm{NV}}\; p_{\mathrm{shell}},\quad "
        r"R = \dfrac{n_{\mathrm{matches}}}{\mathbb{E}[n_{\mathrm{matches}}]}$"
    )

    ax.set_title(title, pad=30, fontsize=18)

    fig.text(
        0.5,
        0.85,
        equations,
        ha="center",
        va="center",
        fontsize=15,
    )

    legend_text = (
        r"$n_{\mathrm{occ}}^{\mathrm{equiv}}$: # of symmetry-equivalent sites "
        r"that are occupied in the dataset; "
        r"$n_{\mathrm{matches}}^{\mathrm{equiv}}$: total matches summed over those sites."
    )
    fig.text(
        0.5,
        0.80,
        legend_text,
        ha="center",
        va="center",
        fontsize=11,
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    # Suppose you already have:
    #   matches_enriched  (from run_full_essem_match_analysis)
    #   orbit_df          (your symmetry-analysis DF)

    HYPERFINE_PATH = "analysis/nv_hyperfine_coupling/nv-2.txt"
