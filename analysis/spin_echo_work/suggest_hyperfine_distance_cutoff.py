import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import kplotlib as kpl
# ---------------------------------------------------------------------
# CONFIG / PATHS
# ---------------------------------------------------------------------

CATALOG_JSON = "analysis/spin_echo_work/essem_freq_kappa_catalog_22A_updated.json"

# ---------------------------------------------------------------------
# CATALOG LOADING
# ---------------------------------------------------------------------

def load_catalog_df(path_json: str = CATALOG_JSON) -> pd.DataFrame:
    """
    Load the ESEEM catalog JSON into a pandas DataFrame and normalize
    column names (distance_A, kappa, etc.).
    """
    with open(path_json, "r") as f:
        recs = json.load(f)

    df = pd.DataFrame(recs)

    # Normalize distance column name
    if "distance_A" not in df.columns and "distance" in df.columns:
        df["distance_A"] = df["distance"]

    # Ensure orientation is stored as tuples for easy filtering
    if "orientation" in df.columns:
        df["orientation"] = df["orientation"].apply(lambda o: tuple(o))

    return df

# ---------------------------------------------------------------------
# CUTOFF SUGGESTION (FROM CATALOG ONLY)
# ---------------------------------------------------------------------

def suggest_distance_cutoff_from_catalog(
    df_cat: pd.DataFrame,
    *,
    orientation=None,
    target_fraction: float = 0.99,     # capture 99% of total κ
    marginal_kappa_min: float = 1e-4,  # stop when marginal κ is tiny
    Ak_min_kHz: float | None = None,   # optional filter on |A_par|
    Ak_max_kHz: float | None = None,
    distance_max: float | None = None,
) -> dict:
    """
    Given the full catalog DataFrame and one NV orientation, compute:

        - cumulative κ vs distance
        - cutoff distance based on:
            (1) reaching target_fraction of total κ
            (2) local average κ falling below marginal_kappa_min

    Returns a dict with the cutoff distances and the per-site table.
    """
    df = df_cat.copy()

    # 1) Filter by orientation if requested
    if orientation is not None:
        ori_t = tuple(int(x) for x in orientation)
        df = df[df["orientation"] == ori_t]

    # 2) Optional filter on hyperfine strength (A_par)
    if "A_par_Hz" in df.columns:
        df["A_par_kHz"] = df["A_par_Hz"] / 1e3
        if Ak_min_kHz is not None:
            df = df[df["A_par_kHz"].abs() >= float(Ak_min_kHz)]
        if Ak_max_kHz is not None:
            df = df[df["A_par_kHz"].abs() <= float(Ak_max_kHz)]

    # 3) Optional distance cutoff
    if distance_max is not None:
        df = df[df["distance_A"] <= float(distance_max)]

    if df.empty:
        raise ValueError("No sites left after filtering for this orientation.")

    # 4) Collapse to unique sites (orientation + site_index)
    group_cols = ["orientation", "site_index", "distance_A"]
    if "kappa" not in df.columns:
        raise KeyError("Catalog is missing 'kappa' column.")

    df_site = (
        df.groupby(group_cols, as_index=False)
          .agg(kappa_max=("kappa", "max"))
    )

    # 5) Sort by distance and build cumulative κ
    df_sorted = df_site.sort_values("distance_A").reset_index(drop=True)
    k_sorted = df_sorted["kappa_max"].to_numpy()

    cum = np.cumsum(k_sorted)
    total = float(cum[-1]) if cum.size else 0.0
    cum_frac = cum / total if total > 0 else np.zeros_like(cum)

    df_sorted["cum_kappa"] = cum
    df_sorted["cum_frac"] = cum_frac

    # ----- Rule 1: reach target_fraction of total κ -----
    tf = min(max(target_fraction, 0.0), 1.0)
    idx_frac = np.searchsorted(cum_frac, tf)
    idx_frac = min(idx_frac, len(df_sorted) - 1)
    cutoff_by_fraction = float(df_sorted.loc[idx_frac, "distance_A"])

    # ----- Rule 2: marginal κ falls below threshold -----
    window = min(10, len(k_sorted))  # small moving window
    if window > 1:
        marg = np.convolve(k_sorted, np.ones(window) / window, mode="same")
        below = np.where(marg < float(marginal_kappa_min))[0]
        if below.size:
            idx_marg = int(below[0])
            cutoff_by_marginal = float(df_sorted.loc[idx_marg, "distance_A"])
        else:
            cutoff_by_marginal = float(df_sorted["distance_A"].iloc[-1])
    else:
        cutoff_by_marginal = float(df_sorted["distance_A"].iloc[-1])

    # Final suggestion = more conservative (smaller) of the two
    cutoff = min(cutoff_by_fraction, cutoff_by_marginal)

    return {
        "cutoff_distance": cutoff,
        "cutoff_by_fraction": cutoff_by_fraction,
        "cutoff_by_marginal": cutoff_by_marginal,
        "total_kappa": total,
        "table": df_sorted,  # columns: distance_A, kappa_max, cum_kappa, cum_frac
    }

# ---------------------------------------------------------------------
# OVERLAY PLOT FOR ALL ORIENTATIONS
# ---------------------------------------------------------------------

def plot_cutoffs_for_all_orientations_from_catalog(
    catalog_path: str = CATALOG_JSON,
    *,
    orientations=((1, 1, 1), (-1, 1, 1), (1, -1, 1), (1, 1, -1)),
    target_fraction: float = 0.99,
    marginal_kappa_min: float = 1e-4,
    Ak_min_kHz: float | None = None,
    Ak_max_kHz: float | None = None,
    distance_max: float | None = None,
):
    """
    Load the catalog, compute suggested distance cutoffs for each NV orientation,
    and plot cumulative κ fraction vs distance with vertical cutoff lines.
    """
    df_cat = load_catalog_df(catalog_path)

    curves = []
    for ori in orientations:
        res = suggest_distance_cutoff_from_catalog(
            df_cat,
            orientation=ori,
            target_fraction=target_fraction,
            marginal_kappa_min=marginal_kappa_min,
            Ak_min_kHz=Ak_min_kHz,
            Ak_max_kHz=Ak_max_kHz,
            distance_max=distance_max,
        )
        dfc = res["table"]
        curves.append(
            dict(
                ori=tuple(int(x) for x in ori),
                distance=dfc["distance_A"].to_numpy(),
                cum_frac=dfc["cum_frac"].to_numpy(),
                cutoff=res["cutoff_distance"],
                cut_frac=res["cutoff_by_fraction"],
                cut_marg=res["cutoff_by_marginal"],
            )
        )

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for c in curves:
        label = f"[{c['ori'][0]}, {c['ori'][1]}, {c['ori'][2]}]"

        # plot cumulative curve and grab its color
        (line,) = ax.plot(c["distance"], c["cum_frac"], lw=1.5, label=label)
        color = line.get_color()

        # vertical line at suggested cutoff, same color
        ax.axvline(
            c["cutoff"],
            ls="-",
            lw=1.0,
            color=color,
        )

        # annotate cutoff distance next to the line
        y_here = np.interp(
            c["cutoff"],
            c["distance"],
            c["cum_frac"],
            left=0.0,
            right=c["cum_frac"][-1] if len(c["cum_frac"]) else 1.0,
        )
        # ax.text(
        #     c["cutoff"],
        #     min(1.02, y_here - 0.4),
        #     f"{c['cutoff']:.2f} Å",
        #     rotation=90,
        #     va="bottom",
        #     ha="right",
        #     fontsize=11,
        # )

    ax.set_xlabel("Distance NV–13C (Å)", fontsize=13)
    ax.set_ylabel("Cumulative κ fraction", fontsize=15)
    ax.set_title(f"Hyperfine distance cutoff{target_fraction}",fontsize=15)
    ax.set_title(
    f"Hyperfine distance cutoffs (≥ {target_fraction*100:.0f}% κ)", fontsize=15
)
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3)
    # ax.legend(title="NV orientation", frameon=False, fontsize=13)
    ax.legend(
    title="NV orientation",
    frameon=False,
    fontsize=13,       # label font size
    title_fontsize=13,  # smaller title
    )

    plt.tight_layout()
    plt.show()

# ---------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------

if __name__ == "__main__":
    kpl.init_kplotlib()
    plot_cutoffs_for_all_orientations_from_catalog(
        catalog_path=CATALOG_JSON,
        orientations=((1, 1, 1), (-1, 1, 1), (1, -1, 1), (1, 1, -1)),
        target_fraction=0.93,
        marginal_kappa_min=1e-4,
        Ak_min_kHz=0.0,
        Ak_max_kHz=20000.0,
        distance_max=None,  # or set a hard cap like 20.0
    )
    kpl.show(block=True)