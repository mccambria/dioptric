import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import kplotlib as kplt

# -------- USER CONFIG --------
kplt.init_kplotlib()
SUMMARY_CSV = (
    r"G:\NV_Widefield_RT_Setup_Enclosure_Temp_Logs\pid_tuning_seq\summary_seq.csv"
)
COST_WEIGHTS = dict(std=1.0, settling=0.01, drift=10.0)  # tweak to taste
SAVE_FIGS = False
OUTDIR = os.path.join(os.path.dirname(SUMMARY_CSV), "per_channel_plots")
# -----------------------------

os.makedirs(OUTDIR, exist_ok=True)


def cost_from_row(r, w):
    std = r["Std"]
    settle = r["SettlingTime_s"]
    drift = abs(r["Drift_deg_per_hour"])
    std = std if np.isfinite(std) else 1e9
    settle = settle if np.isfinite(settle) else 1e9
    drift = drift if np.isfinite(drift) else 1e9
    return w["std"] * std + w["settling"] * settle + w["drift"] * drift


def most_common_pair(df, cols):
    """Return the most common tuple of values for given cols."""
    return df.groupby(cols).size().sort_values(ascending=False).index[0]


# def plot_cost_vs(df_ch, var, hold_cols, title, outpath=None):
#     # choose the most common (fixed) values of hold_cols
#     if hold_cols:
#         fixed_vals = most_common_pair(df_ch, hold_cols)
#         query = " & ".join([f"{c} == {val}" for c, val in zip(hold_cols, fixed_vals)])
#         sub = df_ch.query(query).copy()
#         fixed_str = ", ".join([f"{c}={val}" for c, val in zip(hold_cols, fixed_vals)])
#     else:
#         sub = df_ch.copy()
#         fixed_str = ""

#     if sub.empty:
#         print(f"[WARN] No rows for {title} ({fixed_str}). Skipping.")
#         return

#     sub = sub.sort_values(var)
#     plt.figure(figsize=(6, 4))
#     plt.plot(sub[var], sub["cost"], marker="o")
#     plt.xlabel(var)
#     plt.ylabel("Cost")
#     plt.title(title + (f"\n({fixed_str})" if fixed_str else ""))
#     plt.grid(True, alpha=0.3)
#     plt.tight_layout()
#     if outpath:
#         plt.savefig(outpath, dpi=200)
#     plt.show()


def plot_cost_vs(df_ch, var, hold_cols, title, outpath=None):
    # Filter to the most common pair for hold_cols
    if hold_cols:
        fixed_vals = most_common_pair(df_ch, hold_cols)
        query = " & ".join([f"{c} == {val}" for c, val in zip(hold_cols, fixed_vals)])
        df_sub = df_ch.query(query).copy()
    else:
        df_sub = df_ch.copy()

    # Group by var (in case of duplicates) and take min cost
    df_agg = df_sub.groupby(var, as_index=False)["cost"].min()

    df_agg = df_agg.sort_values(var)
    plt.figure()
    plt.plot(df_agg[var], df_agg["cost"], marker="o")
    plt.xlabel(var)
    plt.ylabel("Cost")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    if outpath:
        plt.savefig(outpath, dpi=200)
    plt.show()


def main():
    df = pd.read_csv(SUMMARY_CSV)

    # Compute cost
    df["cost"] = df.apply(lambda r: cost_from_row(r, COST_WEIGHTS), axis=1)

    # Quick sanity: required columns
    needed = [
        "Channel",
        "P",
        "I",
        "D",
        "Std",
        "SettlingTime_s",
        "Drift_deg_per_hour",
        "cost",
    ]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    for ch, df_ch in df.groupby("Channel"):
        print(f"\n=== Channel {ch} ===")
        # Print relevant parameters sorted by cost
        # Sort by cost
        df_sorted = df_ch[
            ["P", "I", "D", "Std", "SettlingTime_s", "Drift_deg_per_hour", "cost"]
        ].sort_values(by="cost")
        # Print all sorted rows (optional)
        # print("All candidates (sorted by cost):")
        # print(df_sorted.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

        # Get the best PID (lowest cost)
        best_row = df_sorted.iloc[0]
        print(f"\nOptimal PID for Channel {ch}:")
        print(f"  P = {best_row['P']:.3f}")
        print(f"  I = {best_row['I']:.3f}")
        print(f"  D = {best_row['D']:.3f}")
        print(f"  Cost = {best_row['cost']:.4f}")
        print(
            f"  Std = {best_row['Std']:.4f}, Settling = {best_row['SettlingTime_s']:.2f}s, Drift = {best_row['Drift_deg_per_hour']:.4e}"
        )

        # Cost vs P (hold I, D)
        plot_cost_vs(
            df_ch,
            var="P",
            hold_cols=["I", "D"],
            title=f"Channel {ch}: Cost vs P",
            outpath=os.path.join(OUTDIR, f"{ch}_cost_vs_P.png") if SAVE_FIGS else None,
        )

        # Cost vs I (hold P, D)
        plot_cost_vs(
            df_ch,
            var="I",
            hold_cols=["P", "D"],
            title=f"Channel {ch}: Cost vs I",
            outpath=os.path.join(OUTDIR, f"{ch}_cost_vs_I.png") if SAVE_FIGS else None,
        )

        # Cost vs D (hold P, I)
        plot_cost_vs(
            df_ch,
            var="D",
            hold_cols=["P", "I"],
            title=f"Channel {ch}: Cost vs D",
            outpath=os.path.join(OUTDIR, f"{ch}_cost_vs_D.png") if SAVE_FIGS else None,
        )


if __name__ == "__main__":
    main()
    plt.show(block=True)
