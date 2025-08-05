# -*- coding: utf-8 -*-
"""
Created on June 16th, 2023
@author: Saroj B Chand
"""

import datetime
import os

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from scipy import stats  # for linear regression drift estimate

from utils import kplotlib as kplt

kplt.init_kplotlib()
# ----------------------------
# User-configurable parameters
# ----------------------------
base_folder = "G:\\NV_Widefield_RT_Setup_Enclosure_Temp_Logs"
hours = 2  # window to analyze & plot
temp_low, temp_high = 15, 25  # sanity filter limits
PLOT_ADEV = True  # set False if you don't want the Allan plot refreshing

# Define channels and corresponding filenames
channels = {
    "4A": "temp_4A.csv",
    "4B": "temp_4B.csv",
    "4C": "temp_4C.csv",
    "4D": "temp_4D.csv",
    "temp_stick": "temp_stick.csv",
}


# ----------------------------
# Dynamic Allan taus
# ----------------------------
def build_allan_taus(hours: float) -> list:
    # From 60 s (1 min) to the full window, at least 8 points
    n_pts = max(8, int(hours * 2))  # ~2 points/hour
    return list(np.linspace(60, hours * 3600, n_pts))


ALLAN_TAUS = build_allan_taus(hours)

# ----------------------------
# Folder bookkeeping
# ----------------------------
now = datetime.datetime.now()
folder_current = now.strftime("%m%Y")
folder_previous = (now - relativedelta(months=1)).strftime("%m%Y")

data_folders = [
    os.path.join(base_folder, folder_previous),
    os.path.join(base_folder, folder_current),
]

# ----------------------------
# Plot setup
# ----------------------------
plt.ion()
fig, ax = plt.subplots(figsize=(10, 5))
fig_adev, ax_adev = plt.subplots(figsize=(7, 5)) if PLOT_ADEV else (None, None)

timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print(f"Current Time Stamp: {timestamp}")


# ----------------------------
# Helpers
# ----------------------------
def load_channel_df(filename: str) -> pd.DataFrame:
    """Load and concatenate current+previous month CSVs for a channel."""
    dfs = []
    for folder in data_folders:
        file_path = os.path.join(folder, filename)
        if not os.path.exists(file_path):
            continue
        try:
            df = pd.read_csv(
                file_path,
                names=["Timestamp", "Temperature"],
                parse_dates=["Timestamp"],
                dtype={"Temperature": float},
            )
            dfs.append(df)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    if not dfs:
        return pd.DataFrame(columns=["Timestamp", "Temperature"])

    return pd.concat(dfs, ignore_index=True)


def window_and_filter(
    df: pd.DataFrame, hours: float, t_low: float, t_high: float
) -> pd.DataFrame:
    """Apply time window and temp sanity filter."""
    if df.empty:
        return df
    now = datetime.datetime.now()
    df = df[
        (df["Timestamp"] > (now - datetime.timedelta(hours=hours)))
        & (df["Temperature"] > t_low)
        & (df["Temperature"] < t_high)
    ].copy()
    return df


def compute_drift_deg_per_hour(df: pd.DataFrame) -> float:
    """Linear fit (Temperature vs time) slope converted to °C/hour."""
    if len(df) < 3:
        return np.nan
    t = (df["Timestamp"] - df["Timestamp"].iloc[0]).dt.total_seconds().values
    y = df["Temperature"].values
    slope, _, _, _, _ = stats.linregress(t, y)
    return slope * 3600.0  # °C/sec -> °C/hour


def simple_allan_deviation(
    series: pd.Series, dt_seconds: float, taus_seconds: list[int]
) -> dict:
    """
    Simple (non-overlapping) Allan deviation.
    Assumes uniformly-sampled data at dt_seconds.
    """
    y = series.values
    N = len(y)
    adev = {}
    for tau in taus_seconds:
        m = int(round(tau / dt_seconds))  # samples per tau
        if m < 1 or m > 2 * N:
            adev[tau] = np.nan
            continue
        nblocks = N // m
        y_trim = y[: nblocks * m]
        block_means = y_trim.reshape(nblocks, m).mean(axis=1)
        diff = np.diff(block_means)
        avar = 0.5 * np.mean(diff**2)
        adev[tau] = np.sqrt(avar)
    return adev


def compute_metrics(df: pd.DataFrame, allan_taus=ALLAN_TAUS) -> dict:
    """Compute std, sem, peak-to-peak, drift, Allan dev."""
    out = {
        "N": len(df),
        "mean": np.nan,
        "std": np.nan,
        "sem": np.nan,
        "min": np.nan,
        "max": np.nan,
        "p2p": np.nan,
        "drift_deg_per_hr": np.nan,
        "allan_dev": {},
        "med_dt": np.nan,
    }
    if df.empty:
        return out

    temps = df["Temperature"].values
    out["mean"] = float(np.mean(temps))
    out["std"] = float(np.std(temps, ddof=1) if len(temps) > 1 else 0.0)
    out["sem"] = float(out["std"] / np.sqrt(out["N"])) if out["N"] > 0 else np.nan
    out["min"] = float(np.min(temps))
    out["max"] = float(np.max(temps))
    out["p2p"] = float(out["max"] - out["min"])
    out["drift_deg_per_hr"] = compute_drift_deg_per_hour(df)

    # Allan deviation: resample to a uniform cadence first
    try:
        df_u = df.set_index("Timestamp").sort_index()
        med_dt = df_u.index.to_series().diff().median().total_seconds()
        out["med_dt"] = med_dt
        if pd.isna(med_dt) or med_dt <= 0:
            out["allan_dev"] = {tau: np.nan for tau in allan_taus}
        else:
            rule = f"{int(round(med_dt))}S"
            df_res = df_u["Temperature"].resample(rule).mean().interpolate("time")
            out["allan_dev"] = simple_allan_deviation(df_res, med_dt, allan_taus)
    except Exception as e:
        print("Allan deviation computation failed:", e)
        out["allan_dev"] = {tau: np.nan for tau in allan_taus}

    return out


def pretty_print_metrics(label: str, m: dict):
    print(f"\n=== Channel {label} ===")
    print(f"N = {m['N']}")
    print(f"mean = {m['mean']:.6f} °C")
    print(f"std  = {m['std']:.6f} °C")
    print(f"sem  = {m['sem']:.6f} °C")
    print(f"min/max = {m['min']:.6f}/{m['max']:.6f} °C")
    print(f"peak-to-peak = {m['p2p']:.6f} °C")
    print(f"drift = {m['drift_deg_per_hr']:.6e} °C/hour")
    # if m["allan_dev"]:
    #     print("Allan deviation (°C):")
    #     for tau, val in m["allan_dev"].items():
    #         if np.isnan(val):
    #             print(f"  tau = {tau:>6.0f} s :  NaN")
    #         else:
    #             print(f"  tau = {tau:>6.0f} s :  {val:.6e}")


def update_plot():
    ax.clear()
    if PLOT_ADEV:
        ax_adev.clear()

    for label, filename in channels.items():
        df_all = load_channel_df(filename)
        if df_all.empty:
            print(f"No data found for Channel {label}")
            continue

        df_all = window_and_filter(df_all, hours, temp_low, temp_high)
        if df_all.empty:
            print(f"No recent/valid data for Channel {label}")
            continue

        # Compute & print metrics
        metrics = compute_metrics(df_all, ALLAN_TAUS)
        pretty_print_metrics(label, metrics)

        # Plot temperature
        ax.plot(df_all["Timestamp"], df_all["Temperature"], label=f"{label}")

        # Plot Allan deviation
        if PLOT_ADEV and metrics["allan_dev"]:
            taus = np.array(
                [
                    t
                    for t in metrics["allan_dev"].keys()
                    if not np.isnan(metrics["allan_dev"][t])
                ]
            )
            vals = np.array([metrics["allan_dev"][t] for t in taus])
            if taus.size > 0:
                ax_adev.loglog(taus, vals, marker="o", label=label)

    # Decorate time plot
    ax.set_title(f"Temperature Plot (Last {hours}h)", fontsize=13)
    ax.set_xlabel("Time", fontsize=13)
    ax.set_ylabel("Temperature [°C]", fontsize=13)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
    ax.tick_params(axis="both", labelsize=11)
    ax.legend(fontsize=11)
    fig.autofmt_xdate()

    fig.text(
        0.40,
        0.36,
        "4A --> near sample (feedback channel)\n"
        "4B --> air inside duct of experiment enclosure\n"
        "4C --> air inside duct of laser enclosure\n"
        "4D --> air inside laser enclosure\n"
        "temp_stick --> outside monitor",
        ha="left",
        va="bottom",
        fontsize=11,
    )

    if PLOT_ADEV:
        ax_adev.set_title(f"Allan Deviation (Last {hours}h)", fontsize=13)
        ax_adev.set_xlabel("Averaging Time τ [s]", fontsize=13)
        ax_adev.set_ylabel("Allan Deviation [°C]", fontsize=13)
        ax_adev.grid(True, which="both", ls="--", alpha=0.4)
        ax_adev.legend(fontsize=11)

    plt.pause(0.1)


def main():
    print(f"Live plotting from: {data_folders}")
    try:
        while True:
            update_plot()
            if (
                input("Press Enter to refresh or type 'q' to quit: ").strip().lower()
                == "q"
            ):
                break
    finally:
        print("Exiting and closing plots.")
        plt.close("all")


if __name__ == "__main__":
    main()
