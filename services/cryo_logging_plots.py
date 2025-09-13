# services/cryo_plots.py
import os
import re
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker  # add this import
import numpy as np
import pandas as pd
from matplotlib.dates import AutoDateLocator, DateFormatter

from utils import kplotlib as kpl

FILE_PATH = r"G:\nv_cryo_logging\2025_08_16_cryo_logging.txt"

# Choose x-axis: "timestamp" or "time_s"
PLOT_X_AXIS = "timestamp"

# Optional manual override. Example: "2025-08-16 20:48:23"
# If set to a non-empty string, this will be used instead of parsing the file.
START_TIME_OVERRIDE = ""
kpl.init_kplotlib()


def find_header_line(path: str) -> int:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            s = line.strip().lower()
            if "time" in s and (("\t" in line) or len(re.split(r"\s+", s)) >= 5):
                return i
    return 0


def parse_start_time(path: str) -> datetime | None:
    """
    Parse a start timestamp from the first few non-empty lines.
    Handles odd formats like '8/16/2025 _:48:23 P' (interprets as 08:48:23 PM).
    Returns naive datetime or None.
    """
    # Manual override wins
    if START_TIME_OVERRIDE.strip():
        try:
            return datetime.fromisoformat(START_TIME_OVERRIDE.strip())
        except Exception:
            pass  # fall through to auto-parse

    lines = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        # Look at the first ~5 non-empty lines max
        for _ in range(10):
            line = f.readline()
            if not line:
                break
            if line.strip():
                lines.append(line.strip())

    for raw in lines:
        s = raw

        # Normalize weird hour pattern, e.g. ' _:48:23' -> ' 0:48:23'
        s = re.sub(r"\b_:", "0:", s)

        # Normalize ' A'/' P' -> ' AM'/' PM'
        s = re.sub(r"\b([AaPp])\b", r"\1M", s)

        # Extract date (mm/dd/yyyy)
        m_date = re.search(r"(\d{1,2})/(\d{1,2})/(\d{4})", s)
        if not m_date:
            continue
        mm, dd, yyyy = map(int, m_date.groups())

        # Extract time H:MM[:SS] with optional AM/PM
        m_time = re.search(r"(\d{1,2}):(\d{2})(?::(\d{2}))?\s*([AaPp][Mm])?", s)
        if not m_time:
            continue
        hh = int(m_time.group(1))
        mn = int(m_time.group(2))
        ss = int(m_time.group(3)) if m_time.group(3) else 0
        ampm = m_time.group(4)

        try:
            if ampm:
                ampm = ampm.upper()
                if ampm == "PM" and hh != 12:
                    hh += 1
                if ampm == "AM" and hh == 12:
                    hh = 0
            return datetime(yyyy, mm, dd, hh, mn, ss)
        except Exception:
            continue

    return None


def clean_columns(cols):
    out = []
    for c in cols:
        c2 = c.strip().replace("\ufeff", "")
        c2 = c2.replace("%", "percent")
        c2 = re.sub(r"[()\/]", " ", c2)  # remove () and /
        c2 = re.sub(r"\s+", "_", c2).lower()  # spaces -> _
        c2 = re.sub(r"[^a-z0-9_]", "", c2)  # keep a-z0-9_
        c2 = re.sub(r"_+", "_", c2)  # collapse multiple _
        c2 = c2.strip("_")  # trim leading/trailing _
        out.append(c2)
    return out


def _read_with_sep(path: str, header_idx: int, sep, engine="python"):
    return pd.read_csv(path, sep=sep, header=0, skiprows=header_idx, engine=engine)


def load_cryo_log(path: str) -> pd.DataFrame:
    header_idx = find_header_line(path)

    # Try tab first; fallback to whitespace if it looks wrong
    try:
        df = _read_with_sep(path, header_idx, sep="\t", engine="python")
        first_cols = df.columns.tolist()
    except Exception:
        df = None
        first_cols = []

    bad_tokens = {"_s_", "_hz_", "_w_", "_k_", "_t_", "_mbar_"}
    if (
        df is None
        or len(first_cols) > 25
        or any(str(col).strip().lower() in bad_tokens for col in first_cols)
    ):
        df = _read_with_sep(path, header_idx, sep=r"\s+", engine="python")

    df.columns = clean_columns(df.columns.tolist())

    # Drop empty unnamed columns if present
    for col in list(df.columns):
        if col.startswith("unnamed") and df[col].isna().all():
            df.drop(columns=[col], inplace=True)

    # Coerce numeric where possible
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = pd.to_numeric(df[c], errors="ignore")

    return df


def add_timestamp(df: pd.DataFrame, start_dt: datetime) -> pd.DataFrame:
    # Candidate time columns
    candidates = [c for c in df.columns if c.startswith("time")]
    if not candidates:
        return df
    # Prefer names like 'time_s' if present
    tcol = None
    for name in ("time_s", "time", "time_seconds"):
        if name in df.columns:
            tcol = name
            break
    if tcol is None:
        tcol = candidates[0]

    t_sec = pd.to_numeric(df[tcol], errors="coerce")
    if start_dt is None or not np.isfinite(t_sec).any():
        return df
    df["timestamp"] = pd.to_datetime(start_dt) + pd.to_timedelta(t_sec, unit="s")
    return df


def plot_dashboard(df: pd.DataFrame, save_dir: str = None):
    # Decide X-axis
    if PLOT_X_AXIS == "timestamp":
        if "timestamp" not in df.columns:
            raise RuntimeError(
                "You requested PLOT_X_AXIS='timestamp' but no 'timestamp' column exists.\n"
                "Either set START_TIME_OVERRIDE to the correct start time, or switch PLOT_X_AXIS='time_s'."
            )
        x = df["timestamp"]
        x_label = "Time"
    else:
        time_candidates = [c for c in df.columns if c.startswith("time")]
        if not time_candidates:
            raise KeyError(f"No time column detected. Columns: {df.columns.tolist()}")
        # prefer 'time_s'
        tcol = "time_s" if "time_s" in df.columns else time_candidates[0]
        t = pd.to_numeric(df[tcol], errors="coerce")
        if np.isfinite(t).any():
            t0 = np.nanmin(t)
            if np.isfinite(t0):
                t = t - t0
        x = t
        x_label = "Time (s)"

    panels = []

    temp_cols = [
        ("sample_temperature_k", "Sample (K)"),
        # ("magnet_temperature_k", "Magnet (K)"),
        # ("user_temperature_k", "User (K)"),
    ]
    if any(c in df.columns for c, _ in temp_cols):
        panels.append(("Temperatures", "K", temp_cols))

    # df["cryo_in_pressure_mbar"] = df["cryo_in_pressure_mbar"].round(8)
    print(df["cryo_in_pressure_mbar"])
    press_cols = [("cryo_in_pressure_mbar", "Cryo In (mbar)")]
    if any(c in df.columns for c, _ in press_cols):
        panels.append(("Pressure", "mbar", press_cols))

    # heater_cols = [
    #     ("sample_heater_power_w", "Sample Heater (W)"),
    #     ("exchange_heater_power_w", "Exchange Heater (W)"),
    # ]
    # if any(c in df.columns for c, _ in heater_cols):
    #     panels.append(("Heaters", "W", heater_cols))

    # field_cols = [
    #     ("magnetic_field_t", "Magnetic Field (T)"),
    #     ("user_magnetic_field_t", "User Magnetic Field (T)"),
    # ]
    # if any(c in df.columns for c, _ in field_cols):
    #     panels.append(("Magnetic Field", "T", field_cols))

    # pump_cols = [("turbo_pump_frequency_hz", "Turbo Pump (Hz)")]
    # if any(c in df.columns for c, _ in pump_cols):
    #     panels.append(("Turbo Pump", "Hz", pump_cols))

    if not panels:
        raise RuntimeError(
            f"No known data channels found to plot. Detected columns: {df.columns.tolist()}"
        )

    n = len(panels)
    fig, axes = plt.subplots(n, 1, figsize=(12, 4.4 * n), sharex=True)
    if n == 1:
        axes = [axes]

    # Plot
    for ax, (title, ylabel, series_list) in zip(axes, panels):
        plotted = False
        for col_key, label in series_list:
            if col_key in df.columns:
                y = pd.to_numeric(df[col_key], errors="coerce")
                ax.plot(x, y, label=label)
                plotted = True
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        # If this panel is the Pressure plot, show 8 decimals on Y ticks
        # if title == "Pressure":
        #     ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.8f"))
        #     # Optional: avoid scientific notation entirely
        #     # ax.ticklabel_format(axis="y", style="plain", useOffset=False)

        ax.grid(True, alpha=0.3)
        if plotted and sum(col_key in df.columns for col_key, _ in series_list) > 1:
            ax.legend(loc="best")

    # X label and time formatter
    axes[-1].set_xlabel(x_label)
    if PLOT_X_AXIS == "timestamp":
        locator = AutoDateLocator()
        formatter = DateFormatter("%Y-%m-%d %H:%M:%S")
        for ax in axes:
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(formatter)
        fig.autofmt_xdate()

    # plt.tight_layout()
    # if save_dir is None:
    #     save_dir = os.path.dirname(FILE_PATH) or os.getcwd()
    # png_path = os.path.join(save_dir, "cryo_dashboard.png")
    # pdf_path = os.path.join(save_dir, "cryo_dashboard.pdf")
    # plt.savefig(png_path, dpi=150, bbox_inches="tight")
    # plt.savefig(pdf_path, bbox_inches="tight")
    # print("Saved:", png_path)
    # print("Saved:", pdf_path)


def main():
    start_dt = parse_start_time(FILE_PATH)
    print("Parsed start time:", start_dt if start_dt else "None (will use seconds)")

    df = load_cryo_log(FILE_PATH)
    df = add_timestamp(df, start_dt)
    # print("Detected columns (cleaned):", df.columns.tolist())
    print("Using X axis:", PLOT_X_AXIS)
    plot_dashboard(df)


if __name__ == "__main__":
    main()
    kpl.show(block=True)
