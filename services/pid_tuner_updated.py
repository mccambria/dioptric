# -*- coding: utf-8 -*-
"""
Created on July 3rd, 2025

@author: Saroj B Chand

PID sweep: read ALL channels each second, analyze each channel independently,
and save both raw traces and per-channel metrics.
"""

import csv
import datetime
import itertools
import os
import time

import numpy as np
from scipy import stats  # for drift slope

from utils import common

# ----------- USER SETTINGS -----------
TEMP_CHANNELS = {
    "4A": b"4A?\n",
    "4B": b"4B?\n",
    "4C": b"4C?\n",
    "4D": b"4D?\n",
}
OUTPUTCHANNEL = b"OUT1"

DURATION = 360  # seconds to record per PID setting
SAMPLE_PERIOD = 1.0  # seconds between samples
SLEEP_BETWEEN = 2  # seconds wait after PID change before sampling starts
SAVE_DIR = "G:/NV_Widefield_RT_Setup_Enclosure_Temp_Logs/pid_tuning"

# Ranges to sweep
P_vals = [50, 75, 100, 125, 150]
I_vals = [1, 1.5, 2, 2.5, 3]
D_vals = [40, 50, 60, 60, 80]

# Settling-band (choose one of the two modes below)
SETTLING_ABS_BAND = 0.02  # °C band around the final value
USE_REL_BAND = False  # if True, use fraction of final value instead
SETTLING_REL_FRACTION = 0.001  # i.e., 0.1% of final value
# -------------------------------------


def calc_settling_time(
    data, sampling_period, abs_band=0.02, use_rel=False, rel_fraction=0.001
):
    """
    Estimate settling time: the time after which the signal stays within
    ±abs_band (°C) or ±rel_fraction of the final value until the end.

    Returns: settling time (seconds). 0 if already within.
    """
    if len(data) == 0:
        return np.nan

    final_value = data[-1]
    if use_rel:
        band = abs(final_value) * rel_fraction
    else:
        band = abs_band

    lower = final_value - band
    upper = final_value + band

    # Scan back from the end to find the last index outside the band
    last_outside = -1
    for i in range(len(data) - 1, -1, -1):
        if not (lower <= data[i] <= upper):
            last_outside = i
            break

    if last_outside == -1:
        # Always in band
        return 0.0
    else:
        # time from (last_outside+1) to the end
        n_inside_points = len(data) - (last_outside + 1)
        return n_inside_points * sampling_period


def compute_drift_deg_per_hour(timestamps, temps):
    """
    Linear regression slope in °C/hour.
    timestamps: list of datetime objects
    temps: numpy array of temperature values
    """
    if len(temps) < 3:
        return np.nan
    t0 = timestamps[0]
    t_sec = np.array([(ts - t0).total_seconds() for ts in timestamps])
    slope, _, _, _, _ = stats.linregress(t_sec, temps)
    return slope * 3600.0  # °C/sec -> °C/hour


def compute_metrics(timestamps, temps):
    """
    Return dict of metrics for a single channel trace.
    """
    temps = np.asarray(temps)
    N = len(temps)
    if N == 0:
        return {
            "N": 0,
            "mean": np.nan,
            "std": np.nan,
            "sem": np.nan,
            "min": np.nan,
            "max": np.nan,
            "p2p": np.nan,
            "drift_deg_per_hour": np.nan,
            "max_overshoot": np.nan,
            "settling_time_s": np.nan,
        }

    std = float(np.std(temps, ddof=1)) if N > 1 else 0.0
    sem = float(std / np.sqrt(N)) if N > 0 else np.nan
    p2p = float(np.max(temps) - np.min(temps))
    drift = float(compute_drift_deg_per_hour(timestamps, temps))

    # overshoot relative to the starting value
    max_overshoot = float(np.max(temps) - temps[0])

    # settling time (using configuration)
    settling_time = calc_settling_time(
        temps,
        sampling_period=SAMPLE_PERIOD,
        abs_band=SETTLING_ABS_BAND,
        use_rel=USE_REL_BAND,
        rel_fraction=SETTLING_REL_FRACTION,
    )

    return {
        "N": N,
        "mean": float(np.mean(temps)),
        "std": std,
        "sem": sem,
        "min": float(np.min(temps)),
        "max": float(np.max(temps)),
        "p2p": p2p,
        "drift_deg_per_hour": drift,
        "max_overshoot": max_overshoot,
        "settling_time_s": float(settling_time),
    }


def read_all_channels(server):
    """
    Returns (timestamp, temps_dict) where temps_dict[channel] = float temp
    """
    ts = datetime.datetime.now()
    out = {}
    for ch_name, cmd in TEMP_CHANNELS.items():
        try:
            temp = server.get_temp(cmd)  # keeping your original API style
            out[ch_name] = float(temp)
        except Exception as e:
            print(f"Error reading {ch_name}: {e}")
            out[ch_name] = np.nan
    return ts, out


def save_trace_per_channel(save_dir, P, I, D, channel, timestamps, temps):
    """
    Writes one CSV per channel per PID setting.
    """
    file_suffix = f"P{P}_I{I}_D{D}_{channel}.csv"
    path = os.path.join(save_dir, file_suffix)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Temperature"])
        for ts, t in zip(timestamps, temps):
            writer.writerow([ts.isoformat(), f"{t:.6f}"])
    return path


def tune_pid():
    cxn = common.labrad_connect()
    server = cxn.temp_monitor_SRS_ptc10

    os.makedirs(SAVE_DIR, exist_ok=True)
    summary_path = os.path.join(SAVE_DIR, "summary.csv")

    # Prepare the summary file (long format: one row per (P,I,D,channel))
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "P",
                "I",
                "D",
                "Channel",
                "N",
                "Mean",
                "Std",
                "SEM",
                "Min",
                "Max",
                "PeakToPeak",
                "Drift_deg_per_hour",
                "MaxOvershoot",
                "SettlingTime_s",
                "TraceCSV",
            ]
        )

    for P, I, D in itertools.product(P_vals, I_vals, D_vals):
        # 1) Set PID
        print(f"\n>>> Tuning PID: P={P}, I={I}, D={D}")
        try:
            server.set_param(OUTPUTCHANNEL + b".PID.P", P)
            server.set_param(OUTPUTCHANNEL + b".PID.I", I)
            server.set_param(OUTPUTCHANNEL + b".PID.D", D)
        except Exception as e:
            print(f"Error setting PID: {e}")
            continue

        time.sleep(SLEEP_BETWEEN)

        # 2) Collect data for all channels
        timestamps = []
        temp_traces = {ch: [] for ch in TEMP_CHANNELS.keys()}
        start_time = time.time()

        while (time.time() - start_time) < DURATION:
            ts, reading = read_all_channels(server)
            timestamps.append(ts)
            for ch, val in reading.items():
                temp_traces[ch].append(val)

            # print quick progress line
            if len(timestamps) % 10 == 0:
                line = ", ".join(
                    f"{ch}:{temp_traces[ch][-1]:.3f}" for ch in TEMP_CHANNELS.keys()
                )
                print(f"[{ts}] {line}")
            time.sleep(SAMPLE_PERIOD)

        # 3) Analyze & save per channel
        with open(summary_path, "a", newline="") as fsum:
            writer = csv.writer(fsum)
            for ch_name, temps in temp_traces.items():
                metrics = compute_metrics(timestamps, temps)
                trace_path = save_trace_per_channel(
                    SAVE_DIR, P, I, D, ch_name, timestamps, temps
                )

                writer.writerow(
                    [
                        P,
                        I,
                        D,
                        ch_name,
                        metrics["N"],
                        metrics["mean"],
                        metrics["std"],
                        metrics["sem"],
                        metrics["min"],
                        metrics["max"],
                        metrics["p2p"],
                        metrics["drift_deg_per_hour"],
                        metrics["max_overshoot"],
                        metrics["settling_time_s"],
                        trace_path,
                    ]
                )

        # 4) Let system rest a bit before next setting
        time.sleep(20)

    print("\nPID sweep complete. Summary saved to:", summary_path)


if __name__ == "__main__":
    tune_pid()
