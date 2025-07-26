# -*- coding: utf-8 -*-
"""
Sequential PID sweep: tune P first (I,D fixed), then I (P*,D fixed), then D (P*,I* fixed).
Reads ALL channels, optimizes on a chosen channel (e.g., 4A), and logs everything.

@author:
  Saroj B Chand (adapted)
"""

import csv
import datetime
import os
import time

import numpy as np
from scipy import stats

from utils import common

# -------------------- USER SETTINGS --------------------
TEMP_CHANNELS = {
    "4A": b"4A?\n",
    "4B": b"4B?\n",
    "4C": b"4C?\n",
    "4D": b"4D?\n",
}
OUTPUTCHANNEL = b"OUT1"

SAVE_DIR = "G:/NV_Widefield_RT_Setup_Enclosure_Temp_Logs/pid_tuning_seq"

DURATION = 600  # seconds to record per candidate
SAMPLE_PERIOD = 1.0  # seconds between samples
SLEEP_BETWEEN = 60  # seconds after setting PID before sampling
REST_AFTER = 6  # seconds rest after each candidate


# Sequential sweeps
P_SWEEP = [115, 120, 125, 135, 140, 145, 150]
I_SWEEP = [2, 2.5, 3, 3.5, 4.0, 4.5, 5]
D_SWEEP = [100, 105, 110, 115, 120, 125, 130, 135, 140]

# Starting (fixed) values while sweeping others
# P_INIT = 125
I_INIT = 3.5
D_INIT = 120
# Optimize using metrics from this channel (still record all)
OPTIMIZE_ON_CHANNEL = "4A"

# Cost weights for picking best setting
# Cost = w_std * std + w_settle * settling_time_s + w_drift * |drift_deg_per_hour|
COST_WEIGHTS = dict(std=1.0, settling=0.01, drift=10.0)

# Settling band configuration
SETTLING_ABS_BAND = 0.02  # °C
USE_REL_BAND = False
SETTLING_REL_FRACTION = 0.001
# ------------------------------------------------------


def set_pid():
    cxn = common.labrad_connect()
    server = cxn.temp_monitor_SRS_ptc10
    P = 135
    I = 3.5
    D = 120
    server.set_param(OUTPUTCHANNEL + b".PID.P", P)
    server.set_param(OUTPUTCHANNEL + b".PID.I", I)
    server.set_param(OUTPUTCHANNEL + b".PID.D", D)


def calc_settling_time(
    data, sampling_period, abs_band=0.02, use_rel=False, rel_fraction=0.001
):
    if len(data) == 0:
        return np.nan
    final_value = data[-1]
    if use_rel:
        band = abs(final_value) * rel_fraction
    else:
        band = abs_band
    lower, upper = final_value - band, final_value + band

    last_outside = -1
    for i in range(len(data) - 1, -1, -1):
        if not (lower <= data[i] <= upper):
            last_outside = i
            break
    if last_outside == -1:
        return 0.0
    return (len(data) - (last_outside + 1)) * sampling_period


def compute_drift_deg_per_hour(timestamps, temps):
    if len(temps) < 3:
        return np.nan
    t0 = timestamps[0]
    t_sec = np.array([(ts - t0).total_seconds() for ts in timestamps])
    slope, _, _, _, _ = stats.linregress(t_sec, temps)
    return slope * 3600.0


def compute_metrics(timestamps, temps):
    temps = np.asarray(temps)
    N = len(temps)
    if N == 0:
        return dict(
            N=0,
            mean=np.nan,
            std=np.nan,
            sem=np.nan,
            min=np.nan,
            max=np.nan,
            p2p=np.nan,
            drift_deg_per_hour=np.nan,
            max_overshoot=np.nan,
            settling_time_s=np.nan,
        )

    std = float(np.std(temps, ddof=1)) if N > 1 else 0.0
    sem = float(std / np.sqrt(N))
    p2p = float(np.max(temps) - np.min(temps))
    drift = float(compute_drift_deg_per_hour(timestamps, temps))
    max_overshoot = float(np.max(temps) - temps[0])
    settling_time = calc_settling_time(
        temps,
        SAMPLE_PERIOD,
        abs_band=SETTLING_ABS_BAND,
        use_rel=USE_REL_BAND,
        rel_fraction=SETTLING_REL_FRACTION,
    )
    return dict(
        N=N,
        mean=float(np.mean(temps)),
        std=std,
        sem=sem,
        min=float(np.min(temps)),
        max=float(np.max(temps)),
        p2p=p2p,
        drift_deg_per_hour=drift,
        max_overshoot=max_overshoot,
        settling_time_s=float(settling_time),
    )


def read_all_channels(server):
    ts = datetime.datetime.now()
    out = {}
    for ch_name, cmd in TEMP_CHANNELS.items():
        try:
            out[ch_name] = float(server.get_temp(cmd))
        except Exception as e:
            print(f"Error reading {ch_name}: {e}")
            out[ch_name] = np.nan
    return ts, out


def save_trace(save_dir, tag, channel, timestamps, temps):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"trace_{tag}_{channel}.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Timestamp", "Temperature"])
        for ts, t in zip(timestamps, temps):
            w.writerow([ts.isoformat(), f"{t:.6f}"])
    return path


def cost_from_metrics(m, weights):
    return (
        weights["std"] * (m["std"] if np.isfinite(m["std"]) else 1e9)
        + weights["settling"]
        * (m["settling_time_s"] if np.isfinite(m["settling_time_s"]) else 1e9)
        + weights["drift"]
        * (
            abs(m["drift_deg_per_hour"])
            if np.isfinite(m["drift_deg_per_hour"])
            else 1e9
        )
    )


def evaluate_setting(server, P, I, D, summary_writer):
    """Record for DURATION, compute metrics for each channel, log, return cost on OPTIMIZE_ON_CHANNEL."""
    # Set PID
    server.set_param(OUTPUTCHANNEL + b".PID.P", P)
    server.set_param(OUTPUTCHANNEL + b".PID.I", I)
    server.set_param(OUTPUTCHANNEL + b".PID.D", D)
    time.sleep(SLEEP_BETWEEN)

    timestamps = []
    channel_traces = {ch: [] for ch in TEMP_CHANNELS}

    start = time.time()
    tag = f"P{P}_I{I}_D{D}"
    print(f"  -> Recording {tag} ...")

    while time.time() - start < DURATION:
        ts, reading = read_all_channels(server)
        timestamps.append(ts)
        for ch, val in reading.items():
            channel_traces[ch].append(val)

        if len(timestamps) % 10 == 0:
            last_line = ", ".join(
                f"{ch}:{channel_traces[ch][-1]:.3f}" for ch in TEMP_CHANNELS
            )
            print(f"[{ts}] {last_line}")
        time.sleep(SAMPLE_PERIOD)

    # Compute metrics & save per channel
    best_channel_cost = None
    for ch, temps in channel_traces.items():
        metrics = compute_metrics(timestamps, temps)
        trace_path = save_trace(SAVE_DIR, tag, ch, timestamps, temps)

        # Write to summary
        summary_writer.writerow(
            [
                P,
                I,
                D,
                ch,
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

        # If this is the optimization channel, compute cost
        if ch == OPTIMIZE_ON_CHANNEL:
            best_channel_cost = cost_from_metrics(metrics, COST_WEIGHTS)

    # rest
    time.sleep(REST_AFTER)
    return best_channel_cost


def tune_pid_sequential():
    cxn = common.labrad_connect()
    server = cxn.temp_monitor_SRS_ptc10
    os.makedirs(SAVE_DIR, exist_ok=True)

    summary_path = os.path.join(SAVE_DIR, "summary_seq.csv")
    with open(summary_path, "w", newline="") as fsum:
        writer = csv.writer(fsum)
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
        fsum.flush()
        # ------------------ Sweep P ------------------
        print("\n== Sweep P (I,D fixed) ==")
        best_P, best_cost = None, np.inf
        for P in P_SWEEP:
            cost = evaluate_setting(server, P, I_INIT, D_INIT, writer)
            print(f"  P={P}: cost={cost:.6g}")
            if cost < best_cost:
                best_cost = cost
                best_P = P
        print(f"--> Best P = {best_P} (cost={best_cost:.6g})")

        # ------------------ Sweep I ------------------
        print("\n== Sweep I (P*,D fixed) ==")
        best_I, best_cost = None, np.inf
        for I in I_SWEEP:
            cost = evaluate_setting(server, best_P, I, D_INIT, writer)
            print(f"  I={I}: cost={cost:.6g}")
            if cost < best_cost:
                best_cost = cost
                best_I = I
        print(f"--> Best I = {best_I} (cost={best_cost:.6g})")

        # ------------------ Sweep D ------------------
        print("\n== Sweep D (P*,I* fixed) ==")
        best_D, best_cost = None, np.inf
        for D in D_SWEEP:
            cost = evaluate_setting(server, best_P, best_I, D, writer)
            print(f"  D={D}: cost={cost:.6g}")
            if cost < best_cost:
                best_cost = cost
                best_D = D
        print(f"--> Best D = {best_D} (cost={best_cost:.6g})")

    print("\nSequential PID tuning complete.")
    print(f"Best values: P={best_P}, I={best_I}, D={best_D}")
    print("Summary saved to:", summary_path)


if __name__ == "__main__":
    # tune_pid_sequential()
    set_pid()
