# -*- coding: utf-8 -*-
"""
Created on July 3rd, 2025

@author: Saroj B Chand

Live plot for laser power data from NI DAQ logger
"""

import csv
import datetime
import os
import time

import numpy as np

from utils import common

# ----------- USER SETTINGS -----------
# TEMP_CHANNELS = {
#     "4A": b"4A?\n",
#     "4B": b"4B?\n",
#     "4C": b"4C?\n",
#     "4D": b"4D?\n",
# }
CMD = b"4A?\n"
OUTPUTCHANNEL = b"OUT1"
DURATION = 300  # Seconds to record per PID setting
SLEEP_BETWEEN = 2  # Seconds to wait between PID setting changes
SAVE_DIR = "G:/NV_Widefield_RT_Setup_Enclosure_Temp_Logs/pid_tuning"

# Ranges to sweep
P_vals = [50, 100, 150]
I_vals = [1, 2, 3]
D_vals = [40, 60, 80]
# -------------------------------------


def tune_pid():
    cxn = common.labrad_connect()
    server = cxn.temp_monitor_SRS_ptc10

    os.makedirs(SAVE_DIR, exist_ok=True)
    results = []

    for P in P_vals:
        for I in I_vals:
            for D in D_vals:
                # 1. Set PID values
                server.set_param(OUTPUTCHANNEL + b".PID.P", P)
                server.set_param(OUTPUTCHANNEL + b".PID.I", I)
                server.set_param(OUTPUTCHANNEL + b".PID.D", D)
                print(f"\n>>> Tuning PID: P={P}, I={I}, D={D}")
                time.sleep(SLEEP_BETWEEN)

                # 2. Collect data
                temps = []
                timestamps = []
                start_time = time.time()
                while (time.time() - start_time) < DURATION:
                    try:
                        temp = server.get_temp(CMD)
                        temps.append(temp)
                        timestamps.append(datetime.datetime.now())
                        print(f"[{timestamps[-1]}] Temp: {temp:.3f} °C")
                        time.sleep(1)
                    except Exception as e:
                        print(f"Error reading temp: {e}")

                # 3. Analyze
                temps = np.array(temps)
                std_dev = np.std(temps)
                max_overshoot = np.max(temps) - temps[0]
                settling_time = calc_settling_time(temps, threshold=0.02)

                results.append((P, I, D, std_dev, max_overshoot, settling_time))

                # 4. Save data
                file_suffix = f"P{P}_I{I}_D{D}"
                with open(
                    os.path.join(SAVE_DIR, f"temp_trace_{file_suffix}.csv"),
                    "w",
                    newline="",
                ) as f:
                    writer = csv.writer(f)
                    writer.writerow(["Timestamp", "Temperature"])
                    writer.writerows(zip(timestamps, temps))
                # time.sleep(120)
    # 5. Save summary
    with open(os.path.join(SAVE_DIR, "summary.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["P", "I", "D", "StdDev", "MaxOvershoot", "SettlingTime"])
        writer.writerows(results)

    print("\n PID tuning complete. Summary saved to:", SAVE_DIR)


def calc_settling_time(data, threshold=0.02):
    """
    Estimate settling time (time to stay within ±threshold of final value).
    """
    final_value = data[-1]
    lower = final_value * (1 - threshold)
    upper = final_value * (1 + threshold)
    for i in range(len(data) - 1, -1, -1):
        if not (lower <= data[i] <= upper):
            return len(data) - i  # in seconds (assuming 1 Hz)
    return 0


if __name__ == "__main__":
    tune_pid()
