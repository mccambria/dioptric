# -*- coding: utf-8 -*-
"""
Created on July 3rd, 2025

@author: Saroj B Chand

Live plot for laser power data from NI DAQ logger
"""

import datetime
import os

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from dateutil.relativedelta import relativedelta

# === USER SETTINGS ===
base_folder = "G:\\NV_Widefield_RT_Setup_Lasers_Power_Logs"
hours = 48  # Number of hours to plot
plot_power = True  # True = plot power (mW), False = plot voltage (V)

# Label → filename mapping (same as in logger)
channels = {
    "589nm_fiber_out": "laser_589nm_fiber_out.csv",
    # "638nm_back_reflection": "laser_638nm_back_reflection.csv",
}

# === Determine folders for current and previous month
now = datetime.datetime.now()
folder_current = now.strftime("%m%Y")
folder_previous = (now - relativedelta(months=1)).strftime("%m%Y")
data_folders = [
    os.path.join(base_folder, folder_previous),
    os.path.join(base_folder, folder_current),
]

# === Live Plot Setup
plt.ion()
fig, ax = plt.subplots(figsize=(12, 5))


def update_plot():
    ax.clear()
    now = datetime.datetime.now()

    for label, filename in channels.items():
        dfs = []

        for folder in data_folders:
            file_path = os.path.join(folder, filename)
            if not os.path.exists(file_path):
                continue

            try:
                df = pd.read_csv(
                    file_path,
                    names=["Timestamp", "Voltage", "Power_mW"],
                    parse_dates=["Timestamp"],
                    dtype={"Voltage": float, "Power_mW": float},
                )
                dfs.append(df)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

        if not dfs:
            print(f"No data found for {label}")
            continue

        df_all = pd.concat(dfs)
        df_all = df_all[df_all["Timestamp"] > (now - datetime.timedelta(hours=hours))]

        y_col = "Power_mW" if plot_power else "Voltage"
        y_label = "Laser Power [mW]" if plot_power else "Voltage [V]"

        ax.plot(df_all["Timestamp"], df_all[y_col], label=label)

    ax.set_title(f"Laser Power Monitor (Last {hours} h)")
    ax.set_xlabel("Time")
    ax.set_ylabel(y_label)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
    ax.legend()
    fig.autofmt_xdate()
    plt.pause(0.1)


def main():
    print(f"Live plotting from: {base_folder}")
    try:
        while True:
            update_plot()
            if (
                input("Press Enter to refresh or type 'q' to quit: ").strip().lower()
                == "q"
            ):
                break
    finally:
        print("Exiting and closing plot.")
        plt.close()


if __name__ == "__main__":
    main()
