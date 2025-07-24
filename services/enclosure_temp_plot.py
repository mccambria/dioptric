# -*- coding: utf-8 -*-
"""
Created on June 16th, 2023

@author: Saroj B Chand
"""

import datetime
import os
import time

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

from utils import kplotlib as kplt

kplt.init_kplotlib()
# Base folder and current month-year folder
base_folder = "G:\\NV_Widefield_RT_Setup_Enclosure_Temp_Logs"
folder = datetime.datetime.now().strftime("%m%Y")
data_folder = os.path.join(base_folder, folder)

# Determine both current and previous month folders
now = datetime.datetime.now()
folder_current = now.strftime("%m%Y")
folder_previous = (now - relativedelta(months=1)).strftime("%m%Y")

data_folders = [
    os.path.join(base_folder, folder_previous),
    os.path.join(base_folder, folder_current),
]

# Define channels and corresponding filenames
channels = {
    "4A": "temp_4A.csv",
    "4B": "temp_4B.csv",
    "4C": "temp_4C.csv",
    "4D": "temp_4D.csv",
    # "temp_stick": "temp_stick.csv",
}
# Live plot setup
plt.ion()
fig, ax = plt.subplots(figsize=(10, 5))
hours = 2.0  # for plotting

timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print(f"Current Time Stamp: {timestamp}")


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
                    names=["Timestamp", "Temperature"],
                    parse_dates=["Timestamp"],
                    dtype={"Temperature": float},
                )
                dfs.append(df)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

        if not dfs:
            print(f"No data found for Channel {label}")
            continue

        df_all = pd.concat(dfs)
        # df_all = df_all[df_all["Timestamp"] > (now - datetime.timedelta(hours=hours))]
        df_all = pd.concat(dfs)
        # print(f"iqr: {iqr}, median: {median}")
        df_all = df_all[
            (df_all["Timestamp"] > (now - datetime.timedelta(hours=hours)))
            & (df_all["Temperature"] > 15)
            & (df_all["Temperature"] < 25)
        ]
        std = np.std(df_all["Temperature"])
        # rang = np.max(df_all["Temperature"]) - np.min(df_all["Temperature"])
        print(f"std {label}:{std}")
        # Plot
        ax.plot(df_all["Timestamp"], df_all["Temperature"], label=f"Channel {label}")

    ax.set_title(f"Temperature Plot (Last {hours}h)", fontsize=13)
    ax.set_xlabel("Time", fontsize=13)
    ax.set_ylabel("Temperature [°C]", fontsize=13)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
    ax.tick_params(axis="both", labelsize=11)
    ax.legend(fontsize=11)
    fig.autofmt_xdate()

    # fig.text()
    fig.text(
        0.40,
        0.50,
        "4A --> near sample \n4B --> air inside duct of experiment enclosure.\n4C --> air inside duct of laser enclosure. \n4D --> air inside laser enclosure.",
        ha="left",
        va="bottom",
        fontsize=11,
    )
    plt.pause(0.1)


def main():
    print(f"Live plotting from: {data_folder}")
    try:
        while True:
            update_plot()
            if (
                input("Press Enter to rephresh or type 'q' to quit: ").strip().lower()
                == "q"
            ):
                break
    finally:
        print("Exiting and closing plot.")
        plt.close()


if __name__ == "__main__":
    main()
