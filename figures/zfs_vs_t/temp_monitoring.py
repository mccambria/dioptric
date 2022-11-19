# -*- coding: utf-8 -*-
"""
Plot temperature montioring data for zfs experiments

Created on November 17th, 2022

@author: mccambria
"""


# region Imports and constants

import utils.common as common
import utils.kplotlib as kpl
import utils.tool_belt as tool_belt
import numpy as np
import matplotlib.pyplot as plt
import csv
from datetime import datetime
import pandas as pd

nvdata_dir = common.get_nvdata_dir()
compiled_data_file_name = "zfs_vs_t"
compiled_data_path = nvdata_dir / "paper_materials/zfs_temp_dep"

# endregion

# region Functions


def get_data_points():

    override_skips = False

    xl_file_path = compiled_data_path / f"{compiled_data_file_name}.xlsx"
    csv_file_path = compiled_data_path / f"{compiled_data_file_name}.csv"
    compiled_data_file = pd.read_excel(xl_file_path, engine="openpyxl")
    compiled_data_file.to_csv(csv_file_path, index=None, header=True)

    data_points = []
    with open(csv_file_path, newline="") as f:

        reader = csv.reader(f)
        header = True
        for row in reader:
            # Create columns from the header (first row)
            if header:
                columns = row[1:]
                header = False
                continue

            point = {}
            for ind in range(len(columns)):
                column = columns[ind]
                raw_val = row[1 + ind]
                if raw_val == "TRUE":
                    val = True
                else:
                    try:
                        val = eval(raw_val)
                    except Exception:
                        val = raw_val
                point[column] = val

            if override_skips or not point["Skip"]:
                data_points.append(point)

    return data_points


def main(file_list):

    do_plot = False

    path_to_logs = nvdata_dir / "pc_hahn/service_logging/zfs_vs_t"

    ###

    # Get all the times and temps out of the log files
    times = []
    temps = []
    for log_file in path_to_logs.iterdir():
        if not log_file.is_file():
            continue
        with open(log_file) as csv_file:
            reader = csv.reader(csv_file)
            for row in reader:
                times.append(int(row[0]))
                temps.append(float(row[1]))

    if do_plot:
        kpl.init_kplotlib()

    mean_temps = []
    ptp_temps = []
    for data_file in file_list:
        # First 19 characters are human-readable timestamp
        date_time_str = data_file[0:19]
        # Assume timezone is CST
        date_time_str += " CST"
        date_time = datetime.strptime(date_time_str, r"%Y_%m_%d-%H_%M_%S %Z")
        timestamp = date_time.timestamp()
        min_time = timestamp - (15 * 60)
        max_time = timestamp
        file_times = []
        file_temps = []
        for ind in range(len(times)):
            time = times[ind]
            if min_time <= time <= max_time:
                file_times.append((time - timestamp) / 60)
                file_temps.append(temps[ind])
        if len(file_temps) == 0:
            continue
        mean_file_temp = round(np.mean(file_temps), 1)
        mean_temps.append(mean_file_temp)
        ptp_file_temp = round(max(file_temps) - min(file_temps), 3)
        ptp_temps.append(ptp_file_temp)
        file_temps = [val - mean_file_temp for val in file_temps]
        if do_plot:
            fig, ax = plt.subplots()
            ax.set_xlabel("Time to expt end (m)")
            ax.set_ylabel(f"Temp - {mean_file_temp} (K)")
            ax.set_title(kpl.tex_escape(data_file))
            kpl.plot_line(ax, file_times, file_temps)
            kpl.tight_layout(fig)

    print(mean_temps)
    print(ptp_temps)

    if do_plot:
        plt.show(block=True)


# endregion

if __name__ == "__main__":

    data_points = get_data_points()
    file_list = [
        el["ZFS file"] for el in data_points if not el["ZFS file"] == ""
    ]
    # file_list = ["2022_11_18-11_27_31-wu-nv4_zfs_vs_t"]

    main(file_list)
