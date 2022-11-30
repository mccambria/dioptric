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
import zfs_vs_t_main

nvdata_dir = common.get_nvdata_dir()

# endregion

# region Functions


def main(file_list, monitor_list, do_plot=False):

    path_to_logs = nvdata_dir / "paper_materials/zfs_temp_dep/temp_monitoring"

    ###

    # Get all the times and temps out of the log files
    times = {}
    temps = {}
    for monitor_folder in path_to_logs.iterdir():
        times_list = []
        temps_list = []
        for log_file in monitor_folder.iterdir():
            if not log_file.is_file():
                continue
            with open(log_file) as csv_file:
                reader = csv.reader(csv_file)
                for row in reader:
                    times_list.append(int(row[0]))
                    temps_list.append(float(row[1]))
        monitor = monitor_folder.stem
        times[monitor] = times_list
        temps[monitor] = temps_list

    if do_plot:
        kpl.init_kplotlib()

    mean_temps = []
    ptp_temps = []
    for ind in range(len(file_list)):
        data_file = file_list[ind]
        monitor = monitor_list[ind]
        # First 19 characters are human-readable timestamp
        date_time_str = data_file[0:19]
        # Assume timezone is CST
        date_time_str += "-CST"
        date_time = datetime.strptime(date_time_str, r"%Y_%m_%d-%H_%M_%S-%Z")
        timestamp = date_time.timestamp()
        min_time = timestamp - (15 * 60)
        max_time = timestamp
        file_times = []
        file_temps = []
        monitor_times = times[monitor]
        monitor_temps = temps[monitor]
        for ind in range(len(monitor_times)):
            time = monitor_times[ind]
            if min_time <= time <= max_time:
                file_times.append((time - timestamp) / 60)
                file_temps.append(monitor_temps[ind])
        if len(file_temps) == 0:
            file_times.append(-1)
            file_temps.append(-1)
        mean_file_temp = round(np.mean(file_temps), 1)
        mean_temps.append(mean_file_temp)
        ptp_file_temp = round(max(file_temps) - min(file_temps), 3)
        ptp_temps.append(ptp_file_temp)
        if do_plot:
            fig, ax = plt.subplots()
            ax.set_xlabel("Time to expt end (m)")
            ax.set_ylabel(f"Temp - {mean_file_temp} (K)")
            ax.set_title(kpl.tex_escape(data_file))
            file_temps = [val - mean_file_temp for val in file_temps]
            kpl.plot_line(ax, file_times, file_temps)
            kpl.tight_layout(fig)

    print(mean_temps)
    print(ptp_temps)

    if do_plot:
        plt.show(block=True)


# endregion

if __name__ == "__main__":

    data_points = zfs_vs_t_main.get_data_points(override_skips=True)
    file_list = [
        el["ZFS file"] for el in data_points if not el["ZFS file"] == ""
    ]
    monitor_list = [
        el["Monitor"] for el in data_points if not el["ZFS file"] == ""
    ]
    # file_list = ["2022_11_18-11_27_31-wu-nv4_zfs_vs_t"]

    main(file_list, monitor_list, do_plot=False)