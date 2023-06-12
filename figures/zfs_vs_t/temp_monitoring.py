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
            if not log_file.is_file() or log_file.suffix != ".log":
                continue
            with open(log_file) as csv_file:
                reader = csv.reader(csv_file)
                for row in reader:
                    try:
                        time = int(row[0])
                        temp = float(row[1])
                    except Exception as exc:
                        continue
                    times_list.append(time)
                    temps_list.append(temp)
        monitor = monitor_folder.stem
        times[monitor] = times_list
        temps[monitor] = temps_list

    if do_plot:
        kpl.init_kplotlib()

    mean_temps = []
    ptp_temps = []
    std_temps = []
    for ind in range(len(file_list)):
        expt_file = file_list[ind]
        end_time = tool_belt.utc_from_file_name(expt_file)
        # Experiments take at least 10 minutes, so use this for the lookback
        start_time = end_time - (60 * 60)
        monitor = monitor_list[ind]
        expt_times = []
        expt_temps = []
        monitor_times = times[monitor]
        monitor_temps = temps[monitor]
        for ind in range(len(monitor_times)):
            time = monitor_times[ind]
            if start_time <= time <= end_time:
                expt_times.append((time - start_time) / 60)
                expt_temps.append(monitor_temps[ind])
        if len(expt_temps) == 0:
            expt_times.append(-1)
            expt_temps.append(-1)
        mean_expt_temp = round(np.mean(expt_temps), 1)
        mean_temps.append(mean_expt_temp)
        ptp_expt_temp = round(max(expt_temps) - min(expt_temps), 3)
        ptp_temps.append(ptp_expt_temp)
        std_expt_temp = round(np.std(expt_temps), 3)
        std_temps.append(std_expt_temp)
        if do_plot:
            fig, ax = plt.subplots()
            ax.set_xlabel("Time to expt end (m)")
            ax.set_ylabel(f"Temp - {mean_expt_temp} (K)")
            ax.set_title(kpl.tex_escape(expt_file))
            expt_temps = [val - mean_expt_temp for val in expt_temps]
            kpl.plot_line(ax, expt_times, expt_temps)

    print(mean_temps)
    print(ptp_temps)
    print(std_temps)

    if do_plot:
        plt.show(block=True)


# endregion

if __name__ == "__main__":
    # data_points = zfs_vs_t_main.get_data_points()
    # start_time_list = [
    #     el["Start time (UTC)"] for el in data_points if not el["ZFS file"] == ""
    # ]
    # end_time_list = [
    #     el["End time (UTC)"] for el in data_points if not el["ZFS file"] == ""
    # ]
    # file_list = [el["ZFS file"] for el in data_points if not el["ZFS file"] == ""]
    # monitor_list = [el["Monitor"] for el in data_points if not el["ZFS file"] == ""]

    # file_list = ["2022_11_15-14_11_22-wu-nv3_zfs_vs_t"]
    # file_list = [
    #     "2023_01_12-15_21_57-wu-nv7_zfs_vs_t",  # 310
    #     "2023_01_13-11_29_29-wu-nv10_zfs_vs_t",  # 400
    #     "2023_01_14-21_23_31-wu-nv6_zfs_vs_t",  # 500
    # ]
    # monitor_list = ["PT100"] * len(file_list)
    file_list = [
        # "2022_12_03-21_24_05-15micro-nv1_zfs_vs_t",
        "2022_11_15-14_11_22-wu-nv3_zfs_vs_t",  # 15
        # "2022_11_15-14_25_56-wu-nv2_zfs_vs_t",
        # "2022_11_15-14_54_40-wu-nv5_zfs_vs_t",
    ]
    monitor_list = ["lakeshore_X162690"] * len(file_list)

    main(file_list, monitor_list, do_plot=True)
