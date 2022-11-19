# -*- coding: utf-8 -*-
"""
Plot temperature montioring data for zfs experiments

Created on November 17th, 2022

@author: mccambria
"""


# region Imports

import utils.common as common
import utils.kplotlib as kpl
import utils.tool_belt as tool_belt
import numpy as np
import matplotlib.pyplot as plt
import csv
from datetime import datetime

# endregion

# region Constants


nvdata_dir = common.get_nvdata_dir()
data_file_path = "rsysr"


# endregion


# region Colors


def replot(file_list):

    path_to_folder = nvdata_dir / "pc_hahn/service_logging/zfs_vs_t"

    # center_time = 1668712861
    # min_time = center_time - (15 * 60)
    # max_time = center_time + (15 * 60)

    # Get all the times and temps out of the log files
    times = []
    temps = []
    for log_file in path_to_folder.iterdir():
        if not log_file.is_file():
            continue
        with open(log_file) as csv_file:
            reader = csv.reader(csv_file)
            for row in reader:
                times.append(int(row[0]))
                temps.append(float(row[1]))

    kpl.init_kplotlib()

    for data_file in file_list:
        # First 19 characters are human-readable timestamp
        date_time_str = data_file[0:19]
        date_time = datetime.strptime(date_time_str, r"%Y_%m_%d-%H_%M_%S")
        date_time.tzinfo = "CST"
        timestamp = date_time.timestamp()
        fig, ax = plt.subplots()
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Temp (K)")
        kpl.plot_line(ax, times, temps)
        kpl.tight_layout(fig)

    plt.show(block=True)


# endregion


if __name__ == "__main__":

    file_list = ["2022_11_18-11_27_31-wu-nv4_zfs_vs_t"]
    replot(file_list)
    # replot(None)
