# -*- coding: utf-8 -*-
"""
This file contains standardized functions intended to simplify the
creation of plots for publications in a consistent style.

Created on November 17th, 2022

@author: mccambria
"""


# region Imports


import matplotlib.pyplot as plt
from enum import Enum
from colorutils import Color
import re

# endregion

# region Constants


# endregion


# region Colors


def replot(path_to_file):
    

    nvdata_dir = common.get_nvdata_dir()
    path_to_file = (
        nvdata_dir
        # / "pc_hahn/service_logging/old/2022_11_16-calibrated_temp_monitor.log"
        / "pc_hahn/service_logging/calibrated_temp_monitor.log"
    )
    replot(path_to_file)

    center_time = 1668712861
    min_time = center_time - (15 * 60)
    max_time = center_time + (15 * 60)

    times = []
    temps = []

    with open(path_to_file) as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            time = int(row[0])
            if min_time <= time <= max_time:
                times.append(time - min_time)
                temp = float(row[1])
                temps.append(temp)

    kpl.init_kplotlib()
    fig, ax = plt.subplots()
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Temp (K)")
    kpl.plot_line(ax, times, temps)
    kpl.tight_layout(fig)
    plt.show(block=True)


# endregion


