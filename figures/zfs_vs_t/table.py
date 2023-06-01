# -*- coding: utf-8 -*-
"""
Tabulate NV ZFS vs T measurements for LaTeX

Created on May 25th, 2023

@author: mccambria
"""

# region Import and constants

import numpy as np
from utils import common
from majorroutines.pulsed_resonance import return_res_with_error
import majorroutines.pulsed_resonance as pesr
import utils.tool_belt as tool_belt
from utils.tool_belt import bose
import matplotlib.pyplot as plt
import matplotlib as mpl
from utils import kplotlib as kpl
from utils.kplotlib import KplColors
from scipy.optimize import curve_fit
import csv
import pandas as pd
import sys
from figures.zfs_vs_t.zfs_vs_t_main import get_data_points


def main():
    def skip_lambda(point):
        return point["Sample"] != "Wu"

    aliases = {
        "nv1_zfs_vs_t": "A1",
        "nv2_zfs_vs_t": "A2",
        "nv3_zfs_vs_t": "A3",
        "nv4_zfs_vs_t": "A4",
        "nv5_zfs_vs_t": "A5",
        "nv6_zfs_vs_t": "B1",
        "nv7_zfs_vs_t": "B2",
        "nv8_zfs_vs_t": "B3",
        "nv10_zfs_vs_t": "B4",
        "nv11_zfs_vs_t": "B5",
    }
    name_orders = {
        "nv1_zfs_vs_t": 0,
        "nv2_zfs_vs_t": 1,
        "nv3_zfs_vs_t": 2,
        "nv4_zfs_vs_t": 3,
        "nv5_zfs_vs_t": 4,
        "nv6_zfs_vs_t": 5,
        "nv7_zfs_vs_t": 6,
        "nv8_zfs_vs_t": 7,
        "nv10_zfs_vs_t": 8,
        "nv11_zfs_vs_t": 9,
    }

    data_points = get_data_points(skip_lambda)
    data_points_order = []
    for point in data_points:
        temp = point["Monitor temp (K)"]
        name_order = name_orders[point["NV"]]
        order = 0
        for test in data_points:
            test_temp = test["Monitor temp (K)"]
            test_name_order = name_orders[test["NV"]]
            if test_temp < temp - 0.5:
                order += 1
            elif (np.abs(test_temp - temp) < 0.5) and (test_name_order < name_order):
                order += 1
        data_points_order.append(order)

    print_str = ""
    new_line_ctr = 1
    for order in data_points_order:
        point = data_points[order]
        nv = aliases[point["NV"]]
        temp = point["Monitor temp (K)"]
        zfs = point["ZFS (GHz)"]
        zfs_err = point["ZFS (GHz) error"]
        zfs *= 1000
        zfs_err *= 1000
        print_zfs = tool_belt.round_for_print(zfs, zfs_err)
        point_str = f"{nv} & {temp} & {print_zfs}"
        print_str += point_str

        if new_line_ctr % 5 == 0:
            print_str += " \\\\"
        else:
            print_str += " & "
        new_line_ctr += 1

    print(print_str)


if __name__ == "__main__":
    main()
