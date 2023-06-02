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
    # main()

    labels = ["Chen", "Toyli", "Doherty", "Li", "Barson"]
    params_list = [
        [
            2.87739103e00,
            -2.18259231e-06,
            9.35186529e-08,
            -9.06702213e-10,
            1.41418530e-12,
            -8.01851281e-16,
        ],
        [2.87681320e00, 2.18345251e-05, -1.39327457e-07, -2.69568020e-11],
        [2.87693019e00, 1.01813694e-09, -1.66233320e-12],
        [2.87773547e00, 1.92413940e-07, 1.52014969e02],
        [2.87743616e00, -1.77160810e-09, 4.81437544e-12, -3.78265704e-15],
    ]
    param_errs_list = [
        [
            7.59555359e-05,
            2.98832757e-06,
            3.39577677e-08,
            1.57626332e-10,
            3.21814315e-13,
            2.40102970e-16,
        ],
        [4.71585333e-05, 6.81134332e-07, 2.82191816e-09, 3.42222683e-12],
        [1.58206229e-05, 4.31008631e-12, 8.78608098e-15],
        [2.14739748e-05, 1.28774012e-09, 2.19116352e00],
        [1.97216807e-05, 2.49648091e-11, 1.11643152e-13, 1.26067461e-16],
    ]

    for ind in range(5):
        print(labels[ind])
        params = params_list[ind]
        param_errs = param_errs_list[ind]
        print_params = []
        for jnd in range(len(params)):
            num_sig_figs = 6 if jnd == 0 else 4
            # print_params.append(
            #     tool_belt.round_for_print_sci_latex(params[jnd], param_errs[jnd])
            # )
            print_params.append(tool_belt.round_sig_figs(params[jnd], num_sig_figs))
        print(print_params)
