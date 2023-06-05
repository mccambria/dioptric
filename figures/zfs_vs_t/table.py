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

    labels = ["This work", "Chen", "Toyli", "Doherty", "Li", "Barson"]
    params_list = [
        [-7.54373061e-02, -2.49004667e-01, 6.38800583e01, 1.62805204e02],
        [
            -5.60672993e-06,
            1.61129595e-07,
            -1.30943594e-09,
            2.31163392e-12,
            -1.48142926e-15,
        ],
        [1.64910962e-05, -1.27410830e-07, -3.66619701e-11],
        [1.16090179e-09, -1.93169794e-12],
        [2.02884792e-07, 1.68769952e02],
        [-1.81265247e-09, 4.89589231e-12, -3.79468466e-15],
    ]
    param_errs_list = [
        [6.23938155e-03, 3.35225677e-02, 1.45275912e00, 1.15701991e01],
        [
            1.43760347e-06,
            2.10646783e-08,
            1.08840964e-10,
            2.36789920e-13,
            1.84577121e-16,
        ],
        [3.28639435e-07, 1.84219645e-09, 2.52755062e-12],
        [2.83471371e-12, 6.23345595e-15],
        [1.00996510e-09, 1.47103253e00],
        [1.80119384e-11, 8.58117690e-14, 1.01130822e-16],
    ]

    for ind in range(len(labels)):
        print()
        print(labels[ind])
        params = params_list[ind]
        param_errs = param_errs_list[ind]
        print_params = []
        for jnd in range(len(params)):
            # num_sig_figs = 6 if jnd == 0 else 4
            num_sig_figs = 4
            print_params.append(
                tool_belt.round_for_print_sci_latex(params[jnd], param_errs[jnd])
            )
            # print_params.append(tool_belt.round_sig_figs(params[jnd], num_sig_figs))
        print(print_params)
