# -*- coding: utf-8 -*-
"""
Survey of NV zero field lines in the sample Wu

Created on February 16th, 2023

@author: matth
"""


# region Import and constants


import numpy as np
from majorroutines.pulsed_resonance import return_res_with_error
import majorroutines.pulsed_resonance as pesr
import utils.tool_belt as tool_belt
from utils.tool_belt import bose
import matplotlib.pyplot as plt
from utils import common
from utils import kplotlib as kpl
from utils.kplotlib import KplColors
from scipy.optimize import curve_fit
import csv
import pandas as pd
import sys

nvdata_dir = common.get_nvdata_dir()
compiled_data_file_name = "zfs_survey"
compiled_data_path = nvdata_dir / "paper_materials/zfs_temp_dep"


# endregion
# region Functions


def get_data_points(skip_lambda=None):

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
                columns = row
                header = False
                continue

            point = {}
            for ind in range(len(columns)):
                column = columns[ind]
                raw_val = row[ind]
                if raw_val == "TRUE":
                    val = True
                else:
                    try:
                        val = eval(raw_val)
                    except Exception:
                        val = raw_val
                point[column] = val

            skip = skip_lambda is not None and skip_lambda(point)
            if not skip:
                data_points.append(point)

    return data_points


def data_points_to_lists(data_points):
    """Turn a dict of data points into a list that's more convenenient for plotting"""

    zfs_list = []
    zfs_err_list = []
    temp_list = []
    label_list = []
    color_list = []
    # data_color_options = kpl.data_color_cycler.copy()
    # data_color_options.pop(0)
    data_color_options = [
        KplColors.GREEN,
        KplColors.PURPLE,
        KplColors.BROWN,
        KplColors.PINK,
        KplColors.GRAY,
        KplColors.YELLOW,
        KplColors.CYAN,
    ]
    color_dict = {}
    used_labels = []  # For matching colors to labels
    for el in data_points:
        zfs = el["ZFS (GHz)"]
        monitor_temp = el["Monitor temp (K)"]
        if zfs == "" or monitor_temp == "":
            continue
        # if not (min_temp <= reported_temp <= max_temp):
        # if monitor_temp < 296:
        #     # zfs += 0.00042
        #     monitor_temp *= 300.7 / 295
        temp_list.append(monitor_temp)
        zfs_list.append(zfs)
        zfs_err = el["ZFS error (GHz)"]
        zfs_err_list.append(zfs_err)
        label = el["Label"]
        label_list.append(label)
        if label not in used_labels:
            used_labels.append(label)
            color_dict[label] = data_color_options.pop(0)
        color_list.append(color_dict[label])

    return zfs_list, zfs_err_list, temp_list, label_list, color_list


def calc_zfs_from_compiled_data():

    skip_lambda = lambda point: point["Sample"] != "Wu"

    data_points = get_data_points(skip_lambda)
    zfs_list = []
    zfs_err_list = []
    for el in data_points:
        zfs_file_name = el["ZFS file"]
        if zfs_file_name == "":
            zfs_list.append(-1)
            zfs_err_list.append(-1)
            continue
        data = tool_belt.get_raw_data(zfs_file_name)
        res, res_err = return_res_with_error(data)
        zfs_list.append(res)
        zfs_err_list.append(res_err)
    zfs_list = [round(val, 6) for val in zfs_list]
    zfs_err_list = [round(val, 6) for val in zfs_err_list]
    print(zfs_list)
    print(zfs_err_list)


# endregion
# region Fitting functions


# endregion
# region Secondary plots


# endregion


def main():

    pass


if __name__ == "__main__":

    kpl.init_kplotlib()

    main()

    plt.show(block=True)
