# -*- coding: utf-8 -*-
"""
Survey of NV zero field lines in the sample Wu

Created on February 16th, 2023

@author: mccambria
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


def get_header():

    xl_file_path = compiled_data_path / f"{compiled_data_file_name}.xlsx"
    csv_file_path = compiled_data_path / f"{compiled_data_file_name}.csv"
    compiled_data_file = pd.read_excel(xl_file_path, engine="openpyxl")
    compiled_data_file.to_csv(csv_file_path, index=None, header=True)
    with open(csv_file_path, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            header = row
            break
    return header


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

    data_points = condense_data_points(data_points)

    return data_points


def condense_data_points(data_points):
    """Combine measurements on the same NV"""

    header = get_header()

    condensed_data_points = []
    identifier_set = [point["NV"] for point in data_points]
    identifier_set = list(set(identifier_set))
    identifier_set.sort()

    for identifier in identifier_set:
        new_point = {}
        for col in header:
            new_point[col] = []
        for point in data_points:
            test_identifier = point["NV"]
            if test_identifier == identifier:
                for col in header:
                    val = point[col]
                    if val != "":
                        new_point[col].append(point[col])

        for col in header:
            if len(new_point[col]) == 0:
                continue
            first_val = new_point[col][0]
            # Only combine numeric data (floats)
            if type(first_val) is not float:
                new_point[col] = new_point[col][0]
            # Inverse-variance average the points together
            elif col.endswith("error"):
                errors = new_point[col]
                if 0 in errors:
                    new_point[col] = 0
                    continue
                # For inverse-variance weighting, condensed_error**2 = 1/norm
                weights = [val**-2 for val in errors]
                norm = np.sum(weights)
                new_point[col] = np.sqrt(1 / norm)
            else:
                error_col = col + " error"
                errors = new_point[error_col]
                if 0 in errors:
                    ind = np.where(np.array(errors) == 0)[0][0]
                    new_point[col] = new_point[col][ind]
                    continue
                weights = [val**-2 for val in errors]
                new_point[col] = np.average(new_point[col], weights=weights)

        condensed_data_points.append(new_point)

    return condensed_data_points


def data_points_to_lists(data_points):
    """Turn a dict of data points into a list that's more convenenient for plotting"""

    data_lists = {}
    header = get_header()
    for el in header:
        data_lists[el] = []
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
    for point in data_points:
        for col in header:
            data_lists[col].append(point[col])
        # label = el["Label"]
        # label_list.append(label)
        # if label not in used_labels:
        #     used_labels.append(label)
        #     color_dict[label] = data_color_options.pop(0)
        # color_list.append(color_dict[label])

    return data_lists


# endregion
# region Fitting functions


# endregion
# region Secondary plots


# endregion


def main():

    skip_lambda = lambda pt: pt["Skip"]
    # skip_lambda = lambda pt: pt["Skip"] or pt["Region"] == 5

    data_points = get_data_points(skip_lambda)
    data_lists = data_points_to_lists(data_points)
    zfs_list = data_lists["ZFS (GHz)"]
    zfs_err_list = data_lists["ZFS (GHz) error"]
    zfs_devs = [el * 1000 - 2870 for el in zfs_list]
    split_list = data_lists["Splitting (MHz)"]
    split_err_list = data_lists["Splitting (MHz) error"]
    region_list = data_lists["Region"]
    kpl_colors_list = list(KplColors)
    region_colors = {}
    for ind in range(5):
        region_colors[ind + 1] = kpl_colors_list[ind]
    color_list = [region_colors[el] for el in region_list]

    fig, ax = plt.subplots()

    # Histograms
    # kpl.histogram(ax, zfs_devs, kpl.HistType.STEP)
    # ax.set_xlabel("Deviation from 2.87 GHz (MHz)")
    # ax.set_ylabel("Occurrences")

    # Splittings, zfs correlation
    regions_plotted = []
    for point in data_points:
        region = point["Region"]
        if region not in regions_plotted:
            regions_plotted.append(region)
            label = region
        else:
            label = None
        color = region_colors[region]
        kpl.plot_points(
            ax,
            point["Splitting (MHz)"],
            point["ZFS (GHz)"] * 1000 - 2870,
            xerr=point["Splitting (MHz) error"],
            yerr=point["ZFS (GHz) error"],
            color=color,
            label=label,
        )
    ax.set_xlabel("Splitting (MHz)")
    ax.set_ylabel("Deviation from 2.87 GHz (MHz)")
    ax.legend(title="Region")


if __name__ == "__main__":

    kpl.init_kplotlib()

    main()

    plt.show(block=True)
