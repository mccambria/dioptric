# -*- coding: utf-8 -*-
"""
Replot Gergo's ab initio spectral density spin-phonon couplings

Created on May 23rd, 2022

@author: mccambria
"""

# region Imports

import errno
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import csv
import matplotlib.patches as patches
import matplotlib.lines as mlines
from scipy.optimize import curve_fit
import pandas as pd
import utils.tool_belt as tool_belt
import utils.common as common
from scipy.odr import ODR, Model, RealData
import sys
from pathlib import Path
import math
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.gridspec import GridSpec
from numpy import pi
from temp_dependence_fitting import (
    get_temp,
    sample_column_title,
    skip_column_title,
)
from temp_dependence_fitting import omega_column_title, omega_err_column_title
from temp_dependence_fitting import gamma_column_title, gamma_err_column_title
from utils.tool_belt import presentation_round, presentation_round_latex


# endregion

# region Functions


# endregion

# region Main


def main(path, file_name, override_skips=False):
    """This needs to track main in temp_dependence_fitting!"""

    csv_file_path = path / "{}.csv".format(file_name)

    data_points = []
    nv_names = []
    samples = []
    markers = {}
    header = True

    with open(csv_file_path, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            # Create columns from the header (first row)
            if header:
                columns = row[1:]
                header = False
                continue
            point = {}
            sample = row[0]

            # Skip checks
            # Unpopulated first column means this is a padding row
            if not override_skips:
                if sample == "":
                    continue
                elif sample not in ["Wu", "Hopper"]:
                    continue

            if sample == "Hopper":
                nv_name = sample.lower()
            else:
                nv_name = "{}-{}".format(sample.lower(), row[1])
            point[sample_column_title] = sample
            point["nv_name"] = nv_name
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

            # data_points.append(point)
            if override_skips or not point[skip_column_title]:
                data_points.append(point)

            if nv_name not in nv_names:
                nv_names.append(nv_name)

    # The first shall be last
    # data_points.append(data_points.pop(0))

    paper_sample_mapping = {"Hopper": "A", "Wu": "B"}
    paper_nv_names_mapping = {
        # "hopper": "N/A",
        "hopper": "NVA",
        "wu-nv3_2021_12_03": "NVB1",
        "wu-nv6_2021_12_25": "NVB2",
        "wu-nv1_2022_02_10": "NVB3",
        "wu-nv1_2022_03_16": "NVB4",
        "wu-nv6_2022_04_14": "NVB5",
    }
    paper_nv_name_lambda = lambda point: paper_nv_names_mapping[
        point["nv_name"]
    ]

    # Sort the entries, first by NV name (and so sample), then by temp
    sorted_key_lambda = (
        lambda point: paper_nv_name_lambda(point)[0:3]
        + f"{get_temp(point):05.1f}"
    )
    sorted_data_points = sorted(data_points, key=sorted_key_lambda)

    # for point in sorted_data_points:
    num_data_points = len(sorted_data_points)
    break_ind = int(np.ceil(num_data_points / 2))
    for ind_a in range(break_ind):
        print_str = ""
        for ind_b in [ind_a, ind_a + break_ind]:
            if ind_b >= num_data_points:
                break
            point = sorted_data_points[ind_b]
            sample = point[sample_column_title]
            paper_sample = paper_sample_mapping[sample]
            temp = round(get_temp(point), 1)
            nv_name = point["nv_name"]
            paper_nv_name = paper_nv_name_lambda(point)
            omega = point[omega_column_title]
            omega_err = point[omega_err_column_title]
            formatted_omega = presentation_round_latex(omega, omega_err)
            gamma = point[gamma_column_title]
            gamma_err = point[gamma_err_column_title]
            formatted_gamma = presentation_round_latex(gamma, gamma_err)
            if not formatted_omega.strip() or not formatted_gamma.strip():
                test = "debug"
            print_str += "{} & {} & {} & {} & ".format(
                paper_nv_name,
                temp,
                formatted_omega,
                formatted_gamma,
            )
        print_str = print_str[:-2]
        print_str += "\\\\"
        print(print_str)


# endregion

# region Run the file

if __name__ == "__main__":

    home = common.get_nvdata_dir()
    path = home / "paper_materials/relaxation_temp_dependence"
    file_name = "compiled_data"

    main(path, file_name)

# endregion
