# -*- coding: utf-8 -*-
"""
Get the NV temp based on the ZFS, using numbers from: 'Temperature dependent 
energy level shifts of nitrogen-vacancy centers in diamond'

Created on Fri Mar  5 12:42:32 2021

@author: matth
"""


# region Import and constants

import numpy as np
from utils import common
from majorroutines.pulsed_resonance import return_res_with_error
import majorroutines.pulsed_resonance as pesr
import utils.tool_belt as tool_belt
from utils.tool_belt import bose
import matplotlib.pyplot as plt
from utils import kplotlib as kpl
from pathos.multiprocessing import ProcessingPool
from utils.kplotlib import KplColors
from scipy.optimize import curve_fit
from scipy.optimize import minimize
import csv
import pandas as pd
import sys
from analysis import three_level_rabi
import figures.zfs_vs_t.thermal_expansion as thermal_expansion
import csv
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib as mpl
import matplotlib.legend_handler


# Adjust for my poor digitization
# toyli_zfss = np.array(toyli_zfss)
# toyli_zfss -= 2.87
# toyli_zfss *= 0.9857
# toyli_zfss += 2.8701

# ZFS at 0 temp, used for papers that only report shifts
# Matches our fits 0 temp value
base_zfs = 2.877380

nvdata_dir = common.get_nvdata_dir()
compiled_data_file_name = "zfs_vs_t"
compiled_data_path = nvdata_dir / "paper_materials/zfs_temp_dep"


# endregion
# region Functions


def calibrate_digitization(file_name, fit_func):
    temps, zfss = get_prior_work_data(file_name)
    calc_zfss = [fit_func(el) for el in temps]
    num_points = len(temps)

    def adj_zfss_cost(x):
        offset, factor = x
        first_zfs = zfss[0]
        adj_zfss = [(el - first_zfs) * factor + first_zfs + offset for el in zfss]
        cost = 0
        for ind in range(num_points):
            cost += (adj_zfss[ind] - calc_zfss[ind]) ** 2
        return cost

    res = minimize(adj_zfss_cost, (0, 0))
    return res.x


def get_prior_work_data(file_name):
    """Get a list of temps and zfs values from a digitized version of a figure from a prior work"""

    file_path = compiled_data_path / "figures/prior_work_figs" / f"{file_name}.csv"
    temps = []
    zfss = []
    with open(file_path, "r", newline="", encoding="utf-8") as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            if row[0].startswith("header"):
                continue
            temps.append(round(float(row[0]), 3))
            zfss.append(round(float(row[1]), 6))

    ### Adjustments
    if file_name.startswith("doherty"):
        zfss = [base_zfs + el / 1000 for el in zfss]
    elif file_name.startswith("toyli"):
        temps = [round(el, -1) for el in temps]
        first_zfs = zfss[0]
        offset = 1.44959598e-04
        factor = 9.87806786e-01
        zfss = [(el - first_zfs) * factor + first_zfs + offset for el in zfss]
    elif file_name.startswith("chen"):
        temps = [round(el) for el in temps]
        first_zfs = zfss[0]
        offset = 1.22295695e-09
        factor = 1.00000076e00
        zfss = [(el - first_zfs) * factor + first_zfs + offset for el in zfss]
    elif file_name == "li_2017_1b":
        temps = [round(el) for el in temps]
        first_zfs = zfss[0]
        offset = -5.60176242e-07
        factor = 9.90228416e-01
        zfss = [(el - first_zfs) * factor + first_zfs + offset for el in zfss]
    return temps, zfss


def get_data_points(skip_lambda=None, condense_all=False, condense_samples=False):
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
            point["Label"] = f"{point['Sample']}-{point['NV']}"

            nv_ind = point["NV"]
            nv_ind = nv_ind[2:]
            nv_ind = nv_ind.split("_")
            nv_ind = int(nv_ind[0])
            group = "cold" if nv_ind <= 5 else "hot"
            point["Group"] = group

            skip = skip_lambda is not None and skip_lambda(point)
            if not skip:
                data_points.append(point)

    # MCC ad hoc adjustments
    # for point in data_points:
    #     if point["Sample"] == "Wu" and point["Monitor"] == "PT100":
    #         point["ZFS (GHz)"] -= 0.000500
    #     elif point["Sample"] == "15micro" and point["Monitor"] == "PT100":
    #         point["ZFS (GHz)"] += 0.0006
    #         # point["Monitor temp (K)"] += 8

    if condense_all or condense_samples:
        data_points = condense_data_points(data_points, condense_all, condense_samples)

    return data_points


def condense_data_points(data_points, condense_all=False, condense_samples=False):
    """
    Turn the full data points list into a processed version where there is just the
    information necessary for analysis and plotting.
    If condense_all, combine all the data at one temp into one point
    regardless of which sample or NV it came from.
    If condense_samples, combine the data from different NVs within the same
    sample into one point. Each sample at a given temp will have its own point.
    """

    if condense_all or condense_samples:
        id_lambda = lambda point: f"{point['Setpoint temp (K)']}-{point['Sample']}"
    else:
        id_lambda = (
            lambda point: f"{point['Setpoint temp (K)']}-{point['Sample']}-{point['NV']}"
        )

    condensed_data_points = []
    identifier_set = [id_lambda(point) for point in data_points]
    identifier_set = list(set(identifier_set))
    identifier_set.sort()
    for identifier in identifier_set:
        id_split = identifier.split("-")
        setpoint_temp_str = id_split[0]
        if setpoint_temp_str == "":
            setpoint_temp = "room"
        else:
            setpoint_temp = int(float(setpoint_temp_str))
        monitor_temps = []
        zfss = []
        zfs_errors = []
        for point in data_points:
            test_identifier = id_lambda(point)
            if test_identifier == identifier:
                monitor_temps.append(point["Monitor temp (K)"])
                zfss.append(point["ZFS (GHz)"])
                zfs_errors.append(point["ZFS (GHz) error"])
                group = point["Group"]  # Assumes all condensed points share a group
        weights = [val**-2 for val in zfs_errors]
        norm = np.sum(weights)
        # For inverse-variance weighting, condensed_error**2 = 1/norm
        condensed_error = 0
        for ind in range(len(zfs_errors)):
            weight = weights[ind]
            err = zfs_errors[ind]
            condensed_error += (weight * err / norm) ** 2
        condensed_error = np.sqrt(condensed_error)
        if condense_all:
            label = "This work"
        elif condense_samples:
            sample = id_split[1]
            label = sample
        else:
            sample = id_split[1]
            nv = id_split[2]
            label = f"{sample}-{nv}"
        new_point = {
            "Setpoint temp (K)": setpoint_temp,
            "Monitor temp (K)": np.average(monitor_temps),
            "ZFS (GHz)": np.average(zfss, weights=weights),
            "ZFS (GHz) error": condensed_error,
            "Label": label,
            "Group": group,
        }
        condensed_data_points.append(new_point)
    return condensed_data_points


def data_points_to_lists(data_points):
    """Turn a dict of data points into a list that's more convenenient for plotting"""

    zfs_list = []
    zfs_err_list = []
    temp_list = []
    label_list = []
    color_list = []
    group_list = []  # "cold" or "hot"
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
        zfs_err = el["ZFS (GHz) error"]
        zfs_err_list.append(zfs_err)
        label = el["Label"]
        label_list.append(label)
        if label not in used_labels:
            used_labels.append(label)
            color_dict[label] = data_color_options.pop(0)
        color_list.append(color_dict[label])
        group_list.append(el["Group"])

    return zfs_list, zfs_err_list, temp_list, label_list, color_list, group_list


def calc_zfs_from_compiled_data():
    def skip_lambda(point):
        return point["Sample"] != "Wu"

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


def light_polarization():
    # Actual angles of the half-waveplate
    angles = [348, 88, 58, 38, 38, 18, 338, 358, 248, 348, 58]
    # angles = [angles[ind] + 0.1 * ind for ind in range(len(angles))]
    # angles = angles[1:]
    # print(angles)
    # Actual polarization angles
    angles = [np.mod(2 * (el - angles[0]), 180) for el in angles]
    # print(angles)
    zfss = [
        [2.86957, 2.869903, 2.869913, 2.869949, 2.869767],
        [2.869699, 2.869654, 2.869838, 2.869908, 2.869865],
        [2.869846, 2.869777, 2.86989, 2.869863, 2.869619],
        [2.869825, 2.869832, 2.869823, 2.869929, 2.869966],
        [2.869906, 2.869828, 2.870061, 2.869784, 2.869865],  # 38 repeat
        [2.869836, 2.869849, 2.86997, 2.869855, 2.869926],
        [2.869817, 2.869826, 2.869789, 2.869893, 2.870104],
        [2.869873, 2.869767, 2.869866, 2.869859, 2.869953],
        [2.869849, 2.86972, 2.869838, 2.869856, 2.86981],
        [2.869741, 2.869895, 2.869624, 2.8698, 2.869992],
        [2.869835, 2.869805, 2.86986, 2.869986, 2.869934],
    ]
    zfss = np.array(zfss)
    zfss *= 1000
    zfss -= 2870
    errs = [
        [0.000124, 9e-05, 0.000132, 0.000113, 0.000108],
        [0.000114, 0.000103, 0.000133, 0.000122, 0.000107],
        [0.000112, 8.7e-05, 0.000128, 0.000107, 0.000112],
        [0.00011, 8.9e-05, 0.000129, 0.000119, 0.000104],
        [7.9e-05, 6.2e-05, 8.6e-05, 7.9e-05, 7.6e-05],  # 38 repeat
        [0.000111, 9.3e-05, 0.000114, 0.000117, 0.000108],
        [0.000113, 8.6e-05, 0.000112, 0.000112, 0.000107],
        [8.3e-05, 6.5e-05, 7.9e-05, 8.3e-05, 7.4e-05],
        [0.000117, 8.6e-05, 0.000107, 0.000118, 0.0001],
        [0.000119, 8.9e-05, 0.000107, 0.000119, 0.000103],
        [0.000115, 8.6e-05, 0.000107, 0.000112, 9.8e-05],
    ]
    errs = np.array(errs)
    errs *= 1000
    labels = ["NV12", "NV13", "NV14", "NV15", "NV16"]

    # Combine data points at same polarization angle
    do_combine = True
    if do_combine:
        condensed_angles = []
        condensed_zfss = []
        condensed_errs = []
        for ind1 in range(len(angles)):
            angle1 = angles[ind1]
            if angle1 in condensed_angles:
                continue
            sub_zfss = []
            sub_errs = []
            for ind2 in range(len(angles)):
                angle2 = angles[ind2]
                if angle1 != angle2:
                    continue
                sub_zfss.append(zfss[ind2])
                sub_errs.append(errs[ind2])
            sub_zfss = np.array(sub_zfss)
            sub_errs = np.array(sub_errs)
            condensed_sub_zfss = []
            condensed_sub_errs = []
            for nv_ind in range(5):
                weights = sub_errs[:, nv_ind] ** -2
                condensed_sub_zfss.append(
                    np.average(sub_zfss[:, nv_ind], weights=weights)
                )
                condensed_sub_errs.append(np.sqrt(1 / np.sum(weights)))
            condensed_angles.append(angle1)
            condensed_zfss.append(condensed_sub_zfss)
            condensed_errs.append(condensed_sub_errs)
        angles = np.array(condensed_angles)
        zfss = np.array(condensed_zfss)
        errs = np.array(condensed_errs)

    for ind in range(5):
        fig, ax = plt.subplots()
        kpl.plot_points(
            ax,
            angles,
            zfss[:, ind],
            yerr=errs[:, ind],
            label=labels[ind],
            color=kpl.data_color_cycler[ind],
        )
        # ax.set_xlabel("Waveplate angle (deg)")
        ax.set_xlabel("Polarization angle (deg)")
        ax.set_ylabel("ZFS - 2870 (MHz)")
        ax.legend()


def refit_experiments():
    """Re-run fits to experimental data, either plotting and saving the new plots
    or just printing out the fit parameters
    """

    ### User setup
    # Also see below section Sample-dependent fit...

    do_plot = True  # Generate raw data and fit plots?
    do_save = True  # Save the plots?
    do_print = True  # Print out popts and associated error bars?

    skip_lambda = (
        lambda point: point["Skip"]
        # or point["ZFS file"] == ""
        # or point["Sample"] != "15micro"
        or point["Sample"] != "Wu"
        # or point["Setpoint temp (K)"] != ""
        # or point["Setpoint temp (K)"] < 300
    )

    data_points = get_data_points(skip_lambda)
    # file_list = [el["ZFS file"] for el in data_points]
    # data_points = data_points[2:3]
    file_list = []
    guess_param_list = []
    for el in data_points:
        file_list.append(el["ZFS file"])
        guess_params = (
            el["Contrast"],
            el["Width (MHz)"],
            el["ZFS (GHz)"],
            el["Splitting (MHz)"],
        )
        guess_param_list.append(guess_params)
    # print(file_list)
    # file_list = [
    #     # # 1 us
    #     # "2023_03_03-17_23_24-15micro-nv6_zfs_vs_t",
    #     # "2023_03_03-16_55_36-15micro-nv7_zfs_vs_t",
    #     # "2023_03_03-16_28_28-15micro-nv8_zfs_vs_t",
    #     # "2023_03_03-16_00_44-15micro-nv9_zfs_vs_t",
    #     # "2023_03_03-15_32_43-15micro-nv11_zfs_vs_t",
    #     # # 10 us
    #     # "2023_03_03-18_46_02-15micro-nv6_zfs_vs_t",
    #     # "2023_03_03-18_18_54-15micro-nv7_zfs_vs_t",
    #     # "2023_03_03-20_07_05-15micro-nv8_zfs_vs_t",
    #     # "2023_03_03-19_40_03-15micro-nv9_zfs_vs_t",
    #     # "2023_03_03-19_13_03-15micro-nv11_zfs_vs_t",
    #     # # 100 us
    #     # "2023_03_03-21_03_25-15micro-nv6_zfs_vs_t",
    #     # "2023_03_03-22_57_55-15micro-nv7_zfs_vs_t",
    #     # "2023_03_03-22_29_43-15micro-nv8_zfs_vs_t",
    #     # "2023_03_03-21_32_20-15micro-nv9_zfs_vs_t",
    #     # "2023_03_03-22_00_57-15micro-nv11_zfs_vs_t",
    #     # 1 ms
    #     # "2023_03_04-11_43_50-15micro-nv6_zfs_vs_t",
    #     # "2023_03_04-11_06_24-15micro-nv7_zfs_vs_t",
    #     # "2023_03_04-12_58_51-15micro-nv8_zfs_vs_t",
    #     # "2023_03_04-12_21_20-15micro-nv9_zfs_vs_t",
    #     # "2023_03_04-13_36_17-15micro-nv11_zfs_vs_t",
    #     # 1 us, ND 0.3 => 0.5
    #     # "2023_03_04-16_40_09-15micro-nv6_zfs_vs_t",
    #     # "2023_03_04-14_55_01-15micro-nv7_zfs_vs_t",
    #     # "2023_03_04-15_47_39-15micro-nv8_zfs_vs_t",
    #     # "2023_03_04-18_25_23-15micro-nv9_zfs_vs_t",
    #     # "2023_03_04-17_32_26-15micro-nv11_zfs_vs_t",
    #     # # microwave 10 => 0 dBm
    #     # "2023_03_04-21_22_00-15micro-nv6_zfs_vs_t",
    #     # "2023_03_04-23_12_14-15micro-nv7_zfs_vs_t",
    #     # "2023_03_04-20_26_41-15micro-nv8_zfs_vs_t",
    #     # "2023_03_04-22_17_57-15micro-nv9_zfs_vs_t",
    #     # "2023_03_05-00_07_19-15micro-nv11_zfs_vs_t",
    #     # # ND 1.0
    #     # "2023_03_05-13_24_15-15micro-nv6_zfs_vs_t",
    #     # "2023_03_05-11_41_45-15micro-nv7_zfs_vs_t",
    #     # "2023_03_05-14_15_18-15micro-nv8_zfs_vs_t",
    #     # "2023_03_05-12_33_07-15micro-nv9_zfs_vs_t",
    #     # "2023_03_05-10_50_58-15micro-nv11_zfs_vs_t",
    #     # # Temp control disconnected
    #     # "2023_03_06-20_37_53-15micro-nv6_zfs_vs_t",
    #     # "2023_03_06-20_09_53-15micro-nv7_zfs_vs_t",
    #     # "2023_03_06-19_14_17-15micro-nv8_zfs_vs_t",
    #     # "2023_03_06-19_42_31-15micro-nv9_zfs_vs_t",
    #     # "2023_03_06-21_05_38-15micro-nv11_zfs_vs_t",
    #     # 1 ms delay repeat
    #     # "2023_03_07-05_26_17-15micro-nv6_zfs_vs_t",
    #     # "2023_03_07-04_11_05-15micro-nv7_zfs_vs_t",
    #     # "2023_03_07-02_56_13-15micro-nv8_zfs_vs_t",
    #     # "2023_03_07-00_26_02-15micro-nv9_zfs_vs_t",
    #     # "2023_03_07-01_41_04-15micro-nv11_zfs_vs_t",
    #     # uwave polarization
    #     # "2023_03_07-14_27_00-15micro-nv6_zfs_vs_t",
    #     # "2023_03_07-12_34_04-15micro-nv7_zfs_vs_t",
    #     # "2023_03_07-13_30_18-15micro-nv8_zfs_vs_t",
    #     # "2023_03_07-13_58_48-15micro-nv9_zfs_vs_t",
    #     # "2023_03_07-13_02_08-15micro-nv11_zfs_vs_t",
    #     # New NVs 1
    #     # "2023_03_09-13_06_10-15micro-nv6_offset",
    #     # "2023_03_09-12_14_14-15micro-nv7_offset",
    #     # "2023_03_09-12_40_16-15micro-nv8_offset",
    #     # New NVs 2, 348 degrees
    #     # "2023_03_09-16_14_03-15micro-nv12_offset",
    #     # "2023_03_09-15_46_07-15micro-nv13_offset",
    #     # "2023_03_09-15_18_27-15micro-nv14_offset",
    #     # "2023_03_09-14_50_35-15micro-nv15_offset",
    #     # "2023_03_09-14_21_53-15micro-nv16_offset",
    #     # # 88 degrees
    #     # "2023_03_09-23_37_19-15micro-nv12_offset",
    #     # "2023_03_09-18_37_07-15micro-nv13_offset",
    #     # "2023_03_09-23_09_00-15micro-nv14_offset",
    #     # "2023_03_09-17_41_03-15micro-nv15_offset",
    #     # "2023_03_09-18_09_53-15micro-nv16_offset",
    #     # # 58 degrees
    #     # "2023_03_10-13_45_46-15micro-nv12_offset",
    #     # "2023_03_10-14_13_33-15micro-nv13_offset",
    #     # "2023_03_10-15_39_17-15micro-nv14_offset",
    #     # "2023_03_10-15_11_35-15micro-nv15_offset",
    #     # "2023_03_10-14_42_16-15micro-nv16_offset",
    #     # # 38 degrees
    #     # "2023_03_10-17_10_38-15micro-nv12_offset",
    #     # "2023_03_10-19_04_01-15micro-nv13_offset",
    #     # "2023_03_10-17_38_32-15micro-nv14_offset",
    #     # "2023_03_10-18_36_06-15micro-nv15_offset",
    #     # "2023_03_10-18_07_29-15micro-nv16_offset",
    #     # 38 degrees, finer average
    #     # "2023_03_10-23_44_44-15micro-nv12_offset",
    #     # "2023_03_11-01_37_36-15micro-nv13_offset",
    #     # "2023_03_11-03_31_26-15micro-nv14_offset",
    #     # "2023_03_11-02_35_44-15micro-nv15_offset",
    #     # "2023_03_11-00_42_21-15micro-nv16_offset",
    #     # # 18 degrees
    #     # "2023_03_12-13_11_31-15micro-nv12_offset",
    #     # "2023_03_12-13_39_16-15micro-nv13_offset",
    #     # "2023_03_12-14_07_58-15micro-nv14_offset",
    #     # "2023_03_12-14_36_30-15micro-nv15_offset",
    #     # "2023_03_12-12_43_05-15micro-nv16_offset",
    #     # # 338 degrees
    #     # "2023_03_12-18_32_33-15micro-nv12_offset",
    #     # "2023_03_12-19_58_21-15micro-nv13_offset",
    #     # "2023_03_12-20_27_13-15micro-nv14_offset",
    #     # "2023_03_12-19_30_29-15micro-nv15_offset",
    #     # "2023_03_12-19_01_42-15micro-nv16_offset",
    #     # # 358 degrees
    #     # "2023_03_13-00_58_53-15micro-nv12_offset",
    #     # "2023_03_13-02_51_05-15micro-nv13_offset",
    #     # "2023_03_13-03_48_32-15micro-nv14_offset",
    #     # "2023_03_13-01_55_57-15micro-nv15_offset",
    #     # "2023_03_13-00_02_24-15micro-nv16_offset",
    #     # 248 degrees
    #     # "2023_03_13-13_49_30-15micro-nv12_offset",
    #     # "2023_03_13-12_22_52-15micro-nv13_offset",
    #     # "2023_03_13-14_18_30-15micro-nv14_offset",
    #     # "2023_03_13-12_51_39-15micro-nv15_offset",
    #     # "2023_03_13-13_21_04-15micro-nv16_offset",
    #     # 348 degrees
    #     # "2023_03_13-15_33_36-15micro-nv12_offset",
    #     # "2023_03_13-15_05_02-15micro-nv13_offset",
    #     # "2023_03_13-17_00_38-15micro-nv14_offset",
    #     # "2023_03_13-16_02_24-15micro-nv15_offset",
    #     # "2023_03_13-16_31_38-15micro-nv16_offset",
    #     # 58 degrees
    #     "2023_03_13-19_38_01-15micro-nv12_offset",
    #     "2023_03_13-19_09_20-15micro-nv13_offset",
    #     "2023_03_13-18_41_27-15micro-nv14_offset",
    #     "2023_03_13-18_12_28-15micro-nv15_offset",
    #     "2023_03_13-17_43_29-15micro-nv16_offset",
    # ]

    ### Parallel process

    # guess_params = [0.24, 2, 2.853, 4.5]
    # guess_param_list = [0.24, 2, 2.853, 4.5]
    # refit_sub_lambda = lambda f: refit_experiments_sub(
    #     f, do_plot, do_save, guess_params
    # )
    refit_sub_lambda = lambda f, g: refit_experiments_sub(f, g, do_plot, do_save)
    with ProcessingPool() as p:
        results = p.map(refit_sub_lambda, file_list, guess_param_list)
    # results = [refit_sub_lambda(f, g) for f, g in zip(file_list, guess_param_list)]

    # f = "2023_01_14-13_52_57-wu-nv10_zfs_vs_t"
    # popt, pste, red_chi_sq = refit_sub_lambda(f)
    # xl_str = ""
    # for ind in range(len(popt)):
    #     xl_str += f"{round(popt[ind], 6)}, {round(pste[ind], 6)}, "
    # xl_str += str(red_chi_sq)
    # print(xl_str)
    # return

    ### Parse results

    table_popt = []
    table_pste = []
    table_red_chi_sq = []
    num_cols = 5
    for ind in range(num_cols):
        table_popt.append([])
        table_pste.append([])
    for result in results:
        popt, pste, red_chi_sq = result
        for ind in range(num_cols):
            if ind < len(popt):
                val = round(popt[ind], 6)
                err = round(pste[ind], 6)
            else:
                val = None
                err = None
            val_col = table_popt[ind]
            err_col = table_pste[ind]
            val_col.append(val)
            err_col.append(err)
        table_red_chi_sq.append(round(red_chi_sq, 6))

    ### Report the fit parameters

    if do_print:
        print("Reduced chi squared:")
        print(table_red_chi_sq)
        print(np.mean(table_red_chi_sq))
        print()
        print("Fit parameters:")
        for ind in range(len(table_popt)):
            print()
            print(table_popt[ind])
            print()
            print(table_pste[ind])
            print()

    # freq1_errs = np.array(table_pste[2])
    # freq2_errs = np.array(table_pste[3])
    # zfs_errs = np.sqrt(freq1_errs**2 + freq2_errs**2) / 2

    # zfs_vals = np.array(table_popt[2])
    # zfs_errs = np.array(table_pste[2])

    # print()
    # print(np.mean(table_red_chi_sq))
    # print(np.min(table_red_chi_sq))
    # print(np.max(table_red_chi_sq))
    # print()
    # print(zfs_vals)
    # print()
    # print(zfs_errs)

    # print("ZFS vals")
    # for ind in range(len(zfs_vals)):
    #     # print(tool_belt.presentation_round(zfs_vals[ind], zfs_errs[ind]))
    #     print(zfs_vals[ind], zfs_errs[ind])

    # print("ZFS errors")
    # print(zfs_errs)
    # mean_zfs_err = np.mean(zfs_errs)
    # print(mean_zfs_err)


# def refit_experiments_sub(file_name, do_plot=False, do_save=False, guess_params=None):
def refit_experiments_sub(file_name, guess_params, do_plot=False, do_save=False):
    # print(guess_params)

    data = tool_belt.get_raw_data(file_name)
    raw_file_path = tool_belt.get_raw_data_path(file_name)
    freq_center = data["freq_center"]
    freq_range = data["freq_range"]
    num_steps = data["num_steps"]
    ref_counts = data["ref_counts"]
    sig_counts = data["sig_counts"]
    num_reps = data["num_reps"]
    nv_sig = data["nv_sig"]
    sample = nv_sig["name"].split("-")[0]
    readout = nv_sig["spin_readout_dur"]
    uwave_pulse_dur = data["uwave_pulse_dur"]
    # uwave_pulse_dur = None
    try:
        norm_style = tool_belt.NormStyle[str.upper(nv_sig["norm_style"])]
    except Exception:
        # norm_style = NormStyle.POINT_TO_POINT
        norm_style = tool_belt.NormStyle.SINGLE_VALUED

    ret_vals = tool_belt.process_counts(
        sig_counts, ref_counts, num_reps, readout, norm_style
    )
    (
        sig_counts_avg_kcps,
        ref_counts_avg_kcps,
        norm_avg_sig,
        norm_avg_sig_ste,
    ) = ret_vals

    ### Raw data figure

    # if do_plot:
    #     ret_vals = pesr.create_raw_data_figure(
    #         freq_center,
    #         freq_range,
    #         num_steps,
    #         sig_counts_avg_kcps,
    #         ref_counts_avg_kcps,
    #         norm_avg_sig,
    #     )
    #     if do_save:
    #         raw_fig = ret_vals[0]
    #         file_path = raw_file_path.with_suffix(".svg")
    #         tool_belt.save_figure(raw_fig, file_path)

    ### Sample-dependent fit functions and parameters

    if sample == "wu":
        # line_func = (
        #     lambda freq, contrast, rabi_freq, center: pesr.rabi_line_n14_hyperfine(
        #         freq, contrast, rabi_freq, center, uwave_pulse_dur=uwave_pulse_dur
        #     )
        # )
        # # line_func = lambda freq, contrast, rabi_freq, center: pesr.rabi_line(
        # #     freq, contrast, rabi_freq, center, uwave_pulse_dur=uwave_pulse_dur
        # # )
        # guess_params = [0.2, 4, freq_center]
        # guess_params = [0.3, 500 / uwave_pulse_dur, freq_center]
        # guess_params = [0.4, 9, 2.8748]

        line_func = lambda freq, contrast, rabi_freq, center, splitting: three_level_rabi.coherent_line(
            freq, contrast, rabi_freq, center, splitting, uwave_pulse_dur
        )
        # line_func = pesr.lorentzian_split
        # guess_params = [1.5 * np.max(1 - norm_avg_sig), 2, freq_center, 3]
        if guess_params is None:
            guess_params = [0.201135, 4.537905, 2.877451, 3.456755]

        # line_func = pesr.lorentzian_split
        # guess_params = [0.3, 1, freq_center, 1]

        # fit_func = lambda freq, contrast, rabi_freq, center: pesr.dip_sum(
        #     freq, line_func, contrast, rabi_freq, center
        # )
        # popt = guess_params

    elif sample == "15micro":
        # fmt: off
        
        # line_func = lambda freq, contrast, rabi_freq, center, splitting, offset: three_level_rabi.incoherent_line(freq, contrast, rabi_freq, center, splitting, offset, uwave_pulse_dur)
        # guess_params = [0.05, 3, freq_center, 6, 0.005]

        # line_func = pesr.lorentzian_split
        # guess_params = [0.05, 3, freq_center, 6]

        line_func = pesr.lorentzian_split_offset
        if guess_params is None:
            guess_params = [0.05, 3, freq_center, 6, -0.001]

        # line_func = lambda freq, contrast, hwhm, splitting, offset: pesr.lorentzian_split_offset(freq, contrast, hwhm, 2.87, splitting, offset)
        # guess_params = [0.05, 3, 6, 0.005]

        # line_func = pesr.lorentzian_test
        # guess_params = [0.05, 3, freq_center, 6, 0.005, 0.05, 3]

        # fmt: on

    ### Raw data figure

    # try:

    if do_plot:
        fit_fig, _, fit_func, popt, pcov = pesr.create_fit_figure(
            freq_center,
            freq_range,
            num_steps,
            norm_avg_sig,
            norm_avg_sig_ste,
            line_func=line_func,
            guess_params=guess_params,
            # fit_func=fit_func,
            # popt=popt,
        )
        if do_save:
            file_path = raw_file_path.with_name((f"{file_name}-fit"))
            file_path = file_path.with_suffix(".svg")
            tool_belt.save_figure(fit_fig, file_path)

    ### Get fit parameters and error bars
    if not do_plot:
        fit_func, popt, pcov = pesr.fit_resonance(
            freq_center,
            freq_range,
            num_steps,
            norm_avg_sig,
            norm_avg_sig_ste,
            line_func=line_func,
            guess_params=guess_params,
        )

    pste = np.sqrt(np.diag(pcov))

    fit_lambda = lambda freq: fit_func(freq, *popt)
    freqs = pesr.calculate_freqs(freq_center, freq_range, num_steps)
    chi_sq = np.sum(((fit_lambda(freqs) - norm_avg_sig) / norm_avg_sig_ste) ** 2)
    red_chi_sq = chi_sq / (len(norm_avg_sig) - len(popt))

    # except Exception as exc:
    #     num_params = len(guess_params)
    #     popt = np.zeros(num_params)
    #     pste = np.zeros(num_params)
    #     red_chi_sq = 10

    return (popt, pste, red_chi_sq)
    # return "test"

    # Close the plots so they don't clutter everything up
    # plt.close("all")


# endregion
# region Fitting functions


def sub_room_zfs_from_temp(temp):
    coeffs = [2.87771, -4.625e-6, 1.067e-7, -9.325e-10, 1.739e-12, -1.838e-15]
    ret_val = 0
    for ind in range(6):
        ret_val += coeffs[ind] * (temp**ind)
    return ret_val


def sub_room_zfs_from_temp_free(
    temp,
    coeff_1,
    coeff_2,
    coeff_3,
    coeff_4,
    coeff_5,
    coeff_6,
    # temp,
    # coeff_0,
    # # coeff_1,
    # coeff_2,
    # # coeff_3,
    # coeff_4,
    # # coeff_5,
    # coeff_6,
    # skip_derivatives_check=False,
):
    coeffs = [coeff_1, coeff_2, coeff_3, coeff_4, coeff_5, coeff_6]
    # coeffs = [coeff_0, coeff_2, coeff_4, coeff_6]

    # Calculate the zfs and its first and second derivatives
    ret_val = 0
    # Only consider this a valid trial fit function if it has negative first and second derivatives everywhere
    # if not skip_derivatives_check:
    #     num_test_points = 1000
    #     max_test_temp = 300
    #     test_temps = np.linspace(1, max_test_temp, num_test_points)
    #     first_der = 0
    #     second_der = 0
    for ind in range(len(coeffs)):
        # zfs
        exp = ind
        # exp = ind * 2
        ret_val += coeffs[ind] * (temp**exp)

        # if not skip_derivatives_check:
        #     # First derivative
        #     if ind in [0]:
        #         continue
        #     exp = ind - 1
        #     first_der += ind * coeffs[ind] * (test_temps ** exp)

        #     # Second derivative
        #     if ind in [0, 1]:
        #     # if ind in [0]:
        #         continue
        #     exp = ind - 2
        #     second_der += ind * (ind - 1) * coeffs[ind] * (test_temps ** exp)

    # Only consider this a valid trial fit function if it has negative first and second derivatives everywhere
    # if not skip_derivatives_check:
    #     if np.any(first_der > 0) or np.any(second_der > 0):
    #         if type(temp) in [list, np.ndarray]:
    #             ret_val = np.array([0] * len(temp))
    #         else:
    #             ret_val = 0

    return ret_val


def super_room_zfs_from_temp(temp):
    coeffs = [2.8697, 9.7e-5, -3.7e-7, 1.7e-10]
    coeff_errs = [0.0009, 0.6e-5, 0.1e-7, 0.1e-10]
    ret_val = 0
    for ind in range(4):
        ret_val += coeffs[ind] * (temp**ind)
    return ret_val


def zfs_from_temp(temp):
    """
    This is a combination of 2 results. For temp < 300 K, we pull the
    5th order polynomial from 'Temperature dependent energy level shifts
    of nitrogen-vacancy centers in diamond.' Then we stitch that to
    'Measurement and Control of Single Nitrogen-Vacancy Center Spins above
    600 K' above 300 K
    """
    # Branch depending on if temp is single- or multi-valued
    if type(temp) in [list, np.ndarray]:
        ret_vals = []
        for val in temp:
            if val < 300:
                zfs = sub_room_zfs_from_temp(val)
            else:
                zfs = super_room_zfs_from_temp(val)
            ret_vals.append(zfs)
        ret_vals = np.array(ret_vals)
        return ret_vals
    else:
        if temp < 300:
            return sub_room_zfs_from_temp(temp)
        else:
            return super_room_zfs_from_temp(temp)


def zfs_from_temp_barson(temp):
    """
    Comes from Barson 2019!
    """

    zfs0 = base_zfs  # GHz
    # zfs0 = 2.884624012121079  # GHz, lowest temp (6 K) value from digitized Fig. 2a
    X1 = 0.4369e-7  # 1 / K
    X2 = 15.7867e-7  # 1 / K
    X3 = 42.5598e-7  # 1 / K
    Theta1 = 200  # K
    Theta2 = 880  # K
    Theta3 = 2137.5  # K

    return zfs_from_temp_barson_free(temp, zfs0, X1, X2, X3, Theta1, Theta2, Theta3)


def zfs_from_temp_li(temp):
    """
    Li 2017, table I for ensemble
    """

    # Ensemble
    zfs0 = 2.87769  # GHz
    A = 5.6e-7  # GHz / K**2
    B = 490  # K

    # NV2
    # zfs0 = 2.87882  # GHz
    # A = 1.4e-7  # GHz / K**2
    # B = 85  # K

    zfs = zfs0 - A * temp**4 / ((temp + B) ** 2)

    return zfs


def fractional_thermal_expansion(temp):
    X1 = 0.4369e-7  # 1 / K
    X2 = 15.7867e-7  # 1 / K
    X3 = 42.5598e-7  # 1 / K
    Theta1 = 200  # K
    Theta2 = 880  # K
    Theta3 = 2137.5  # K

    return fractional_thermal_expansion_free(temp, X1, X2, X3, Theta1, Theta2, Theta3)


def fractional_thermal_expansion_free(temp, X1, X2, X3, Theta1, Theta2, Theta3):
    dV_over_V_partial = lambda X, Theta, T: (X * Theta) / (np.exp(Theta / T) - 1)
    dV_over_V = (
        lambda T: np.exp(
            3
            * (
                dV_over_V_partial(X1, Theta1, T)
                + dV_over_V_partial(X2, Theta2, T)
                + dV_over_V_partial(X3, Theta3, T)
            )
        )
        - 1
    )

    return dV_over_V(temp)


def zfs_from_temp_barson_free(temp, zfs0, X1, X2, X3, Theta1, Theta2, Theta3):
    dV_over_V = lambda temp: fractional_thermal_expansion_free(
        temp, X1, X2, X3, Theta1, Theta2, Theta3
    )

    A = 14.6  # MHz /GPa
    B = 442  # GPa/strain
    b4 = -1.44e-9
    b5 = 3.1e-12
    b6 = -1.8e-15
    D_of_T = (
        lambda T: zfs0
        + (-(A * B * dV_over_V(T)) + (b4 * T**4 + b5 * T**5 + b6 * T**6)) / 1000
    )
    # D_of_T = lambda T: -D_of_T_sub(1) + D_of_T_sub(T)
    if type(temp) in [list, np.ndarray]:
        ret_vals = []
        for val in temp:
            ret_vals.append(D_of_T(val))
        ret_vals = np.array(ret_vals)
        return ret_vals
    else:
        return D_of_T(temp)


# def cambria_test(temp, zfs0, A1, A2, Theta1, Theta2, A3):
# def cambria_test(temp, zfs0, A1, A2, Theta1, Theta2):
def cambria_test(temp, zfs0, A1, A2):
    Theta1 = 65
    Theta2 = 150

    ret_val = zfs0
    for ind in range(2):
        adj_ind = ind + 1
        ret_val += eval(f"A{adj_ind}") * bose(eval(f"Theta{adj_ind}"), temp)

    # A3 = -14.6 * 442 / 1000  # (MHz/GPa) * (GPa/strain)
    # ret_val += A3 * fractional_thermal_expansion(temp)

    return ret_val


def cambria_fixed(temp):
    zfs0, A1, A2 = [2.87781899, -0.08271508, -0.22871962]
    Theta1 = 65
    Theta2 = 150

    ret_val = zfs0
    for ind in range(2):
        adj_ind = ind + 1
        ret_val += eval(f"A{adj_ind}") * bose(eval(f"Theta{adj_ind}"), temp)

    # A3 = -14.6 * 442 / 1000  # (MHz/GPa) * (GPa/strain)
    # ret_val += A3 * fractional_thermal_expansion(temp)

    return ret_val


def cambria_test2(temp, A1, A2, Theta1, Theta2):
    # Fix the ZFS at T=0 to the accepted value
    zfs0 = 2.8777

    # Calculate A2 by fixing to Toyli at 700 K
    # toyli_700 = 2.81461
    # A2 = (toyli_700 - zfs0 - A1 * bose(Theta1, 700)) / bose(Theta2, 700)

    ret_val = zfs0
    for ind in range(2):
        adj_ind = ind + 1
        ret_val += eval(f"A{adj_ind}") * bose(eval(f"Theta{adj_ind}"), temp)

    return ret_val


def cambria_test3(temp, zfs0, A1, A2, Theta1, Theta2):
    ret_val = zfs0
    for ind in range(2):
        adj_ind = ind + 1
        ret_val += eval(f"A{adj_ind}") * bose(eval(f"Theta{adj_ind}"), temp)

    return ret_val


def two_mode_qh(temp, zfs0, A1, A2, Theta1, Theta2):
    ret_val = zfs0
    for ind in range(2):
        adj_ind = ind + 1
        ret_val += eval(f"A{adj_ind}") * bose(eval(f"Theta{adj_ind}"), temp)

    return ret_val


def einstein_heat_capacity(e, T):
    return (((e / T) ** 2) * np.exp(e / T)) / ((np.exp(e / T) - 1) ** 2)


def jacobson(temp, zfs0, coeff):
    """Coefficient of thermal expansion from Jacobson and Stoupin 2019"""
    lattice_constant = thermal_expansion.jacobson_lattice_constant
    # The subtracted term below should really be at T=0 but then we get
    # a divide by 0 in the occupation number calculator. The lattice constant
    # doesn't change really at all between 0 and 10 K so just use 10 K.
    delta_a = lattice_constant(temp) - lattice_constant(10)
    return zfs0 + coeff * delta_a


def cambria_test4(temp, zfs0, A1, Theta1):
    ret_val = zfs0
    for ind in range(1):
        adj_ind = ind + 1
        ret_val += eval(f"A{adj_ind}") * bose(eval(f"Theta{adj_ind}"), temp)

    return ret_val


# endregion
# region Secondary plots


def derivative_comp():
    # Low temp fit
    skip_lambda = lambda point: point["Skip"] or point["Monitor temp (K)"] < 295
    data_points = get_data_points(skip_lambda, condense_all=True)
    zfs_list, zfs_err_list, temp_list, _, _, _ = data_points_to_lists(data_points)
    guess_params = [2.87771, -8e-2, -4e-1, 65, 165]
    fit_func = cambria_test3
    low_popt, _ = curve_fit(
        fit_func,
        temp_list,
        zfs_list,
        sigma=zfs_err_list,
        absolute_sigma=True,
        p0=guess_params,
    )
    low_lambda = lambda temp: fit_func(temp, *low_popt)

    # High temp fit
    skip_lambda = lambda point: point["Skip"] or point["Monitor temp (K)"] >= 295
    data_points = get_data_points(skip_lambda, condense_all=True)
    zfs_list, zfs_err_list, temp_list, _, _, _ = data_points_to_lists(data_points)
    guess_params = [2.87771, -8e-2, -4e-1, 65, 165]
    fit_func = cambria_test3
    high_popt, _ = curve_fit(
        fit_func,
        temp_list,
        zfs_list,
        sigma=zfs_err_list,
        absolute_sigma=True,
        p0=guess_params,
    )
    high_lambda = lambda temp: fit_func(temp, *high_popt)

    fig, ax = plt.subplots()

    temp_linspace = np.linspace(0, 500, 1000)
    # low_der = np.gradient(low_lambda(temp_linspace), temp_linspace)
    low_der = low_lambda(temp_linspace)
    kpl.plot_line(ax, temp_linspace, low_der, label="< 295 K")

    # high_der = np.gradient(high_lambda(temp_linspace), temp_linspace)
    high_der = high_lambda(temp_linspace)
    kpl.plot_line(ax, temp_linspace, high_der, label=">= 295 K")

    ax.legend()


def get_fitted_model(temp_list, zfs_list, zfs_err_list):
    guess_params = [
        2.87771,
        -20,
        -300,
        65,
        165,
    ]
    # fit_func = cambria_test
    fit_func = two_mode_qh
    # fit_func = jacobson
    if None in zfs_err_list:
        zfs_err_list = None
        absolute_sigma = False
    else:
        absolute_sigma = True
    popt, pcov = curve_fit(
        fit_func,
        temp_list,
        zfs_list,
        sigma=zfs_err_list,
        absolute_sigma=absolute_sigma,
        p0=guess_params,
    )
    print(popt)
    print(np.sqrt(np.diag(pcov)))
    # popt[2] = 0
    cambria_lambda = lambda temp: fit_func(
        temp,
        *popt,
        # *guess_params,
    )
    print(f"Predicted ZFS at 296 K: {cambria_lambda(296)}")
    ssr = 0
    num_points = len(temp_list)
    num_params = len(guess_params)
    if zfs_err_list is not None:
        for temp, zfs, zfs_err in zip(temp_list, zfs_list, zfs_err_list):
            calc_zfs = cambria_lambda(temp)
            ssr += ((zfs - calc_zfs) / zfs_err) ** 2
        dof = num_points - num_params
        red_chi_sq = ssr / dof
        print(red_chi_sq)

    return cambria_lambda


# endregion
# region Main plots


def fig_main():
    temp_range = [-10, 510]
    y_range = [2.847, 2.879]
    plot_data = True
    condense_all = False
    condense_samples = True
    plot_prior_models = True
    desaturate_prior = True
    plot_new_model = True

    skip_lambda = lambda point: (
        point["Skip"]
        or point["Sample"] != "Wu"
        # or point["Sample"] != "15micro"
        # or point["ZFS file"] == ""
        # or point["Monitor temp (K)"] >= 296
    )

    min_temp, max_temp = temp_range
    min_temp = 0.1 if min_temp <= 0 else min_temp
    temp_linspace = np.linspace(min_temp, max_temp, 1000)

    # kpl_figsize = kpl.figsize
    # adj_figsize = (kpl_figsize[0], 1.75 * kpl_figsize[1])
    # fig, axes_pack = plt.subplots(2, 1, figsize=adj_figsize)
    fig, ax = plt.subplots()

    data_points = get_data_points(skip_lambda, condense_all, condense_samples)
    (
        zfs_list,
        zfs_err_list,
        temp_list,
        label_list,
        color_list,
        group_list,
    ) = data_points_to_lists(data_points)

    label_set = set(label_list)
    color_dict = {}
    for ind in range(len(zfs_list)):
        color_dict[label_list[ind]] = color_list[ind]

    cambria_lambda = get_fitted_model(temp_list, zfs_list, zfs_err_list)

    ### Plots

    min_temp, max_temp = temp_range
    min_temp = 0.1 if min_temp <= 0 else min_temp
    temp_linspace = np.linspace(min_temp, max_temp, 1000)

    used_data_labels = []
    if plot_data:
        for ind in range(len(zfs_list)):
            temp = temp_list[ind]
            val = zfs_list[ind]
            val_err = zfs_err_list[ind] if (zfs_err_list is not None) else None
            # label = None
            # color = KplColors.DARK_GRAY
            color = color_list[ind]
            label = label_list[ind]
            if label in used_data_labels:
                label = None
            else:
                used_data_labels.append(label)
            kpl.plot_points(
                ax,
                temp,
                val,
                yerr=val_err,
                color=color,
                zorder=-1,
                # zorder=temp - 1000,
                # label=label,
            )
            # print(name, val, temp)
        if len(used_data_labels) > 1:
            ax.legend(loc=kpl.Loc.LOWER_LEFT)

    if plot_new_model:
        color = "#0f49bd"
        kpl.plot_line(
            ax,
            temp_linspace,
            cambria_lambda(temp_linspace),
            label="This work",
            color=color,
            zorder=10,
        )

    ### Prior models

    # prior_models_to_plot = ["Toyli", "Barson"]
    prior_models_to_plot = ["Toyli", "Barson", "Li", "Chen"]
    # prior_models_to_plot = ["Toyli"]
    if plot_prior_models:
        prior_model_colors = [
            KplColors.GREEN,
            KplColors.PURPLE,
            KplColors.RED,
            KplColors.ORANGE,
        ]
        prior_model_colors.reverse()
        prior_model_zorder = 2
        if desaturate_prior:
            prior_model_colors = [
                kpl.lighten_color_hex(el) for el in prior_model_colors
            ]
            prior_model_zorder = -1500
        if "Chen" in prior_models_to_plot:
            kpl.plot_line(
                ax,
                temp_linspace,
                sub_room_zfs_from_temp(temp_linspace),
                label="Chen",
                color=prior_model_colors[0],
                zorder=prior_model_zorder,
            )
        if "Toyli" in prior_models_to_plot:
            kpl.plot_line(
                ax,
                temp_linspace,
                super_room_zfs_from_temp(temp_linspace),
                label="Toyli",
                color=prior_model_colors[1],
                zorder=prior_model_zorder,
            )
        if "Barson" in prior_models_to_plot:
            kpl.plot_line(
                ax,
                temp_linspace,
                zfs_from_temp_barson(temp_linspace),
                label="Barson",
                color=prior_model_colors[2],
                zorder=prior_model_zorder,
            )
        if "Li" in prior_models_to_plot:
            kpl.plot_line(
                ax,
                temp_linspace,
                zfs_from_temp_li(temp_linspace),
                label="Li",
                color=prior_model_colors[3],
                zorder=prior_model_zorder,
            )

    ### Plot wrap up
    if plot_prior_models:
        ax.legend(loc="lower left")
    ax.set_xlabel("Temperature $\mathit{T}$ (K)")
    ax.set_ylabel("Zero-field splitting $\mathit{D}$ (GHz)")
    ax.set_xlim(*temp_range)
    ax.set_ylim(*y_range)


def fig(
    temp_range=[0, 515],
    y_range=[2.847, 2.879],
    plot_data=True,
    condense_all=False,
    condense_samples=True,
    plot_prior_models=False,
    desaturate_prior=True,
    plot_new_model=True,
    plot_prior_data=False,
    inverse_temp=False,
    yscale="linear",
    new_model_diff=False,
    dash_predictions=False,
    inset_comp=False,
    inset_resid=False,
):
    fig, ax = plt.subplots()

    fig_sub(
        ax,
        temp_range,
        y_range,
        plot_data,
        condense_all,
        condense_samples,
        plot_prior_models,
        desaturate_prior,
        plot_new_model,
        plot_prior_data,
        inverse_temp,
        yscale,
        new_model_diff,
        dash_predictions,
    )

    if inverse_temp:
        # tick_locs = [100, 150, 300, 1000]
        tick_locs = [150, 200, 400, 1000]
        ax.set_xticks([1 / val for val in tick_locs])
        ax.set_xticklabels([f"1/{int(val)}" for val in tick_locs])

    if inset_comp:
        axins = inset_axes(
            ax,
            width="100%",
            height="100%",
            bbox_to_anchor=(
                0.1,
                0.11,
                0.52,
                0.49,
            ),
            bbox_transform=ax.transAxes,
            loc=1,
        )
        fig_sub(
            axins,
            # [0, 300],
            # [2.870, 2.878],
            [0, 175],
            [2.876, 2.8781],
            plot_data,
            condense_all,
            condense_samples,
            plot_prior_models,
            desaturate_prior,
            plot_new_model,
            plot_prior_data,
            inverse_temp,
            yscale,
            new_model_diff=False,
            dash_predictions=dash_predictions,
            no_axis_labels=True,
        )
        # axins.set_yticks([2.870, 2.874, 2.878])
        axins.set_yticks([2.876, 2.877, 2.878])
        axins.tick_params(axis="both", which="major", labelsize=16)
        plt.setp(axins.yaxis.get_majorticklabels(), rotation=90, va="center")
        axins.patch.set_alpha(0.7)

    if inset_resid:
        x_pos = 0.57
        y_pos = 0.87
        text = r"\noindent $D(T) = D_{0} + c_{1}n_{1} + c_{2}n_{2}$"
        text += r"\\"
        text += r"$n=\left(\exp(\Delta_{i} / k_{\mathrm{B}}T)-1\right)^{-1}$"
        # ax.text(x_pos, y_pos, text, transform=ax.transAxes, fontsize=15, usetex=True)
        kpl.anchored_text(ax, text, kpl.Loc.UPPER_RIGHT, kpl.Size.SMALL, usetex=True)

        axins = inset_axes(
            ax,
            width="100%",
            height="100%",
            bbox_to_anchor=(
                0.19,
                0.11,
                0.52,
                0.48,
            ),
            bbox_transform=ax.transAxes,
            loc=1,
        )
        fig_sub(
            axins,
            temp_range,
            [-0.45, 0.45],
            # [-450, 450],
            plot_data,
            condense_all,
            condense_samples,
            plot_prior_models,
            desaturate_prior,
            plot_new_model,
            plot_prior_data,
            inverse_temp,
            yscale,
            new_model_diff=True,
            dash_predictions=False,
            no_axis_labels=True,
        )
        axins.tick_params(axis="both", which="major", labelsize=16)
        # plt.setp(axins.yaxis.get_majorticklabels(), rotation=90, va="center")
        # axins.patch.set_alpha(0.7)
        # tick_locs = [150, 200, 400, 1000]
        axins.set_yticks([-0.4, -0.2, 0.0, 0.2, 0.4])
        axins.set_ylabel("Residuals (MHz)")
        # axins.set_ylabel("Residuals (kHz)")


def fig_sub(
    ax,
    temp_range=[-10, 510],
    y_range=[2.847, 2.879],
    plot_data=True,
    condense_all=False,
    condense_samples=True,
    plot_prior_models=False,
    desaturate_prior=True,
    plot_new_model=True,
    plot_prior_data=False,
    inverse_temp=False,
    yscale="linear",
    new_model_diff=False,
    dash_predictions=False,
    no_axis_labels=False,
):
    ### Setup

    skip_lambda = lambda point: (
        point["Skip"]
        or point["Sample"] != "Wu"
        # or point["Sample"] != "15micro"
        # or point["ZFS file"] == ""
        # or point["Monitor temp (K)"] >= 296
    )

    # prior_data_to_plot = ["Toyli", "Barson", "Chen", "Li", "Doherty"]
    prior_data_to_plot = ["Toyli", "Chen", "Li", "Doherty"]
    # prior_data_to_plot = ["Toyli"]

    # prior_models_to_plot = ["Toyli", "Barson"]
    prior_models_to_plot = ["Toyli", "Barson", "Li", "Chen"]
    # prior_models_to_plot = ["Toyli"]
    prior_model_data_ranges = {
        "Toyli": [300, 710],
        "Barson": [0, 710],
        "Li": [0, 295],
        "Chen": [0, 295],
    }

    # prior_models_to_plot = prior_data_to_plot

    ###

    this_work_data_color = KplColors.BLUE
    this_work_model_color = "#0f49bd"
    prior_work_colors = {
        "Chen": KplColors.ORANGE,
        "Toyli": KplColors.RED,
        "Barson": KplColors.PURPLE,
        "Doherty": KplColors.PURPLE,
        "Li": KplColors.GREEN,
    }
    prior_model_fns = {
        "Chen": sub_room_zfs_from_temp,
        "Toyli": super_room_zfs_from_temp,
        "Barson": zfs_from_temp_barson,
        "Li": zfs_from_temp_li,
    }
    prior_data_file_names = {
        "Chen": "chen_2011_3a",
        "Toyli": "toyli_2012_5c",
        "Barson": "barson_2019_2a",
        "Li": "li_2017_1b",  # a is single, b is ensemble
        "Doherty": "doherty_2014_2a",
    }
    prior_data_sets = {}
    for prior_work in prior_data_to_plot:
        file_name = prior_data_file_names[prior_work]
        prior_temps, prior_zfss = get_prior_work_data(file_name)
        prior_data_sets[prior_work] = {"temps": prior_temps, "zfss": prior_zfss}

    min_temp, max_temp = temp_range
    min_temp = 0.1 if min_temp <= 0 else min_temp
    temp_linspace = np.linspace(min_temp, max_temp, 1000)
    plot_temp_linspace = 1 / temp_linspace if inverse_temp else temp_linspace

    # kpl_figsize = kpl.figsize
    # adj_figsize = (kpl_figsize[0], 1.75 * kpl_figsize[1])
    # fig, axes_pack = plt.subplots(2, 1, figsize=adj_figsize)

    data_points = get_data_points(skip_lambda, condense_all, condense_samples)
    (
        zfs_list,
        zfs_err_list,
        temp_list,
        label_list,
        color_list,
        group_list,
    ) = data_points_to_lists(data_points)

    label_set = set(label_list)
    color_dict = {}
    for ind in range(len(zfs_list)):
        color_dict[label_list[ind]] = color_list[ind]

    cambria_lambda = get_fitted_model(temp_list, zfs_list, zfs_err_list)

    # zfs_base = 2.878

    ### Plots

    min_temp, max_temp = temp_range
    min_temp = 0.1 if min_temp <= 0 else min_temp
    temp_linspace = np.linspace(min_temp, max_temp, 1000)
    marker_size = (
        kpl.Size.SMALL if plot_prior_data or new_model_diff else kpl.Size.NORMAL
    )

    used_data_label_keys = []
    if plot_data:
        for ind in range(len(zfs_list)):
            temp = temp_list[ind]
            plot_temp = 1 / temp if inverse_temp else temp
            val = zfs_list[ind]
            if inverse_temp:
                plot_val = val - zfs_base
            elif new_model_diff:
                plot_val = 1e3 * (val - cambria_lambda(temp))
            else:
                plot_val = val
            val_err = zfs_err_list[ind] if (zfs_err_list is not None) else None
            # label = None
            # color = KplColors.DARK_GRAY
            color = color_list[ind]
            if len(label_set) == 1:
                label = "This work"
            else:
                label = label_list[ind]
            group = group_list[ind]
            label_key = f"{label}-{group}"
            if (
                plot_prior_data or len(label_set) > 1
            ) and label_key not in used_data_label_keys:
                used_data_label_keys.append(label_key)
            else:
                label = None
            if plot_prior_data:
                yerr = None
            elif new_model_diff:
                yerr = 1e3 * val_err
            else:
                yerr = val_err
            group = group_list[ind]
            marker = "o" if group == "hot" else "s"
            kpl.plot_points(
                ax,
                plot_temp,
                plot_val,
                marker_size,
                yerr=yerr,
                color=this_work_data_color,
                # zorder=15,
                zorder=temp,
                label=label,
                marker=marker,
            )
            # print(name, val, temp)
        if not no_axis_labels and len(used_data_label_keys) > 1:
            ax.legend(loc=kpl.Loc.LOWER_LEFT)

    if plot_prior_data:
        for prior_data in prior_data_to_plot:
            if prior_data not in prior_data_sets:
                continue
            color = prior_work_colors[prior_data]
            plot_temps = np.array(prior_data_sets[prior_data]["temps"])
            if inverse_temp:
                plot_temps = 1 / plot_temps
            vals = np.array(prior_data_sets[prior_data]["zfss"])
            if inverse_temp:
                plot_vals = vals - zfs_base
            elif new_model_diff:
                plot_vals = vals - cambria_lambda(plot_temps)
            else:
                plot_vals = vals
            kpl.plot_points(
                ax,
                plot_temps,
                plot_vals,
                marker_size,
                color=color,
                # zorder=-5,
                zorder=11,
                label=prior_data,
                marker="D",
            )

    zfs_base = cambria_lambda(1)
    zfs_base_new_model = zfs_base
    if plot_new_model and not new_model_diff:
        zorder = 10 if plot_prior_models else -2
        vals = cambria_lambda(temp_linspace)
        if inverse_temp:
            plot_vals = zfs_base - vals
        else:
            plot_vals = vals
        label = None if plot_data else "This work"
        color = this_work_model_color
        if dash_predictions:
            pmdr = [0, 500]
            if inverse_temp:
                if pmdr[0] == 0:
                    pmdr[0] = 1
                pmdr = [1 / pmdr[1], 1 / pmdr[0]]
            in_range = np.array([pmdr[0] < el < pmdr[1] for el in plot_temp_linspace])
            data_inds = np.nonzero(in_range)
            pred_inds = np.nonzero(np.logical_not(in_range))
            kpl.plot_line(
                ax,
                plot_temp_linspace[data_inds],
                plot_vals[data_inds],
                label=label,
                color=color,
                zorder=zorder,
            )
            light_color = kpl.lighten_color_hex(color)
            kpl.plot_line(
                ax,
                plot_temp_linspace[pred_inds],
                plot_vals[pred_inds],
                # label=label,
                # color=light_color,
                color=color,
                linestyle="dashed",
                zorder=zorder,
            )
        else:
            kpl.plot_line(
                ax,
                plot_temp_linspace,
                plot_vals,
                label=label,
                color=color,
                zorder=zorder,
            )

    ### Prior models

    if plot_prior_models:
        prior_model_zorder = 2
        if desaturate_prior:
            for key in prior_work_colors:
                current = prior_work_colors[key]
                prior_work_colors[key] = kpl.lighten_color_hex(current)
            prior_model_zorder = -1500
        for prior_model in prior_models_to_plot:
            color = prior_work_colors[prior_model]
            fn = prior_model_fns[prior_model]
            vals = fn(temp_linspace)
            zfs_base = zfs_base_new_model if prior_model == "Toyli" else fn(1)

            if inverse_temp:
                plot_vals = zfs_base - vals
            elif new_model_diff:
                plot_vals = vals - cambria_lambda(plot_temp_linspace)
            else:
                plot_vals = vals
            label = None if plot_prior_data else prior_model
            if dash_predictions:
                pmdr = prior_model_data_ranges[prior_model]
                if inverse_temp:
                    if pmdr[0] == 0:
                        pmdr[0] = 1
                    pmdr = [1 / pmdr[1], 1 / pmdr[0]]
                in_range = np.array(
                    [pmdr[0] < el < pmdr[1] for el in plot_temp_linspace]
                )
                data_inds = np.nonzero(in_range)
                pred_inds_total = np.nonzero(np.logical_not(in_range))
                pred_inds_list = []
                last_ind = -10
                pred_inds = None
                for ind in pred_inds_total[0]:
                    if ind - last_ind != 1:
                        if pred_inds is not None:
                            pred_inds_list.append(np.array(pred_inds))
                        pred_inds = []
                    pred_inds.append(ind)
                    last_ind = ind
                if pred_inds is not None:
                    pred_inds_list.append(np.array(pred_inds))
                kpl.plot_line(
                    ax,
                    plot_temp_linspace[data_inds],
                    plot_vals[data_inds],
                    label=label,
                    color=color,
                )
                light_color = kpl.lighten_color_hex(color)
                for pred_inds in pred_inds_list:
                    kpl.plot_line(
                        ax,
                        plot_temp_linspace[pred_inds],
                        plot_vals[pred_inds],
                        # label=label,
                        # color=light_color,
                        color=color,
                        linestyle="dashed",
                    )
            else:
                kpl.plot_line(
                    ax,
                    plot_temp_linspace,
                    plot_vals,
                    label=label,
                    color=color,
                    zorder=prior_model_zorder,
                )

    ### Plot wrap up
    if not no_axis_labels:
        leg_loc = (
            kpl.Loc.UPPER_RIGHT
            if inverse_temp or plot_prior_models
            # if True
            else kpl.Loc.LOWER_LEFT
        )
        # handlelength = 0.5 if plot_data else 1.5
        handlelength = 1.0
        if plot_prior_models:
            handles, labels = ax.get_legend_handles_labels()
            adj_labels = []
            for ind in range(len(handles)):
                label = labels[ind]
                if label not in adj_labels:
                    adj_labels.append(label)
            adj_handles = [[] for ind in range(len(adj_labels))]
            for ind in range(len(handles)):
                handle = handles[ind]
                label = labels[ind]
                label_ind = adj_labels.index(label)
                adj_handles[label_ind].append(handle)
            adj_labels = tuple(adj_labels)
            adj_handles = [tuple(el[::-1]) for el in adj_handles]
            adj_handles = tuple(adj_handles)
            ax.legend(
                handles=adj_handles,
                labels=adj_labels,
                loc=leg_loc,
                handlelength=handlelength,
                fontsize=15,
                handler_map={tuple: matplotlib.legend_handler.HandlerTuple(None)},
            )
        # ax.set_xlabel("Temperature $\mathit{T}$ (K)")
        # ax.set_ylabel("Zero-field splitting $\mathit{D}$ (GHz)")
        if inverse_temp:
            ax.set_xlabel("Inverse temperature (K)")
        else:
            ax.set_xlabel("Temperature (K)")
        if inverse_temp:
            ax.set_ylabel("$\Delta D$ (GHz)")
        elif new_model_diff:
            ax.set_ylabel("Residuals (MHz)")
        else:
            ax.set_ylabel("ZFS (GHz)")
    xlim = (1 / temp_range[1], 1 / temp_range[0]) if inverse_temp else temp_range
    ax.set_xlim(*xlim)
    ax.set_ylim(*y_range)
    ax.set_yscale(yscale)


def main():
    # temp_range = [-10, 1000]
    # y_range = [2.74, 2.883]
    # temp_range = [-10, 720]
    # y_range = [2.80, 2.883]
    temp_range = [-10, 500]
    y_range = [2.847, 2.879]
    # temp_range = [-10, 310]
    # y_range = [2.8685, 2.8785]
    # temp_range = [280, 320]
    # y_range = [2.867, 2.873]
    # temp_range = [-10, 310]
    # y_range = [-0.0012, 0.0012]

    plot_data = True
    plot_residuals = False
    hist_residuals = False  # Must specify nv_to_plot down below
    condense_all = False
    condense_samples = True
    plot_prior_models = True
    desaturate_prior = True
    plot_new_model = True
    toyli_extension = False

    skip_lambda = lambda point: (
        point["Skip"]
        or point["Sample"] != "Wu"
        # or point["Sample"] != "15micro"
        # or point["ZFS file"] == ""
        # or point["Monitor temp (K)"] >= 296
    )

    ###

    min_temp, max_temp = temp_range
    min_temp = 0.1 if min_temp <= 0 else min_temp
    temp_linspace = np.linspace(min_temp, max_temp, 1000)
    fig, ax = plt.subplots()

    data_points = get_data_points(skip_lambda, condense_all, condense_samples)
    (
        zfs_list,
        zfs_err_list,
        temp_list,
        label_list,
        color_list,
        group_list,
    ) = data_points_to_lists(data_points)

    if toyli_extension:
        zfs_list.extend(toyli_zfss)
        temp_list.extend(toyli_temps)
        label_list.extend(["Toyli"] * len(toyli_temps))
        color_list.extend(KplColors.RED * len(toyli_temps))

    label_set = set(label_list)
    color_dict = {}
    for ind in range(len(zfs_list)):
        color_dict[label_list[ind]] = color_list[ind]

    ### New model

    # guess_params = [
    #     2.8778,
    #     0,  # -3.287e-15,
    #     -3e-08,
    #     0,  # -2.4e-10,
    #     0,  # -1.7e-13,
    #     0,  # -0.8e-23,
    # ]
    # guess_params = [
    #     2.87771,
    #     -4.625e-6,
    #     1.067e-7,
    #     -9.325e-10,
    #     1.739e-12,
    #     -1.838e-15,
    # ]  # , -1.838e-17]
    # guess_params = [
    #     2.87771,
    #     # -4.625e-6,
    #     -1.067e-7,
    #     # -9.325e-10,
    #     -1.739e-12,
    #     # -1.838e-15,
    #     -1.838e-17,
    # ]
    # fit_func = sub_room_zfs_from_temp_free
    # fit_func = zfs_from_temp_barson_free
    guess_params = [
        2.87771,
        -20,
        -300,
        # -4e-1,
        65,
        165,
        # 6.5,
    ]
    # fit_func = cambria_test
    fit_func = two_mode_qh
    # fit_func = jacobson
    if None in zfs_err_list:
        zfs_err_list = None
        absolute_sigma = False
    else:
        absolute_sigma = True
    popt, pcov = curve_fit(
        fit_func,
        temp_list,
        zfs_list,
        sigma=zfs_err_list,
        absolute_sigma=absolute_sigma,
        p0=guess_params,
    )
    print(popt)
    print(np.sqrt(np.diag(pcov)))
    # popt[2] = 0
    cambria_lambda = lambda temp: fit_func(
        temp,
        *popt,
        # *guess_params,
    )
    print(f"Predicted ZFS at 296 K: {cambria_lambda(296)}")
    ssr = 0
    num_points = len(temp_list)
    num_params = len(guess_params)
    if zfs_err_list is not None:
        for temp, zfs, zfs_err in zip(temp_list, zfs_list, zfs_err_list):
            calc_zfs = cambria_lambda(temp)
            ssr += ((zfs - calc_zfs) / zfs_err) ** 2
        dof = num_points - num_params
        red_chi_sq = ssr / dof
        print(red_chi_sq)

    used_data_labels = []
    if plot_data or plot_residuals:
        for ind in range(len(zfs_list)):
            temp = temp_list[ind]
            val = (
                zfs_list[ind] - cambria_lambda(temp)
                if plot_residuals
                else zfs_list[ind]
            )
            val_err = zfs_err_list[ind] if (zfs_err_list is not None) else None
            # label = None
            # color = KplColors.DARK_GRAY
            color = color_list[ind]
            label = label_list[ind]
            if label in used_data_labels:
                label = None
            else:
                used_data_labels.append(label)
            kpl.plot_points(
                ax,
                temp,
                val,
                yerr=val_err,
                color=color,
                zorder=-1,
                # zorder=temp - 1000,
                label=label,
            )
            # print(name, val, temp)
        if len(used_data_labels) > 1:
            ax.legend(loc=kpl.Loc.LOWER_LEFT)

    if hist_residuals:
        residuals = {}
        for el in label_set:
            residuals[el] = []
        for ind in range(len(zfs_list)):
            label = label_list[ind]
            temp = temp_list[ind]
            # if not (150 <= temp <= 500):
            #     continue
            # val = (zfs_list[ind] - cambria_lambda(temp)) / zfs_err_list[ind]
            val = zfs_list[ind] - cambria_lambda(temp)
            residuals[label].append(val)
        nv_to_plot = label_list[4]
        devs = residuals[nv_to_plot]
        # max_dev = 3
        # num_bins = max_dev * 4
        devs = [1000 * el for el in devs]
        max_dev = 0.0006 * 1000
        num_bins = 12
        large_errors = [abs(val) > max_dev for val in devs]
        if True in large_errors:
            print("Got a large error that won't be shown in hist!")
        hist, bin_edges = np.histogram(devs, bins=num_bins, range=(-max_dev, max_dev))
        x_vals = []
        y_vals = []
        for ind in range(len(bin_edges) - 1):
            x_vals.append(bin_edges[ind])
            x_vals.append(bin_edges[ind + 1])
            y_vals.append(hist[ind])
            y_vals.append(hist[ind])
        color = color_dict[nv_to_plot]
        kpl.plot_line(ax, x_vals, y_vals, color=color)
        ax.fill_between(x_vals, y_vals, color=kpl.lighten_color_hex(color))
        ylim = max(y_vals) + 1

    if plot_new_model:
        # color = KplColors.BLUE
        color = "#0f49bd"
        kpl.plot_line(
            ax,
            temp_linspace,
            cambria_lambda(temp_linspace),
            label="This work",
            color=color,
            zorder=10,
        )
        # color = KplColors.GRAY
        # kpl.plot_line(
        #     ax,
        #     temp_linspace,
        #     cambria_fixed(temp_linspace),
        #     label="MCAW proposed",
        #     color=color,
        #     zorder=10,
        # )
        # ax.legend()

    ### Prior models

    # prior_models_to_plot = ["Toyli", "Barson"]
    prior_models_to_plot = ["Toyli", "Barson", "Li", "Chen"]
    # prior_models_to_plot = ["Toyli"]
    if plot_prior_models:
        prior_model_colors = [
            KplColors.GREEN,
            KplColors.PURPLE,
            KplColors.RED,
            KplColors.ORANGE,
        ]
        prior_model_colors.reverse()
        prior_model_zorder = 2
        if desaturate_prior:
            prior_model_colors = [
                kpl.lighten_color_hex(el) for el in prior_model_colors
            ]
            prior_model_zorder = -1500
        if "Chen" in prior_models_to_plot:
            kpl.plot_line(
                ax,
                temp_linspace,
                sub_room_zfs_from_temp(temp_linspace),
                label="Chen",
                color=prior_model_colors[0],
                zorder=prior_model_zorder,
            )
        # print(super_room_zfs_from_temp(294))
        # print(super_room_zfs_from_temp(296))
        # print(super_room_zfs_from_temp(298))
        # return
        if "Toyli" in prior_models_to_plot:
            kpl.plot_line(
                ax,
                temp_linspace,
                super_room_zfs_from_temp(temp_linspace),
                label="Toyli",
                color=prior_model_colors[1],
                zorder=prior_model_zorder,
            )
        if "Barson" in prior_models_to_plot:
            kpl.plot_line(
                ax,
                temp_linspace,
                zfs_from_temp_barson(temp_linspace),
                label="Barson",
                color=prior_model_colors[2],
                zorder=prior_model_zorder,
            )
        if "Li" in prior_models_to_plot:
            kpl.plot_line(
                ax,
                temp_linspace,
                zfs_from_temp_li(temp_linspace),
                label="Li",
                color=prior_model_colors[3],
                zorder=prior_model_zorder,
            )

    ### Plot wrap up
    if plot_prior_models:
        ax.legend(loc="lower left")
        # ax.legend(bbox_to_anchor=(0.37, 0.46))
        # ax.legend(bbox_to_anchor=(0.329, 0.46))
    if hist_residuals:
        # ax.set_xlabel("Normalized residual")
        ax.set_xlabel("Residual (MHz)")
        ax.set_ylabel("Frequency")
        ax.set_xlim(-max_dev, max_dev)
        ax.set_ylim(0, ylim)
    else:
        ax.set_xlabel("Temperature $\mathit{T}$ (K)")
        ax.set_ylabel("$\mathit{D}$ (GHz)")
        ax.set_xlim(*temp_range)
        ax.set_ylim(*y_range)

    # fig, ax = plt.subplots()
    # chen_proposed_diff = lambda temp: sub_room_zfs_from_temp(temp) - cambria_lambda(temp)
    # kpl.plot_line(
    #     ax,
    #     temp_linspace,
    #     1000 * chen_proposed_diff(temp_linspace)
    # )
    # ax.set_xlabel(r"Temperature $\mathit{T}$ (K)")
    # ax.set_ylabel("Chen - proposed (MHz)")


# endregion

if __name__ == "__main__":
    # print(calibrate_digitization("chen_2011_3a", sub_room_zfs_from_temp))
    # print(calibrate_digitization("toyli_2012_5c", super_room_zfs_from_temp))
    # print(calibrate_digitization("li_2017_1b", zfs_from_temp_li))
    # sys.exit()

    # calc_zfs_from_compiled_data()
    # sys.exit()

    kpl.init_kplotlib()

    # main()
    fig(inset_resid=True)  # Main
    # fig(  # Comps
    #     #     temp_range=[-20, 820],
    #     #     y_range=[2.80, 2.88],
    #     #
    #     # temp_range=[-20, 1020],
    #     # y_range=[2.76, 2.88],
    #     # y_range=[-0.01, 0.01],
    #     #
    #     temp_range=[0, 1000],
    #     y_range=[2.76, 2.88],
    #     #
    #     plot_data=True,
    #     condense_all=False,
    #     condense_samples=True,
    #     plot_prior_models=True,
    #     desaturate_prior=False,
    #     plot_new_model=True,
    #     plot_prior_data=True,
    #     new_model_diff=False,
    #     dash_predictions=True,
    #     inset_comp=True,
    #     inset_resid=False,
    # )
    # fig(  # Comps semi-log vs inverse temp
    #     # temp_range=[100, 1000],
    #     # y_range=[1e-5, 1],
    #     temp_range=[150, 1000],
    #     y_range=[5e-4, 1],
    #     plot_data=False,
    #     condense_all=False,
    #     condense_samples=True,
    #     plot_prior_models=True,
    #     desaturate_prior=False,
    #     plot_new_model=True,
    #     plot_prior_data=False,
    #     inverse_temp=True,
    #     yscale="log",
    #     dash_predictions=True,
    # )
    # refit_experiments()
    # # # derivative_comp()
    # light_polarization()

    plt.show(block=True)
    sys.exit()

    vals = [
        [2.869549, 2.869409, 2.869265, 2.869323, 2.869320],
        [2.869514, 2.869471, 2.869382, 2.869566, 2.869293],
        [2.869116, 2.869526, 2.869619, 2.869388, 2.869451],
        [2.869500, 2.870178, 2.869463, 2.870033, 2.869479],
        [2.869472, 2.869201, 2.869634, 2.869352, 2.869265],
        [2.868786, 2.869506, 2.869398, 2.869324, 2.869297],
        [2.868402, 2.869612, 2.868951, 2.869346, 2.869729],
        [2.869555, 2.869349, 2.869200, 2.869306, 2.869191],
        [2.869638, 2.869301, 2.869307, 2.869410, 2.869455],
        [2.869586, 2.869742, 2.86961, 2.869567, 2.869645],
        [2.86957, 2.869903, 2.869913, 2.869949, 2.869767],
        [2.869699, 2.869654, 2.869838, 2.869908, 2.869865],
        [2.869846, 2.869777, 2.86989, 2.869863, 2.869619],
        [2.869825, 2.869832, 2.869823, 2.869929, 2.869966],
        [2.869906, 2.869865, 2.869828, 2.869784, 2.870061],
    ]
    errs = [
        [0.000122, 0.000151, 0.000122, 0.000126, 0.000136],
        [0.000125, 0.000162, 0.000131, 0.000129, 0.000137],
        [0.00015, 0.000187, 0.000156, 0.00015, 0.000158],
        [0.000353, 0.000354, 0.000323, 0.00036, 0.000379],
        [0.000132, 0.000188, 0.00014, 0.000131, 0.000138],
        [0.00026, 0.000301, 0.000262, 0.000256, 0.000249],
        [0.000332, 0.000436, 0.00035, 0.000332, 0.000363],
        [0.000119, 0.000153, 0.000116, 0.000121, 0.000133],
        [0.000246, 0.000309, 0.000244, 0.000229, 0.000231],
        [0.000119, 0.000148, 0.000123, 0.000117, 0.000129],
        [0.000124, 9e-05, 0.000132, 0.000113, 0.000108],
        [0.000114, 0.000103, 0.000133, 0.000122, 0.000107],
        [0.000112, 8.7e-05, 0.000128, 0.000107, 0.000112],
        [0.00011, 8.9e-05, 0.000129, 0.000119, 0.000104],
        [7.9e-05, 7.6e-05, 6.2e-05, 7.9e-05, 8.6e-05],
    ]
    vals = np.array(vals)
    errs = np.array(errs)

    sub_vals = vals[-1]
    sub_errs = errs[-1]
    for ind in range(len(sub_vals)):
        val = sub_vals[ind]
        err = sub_errs[ind]
        print(tool_belt.round_for_print(val, err))
    sys.exit()

    num_inds = len(vals)
    # num_inds = 5
    for ind in range(num_inds):
        sub_vals = vals[ind]
        sub_errs = errs[ind]
        # sub_vals = vals[:, ind]
        # sub_errs = errs[:, ind]

        avg = np.average(sub_vals, weights=1 / (sub_errs**2))
        err = np.sqrt(1 / np.sum(1 / (sub_errs**2)))

        print(tool_belt.presentation_round(avg, err))
        print()
