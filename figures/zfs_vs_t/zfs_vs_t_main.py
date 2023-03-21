# -*- coding: utf-8 -*-
"""
Get the NV temp based on the ZFS, using numbers from: 'Temperature dependent 
energy level shifts of nitrogen-vacancy centers in diamond'

Created on Fri Mar  5 12:42:32 2021

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
from analysis import three_level_rabi
from scipy.integrate import quad

# fmt: off
toyli_digitized = [300, 2.87, 309.9858044201037, 2.8690768841409784, 320.04280071681194, 2.868366259576263, 330.32149670254546, 2.8673945841666115, 340.3583384820696, 2.866304172245094, 350.05837349046874, 2.8655253065868678, 360.1625766242179, 2.8644972039180088, 370.064695695292, 2.8633133281175045, 380.2362601832661, 2.8622540708223165, 390.13837925434024, 2.8611013496481412, 399.9731369711893, 2.8600109377266243, 410.00997875071346, 2.858858216552449, 420.0468205302376, 2.857362794488654, 430.4878304351117, 2.856176937898957, 440.3899495061858, 2.8549015790086583, 450.02262316036, 2.8535619300765087, 460.1268262941091, 2.852066508012714, 469.96158401095823, 2.850633395201577, 480.5373166242823, 2.849387210148415, 490.30471298690645, 2.84789178808462, 500.2068320579806, 2.8465209845261414, 510.04158977482973, 2.844994407836017, 520.2805156170289, 2.843374367266906, 530.452080105003, 2.8417854813241243, 540.354199176077, 2.840258904634, 550.1215955387013, 2.838638864064889, 560.1584373182253, 2.837361524385398, 570.3300018061993, 2.8357103291899572, 580.2321208772735, 2.8341545786626967, 590.4036853652476, 2.8322813395045685, 600.0363590194218, 2.830756743603637, 610.005839444721, 2.829354785418829, 619.9079585157951, 2.8275789717180726, 630.4163297748942, 2.826052395027949, 640.3184488459683, 2.824556972964154, 650.2879292712674, 2.8227500046370686, 660.1269697755595, 2.821005345562641, 669.8900833507407, 2.8189160048094015, 680.4658159640647, 2.816922108724342, 690.5700190978139, 2.8151482758127777, 700.472138168888, 2.8134950998281454, 710.0374504688373, 2.812188586311517]
# fmt: on
toyli_temps = toyli_digitized[0::2]
toyli_temps = [round(val, -1) for val in toyli_temps]
toyli_zfss = toyli_digitized[1::2]
# Adjust for my poor digitization
toyli_zfss = np.array(toyli_zfss)
toyli_zfss -= 2.87
toyli_zfss *= 0.9857
toyli_zfss += 2.8701

nvdata_dir = common.get_nvdata_dir()
compiled_data_file_name = "zfs_vs_t"
compiled_data_path = nvdata_dir / "paper_materials/zfs_temp_dep"


# endregion
# region Functions


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
            label = "Cambria"
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

    # skip_lambda = lambda point: point["Skip"] or point["ZFS file"] == ""
    skip_lambda = (
        lambda point: point["Skip"]
        # or point["ZFS file"] == ""
        or point["Sample"] != "15micro"
        # or point["Sample"] != "Wu"
        # or point["Setpoint temp (K)"] != ""
        # or point["Setpoint temp (K)"] < 300
    )

    data_points = get_data_points(skip_lambda)
    file_list = [el["ZFS file"] for el in data_points]
    file_list = file_list[164:165]
    file_list = [
        # # 1 us
        # "2023_03_03-17_23_24-15micro-nv6_zfs_vs_t",
        # "2023_03_03-16_55_36-15micro-nv7_zfs_vs_t",
        # "2023_03_03-16_28_28-15micro-nv8_zfs_vs_t",
        # "2023_03_03-16_00_44-15micro-nv9_zfs_vs_t",
        # "2023_03_03-15_32_43-15micro-nv11_zfs_vs_t",
        # # 10 us
        # "2023_03_03-18_46_02-15micro-nv6_zfs_vs_t",
        # "2023_03_03-18_18_54-15micro-nv7_zfs_vs_t",
        # "2023_03_03-20_07_05-15micro-nv8_zfs_vs_t",
        # "2023_03_03-19_40_03-15micro-nv9_zfs_vs_t",
        # "2023_03_03-19_13_03-15micro-nv11_zfs_vs_t",
        # # 100 us
        # "2023_03_03-21_03_25-15micro-nv6_zfs_vs_t",
        # "2023_03_03-22_57_55-15micro-nv7_zfs_vs_t",
        # "2023_03_03-22_29_43-15micro-nv8_zfs_vs_t",
        # "2023_03_03-21_32_20-15micro-nv9_zfs_vs_t",
        # "2023_03_03-22_00_57-15micro-nv11_zfs_vs_t",
        # 1 ms
        # "2023_03_04-11_43_50-15micro-nv6_zfs_vs_t",
        # "2023_03_04-11_06_24-15micro-nv7_zfs_vs_t",
        # "2023_03_04-12_58_51-15micro-nv8_zfs_vs_t",
        # "2023_03_04-12_21_20-15micro-nv9_zfs_vs_t",
        # "2023_03_04-13_36_17-15micro-nv11_zfs_vs_t",
        # 1 us, ND 0.3 => 0.5
        # "2023_03_04-16_40_09-15micro-nv6_zfs_vs_t",
        # "2023_03_04-14_55_01-15micro-nv7_zfs_vs_t",
        # "2023_03_04-15_47_39-15micro-nv8_zfs_vs_t",
        # "2023_03_04-18_25_23-15micro-nv9_zfs_vs_t",
        # "2023_03_04-17_32_26-15micro-nv11_zfs_vs_t",
        # # microwave 10 => 0 dBm
        # "2023_03_04-21_22_00-15micro-nv6_zfs_vs_t",
        # "2023_03_04-23_12_14-15micro-nv7_zfs_vs_t",
        # "2023_03_04-20_26_41-15micro-nv8_zfs_vs_t",
        # "2023_03_04-22_17_57-15micro-nv9_zfs_vs_t",
        # "2023_03_05-00_07_19-15micro-nv11_zfs_vs_t",
        # # ND 1.0
        # "2023_03_05-13_24_15-15micro-nv6_zfs_vs_t",
        # "2023_03_05-11_41_45-15micro-nv7_zfs_vs_t",
        # "2023_03_05-14_15_18-15micro-nv8_zfs_vs_t",
        # "2023_03_05-12_33_07-15micro-nv9_zfs_vs_t",
        # "2023_03_05-10_50_58-15micro-nv11_zfs_vs_t",
        # # Temp control disconnected
        # "2023_03_06-20_37_53-15micro-nv6_zfs_vs_t",
        # "2023_03_06-20_09_53-15micro-nv7_zfs_vs_t",
        # "2023_03_06-19_14_17-15micro-nv8_zfs_vs_t",
        # "2023_03_06-19_42_31-15micro-nv9_zfs_vs_t",
        # "2023_03_06-21_05_38-15micro-nv11_zfs_vs_t",
        # 1 ms delay repeat
        # "2023_03_07-05_26_17-15micro-nv6_zfs_vs_t",
        # "2023_03_07-04_11_05-15micro-nv7_zfs_vs_t",
        # "2023_03_07-02_56_13-15micro-nv8_zfs_vs_t",
        # "2023_03_07-00_26_02-15micro-nv9_zfs_vs_t",
        # "2023_03_07-01_41_04-15micro-nv11_zfs_vs_t",
        # uwave polarization
        # "2023_03_07-14_27_00-15micro-nv6_zfs_vs_t",
        # "2023_03_07-12_34_04-15micro-nv7_zfs_vs_t",
        # "2023_03_07-13_30_18-15micro-nv8_zfs_vs_t",
        # "2023_03_07-13_58_48-15micro-nv9_zfs_vs_t",
        # "2023_03_07-13_02_08-15micro-nv11_zfs_vs_t",
        # New NVs 1
        # "2023_03_09-13_06_10-15micro-nv6_offset",
        # "2023_03_09-12_14_14-15micro-nv7_offset",
        # "2023_03_09-12_40_16-15micro-nv8_offset",
        # New NVs 2, 348 degrees
        # "2023_03_09-16_14_03-15micro-nv12_offset",
        # "2023_03_09-15_46_07-15micro-nv13_offset",
        # "2023_03_09-15_18_27-15micro-nv14_offset",
        # "2023_03_09-14_50_35-15micro-nv15_offset",
        # "2023_03_09-14_21_53-15micro-nv16_offset",
        # # 88 degrees
        # "2023_03_09-23_37_19-15micro-nv12_offset",
        # "2023_03_09-18_37_07-15micro-nv13_offset",
        # "2023_03_09-23_09_00-15micro-nv14_offset",
        # "2023_03_09-17_41_03-15micro-nv15_offset",
        # "2023_03_09-18_09_53-15micro-nv16_offset",
        # # 58 degrees
        # "2023_03_10-13_45_46-15micro-nv12_offset",
        # "2023_03_10-14_13_33-15micro-nv13_offset",
        # "2023_03_10-15_39_17-15micro-nv14_offset",
        # "2023_03_10-15_11_35-15micro-nv15_offset",
        # "2023_03_10-14_42_16-15micro-nv16_offset",
        # # 38 degrees
        # "2023_03_10-17_10_38-15micro-nv12_offset",
        # "2023_03_10-19_04_01-15micro-nv13_offset",
        # "2023_03_10-17_38_32-15micro-nv14_offset",
        # "2023_03_10-18_36_06-15micro-nv15_offset",
        # "2023_03_10-18_07_29-15micro-nv16_offset",
        # 38 degrees, finer average
        # "2023_03_10-23_44_44-15micro-nv12_offset",
        # "2023_03_11-01_37_36-15micro-nv13_offset",
        # "2023_03_11-03_31_26-15micro-nv14_offset",
        # "2023_03_11-02_35_44-15micro-nv15_offset",
        # "2023_03_11-00_42_21-15micro-nv16_offset",
        # # 18 degrees
        # "2023_03_12-13_11_31-15micro-nv12_offset",
        # "2023_03_12-13_39_16-15micro-nv13_offset",
        # "2023_03_12-14_07_58-15micro-nv14_offset",
        # "2023_03_12-14_36_30-15micro-nv15_offset",
        # "2023_03_12-12_43_05-15micro-nv16_offset",
        # # 338 degrees
        # "2023_03_12-18_32_33-15micro-nv12_offset",
        # "2023_03_12-19_58_21-15micro-nv13_offset",
        # "2023_03_12-20_27_13-15micro-nv14_offset",
        # "2023_03_12-19_30_29-15micro-nv15_offset",
        # "2023_03_12-19_01_42-15micro-nv16_offset",
        # # 358 degrees
        # "2023_03_13-00_58_53-15micro-nv12_offset",
        # "2023_03_13-02_51_05-15micro-nv13_offset",
        # "2023_03_13-03_48_32-15micro-nv14_offset",
        # "2023_03_13-01_55_57-15micro-nv15_offset",
        # "2023_03_13-00_02_24-15micro-nv16_offset",
        # 248 degrees
        # "2023_03_13-13_49_30-15micro-nv12_offset",
        # "2023_03_13-12_22_52-15micro-nv13_offset",
        # "2023_03_13-14_18_30-15micro-nv14_offset",
        # "2023_03_13-12_51_39-15micro-nv15_offset",
        # "2023_03_13-13_21_04-15micro-nv16_offset",
        # 348 degrees
        # "2023_03_13-15_33_36-15micro-nv12_offset",
        # "2023_03_13-15_05_02-15micro-nv13_offset",
        # "2023_03_13-17_00_38-15micro-nv14_offset",
        # "2023_03_13-16_02_24-15micro-nv15_offset",
        # "2023_03_13-16_31_38-15micro-nv16_offset",
        # 58 degrees
        "2023_03_13-19_38_01-15micro-nv12_offset",
        "2023_03_13-19_09_20-15micro-nv13_offset",
        "2023_03_13-18_41_27-15micro-nv14_offset",
        "2023_03_13-18_12_28-15micro-nv15_offset",
        "2023_03_13-17_43_29-15micro-nv16_offset",
    ]

    ### Loop

    table_popt = None
    table_pste = None

    for file_name in file_list:

        ### Data extraction and processing

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
        # uwave_pulse_dur = data["uwave_pulse_dur"]
        uwave_pulse_dur = None
        try:
            norm_style = tool_belt.NormStyle[str.upper(nv_sig["norm_style"])]
        except Exception as exc:
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
            # line_func = lambda freq, contrast, rabi_freq, center: pesr.rabi_line(
            #     freq, contrast, rabi_freq, center, uwave_pulse_dur=uwave_pulse_dur
            # )
            # guess_params = [0.2, 4, freq_center]
            # guess_params = [0.3, 500 / uwave_pulse_dur, freq_center]
            # guess_params = [0.4, 9, 2.8748]

            line_func = lambda freq, contrast, rabi_freq, center, splitting: three_level_rabi.coherent_line(
                freq, contrast, rabi_freq, center, splitting, uwave_pulse_dur
            )
            guess_params = [0.2, 4, freq_center, 5]

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
            guess_params = [0.05, 3, freq_center, 6, -0.001]

            # line_func = lambda freq, contrast, hwhm, splitting, offset: pesr.lorentzian_split_offset(freq, contrast, hwhm, 2.87, splitting, offset)
            # guess_params = [0.05, 3, 6, 0.005]

            # line_func = pesr.lorentzian_test
            # guess_params = [0.05, 3, freq_center, 6, 0.005, 0.05, 3]

            # fmt: on

        ### Raw data figure

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
        if table_popt is None:
            table_popt = []
            table_pste = []
            table_red_chi_sq = []
            # for ind in range(len(popt)):
            for ind in range(5):
                table_popt.append([])
                table_pste.append([])
        for ind in range(len(popt)):
            val = popt[ind]
            err = np.sqrt(pcov[ind, ind])
            val_col = table_popt[ind]
            err_col = table_pste[ind]
            val_col.append(round(val, 6))
            err_col.append(round(err, 6))

        fit_lambda = lambda freq: fit_func(freq, *popt)
        freqs = pesr.calculate_freqs(freq_center, freq_range, num_steps)
        chi_sq = np.sum(((fit_lambda(freqs) - norm_avg_sig) / norm_avg_sig_ste) ** 2)
        red_chi_sq = chi_sq / (len(norm_avg_sig) - len(popt))
        table_red_chi_sq.append(red_chi_sq)

        # Close the plots so they don't clutter everything up
        # plt.close("all")

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

    zfs_vals = np.array(table_popt[2])
    zfs_errs = np.array(table_pste[2])

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
    Comes from Barson paper!
    """

    zfs0 = 2.87771  # GHz
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

    zfs0 = 2.87769  # GHz
    A = 5.6e-7  # GHz / K**2
    B = 490  # K

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


def einstein_heat_capacity(e, T):
    return (((e / T) ** 2) * np.exp(e / T)) / ((np.exp(e / T) - 1) ** 2)


def jacobson(temp, zfs0, coeff):
    """Coefficient of thermal expansion from Jacobson and Stoupin 2019"""
    # X1[10−6/K]  0.0096  0.0210
    # X2[10−6/K]  0.2656  0.3897
    # X3[10−6/K]  2.6799  3.4447
    # X4[10−6/K]  2.3303  2.2796
    # Θ1 [K]      159.3   225.2
    # Θ2 [K]      548.5   634.0
    # Θ3 [K]      1237.9  1365.5
    # Θ4 [K]      2117.8  3068.8

    # coeffs = [0.0096, 0.2656, 2.6799, 2.3303]
    # energies = [159.3, 548.5, 1237.9, 2117.8]  # K
    coeffs = [0.0210, 0.3897, 3.4447, 2.2796]
    energies = [225.2, 634.0, 1365.5, 3068.8]  # K
    jacobson_total = None
    for ind in range(4):
        energy = energies[ind]
        alpha_coeff = coeffs[ind]
        if jacobson_total is None:
            jacobson_total = alpha_coeff * bose(energy, temp)
        else:
            jacobson_total += alpha_coeff * bose(energy, temp)
        # kpl.plot_line(ax, temp_linspace, sub_lambda(temp_linspace), label=ind)

    return zfs0 * np.exp(jacobson_total)


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
    zfs_list, zfs_err_list, temp_list, _, _ = data_points_to_lists(data_points)
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
    zfs_list, zfs_err_list, temp_list, _, _ = data_points_to_lists(data_points)
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


# endregion
# region Main plots


def main():

    # temp_range = [-10, 1000]
    # y_range = [2.74, 2.883]
    # temp_range = [-10, 720]
    # y_range = [2.80, 2.883]
    temp_range = [-10, 2000]
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
    plot_prior_models = False
    desaturate_prior = False
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
    zfs_list, zfs_err_list, temp_list, label_list, color_list = data_points_to_lists(
        data_points
    )

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
        # -8e-2,
        -300,
        # -4e-1,
        # 65,
        # 165,
        # 6.5,
    ]
    # fit_func = cambria_test
    # fit_func = cambria_test3
    fit_func = jacobson
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
    popt = [3.567, 1e-4]
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
            label="Proposed",
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
    prior_models_to_plot = ["Toyli"]
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

    # print(cambria_fixed(15))
    # sys.exit()

    # calc_zfs_from_compiled_data()
    # sys.exit()

    kpl.init_kplotlib()

    main()
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
