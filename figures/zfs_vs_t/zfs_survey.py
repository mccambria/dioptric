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
from utils.tool_belt import States, NormStyle
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


def reanalyze():
    file_list = [
        "2023_02_09-13_52_02-wu-nv6_zfs_vs_t",
        "2023_02_09-13_29_32-wu-nv7_zfs_vs_t",
        "2023_02_09-14_14_33-wu-nv8_zfs_vs_t",
        "2023_02_09-13_07_10-wu-nv10_zfs_vs_t",
        "2023_02_09-14_37_43-wu-nv11_zfs_vs_t",
        "2023_02_09-17_28_01-wu-nv1_region2",
        "2023_02_09-18_02_43-wu-nv2_region2",
        "2023_02_09-18_14_01-wu-nv3_region2",
        "2023_02_09-17_51_24-wu-nv4_region2",
        "2023_02_09-17_39_51-wu-nv5_region2",
        "2023_02_09-23_28_39-wu-nv1_region3",
        "2023_02_09-23_51_39-wu-nv2_region3",
        "2023_02_10-00_14_56-wu-nv3_region3",
        "2023_02_10-00_37_40-wu-nv4_region3",
        "2023_02_10-00_59_59-wu-nv5_region3",
        "2023_02_10-19_13_33-wu-nv1_region4",
        "2023_02_10-18_51_08-wu-nv2_region4",
        "2023_02_10-18_28_42-wu-nv3_region4",
        "2023_02_10-18_06_16-wu-nv4_region4",
        "2023_02_10-19_36_05-wu-nv5_region4",
        "2023_02_13-11_54_40-wu-nv1_region5",
        "2023_02_13-10_47_07-wu-nv2_region5",
        "2023_02_13-11_32_11-wu-nv3_region5",
        "2023_02_13-11_09_39-wu-nv4_region5",
        "2023_02_13-12_17_20-wu-nv5_region5",
        "2023_02_14-19_34_18-wu-nv6_region5",
        "2023_02_15-11_34_42-wu-nv6_region5",
        "2023_02_14-18_25_12-wu-nv7_region5",
        "2023_02_15-10_49_10-wu-nv7_region5",
        "2023_02_14-16_31_33-wu-nv8_region5",
        "2023_02_15-10_03_52-wu-nv8_region5",
        "2023_02_14-19_56_53-wu-nv9_region5",
        "2023_02_15-09_17_38-wu-nv9_region5",
        "2023_02_14-17_39_49-wu-nv10_region5",
        "2023_02_15-08_54_44-wu-nv10_region5",
        "2023_02_14-18_02_32-wu-nv11_region5",
        "2023_02_15-08_31_53-wu-nv11_region5",
        "2023_02_14-19_11_31-wu-nv12_region5",
        "2023_02_15-11_12_05-wu-nv12_region5",
        "2023_02_14-16_54_38-wu-nv13_region5",
        "2023_02_15-09_41_02-wu-nv13_region5",
        "2023_02_14-17_17_04-wu-nv14_region5",
        "2023_02_15-11_57_26-wu-nv14_region5",
        "2023_02_14-18_47_39-wu-nv15_region5",
        "2023_02_15-10_26_32-wu-nv15_region5",
        "2023_02_16-11_38_00-wu-nv16_region5",
        "2023_02_16-15_21_12-wu-nv16_region5",
        "2023_02_16-12_45_08-wu-nv17_region5",
        "2023_02_16-16_28_17-wu-nv17_region5",
        "2023_02_16-13_07_28-wu-nv18_region5",
        "2023_02_16-17_58_56-wu-nv18_region5",
        "2023_02_16-13_52_11-wu-nv19_region5",
        "2023_02_16-17_36_17-wu-nv19_region5",
        "2023_02_16-14_14_37-wu-nv20_region5",
        "2023_02_16-14_36_43-wu-nv20_region5",
        "2023_02_16-11_15_49-wu-nv21_region5",
        "2023_02_16-16_51_10-wu-nv21_region5",
        "2023_02_16-12_00_24-wu-nv22_region5",
        "2023_02_16-14_59_00-wu-nv22_region5",
        "2023_02_16-12_22_59-wu-nv23_region5",
        "2023_02_16-17_14_04-wu-nv23_region5",
        "2023_02_16-13_29_52-wu-nv24_region5",
        "2023_02_16-15_43_44-wu-nv24_region5",
        "2023_02_16-10_53_23-wu-nv25_region5",
        "2023_02_16-16_05_52-wu-nv25_region5",
    ]

    file_list = [
        "2023_02_22-18_06_53-15micro-nv6_zfs_vs_t",
        "2023_02_22-19_30_08-15micro-nv7_zfs_vs_t",
        "2023_02_22-18_34_55-15micro-nv8_zfs_vs_t",
        "2023_02_22-19_02_58-15micro-nv9_zfs_vs_t",
        "2023_02_22-20_51_42-15micro-nv11_zfs_vs_t",
    ]

    # file_list = file_list[21:22]

    for file_name in file_list:
        if "nv14_region5" in file_name:
            print(0.0)
            continue

        data = tool_belt.get_raw_data(file_name)

        # print(file_name)
        # print(return_res_with_error(data, fit_func, guess_params))
        # print()
        # sys.exit()

        freq_center = data["freq_center"]
        freq_range = data["freq_range"]
        num_steps = data["num_steps"]
        ref_counts = data["ref_counts"]
        sig_counts = data["sig_counts"]
        num_reps = data["num_reps"]
        nv_sig = data["nv_sig"]
        readout = nv_sig["spin_readout_dur"]
        # uwave_pulse_dur = data["uwave_pulse_dur"]
        # uwave_pulse_dur = 300
        uwave_pulse_dur = None
        # uwave_pulse_dur = 150
        try:
            norm_style = NormStyle[str.upper(nv_sig["norm_style"])]
        except Exception as exc:
            # norm_style = NormStyle.POINT_TO_POINT
            norm_style = NormStyle.SINGLE_VALUED

        # line_func = lorentzian
        line_func = pesr.lorentzian_split
        # line_func = lorentzian_sum
        # line_func = gaussian
        # line_func = lambda freq, contrast, width, center: rabi_line_n14_hyperfine(
        #     freq,
        #     contrast,
        #     width,
        #     center,
        #     uwave_pulse_dur=uwave_pulse_dur,
        #     coherent=False,
        # )
        # line_func = lambda freq, contrast, width, center: rabi_line(
        #     freq,
        #     contrast,
        #     width,
        #     center,
        #     uwave_pulse_dur=uwave_pulse_dur,
        #     coherent=True,
        # )
        # line_func = lambda freq, contrast, width, center: lorentzian_sum(
        #     freq, contrast, width, center, freq_range
        # )
        num_resonances = None
        # guess_params = [0.01, 2, 2.867, 0.01, 2, 2.873]
        # guess_params = [0.01, 2, 2.87]
        guess_params = [0.01, 6, 2.87, 7]
        # guess_params = None

        fit_func = None
        popt = None

        # popt = [0.18, 2.0, 2.867, 0.18, 2.0, 2.873]
        # fit_func = lambda freq, *res_args: dip_sum(freq, line_func, *res_args)

        # fig, ax = plt.subplots()
        # freqs_linspace = np.linspace(2.85, 2.89, 100)
        # fit_func = lambda freq: dip_sum(freq, line_func, 0.2, 2.0, 2.87)
        # kpl.plot_line(ax, freqs_linspace, fit_func(freqs_linspace))
        # break

        ret_vals = tool_belt.process_counts(
            sig_counts, ref_counts, num_reps, readout, norm_style
        )
        (
            sig_counts_avg_kcps,
            ref_counts_avg_kcps,
            norm_avg_sig,
            norm_avg_sig_ste,
        ) = ret_vals
        # create_raw_data_figure(
        #     freq_center,
        #     freq_range,
        #     num_steps,
        #     sig_counts_avg_kcps,
        #     ref_counts_avg_kcps,
        #     norm_avg_sig,
        # )
        fit_fig, _, _, popt, pcov = pesr.create_fit_figure(
            freq_center,
            freq_range,
            num_steps,
            norm_avg_sig,
            norm_avg_sig_ste,
            popt=popt,
            fit_func=fit_func,
            line_func=line_func,
            num_resonances=num_resonances,
            guess_params=guess_params,
        )
        # fit_func, popt, pcov = fit_resonance(
        #     freq_center,
        #     freq_range,
        #     num_steps,
        #     norm_avg_sig,
        #     norm_avg_sig_ste,
        # popt=popt,
        # fit_func=fit_func,
        #     line_func=line_func,
        #     num_resonances=num_resonances,
        #     guess_params=guess_params,
        # )

        # pste = np.sqrt(np.diag(pcov))
        # # Reverse for presentation
        # popt = popt[::-1]
        # pste = pste[::-1]
        # round_popt = [tool_belt.round_sig_figs(val, 7) for val in popt]
        # round_pste = [tool_belt.round_sig_figs(val, 3) for val in pste]
        # print_list = []
        # for ind in range(len(popt)):
        #     print_list.append(round_popt[ind])
        #     print_list.append(round_pste[ind])
        # print(print_list)

        # print(
        #     round(1000 * ((popt[3] + popt[1]) / 2) - 2870, 1),
        #     round(1000 * (popt[3] - popt[1]), 1),
        # )

        # file_path = tool_belt.get_raw_data_path(file_name)
        # file_path = file_path.with_stem(file_name + "-fit").with_suffix("")
        # tool_belt.save_figure(fit_fig, file_path)

        # break


# endregion


def main():
    skip_lambda = lambda pt: pt["Skip"]
    # skip_lambda = lambda pt: pt["Skip"] or pt["Region"] != 5

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
    # reanalyze()

    plt.show(block=True)
