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


def calc_zfs_from_compiled_data():

    skip_lambda = lambda point: point["Sample"] != "15micro"

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


def refit_experiments():
    """Re-run fits to experimental data, either plotting and saving the new plots
    or just printing out the fit parameters
    """

    ### User setup
    # Also see below section Sample-dependent fit...

    do_plot = False  # Generate raw data and fit plots?
    do_save = False  # Save the plots?
    do_print = True  # Print out popts and associated error bars?

    data_points = get_data_points()
    # sample = "Wu"
    sample = "15micro"
    file_list = [el["ZFS file"] for el in data_points if el["Sample"] == sample]

    ### Loop

    table_popt = None
    table_pste = None

    for file_name in file_list:

        ### Data extraction and processing

        # file_name = "2022_11_19-09_14_08-wu-nv1_zfs_vs_t"
        data = tool_belt.get_raw_data(file_name)
        raw_file_path = tool_belt.get_raw_data_path(file_name)
        freq_center = data["freq_center"]
        freq_range = data["freq_range"]
        num_steps = data["num_steps"]
        ref_counts = data["ref_counts"]
        sig_counts = data["sig_counts"]
        num_reps = data["num_reps"]
        nv_sig = data["nv_sig"]
        readout = nv_sig["spin_readout_dur"]
        try:
            norm_style = tool_belt.NormStyle[str.upper(nv_sig["norm_style"])]
        except Exception as exc:
            # norm_style = NormStyle.POINT_TO_POINT
            norm_style = tool_belt.NormStyle.SINGLE_VALUED

        ret_vals = pesr.process_counts(
            sig_counts, ref_counts, num_reps, readout, norm_style
        )
        (
            sig_counts_avg_kcps,
            ref_counts_avg_kcps,
            norm_avg_sig,
            norm_avg_sig_ste,
        ) = ret_vals

        ### Raw data figure

        if do_plot:
            ret_vals = pesr.create_raw_data_figure(
                freq_center,
                freq_range,
                num_steps,
                sig_counts_avg_kcps,
                ref_counts_avg_kcps,
                norm_avg_sig,
            )
            if do_save:
                raw_fig = ret_vals[0]
                file_path = raw_file_path.with_suffix(".svg")
                tool_belt.save_figure(raw_fig, file_path)

        ### Sample-dependent fit functions and parameters

        if sample == "Wu":
            fit_func = lambda freq, contrast, rabi_freq, center: pesr.single_dip(
                freq, contrast, rabi_freq, center, dip_func=pesr.rabi_line_hyperfine
            )
            guess_params = [0.1, 5, freq_center]
        elif sample == "15micro":
            fit_func = lambda freq, low_contrast, low_width, low_center, high_contrast, high_width, high_center: pesr.double_dip(
                freq,
                low_contrast,
                low_width,
                low_center,
                high_contrast,
                high_width,
                high_center,
                dip_func=pesr.lorentzian,
            )
            guess_params = [
                0.015,
                7,
                freq_center - 0.005,
                0.015,
                7,
                freq_center + 0.005,
            ]

        ### Raw data figure

        if do_plot:
            fit_fig, fit_func, popt, pcov = pesr.create_fit_figure(
                freq_center,
                freq_range,
                num_steps,
                norm_avg_sig,
                norm_avg_sig_ste,
                fit_func=fit_func,
                guess_params=guess_params,
            )
            if do_save:
                file_path = raw_file_path.with_name((f"{file_name}-fit"))
                file_path = file_path.with_suffix(".svg")
                tool_belt.save_figure(fit_fig, file_path)

        ### Get fit parameters and error bars

        if do_print:
            if not do_plot:
                fit_func, popt, pcov = pesr.fit_resonance(
                    freq_center,
                    freq_range,
                    num_steps,
                    norm_avg_sig,
                    norm_avg_sig_ste,
                    fit_func=fit_func,
                    guess_params=guess_params,
                )
            if table_popt is None:
                table_popt = []
                table_pste = []
                for ind in range(len(popt)):
                    table_popt.append([])
                    table_pste.append([])
            for ind in range(len(popt)):
                val = popt[ind]
                err = np.sqrt(pcov[ind, ind])
                val_col = table_popt[ind]
                err_col = table_pste[ind]
                val_col.append(round(val, 6))
                err_col.append(round(err, 6))

        # Close the plots so they don't clutter everything up
        # plt.close("all")

    ### Report the fit parameters

    if do_print:
        for ind in range(len(table_popt)):
            print()
            print(table_popt[ind])
            print()
            print(table_pste[ind])
            print()


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


def cambria_test4(temp, zfs0, A1, Theta1):

    ret_val = zfs0
    for ind in range(1):
        adj_ind = ind + 1
        ret_val += eval(f"A{adj_ind}") * bose(eval(f"Theta{adj_ind}"), temp)

    return ret_val


# endregion

# region Main plots


def main():

    # temp_range = [-10, 1000]
    # y_range = [2.74, 2.883]
    # temp_range = [-10, 720]
    # y_range = [2.80, 2.883]
    temp_range = [-10, 310]
    y_range = [2.8685, 2.8785]
    # temp_range = [280, 320]
    # y_range = [2.867, 2.873]
    # temp_range = [-10, 310]
    # y_range = [-0.0012, 0.0012]

    plot_data = True
    condense_data = True
    plot_residuals = False
    hist_residuals = False  # Must specify nv_to_plot down below
    separate_samples = False
    separate_nvs = False
    plot_prior_models = True
    desaturate_prior = False
    plot_new_model = True

    skip_lambda = lambda point: point["Skip"] or point["Sample"] != "Wu"

    ###

    min_temp, max_temp = temp_range
    min_temp = 0.1 if min_temp <= 0 else min_temp
    temp_linspace = np.linspace(min_temp, max_temp, 1000)
    fig, ax = plt.subplots()

    data_points = get_data_points(skip_lambda)

    if condense_data:
        condensed_data_points = []
        setpoint_temp_set = []
        for point in data_points:
            setpoint_temp_set.append(point["Setpoint temp (K)"])
        setpoint_temp_set = list(set(setpoint_temp_set))
        setpoint_temp_set.sort()
        for temp in setpoint_temp_set:
            monitor_temps = []
            zfss = []
            zfs_errors = []
            for point in data_points:
                if point["Setpoint temp (K)"] == temp:
                    monitor_temps.append(point["Monitor temp (K)"])
                    zfss.append(point["ZFS (GHz)"])
                    zfs_errors.append(point["ZFS error (GHz)"])
            sq_zfs_errors = [err**2 for err in zfs_errors]
            sum_sq_errors = np.sum(sq_zfs_errors)
            condensed_error = np.sqrt(sum_sq_errors) / len(sq_zfs_errors)
            new_point = {
                "Setpoint temp (K)": temp,
                "Monitor temp (K)": np.average(monitor_temps),
                "ZFS (GHz)": np.average(zfss, weights=zfs_errors),
                # + 0.0006,  # MCC
                "ZFS error (GHz)": condensed_error,
                "Sample": "Wu",
                "NV": "",
            }
            condensed_data_points.append(new_point)
        data_points = condensed_data_points

    zfs_list = []
    zfs_err_list = []
    temp_list = []
    nv_names = []
    nv_samples = []
    data_colors = {}
    data_color_options = kpl.data_color_cycler.copy()
    data_labels = {}
    for el in data_points:
        zfs = el["ZFS (GHz)"]
        monitor_temp = el["Monitor temp (K)"]
        if zfs == "" or monitor_temp == "":
            continue
        # if not (min_temp <= reported_temp <= max_temp):
        # if not (150 <= reported_temp <= 500):
        #     continue
        temp_list.append(monitor_temp)
        zfs_list.append(zfs)
        zfs_err = el["ZFS error (GHz)"]
        zfs_err_list.append(zfs_err)
        sample = el["Sample"]
        nv_samples.append(sample)
        el_nv = el["NV"]
        name = f"{sample}-{el_nv}"
        nv_names.append(name)
        if separate_samples:
            data_colors_keys = list(data_colors.keys())
            data_colors_samples = [el.split("-")[0] for el in data_colors_keys]
            if sample not in data_colors_samples:
                data_colors[name] = data_color_options.pop(0)
            data_labels[name] = sample
        elif separate_nvs:
            if name not in data_colors:
                data_colors[name] = data_color_options.pop(0)
            data_labels[name] = name
        else:
            data_colors[name] = KplColors.RED
    nv_names_set = list(set(nv_names))
    nv_names_set.sort()

    # zfs_list.extend(toyli_zfss)
    # temp_list.extend(toyli_temps)
    # nv_names.extend(["Toyli"] * len(toyli_temps))
    # zfs_err_list.extend([None] * len(toyli_temps))

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
        -8e-2,
        # -4e-1,
        65,
        # 165,
        # 6.5,
    ]
    fit_func = cambria_test
    # fit_func = cambria_test4
    popt, pcov = curve_fit(
        fit_func,
        temp_list,
        zfs_list,
        sigma=zfs_err_list,
        absolute_sigma=True,
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
    ssr = 0
    num_points = len(temp_list)
    num_params = len(guess_params)
    if None not in zfs_err_list:
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
            name = nv_names[ind]
            if plot_residuals:
                val = zfs_list[ind] - cambria_lambda(temp)
            else:
                val = zfs_list[ind]
            val_err = zfs_err_list[ind]
            label = None
            color = KplColors.DARK_GRAY
            if name in data_colors:
                color = data_colors[name]
            if separate_samples or separate_nvs:
                label = data_labels[name]
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
                label=label,
            )
            # ax.legend(loc="lower right")

    if hist_residuals:
        residuals = {}
        for el in nv_names_set:
            residuals[el] = []
        for ind in range(len(zfs_list)):
            name = nv_names[ind]
            temp = temp_list[ind]
            # if not (150 <= temp <= 500):
            #     continue
            # val = (zfs_list[ind] - cambria_lambda(temp)) / zfs_err_list[ind]
            val = zfs_list[ind] - cambria_lambda(temp)
            residuals[name].append(val)
        nv_to_plot = nv_names_set[4]
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
        color = data_colors[nv_to_plot]
        kpl.plot_line(ax, x_vals, y_vals, color=color)
        ax.fill_between(x_vals, y_vals, color=kpl.lighten_color_hex(color))
        ylim = max(y_vals) + 1

    if plot_new_model:
        color = KplColors.BLUE
        # color = "#0f49bd"
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

    if plot_prior_models:
        prior_model_colors = [
            KplColors.GREEN,
            KplColors.PURPLE,
            KplColors.RED,
            KplColors.ORANGE,
        ]
        prior_model_zorder = 2
        if desaturate_prior:
            prior_model_colors = [
                kpl.lighten_color_hex(el) for el in prior_model_colors
            ]
            prior_model_zorder = -5
        kpl.plot_line(
            ax,
            temp_linspace,
            sub_room_zfs_from_temp(temp_linspace),
            label="Chen",
            color=prior_model_colors[0],
            zorder=prior_model_zorder,
        )
        # print(super_room_zfs_from_temp(700))
        # return
        kpl.plot_line(
            ax,
            temp_linspace,
            super_room_zfs_from_temp(temp_linspace),
            label="Toyli",
            color=prior_model_colors[1],
            zorder=prior_model_zorder,
        )
        # kpl.plot_line(
        #     ax,
        #     temp_linspace,
        #     zfs_from_temp_barson(temp_linspace),
        #     label="Barson",
        #     color=prior_model_colors[2],
        #     zorder=prior_model_zorder,
        # )
        # kpl.plot_line(
        #     ax,
        #     temp_linspace,
        #     zfs_from_temp_li(temp_linspace),
        #     label="Li",
        #     color=prior_model_colors[3],
        #     zorder=prior_model_zorder,
        # )

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
        ax.set_xlabel(r"Temperature $\mathit{T}$ (K)")
        ax.set_ylabel("D (GHz)")
        ax.set_xlim(*temp_range)
        ax.set_ylim(*y_range)
    kpl.tight_layout(fig)

    # fig, ax = plt.subplots()
    # chen_proposed_diff = lambda temp: sub_room_zfs_from_temp(temp) - cambria_lambda(temp)
    # kpl.plot_line(
    #     ax,
    #     temp_linspace,
    #     1000 * chen_proposed_diff(temp_linspace)
    # )
    # ax.set_xlabel(r"Temperature $\mathit{T}$ (K)")
    # ax.set_ylabel("Chen - proposed (MHz)")
    # kpl.tight_layout(fig)


# endregion

if __name__ == "__main__":

    # print(cambria_fixed(15))
    # sys.exit()

    # calc_zfs_from_compiled_data()

    # kpl.init_kplotlib()

    # main()
    refit_experiments()

    # plt.show(block=True)
