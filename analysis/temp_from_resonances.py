# -*- coding: utf-8 -*-
"""
Get the NV temp based on the ZFS, using numbers from: 'Temperature dependent 
energy level shifts of nitrogen-vacancy centers in diamond'

Created on Fri Mar  5 12:42:32 2021

@author: matth
"""


# %% Imports


import numpy as np
from numpy.core.shape_base import block
from scipy.optimize import root_scalar
from majorroutines.pulsed_resonance import return_res_with_error
import utils.tool_belt as tool_belt
from majorroutines.spin_echo import zfs_cost_func
from scipy.optimize import minimize_scalar
import time
from figures.relaxation_temp_dependence.temp_dependence_fitting import (
    get_data_points,
    nominal_temp_column_title,
    reported_temp_column_title,
    low_res_file_column_title,
    high_res_file_column_title,
    bose,
)
import matplotlib.pyplot as plt
from utils import common
from utils import kplotlib as kpl
from scipy.optimize import curve_fit
from numpy import inf

bad_zfs_temps = 350


# %% Functions


def process_res_list():

    nominal_temps = []
    resonances = []

    for ind in range(len(resonances)):
        nominal_temp = nominal_temps[ind]
        res_pair = resonances[ind]
        print("Nominal temp: {}".format(nominal_temp))
        main_files(res_pair)
        print()


def process_temp_dep_res_files():

    file_name = "compiled_data"
    home = common.get_nvdata_dir()
    path = home / "paper_materials/relaxation_temp_dependence"

    data_points = get_data_points(path, file_name)
    nominal_temps = []
    resonances = []
    for el in data_points:
        if el[low_res_file_column_title] == "":
            continue
        # if float(el[nominal_temp_column_title]) not in [5.5, 50]:
        #     continue
        nominal_temps.append(el[nominal_temp_column_title])
        resonances.append(
            [el[low_res_file_column_title], el[high_res_file_column_title]]
        )

    for ind in range(len(resonances)):
        nominal_temp = nominal_temps[ind]
        res_pair = resonances[ind]
        # print("Nominal temp: {}".format(nominal_temp))
        try:
            main_files(res_pair)
        except Exception as exc:
            print(exc)
        # print()


def expt_guess(temp):
    if type(temp) in [list, np.ndarray]:
        ret_vals = [expt_guess_sub(el) for el in temp]
        return ret_vals
    else:
        return expt_guess_sub(temp)


def expt_guess_sub(temp):
    if temp < 300:  # Chen
        return sub_room_zfs_from_temp(temp)
    else:  # Toyli
        return super_room_zfs_from_temp(temp)


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
            if val < bad_zfs_temps:
                zfs = sub_room_zfs_from_temp(val)
            else:
                zfs = super_room_zfs_from_temp(val)
            ret_vals.append(zfs)
        ret_vals = np.array(ret_vals)
        return ret_vals
    else:
        if temp < bad_zfs_temps:
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

    return zfs_from_temp_barson_free(
        temp, zfs0, X1, X2, X3, Theta1, Theta2, Theta3
    )


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

    return fractional_thermal_expansion_free(
        temp, X1, X2, X3, Theta1, Theta2, Theta3
    )


def fractional_thermal_expansion_free(
    temp, X1, X2, X3, Theta1, Theta2, Theta3
):

    dV_over_V_partial = lambda X, Theta, T: (X * Theta) / (
        np.exp(Theta / T) - 1
    )
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
        + (-(A * B * dV_over_V(T)) + (b4 * T**4 + b5 * T**5 + b6 * T**6))
        / 1000
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
    Theta2 = 160

    ret_val = zfs0
    for ind in range(2):
        adj_ind = ind + 1
        ret_val += eval(f"A{adj_ind}") * bose(eval(f"Theta{adj_ind}"), temp)

    A3 = -14.6 * 442 / 1000  # (MHz/GPa) * (GPa/strain)
    ret_val += A3 * fractional_thermal_expansion(temp)

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


def cambria_test4(temp, zfs0, A1):

    Theta1 = 65
    Theta2 = 150
    A3 = -14.6 * 442 / 1000  # (MHz/GPa) * (GPa/strain)

    ret_val = zfs0
    ret_val += A1 * bose(Theta1, temp)
    # ret_val += A3 * fractional_thermal_expansion(temp)

    # Constrain A2 to match Toyli's 700 K value of 2.81461 GHz
    test_temp = 700
    A2 = (
        super_room_zfs_from_temp(test_temp)
        - zfs0
        - A1 * bose(Theta1, test_temp)
        # - A3 * fractional_thermal_expansion(test_temp)
    ) / bose(Theta2, test_temp)
    ret_val += A2 * bose(Theta2, temp)

    return ret_val


# def get_fake_data()
# expt_guess(temp_linspace)


def experimental_zfs_versus_t(path, file_name):

    # print(super_room_zfs_from_temp(700))
    # return
    temp_range = [-10, 1000]
    y_range = [2.74, 2.883]
    # temp_range = [-10, 510]
    # y_range = [2.843, 2.881]
    plot_data = False
    plot_prior_models = False
    plot_my_model = True

    min_temp, max_temp = temp_range
    min_temp = 0.1 if min_temp <= 0 else min_temp
    temp_linspace = np.linspace(min_temp, max_temp, 1000)
    csv_file_path = path / "{}.csv".format(file_name)
    fig, ax = plt.subplots(figsize=kpl.figsize)

    filter_temp_range = [0, 300]
    data_points = get_data_points(path, file_name)
    filtered_data_points = []
    for el in data_points:
        el_temp = el["Reported temp (K)"]
        if filter_temp_range[0] < el_temp < filter_temp_range[1]:
            filtered_data_points.append(el)
    data_points = filtered_data_points
    # data_points = get_fake_data(path, file_name)
    zfs_list = []
    zfs_err_list = []
    temp_list = []
    for el in data_points:
        if el[low_res_file_column_title] == "":
            continue
        reported_temp = el[reported_temp_column_title]
        # if not (min_temp <= reported_temp <= max_temp):
        # if not (min_temp <= reported_temp <= 300):
        # continue
        temp_list.append(reported_temp)
        low_res_file = el[low_res_file_column_title]
        high_res_file = el[high_res_file_column_title]
        resonances = []
        res_errs = []
        for f in [low_res_file, high_res_file]:
            data = tool_belt.get_raw_data(f)
            res, res_err = return_res_with_error(data)
            resonances.append(res)
            res_errs.append(res_err)
        zfs = (resonances[0] + resonances[1]) / 2
        zfs_err = np.sqrt(res_errs[0] ** 2 + res_errs[1] ** 2) / 2
        zfs_list.append(zfs)
        zfs_err_list.append(zfs_err)

    color = kpl.KplColors.PURPLE.value
    facecolor = kpl.lighten_color_hex(color)
    if plot_data:
        # ax.errorbar(
        #     temp_list,
        #     zfs_list,
        #     zfs_err_list,
        #     linestyle="None",
        #     marker="o",
        #     ms=kpl.marker_size,
        #     color=color,
        #     markerfacecolor=facecolor,
        #     lw=kpl.line_width,
        #     markeredgewidth=kpl.line_width,
        # )
        kpl.plot_points(
            ax, temp_list, zfs_list, yerr=zfs_err_list, color=color
        )

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
        # 65,
        # 165,
        # 6.5,
    ]
    # fit_func = cambria_test
    fit_func = cambria_test4
    popt, pcov = curve_fit(
        fit_func,
        temp_list,
        zfs_list,
        p0=guess_params,
        sigma=zfs_err_list,
        absolute_sigma=True,
        # bounds=((-inf, -inf, -inf, -inf, -inf, -inf), (inf, 0, 0, 0, 0, 0)),
    )
    print(popt)
    print(np.sqrt(np.diag(pcov)))
    cambria_lambda = lambda temp: fit_func(
        temp,
        *popt,
        # *guess_params,
    )
    ssr = 0
    num_points = len(temp_list)
    num_params = len(guess_params)
    for temp, zfs, zfs_err in zip(temp_list, zfs_list, zfs_err_list):
        calc_zfs = cambria_lambda(temp)
        ssr += ((zfs - calc_zfs) / zfs_err) ** 2
    dof = num_points - num_params
    red_chi_sq = ssr / dof
    print(red_chi_sq)

    ### Prior models

    if plot_prior_models:
        kpl.plot_line(
            ax,
            temp_linspace,
            sub_room_zfs_from_temp(temp_linspace),
            label="Chen",
        )
        # print(super_room_zfs_from_temp(700))
        # return
        kpl.plot_line(
            ax,
            temp_linspace,
            super_room_zfs_from_temp(temp_linspace),
            label="Toyli",
        )
        kpl.plot_line(
            ax,
            temp_linspace,
            zfs_from_temp_barson(temp_linspace),
            label="Barson",
        )
        kpl.plot_line(
            ax, temp_linspace, zfs_from_temp_li(temp_linspace), label="Li"
        )

    # Plot mine last for the reveal
    if plot_my_model:
        kpl.plot_line(
            ax,
            temp_linspace,
            cambria_lambda(temp_linspace),
            label="Proposed",
            color=color,
        )

    # Generated experimental
    kpl.plot_line(
        ax,
        temp_linspace,
        expt_guess(temp_linspace),
        label="expt",
        color="black",
        linestyle="dotted",
    )

    ### Plot wrap up
    if plot_prior_models:
        ax.legend(loc="lower left")
        # ax.legend(bbox_to_anchor=(0.37, 0.46))
        # ax.legend(bbox_to_anchor=(0.329, 0.46))
    ax.set_xlabel(r"Temperature $\mathit{T}$ (K)")
    ax.set_ylabel("D (GHz)")
    ax.set_xlim(*temp_range)
    ax.set_ylim(*y_range)
    kpl.tight_layout(fig)


# %% Main


def main_files(files, mag_B=None, theta_B_deg=None):

    resonances = []
    res_errs = []

    for ind in range(2):
        file = files[ind]
        data = tool_belt.get_raw_data(file)
        res, res_err = return_res_with_error(data)
        resonances.append(res)
        res_errs.append(res_err)

    return main_res(resonances, res_errs, mag_B, theta_B_deg)


def main_res(resonances, res_errs, mag_B=None, theta_B_deg=None):

    if mag_B is not None:
        theta_B = theta_B_deg * (np.pi / 180)
        args = (mag_B, theta_B, *resonances)
        result = minimize_scalar(
            zfs_cost_func, bounds=(2.83, 2.88), args=args, method="bounded"
        )
        zfs = result.x
        zfs_err = 0
    else:
        zfs = (resonances[0] + resonances[1]) / 2
        zfs_err = np.sqrt(res_errs[0] ** 2 + res_errs[1] ** 2) / 2

    return main(zfs, zfs_err)


def main(zfs, zfs_err=None, no_print=None):

    # func_to_invert = zfs_from_temp_barson
    func_to_invert = zfs_from_temp

    x0 = 199
    x1 = 201

    next_to_zero = 1e-5

    zfs_diff = lambda temp: func_to_invert(temp) - zfs
    if zfs_diff(next_to_zero) < 0:
        temp_mid = 0
    else:
        results = root_scalar(zfs_diff, x0=x0, x1=x1)
        temp_mid = results.root
        # print(zfs_diff(results.root))
    # x_vals = np.linspace(0, 2, 100)
    # plt.plot(x_vals, zfs_diff(x_vals))
    # print(zfs)
    # plt.show(block=True)

    temp_err = None
    if zfs_err is not None:
        zfs_lower = zfs - zfs_err
        zfs_diff = lambda temp: func_to_invert(temp) - zfs_lower
        if zfs_diff(next_to_zero) < 0:
            temp_higher = 0
        else:
            results = root_scalar(zfs_diff, x0=x0, x1=x1)
            temp_higher = results.root

        zfs_higher = zfs + zfs_err
        zfs_diff = lambda temp: func_to_invert(temp) - zfs_higher
        # print(zfs_diff(50))
        if zfs_diff(next_to_zero) < 0:
            temp_lower = 0
        else:
            results = root_scalar(zfs_diff, x0=x0, x1=x1)
            temp_lower = results.root

        if not no_print:
            print("{}\t{}\t{}".format(temp_lower, temp_mid, temp_higher))
        temp_err = ((temp_mid - temp_lower) + (temp_higher - temp_mid)) / 2
        return temp_mid, temp_err
    else:
        if not no_print:
            print(temp_mid)

    return temp_mid

    # print("T: [{}\t{}\t{}]".format(temp_lower, temp_mid, temp_higher))
    # temp_error = np.average([temp_mid - temp_lower, temp_higher - temp_mid])
    # print("T: [{}\t{}]".format(temp_mid, temp_error))


# %% Run the file


if __name__ == "__main__":

    # files = [
    #     "2022_07_06-17_07_38-hopper-search",
    #     "2022_07_06-17_36_45-hopper-search",
    # ]
    # files = [
    #     "2022_07_06-18_06_22-hopper-search",
    #     "2022_07_06-18_35_47-hopper-search",
    # ]

    # files = [
    #     "2022_07_02-20_32_55-hopper-search",
    #     "2022_07_02-21_02_41-hopper-search",
    # ]
    # files = [
    #     "2022_07_02-21_32_49-hopper-search",
    #     "2022_07_02-22_02_46-hopper-search",
    # ]

    # files = [
    #     "2022_06_26-22_31_49-hopper-search",
    #     "2022_06_26-22_47_13-hopper-search",
    # ]

    # main_files(files)

    # zfs = (2.80437571982632 + 2.9361011568838298) / 2
    # zfs_err = np.sqrt(2.3665998318251086e-05**2 + 2.7116675293791154e-05**2) / 2
    # zfs = (2.804495746863994 + 2.936131974306111) / 2
    # zfs_err = (
    #     np.sqrt(2.6001293985452265e-05 ** 2 + 2.9769666707425683e-05 ** 2) / 2
    # )
    # main(zfs, zfs_err)

    # process_temp_dep_res_files()

    #    print(zfs_from_temp(280))

    # temp = 0
    # print(zfs_from_temp(temp))
    # print(zfs_from_temp(temp) - zfs_from_temp_barson(temp))

    # plt.ion()

    # temps = np.linspace(10, 700, 1000)
    # plt.plot(temps, zfs_from_temp_barson(temps))
    # plt.plot(temps, zfs_from_temp(temps))
    # # fig, ax = plt.subplots()
    # # ax.plot(temps, sub_room_zfs_from_temp(temps), label='sub')
    # # ax.plot(temps, super_room_zfs_from_temp(temps), label='super')
    # # ax.legend()

    kpl.init_kplotlib()

    home = common.get_nvdata_dir()
    path = home / "paper_materials/relaxation_temp_dependence"
    file_name = "compiled_data"

    experimental_zfs_versus_t(path, file_name)

    plt.show(block=True)
