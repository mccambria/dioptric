# -*- coding: utf-8 -*-
"""
Get the NV temp based on the ZFS, using numbers from: 'Temperature dependent 
energy level shifts of nitrogen-vacancy centers in diamond'

Created on Fri Mar  5 12:42:32 2021

@author: matth
"""


# %% Imports


import numpy
from scipy.optimize import root_scalar
from majorroutines.pulsed_resonance import return_res_with_error
import utils.tool_belt as tool_belt
from majorroutines.spin_echo import zfs_cost_func
from scipy.optimize import minimize_scalar
import time
from figures.relaxation_temp_dependence.revision1.temp_dependence_fitting import (
    get_data_points,
    nominal_temp_column_title,
    low_res_file_column_title,
    high_res_file_column_title,
)
import matplotlib.pyplot as plt
from utils import common


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
        # if int(el[nominal_temp_column_title]) != 50:
        #     continue
        nominal_temps.append(el[nominal_temp_column_title])
        resonances.append(
            [el[low_res_file_column_title], el[high_res_file_column_title]]
        )

    for ind in range(len(resonances)):
        nominal_temp = nominal_temps[ind]
        res_pair = resonances[ind]
        print("Nominal temp: {}".format(nominal_temp))
        try:
            main_files(res_pair)
        except Exception as exc:
            print(exc)
        print()


def sub_room_zfs_from_temp(temp):
    coeffs = [2.87771, -4.625e-6, 1.067e-7, -9.325e-10, 1.739e-12, -1.838e-15]
    ret_val = 0
    for ind in range(6):
        ret_val += coeffs[ind] * (temp ** ind)
    return ret_val


def super_room_zfs_from_temp(temp):
    coeffs = [2.8697, 9.7e-5, -3.7e-7, 1.7e-10]
    coeff_errs = [0.0009, 0.6e-5, 0.1e-7, 0.1e-10]
    ret_val = 0
    for ind in range(4):
        ret_val += coeffs[ind] * (temp ** ind)
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
    if type(temp) in [list, numpy.ndarray]:
        ret_vals = []
        for val in temp:
            if val < 300:
                zfs = sub_room_zfs_from_temp(val)
            else:
                zfs = super_room_zfs_from_temp(val)
            ret_vals.append(zfs)
        ret_vals = numpy.array(ret_vals)
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

    X1 = 0.4369e-7  # 1 / K
    X2 = 15.7867e-7  # 1 / K
    X3 = 42.5598e-7  # 1 / K
    Theta1 = 200  # K
    Theta2 = 880  # K
    Theta3 = 2137.5  # K
    dV_over_V_partial = lambda X, Theta, T: (X * Theta) / (
        numpy.exp(Theta / T) - 1
    )
    dV_over_V = (
        lambda T: numpy.exp(
            3
            * (
                dV_over_V_partial(X1, Theta1, T)
                + dV_over_V_partial(X2, Theta2, T)
                + dV_over_V_partial(X3, Theta3, T)
            )
        )
        - 1
    )

    A = 14.6  # MHz /GPa
    B = 442  # GPa/strain
    b4 = -1.44e-9
    b5 = 3.1e-12
    b6 = -1.8e-15
    D_of_T = (
        lambda T: 2.877656435574434
        + (-(A * B * dV_over_V(T)) + (b4 * T ** 4 + b5 * T ** 5 + b6 * T ** 6))
        / 1000
    )
    # D_of_T = lambda T: -D_of_T_sub(1) + D_of_T_sub(T)
    if type(temp) in [list, numpy.ndarray]:
        ret_vals = []
        for val in temp:
            ret_vals.append(D_of_T(val))
        ret_vals = numpy.array(ret_vals)
        return ret_vals
    else:
        return D_of_T(temp)


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

    main_res(resonances, res_errs, mag_B, theta_B_deg)


def main_res(resonances, res_errs, mag_B=None, theta_B_deg=None):

    if mag_B is not None:
        theta_B = theta_B_deg * (numpy.pi / 180)
        args = (mag_B, theta_B, *resonances)
        result = minimize_scalar(
            zfs_cost_func, bounds=(2.83, 2.88), args=args, method="bounded"
        )
        zfs = result.x
        zfs_err = 0
    else:
        zfs = (resonances[0] + resonances[1]) / 2
        zfs_err = numpy.sqrt(res_errs[0] ** 2 + res_errs[1] ** 2) / 2

    main(zfs, zfs_err)


def main(zfs, zfs_err):

    func_to_invert = zfs_from_temp_barson

    zfs_diff = lambda temp: func_to_invert(temp) - zfs
    results = root_scalar(zfs_diff, x0=50, x1=500)
    temp_mid = results.root

    zfs_lower = zfs - zfs_err
    zfs_diff = lambda temp: func_to_invert(temp) - zfs_lower
    results = root_scalar(zfs_diff, x0=50, x1=500)
    temp_higher = results.root

    zfs_higher = zfs + zfs_err
    zfs_diff = lambda temp: func_to_invert(temp) - zfs_higher
    results = root_scalar(zfs_diff, x0=50, x1=500)
    temp_lower = results.root

    print("T: [{}\t{}\t{}]".format(temp_lower, temp_mid, temp_higher))
    temp_error = numpy.average([temp_mid - temp_lower, temp_higher - temp_mid])
    print("T: [{}\t{}]".format(temp_mid, temp_error))


# %% Run the file


if __name__ == "__main__":

    # files = [
    #     "2021_10_01-16_01_47-hopper-search",
    #     "2021_10_01-16_11_08-hopper-search",
    # ]

    # main_files(files)

    # process_temp_dep_res_files()

    #    print(zfs_from_temp(280))

    temp = 0
    print(zfs_from_temp(temp))
    # print(zfs_from_temp(temp) - zfs_from_temp_barson(temp))

    # plt.ion()

    # temps = numpy.linspace(10, 700, 1000)
    # plt.plot(temps, zfs_from_temp_barson(temps))
    # plt.plot(temps, zfs_from_temp(temps))
    # # fig, ax = plt.subplots()
    # # ax.plot(temps, sub_room_zfs_from_temp(temps), label='sub')
    # # ax.plot(temps, super_room_zfs_from_temp(temps), label='super')
    # # ax.legend()

    # plt.show(block=True)
