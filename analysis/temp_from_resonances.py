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


# %% Functions


def process_res_list():
    nominal_temps = [
        275,
        200,
        262.5,
        225,
        212.5,
        237.5,
        287.5,
        300,
        250,
        150,
        85,
        175,
        125,
        295,
        350,
        400,
    ]

    resonances = [
        [
            "2021_04_30-20_20_44-hopper-nv1_2021_03_16",
            "2021_04_30-20_25_01-hopper-nv1_2021_03_16",
        ],  # 275
        [
            "2021_05_02-19_46_43-hopper-nv1_2021_03_16",
            "2021_05_02-19_51_00-hopper-nv1_2021_03_16",
        ],  # 200
        [
            "2021_05_05-16_52_20-hopper-nv1_2021_03_16",
            "2021_05_05-16_58_19-hopper-nv1_2021_03_16",
        ],  # 262.5
        [
            "2021_05_06-10_57_36-hopper-nv1_2021_03_16",
            "2021_05_06-11_03_36-hopper-nv1_2021_03_16",
        ],  # 225
        [
            "2021_05_07-11_29_42-hopper-nv1_2021_03_16",
            "2021_05_07-11_35_40-hopper-nv1_2021_03_16",
        ],  # 212.5
        [
            "2021_05_09-00_28_20-hopper-nv1_2021_03_16",
            "2021_05_09-00_34_24-hopper-nv1_2021_03_16",
        ],  # 237.5
        [
            "2021_05_10-09_47_22-hopper-nv1_2021_03_16",
            "2021_05_10-09_53_17-hopper-nv1_2021_03_16",
        ],  # 287.5
        [
            "2021_05_11-08_21_59-hopper-nv1_2021_03_16",
            "2021_05_11-08_27_51-hopper-nv1_2021_03_16",
        ],  # 300
        [
            "2021_05_11-23_13_54-hopper-nv1_2021_03_16",
            "2021_05_11-23_19_48-hopper-nv1_2021_03_16",
        ],  # 250
        [
            "2021_05_12-23_08_06-hopper-nv1_2021_03_16",
            "2021_05_12-23_13_52-hopper-nv1_2021_03_16",
        ],  # 150
        [
            "2021_05_20-23_57_25-hopper-nv1_2021_03_16",
            "2021_05_20-23_52_42-hopper-nv1_2021_03_16",
        ],  # 85
        [
            "2021_06_05-16_57_26-hopper-nv1_2021_03_16",
            "2021_06_05-17_03_55-hopper-nv1_2021_03_16",
        ],  # 175
        [
            "2021_06_09-22_52_51-hopper-nv1_2021_03_16",
            "2021_06_09-22_58_18-hopper-nv1_2021_03_16",
        ],  # 125
        [
            "2021_07_07-22_21_10-hopper-nv1_2021_03_16",
            "2021_07_07-22_22_55-hopper-nv1_2021_03_16",
        ],  # 295
        [
            "2021_07_08-18_16_31-hopper-nv1_2021_03_16",
            "2021_07_08-18_19_47-hopper-nv1_2021_03_16",
        ],  # 350
        [
            "2021_07_09-18_04_22-hopper-nv1_2021_03_16",
            "2021_07_09-18_07_34-hopper-nv1_2021_03_16",
        ],  # 400
    ]

    for ind in range(len(resonances)):
        nominal_temp = nominal_temps[ind]
        res_pair = resonances[ind]
        print("Nominal temp: {}".format(nominal_temp))
        main_files(res_pair)
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


# %% Main


def main_files(files, mag_B=None, theta_B_deg=None):

    resonances = []
    res_errs = []
    
    path_from_nvdata = "pc_hahn/branch_master/pulsed_resonance/2021_09"

    for ind in range(2):
        file = files[ind]
        data = tool_belt.get_raw_data(file, path_from_nvdata)
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

    zfs_diff = lambda temp: zfs_from_temp(temp) - zfs
    results = root_scalar(zfs_diff, x0=50, x1=500)
    temp_mid = results.root

    zfs_lower = zfs - zfs_err
    zfs_diff = lambda temp: zfs_from_temp(temp) - zfs_lower
    results = root_scalar(zfs_diff, x0=50, x1=500)
    temp_higher = results.root

    zfs_higher = zfs + zfs_err
    zfs_diff = lambda temp: zfs_from_temp(temp) - zfs_higher
    results = root_scalar(zfs_diff, x0=50, x1=500)
    temp_lower = results.root

    print("T: [{}\t{}\t{}]".format(temp_lower, temp_mid, temp_higher))
    temp_error = numpy.average([temp_mid - temp_lower, temp_higher - temp_mid])
    print("T: [{}\t{}]".format(temp_mid, temp_error))


# %% Run the file


if __name__ == "__main__":

    files = [
        "2021_09_15-01_16_57-hopper-search",
        "2021_09_15-01_25_47-hopper-search",
    ]

    main_files(files)

    # process_res_list()

#    print(zfs_from_temp(280))

# temps = numpy.linspace(5,500,1000)
# # plt.plot(temps, zfs_from_temp(temps))
# fig, ax = plt.subplots()
# ax.plot(temps, sub_room_zfs_from_temp(temps), label='sub')
# ax.plot(temps, super_room_zfs_from_temp(temps), label='super')
# ax.legend()
