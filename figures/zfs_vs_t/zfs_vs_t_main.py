# -*- coding: utf-8 -*-
"""
Get the NV temp based on the ZFS, using numbers from: 'Temperature dependent 
energy level shifts of nitrogen-vacancy centers in diamond'

Created on Fri Mar  5 12:42:32 2021

@author: matth
"""


# region Import and constants


import numpy as np
from numpy.core.shape_base import block
from scipy.optimize import root_scalar
from majorroutines.pulsed_resonance import return_res_with_error
import utils.tool_belt as tool_belt
from majorroutines.spin_echo import zfs_cost_func
from scipy.optimize import minimize_scalar
import time
from figures.relaxation_temp_dependence.temp_dependence_fitting import bose
import matplotlib.pyplot as plt
from utils import common
from utils import kplotlib as kpl
from utils.kplotlib import KplColors
from scipy.optimize import curve_fit
from numpy import inf
import sys
import csv
import pandas as pd

toyli_digitized = [
    300,
    2.87,
    309.9858044201037,
    2.8690768841409784,
    320.04280071681194,
    2.868366259576263,
    330.32149670254546,
    2.8673945841666115,
    340.3583384820696,
    2.866304172245094,
    350.05837349046874,
    2.8655253065868678,
    360.1625766242179,
    2.8644972039180088,
    370.064695695292,
    2.8633133281175045,
    380.2362601832661,
    2.8622540708223165,
    390.13837925434024,
    2.8611013496481412,
    399.9731369711893,
    2.8600109377266243,
    410.00997875071346,
    2.858858216552449,
    420.0468205302376,
    2.857362794488654,
    430.4878304351117,
    2.856176937898957,
    440.3899495061858,
    2.8549015790086583,
    450.02262316036,
    2.8535619300765087,
    460.1268262941091,
    2.852066508012714,
    469.96158401095823,
    2.850633395201577,
    480.5373166242823,
    2.849387210148415,
    490.30471298690645,
    2.84789178808462,
    500.2068320579806,
    2.8465209845261414,
    510.04158977482973,
    2.844994407836017,
    520.2805156170289,
    2.843374367266906,
    530.452080105003,
    2.8417854813241243,
    540.354199176077,
    2.840258904634,
    550.1215955387013,
    2.838638864064889,
    560.1584373182253,
    2.837361524385398,
    570.3300018061993,
    2.8357103291899572,
    580.2321208772735,
    2.8341545786626967,
    590.4036853652476,
    2.8322813395045685,
    600.0363590194218,
    2.830756743603637,
    610.005839444721,
    2.829354785418829,
    619.9079585157951,
    2.8275789717180726,
    630.4163297748942,
    2.826052395027949,
    640.3184488459683,
    2.824556972964154,
    650.2879292712674,
    2.8227500046370686,
    660.1269697755595,
    2.821005345562641,
    669.8900833507407,
    2.8189160048094015,
    680.4658159640647,
    2.816922108724342,
    690.5700190978139,
    2.8151482758127777,
    700.472138168888,
    2.8134950998281454,
    710.0374504688373,
    2.812188586311517,
]
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


def get_data_points(override_skips=False):

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

            if override_skips or not point["Skip"]:
                data_points.append(point)

    return data_points


def calc_zfs_from_compiled_data():

    data_points = get_data_points(override_skips=True)
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


# endregion

# region Main plots


def main():

    # temp_range = [-10, 1000]
    # y_range = [2.74, 2.883]
    temp_range = [-10, 300]
    y_range = [2.868, 2.88]
    # temp_range = [-10, 250]
    # y_range = [-0.0015, 0.0015]

    plot_data = True
    plot_data_model_diff = False
    hist_deviations = False
    separate_samples = False
    separate_nvs = True
    plot_prior_models = False
    desaturate_prior = False
    plot_new_model = True

    ###

    min_temp, max_temp = temp_range
    min_temp = 0.1 if min_temp <= 0 else min_temp
    temp_linspace = np.linspace(min_temp, max_temp, 1000)
    fig, ax = plt.subplots(figsize=kpl.figsize)

    data_points = get_data_points()
    zfs_list = []
    zfs_err_list = []
    temp_list = []
    nv_names = []
    nv_samples = []
    data_colors = {}
    data_color_options = kpl.data_color_cycler.copy()
    for el in data_points:
        zfs = el["ZFS (GHz)"]
        reported_temp = el["Monitor temp (K)"]
        if zfs == "" or reported_temp == "":
            continue
        # if not (min_temp <= reported_temp <= max_temp):
        if not (min_temp <= reported_temp <= 295):
            continue
        temp_list.append(reported_temp)
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
        elif separate_nvs:
            if name not in data_colors:
                data_colors[name] = data_color_options.pop(0)
        else:
            data_colors[name] = KplColors.DARK_GRAY.value

    # zfs_list.extend(toyli_zfss)
    # temp_list.extend(toyli_temps)

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
        -4e-1,
        # 65,
        # 165,
        # 6.5,
    ]
    fit_func = cambria_test
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

    if plot_data:
        for ind in range(len(zfs_list)):
            temp = temp_list[ind]
            name = nv_names[ind]
            color = data_colors[name]
            if plot_data_model_diff:
                val = zfs_list[ind] - cambria_lambda(temp)
            else:
                val = zfs_list[ind]
            val_err = zfs_err_list[ind]
            kpl.plot_points(
                ax,
                temp,
                val,
                yerr=val_err,
                color=color,
                zorder=-1,
            )

    if hist_deviations:
        pass

    if plot_new_model:
        color = KplColors.BLUE.value
        # color = "#0f49bd"
        kpl.plot_line(
            ax,
            temp_linspace,
            cambria_lambda(temp_linspace),
            label="Proposed",
            color=color,
            zorder=10,
        )

    ### Prior models

    if plot_prior_models:
        prior_model_colors = [
            KplColors.GREEN.value,
            KplColors.PURPLE.value,
            KplColors.RED.value,
            KplColors.ORANGE.value,
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
            color=prior_model_colors.pop(),
            zorder=prior_model_zorder,
        )
        # print(super_room_zfs_from_temp(700))
        # return
        kpl.plot_line(
            ax,
            temp_linspace,
            super_room_zfs_from_temp(temp_linspace),
            label="Toyli",
            color=prior_model_colors.pop(),
            zorder=prior_model_zorder,
        )
        kpl.plot_line(
            ax,
            temp_linspace,
            zfs_from_temp_barson(temp_linspace),
            label="Barson",
            color=prior_model_colors.pop(),
            zorder=prior_model_zorder,
        )
        kpl.plot_line(
            ax,
            temp_linspace,
            zfs_from_temp_li(temp_linspace),
            label="Li",
            color=prior_model_colors.pop(),
            zorder=prior_model_zorder,
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


# endregion

if __name__ == "__main__":

    # print(cambria_fixed(15))
    # sys.exit()

    # calc_zfs_from_compiled_data()

    kpl.init_kplotlib()

    main()

    plt.show(block=True)
