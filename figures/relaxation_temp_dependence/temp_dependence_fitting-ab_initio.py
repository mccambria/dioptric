# -*- coding: utf-8 -*-
"""
Reproduce Jarmola 2012 temperature scalings

Created on Fri Jun 26 17:40:09 2020

@author: matth
"""


# %% Imports


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
import copy
import utils.kplotlib as kpl
from utils.tool_belt import presentation_round, presentation_round_latex
from utils.kplotlib import figsize, double_figsize
from ab_initio_rates import get_ab_initio_rates


# %% Constants


Boltzmann = 8.617e-2  # meV / K
# from scipy.constants import Boltzmann  # J / K

# Rate coefficients in s^-1 from Jarmola. Not accurate right now
# A_1 = 0.007  # Constant for S3
# A_2 = 5.10e2  # Orbach
# # A_2 = 1.7e3  # Test
# A_3 = 1.38e-11  # T^5
# # A_3 = 2.5e-11  # Test
# A_4 = 4.3e-6  # T^3
# A_7 = 2.55e-20

A_1 = 0.007  # Constant for S3
A_2 = 5.10e2  # Orbach
# A_2 = 1.7e3  # Test
A_3 = 1.38e-11  # T^5
# A_3 = 2.5e-11  # Test
A_4 = 4.3e-6  # T^3
A_7 = 2.55e-20

# Quasilocalized mode activation energy
quasi = 76.0  # meV, empirical fit
# quasi = 69.0  # meV, empirical fit
# quasi = 65.0  # meV, quasilocalized resonance
# quasi = 1.17e-20  # J

# marker_size = 7
# line_width = 3
marker_size = 7
line_width = 1.5
# line_width = 2.5
marker_edge_width = line_width

gamma_face_color = "#CC99CC"
gamma_edge_color = "#993399"
omega_face_color = "#FFCC33"
omega_edge_color = "#FF9933"
ratio_face_color = "#FB9898"
ratio_edge_color = "#EF2424"
# qubit_max_face_color = "#81bfeb"
# qubit_max_edge_color = "#1f77b4"
# qutrit_max_face_color = "#e5e667"
# qutrit_max_edge_color = "#bcbd22"
qutrit_color = "#bcbd22"
qubit_color = "#1f77b4"

sample_column_title = "Sample"
skip_column_title = "Skip"
no_x_errs = True

nominal_temp_column_title = "Nominal temp (K)"
reported_temp_column_title = "Reported temp (K)"
temp_column_title = "ZFS temp (K)"
temp_lb_column_title = "ZFS temp, lb (K)"
temp_ub_column_title = "ZFS temp, ub (K)"

low_res_file_column_title = "-1 resonance file"
high_res_file_column_title = "+1 resonance file"

omega_column_title = "Omega (s^-1)"
omega_err_column_title = "Omega err (s^-1)"
gamma_column_title = "gamma (s^-1)"
gamma_err_column_title = "gamma err (s^-1)"

# bad_zfs_temps = 300  # Below this consider zfs temps inaccurate


# %% Processes and sum functions


def bose(energy, temp):
    # For very low temps we can get divide by zero and overflow warnings.
    # Fortunately, numpy is smart enough to know what we mean when this
    # happens, so let's let numpy figure it out and suppress the warnings.
    old_settings = np.seterr(divide="ignore", over="ignore")
    # print(energy / (Boltzmann * temp))
    val = 1 / (np.exp(energy / (Boltzmann * temp)) - 1)
    # Return error handling to default state for other functions
    np.seterr(**old_settings)
    return val


def orbach(temp):
    """
    This is for quasilocalized phonons interacting by a Raman process, which
    reproduces an Orbach scaling even though it's not really an Orbach.
    process. As such, the proper scaling is
    n(omega)(n(omega)+1) approx n(omega) for omega << kT
    """
    # return A_2 * bose(quasi, temp) * (bose(quasi, temp) + 1)
    return A_2 * bose(quasi, temp)
    # return A_2 / (np.exp(quasi / (Boltzmann * temp)))


def orbach_free(temp, coeff, activation):
    return coeff * bose(activation, temp)


def raman(temp):
    return A_3 * (temp ** 5)


def test_T_cubed(temp):
    return A_4 * (temp ** 3)


def test_T_seventh(temp):
    return A_7 * (temp ** 7)


def orbach_T5_free(temp, coeff_orbach, activation, coeff_T5):
    # activation = 78
    # coeff_T5 = 0
    return (coeff_orbach * bose(activation, temp)) + (coeff_T5 * temp ** 5)


def orbach_T5_free_const(temp, coeff_orbach, activation, coeff_T5, const):
    full_scaling = True
    if full_scaling:
        n1 = bose(activation, temp)
        return const + (coeff_orbach * n1 * (n1 + 1)) + (coeff_T5 * temp ** 5)
    else:
        return (
            const
            + (coeff_orbach * bose(activation, temp))
            + (coeff_T5 * temp ** 5)
        )


def orbach_variable_exp_const(
    temp, coeff_orbach, activation, coeff_power, exp, const
):
    full_scaling = True
    if full_scaling:
        n1 = bose(activation, temp)
        return (
            const
            + (coeff_orbach * n1 * (n1 + 1))
            + (coeff_power * temp ** exp)
        )
    else:
        return (
            const
            + (coeff_orbach * bose(activation, temp))
            + (coeff_power * temp ** exp)
        )


def double_orbach(temp, coeff1, delta1, coeff2, delta2, const):
    full_scaling = True
    if full_scaling:
        n1 = bose(delta1, temp)
        n2 = bose(delta2, temp)
        return const + (coeff1 * n1 * (n1 + 1)) + (coeff2 * n2 * (n2 + 1))
    else:
        return (
            const
            + (coeff1 * bose(delta1, temp))
            + (coeff2 * bose(delta2, temp))
        )


def double_orbach_ratio(
    temp, orbach_coeff, gamma_to_omega_ratio, delta1, delta2, const
):
    full_scaling = True
    coeff = orbach_coeff * gamma_to_omega_ratio
    if full_scaling:
        n1 = bose(delta1, temp)
        n2 = bose(delta2, temp)
        return const + (coeff * n1 * (n1 + 1)) + (coeff * n2 * (n2 + 1))
    else:
        return (
            const + (coeff * bose(delta1, temp)) + (coeff * bose(delta2, temp))
        )


def triple_orbach(temp, coeff1, delta1, coeff2, delta2, coeff3, delta3, const):
    full_scaling = True
    if full_scaling:
        n1 = bose(delta1, temp)
        n2 = bose(delta2, temp)
        n3 = bose(delta3, temp)
        return (
            const
            + (coeff1 * n1 * (n1 + 1))
            + (coeff2 * n2 * (n2 + 1))
            + (coeff3 * n3 * (n3 + 1))
        )
    else:
        return (
            const
            + (coeff1 * bose(delta1, temp))
            + (coeff2 * bose(delta2, temp))
            + (coeff3 * bose(delta3, temp))
        )


def orbach_T5_free_linear(
    temp, coeff_orbach, activation, coeff_T5, coeff_linear
):
    return (
        (coeff_orbach * bose(activation, temp))
        + (coeff_T5 * temp ** 5)
        + (coeff_linear * temp)
    )


def orbach_T7_free(temp, coeff_orbach, activation, coeff_T7):
    return (coeff_orbach * bose(activation, temp)) + (coeff_T7 * temp ** 7)


def orbach_T3_free(temp, coeff_orbach, activation, coeff_T3):
    return (coeff_orbach * bose(activation, temp)) + (coeff_T3 * temp ** 3)


def T5_free(temp, coeff_T5):
    return coeff_T5 * temp ** 5


# %% Other functions


def get_past_results(res):

    omega_temps = None
    omega_rates = None
    gamma_temps = None
    gamma_rates = None

    # fmt: off
    if res == "redman":
        omega_temps = [476.4177416661098, 451.11715421405614, 427.14250537007416, 401.70168043525575, 372.6536407580296, 348.06445892052716, 320.67189366418336, 295.4432631793261, 249.08174616160218, 198.8186984331038, 168.76315171076862, 139.3829999492335, 119.91885637433246, 99.71082373901912,]
        omega_rates = [411.12308417220487,365.9018246860615,303.663725116546,257.9542650146412,209.14601567073342,165.66662339760202,114.10159115954427,82.33629032388878,42.87365275566925,16.10976746190582,8.195340538524343,3.380271780205849,1.362118143087146,0.488506838976792,]
    elif res == "takahashi":
        omega_temps = [39.708076869092565, 60.18108720206314, 79.61728330307751, 99.71082373901912, 149.1639405437168, 200.11702947793873, 301.43114785530264, ]
        omega_rates = [0.08310743505465269, 0.12642300635774165, 0.3287046673635948, 0.488506838976792, 2.0243218136492307, 9.647553749055826, 43.88457973623873, ]
    elif res == r"jarmola\_s2":
        omega_temps = [5.052927989240913, 9.986632403181524, 20.00811570715186, 29.90545574361996, 39.81280330854725, 60.32233775511956, 79.75794398477714, 120.85851898425719, 198.86531379873264, 252.5578517930507, 295.50030798309376, 325.09776992170424, 350.4582003215088, 388.22259119422284, 407.2346885164248, 479.68759119360243, ]
        omega_rates = [6.342069135479235, 6.801353446821833, 7.125875978477722, 6.801353446821833, 7.125875978477722, 6.644677189578252, 6.491610133029184, 7.293898524349615, 23.94164841827978, 63.71699176359265, 114.10159115954427, 131.22616761282828, 177.66398252665408, 229.58072627922334, 283.1578087632278, 430.7395774394513, ]
    elif res == r"jarmola\_s3":
        omega_temps = [6.427375637623035, 9.937587857375618, 19.916720690963206, 39.91505366735992, 49.63491901713573, 69.84021996713274, 78.96668063167132, 94.35441700897722, 119.05215177843596, 160.81553767084844, 202.87342606577826, 297.39307938738887, 324.9857097772839, 350.3373985191751, 375.0871532725971, 401.58536640626534, 424.1313933316525, 451.01762892382345, 479.5883763012125, ]
        omega_rates = [0.0024620666073597885, 0.0016566662339760237, 0.0031082387670171665, 0.0054378766564793, 0.004953857646692448, 0.022533998415870633, 0.031227604065269598, 0.14539680344353337, 0.6461422475342434, 3.226329393457812, 10.59017256160916, 55.402137213788336, 73.27975496207623, 99.21171472104378, 125.24994121458094, 158.12192963668738, 195.02272616248604, 252.01201997495744, 303.663725116546, ]
    elif res == r"jarmola\_s8":
        omega_temps = [9.950478138502218, 19.80060613286093, 29.800477508344034, 39.67414115937397, 49.680800221318, 59.72553858233379, 79.56130794778521, 119.89239992311232, 160.83993406671632, 200.15014817692008, 252.45338834977025, 293.3848300819487, 322.81046691031594, 352.7565006568838, 429.97927603430963, 479.60160376778975, 375.1233628312281, 398.8864136467382, ]
        omega_rates = [0.014813307038192704, 0.015886067264611867, 0.0178494025680193, 0.019593386907583058, 0.023609194738400396, 0.033489066108904834, 0.10014091119973172, 0.9381482177041631, 4.169123278855502, 12.760705818733683, 31.667336635581677, 60.81522929409823, 86.26491741142065, 111.47314222093364, 219.12529332802444, 318.15285902458135, 147.44421348369355, 177.66398252665408, ]
    elif res == r"lin":
        omega_temps = [300, 325, 350, 375, 400, 425, 450, 500, 550, 600]
        omega_rates = [0.09401709401709413, 0.14102564102564097, 0.170940170940171, 0.20512820512820507, 0.2435897435897436, 0.30341880341880345, 0.3547008547008548, 0.5256410256410258, 0.777777777777778, 1.153846153846154, ]
        omega_rates = [1000*el for el in omega_rates]
        gamma_temps = [300, 350, 400, 450, 500, 550, 600]
        gamma_rates = [0.19264214046822736, 0.24882943143812708, 0.33177257525083614, 0.38795986622073575, 0.4655518394648829, 0.6555183946488294, 0.7785953177257525, ]
        gamma_rates = [1000*el for el in gamma_rates]
    # fmt: on

    return omega_temps, omega_rates, gamma_temps, gamma_rates


def omega_calc(temp):
    popt = [421.88, 69.205, 1.1124e-11]
    return orbach_T5_free(temp, *popt)


def gamma_calc(temp):
    popt = [1357.2, 69.205, 9.8064e-12]
    return orbach_T5_free(temp, *popt)


def get_temp(point):
    # Return whatever is in the excel sheet, which has its own logic
    reported_temp = point[reported_temp_column_title]
    return reported_temp
    # nominal_temp = point[nominal_temp_column_title]
    # if nominal_temp <= bad_zfs_temps:
    #     temp = nominal_temp
    # else:
    #     temp = point[temp_column_title]
    #     if temp == "":
    #         temp = point[nominal_temp_column_title]
    # return temp


def get_temp_bounds(point):
    if temp_lb_column_title == nominal_temp_column_title:
        return None
    nominal_temp = point[nominal_temp_column_title]
    # if nominal_temp < bad_zfs_temps:
    if nominal_temp < 295:
        return [nominal_temp - 3, nominal_temp + 3]
    else:
        lower_bound = point[temp_lb_column_title]
        if lower_bound == "":
            return None
        upper_bound = point[temp_ub_column_title]
        return [lower_bound, upper_bound]


def get_temp_error(point):
    temp = get_temp(point)
    temp_bounds = get_temp_bounds(point)
    if temp_bounds is None:
        return 1.0
    else:
        return np.average([temp - temp_bounds[0], temp_bounds[1] - temp])


def simultaneous_test_lambda(
    temps,
    beta,
    omega_hopper_fit_func,
    omega_wu_fit_func,
    gamma_hopper_fit_func,
    gamma_wu_fit_func,
    sample_break,
):
    """
    Lambda variation of simultaneous_test
    """

    ret_vals = []
    num_vals = len(temps)
    for ind in range(num_vals):
        temp_val = temps[ind]
        if ind < sample_break:
            sample = "hopper"
        else:
            sample = "wu"
        # Omegas are even indexed, gammas are odd indexed
        if ind % 2 == 0:
            rate = "omega"
        else:
            rate = "gamma"
        fit_func = eval("{}_{}_fit_func".format(rate, sample))
        ret_vals.append(fit_func(temp_val, beta))

    return np.array(ret_vals)


def fit_simultaneous(data_points, fit_mode=None):

    # To fit to Omega and gamma simultaneously, set up a combined list of the
    # rates. Parity determines which rate is where. Even is Omega, odd is
    # gamma.
    temps = []
    temp_errors = []
    combined_rates = []
    combined_errs = []
    sample_breaks = []
    for sample in ["Hopper", "Wu"]:
        for point in data_points:
            # Crash if we're trying to work with incomplete data
            if (point[omega_column_title] is None) or (
                point[gamma_column_title] is None
            ):
                crash = 1 / 0
            if point[sample_column_title] != sample:
                continue
            temp = get_temp(point)
            temps.append(temp)
            temp_bounds = get_temp_bounds(point)
            temp_error = get_temp_error(point)
            temp_errors.append(temp_error)
            combined_rates.append(point[omega_column_title])
            combined_errs.append(point[omega_err_column_title])
            temps.append(temp)
            temp_errors.append(temp_error)
            combined_rates.append(point[gamma_column_title])
            combined_errs.append(point[gamma_err_column_title])
        sample_breaks.append(len(combined_rates))

    # region DECLARE FIT FUNCTIONS HERE

    if fit_mode is None:
        # fit_mode = "variable_exp"
        # fit_mode = "T5_fixed_coeffs"
        # fit_mode = "T5"
        fit_mode = "double_orbach"
        # fit_mode = "other"

    # Varible exponent
    if fit_mode == "variable_exp":
        init_params = (
            1.38e-11,
            1.38e-11,
            5,
            510,
            2000,
            72.0,
            0.01,
            0.01,
            0.07,
            0.15,
        )
        omega_hopper_fit_func = lambda temp, beta: orbach_variable_exp_const(
            temp, beta[3], beta[5], beta[0], beta[2], beta[6]
        )
        omega_wu_fit_func = lambda temp, beta: orbach_variable_exp_const(
            temp, beta[3], beta[5], beta[0], beta[2], beta[7]
        )
        gamma_hopper_fit_func = lambda temp, beta: orbach_variable_exp_const(
            temp, beta[4], beta[5], beta[1], beta[2], beta[8]
        )
        gamma_wu_fit_func = lambda temp, beta: orbach_variable_exp_const(
            temp, beta[4], beta[5], beta[1], beta[2], beta[9]
        )
        beta_desc = [
            "Omega exp coeff (K^-exp s^-1)",
            "gamma exp coeff (K^-exp s^-1)",
            "Power law exp",
            "Omega Orbach coeff (s^-1)",
            "gamma Orbach coeff (s^-1)",
            "Orbach Delta (meV)",
            "Omega Hopper constant (s^-1)",
            "Omega Wu constant (s^-1)",
            "gamma Hopper constant (s^-1)",
            "gamma Wu constant (s^-1)",
        ]

    # T5 free coeffs + constant
    elif fit_mode == "T5":
        init_params = (
            1.38e-11,
            1.38e-11,
            510,
            2000,
            72.0,
            0.01,
            0.01,
            0.07,
            0.15,
        )
        omega_hopper_fit_func = lambda temp, beta: orbach_T5_free_const(
            temp, beta[2], beta[4], beta[0], beta[5]
        )
        omega_wu_fit_func = lambda temp, beta: orbach_T5_free_const(
            temp, beta[2], beta[4], beta[0], beta[6]
        )
        gamma_hopper_fit_func = lambda temp, beta: orbach_T5_free_const(
            temp, beta[3], beta[4], beta[1], beta[7]
        )
        gamma_wu_fit_func = lambda temp, beta: orbach_T5_free_const(
            temp, beta[3], beta[4], beta[1], beta[8]
        )
        beta_desc = [
            "Omega T5 coeff (K^-5 s^-1)",
            "gamma T5 coeff (K^-5 s^-1)",
            "gamma Omega Orbach coeff (s^-1)",
            "gamma Orbach coeff (s^-1)",
            "Orbach Delta (meV)",
            "Omega Hopper constant (s^-1)",
            "Omega Wu constant (s^-1)",
            "gamma Hopper constant (s^-1)",
            "gamma Wu constant (s^-1)",
        ]

    # T5 fixed + constant
    elif fit_mode == "T5_fixed_coeffs":
        init_params = (1.38e-11, 510, 2000, 72.0, 0.01, 0.01, 0.07, 0.15)
        omega_hopper_fit_func = lambda temp, beta: orbach_T5_free_const(
            temp, beta[1], beta[3], beta[0], beta[4]
        )
        omega_wu_fit_func = lambda temp, beta: orbach_T5_free_const(
            temp, beta[1], beta[3], beta[0], beta[5]
        )
        gamma_hopper_fit_func = lambda temp, beta: orbach_T5_free_const(
            temp, beta[2], beta[3], beta[0], beta[6]
        )
        gamma_wu_fit_func = lambda temp, beta: orbach_T5_free_const(
            temp, beta[2], beta[3], beta[0], beta[7]
        )
        beta_desc = [
            "T5 coeff (K^-5 s^-1)",
            "Omega Orbach coeff (s^-1)",
            "gamma Orbach coeff (s^-1)",
            "Orbach Delta (meV)",
            "Omega Hopper constant (s^-1)",
            "Omega Wu constant (s^-1)",
            "gamma Hopper constant (s^-1)",
            "gamma Wu constant (s^-1)",
        ]

    # Double Orbach
    elif fit_mode == "double_orbach":
        init_params = (
            450,
            1200,
            65,
            11000,
            11000,
            160,
            0.01,
            0.01,
            0.07,
            0.15,
        )
        omega_hopper_fit_func = lambda temp, beta: double_orbach(
            temp,
            beta[0],
            beta[2],
            beta[3],
            beta[5],
            beta[6],
        )
        omega_wu_fit_func = lambda temp, beta: double_orbach(
            temp,
            beta[0],
            beta[2],
            beta[3],
            beta[5],
            beta[7],
        )
        gamma_hopper_fit_func = lambda temp, beta: double_orbach(
            temp,
            beta[1],
            beta[2],
            beta[4],
            beta[5],
            beta[8],
        )
        gamma_wu_fit_func = lambda temp, beta: double_orbach(
            temp,
            beta[1],
            beta[2],
            beta[4],
            beta[5],
            beta[9],
        )
        beta_desc = [
            "Omega Orbach 1 coeff (s^-1)",
            "gamma Orbach 1 coeff (s^-1)",
            "Orbach 1 Delta (meV)",
            "Orbach 2 coeff (s^-1)",
            "Omega Orbach 2 Delta (meV)",
            "gamma Orbach 2 Delta (meV)",
            "Omega Hopper constant (s^-1)",
            "Omega Wu constant (s^-1)",
            "gamma Hopper constant (s^-1)",
            "gamma Wu constant (s^-1)",
        ]

    # Double Orbach, fixed high Orbach coeffs
    elif fit_mode == "double_orbach_fixed":
        init_params = (450, 1200, 65, 11000, 160, 0.01, 0.01, 0.07, 0.15)
        omega_hopper_fit_func = lambda temp, beta: double_orbach(
            temp,
            beta[0],
            beta[2],
            beta[3],
            beta[4],
            beta[5],
        )
        omega_wu_fit_func = lambda temp, beta: double_orbach(
            temp,
            beta[0],
            beta[2],
            beta[3],
            beta[4],
            beta[6],
        )
        gamma_hopper_fit_func = lambda temp, beta: double_orbach(
            temp,
            beta[1],
            beta[2],
            beta[3],
            beta[4],
            beta[7],
        )
        gamma_wu_fit_func = lambda temp, beta: double_orbach(
            temp,
            beta[1],
            beta[2],
            beta[3],
            beta[4],
            beta[8],
        )
        beta_desc = [
            "Omega Orbach 1 coeff (s^-1)",
            "gamma Orbach 1 coeff (s^-1)",
            "Orbach 1 Delta (meV)",
            "Orbach 2 coeff (s^-1)",
            "Orbach 2 Delta (meV)",
            "Omega Hopper constant (s^-1)",
            "Omega Wu constant (s^-1)",
            "gamma Hopper constant (s^-1)",
            "gamma Wu constant (s^-1)",
        ]

    # Triple Orbach
    # init_params = (450, 1200, 65, 1200, 95, 11000, 150, 0.01, 0.01, 0.07, 0.15)
    # omega_hopper_fit_func = lambda temp, beta: triple_orbach(
    #     temp,
    #     beta[0],
    #     beta[2],
    #     beta[3],
    #     beta[4],
    #     beta[5],
    #     beta[6],
    #     beta[7],
    # )
    # omega_wu_fit_func = lambda temp, beta: triple_orbach(
    #     temp,
    #     beta[0],
    #     beta[2],
    #     beta[3],
    #     beta[4],
    #     beta[5],
    #     beta[6],
    #     beta[8],
    # )
    # gamma_hopper_fit_func = lambda temp, beta: triple_orbach(
    #     temp,
    #     beta[1],
    #     beta[2],
    #     beta[3],
    #     beta[4],
    #     beta[5],
    #     beta[6],
    #     beta[9],
    # )
    # gamma_wu_fit_func = lambda temp, beta: triple_orbach(
    #     temp,
    #     beta[1],
    #     beta[2],
    #     beta[3],
    #     beta[4],
    #     beta[5],
    #     beta[6],
    #     beta[10],
    # )
    # beta_desc = [
    #     "Omega Orbach 1 coeff (s^-1)",
    #     "gamma Orbach 1 coeff (s^-1)",
    #     "Orbach 1 Delta (meV)",
    #     "Orbach 2 coeff (s^-1)",
    #     "Orbach 2 Delta (meV)",
    #     "Orbach 3 coeff (s^-1)",
    #     "Orbach 3 Delta (meV)",
    #     "Omega Hopper constant (s^-1)",
    #     "Omega Wu constant (s^-1)",
    #     "gamma Hopper constant (s^-1)",
    #     "gamma Wu constant (s^-1)",
    # ]

    # Triple Orbach, fixed energies
    # Delta_1 = 70
    # Delta_2 = 90
    # Delta_3 = 150
    # init_params = (450, 1200, 1200, 11000, 0.01, 0.01, 0.07, 0.15)
    # omega_hopper_fit_func = lambda temp, beta: triple_orbach(
    #     temp,
    #     beta[0],
    #     Delta_1,
    #     beta[2],
    #     Delta_2,
    #     beta[3],
    #     Delta_3,
    #     beta[4],
    # )
    # omega_wu_fit_func = lambda temp, beta: triple_orbach(
    #     temp,
    #     beta[0],
    #     Delta_1,
    #     beta[2],
    #     Delta_2,
    #     beta[3],
    #     Delta_3,
    #     beta[5],
    # )
    # gamma_hopper_fit_func = lambda temp, beta: triple_orbach(
    #     temp,
    #     beta[1],
    #     Delta_1,
    #     beta[2],
    #     Delta_2,
    #     beta[3],
    #     Delta_3,
    #     beta[6],
    # )
    # gamma_wu_fit_func = lambda temp, beta: triple_orbach(
    #     temp,
    #     beta[1],
    #     Delta_1,
    #     beta[2],
    #     Delta_2,
    #     beta[3],
    #     Delta_3,
    #     beta[7],
    # )
    # beta_desc = [
    #     "Omega Orbach 1 coeff (s^-1)",
    #     "gamma Orbach 1 coeff (s^-1)",
    #     "Orbach 2 coeff (s^-1)",
    #     "Orbach 3 coeff (s^-1)",
    #     "Omega Hopper constant (s^-1)",
    #     "Omega Wu constant (s^-1)",
    #     "gamma Hopper constant (s^-1)",
    #     "gamma Wu constant (s^-1)",
    # ]

    # Double Orbach, fixed energies
    # init_params = (450, 1200, 11000, 0.01, 0.01, 0.07, 0.15)
    # Delta_1 = 70
    # Delta_2 = 150
    # omega_hopper_fit_func = lambda temp, beta: double_orbach(
    #     temp,
    #     beta[0],
    #     Delta_1,
    #     beta[2],
    #     Delta_2,
    #     beta[3],
    # )
    # omega_wu_fit_func = lambda temp, beta: double_orbach(
    #     temp,
    #     beta[0],
    #     Delta_1,
    #     beta[2],
    #     Delta_2,
    #     beta[4],
    # )
    # gamma_hopper_fit_func = lambda temp, beta: double_orbach(
    #     temp,
    #     beta[1],
    #     Delta_1,
    #     beta[2],
    #     Delta_2,
    #     beta[5],
    # )
    # gamma_wu_fit_func = lambda temp, beta: double_orbach(
    #     temp,
    #     beta[1],
    #     Delta_1,
    #     beta[2],
    #     Delta_2,
    #     beta[6],
    # )
    # beta_desc = [
    #     "Omega Orbach 1 coeff (s^-1)",
    #     "gamma Orbach 1 coeff (s^-1)",
    #     "Orbach 2 coeff (s^-1)",
    #     "Omega Hopper constant (s^-1)",
    #     "Omega Wu constant (s^-1)",
    #     "gamma Hopper constant (s^-1)",
    #     "gamma Wu constant (s^-1)",
    # ]

    # Double Orbach, second Orbach fixed
    # init_params = (450, 1200, 65, 11000, 0.01, 0.01, 0.07, 0.15)
    # delta_2 = 165
    # omega_hopper_fit_func = lambda temp, beta: double_orbach(
    #     temp,
    #     beta[0],
    #     beta[2],
    #     beta[3],
    #     delta_2,
    #     beta[4],
    # )
    # omega_wu_fit_func = lambda temp, beta: double_orbach(
    #     temp,
    #     beta[0],
    #     beta[2],
    #     beta[3],
    #     delta_2,
    #     beta[5],
    # )
    # gamma_hopper_fit_func = lambda temp, beta: double_orbach(
    #     temp,
    #     beta[1],
    #     beta[2],
    #     beta[3],
    #     delta_2,
    #     beta[6],
    # )
    # gamma_wu_fit_func = lambda temp, beta: double_orbach(
    #     temp,
    #     beta[1],
    #     beta[2],
    #     beta[3],
    #     delta_2,
    #     beta[7],
    # )
    # beta_desc = [
    #     "Omega Orbach 1 coeff (s^-1)",
    #     "gamma Orbach 1 coeff (s^-1)",
    #     "Orbach 1 Delta (meV)",
    #     "Orbach 2 coeff (s^-1)",
    #     "Omega Hopper constant (s^-1)",
    #     "Omega Wu constant (s^-1)",
    #     "gamma Hopper constant (s^-1)",
    #     "gamma Wu constant (s^-1)",
    # ]

    # Double Orbach, fixed Orbach ratios
    # init_params = (800, 2, 65, 160, 0.01, 0.01, 0.07, 0.15)
    # omega_hopper_fit_func = lambda temp, beta: double_orbach_ratio(
    #     temp,
    #     beta[0],
    #     1,
    #     beta[2],
    #     beta[3],
    #     beta[4],
    # )
    # omega_wu_fit_func = lambda temp, beta: double_orbach_ratio(
    #     temp,
    #     beta[0],
    #     1,
    #     beta[2],
    #     beta[3],
    #     beta[5],
    # )
    # gamma_hopper_fit_func = lambda temp, beta: double_orbach_ratio(
    #     temp,
    #     beta[0],
    #     beta[1],
    #     beta[2],
    #     beta[3],
    #     beta[6],
    # )
    # gamma_wu_fit_func = lambda temp, beta: double_orbach_ratio(
    #     temp,
    #     beta[0],
    #     beta[1],
    #     beta[2],
    #     beta[3],
    #     beta[7],
    # )
    # beta_desc = [
    #     "Omega Orbach 1 coeff (s^-1)",
    #     "gamma: Omega Orbach coeff ratio",
    #     "Orbach 1 Delta (meV)",
    #     "Orbach 2 Delta (meV)",
    #     "Omega Hopper constant (s^-1)",
    #     "Omega Wu constant (s^-1)",
    #     "gamma Hopper constant (s^-1)",
    #     "gamma Wu constant (s^-1)",
    # ]

    # endregion

    fit_func = lambda beta, temp: simultaneous_test_lambda(
        temp,
        beta,
        omega_hopper_fit_func,
        omega_wu_fit_func,
        gamma_hopper_fit_func,
        gamma_wu_fit_func,
        sample_breaks[0],
    )
    # data = data = RealData(temps, combined_rates, temp_errors, combined_errs)
    data = data = RealData(temps, combined_rates, sy=combined_errs)
    model = Model(fit_func)
    odr = ODR(data, model, beta0=np.array(init_params))
    odr.set_job(fit_type=0)
    output = odr.run()
    popt = output.beta
    pcov = output.cov_beta
    pvar = output.sd_beta ** 2
    red_chi_square = output.res_var
    red_chi_square_report = tool_belt.round_sig_figs(red_chi_square, 3)
    print("Reduced chi squared: {}".format(red_chi_square_report))
    ssr = output.sum_square
    ssr_report = tool_belt.round_sig_figs(ssr, 3)
    print("Sum of squared residuals: {}".format(ssr_report))
    print("redChi2={}, SSR={}".format(red_chi_square_report, ssr_report))

    return (
        popt,
        np.diag(pcov),
        beta_desc,
        omega_hopper_fit_func,
        omega_wu_fit_func,
        gamma_hopper_fit_func,
        gamma_wu_fit_func,
    )


def get_data_points(
    path,
    file_name,
    temp_range=None,
    marker_type="sample",
    override_skips=False,
):

    file_path = path / "{}.xlsx".format(file_name)
    csv_file_path = path / "{}.csv".format(file_name)

    file = pd.read_excel(file_path, engine="openpyxl")
    file.to_csv(csv_file_path, index=None, header=True)

    # Marker and color combination to distinguish samples
    marker_ind = 0
    markers_list = [
        "o",
        "^",
        "s",
        "X",
        "D",
        "H",
    ]

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

            # Set up markers if the temp is in the plotting range
            temp = get_temp(point)
            if (temp_range is not None) and not (
                temp_range[0] < temp < temp_range[1]
            ):
                continue
            if nv_name not in nv_names:
                if marker_type == "nv":
                    markers[nv_name] = markers_list[marker_ind]
                    marker_ind += 1
                nv_names.append(nv_name)
            if sample not in samples:
                if marker_type == "sample":
                    markers[sample] = markers_list[marker_ind]
                    marker_ind += 1
                samples.append(sample)
            if marker_type == "nv":
                point["marker"] = markers[nv_name]
            elif marker_type == "sample":
                point["marker"] = markers[sample]

    # The first shall be last
    # data_points.append(data_points.pop(0))

    return data_points


def plot_scalings(
    process_to_plot,
    temp_range=[190, 310],
    rate_range=None,
    xscale="linear",
    yscale="linear",
):

    # %% Setup

    min_temp = temp_range[0]
    max_temp = temp_range[1]

    # temp_linspace = np.linspace(5, 600, 1000)
    temp_linspace = np.linspace(min_temp, max_temp, 1000)
    # temp_linspace = np.linspace(5, 300, 1000)
    # temp_linspace = np.linspace(5, 5000, 1000)
    fig, ax = plt.subplots(figsize=figsize)
    fig.set_tight_layout(True)
    # ax.set_title('Relaxation rates')

    if process_to_plot in ["Walker", "both"]:
        process_lambda = lambda temp: orbach_T5_free(
            temp, 0, 75, 1 / (300 ** 5)
        )
        process_edge_color = "blue"
        ax.plot(
            temp_linspace,
            process_lambda(temp_linspace),
            color=process_edge_color,
            label="Walker",
        )
    if process_to_plot in ["Orbach", "both"]:
        process_lambda = lambda temp: orbach_T5_free(
            temp, 1 / bose(75, 300), 75, 0
        )
        process_edge_color = "red"
        ax.plot(
            temp_linspace,
            process_lambda(temp_linspace),
            color=process_edge_color,
            label="Orbach",
        )

    ax.set_xlabel(r"Temperature $\mathit{T}$ (K)")
    ax.set_ylabel(r"Relaxation rate (arb. units)")
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.set_xlim(min_temp, max_temp)
    if rate_range is not None:
        ax.set_ylim(rate_range[0], rate_range[1])
    if process_to_plot in ["Walker", "Orbach"]:
        ax.set_title(
            "{} Process Temperature Dependence".format(process_to_plot)
        )
    elif process_to_plot == "both":
        ax.set_title("Relaxation Process Temperature Dependence")
        ax.legend(loc="upper left")


def plot_T2_max(
    omega_popt,
    gamma_popt,
    temp_range=[190, 310],
    xscale="linear",
    yscale="linear",
):

    omega_fit_func = orbach_T5_free
    gamma_fit_func = orbach_free

    omega_lambda = lambda temp: omega_fit_func(temp, *omega_popt)
    gamma_lambda = lambda temp: gamma_fit_func(temp, *gamma_popt)
    T2_max = lambda temp: 2 / (3 * omega_lambda(temp) + gamma_lambda(temp))

    min_temp = temp_range[0]
    max_temp = temp_range[1]

    temp_linspace = np.linspace(min_temp, max_temp, 1000)
    fig, ax = plt.subplots(figsize=figsize)
    fig.set_tight_layout(True)

    ax.plot(temp_linspace, T2_max(temp_linspace))

    ax.set_xlabel(r"Temperature $\mathit{T}$ (K)")
    ax.set_ylabel(r"$\mathit{T}_{\mathrm{2,max}}$ (s)")
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)


def normalized_residuals_histogram(rates_to_plot):

    data_points = get_data_points(path, file_name)
    # Fit to Omega and gamma simultaneously
    popt, _, _, omega_fit_func, gamma_fit_func = fit_simultaneous(data_points)
    omega_lambda = lambda temp: omega_fit_func(temp, popt)
    gamma_lambda = lambda temp: gamma_fit_func(temp, popt)

    fig, ax = plt.subplots(figsize=figsize)
    if rates_to_plot == "Omega":
        title_suffix = "Omega only"
    if rates_to_plot == "gamma":
        title_suffix = "gamma only"
    if rates_to_plot == "both":
        title_suffix = "both Omega and gamma"
    ax.set_title(r"Normalized residuals histogram: {}".format(title_suffix))
    ax.set_xlabel(r"Normalized residual")
    ax.set_ylabel(r"Frequency")
    x_range = [-3, +3]
    ax.set_xlim(*x_range)
    ax.set_ylim(0, 0.5)

    normalized_residuals = []

    for point in data_points:
        temp = get_temp(point)
        if rates_to_plot in ["Omega", "both"]:
            rate = point[omega_column_title]
            rate_err = point[omega_err_column_title]
            norm_res = (rate - omega_lambda(temp)) / rate_err
            normalized_residuals.append(norm_res)
        if rates_to_plot in ["gamma", "both"]:
            rate = point[gamma_column_title]
            rate_err = point[gamma_err_column_title]
            norm_res = (rate - gamma_lambda(temp)) / rate_err
            normalized_residuals.append(norm_res)

    bin_width = 0.5
    bin_edges = np.arange(x_range[0], x_range[1] + bin_width, bin_width)
    # hist, bin_edges = np.histogram(
    #     normalized_residuals, bins=bin_edges, density=True
    # )
    # bin_centers = [
    #     (bin_edges[ind] + bin_edges[ind + 1]) / 2
    #     for ind in range(0, len(bin_edges) - 1)
    # ]
    ax.hist(normalized_residuals, bins=bin_edges, density=True)

    inv_root_2_pi = 1 / np.sqrt(2 * np.pi)
    norm_gaussian = lambda norm_res: inv_root_2_pi * np.exp(
        -(norm_res ** 2) / 2
    )
    norm_res_linspace = np.linspace(*x_range, 1000)
    ax.plot(norm_res_linspace, norm_gaussian(norm_res_linspace), lw=line_width)

    fig.tight_layout(pad=0.3)


def plot_orbach_scalings(temp_range, xscale, yscale, y_range):

    min_temp = temp_range[0]
    max_temp = temp_range[1]

    temp_linspace = np.linspace(min_temp, max_temp, 1000)
    fig, ax = plt.subplots(figsize=figsize)

    normalized = True
    Delta_1 = 70
    Delta_2 = 90
    Delta_3 = 150
    orbach_1 = lambda temp: bose(Delta_1, temp)
    orbach_2 = lambda temp: bose(Delta_2, temp)
    orbach_3 = lambda temp: bose(Delta_3, temp)
    orbach_1_full = lambda temp: bose(Delta_1, temp) * (
        1 + bose(Delta_1, temp)
    )
    orbach_2_full = lambda temp: bose(Delta_2, temp) * (
        1 + bose(Delta_2, temp)
    )
    orbach_3_full = lambda temp: bose(Delta_3, temp) * (
        1 + bose(Delta_3, temp)
    )
    for ind in [1, 2, 3]:
        orbach_lambda = eval("orbach_{}".format(ind))
        orbach_full_lambda = eval("orbach_{}_full".format(ind))
        if normalized:
            factor = orbach_lambda(300)
            factor_full = orbach_full_lambda(300)
            plot_orbach_lambda = lambda temp: (1 / factor) * orbach_lambda(
                temp
            )
            plot_orbach_full_lambda = lambda temp: (
                1 / factor_full
            ) * orbach_full_lambda(temp)
        else:
            plot_orbach_lambda = orbach_lambda
            plot_orbach_full_lambda = orbach_full_lambda
        label = str(eval("Delta_{}".format(ind)))
        label_full = str(eval("Delta_{}".format(ind))) + " (full)"
        ax.plot(temp_linspace, plot_orbach_lambda(temp_linspace), label=label)
        ax.plot(
            temp_linspace,
            plot_orbach_full_lambda(temp_linspace),
            label=label_full,
        )
    ax.legend(title=r"\(\Delta\) (meV)")
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.set_xlim(min_temp, max_temp)
    ax.set_xlabel(r"Temperature $\mathit{T}$ (K)")
    if y_range is not None:
        ax.set_ylim(y_range[0], y_range[1])
    if normalized:
        ax.set_ylabel(r"Normalized Orbach scaling")
    else:
        ax.set_ylabel(r"Orbach scaling")
    fig.tight_layout(pad=0.3)
    return


def figure_2(file_name, path, dosave=False, supp_comparison=False):

    data_points = get_data_points(path, file_name)  # , temp_range)
    # fit_modes = ["double_orbach_fixed", "T5"]
    # fit_modes = ["double_orbach", "variable_exp"]
    # fit_modes = ["double_orbach", "T5_fixed_coeffs"]
    fit_modes = ["double_orbach", "T5"]
    rates = ["gamma", "Omega"]
    labels = ["(a)", "(b)"]

    # figsize = (figsize[0], 2 * figsize[1])
    # adj_figsize = (figsize[0], (2 * figsize[1]) + 1.0)
    adj_figsize = (2 * figsize[0], figsize[1])
    fig_a, ax_a = plt.subplots(figsize=adj_figsize)
    if not supp_comparison:
        fig_b = plt.figure(figsize=figsize)
    # gs_sep = 0.09
    # gs_a_bottom = 0.55
    # gs_a = fig.add_gridspec(
    #     nrows=1,
    #     ncols=1,
    #     left=0.07,
    #     right=0.49,
    #     bottom=0.13,
    #     top=0.99,
    # )
    if not supp_comparison:
        gs_b = fig_b.add_gridspec(
            nrows=2,
            # ncols=2,
            ncols=4,
            left=0.11,
            right=1.0,
            bottom=0.13,
            top=0.94,
            wspace=0,
            hspace=0,
            # width_ratios=[1, 0.2, 0.2, 1],
            width_ratios=[1, 0.16, 1, 0.16],
        )
        # ax_a = fig.add_subplot(gs_a[:, :])
        scatter_axes_b = [[None, None], [None, None]]
        scatter_axes_b[0][0] = fig_b.add_subplot(gs_b[0, 0])
        scatter_axes_b[0][1] = fig_b.add_subplot(gs_b[0, 2])
        scatter_axes_b[1][0] = fig_b.add_subplot(gs_b[1, 0])
        scatter_axes_b[1][1] = fig_b.add_subplot(gs_b[1, 2])

        hist_axes_b = [[None, None], [None, None]]
        hist_axes_b[0][0] = fig_b.add_subplot(gs_b[0, 1])
        hist_axes_b[0][1] = fig_b.add_subplot(gs_b[0, 3])
        hist_axes_b[1][0] = fig_b.add_subplot(gs_b[1, 1])
        hist_axes_b[1][1] = fig_b.add_subplot(gs_b[1, 3])

    # Generic setup

    # fig.text(
    #     -0.16,
    #     0.96,
    #     "(a)",
    #     transform=ax_a.transAxes,
    #     color="black",
    #     fontsize=18,
    # )
    # fig.text(
    #     1.02,
    #     0.96,
    #     "(b)",
    #     transform=ax_a.transAxes,
    #     color="black",
    #     fontsize=18,
    # )

    axins_a = None
    if not supp_comparison:
        inset_bottom = 0.105
        inset_height = 0.47
        # inset_left = 0.6
        inset_left = 0.46
        # inset_width = 1 - inset_left
        inset_width = 0.55

        axins_a = inset_axes(
            ax_a,
            width="100%",
            height="100%",
            bbox_to_anchor=(
                inset_left,
                inset_bottom,
                inset_width,
                inset_height,
            ),
            bbox_transform=ax_a.transAxes,
            loc=1,
        )

    # Plot the experimental data

    figure_2_raw_data(ax_a, axins_a, data_points)

    # Plot the fits in (a)

    for fit_mode in fit_modes:
        figure_2_fits(ax_a, axins_a, data_points, fit_mode)

    # Plot the residuals

    if not supp_comparison:
        # scatter_axes_b[0][0].set_title("Double Orbach")
        # scatter_axes_b[0][1].set_title(r"Orbach $+ T^{5}$")
        scatter_axes_b[0][0].set_title("Proposed model")
        scatter_axes_b[0][1].set_title("Prior model")
        # scatter_axes_b[0][0].set_title(
        #     r"$C + A_{1} O(\Delta_{1}, T) + A_{2} O(\Delta_{2}, T)$"
        # )
        # scatter_axes_b[0][1].set_title(r"$C + A_{1} O(\Delta, T) + A_{2} T^{5}$")

        scatter_axes_b[0][0].get_xaxis().set_visible(False)
        scatter_axes_b[0][1].get_xaxis().set_visible(False)
        scatter_axes_b[0][1].get_yaxis().set_visible(False)
        scatter_axes_b[1][1].get_yaxis().set_visible(False)

        scatter_axes_b[0][0].set_ylabel(r"$\mathit{\gamma}$ residual")
        scatter_axes_b[1][0].set_ylabel(r"$\mathrm{\Omega}$ residual")
        x_label = r"Temperature $\mathit{T}$ (K)"
        scatter_axes_b[1][0].set_xlabel(x_label)
        scatter_axes_b[1][1].set_xlabel(x_label)

        for rate_ind in range(2):
            rate = rates[rate_ind]
            for fit_mode_ind in range(2):

                scatter_ax = scatter_axes_b[rate_ind][fit_mode_ind]
                hist_ax = hist_axes_b[rate_ind][fit_mode_ind]
                hist_ax.get_xaxis().set_visible(False)
                hist_ax.get_yaxis().set_visible(False)

                fit_mode = fit_modes[fit_mode_ind]

                xlim = [-10, 490]
                scatter_ax.set_xlim(xlim[0], xlim[1])
                scatter_ax.plot(
                    xlim, [0, 0], color="silver", zorder=-10, lw=line_width
                )

                # axins.set_ylim(-3.25, 3.25)
                # axins.set_yticks(np.linspace(-3, 3, 7))

                ax_ylim = 2.5
                scatter_ax.set_ylim(-ax_ylim, ax_ylim)
                hist_ax.set_ylim(-ax_ylim, ax_ylim)
                ylim_floor = math.floor(ax_ylim)
                num_yticks = (ylim_floor * 2) + 1
                yticks = np.linspace(-ylim_floor, ylim_floor, num_yticks)
                scatter_ax.set_yticks(yticks)

                figure_2_residuals(
                    scatter_ax, hist_ax, rate, data_points, fit_mode
                )

    if supp_comparison:
        past_results = [
            "redman",
            "takahashi",
            r"jarmola\_s2",
            r"jarmola\_s3",
            r"jarmola\_s8",
            r"lin",
        ]
        for res in past_results:
            (
                omega_temps,
                omega_rates,
                gamma_temps,
                gamma_rates,
            ) = get_past_results(res)
            ax_a.plot(
                omega_temps,
                omega_rates,
                label=r"$\mathrm{\Omega}$",
                marker="D",
                color=omega_edge_color,
                markerfacecolor=omega_face_color,
                linestyle="None",
                ms=marker_size,
                lw=line_width,
                markeredgewidth=marker_edge_width,
            )
            # gamma
            if gamma_temps is not None:
                ax_a.plot(
                    gamma_temps,
                    gamma_rates,
                    label=r"$\mathit{\gamma}$",
                    marker="D",
                    color=gamma_edge_color,
                    markerfacecolor=gamma_face_color,
                    linestyle="None",
                    ms=marker_size,
                    lw=line_width,
                    markeredgewidth=marker_edge_width,
                )

    # fig.tight_layout(pad=0.3)
    # tool_belt.non_math_ticks(ax_a)
    # for el in scatter_axes_b:
    #     for sub_el in el:
    #         tool_belt.non_math_ticks(sub_el)
    # for el in hist_axes_b:
    #     for sub_el in el:
    #         tool_belt.non_math_ticks(sub_el)

    # fontProperties = {'family':'sans-serif'}
    # ax_a.set_xticklabels(ax_a.get_xticks(), fontProperties)
    # ax_a.set_yticklabels(ax_a.get_yticks(), fontProperties)
    fig_a.tight_layout(pad=0.3)
    if not supp_comparison:
        fig_b.tight_layout(pad=0.3)


def figure_2_raw_data(ax, axins, data_points):

    # %% Setup

    if axins is None:
        axes = [ax]
    else:
        axes = [ax, axins]

    # temp_ranges = [[-5, 480], [145, 215]]
    # rate_ranges = [[0.0036, 1100], [2, 50]]
    # yscales = ["log", "log"]
    # ytickss = [None, [3, 10, 30]]

    temp_ranges = [[-5, 480], [-5, 487]]
    rate_ranges = [[0.004, 750], [-25, 670]]
    yscales = ["log", "linear"]
    ytickss = [None, None]

    no_legends = [False, True]
    mss = [marker_size, marker_size - 1]
    lws = [line_width, line_width - 0.25]
    xlabels = [
        r"Temperature $\mathit{T}$ (K)",
        None,
    ]
    ylabels = [
        r"Relaxation rates (s$^{\text{-1}}$)",
        None,
    ]

    for ind in range(len(axes)):
        ax = axes[ind]
        temp_range = temp_ranges[ind]
        rate_range = rate_ranges[ind]
        yticks = ytickss[ind]
        yscale = yscales[ind]
        no_legend = no_legends[ind]
        samples_to_plot = ["hopper", "wu"]
        linestyles = {"hopper": "dotted", "wu": "dashed"}
        marker_type = "sample"
        include_sample_legend = not no_legend
        ms = mss[ind]
        lw = lws[ind]
        xlabel = xlabels[ind]
        ylabel = ylabels[ind]

        # Sample-dependent vs phonon-limited line
        ax.axvline(x=125, color="silver", zorder=-10, lw=lw)

        # marker_type = "nv"

        min_temp = temp_range[0]
        max_temp = temp_range[1]

        linspace_min_temp = min_temp if min_temp > 0 else 0
        temp_linspace = np.linspace(linspace_min_temp, max_temp, 1000)
        ax.set_xlim(min_temp, max_temp)

        # Plot setup
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_yscale(yscale)
        if rate_range is not None:
            ax.set_ylim(rate_range[0], rate_range[1])
        if yticks is not None:
            ax.set_yticks(yticks)
            labels = [str(el) for el in yticks]
            ax.set_yticklabels(labels)

        # %% Plot the points

        samples = []
        nv_names = []
        markers_list = []

        for point in data_points:

            if "marker" not in point:
                continue
            sample = point[sample_column_title]
            nv_name = point["nv_name"]
            sample_lower = sample.lower()
            marker = point["marker"]

            if nv_name not in nv_names:
                nv_names.append(nv_name)
            if sample not in samples:
                samples.append(sample)
            if marker not in markers_list:
                markers_list.append(marker)
            if sample.lower() not in samples_to_plot:
                continue

            temp = get_temp(point)
            if no_x_errs:
                temp_error = None
            else:
                temp_error = get_temp_error(point)

            # Omega
            rate = point[omega_column_title]
            rate_err = point[omega_err_column_title]
            val = rate
            val_err = rate_err
            ax.errorbar(
                temp,
                val,
                yerr=val_err,
                xerr=temp_error,
                label=r"$\mathrm{\Omega}$",
                marker=marker,
                color=omega_edge_color,
                markerfacecolor=omega_face_color,
                linestyle="None",
                ms=ms,
                lw=line_width,
                markeredgewidth=marker_edge_width,
            )

            # gamma
            rate = point[gamma_column_title]
            rate_err = point[gamma_err_column_title]
            val = rate
            val_err = rate_err
            ax.errorbar(
                temp,
                val,
                yerr=val_err,
                xerr=temp_error,
                label=r"$\mathit{\gamma}$",
                marker=marker,
                color=gamma_edge_color,
                markerfacecolor=gamma_face_color,
                linestyle="None",
                ms=ms,
                lw=line_width,
                markeredgewidth=marker_edge_width,
            )

        # Rate legend
        if not no_legend:
            omega_patch = patches.Patch(
                label=r"$\mathrm{\Omega}$",
                facecolor=omega_face_color,
                edgecolor=omega_edge_color,
                lw=marker_edge_width,
            )
            gamma_patch = patches.Patch(
                label=r"$\mathit{\gamma}$",
                facecolor=gamma_face_color,
                edgecolor=gamma_edge_color,
                lw=marker_edge_width,
            )
            leg1 = ax.legend(
                handles=[gamma_patch, omega_patch],
                loc="upper left",
                title="Rate",
                handlelength=1.5,
            )
            # leg1 = ax.legend(
            #     handles=[omega_patch, gamma_patch], loc="upper left", frameon=False
            # )

        # Sample legend
        if include_sample_legend:
            include_fit_lines = False
            x_loc = 0.18
            handlelength = 1
            if include_fit_lines:
                nv_patches = []
                for ind in range(len(markers_list)):
                    nv_name = nv_names[ind].replace("_", "\_")
                    sample = nv_name.split("-")[0]
                    if sample == "prresearch":
                        nv_name = "[1]"
                    # else:
                    #     label = "New results"
                    ls = linestyles[sample]
                    if marker_type == "nv":
                        label = nv_name
                        title = "sample-nv"
                    elif marker_type == "sample":
                        label = sample[0].upper() + sample[1:]
                        title = "Sample"
                    patch = mlines.Line2D(
                        [],
                        [],
                        color="black",
                        marker=markers_list[ind],
                        linestyle=linestyles[sample],
                        markersize=marker_size,
                        markeredgewidth=marker_edge_width,
                        label=label,
                    )
                    nv_patches.append(patch)
                ax.legend(
                    handles=nv_patches,
                    loc="upper left",
                    title=title,
                    # title="Samples",
                    bbox_to_anchor=(x_loc, 1.0),
                    framealpha=1.0,
                    handlelength=handlelength,
                )
            else:
                nv_patches = []
                for ind in range(len(markers_list)):
                    nv_name = nv_names[ind].replace("_", "\_")
                    sample = nv_name.split("-")[0]
                    if sample == "prresearch":
                        nv_name = "[1]"
                    # else:
                    #     label = "New results"
                    ls = linestyles[sample]
                    if marker_type == "nv":
                        label = nv_name
                        title = "sample-nv"
                    elif marker_type == "sample":
                        # label = sample[0].upper() + sample[1:]
                        if sample == "hopper":
                            label = "A"
                        elif sample == "wu":
                            label = "B"
                        title = "Sample"
                    patch = mlines.Line2D(
                        [],
                        [],
                        color="black",
                        marker=markers_list[ind],
                        linestyle="None",
                        markersize=marker_size,
                        markeredgewidth=marker_edge_width,
                        label=label,
                    )
                    nv_patches.append(patch)
                ax.legend(
                    handles=nv_patches,
                    loc="upper left",
                    title=title,
                    # title="Samples",
                    bbox_to_anchor=(x_loc, 1.0),
                    framealpha=1.0,
                    handlelength=handlelength,
                )

        # Final steps

        # Sample-dependent vs phonon-limited line
        include_sample_dep_line_label = False
        if include_sample_dep_line_label:
            text_font_size = 11.25
            arrow_font_size = 16
            ax = axes[0]
            args = {
                "transform": ax.transAxes,
                "color": "black",
                "fontsize": text_font_size,
                "ha": "right",
            }
            x_loc = 0.253
            y_loc = 0.765
            linespacing = 0.04
            ax.text(x_loc, y_loc, r"Sample-", **args)
            ax.text(x_loc, y_loc - linespacing, r"dependent", **args)
            prev = args["fontsize"]
            args["fontsize"] = arrow_font_size
            ax.text(
                x_loc,
                y_loc - 2.25 * linespacing,
                r"$\boldsymbol{\leftarrow}$",
                **args,
            )
            args["fontsize"] = prev

            args["ha"] = "left"
            x_loc += 0.03
            ax.text(x_loc, y_loc, r"Phonon-", **args)
            ax.text(x_loc, y_loc - linespacing, r"limited", **args)
            prev = args["fontsize"]
            args["fontsize"] = arrow_font_size
            ax.text(
                x_loc,
                y_loc - 2.25 * linespacing,
                r"$\boldsymbol{\rightarrow}$",
                **args,
            )
            args["fontsize"] = prev

        if include_sample_legend:
            ax.add_artist(leg1)


def figure_2_fits(ax_a, axins_a, data_points, fit_mode):

    samples_to_plot = ["hopper", "wu"]
    linestyles = {"hopper": "dotted", "wu": "dashed"}
    # linestyles = {"hopper": "dotted", "wu": "solid"}

    zorder = 0
    if fit_mode != "double_orbach":
        zorder = -1

    if axins_a is None:
        axes = [ax_a]
    else:
        axes = [ax_a, axins_a]

    for ax in axes:

        min_temp, max_temp = ax.get_xlim()

        linspace_min_temp = min_temp if min_temp > 0 else 0
        temp_linspace = np.linspace(linspace_min_temp, max_temp, 1000)

        # Fit to Omega and gamma simultaneously
        (
            popt,
            pvar,
            beta_desc,
            omega_hopper_fit_func,
            omega_wu_fit_func,
            gamma_hopper_fit_func,
            gamma_wu_fit_func,
        ) = fit_simultaneous(data_points, fit_mode)
        omega_hopper_lambda = lambda temp: omega_hopper_fit_func(temp, popt)
        omega_wu_lambda = lambda temp: omega_wu_fit_func(temp, popt)
        gamma_hopper_lambda = lambda temp: gamma_hopper_fit_func(temp, popt)
        gamma_wu_lambda = lambda temp: gamma_wu_fit_func(temp, popt)
        print("parameter description: popt, psd")
        for ind in range(len(popt)):
            desc = beta_desc[ind]
            val = tool_belt.round_sig_figs(popt[ind], 5)
            err = tool_belt.round_sig_figs(np.sqrt(pvar[ind]), 2)
            print("{}: {}, {}".format(desc, val, err))
            print(presentation_round_latex(val, err))

        # Plot the rate fits
        line_color = omega_edge_color
        if fit_mode != "double_orbach":
            line_color = "#fcd4ac"
        for sample in samples_to_plot:
            fit_func = eval("omega_{}_lambda".format(sample))
            ls = linestyles[sample]
            ax.plot(
                temp_linspace,
                fit_func(temp_linspace),
                linestyle=ls,
                label=r"$\mathrm{\Omega}$ fit",
                color=line_color,
                linewidth=line_width,
                zorder=zorder,
            )
        line_color = gamma_edge_color
        if fit_mode != "double_orbach":
            line_color = "#e09de0"
        for sample in samples_to_plot:
            fit_func = eval("gamma_{}_lambda".format(sample))
            ls = linestyles[sample]
            ax.plot(
                temp_linspace,
                fit_func(temp_linspace),
                linestyle=ls,
                label=r"$\mathit{\gamma}$ fit",
                color=line_color,
                linewidth=line_width,
                zorder=zorder,
            )


def figure_2_residuals(scatter_ax, hist_ax, plot_rate, data_points, fit_mode):

    plot_rate = plot_rate.lower()
    samples_to_plot = ["hopper", "wu"]
    # temp_range = [-5, 480]
    # min_temp = temp_range[0]
    # max_temp = temp_range[1]
    # ax.set_xlim(min_temp, max_temp)

    # if rate == "gamma":
    #     # title = r"$\gamma$ Normalized residual"
    #     title = r"$\gamma$ residual"
    # if rate == "omega":
    #     # title = r"$\Omega$ Normalized residual"
    #     title = r"$\Omega$ residual"
    # ax.set_ylabel(title)
    # ax.set_xlabel(r"T(K)")

    # Fit to Omega and gamma simultaneously
    (
        popt,
        pvar,
        beta_desc,
        omega_hopper_fit_func,
        omega_wu_fit_func,
        gamma_hopper_fit_func,
        gamma_wu_fit_func,
    ) = fit_simultaneous(data_points, fit_mode)
    omega_hopper_lambda = lambda temp: omega_hopper_fit_func(temp, popt)
    omega_wu_lambda = lambda temp: omega_wu_fit_func(temp, popt)
    gamma_hopper_lambda = lambda temp: gamma_hopper_fit_func(temp, popt)
    gamma_wu_lambda = lambda temp: gamma_wu_fit_func(temp, popt)

    samples = []
    nv_names = []
    markers_list = []
    max_norm_err = 0
    ms = (marker_size - 1) ** 2
    lw = line_width - 0.25
    err_list = []
    edgecolor = eval("{}_edge_color".format(plot_rate))
    facecolor = eval("{}_face_color".format(plot_rate))

    for point in data_points:

        if "marker" not in point:
            continue
        sample = point[sample_column_title]
        nv_name = point["nv_name"]
        sample_lower = sample.lower()
        marker = point["marker"]

        if nv_name not in nv_names:
            nv_names.append(nv_name)
        if sample not in samples:
            samples.append(sample)
        if marker not in markers_list:
            markers_list.append(marker)
        if sample.lower() not in samples_to_plot:
            continue

        temp = get_temp(point)
        temp_error = get_temp_error(point)

        column_title = eval("{}_column_title".format(plot_rate))
        err_column_title = eval("{}_err_column_title".format(plot_rate))
        rate = point[column_title]
        rate_err = point[err_column_title]
        rate_lambda = eval("{}_{}_lambda".format(plot_rate, sample_lower))
        val = (rate - rate_lambda(temp)) / rate_err
        err_list.append(val)
        if abs(val) > max_norm_err:
            max_norm_err = abs(val)
        scatter_ax.scatter(
            temp,
            val,
            # label=r"$\Omega$",
            marker=marker,
            edgecolor=edgecolor,
            facecolor=facecolor,
            linestyle="None",
            s=ms,
            linewidth=lw,
        )

    bins = np.linspace(-2.5, 2.5, 8)
    hist_ax.hist(
        err_list,
        bins,
        orientation="horizontal",
        edgecolor=edgecolor,
        color=facecolor,
        linewidth=marker_edge_width,
        density=True,
    )
    hist_ax.set_xlim([0, 0.5])
    hist_xlim = hist_ax.get_xlim()
    hist_ylim = hist_ax.get_ylim()
    # if fit_mode == "double_orbach":
    # if fit_mode == "T5":
    #     hist_ax.set_xlim(hist_xlim[::-1])
    inv_root_2_pi = 1 / np.sqrt(2 * np.pi)
    normal_density = lambda x: inv_root_2_pi * np.exp(-(x ** 2) / 2)
    err_linspace = np.linspace(hist_ylim[0], hist_ylim[1], 1000)
    hist_ax.plot(
        normal_density(err_linspace), err_linspace, color=edgecolor, zorder=1
    )
    hist_ax.axis("off")

    # print(max_norm_err)


# %% Main


def main(
    file_name,
    path,
    plot_type,
    rates_to_plot,
    temp_range=[190, 310],
    rate_range=None,
    xscale="linear",
    yscale="linear",
    dosave=False,
):

    # %% Setup

    if plot_type in "T2_max_supp":

        ### (a) and (b) version
        # fs = double_figsize
        # fig, axes_pack = plt.subplots(1, 2, figsize=fs)
        # ax1, ax2 = axes_pack

        ### Inset version
        fs = figsize
        fig, ax1 = plt.subplots(figsize=fs)
        inset_bottom = 0.58
        inset_height = 0.43
        inset_left = 0.52
        inset_width = 0.485
        ax2 = inset_axes(
            ax1,
            width="100%",
            height="100%",
            bbox_to_anchor=(
                inset_left,
                inset_bottom,
                inset_width,
                inset_height,
            ),
            bbox_transform=ax1.transAxes,
            loc=1,
        )

        _, _, leg1, T2_max_qubit_hopper_temp = main_sub(
            fig,
            ax1,
            file_name,
            path,
            "T2_max",
            rates_to_plot[0],
            temp_range[0],
            rate_range[0],
            xscale[0],
            yscale[0],
            dosave,
        )
        _ = main_sub(
            fig,
            ax2,
            file_name,
            path,
            "T2_frac",
            rates_to_plot[1],
            temp_range[1],
            rate_range[1],
            xscale[1],
            yscale[1],
            dosave,
        )
        # fig.text(
        #     -0.16,
        #     0.96,
        #     "(a)",
        #     transform=ax1.transAxes,
        #     color="black",
        #     fontsize=18,
        # )
        # fig.text(
        #     -0.138,
        #     0.96,
        #     "(b)",
        #     transform=ax2.transAxes,
        #     color="black",
        #     fontsize=18,
        # )
        # fig.subplots_adjust(wspace=0.16)
        return fig, ax1, ax2, leg1, T2_max_qubit_hopper_temp
    elif plot_type == "rates":
        fs = figsize
        fig, ax1 = plt.subplots(figsize=fs)
        inset_bottom = 0.1
        inset_height = 0.43
        inset_left = 0.52
        inset_width = 0.485
        ax2 = inset_axes(
            ax1,
            width="100%",
            height="100%",
            bbox_to_anchor=(
                inset_left,
                inset_bottom,
                inset_width,
                inset_height,
            ),
            bbox_transform=ax1.transAxes,
            loc=1,
        )

        main_sub(
            fig,
            ax1,
            file_name,
            path,
            "rates",
            rates_to_plot[0],
            temp_range[0],
            rate_range[0],
            xscale[0],
            yscale[0],
            dosave,
        )
        main_sub(
            fig,
            ax2,
            file_name,
            path,
            "rates",
            rates_to_plot[1],
            temp_range[1],
            rate_range[1],
            xscale[1],
            yscale[1],
            dosave,
            inset=True,
        )
    else:
        fs = figsize
        fig, ax = plt.subplots(figsize=fs)
        return main_sub(
            fig,
            ax,
            file_name,
            path,
            plot_type,
            rates_to_plot,
            temp_range,
            rate_range,
            xscale,
            yscale,
            dosave,
        )


def main_sub(
    fig,
    ax,
    file_name,
    path,
    plot_type,
    rates_to_plot,
    temp_range,
    rate_range,
    xscale,
    yscale,
    dosave,
    inset=False,
):

    data_points = get_data_points(path, file_name, temp_range)

    marker_type = "sample"
    min_temp = temp_range[0]
    max_temp = temp_range[1]

    linspace_min_temp = max(0, min_temp)
    temp_linspace = np.linspace(linspace_min_temp, max_temp, 1000)

    # Fit to Omega and gamma simultaneously
    (
        popt,
        pvar,
        beta_desc,
        omega_hopper_fit_func,
        omega_wu_fit_func,
        gamma_hopper_fit_func,
        gamma_wu_fit_func,
    ) = fit_simultaneous(data_points, "double_orbach")

    # omega_lambda = lambda temp: orbach_free(temp, 5.4603e02, 71)
    # gamma_lambda = lambda temp: orbach_free(temp, 1.5312e03, 71)
    # omega_lambda = lambda temp: orbach_free(temp, 1e8, 400)
    # gamma_lambda = omega_lambda
    omega_hopper_lambda = lambda temp: omega_hopper_fit_func(temp, popt)
    omega_wu_lambda = lambda temp: omega_wu_fit_func(temp, popt)
    gamma_hopper_lambda = lambda temp: gamma_hopper_fit_func(temp, popt)
    gamma_wu_lambda = lambda temp: gamma_wu_fit_func(temp, popt)

    # for temp in np.arange(487.5, 555, 12.5):
    #     boilerplate = "data_points.append(gen_fake_data_point({}, {}, {}))"
    #     temp = round(temp, 1)
    #     omega = round(omega_lambda(temp))
    #     gamma = round(gamma_lambda(temp))
    #     print(boilerplate.format(temp, omega, gamma))

    sim_temps, sim_omega, sim_gamma = get_ab_initio_rates()
    sim_ls = "solid"

    print("parameter description: popt, psd")
    for ind in range(len(popt)):
        desc = beta_desc[ind]
        val = tool_belt.round_sig_figs(popt[ind], 5)
        err = tool_belt.round_sig_figs(np.sqrt(pvar[ind]), 2)
        print("{}: {}, {}".format(desc, val, err))
        print(presentation_round_latex(val, err))
    samples_to_plot = ["hopper", "wu"]
    # samples_to_plot = ["hopper"]
    # samples_to_plot = ["wu"]
    linestyles = {"hopper": "dotted", "wu": "dashed"}
    if (plot_type == "rates") and (rates_to_plot in ["both", "Omega"]):
        for sample in samples_to_plot:
            fit_func = eval("omega_{}_lambda".format(sample))
            ls = linestyles[sample]
            ax.plot(
                temp_linspace,
                fit_func(temp_linspace),
                linestyle=ls,
                label=r"$\mathrm{\Omega}$ fit",
                color=omega_edge_color,
                linewidth=line_width,
            )
        # Plot Jarmola 2012 Eq. 1 for S3
        # ax.plot(temp_linspace, omega_calc(temp_linspace),
        #         label=r'$\Omega$ fit', color=omega_edge_color)
        # Ab initio plot
        ax.plot(
            sim_temps,
            sim_omega,
            linestyle=sim_ls,
            label=r"$\mathrm{\Omega}$ fit",
            color=omega_face_color,
            linewidth=line_width,
        )

    if (plot_type == "rates") and (rates_to_plot in ["both", "gamma"]):
        for sample in samples_to_plot:
            fit_func = eval("gamma_{}_lambda".format(sample))
            ls = linestyles[sample]
            ax.plot(
                temp_linspace,
                fit_func(temp_linspace),
                linestyle=ls,
                color=gamma_edge_color,
                linewidth=line_width,
            )
        # Ab initio plot
        ax.plot(
            sim_temps,
            sim_gamma,
            linestyle=sim_ls,
            color=gamma_face_color,
            linewidth=line_width,
        )
    # print(omega_lambda(50))
    # print(gamma_lambda(50))

    # Plot ratio
    ratio_hopper_lambda = lambda temp: gamma_hopper_lambda(
        temp
    ) / omega_hopper_lambda(temp)
    ratio_wu_lambda = lambda temp: gamma_wu_lambda(temp) / omega_wu_lambda(
        temp
    )
    if plot_type in ["ratios", "ratio_fits"]:
        # for func in [ratio_hopper_lambda, ratio_wu_lambda]:
        for sample in rates_to_plot:
            func = eval("ratio_{}_lambda".format(sample))
            ax.plot(
                temp_linspace,
                func(temp_linspace),
                label=r"$\mathit{\gamma}/\mathrm{\Omega}",
                color=gamma_edge_color,
                linewidth=line_width,
            )
    if plot_type in ["T2_max", "T2_frac"]:
        T2_max_qubit = lambda omega, gamma: 2 / (3 * omega + gamma)
        T2_max_qubit_hopper_temp = lambda temp: T2_max_qubit(
            omega_hopper_lambda(temp), gamma_hopper_lambda(temp)
        )
        T2_max_qubit_wu_temp = lambda temp: T2_max_qubit(
            omega_wu_lambda(temp), gamma_wu_lambda(temp)
        )
        T2_max_qubit_err = lambda T2max, omega_err, gamma_err: (
            (T2max ** 2) / 2
        ) * np.sqrt((3 * omega_err) ** 2 + gamma_err ** 2)

        if plot_type == "T2_max":
            for sample, linestyle, label in [
                # ("hopper", "dotted", r"$\mathrm{\{\ket{0}, \ket{\pm 1}\}}$"),
                ("hopper", "dotted", "SQ"),
                ("wu", "dashed", None),
            ]:
                if sample not in rates_to_plot:
                    continue
                func = eval(f"T2_max_qubit_{sample}_temp")
                if len(rates_to_plot) == 1:
                    linestyle = "solid"
                ax.plot(
                    temp_linspace,
                    func(temp_linspace),
                    label=label,
                    color=qubit_color,
                    linewidth=line_width,
                    ls=linestyle,
                )
        T2_max_qutrit = lambda omega, gamma: 1 / (omega + gamma)
        T2_max_qutrit_err = lambda T2max, omega_err, gamma_err: (
            T2max ** 2
        ) * np.sqrt(omega_err ** 2 + gamma_err ** 2)
        T2_max_qutrit_hopper_temp = lambda temp: T2_max_qutrit(
            omega_hopper_lambda(temp), gamma_hopper_lambda(temp)
        )
        T2_max_qutrit_wu_temp = lambda temp: T2_max_qutrit(
            omega_wu_lambda(temp), gamma_wu_lambda(temp)
        )
        if plot_type == "T2_max":
            for sample, linestyle, label in [
                # ("hopper", "dotted", r"$\mathrm{\{\ket{-1}, \ket{+1}\}}$"),
                ("hopper", "dotted", "DQ"),
                ("wu", "dashed", None),
            ]:
                if sample not in rates_to_plot:
                    continue
                func = eval(f"T2_max_qutrit_{sample}_temp")
                if len(rates_to_plot) == 1:
                    linestyle = "solid"
                ax.plot(
                    temp_linspace,
                    func(temp_linspace),
                    label=label,
                    color=qutrit_color,
                    linewidth=line_width,
                    ls=linestyle,
                )

        ax.axvline(x=125, color="silver", zorder=-10)

    if not inset:
        ax.set_xlabel(r"Temperature $\mathit{T}$ (K)")
        if plot_type == "rates":
            ax.set_ylabel(r"Relaxation rates (s$^{-1}$)")
        elif plot_type == "ratios":
            ax.set_ylabel(r"Ratios")
        elif plot_type == "ratio_fits":
            ax.set_ylabel(r"Ratio of fits")
        elif plot_type == "residuals":
            ax.set_ylabel(r"Residuals (s$^{-1}$)")
        elif plot_type == "normalized_residuals":
            ax.set_ylabel(r"Normalized residuals")
        elif plot_type == "T2_max":
            ax.set_ylabel(r"$\mathit{T}_{\mathrm{2,max}}$ (s)")
        elif plot_type == "T2_frac":
            ax.set_ylabel(
                r"$\mathit{T}_{\mathrm{2}} / \mathit{T}_{\mathrm{2,max}}$"
            )
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.set_xlim(min_temp, max_temp)
    if rate_range is not None:
        ax.set_ylim(rate_range[0], rate_range[1])

    # %% Plot the points

    samples = []
    nv_names = []
    markers_list = []

    for point in data_points:

        if "marker" not in point:
            continue
        sample = point[sample_column_title]
        nv_name = point["nv_name"]
        sample_lower = sample.lower()
        marker = point["marker"]

        if nv_name not in nv_names:
            nv_names.append(nv_name)
        if sample not in samples:
            samples.append(sample)
        if marker not in markers_list:
            markers_list.append(marker)
        if sample.lower() not in samples_to_plot:
            continue

        temp = get_temp(point)
        if no_x_errs:
            temp_error = None
        else:
            temp_error = get_temp_error(point)

        if plot_type in ["rates", "residuals", "normalized_residuals"]:
            # Omega
            rate = point[omega_column_title]
            rate_err = point[omega_err_column_title]
            omega_lambda = eval("omega_{}_lambda".format(sample_lower))
            if (rate is not None) and (rates_to_plot in ["both", "Omega"]):
                if plot_type == "rates":
                    val = rate
                    val_err = rate_err
                elif plot_type == "residuals":
                    val = rate - omega_lambda(temp)
                    val_err = rate_err
                elif plot_type == "normalized_residuals":
                    val = (rate - omega_lambda(temp)) / rate_err
                    val_err = 0
                ax.errorbar(
                    temp,
                    val,
                    yerr=val_err,
                    xerr=temp_error,
                    label=r"$\mathrm{\Omega}$",
                    marker=marker,
                    color=omega_edge_color,
                    markerfacecolor=omega_face_color,
                    linestyle="None",
                    ms=marker_size,
                    lw=line_width,
                    markeredgewidth=marker_edge_width,
                )
            # gamma
            rate = point[gamma_column_title]
            rate_err = point[gamma_err_column_title]
            gamma_lambda = eval("gamma_{}_lambda".format(sample_lower))
            if (rate is not None) and (rates_to_plot in ["both", "gamma"]):
                if plot_type == "rates":
                    val = rate
                    val_err = rate_err
                elif plot_type == "residuals":
                    val = rate - gamma_lambda(temp)
                    val_err = rate_err
                elif plot_type == "normalized_residuals":
                    val = (rate - gamma_lambda(temp)) / rate_err
                    val_err = 0
                ax.errorbar(
                    temp,
                    val,
                    yerr=val_err,
                    xerr=temp_error,
                    label=r"$\mathit{\gamma}$",
                    marker=marker,
                    color=gamma_edge_color,
                    markerfacecolor=gamma_face_color,
                    linestyle="None",
                    ms=marker_size,
                    lw=line_width,
                    markeredgewidth=marker_edge_width,
                )
            # print(omega_lambda(475))
            # print(gamma_lambda(475))
            # return

        elif plot_type == "ratios":
            omega_val = point[omega_column_title]
            omega_err = point[omega_err_column_title]
            gamma_val = point[gamma_column_title]
            gamma_err = point[gamma_err_column_title]
            if (omega_val is not None) and (gamma_val is not None):
                ratio = gamma_val / omega_val
                ratio_err = ratio * np.sqrt(
                    (omega_err / omega_val) ** 2 + (gamma_err / gamma_val) ** 2
                )
                ax.errorbar(
                    temp,
                    ratio,
                    yerr=ratio_err,
                    xerr=temp_error,
                    label=r"$\mathit{\gamma}/\mathrm{\Omega}$",
                    marker=marker,
                    color=ratio_edge_color,
                    markerfacecolor=ratio_face_color,
                    linestyle="None",
                    ms=marker_size,
                    lw=line_width,
                )
        # elif plot_type == "T2_max":
        #     omega_val = point[omega_column_title]
        #     omega_err = point[omega_err_column_title]
        #     gamma_val = point[gamma_column_title]
        #     gamma_err = point[gamma_err_column_title]
        #     if (omega_val is not None) and (gamma_val is not None):
        #         qubit_max_val = T2_max_qubit(omega_val, gamma_val)
        #         qubit_max_err = T2_max_qubit_err(
        #             qubit_max_val, omega_err, gamma_err
        #         )
        #         ax.errorbar(
        #             temp,
        #             qubit_max_val,
        #             yerr=qubit_max_err,
        #             xerr=temp_error,
        #             marker=marker,
        #             color=qubit_max_edge_color,
        #             markerfacecolor=qubit_max_face_color,
        #             linestyle="None",
        #             ms=marker_size,
        #             lw=line_width,
        #         )
        #         qutrit_max_val = T2_max_qutrit(omega_val, gamma_val)
        #         qutrit_max_err = T2_max_qutrit_err(
        #             qutrit_max_val, omega_err, gamma_err
        #         )
        #         ax.errorbar(
        #             temp,
        #             qutrit_max_val,
        #             yerr=qutrit_max_err,
        #             xerr=temp_error,
        #             marker=marker,
        #             color=qutrit_max_edge_color,
        #             markerfacecolor=qutrit_max_face_color,
        #             linestyle="None",
        #             ms=marker_size,
        #             lw=line_width,
        #         )

    # Legend x location
    x_loc = 0.14
    # x_loc = 0.16
    # x_loc = 0.22

    # %% Plot past data
    leg0 = None
    plot_past_data = False
    # plot_past_data = True
    if plot_past_data:
        past_results = [
            "redman",
            "takahashi",
            r"jarmola\_s2",
            r"jarmola\_s3",
            r"jarmola\_s8",
        ]
        past_result_markers_list = ["^", "X", "D", "H", "P"]
        past_result_patches = []
        for ind in range(len(past_results)):
            res = past_results[ind]
            marker = past_result_markers_list[ind]
            temps, vals = get_past_results(res)
            ax.plot(
                temps,
                vals,
                label=res,
                marker=marker,
                linestyle="None",
                ms=marker_size,
                markeredgewidth=marker_edge_width,
                color="0.0",  # Black
                markerfacecolor="0.5",  # Dark gray
                zorder=500,
            )
            patch = mlines.Line2D(
                [],
                [],
                color="0.0",
                markerfacecolor="0.5",  # Dark gray
                marker=marker,
                linestyle="None",
                markersize=marker_size,
                markeredgewidth=marker_edge_width,
                label=res,
            )
            past_result_patches.append(patch)
        if yscale == "linear":
            loc = "upper left"
            pos = (0, 0.82)
        elif yscale == "log":
            loc = "lower right"
            pos = None
        leg0 = ax.legend(
            handles=past_result_patches,
            loc=loc,
            title="Past results",
            bbox_to_anchor=pos,
            ncol=2,
        )

    # %% Legend
    leg1 = None
    if not inset:

        if plot_type in ["rates", "residuals", "normalized_residuals"]:
            omega_patch = patches.Patch(
                label=r"$\mathrm{\Omega}$",
                facecolor=omega_face_color,
                edgecolor=omega_edge_color,
                lw=marker_edge_width,
            )
            gamma_patch = patches.Patch(
                label=r"$\mathit{\gamma}$",
                facecolor=gamma_face_color,
                edgecolor=gamma_edge_color,
                lw=marker_edge_width,
            )
            leg1 = ax.legend(
                handles=[omega_patch, gamma_patch],
                loc="upper left",
                title="Rates",
            )

        elif plot_type == "ratios":
            ratio_patch = patches.Patch(
                label=r"$\mathit{\gamma}/\mathrm{\Omega}$",
                facecolor=ratio_face_color,
                edgecolor=ratio_edge_color,
                lw=marker_edge_width,
            )
            leg1 = ax.legend(handles=[ratio_patch], loc="upper left")

        # Samples
        if plot_type in [
            "rates",
            "ratios",
            "residuals",
            "normalized_residuals",
        ]:
            nv_patches = []
            for ind in range(len(markers_list)):
                nv_name = nv_names[ind].replace("_", "\_")
                sample = nv_name.split("-")[0]
                if sample == "prresearch":
                    nv_name = "[1]"
                # else:
                #     label = "New results"
                ls = linestyles[sample]
                if marker_type == "nv":
                    label = nv_name
                    title = "sample-nv"
                elif marker_type == "sample":
                    label = sample[0].upper() + sample[1:]
                    title = "Sample"
                patch = mlines.Line2D(
                    [],
                    [],
                    color="black",
                    marker=markers_list[ind],
                    linestyle=linestyles[sample],
                    markersize=marker_size,
                    markeredgewidth=marker_edge_width,
                    label=label,
                )
                nv_patches.append(patch)
            ax.legend(
                handles=nv_patches,
                loc="upper left",
                title=title,
                # title="Samples",
                bbox_to_anchor=(x_loc, 1.0),
            )

        if leg0 is not None:
            ax.add_artist(leg0)
        if leg1 is not None:
            ax.add_artist(leg1)

        if plot_type == "T2_max":
            handles, labels = ax.get_legend_handles_labels()
            mod_handles = []
            for el in handles:
                mod_handle = mlines.Line2D(
                    [],
                    [],
                    color=el.get_color(),
                    linewidth=line_width,
                    linestyle="solid",
                )
                mod_handles.append(mod_handle)
            leg1 = ax.legend(
                mod_handles,
                labels,
                title="Subspace",
                # bbox_to_anchor=(0.743, 1.0),
                # loc="lower left",
                loc="upper right",
                handlelength=1.5,
                handletextpad=0.5,
                # borderpad=0.3,
                # borderaxespad=0.3,
            )

    fig.tight_layout(pad=0.3)

    if dosave:
        nvdata_dir = common.get_nvdata_dir()
        if plot_type == "T2_max":
            # ext = "png"
            ext = "svg"
            file_path = str(
                nvdata_dir
                / "paper_materials/relaxation_temp_dependence/figures/main4.{}".format(
                    ext
                )
            )
            # fig.savefig(file_path, dpi=500)
            fig.savefig(file_path)
        else:
            timestamp = tool_belt.get_time_stamp()
            datestamp = timestamp.split("-")[0]
            file_name = "{}-{}-{}".format(datestamp, plot_type, yscale)
            file_path = str(
                nvdata_dir
                / "paper_materials"
                / "relaxation_temp_dependence"
                / file_name
            )
            tool_belt.save_figure(fig, file_path)

    if plot_type in ["T2_max", "T2_frac", "T2_supp"]:
        return fig, ax, leg1, T2_max_qubit_hopper_temp
    else:
        return fig, ax, leg1


# %% Run the file


if __name__ == "__main__":

    # temp = 300
    # # delta1 = 4
    # delta1 = 68.2
    # # delta2 = 167
    # # A_1 = 580
    # # A_2 = 9000
    # n1 = bose(delta1, temp)
    # # n2 = bose(delta2, temp)
    # print(n1)
    # # print(A_1 * n1 * (n1 + 1))
    # # print(A_2 * n2 * (n2 + 1))
    # # # print(bose(0.01241, 150))
    # # # print(bose(65, 450) * (bose(65, 450) + 1))
    # # # print(presentation_round_latex(145.88, 26.55))
    # # # print(presentation_round_latex(145.88, 16.55))
    # # # print(presentation_round_latex(145.88, 1.2))
    # # # print(presentation_round_latex(145.88, 6.55))
    # # # print(presentation_round_latex(145.88999, 0.002))
    # # # print(presentation_round_latex(15.88999, 0.00167))
    # # # print(presentation_round_latex(0.0288999, 0.0000167))
    # sys.exit()

    # tool_belt.init_matplotlib()
    kpl.init_kplotlib()
    matplotlib.rcParams["axes.linewidth"] = 1.0

    plot_type = "rates"
    # plot_type = "T2_max"
    # plot_type = "ratios"
    # plot_type = "ratio_fits"
    # plot_type = 'residuals'
    # plot_type = "normalized_residuals"

    rates_to_plot = "both"
    # rates_to_plot = "Omega"
    # rates_to_plot = 'gamma'

    # temp_range = [0, 600]
    temp_range = [0, 480]
    # temp_range = [305, 480]
    # temp_range = [469.5, 475]
    xscale = "linear"
    # temp_range = [1, 500]
    # xscale = "log"

    file_name = "compiled_data"
    # file_name = "compiled_data-single_ref"
    # file_name = "spin_phonon_temp_dependence"
    home = common.get_nvdata_dir()
    path = home / "paper_materials/relaxation_temp_dependence"
    # path = Path.home() / "lab/experimental_update-2022_04_26/"

    if plot_type == "rates":
        # y_params = [[[-10, 1000], "linear"]]
        y_params = [[[-10, 600], "linear"], [[5e-3, 1000], "log"]]
    elif plot_type == "T2_max":
        # y_params = [[[-1, 6], "linear"], [[1e-3, 50], "log"]]
        y_params = [[[5e-4, 50], "log"]]
    elif plot_type == "ratios":
        y_params = [[[0, 5], "linear"]]
    elif plot_type == "ratio_fits":
        y_params = [[[0, 5], "linear"]]
    elif plot_type == "residuals":
        pass
    elif plot_type == "normalized_residuals":
        y_params = [[[-3, 3], "linear"]]
        # rates_to_plot = "Omega"
        rates_to_plot = "gamma"
    # y_params = [y_params[1]]
    # y_params = [[None, "linear"], [[0.001, 20], "log"]]
    # for el in y_params:
    #     y_range, yscale = el
    #     # plot_orbach_scalings(temp_range, xscale, yscale, y_range)
    #     # continue
    #     main(
    #         file_name,
    #         path,
    #         plot_type,
    #         rates_to_plot,
    #         temp_range,
    #         y_range,
    #         xscale,
    #         yscale,
    #         dosave=False,
    #     )

    rates_to_plot = ["both", "both"]
    temp_ranges = [[100, 480], [-10, 490]]
    y_ranges = [[5e-3, 1000], [-20, 650]]
    yscales = ["log", "linear"]
    xscales = ["log", "linear"]
    main(
        file_name,
        path,
        plot_type,
        rates_to_plot,
        temp_ranges,
        y_ranges,
        xscales,
        yscales,
        dosave=False,
    )
    #     print()
    # normalized_residuals_histogram(rates_to_plot)

    # supp_comparison = True
    # supp_comparison = False
    # figure_2(file_name, path, dosave=False, supp_comparison=supp_comparison)

    # # process_to_plot = 'Walker'
    # # process_to_plot = 'Orbach'
    # process_to_plot = 'both'

    # plot_scalings(process_to_plot, temp_range, rate_range, xscale, yscale)

    # May 31st 2021
    # omega_popt = [448.05202972439383, 73.77518971996268, 1.4221406909199286e-11]
    # gamma_popt = [2049.116503275054, 73.77518971996268]
    # plot_T2_max(omega_popt, gamma_popt, temp_range, 'log', 'log')

    plt.show(block=True)

    # print(bose(65,295))
    # print(bose(165,295))

    # Parameter description: [T5_coeff (K^-5 s^-1), omega_exp_coeff (s^-1), gamma_exp_coeff (s^-1), activation (meV)]

    # ZFS:
    # popt: [1.0041e-11 4.7025e+02 1.3495e+03 6.9394e+01]
    # psd: [5.7e-13 6.9e+01 1.6e+02 2.5e+00]

    # Nominal:
    # popt: [7.1350e-12 6.5556e+02 1.6383e+03 7.2699e+01]
    # psd: [5.5e-13 8.4e+01 1.7e+02 2.2e+00]
