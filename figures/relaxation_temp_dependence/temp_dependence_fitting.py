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
marker_edge_width = line_width

gamma_face_color = "#CC99CC"
gamma_edge_color = "#993399"
omega_face_color = "#FFCC33"
omega_edge_color = "#FF9933"
ratio_face_color = "#FB9898"
ratio_edge_color = "#EF2424"
qubit_max_face_color = "#81bfeb"
qubit_max_edge_color = "#1f77b4"
qutrit_max_face_color = "#e5e667"
qutrit_max_edge_color = "#bcbd22"

figsize = [6.5, 5.0]  # default
# figsize = [0.7 * el for el in figsize]

sample_column_title = "Sample"
skip_column_title = "Skip"

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

bad_zfs_temps = 300  # Below this consider zfs temps inaccurate


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

    if res == "redman":
        res_list = [
            476.4177416661098,
            1233.3692525166146,
            451.11715421405614,
            1097.7054740581846,
            427.14250537007416,
            910.991175349638,
            401.70168043525575,
            773.8627950439235,
            372.6536407580296,
            627.4380470122003,
            348.06445892052716,
            496.9998701928061,
            320.67189366418336,
            342.3047734786328,
            295.4432631793261,
            247.00887097166634,
            249.08174616160218,
            128.62095826700775,
            198.8186984331038,
            48.32930238571747,
            168.76315171076862,
            24.58602161557303,
            139.3829999492335,
            10.140815340617547,
            119.91885637433246,
            4.086354429261438,
            99.71082373901912,
            1.465520516930376,
        ]
    elif res == "takahashi":
        res_list = [
            39.708076869092565,
            0.2493223051639581,
            60.18108720206314,
            0.37926901907322497,
            79.61728330307751,
            0.9861140020907845,
            99.71082373901912,
            1.465520516930376,
            149.1639405437168,
            6.072965440947692,
            200.11702947793873,
            28.942661247167475,
            301.43114785530264,
            131.6537392087162,
        ]
    elif res == r"jarmola\_s2":
        res_list = [
            5.052927989240913,
            19.026207406437706,
            9.986632403181524,
            20.4040603404655,
            20.00811570715186,
            21.377627935433164,
            29.90545574361996,
            20.4040603404655,
            39.81280330854725,
            21.377627935433164,
            60.32233775511956,
            19.934031568734756,
            79.75794398477714,
            19.47483039908755,
            120.85851898425719,
            21.881695573048844,
            198.86531379873264,
            71.82494525483934,
            252.5578517930507,
            191.15097529077795,
            295.50030798309376,
            342.3047734786328,
            325.09776992170424,
            393.6785028384848,
            350.4582003215088,
            532.9919475799622,
            388.22259119422284,
            688.74217883767,
            407.2346885164248,
            849.4734262896834,
            479.68759119360243,
            1292.218732318354,
        ]
    elif res == r"jarmola\_s3":
        res_list = [
            6.427375637623035,
            0.007386199822079366,
            9.937587857375618,
            0.004969998701928071,
            19.916720690963206,
            0.0093247163010515,
            39.91505366735992,
            0.0163136299694379,
            49.63491901713573,
            0.014861572940077342,
            69.84021996713274,
            0.0676019952476119,
            78.96668063167132,
            0.09368281219580879,
            94.35441700897722,
            0.4361904103306001,
            119.05215177843596,
            1.9384267426027304,
            160.81553767084844,
            9.678988180373436,
            202.87342606577826,
            31.77051768482748,
            297.39307938738887,
            166.206411641365,
            324.9857097772839,
            219.83926488622868,
            350.3373985191751,
            297.6351441631313,
            375.0871532725971,
            375.74982364374284,
            401.58536640626534,
            474.3657889100621,
            424.1313933316525,
            585.0681784874581,
            451.01762892382345,
            756.0360599248723,
            479.5883763012125,
            910.991175349638,
        ]
    elif res == r"jarmola\_s8":
        res_list = [
            9.950478138502218,
            0.04443992111457811,
            19.80060613286093,
            0.047658201793835606,
            29.800477508344034,
            0.053548207704057896,
            39.67414115937397,
            0.05878016072274918,
            49.680800221318,
            0.07082758421520119,
            59.72553858233379,
            0.10046719832671451,
            79.56130794778521,
            0.30042273359919514,
            119.89239992311232,
            2.8144446531124894,
            160.83993406671632,
            12.507369836566506,
            200.15014817692008,
            38.28211745620105,
            252.45338834977025,
            95.00200990674503,
            293.3848300819487,
            182.4456878822947,
            322.81046691031594,
            258.794752234262,
            352.7565006568838,
            334.4194266628009,
            429.97927603430963,
            657.3758799840733,
            479.60160376778975,
            954.458577073744,
            375.1233628312281,
            442.33264045108064,
            398.8864136467382,
            532.9919475799622,
        ]

    temps = []
    vals = []
    num_pairs = len(res_list) // 2
    for ind in range(num_pairs):
        temps.append(res_list[ind * 2])
        raw_val = res_list[(ind * 2) + 1]
        omega = raw_val / 3
        vals.append(omega)

    return temps, vals


def presentation_round(val, err):
    err_mag = math.floor(math.log10(err))
    sci_err = err / (10 ** err_mag)
    first_err_digit = int(str(sci_err)[0])
    if first_err_digit == 1:
        err_sig_figs = 2
    else:
        err_sig_figs = 1
    power_of_10 = math.floor(math.log10(val))
    mag = 10 ** power_of_10
    rounded_err = tool_belt.round_sig_figs(err, err_sig_figs) / mag
    rounded_val = round(val / mag, (power_of_10 - err_mag) + err_sig_figs - 1)
    return [rounded_val, rounded_err, power_of_10]


def presentation_round_latex(val, err):
    if val <= 0 or err > val:
        return ""
    rounded_val, rounded_err, power_of_10 = presentation_round(val, err)
    err_mag = math.floor(math.log10(rounded_err))
    val_mag = math.floor(math.log10(rounded_val))

    # Turn 0.0000016 into 0.16
    # The round is to deal with floating point leftovers eg 9 = 9.00000002
    shifted_rounded_err = round(rounded_err / 10 ** (err_mag + 1), 5)
    # - 1 to remove the "0." part
    err_last_decimal_mag = len(str(shifted_rounded_err)) - 2
    pad_val_to = -err_mag + err_last_decimal_mag

    if err_mag > val_mag:
        return 1 / 0
    elif err_mag == val_mag:
        print_err = rounded_err
    else:
        print_err = int(str(shifted_rounded_err).replace(".", ""))

    str_val = str(rounded_val)
    decimal_pos = str_val.find(".")
    num_padding_zeros = pad_val_to - len(str_val[decimal_pos:])
    padded_val = str(rounded_val) + "0" * num_padding_zeros
    # return "{}({})e{}".format(padded_val, print_err, power_of_10)
    return r"\num{{{}({})e{}}}".format(padded_val, print_err, power_of_10)


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
    if nominal_temp < bad_zfs_temps:
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

    fit_func = simultaneous_test_lambda

    # region DECLARE FIT FUNCTIONS HERE

    mode = "T5"
    mode = "double_orbach"
    mode = "other"

    # T5 fixed + constant
    if mode in ["T5", "other"]:
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
    if mode in ["double_orbach", "other"]:
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
    data = data = RealData(temps, combined_rates, temp_errors, combined_errs)
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
        "s",
        "^",
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
            if (temp_range is None) or not (
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

    ax.set_xlabel(r"T (K)")
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

    ax.set_xlabel(r"T (K)")
    ax.set_ylabel(r"$T_{2,\text{max}}$ (s)")
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
    ax.plot(norm_res_linspace, norm_gaussian(norm_res_linspace))

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
    ax.set_xlabel(r"T (K)")
    if y_range is not None:
        ax.set_ylim(y_range[0], y_range[1])
    if normalized:
        ax.set_ylabel(r"Normalized Orbach scaling")
    else:
        ax.set_ylabel(r"Orbach scaling")
    fig.tight_layout(pad=0.3)
    return


def figure_2(
    file_name,
    path,
    dosave=False,
):

    data_points = get_data_points(path, file_name, temp_range)
    fig, ax = plt.subplots(figsize=figsize)

    for fit_mode in ["double_orbach", "T5"]:
        figure_2_sub(ax, data_points, fit_mode)

    if dosave:
        timestamp = tool_belt.get_time_stamp()
        datestamp = timestamp.split("-")[0]
        file_name = "{}-{}-{}".format(datestamp, plot_type, yscale)
        nvdata_dir = common.get_nvdata_dir()
        file_path = str(
            nvdata_dir
            / "paper_materials"
            / "relaxation_temp_dependence"
            / file_name
        )
        tool_belt.save_figure(fig, file_path)


def figure_2_sub(
    ax,
    data_points,
    fit_mode,
):

    # %% Setup

    temp_range = [0, 480]
    samples_to_plot = ["hopper", "wu"]
    linestyles = {"hopper": "dotted", "wu": "dashed"}
    marker_type = "sample"
    # marker_type = "nv"

    min_temp = temp_range[0]
    max_temp = temp_range[1]

    temp_linspace = np.linspace(min_temp, max_temp, 1000)
    ax.set_xlim(min_temp, max_temp)

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

    # Plot setup
    ax.set_xlabel(r"T (K)")
    ax.set_ylabel(r"Relaxation rates (s$^{-1}$)")
    ax.set_yscale("log")
    rate_range = None
    if rate_range is not None:
        ax.set_ylim(rate_range[0], rate_range[1])

    # Plot the rate fits
    for sample in samples_to_plot:
        fit_func = eval("omega_{}_lambda".format(sample))
        ls = linestyles[sample]
        ax.plot(
            temp_linspace,
            fit_func(temp_linspace),
            linestyle=ls,
            label=r"$\Omega$ fit",
            color=omega_edge_color,
            linewidth=line_width,
        )
    for sample in samples_to_plot:
        fit_func = eval("gamma_{}_lambda".format(sample))
        ls = linestyles[sample]
        ax.plot(
            temp_linspace,
            fit_func(temp_linspace),
            linestyle=ls,
            label=r"$\gamma$ fit",
            color=gamma_edge_color,
            linewidth=line_width,
        )

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
            label=r"$\Omega$",
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
        val = rate
        val_err = rate_err
        ax.errorbar(
            temp,
            val,
            yerr=val_err,
            xerr=temp_error,
            label=r"$\gamma$",
            marker=marker,
            color=gamma_edge_color,
            markerfacecolor=gamma_face_color,
            linestyle="None",
            ms=marker_size,
            lw=line_width,
            markeredgewidth=marker_edge_width,
        )

    # Rate legend
    omega_patch = patches.Patch(
        label=r"$\Omega$",
        facecolor=omega_face_color,
        edgecolor=omega_edge_color,
        lw=marker_edge_width,
    )
    gamma_patch = patches.Patch(
        label=r"$\gamma$",
        facecolor=gamma_face_color,
        edgecolor=gamma_edge_color,
        lw=marker_edge_width,
    )
    leg1 = ax.legend(
        handles=[omega_patch, gamma_patch], loc="upper left", title="Rates"
    )

    # Sample legend
    x_loc = 0.14
    # x_loc = 0.16
    # x_loc = 0.22
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

    # Inset plot of normalized residuals

    axins_gamma = inset_axes(
        ax,
        width="100%",
        height="100%",
        # borderpad=1.5,
        # bbox_to_anchor=(0.9, 0.8, 0.5, 0.5),
        bbox_to_anchor=(0.6, 0.53, 0.4, 0.4),
        bbox_transform=ax.transAxes,
        loc=1,
    )
    axins_omega = inset_axes(
        ax,
        width="100%",
        height="100%",
        bbox_to_anchor=(0.6, 0.13, 0.4, 0.4),
        bbox_transform=ax.transAxes,
        loc=1,
    )

    for rate in ["gamma", "omega"]:
        if rate == "gamma":
            title = r"$\gamma$ Normalized residual"
        if rate == "omega":
            title = r"$\Omega$ Normalized residual"
        axins = eval("axins_{}".format(rate))
        axins.set_ylabel(title)
        axins.set_xlabel(r"T(K)")
        axins.set_xlim(min_temp, max_temp)
        axins.set_ylim(-3.25, 3.25)

    samples = []
    nv_names = []
    markers_list = []
    max_norm_err = 0

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

        # Omega
        rate = point[omega_column_title]
        rate_err = point[omega_err_column_title]
        omega_lambda = eval("omega_{}_lambda".format(sample_lower))
        val = (rate - omega_lambda(temp)) / rate_err
        if abs(val) > max_norm_err:
            max_norm_err = abs(val)
        axins_omega.scatter(
            temp,
            val,
            label=r"$\Omega$",
            marker=marker,
            edgecolor=omega_edge_color,
            facecolor=omega_face_color,
            linestyle="None",
            s=marker_size,
            # lw=line_width,
            # markeredgewidth=marker_edge_width,
            linewidth=marker_edge_width,
        )

        # gamma
        rate = point[gamma_column_title]
        rate_err = point[gamma_err_column_title]
        gamma_lambda = eval("gamma_{}_lambda".format(sample_lower))
        val = (rate - gamma_lambda(temp)) / rate_err
        if abs(val) > max_norm_err:
            max_norm_err = abs(val)
        axins_gamma.scatter(
            temp,
            val,
            label=r"$\gamma$",
            marker=marker,
            edgecolor=gamma_edge_color,
            facecolor=gamma_face_color,
            s=marker_size,
            # lw=line_width,
            # markeredgewidth=marker_edge_width,
            linewidth=marker_edge_width,
        )

    print(max_norm_err)

    # Final steps

    if leg1 is not None:
        ax.add_artist(leg1)
    fig.tight_layout(pad=0.3)


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

    marker_type = "sample"
    # marker_type = "nv"
    data_points = get_data_points(path, file_name, temp_range)

    min_temp = temp_range[0]
    max_temp = temp_range[1]

    temp_linspace = np.linspace(min_temp, max_temp, 1000)
    fig, ax = plt.subplots(figsize=figsize)

    # Fit to Omega and gamma simultaneously
    (
        popt,
        pvar,
        beta_desc,
        omega_hopper_fit_func,
        omega_wu_fit_func,
        gamma_hopper_fit_func,
        gamma_wu_fit_func,
    ) = fit_simultaneous(data_points)

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
                label=r"$\Omega$ fit",
                color=omega_edge_color,
                linewidth=line_width,
            )
        # Plot Jarmola 2012 Eq. 1 for S3
        # ax.plot(temp_linspace, omega_calc(temp_linspace),
        #         label=r'$\Omega$ fit', color=omega_edge_color)

    if (plot_type == "rates") and (rates_to_plot in ["both", "gamma"]):
        for sample in samples_to_plot:
            fit_func = eval("gamma_{}_lambda".format(sample))
            ls = linestyles[sample]
            ax.plot(
                temp_linspace,
                fit_func(temp_linspace),
                linestyle=ls,
                label=r"$\gamma$ fit",
                color=gamma_edge_color,
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
        for func in [ratio_hopper_lambda, ratio_wu_lambda]:
            ax.plot(
                temp_linspace,
                func(temp_linspace),
                label=r"$\gamma/\Omega$",
                color=gamma_edge_color,
                linewidth=line_width,
            )
    if plot_type == "T2_max":
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
        for func in [T2_max_qubit_hopper_temp, T2_max_qubit_wu_temp]:
            ax.plot(
                temp_linspace,
                func(temp_linspace),
                label=r"Superposition of $\ket{0}$, $\ket{\pm 1}$",
                # label=r"Qubit T2 max",
                color=qubit_max_edge_color,
                linewidth=line_width,
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
        for func in [T2_max_qutrit_hopper_temp, T2_max_qutrit_wu_temp]:
            ax.plot(
                temp_linspace,
                func(temp_linspace),
                label=r"Superposition of $\ket{-1}$, $\ket{+1}$",
                # label=r"Qutrit T2 max",
                color=qutrit_max_edge_color,
                linewidth=line_width,
            )

    # ax.plot(temp_linspace, orbach(temp_linspace) * 0.7, label='Orbach')
    # ax.plot(temp_linspace, raman(temp_linspace)/3, label='Raman')

    ax.set_xlabel(r"T (K)")
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
        ax.set_ylabel(r"$T_{2,\text{max}}$ (s)")
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
                    label=r"$\Omega$",
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
                    label=r"$\gamma$",
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
                    label=r"$\gamma/\Omega$",
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

    if plot_type in ["rates", "residuals", "normalized_residuals"]:
        omega_patch = patches.Patch(
            label=r"$\Omega$",
            facecolor=omega_face_color,
            edgecolor=omega_edge_color,
            lw=marker_edge_width,
        )
        gamma_patch = patches.Patch(
            label=r"$\gamma$",
            facecolor=gamma_face_color,
            edgecolor=gamma_edge_color,
            lw=marker_edge_width,
        )
        leg1 = ax.legend(
            handles=[omega_patch, gamma_patch], loc="upper left", title="Rates"
        )

    elif plot_type == "ratios":
        ratio_patch = patches.Patch(
            label=r"$\gamma/\Omega$",
            facecolor=ratio_face_color,
            edgecolor=ratio_edge_color,
            lw=marker_edge_width,
        )
        leg1 = ax.legend(handles=[ratio_patch], loc="upper left")

    # Samples
    if plot_type in ["rates", "ratios", "residuals", "normalized_residuals"]:
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
        ax.legend()

    fig.tight_layout(pad=0.3)

    if dosave:
        timestamp = tool_belt.get_time_stamp()
        datestamp = timestamp.split("-")[0]
        file_name = "{}-{}-{}".format(datestamp, plot_type, yscale)
        nvdata_dir = common.get_nvdata_dir()
        file_path = str(
            nvdata_dir
            / "paper_materials"
            / "relaxation_temp_dependence"
            / file_name
        )
        tool_belt.save_figure(fig, file_path)


# %% Run the file


if __name__ == "__main__":

    # print(bose(65, 450))
    # print(bose(65, 450) * (bose(65, 450) + 1))
    # print(presentation_round_latex(145.88, 26.55))
    # print(presentation_round_latex(145.88, 16.55))
    # print(presentation_round_latex(145.88, 1.2))
    # print(presentation_round_latex(145.88, 6.55))
    # print(presentation_round_latex(145.88999, 0.002))
    # print(presentation_round_latex(15.88999, 0.00167))
    # print(presentation_round_latex(0.0288999, 0.0000167))
    # sys.exit()

    tool_belt.init_matplotlib()
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
        y_params = [[[-1, 6], "linear"], [[1e-3, 50], "log"]]
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
    #     print()
    # normalized_residuals_histogram(rates_to_plot)

    figure_2(
        file_name,
        path,
        dosave=False,
    )

    # # process_to_plot = 'Walker'
    # # process_to_plot = 'Orbach'
    # process_to_plot = 'both'

    # plot_scalings(process_to_plot, temp_range, rate_range, xscale, yscale)

    # May 31st 2021
    # omega_popt = [448.05202972439383, 73.77518971996268, 1.4221406909199286e-11]
    # gamma_popt = [2049.116503275054, 73.77518971996268]
    # plot_T2_max(omega_popt, gamma_popt, temp_range, 'log', 'log')

    plt.show(block=True)

    # Parameter description: [T5_coeff (K^-5 s^-1), omega_exp_coeff (s^-1), gamma_exp_coeff (s^-1), activation (meV)]

    # ZFS:
    # popt: [1.0041e-11 4.7025e+02 1.3495e+03 6.9394e+01]
    # psd: [5.7e-13 6.9e+01 1.6e+02 2.5e+00]

    # Nominal:
    # popt: [7.1350e-12 6.5556e+02 1.6383e+03 7.2699e+01]
    # psd: [5.5e-13 8.4e+01 1.7e+02 2.2e+00]
