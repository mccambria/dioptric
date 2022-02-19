# -*- coding: utf-8 -*-
"""
Reproduce Jarmola 2012 temperature scalings

Created on Fri Jun 26 17:40:09 2020

@author: matth
"""


# %% Imports


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
# temp_model = "Barson"
temp_model = "comp"
temp_column_title = "ZFS temp, {} (K)".format(temp_model)
# temp_column_title = "Nominal temp (K)"
temp_lb_column_title = "ZFS temp, lb, {} (K)".format(temp_model)
temp_ub_column_title = "ZFS temp, ub, {} (K)".format(temp_model)

low_res_file_column_title = "-1 resonance file"
high_res_file_column_title = "+1 resonance file"

omega_column_title = "Omega (s^-1)"
omega_err_column_title = "Omega err (s^-1)"
gamma_column_title = "gamma (s^-1)"
gamma_err_column_title = "gamma err (s^-1)"

bad_zfs_temps = 300.1  # Below this consider zfs temps inaccurate


# %% Processes and sum functions


def bose(energy, temp):
    # For very low temps we can get divide by zero and overflow warnings.
    # Fortunately, numpy is smart enough to know what we mean when this
    # happens, so let's let numpy figure it out and suppress the warnings.
    old_settings = np.seterr(divide="ignore", over="ignore")
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
    return (
        const
        + (coeff_orbach * bose(activation, temp))
        + (coeff_T5 * temp ** 5)
    )


def double_orbach(temp, coeff1, delta1, coeff2, delta2, const):
    return (
        const + (coeff1 * bose(delta1, temp)) + (coeff2 * bose(delta2, temp))
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


def omega_calc(temp):
    popt = [421.88, 69.205, 1.1124e-11]
    return orbach_T5_free(temp, *popt)


def gamma_calc(temp):
    popt = [1357.2, 69.205, 9.8064e-12]
    return orbach_T5_free(temp, *popt)


def get_temp(point):
    nominal_temp = point[nominal_temp_column_title]
    if nominal_temp <= bad_zfs_temps:
        temp = nominal_temp
    else:
        temp = point[temp_column_title]
        if temp == "":
            temp = point[nominal_temp_column_title]
    return temp


def get_temp_bounds(point):
    if temp_lb_column_title == nominal_temp_column_title:
        return None
    nominal_temp = point[nominal_temp_column_title]
    if nominal_temp <= bad_zfs_temps:
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
    temps, beta, omega_rate_lambda, gamma_rate_lambda
):
    """
    Lambda variation of simultaneous_test
    """

    ret_vals = []
    num_vals = len(temps)
    for ind in range(num_vals):
        temp_val = temps[ind]
        # Omegas are even indexed
        if ind % 2 == 0:
            ret_vals.append(omega_rate_lambda(temp_val, beta))
        # gammas are odd indexed
        else:
            ret_vals.append(gamma_rate_lambda(temp_val, beta))

    return np.array(ret_vals)


def fit_simultaneous(data_points):

    # To fit to Omega and gamma simultaneously, set up a combined list of the
    # rates. Parity determines which rate is where. Even is Omega, odd is
    # gamma.
    temps = []
    temp_errors = []
    combined_rates = []
    combined_errs = []
    for point in data_points:
        # Crash if we're trying to work with incomplete data
        if (point[omega_column_title] is None) or (
            point[gamma_column_title] is None
        ):
            crash = 1 / 0
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

    fit_func = simultaneous_test_lambda

    # region DECLARE FIT FUNCTIONS HERE

    # Just exp
    # init_params = (510, 1.38e-11, 2000, 72.0)
    # omega_fit_func = lambda temp, beta: orbach_T5_free(
    #     temp, beta[0], beta[3], beta[1]
    # )
    # gamma_fit_func = lambda temp, beta: orbach_free(temp, beta[2], beta[3])
    # beta_desc = (
    #     "[omega_exp_coeff (s^-1), omega_T5_coeff (K^-5 s^-1), gamma_exp_coeff"
    #     " (s^-1), activation (meV)]"
    # )

    # T5 free
    # init_params = (510, 1.38e-11, 2000, 1.38e-11, 72.0)
    # omega_fit_func = lambda temp, beta: orbach_T5_free(
    #     temp, beta[0], beta[4], beta[1]
    # )
    # gamma_fit_func = lambda temp, beta: orbach_T5_free(
    #     temp, beta[2], beta[4], beta[3]
    # )
    # beta_desc = (
    #     "[T5_coeff (K^-5 s^-1), omega_exp_coeff (s^-1), gamma_exp_coeff"
    #     " (s^-1), activation (meV)]"
    # )

    # T5 fixed + linear
    # init_params = (1.38e-11, 510, 2000, 72.0, 0.07, 0.035)
    # omega_fit_func = lambda temp, beta: orbach_T5_free_linear(
    #     temp, beta[1], beta[3], beta[0], beta[4]
    # )
    # gamma_fit_func = lambda temp, beta: orbach_T5_free_linear(
    #     temp, beta[2], beta[3], beta[0], beta[5]
    # )
    # beta_desc = (
    #     "[T5_coeff (K^-5 s^-1), omega_exp_coeff (s^-1), gamma_exp_coeff"
    #     " (s^-1), activation (meV), linear_coeff (K^-1 s^-1)]"
    # )

    # T5 fixed + constant
    # init_params = (1.38e-11, 510, 2000, 72.0, 0.01, 0.07)
    # omega_fit_func = lambda temp, beta: orbach_T5_free_const(
    #     temp, beta[1], beta[3], beta[0], beta[4]
    # )
    # gamma_fit_func = lambda temp, beta: orbach_T5_free_const(
    #     temp, beta[2], beta[3], beta[0], beta[5]
    # )
    # beta_desc = [
    #     "T5_coeff (K^-5 s^-1)",
    #     "omega_exp_coeff (s^-1)",
    #     "gamma_exp_coeff (s^-1)",
    #     "activation (meV)",
    #     "Omega constant (K^-1 s^-1)",
    #     "gamma constant (K^-1 s^-1)",
    # ]

    # Double Orbach
    # init_params = (500, 1500, 72, 2000, 2000, 400, 0.01, 0.07)
    # # init_params = (500, 1500, 72, 2000, 2000, 0.01, 0.07)
    # omega_fit_func = lambda temp, beta: double_orbach(
    #     temp,
    #     beta[0],
    #     beta[2],
    #     beta[3],
    #     beta[-3],
    #     beta[-2],  # 400, beta[5]
    # )
    # gamma_fit_func = lambda temp, beta: double_orbach(
    #     temp,
    #     beta[1],
    #     beta[2],
    #     beta[4],
    #     beta[5],
    #     beta[7],  # 400, beta[6]
    # )
    # beta_desc = [
    #     "Omega Orbach 1 coeff (s^-1)",
    #     "gamma Orbach 1 coeff (s^-1)",
    #     "Orbach 1 Delta (meV)",
    #     "Omega Orbach 2 coeff (s^-1)",
    #     "gamma Orbach 2 coeff (s^-1)",
    #     "Orbach 2 Delta (meV)",
    #     "Omega constant (K^-1 s^-1)",
    #     "gamma constant (K^-1 s^-1)",
    # ]

    # Double Orbach fixed
    init_params = (450, 1200, 65, 11000, 160, 0.01, 0.07)
    omega_fit_func = lambda temp, beta: double_orbach(
        temp,
        beta[0],
        beta[2],
        beta[3],
        beta[4],
        beta[5],
    )
    gamma_fit_func = lambda temp, beta: double_orbach(
        temp,
        beta[1],
        beta[2],
        beta[3],
        beta[4],
        beta[6],
    )
    beta_desc = [
        "Omega Orbach 1 coeff (s^-1)",
        "gamma Orbach 1 coeff (s^-1)",
        "Orbach 1 Delta (meV)",
        "Orbach 2 coeff (s^-1)",
        "Orbach 2 Delta (meV)",
        "Omega constant (K^-1 s^-1)",
        "gamma constant (K^-1 s^-1)",
    ]

    # Double Orbach fixed
    # orbach1_delta = 60
    # orbach2_delta = 400
    # init_params = (500, 1500, 2000, 60, 0.01, 0.07)
    # omega_fit_func = lambda temp, beta: double_orbach(
    #     temp,
    #     beta[0],
    #     beta[3],
    #     # orbach1_delta,
    #     beta[2],
    #     orbach2_delta,
    #     beta[-2],
    # )
    # gamma_fit_func = lambda temp, beta: double_orbach(
    #     temp,
    #     beta[1],
    #     beta[3],
    #     # orbach1_delta,
    #     beta[2],
    #     orbach2_delta,
    #     beta[-1],
    # )
    # beta_desc = (
    #     "Omega Orbach 1 coeff (s^-1), gamma Orbach 1 coeff (s^-1), Omega"
    #     " Orbach 2 coeff (s^-1), gamma Orbach 2 coeff (s^-1), Omega constant"
    #     " (K^-1 s^-1), gamma constant (K^-1 s^-1)]"
    # )

    # T5 fixed
    # init_params = (1.38e-11, 510, 2000, 72.0)
    # omega_fit_func = lambda temp, beta: orbach_T5_free(
    #     temp, beta[1], beta[3], beta[0]
    # )
    # gamma_fit_func = lambda temp, beta: orbach_T5_free(
    #     temp, beta[2], beta[3], beta[0]
    # )
    # beta_desc = (
    #     "[T5_coeff (K^-5 s^-1), omega_exp_coeff (s^-1), gamma_exp_coeff"
    #     " (s^-1), activation (meV)]"
    # )

    # T7
    # init_params = (510, 1.38e-11, 2000, 1.38e-15, 72.0)
    # omega_fit_func = lambda temp, beta: orbach_T5_free(
    #     temp, beta[0], beta[4], beta[1]
    # )
    # gamma_fit_func = lambda temp, beta: orbach_T7_free(
    #     temp, beta[2], beta[4], beta[3]
    # )
    # beta_desc = (
    #     "[omega_exp_coeff (s^-1), omega_T5_coeff (K^-5 s^-1), gamma_exp_coeff"
    #     " (s^-1), gamma_T7_coeff (K^-7 s^-1), activation (meV)]"
    # )

    # Ariel
    # init_params = (2000,)
    # ariel_params = (653, 73, 6.87e-12)
    # omega_fit_func = lambda temp, beta: orbach_T5_free(temp, *ariel_params)
    # gamma_fit_func = lambda temp, beta: orbach_T5_free(
    #     temp, beta[0], *ariel_params[1:]
    # )
    # beta_desc = "[gamma_exp_coeff]"

    # endregion

    fit_func = lambda beta, temp: simultaneous_test_lambda(
        temp, beta, omega_fit_func, gamma_fit_func
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

    return popt, np.diag(pcov), beta_desc, omega_fit_func, gamma_fit_func


def gen_fake_data_point(temp, omega, gamma, sample_normal=False):

    fake_point = {}

    error_level = 0.05
    omega_err = error_level * omega
    gamma_err = error_level * gamma

    # Generate the rates according to whether we want to simulate the
    # effects of the error bars or use precise values
    sample_normal = True
    if sample_normal:
        omega_sample = np.random.normal(omega, omega_err)
        gamma_sample = np.random.normal(gamma, gamma_err)
        omega_sample = tool_belt.round_sig_figs(omega_sample, 3)
        gamma_sample = tool_belt.round_sig_figs(gamma_sample, 3)
        # print("{}, {}".format(omega_sample, gamma_sample))
        fake_point = {
            omega_column_title: omega_sample,
            gamma_column_title: gamma_sample,
        }
    else:
        fake_point = {
            omega_column_title: omega,
            gamma_column_title: gamma,
        }

    # Generate the common properties
    common = {
        "marker": "D",
        sample_column_title: "FAKE",
        nominal_temp_column_title: temp,
        temp_column_title: temp,
        temp_lb_column_title: temp - 3,
        temp_ub_column_title: temp + 3,
        omega_err_column_title: omega_err,
        gamma_err_column_title: gamma_err,
    }
    fake_point.update(common)

    return fake_point


def get_data_points(path, file_name):

    file_path = path / "{}.xlsx".format(file_name)
    csv_file_path = path / "{}.csv".format(file_name)

    file = pd.read_excel(file_path, engine="openpyxl")
    file.to_csv(csv_file_path, index=None, header=True)

    # Marker and color combination to distinguish samples
    marker_ind = 0
    markers = [
        "o",
        "s",
        "^",
        "X",
    ]

    data_points = []
    samples = []
    sample_markers = {}
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
            # The first row should be populated for every data point. If it's
            # not, then assume we're looking at a padding row at the bottom
            # of the csv
            if sample == "":
                continue
            if sample not in samples:
                sample_markers[sample] = markers[marker_ind]
                marker_ind += 1
                samples.append(sample)
            point["marker"] = sample_markers[sample]
            point[sample_column_title] = sample
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
            if not point[skip_column_title]:
                data_points.append(point)
            # data_points.append(point)

    # The first shall be last
    data_points.append(data_points.pop(0))

    # Generate fake data for high temp tests

    # Genned from Orbach + T5
    # data_points.append(gen_fake_data_point(487.5, 382, 603))
    # data_points.append(gen_fake_data_point(500.0, 423, 656))
    # data_points.append(gen_fake_data_point(512.5, 469, 713))
    # data_points.append(gen_fake_data_point(525.0, 518, 775))
    # data_points.append(gen_fake_data_point(537.5, 572, 841))
    # data_points.append(gen_fake_data_point(550.0, 630, 911))
    # Genned from double Orbach
    # data_points.append(gen_fake_data_point(487.5, 354, 557))
    # data_points.append(gen_fake_data_point(500.0, 383, 596))
    # data_points.append(gen_fake_data_point(512.5, 413, 637))
    # data_points.append(gen_fake_data_point(525.0, 445, 680))
    # data_points.append(gen_fake_data_point(537.5, 478, 723))
    # data_points.append(gen_fake_data_point(550.0, 511, 767))

    # Pre-sampled normal distribution
    # Genned from Orbach + T5
    # data_points.append(gen_fake_data_point(487.5, 443.0, 631.0))
    # data_points.append(gen_fake_data_point(500.0, 356.0, 628.0))
    # data_points.append(gen_fake_data_point(512.5, 446.0, 704.0))
    # data_points.append(gen_fake_data_point(525.0, 519.0, 686.0))
    # data_points.append(gen_fake_data_point(537.5, 527.0, 768.0))
    # data_points.append(gen_fake_data_point(550.0, 654.0, 939.0))
    # Genned from double Orbach
    # data_points.append(gen_fake_data_point(487.5, 331.0, 556.0))
    # data_points.append(gen_fake_data_point(500.0, 413.0, 586.0))
    # data_points.append(gen_fake_data_point(512.5, 382.0, 704.0))
    # data_points.append(gen_fake_data_point(525.0, 508.0, 728.0))
    # data_points.append(gen_fake_data_point(537.5, 474.0, 791.0))
    # data_points.append(gen_fake_data_point(550.0, 515.0, 843.0))

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

    data_points = get_data_points(path, file_name)

    min_temp = temp_range[0]
    max_temp = temp_range[1]

    temp_linspace = np.linspace(min_temp, max_temp, 1000)
    fig, ax = plt.subplots(figsize=figsize)
    # ax.plot([1, 3, 2, 4, 3, 5])
    # return

    # Fit to Omega and gamma simultaneously
    popt, pvar, beta_desc, omega_fit_func, gamma_fit_func = fit_simultaneous(
        data_points
    )

    # omega_lambda = lambda temp: orbach_free(temp, 5.4603e02, 71)
    # gamma_lambda = lambda temp: orbach_free(temp, 1.5312e03, 71)
    # omega_lambda = lambda temp: orbach_free(temp, 1e8, 400)
    # gamma_lambda = omega_lambda
    omega_lambda = lambda temp: omega_fit_func(temp, popt)
    gamma_lambda = lambda temp: gamma_fit_func(temp, popt)

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
    if (plot_type == "rates") and (rates_to_plot in ["both", "Omega"]):
        ax.plot(
            temp_linspace,
            omega_lambda(temp_linspace),
            label=r"$\Omega$ fit",
            color=omega_edge_color,
            linewidth=line_width,
        )
        # Plot Jarmola 2012 Eq. 1 for S3
        # ax.plot(temp_linspace, omega_calc(temp_linspace),
        #         label=r'$\Omega$ fit', color=omega_edge_color)

    if (plot_type == "rates") and (rates_to_plot in ["both", "gamma"]):
        ax.plot(
            temp_linspace,
            gamma_lambda(temp_linspace),
            label=r"$\gamma$ fit",
            color=gamma_edge_color,
            linewidth=line_width,
        )
    # print(omega_lambda(50))
    # print(gamma_lambda(50))

    # Plot ratio
    ratio_lambda = lambda temp: gamma_lambda(temp) / omega_lambda(temp)
    if plot_type in ["ratios", "ratio_fits"]:
        ax.plot(
            temp_linspace,
            ratio_lambda(temp_linspace),
            label=r"$\gamma/\Omega$",
            color=gamma_edge_color,
            linewidth=line_width,
        )
    if plot_type == "T2_max":
        T2_max_qubit = lambda omega, gamma: 2 / (3 * omega + gamma)
        T2_max_qubit_temp = lambda temp: T2_max_qubit(
            omega_lambda(temp), gamma_lambda(temp)
        )
        T2_max_qubit_err = lambda T2max, omega_err, gamma_err: (
            (T2max ** 2) / 2
        ) * np.sqrt((3 * omega_err) ** 2 + gamma_err ** 2)
        ax.plot(
            temp_linspace,
            T2_max_qubit_temp(temp_linspace),
            label=r"Superposition of $\ket{0}$, $\ket{\pm 1}$",
            # label=r"Qubit T2 max",
            color=qubit_max_edge_color,
            linewidth=line_width,
        )
        T2_max_qutrit = lambda omega, gamma: 1 / (omega + gamma)
        T2_max_qutrit_err = lambda T2max, omega_err, gamma_err: (
            T2max ** 2
        ) * np.sqrt(omega_err ** 2 + gamma_err ** 2)
        T2_max_qutrit_temp = lambda temp: T2_max_qutrit(
            omega_lambda(temp), gamma_lambda(temp)
        )
        ax.plot(
            temp_linspace,
            T2_max_qutrit_temp(temp_linspace),
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
    markers = []

    for point in data_points:

        sample = point[sample_column_title]
        marker = point["marker"]

        if sample not in samples:
            samples.append(sample)
        if marker not in markers:
            markers.append(marker)

        temp = get_temp(point)
        temp_error = get_temp_error(point)

        if plot_type in ["rates", "residuals", "normalized_residuals"]:
            # Omega
            rate = point[omega_column_title]
            rate_err = point[omega_err_column_title]
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
        sample_patches = []
        for ind in range(len(samples)):
            label = samples[ind]
            if label == "PRResearch":
                label = "[1]"
            # else:
            #     label = "New results"
            patch = mlines.Line2D(
                [],
                [],
                color="black",
                marker=markers[ind],
                linestyle="None",
                markersize=marker_size,
                markeredgewidth=marker_edge_width,
                label=label,
            )
            sample_patches.append(patch)
        x_loc = 0.14
        # x_loc = 0.16
        # x_loc = 0.22
        ax.legend(
            handles=sample_patches,
            loc="upper left",
            title="Samples",
            bbox_to_anchor=(x_loc, 1.0),
        )

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

    tool_belt.init_matplotlib()
    matplotlib.rcParams["axes.linewidth"] = 1.0

    # plot_type = "rates"
    # plot_type = "T2_max"
    # plot_type = "ratios"
    # plot_type = "ratio_fits"
    # plot_type = 'residuals'
    plot_type = "normalized_residuals"

    rates_to_plot = "both"
    # rates_to_plot = 'Omega'
    # rates_to_plot = 'gamma'

    # temp_range = [0, 600]
    temp_range = [0, 500]
    xscale = "linear"
    # temp_range = [1, 500]
    # xscale = "log"

    file_name = "compiled_data"
    # file_name = "spin_phonon_temp_dependence"
    home = common.get_nvdata_dir()
    path = home / "paper_materials/relaxation_temp_dependence"

    # if plot_type == "rates":
    #     # y_params = [[[-10, 1000], "linear"]]
    #     y_params = [[[-10, 1000], "linear"], [[5e-3, 1200], "log"]]
    # elif plot_type == "T2_max":
    #     y_params = [[[-1, 6], "linear"], [[1e-3, 50], "log"]]
    # elif plot_type == "ratios":
    #     y_params = [[[0, 5], "linear"]]
    # elif plot_type == "ratio_fits":
    #     y_params = [[[0, 5], "linear"]]
    # elif plot_type == "residuals":
    #     pass
    # elif plot_type == "normalized_residuals":
    #     y_params = [[[-3, 3], "linear"]]
    #     # rates_to_plot = "Omega"
    #     rates_to_plot = "gamma"
    # y_params = [y_params[1]]
    # for el in y_params:
    #     y_range, yscale = el
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
    normalized_residuals_histogram(rates_to_plot)

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
