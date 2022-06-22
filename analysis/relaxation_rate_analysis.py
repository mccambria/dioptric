# -*- coding: utf-8 -*-
"""
Created on Wed Sep 4 14:52:43 2019

This analysis script will plot and evaluate the omega and gamma rates for the
modified rate equations [(0,0) - (0,1) and (1,1) - (1,-1)] for the complete
data set. It calculates a standard error of each data point based on the
statistics over the number of runs. With the standard error on each point, the
subtracted data is then fit to a single exponential. From the (0,0) - (0,1)
exponential, we extact 3*Omega from the exponent, along with the standard
error on omega from the covariance of the fit.

From the (1,1) - (1,-1) exponential, we extract (2*gamma + Omega). Using the
Omega we just found, we calculate gamma and the associated standard error
from the covariance of the fit.

-User can specify if the offset should be a free parameter or if it should be
  set to 0. All our analysis of rates has been done without offset as a free
  param.

-If a value for omega and the omega uncertainty is passed, file will just
  evaluate gamma (t=with the omega provided).


@author: agardill
"""

# %% Imports

import numpy
from numpy import exp
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import os
import sys

import utils.tool_belt as tool_belt
import utils.common as common
from utils.tool_belt import States
from figures.relaxation_temp_dependence.temp_dependence_fitting import (
    omega_calc,
    gamma_calc,
    get_data_points,
)

# %% Constants

manual_offset_gamma = 0.00
# %% Functions

# The exponential function without an offset
def exp_eq_omega(t, rate, amp):
    return amp * exp(-rate * t)


def exp_eq_gamma(t, rate, amp):
    return amp * exp(-rate * t) + manual_offset_gamma


def biexp(t, omega, rate1, amp1, amp2):
    return amp1 * exp(-rate1 * t) + amp2  # * exp(-3*omega*t)


# The exponential function with an offset
def exp_eq_offset(t, rate, amp, offset):
    return offset + amp * exp(-rate * t)


# A function to collect folders in mass analysis
def get_folder_list(keyword):
    nvdata_dir = tool_belt.get_nvdata_dir()
    path = nvdata_dir / "t1_double_quantum"

    folders = []

    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for folder in d:
            if keyword in folder:
                folders.append(folder)

    return folders


# This function sorts the data from one folder of an experiment and passes it
# into main
def get_data_lists(folder_name, simple_print=False):
    # Get the file list from this folder
    file_list = tool_belt.get_file_list(folder_name, ".txt")

    # Define booleans to be used later in putting data into arrays in the
    # correct order. This was mainly put in place for older data where we
    # took measurements in an inconsistent way (unlike we are now)
    zero_zero_bool = False
    zero_plus_bool = False
    plus_plus_bool = False
    plus_minus_bool = False

    # Initially create empty lists, so that if no data is recieved, a list is
    # still returned from this function
    zero_zero_counts = []
    zero_zero_ste = []
    zero_plus_counts = []
    zero_plus_ste = []
    zero_zero_time = []
    plus_plus_counts = []
    plus_plus_ste = []
    plus_minus_counts = []
    plus_minus_ste = []
    plus_plus_time = []

    # Unpack the data

    # Unpack the data and sort into arrays. This allows multiple measurements of
    # the same type to be correctly sorted into one array
    for file in file_list:
        data = tool_belt.get_raw_data(file[:-4], folder_name)
        try:
            # if True:

            init_state_name = data["init_state"]
            read_state_name = data["read_state"]

            # older files still used 1,-1,0 convention. This will allow old
            # and new files to be evaluated
            if (
                init_state_name == 1
                or init_state_name == -1
                or init_state_name == 0
            ):
                high_state_name = 1
                low_state_name = -1
                zero_state_name = 0
            else:
                high_state_name = States.HIGH.name
                low_state_name = States.LOW.name
                zero_state_name = States.ZERO.name
            relaxation_time_range = numpy.array(data["relaxation_time_range"])
            num_steps = data["num_steps"]

            num_runs = data["num_runs"]
            sig_counts = numpy.array(data["sig_counts"])
            ref_counts = numpy.array(data["ref_counts"])

            # For low counts/run, combine runs to avoid div by zero in normalization, at least
            combine_runs = 1
            if combine_runs > 1:
                sig_counts_buffer = []
                ref_counts_buffer = []
                combine_num_runs = num_runs // combine_runs
                clip_num_runs = combine_num_runs * combine_runs
                for ind in range(clip_num_runs):
                    if ind % combine_runs != 0:
                        continue
                    sig_val = 0
                    ref_val = 0
                    for sub_ind in range(combine_runs):
                        sig_val += sig_counts[ind + sub_ind]
                        ref_val += ref_counts[ind + sub_ind]
                    sig_counts_buffer.append(sig_val)
                    ref_counts_buffer.append(ref_val)
                num_runs = combine_num_runs
                sig_counts = numpy.array(sig_counts_buffer)
                ref_counts = numpy.array(ref_counts_buffer)

            # Calculate time arrays in us
            min_relaxation_time, max_relaxation_time = (
                relaxation_time_range / 10 ** 6
            )
            time_array = numpy.linspace(
                min_relaxation_time, max_relaxation_time, num=num_steps
            )

            # Calculate the average signal counts over the runs, and ste
            # if 0 in ref_counts:
            #     crash = 1 / 0

            # Assume reference is constant and can be approximated to one value
            single_ref = True
            if single_ref:
                avg_sig_counts = numpy.average(sig_counts[:num_runs], axis=0)
                ste_sig_counts = numpy.std(
                    sig_counts[:num_runs], axis=0, ddof=1
                ) / numpy.sqrt(num_runs)

                avg_ref = numpy.average(ref_counts[:num_runs])

                # Divide signal by reference to get normalized counts and st error
                norm_avg_sig = avg_sig_counts / avg_ref
                norm_avg_sig_ste = ste_sig_counts / avg_ref

            else:
                norm_sig = sig_counts[:num_runs] / ref_counts[:num_runs]
                norm_avg_sig = numpy.average(norm_sig, axis=0)
                norm_avg_sig_ste = numpy.std(
                    norm_sig, axis=0, ddof=1
                ) / numpy.sqrt(num_runs)

            # Check to see which data set the file is for, and append the data
            # to the corresponding array
            if (
                init_state_name == zero_state_name
                and read_state_name == zero_state_name
            ):
                # Check to see if data has already been added to a list for
                # this experiment. If it hasn't, then create arrays of the data.
                if zero_zero_bool == False:
                    zero_zero_counts = norm_avg_sig
                    zero_zero_ste = norm_avg_sig_ste
                    zero_zero_time = time_array

                    zero_zero_ref_max_time = max_relaxation_time
                    zero_zero_bool = True
                # If data has already been sorted for this experiment, then check
                # to see if this current data is the shorter or longer measurement,
                # and either append before or after the prexisting data
                else:

                    if max_relaxation_time > zero_zero_ref_max_time:
                        zero_zero_counts = numpy.concatenate(
                            (zero_zero_counts, norm_avg_sig)
                        )
                        zero_zero_ste = numpy.concatenate(
                            (zero_zero_ste, norm_avg_sig_ste)
                        )
                        zero_zero_time = numpy.concatenate(
                            (zero_zero_time, time_array)
                        )

                    elif max_relaxation_time < zero_zero_ref_max_time:
                        zero_zero_counts = numpy.concatenate(
                            (norm_avg_sig, zero_zero_counts)
                        )
                        zero_zero_ste = numpy.concatenate(
                            (norm_avg_sig_ste, zero_zero_ste)
                        )
                        zero_zero_time = numpy.concatenate(
                            (time_array, zero_zero_time)
                        )

            # if init_state_name == zero_state_name and \
            #                     read_state_name == high_state_name:
            # if init_state_name == zero_state_name and \
            #                     read_state_name == low_state_name:
            if (
                init_state_name == zero_state_name
                and read_state_name == high_state_name
            ) or (
                init_state_name == zero_state_name
                and read_state_name == low_state_name
            ):
                if zero_plus_bool == False:
                    zero_plus_counts = norm_avg_sig
                    zero_plus_ste = norm_avg_sig_ste
                    zero_plus_time = time_array

                    zero_plus_ref_max_time = max_relaxation_time
                    zero_plus_bool = True
                else:

                    if max_relaxation_time > zero_plus_ref_max_time:
                        zero_plus_counts = numpy.concatenate(
                            (zero_plus_counts, norm_avg_sig)
                        )
                        zero_plus_ste = numpy.concatenate(
                            (zero_plus_ste, norm_avg_sig_ste)
                        )

                        zero_plus_time = numpy.concatenate(
                            (zero_plus_time, time_array)
                        )

                    elif max_relaxation_time < zero_plus_ref_max_time:
                        zero_plus_counts = numpy.concatenate(
                            (norm_avg_sig, zero_plus_counts)
                        )
                        zero_plus_ste = numpy.concatenate(
                            (norm_avg_sig_ste, zero_plus_ste)
                        )

                        zero_plus_time = numpy.concatenate(
                            (time_array, zero_plus_time)
                        )

            # if (init_state_name == high_state_name) and \
            #     (read_state_name == high_state_name):
            # if (init_state_name == low_state_name) and \
            #     (read_state_name == low_state_name):
            if (
                init_state_name == high_state_name
                and read_state_name == high_state_name
            ) or (
                init_state_name == low_state_name
                and read_state_name == low_state_name
            ):
                if plus_plus_bool == False:
                    plus_plus_counts = norm_avg_sig
                    plus_plus_ste = norm_avg_sig_ste
                    plus_plus_time = time_array

                    plus_plus_ref_max_time = max_relaxation_time
                    plus_plus_bool = True
                else:

                    if max_relaxation_time > plus_plus_ref_max_time:
                        plus_plus_counts = numpy.concatenate(
                            (plus_plus_counts, norm_avg_sig)
                        )
                        plus_plus_ste = numpy.concatenate(
                            (plus_plus_ste, norm_avg_sig_ste)
                        )
                        plus_plus_time = numpy.concatenate(
                            (plus_plus_time, time_array)
                        )

                    elif max_relaxation_time < plus_plus_ref_max_time:
                        plus_plus_counts = numpy.concatenate(
                            (norm_avg_sig, plus_plus_counts)
                        )
                        plus_plus_ste = numpy.concatenate(
                            (norm_avg_sig_ste, plus_plus_ste)
                        )
                        plus_plus_time = numpy.concatenate(
                            (time_array, plus_plus_time)
                        )

            # if init_state_name == high_state_name and \
            #                     read_state_name == low_state_name:
            # if init_state_name == low_state_name and \
            #                     read_state_name == high_state_name:
            if (
                init_state_name == high_state_name
                and read_state_name == low_state_name
            ) or (
                init_state_name == low_state_name
                and read_state_name == high_state_name
            ):
                # We will want to put the MHz splitting in the file metadata
                uwave_freq_init = data["uwave_freq_init"]
                uwave_freq_read = data["uwave_freq_read"]

                if plus_minus_bool == False:
                    plus_minus_counts = norm_avg_sig
                    plus_minus_ste = norm_avg_sig_ste
                    plus_minus_time = time_array

                    plus_minus_ref_max_time = max_relaxation_time
                    plus_minus_bool = True
                else:

                    if max_relaxation_time > plus_minus_ref_max_time:
                        plus_minus_counts = numpy.concatenate(
                            (plus_minus_counts, norm_avg_sig)
                        )
                        plus_minus_ste = numpy.concatenate(
                            (plus_minus_ste, norm_avg_sig_ste)
                        )
                        plus_minus_time = numpy.concatenate(
                            (plus_minus_time, time_array)
                        )

                    elif max_relaxation_time < plus_minus_ref_max_time:
                        plus_minus_counts = numpy.concatenate(
                            (norm_avg_sig, plus_minus_counts)
                        )
                        plus_minus_ste = numpy.concatenate(
                            (norm_avg_sig_ste, plus_minus_ste)
                        )
                        plus_minus_time = numpy.concatenate(
                            (time_array, plus_minus_time)
                        )

                splitting_MHz = (
                    abs(uwave_freq_init - uwave_freq_read) * 10 ** 3
                )

        except Exception as exc:
            if not simple_print:
                print(exc)
                print("Skipping {}".format(str(file)))
            continue

    omega_exp_list = [
        zero_zero_counts,
        zero_zero_ste,
        zero_plus_counts,
        zero_plus_ste,
        zero_zero_time,
    ]
    gamma_exp_list = [
        plus_plus_counts,
        plus_plus_ste,
        plus_minus_counts,
        plus_minus_ste,
        plus_plus_time,
    ]
    return omega_exp_list, gamma_exp_list, num_runs, splitting_MHz


# %% Main


def main(
    path,
    folder,
    omega=None,
    omega_ste=None,
    doPlot=False,
    offset=True,
    return_gamma_data=False,
    simple_print=True,
):

    slow = True

    path_folder = path + folder
    # Get the file list from the folder
    omega_exp_list, gamma_exp_list, num_runs, splitting_MHz = get_data_lists(
        path_folder, simple_print=simple_print
    )

    # %% Fit the data

    if doPlot:
        fig, axes_pack = plt.subplots(1, 2, figsize=(17, 8))
        fig.set_tight_layout(True)

    omega_fit_failed = False
    gamma_fit_failed = False

    ax = None

    # If omega value is passed into the function, skip the omega fitting.
    if omega is not None and omega_ste is not None:
        omega_opti_params = numpy.array([None])
        zero_relaxation_counts = numpy.array([None])
        zero_relaxation_ste = numpy.array([None])
        zero_zero_time = numpy.array([None])
    else:
        # Fit to the (0,0) - (0,1) data to find Omega

        zero_zero_counts = omega_exp_list[0]
        zero_zero_ste = omega_exp_list[1]
        zero_plus_counts = omega_exp_list[2]
        zero_plus_ste = omega_exp_list[3]
        zero_zero_time = omega_exp_list[4]
        zero_relaxation_counts = zero_zero_counts - zero_plus_counts
        zero_relaxation_ste = numpy.sqrt(
            zero_zero_ste ** 2 + zero_plus_ste ** 2
        )

        if slow:
            init_params_list = [0.24 / 1000, 0.16]
        else:
            init_params_list = [0.1, 0.3]

        try:
            if offset:
                init_params_list.append(0)
                init_params = tuple(init_params_list)
                omega_opti_params, cov_arr = curve_fit(
                    exp_eq_offset,
                    zero_zero_time,
                    zero_relaxation_counts,
                    p0=init_params,
                    sigma=zero_relaxation_ste,
                    absolute_sigma=True,
                )

            else:
                init_params = tuple(init_params_list)
                omega_opti_params, cov_arr = curve_fit(
                    exp_eq_omega,
                    zero_zero_time,
                    zero_relaxation_counts,
                    p0=init_params,
                    sigma=zero_relaxation_ste,
                    absolute_sigma=True,
                )
                # if slow:
                #     omega_opti_params = numpy.array(init_params)
                #     cov_arr = numpy.array([[0,0],[0,0]])

            # MCC
            if not simple_print:
                print(omega_opti_params)

        except Exception:

            omega_fit_failed = True

            if doPlot:
                ax = axes_pack[0]
                ax.errorbar(
                    zero_zero_time,
                    zero_relaxation_counts,
                    yerr=zero_relaxation_ste,
                    label="data",
                    fmt="o",
                    color="blue",
                )
                ax.set_xlabel("Relaxation time (ms)")
                ax.set_ylabel("Normalized signal Counts")
                ax.legend()

        if not omega_fit_failed:
            # Calculate omega nad its ste
            omega = omega_opti_params[0] / 3.0
            omega_ste = numpy.sqrt(cov_arr[0, 0]) / 3.0

            if not simple_print:
                print(
                    "Omega: {} +/- {} s^-1".format(
                        "%.3f" % (omega * 1000), "%.3f" % (omega_ste * 1000)
                    )
                )
            # Plotting the data
            if doPlot:
                zero_time_linspace = numpy.linspace(
                    0, zero_zero_time[-1], num=1000
                )
                ax = axes_pack[0]
                ax.errorbar(
                    zero_zero_time,
                    zero_relaxation_counts,
                    yerr=zero_relaxation_ste,
                    label="data",
                    fmt="o",
                    color="blue",
                )
                if offset:
                    ax.plot(
                        zero_time_linspace,
                        exp_eq_offset(zero_time_linspace, *omega_opti_params),
                        "r",
                        label="fit",
                    )
                else:
                    ax.plot(
                        zero_time_linspace,
                        exp_eq_omega(zero_time_linspace, *omega_opti_params),
                        "r",
                        label="fit",
                    )
                ax.set_xlabel("Relaxation time (ms)")
                ax.set_ylabel("Normalized signal Counts")
                ax.legend()
                units = r"s$^{-1}$"
                text = r"$\Omega = $ {} $\pm$ {} {}".format(
                    "%.3f" % (omega * 1000), "%.3f" % (omega_ste * 1000), units
                )

                props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
                ax.text(
                    0.55,
                    0.9,
                    text,
                    transform=ax.transAxes,
                    fontsize=12,
                    verticalalignment="top",
                    bbox=props,
                )

    if omega_fit_failed:
        print("Omega fit failed")
        return

    if ax is not None:
        ax.set_title("Omega")
        # ax.set_title('(0,0) - (0,-1)')
        # ax.set_title('(0,0) - (0,+1)')

    # %% Fit to the (1,1) - (1,-1) data to find Gamma, only if Omega waas able
    # to fit

    plus_plus_counts = gamma_exp_list[0]
    plus_plus_ste = gamma_exp_list[1]
    plus_minus_counts = gamma_exp_list[2]
    plus_minus_ste = gamma_exp_list[3]
    plus_plus_time = gamma_exp_list[4]

    # Define the counts for the plus relaxation equation
    plus_relaxation_counts = plus_plus_counts - plus_minus_counts
    plus_relaxation_ste = numpy.sqrt(plus_plus_ste ** 2 + plus_minus_ste ** 2)

    # Skip values at t=0 to get rid of pi pulse decoherence systematic
    # See wiki March 31st, 2021
    inds_to_remove = []
    for ind in range(len(plus_plus_time)):
        t = plus_plus_time[ind]
        if t == 0:
            inds_to_remove.append(ind)
    plus_plus_time = numpy.delete(plus_plus_time, inds_to_remove)
    plus_relaxation_counts = numpy.delete(
        plus_relaxation_counts, inds_to_remove
    )
    plus_relaxation_ste = numpy.delete(plus_relaxation_ste, inds_to_remove)

    if slow:
        init_params_list = [3 * omega, 0.16]
    else:
        init_params_list = [2 * omega, 0.40]

    try:
        if offset:

            init_params_list.append(0)
            init_params = tuple(init_params_list)
            gamma_opti_params, cov_arr = curve_fit(
                exp_eq_offset,
                plus_plus_time,
                plus_relaxation_counts,
                p0=init_params,
                sigma=plus_relaxation_ste,
                absolute_sigma=True,
            )

        else:
            # MCC
            init_params = tuple(init_params_list)
            gamma_fit_func = exp_eq_gamma
            gamma_opti_params, cov_arr = curve_fit(
                exp_eq_gamma,
                plus_plus_time,
                plus_relaxation_counts,
                p0=init_params,
                sigma=plus_relaxation_ste,
                absolute_sigma=True,
            )
            # init_params = (0.22, 0.17, 0.0)
            # gamma_fit_func = lambda t, rate1, amp1, amp2: biexp(t, omega, rate1, amp1, amp2)
            # gamma_opti_params, cov_arr = curve_fit(gamma_fit_func,
            #                   plus_plus_time, plus_relaxation_counts,
            #                   p0 = init_params, sigma = plus_relaxation_ste,
            #                   absolute_sigma=True)
            # print(gamma_opti_params)
            # gamma_opti_params = numpy.array([0.0,0.0,0])
            # cov_arr = numpy.array([[0,0,0],[0,0,0],[0,0,0]])
            # if slow:
            #     gamma_opti_params = numpy.array(init_params)
            #     cov_arr = numpy.array([[0,0],[0,0]])

        # MCC
        if not simple_print:
            print(gamma_opti_params)

        if return_gamma_data:
            amplitude = gamma_opti_params[1]
            data_decay = plus_relaxation_counts / amplitude
            ste_decay = plus_relaxation_ste / amplitude
            times_decay = plus_plus_time
            return data_decay, ste_decay, times_decay

    except Exception as e:
        gamma_fit_failed = True
        if not simple_print:
            print(e)

        if doPlot:
            ax = axes_pack[1]
            ax.errorbar(
                plus_plus_time,
                plus_relaxation_counts,
                yerr=plus_relaxation_ste,
                label="data",
                fmt="o",
                color="blue",
            )
            ax.set_xlabel("Relaxation time (ms)")
            ax.set_ylabel("Normalized signal Counts")

    if not gamma_fit_failed:

        # Calculate gamma and its ste
        gamma = (gamma_opti_params[0] - omega) / 2.0
        gamma_ste = 0.5 * numpy.sqrt(cov_arr[0, 0] + omega_ste ** 2)

        # Test MCC
        # gamma = 0.070
        # gamma_opti_params[0] = (2 * gamma) + omega
        # gamma_opti_params[1] = 0.20
        if not simple_print:
            print(
                "gamma: {} +/- {} s^-1".format(
                    "%.3f" % (gamma * 1000), "%.3f" % (gamma_ste * 1000)
                )
            )

        # Plotting
        if doPlot:
            plus_time_linspace = numpy.linspace(
                0, plus_plus_time[-1], num=1000
            )
            ax = axes_pack[1]
            ax.errorbar(
                plus_plus_time,
                plus_relaxation_counts,
                yerr=plus_relaxation_ste,
                label="data",
                fmt="o",
                color="blue",
            )
            if offset:
                ax.plot(
                    plus_time_linspace,
                    exp_eq_offset(plus_time_linspace, *gamma_opti_params),
                    "r",
                    label="fit",
                )
            else:
                ax.plot(
                    plus_time_linspace,
                    # exp_eq_gamma(plus_time_linspace, *gamma_opti_params),  # MCC
                    gamma_fit_func(plus_time_linspace, *gamma_opti_params),
                    "r",
                    label="fit",
                )
            ax.set_xlabel("Relaxation time (ms)")
            ax.set_ylabel("Normalized signal Counts")
            ax.legend()
            units = r"s$^{-1}$"
            text = r"$\gamma = $ {} $\pm$ {} {}".format(
                "%.3f" % (gamma * 1000), "%.3f" % (gamma_ste * 1000), units
            )
            #            ax.set_xlim([-0.001, 0.05])

            props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
            ax.text(
                0.55,
                0.90,
                text,
                transform=ax.transAxes,
                fontsize=12,
                verticalalignment="top",
                bbox=props,
            )

    if doPlot:
        ax.set_title("gamma")
        # ax.set_title('(+1,+1) - (+1,-1)')
        # ax.set_title('(-1,-1) - (-1,+1)')
        fig.canvas.draw()
        fig.canvas.flush_events()

        # Saving the data
        data_dir = common.get_nvdata_dir()

        time_stamp = tool_belt.get_time_stamp()
        raw_data = {
            "time_stamp": time_stamp,
            "splitting_MHz": splitting_MHz,
            "splitting_MHz-units": "MHz",
            #                    'offset_free_param?': offset,
            "manual_offset_gamma": manual_offset_gamma,
            "omega": omega,
            "omega-units": "kHz",
            "omega_ste": omega_ste,
            "omega_ste-units": "khz",
            #                    'gamma': gamma,
            #                    'gamma-units': 'kHz',
            #                    'gamma_ste': gamma_ste,
            #                    'gamma_ste-units': 'khz',
            "zero_relaxation_counts": zero_relaxation_counts.tolist(),
            "zero_relaxation_counts-units": "counts",
            "zero_relaxation_ste": zero_relaxation_ste.tolist(),
            "zero_relaxation_ste-units": "counts",
            "zero_zero_time": zero_zero_time.tolist(),
            "zero_zero_time-units": "ms",
            #                    'plus_relaxation_counts': plus_relaxation_counts.tolist(),
            #                    'plus_relaxation_counts-units': 'counts',
            #                    'plus_relaxation_ste': plus_relaxation_ste.tolist(),
            #                    'plus_relaxation_ste-units': 'counts',
            #                    'plus_plus_time': plus_plus_time.tolist(),
            #                    'plus_plus_time-units': 'ms',
            "omega_opti_params": omega_opti_params.tolist(),
            #                    'gamma_opti_params': gamma_opti_params.tolist(),
        }

        file_name = "{}-analysis".format(folder)
        file_path = str(data_dir / path_folder / file_name)
        tool_belt.save_raw_data(raw_data, file_path)
        tool_belt.save_figure(fig, file_path)

    if gamma_fit_failed:
        print("gamma fit failed")
        return

    # String to paste into excel
    try:
        print(
            "{}\t{}\t{}\t{}".format(
                "%.3f" % (omega * 1000),
                "%.3f" % (omega_ste * 1000),
                "%.3f" % (gamma * 1000),
                "%.3f" % (gamma_ste * 1000),
            )
        )
    except Exception as exc:
        print(exc)

    return gamma, gamma_ste


# %% Run the file

if __name__ == "__main__":

    temp = 295
    folder = "hopper-nv1_2022_06_15-{}K-5mW".format(temp)

    # mode = "prediction"
    mode = "analysis"
    # mode = "batch_analysis"

    if mode == "prediction":
        est_omega = omega_calc(temp)
        est_gamma = gamma_calc(temp)
        print("good times in ms")
        # print("Omega: {}".format(4000 / (3 * est_omega)))
        # print("gamma: {}".format(4000 / (2 * est_gamma + est_omega)))
        print("Omega: {}".format(1000 * 1 / (est_omega)))
        print("gamma: {}".format(1000 * (3 / 2) / (est_gamma + est_omega)))
        # print('Omega: {}'.format(est_omega))
        # print('gamma: {}'.format(est_gamma))

    elif mode == "analysis":

        plt.ion()

        path = "pc_hahn/branch_master/t1_dq_main/data_collections-optically_enhanced/"

        main(
            path,
            folder,
            omega=None,
            omega_ste=None,
            doPlot=True,
            offset=False,
            simple_print=True,
        )

        # plt.show(block=True)

    elif mode == "batch_analysis":

        # file_name = "compiled_data"
        file_name = "compiled_data-single_ref"
        home = common.get_nvdata_dir()
        path = home / "paper_materials/relaxation_temp_dependence"
        data_points = get_data_points(path, file_name, override_skips=True)

        for point in data_points:
            full_data_path = point["Path"]
            if full_data_path == "":
                print("None")
                continue
            full_data_path_split = full_data_path.split("/")
            # if full_data_path_split[0] != "pc_rabi":
            #     continue
            data_path = "/".join(full_data_path_split[0:-2]) + "/"
            folder = full_data_path_split[-2]
            # print(data_path)
            # print(folder)

            main(
                data_path,
                folder,
                omega=None,
                omega_ste=None,
                doPlot=False,
                offset=False,
                simple_print=True,
            )
