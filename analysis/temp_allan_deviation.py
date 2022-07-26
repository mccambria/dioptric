# -*- coding: utf-8 -*-
"""
Analysis of four-point ESR measurements

Created on July 21st, 2022

@author: mccambria
"""

# region Imports

import utils.tool_belt as tool_belt
import matplotlib.pyplot as plt
import numpy as np
from numpy import pi
import utils.kplotlib as kpl
import majorroutines.four_point_esr as four_point_esr
import analysis.temp_from_resonances as temp_from_resonances
from pathos.multiprocessing import ProcessingPool as Pool
import allantools


# endregion

# region Constants


# endregion

# region Functions


def get_temp_from_file_pair(f_pair):

    low_file, high_file = f_pair
    low_res, low_error = four_point_esr.calc_resonance_from_file(low_file)
    high_res, high_error = four_point_esr.calc_resonance_from_file(high_file)

    zfs = (low_res + high_res) / 2
    zfs_err = np.sqrt(low_error ** 2 + high_error ** 2) / 2
    temp, temp_err = temp_from_resonances.main(zfs, zfs_err, no_print=True)

    # return temp, temp_err
    return zfs, zfs_err
    # return low_res, low_error
    # return high_res, high_error


def get_temps_from_files(files):

    with Pool() as p:
        temps_and_errs = p.map(get_temp_from_file_pair, files)
    # temps_and_errs = [get_temp_from_file_pair(pair) for pair in files]

    first_time = tool_belt.utc_from_file_name(files[0][0])
    times = [
        tool_belt.utc_from_file_name(pair[0]) - first_time for pair in files
    ]

    temps = []
    temp_errs = []

    for el in temps_and_errs:
        temps.append(el[0])
        temp_errs.append(el[1])

    return np.array(temps), np.array(temp_errs), times


# endregion


# region Main functions


def allan_deviation(sig_files, ref_files):

    fig, ax = plt.subplots()

    sig_vals, sig_errs, times = get_temps_from_files(sig_files)
    ref_vals, ref_errs, times = get_temps_from_files(ref_files)

    diff_vals = sig_vals - ref_vals
    diff_errs = np.sqrt(sig_errs ** 2 + ref_errs ** 2)

    test_data = diff_vals
    period = (times[-1] - times[0]) / (len(times) - 1)
    (t2, ad, ade, adn) = allantools.oadev(
        test_data / np.mean(test_data),
        rate=1 / period,
        data_type="freq",
        taus="all",
    )

    ax.errorbar(t2, ad, yerr=ade, ls="None", fmt="o")
    # ax.plot(t2, ad)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Time Cluster (sec)")
    ax.set_ylabel(r"$\mathrm{\Delta}\mathit{T}$ (K) Allan Deviation")

    fig.tight_layout()


def temp_vs_time(sig_files, ref_files):

    fig, ax = plt.subplots()

    sig_vals, sig_errs, times = get_temps_from_files(sig_files)
    ref_vals, ref_errs, times = get_temps_from_files(ref_files)
    num_exps = len(sig_vals)

    # ax.errorbar(x_vals, sig_vals, yerr=sig_errs, label="sig")
    # ax.plot(x_vals, sig_vals, label="sig")

    # ax.errorbar(x_vals, ref_vals, yerr=ref_errs, label="ref")
    # ax.plot(x_vals, ref_vals, label="ref")

    diff_vals = sig_vals - ref_vals
    diff_errs = np.sqrt(sig_errs ** 2 + ref_errs ** 2)
    ax.errorbar(times, diff_vals, yerr=diff_errs, label="diff", fmt="o")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"$\mathrm{\Delta}\mathit{T}$ (K)")

    eval_type = "diff"
    vals = eval(f"{eval_type}_vals")
    errs = eval(f"{eval_type}_errs")
    val_mean = np.mean(vals)
    err_mean = np.mean(errs)
    ste = np.sqrt(np.sum(errs ** 2)) / num_exps
    print(val_mean)
    print(np.std((vals - val_mean) / err_mean))
    print(ste)

    ax.legend()

    fig.tight_layout()


# endregion

# region Run the file

if __name__ == "__main__":

    kpl.init_kplotlib()

    f = "2022_07_24-22_03_54-hopper-search"
    data = tool_belt.get_raw_data(f)
    sig_files = data["sig_files"]
    ref_files = data["ref_files"]

    # temp_vs_time(sig_files, ref_files)
    allan_deviation(sig_files, ref_files)

    plt.show(block=True)

# endregion
