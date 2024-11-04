# -*- coding: utf-8 -*-
"""This file contains functions, classes, and other objects that are useful
in a variety of contexts. Since they are expected to be used in manyNormMode
files, I put them all in one place so that they don't have to be redefined
in each file.

Created on November 23rd, 2018

@author: mccambria
"""

import json
import logging
import math
import signal
import smtplib
import socket
import time
import traceback
from decimal import Decimal
from email.mime.text import MIMEText
from enum import Enum
from functools import cache

import keyring
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
from numpy import exp
from scipy.optimize import curve_fit
from scipy.special import erf, factorial
from scipy.stats import norm, skewnorm

from utils import common
from utils.constants import Boltzmann, Digital, ModMode, NormMode

# region Server utils
# Utility functions to be used by LabRAD servers


def configure_logging(inst, level=logging.INFO):
    """Setup logging for a LabRAD server

    Parameters
    ----------
    inst : Class instance
        Pass self from the LabRAD server class
    level : logging level, optional
        by default logging.DEBUG
    """
    folder_path = common.get_labrad_logging_folder()
    filename = folder_path / f"{inst.name}.log"
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%y-%m-%d_%H-%M-%S",
        filename=filename,
    )


# endregion
# region Laser utils


def get_mod_type(laser_name):
    config = common.get_config_dict()
    mod_type = config["Optics"][laser_name]["mod_type"]
    return mod_type.name


def laser_off(laser_name):
    laser_switch_sub(False, laser_name)


def laser_on(laser_name, laser_power=None):
    laser_switch_sub(True, laser_name, laser_power)


def laser_switch_sub(turn_on, laser_name, laser_power=None):
    config = common.get_config_dict()
    mod_type = config["Optics"][laser_name]["mod_type"]
    pulse_gen = get_server_pulse_gen()

    if mod_type is ModMode.DIGITAL:
        if turn_on:
            laser_chan = config["Wiring"]["PulseGen"][f"do_{laser_name}_dm"]
            pulse_gen.constant([laser_chan])
    elif mod_type is ModMode.ANALOG:
        if turn_on:
            laser_chan = config["Wiring"]["PulseGen"][f"do_{laser_name}_dm"]
            if laser_chan == 0:
                pulse_gen.constant([], 0.0, laser_power)
            elif laser_chan == 1:
                pulse_gen.constant([], laser_power, 0.0)

    # If we're turning things off, turn everything off. If we wanted to really
    # do this nicely we'd find a way to only turn off the specific channel,
    # but it's not worth the effort.
    if not turn_on:
        pulse_gen.constant([])


def set_laser_power(nv_sig=None, laser_key=None, laser_name=None, laser_power=None):
    """Set a laser power, or return it for analog modulation.
    Specify either a laser_key/nv_sig or a laser_name/laser_power.
    """

    return None  # MCC consider deprecated

    if (nv_sig is not None) and (laser_key is not None):
        laser_dict = nv_sig[laser_key]
        laser_name = laser_dict["name"]
        # If the power isn't specified, then we assume it's set some other way
        if "power" in laser_dict:
            laser_power = laser_dict["power"]
    elif (laser_name is not None) and (laser_power is not None):
        pass  # All good
    else:
        raise Exception(
            "Specify either a laser_key/nv_sig or a laser_name/laser_power."
        )

    # If the power is controlled by analog modulation, we'll need to pass it
    # to the pulse streamer
    config = common.get_config_dict()
    mod_type = config["Optics"][laser_name]["mod_mode"]
    if mod_type == ModMode.ANALOG:
        return laser_power
    else:
        laser_server = get_filter_server(laser_name)
        if (laser_power is not None) and (laser_server is not None):
            laser_server.set_laser_power(laser_power)
        return None


def set_filter(optics_name, filter_name):
    filter_server = get_filter_server(optics_name)
    if filter_server is None:
        return
    config = common.get_config_dict()
    pos = config["Optics"][optics_name]["filter_mapping"][filter_name]
    filter_server.set_filter(pos)


def get_filter_server(optics_name):
    """Try to get a filter server. If there isn't one listed in the config,
    just return None.
    """
    try:
        config = common.get_config_dict()
        server_name = config["Optics"][optics_name]["filter_server"]
        cxn = common.labrad_connect()
        return getattr(cxn, server_name)
    except Exception:
        return None


def get_laser_server(laser_name):
    """Try to get a laser server. If there isn't one listed in the config,
    just return None.
    """
    try:
        config = common.get_config_dict()
        server_name = config["Optics"][laser_name]["laser_server"]
        cxn = common.labrad_connect()
        return getattr(cxn, server_name)
    except Exception:
        return None


# endregion
# region Pulse generator utils


def process_laser_seq(seq, laser_name, laser_power, train):
    """
    Automatically process simple laser sequences. Simple here means that the modulation
    is digital or, if the modulation is analog, then only one power is used)
    """

    config = common.get_config_dict()
    # print(config)
    pulser_wiring = config["Wiring"]["PulseGen"]
    mod_type = config["Optics"][laser_name]["mod_type"]

    # Digital: do nothing
    if mod_type is ModMode.DIGITAL:
        pulser_laser_mod = pulser_wiring["do_{}_dm".format(laser_name)]
        seq.setDigital(pulser_laser_mod, train)
    # Analog:convert LOW / HIGH to 0.0 / analog voltage
    elif mod_type is ModMode.ANALOG:
        processed_train = []
        power_dict = {Digital.LOW: 0.0, Digital.HIGH: laser_power}
        for el in train:
            dur = el[0]
            val = el[1]
            processed_train.append((dur, power_dict[val]))
        pulser_laser_mod = pulser_wiring["ao_{}_am".format(laser_name)]
        # print(processed_train)
        seq.setAnalog(pulser_laser_mod, processed_train)


def set_delays_to_zero(config):
    """Pass this a config dictionary and it'll set all the delays to zero.
    Useful for testing sequences without having to worry about delays.
    """
    for key in config:
        # Check if any entries are delays and set them to 0
        if key.endswith("delay"):
            config[key] = 0
            return
        # Check if we're at a sublevel - if so, recursively set its delay to 0
        val = config[key]
        if isinstance(val, dict):
            set_delays_to_zero(val)


def set_delays_to_sixteen(config):
    """Pass this a config dictionary and it'll set all the delays to 16ns,
    which is the minimum wait() time for the OPX. Useful for testing
    sequences without having to worry about delays.
    """
    for key in config:
        # Check if any entries are delays and set them to 0
        if key.endswith("delay"):
            config[key] = 16
            return
        # Check if we're at a sublevel - if so, recursively set its delay to 0
        val = config[key]
        if isinstance(val, dict):
            set_delays_to_sixteen(val)


def seq_train_length_check(train):
    """Print out the length of a the sequence train for a specific channel.
    Useful for debugging sequences
    """
    total = 0
    for el in train:
        total += el[0]
    print(total)


def encode_seq_args(seq_args):
    # Recast np ints to Python ints so json knows what to do
    for ind in range(len(seq_args)):
        el = seq_args[ind]
        if type(el) is np.int32:
            seq_args[ind] = int(el)
        if isinstance(el, Enum):
            seq_args[ind] = str(el)
    return json.dumps(seq_args)


def decode_seq_args(seq_args_string):
    if seq_args_string == "":
        return []
    else:
        return json.loads(seq_args_string)


def get_pulse_streamer_wiring():
    config = common.get_config_dict()
    wiring = config["Wiring"]["PulseGen"]
    return wiring


def get_tagger_wiring():
    config = common.get_config_dict()
    wiring = config["Wiring"]["Tagger"]
    return wiring


# endregion
# region Math functions


def poisson_dist(x, rate):
    return (rate**x) * np.exp(-rate) / factorial(x)


def poisson_cdf(x, rate):
    """Cumulative distribution function for poisson pdf. Integrates
    up to and including x"""
    x_floor = int(np.floor(x))
    val = 0
    for ind in range(x_floor):
        val += poisson_dist(ind, rate)
    return val


def bimodal_dist(x, prob_nv0, mean_counts_nv0, mean_counts_nvn):
    prob_nvn = 1 - prob_nv0
    val_nv0 = poisson_dist(x, mean_counts_nv0)
    val_nvn = poisson_dist(x, mean_counts_nvn)
    return prob_nv0 * val_nv0 + prob_nvn * val_nvn


def bimodal_gaussian(x, prob_nv0, mean_nv0, std_nv0, mean_nvn, std_nvn):
    prob_nvn = 1 - prob_nv0
    val_nv0 = gaussian_pdf(x, mean_nv0, std_nv0)
    val_nvn = gaussian_pdf(x, mean_nvn, std_nvn)
    return prob_nv0 * val_nv0 + prob_nvn * val_nvn


def gaussian_pdf(x, mean, std):
    return norm(loc=mean, scale=std).pdf(x)


def gaussian_cdf(x, mean, std):
    return norm(loc=mean, scale=std).cdf(x)


def bimodal_skew_gaussian(
    x, prob_nv0, mean_nv0, std_nv0, skew_nv0, mean_nvn, std_nvn, skew_nvn
):
    prob_nvn = 1 - prob_nv0
    val_nv0 = skew_gaussian_pdf(x, mean_nv0, std_nv0, skew_nv0)
    val_nvn = skew_gaussian_pdf(x, mean_nvn, std_nvn, skew_nvn)
    return prob_nv0 * val_nv0 + prob_nvn * val_nvn


def skew_gaussian_pdf(x, mean, std, skew):
    return skewnorm(a=skew, loc=mean, scale=std).pdf(x)


def skew_gaussian_cdf(x, mean, std, skew):
    return skewnorm(a=skew, loc=mean, scale=std).cdf(x)


def fit_charge_state_histogram(counts_list, no_print=False):
    """counts_list should have some population in both NV- and NV0"""

    counts_list = counts_list.flatten()

    # Remove outliers
    median = np.median(counts_list)
    std = np.std(counts_list)
    counts_list = counts_list[counts_list < median + 10 * std]

    # Histogram the counts
    counts_list = np.array([round(el) for el in counts_list])
    max_count = max(counts_list)
    x_vals = np.linspace(0, max_count, max_count + 1)
    hist, _ = np.histogram(
        counts_list, bins=max_count + 1, range=(0, max_count), density=True
    )

    # Fit the histogram
    fit_fn = bimodal_skew_gaussian
    mean_nv0_guess = round(np.quantile(counts_list, 0.2))
    mean_nvn_guess = round(np.quantile(counts_list, 0.98))
    guess_params = (
        0.7,
        mean_nv0_guess,
        2 * np.sqrt(mean_nv0_guess),
        2,
        mean_nvn_guess,
        2 * np.sqrt(mean_nvn_guess),
        -2,
    )
    try:
        popt, _ = curve_fit(fit_fn, x_vals, hist, p0=guess_params)
        if not no_print:
            print(popt)
        return popt
    except Exception:
        return None


def determine_charge_state_threshold(
    counts_list, nvn_ratio=None, no_print=False, ret_fidelity=False
):
    popt = fit_charge_state_histogram(counts_list, no_print)

    # Popt will be None
    if popt is None:
        if ret_fidelity:
            return None, None
        else:
            return None

    if nvn_ratio is None:
        nvn_ratio = 1 - popt[0]
    nv0_ratio = 1 - nvn_ratio

    # Assume some kind of bimodal distribution where each mode has the same form
    # and there is one parameter that describes the relative weight of each mode.
    num_single_dist_params = int((len(popt) - 1) / 2)

    # Calculate fidelities for given threshold
    mean_counts_nv0, mean_counts_nvn = popt[1], popt[1 + num_single_dist_params]
    mean_counts_nv0 = round(mean_counts_nv0)
    mean_counts_nvn = round(mean_counts_nvn)
    thresh_options = np.arange(0.5, np.max(counts_list) + 0.5, 1)
    fidelities = []
    left_fidelities = []
    right_fidelities = []
    for val in thresh_options:
        nv0_left_prob = skew_gaussian_cdf(val, *popt[1 : 1 + num_single_dist_params])
        nvn_left_prob = skew_gaussian_cdf(val, *popt[1 + num_single_dist_params :])
        nv0_right_prob = 1 - nv0_left_prob
        nvn_right_prob = 1 - nvn_left_prob
        fidelity = nv0_ratio * nv0_left_prob + nvn_ratio * nvn_right_prob
        left_fidelity = (nv0_ratio * nv0_left_prob) / (
            nv0_ratio * nv0_left_prob + nvn_ratio * nvn_left_prob
        )
        right_fidelity = (nvn_ratio * nvn_right_prob) / (
            nvn_ratio * nvn_right_prob + nv0_ratio * nv0_right_prob
        )
        fidelities.append(fidelity)
        left_fidelities.append(left_fidelity)
        right_fidelities.append(right_fidelity)
    fidelity = np.max(fidelities)
    threshold = thresh_options[np.argmax(fidelities)]

    if not no_print:
        print(f"Optimum readout fidelity {fidelity} achieved at threshold {threshold}")

    if ret_fidelity:
        return threshold, fidelity
    else:
        return threshold


def determine_dual_threshold(
    counts_list,
    nvn_ratio=None,
    min_fidelity=0.9,
    no_print=False,
):
    """Not fully implemented yet. Version of determine_threshold for a trinary
    threshold system where for a < b: if counts < a, we call dark state; if
    counts > b, we call bright state; and if a < counts < b, we make no call.
    """

    counts_list = counts_list.flatten()

    # Remove outliers
    median = np.median(counts_list)
    std = np.std(counts_list)
    counts_list = counts_list[counts_list < median + 10 * std]

    # Histogram the counts
    counts_list = np.array([round(el) for el in counts_list])
    max_count = max(counts_list)
    x_vals = np.linspace(0, max_count, max_count + 1)
    hist, _ = np.histogram(
        counts_list, bins=max_count + 1, range=(0, max_count), density=True
    )

    # Fit the histogram
    fit_fn = bimodal_skew_gaussian
    num_single_dist_params = 3
    mean_nv0_guess = round(np.quantile(counts_list, 0.2))
    mean_nvn_guess = round(np.quantile(counts_list, 0.98))
    guess_params = (
        0.7,
        mean_nv0_guess,
        2 * np.sqrt(mean_nv0_guess),  # 1.5 factor for broadening
        2,
        mean_nvn_guess,
        2 * np.sqrt(mean_nvn_guess),
        -2,
    )
    popt, _ = curve_fit(fit_fn, x_vals, hist, p0=guess_params)
    if not no_print:
        print(popt)

    if nvn_ratio is None:
        nvn_ratio = 1 - popt[0]
    nv0_ratio = 1 - nvn_ratio

    # Calculate fidelities for given threshold
    mean_counts_nv0, mean_counts_nvn = popt[1], popt[1 + num_single_dist_params]
    mean_counts_nv0 = round(mean_counts_nv0)
    mean_counts_nvn = round(mean_counts_nvn)
    thresh_options = np.arange(0.5, np.max(counts_list) + 0.5, 1)
    num_options = len(thresh_options)
    fidelities = []
    left_fidelities = []
    right_fidelities = []
    for val in thresh_options:
        nv0_left_prob = skew_gaussian_cdf(val, *popt[1 : 1 + num_single_dist_params])
        nvn_left_prob = skew_gaussian_cdf(val, *popt[1 + num_single_dist_params :])
        nv0_right_prob = 1 - nv0_left_prob
        nvn_right_prob = 1 - nvn_left_prob
        fidelity = nv0_ratio * nv0_left_prob + nvn_ratio * nvn_right_prob
        left_fidelity = (nv0_ratio * nv0_left_prob) / (
            nv0_ratio * nv0_left_prob + nvn_ratio * nvn_left_prob
        )
        right_fidelity = (nvn_ratio * nvn_right_prob) / (
            nvn_ratio * nvn_right_prob + nv0_ratio * nv0_right_prob
        )
        fidelities.append(fidelity)
        left_fidelities.append(left_fidelity)
        right_fidelities.append(right_fidelity)
    single_threshold = thresh_options[np.argmax(fidelities)]
    best_fidelity = np.max(fidelities)

    # Calculate normalized probabilities for given integrated counts value
    norm_nv0_probs = []
    norm_nvn_probs = []
    for val in x_vals:
        nv0_prob = skew_gaussian_pdf(val, *popt[1 : 1 + num_single_dist_params])
        nvn_prob = skew_gaussian_pdf(val, *popt[1 + num_single_dist_params :])
        norm_nv0_prob = (
            nv0_ratio * nv0_prob / (nv0_ratio * nv0_prob + nvn_ratio * nvn_prob)
        )
        norm_nvn_prob = (
            nvn_ratio * nvn_prob / (nv0_ratio * nv0_prob + nvn_ratio * nvn_prob)
        )
        norm_nv0_probs.append(norm_nv0_prob)
        norm_nvn_probs.append(norm_nvn_prob)
    if not single_or_dual:
        ### Manual approach
        # threshold = [single_threshold - 4, single_threshold + 1]

        ### CDF
        # threshold = [np.min(thresh_options), np.max(thresh_options)]
        # for ind in range(num_options):
        #     left_fidelity = left_fidelities[ind]
        #     right_fidelity = right_fidelities[ind]
        #     thresh_option = thresh_options[ind]
        #     if (
        #         left_fidelity > dual_threshold_min_fidelity
        #         and thresh_option > threshold[0]
        #     ):
        #         threshold[0] = thresh_option
        #     if (
        #         right_fidelity > dual_threshold_min_fidelity
        #         and thresh_option < threshold[1]
        #     ):
        #         threshold[1] = thresh_option

        ### PDF
        norm_nv0_probs = np.array(norm_nv0_probs)
        norm_nvn_probs = np.array(norm_nvn_probs)
        adj_norm_nv0_probs = np.where(norm_nv0_probs > min_fidelity, norm_nv0_probs, 1)
        adj_norm_nvn_probs = np.where(norm_nvn_probs > min_fidelity, norm_nvn_probs, 1)
        threshold = [
            x_vals[np.argmin(adj_norm_nv0_probs)] + 0.5,
            x_vals[np.argmin(adj_norm_nvn_probs)] - 0.5,
        ]

    # if there's no ambiguous zone for dual-thresholding just use a single value
    if single_or_dual or threshold[0] >= threshold[1]:
        # if single_or_dual:
        threshold = single_threshold

    # if not single_or_dual:
    if False:
        smooth_x_vals = np.linspace(0, max_count, 10 * (max_count + 1))
        fig, ax = plt.subplots()
        max_data = max(counts_list)
        rng = (-0.5, max_data + 0.5)
        nbins = max_data + 1
        color = "#1f77b4"
        alpha = 0.3
        hex_alpha = hex(round(alpha * 255))
        if len(hex_alpha) == 3:
            hex_alpha = f"0{hex_alpha[-1]}"
        else:
            hex_alpha = hex_alpha[-2:]
        facecolor = f"{color}{hex_alpha}"
        ax.hist(
            counts_list,
            histtype="step",
            bins=nbins,
            facecolor=facecolor,
            fill=True,
            range=rng,
            # label="Histogram",
            density=True,
        )
        # ax.plot(x_vals, fit_fn(x_vals, *guess_params))
        # popt: prob_nv0, mean_nv0, std_nv0, skew_nv0, mean_nvn, std_nvn, skew_nvn
        ax.plot(
            smooth_x_vals,
            popt[0] * skew_gaussian_pdf(smooth_x_vals, *popt[1:4]),
            color="#d62728",
            label="NV⁰ mode",
        )
        ax.plot(
            smooth_x_vals,
            (1 - popt[0]) * skew_gaussian_pdf(smooth_x_vals, *popt[4:]),
            color="#2ca02c",
            label="NV⁻ mode",
        )
        ax.plot(
            smooth_x_vals,
            fit_fn(smooth_x_vals, *popt),
            color="#1f77b4",
            label="Combined",
        )
        if single_or_dual:
            ax.axvline(threshold, color="#7f7f7f", linestyle="dashed", linewidth=2)
        else:
            ax.axvline(threshold[0], color="red")
            ax.axvline(threshold[1], color="black")
        ax.set_xlabel("Integrated counts")
        ax.set_ylabel("Probability")
        ax.legend()
        plt.show(block=True)

    # if not single_or_dual:
    #     threshold = threshold[1]

    if not no_print:
        print(f"Optimum threshold: {threshold}")
        if single_or_dual:
            print(f"Fidelity: {best_fidelity}")

    return threshold


def threshold(val, thresh):
    try:
        if len(thresh) == 2:
            return _dual_threshold(val, *thresh)
    except Exception:
        pass
    where_thresh = np.array(thresh, dtype=bool)
    thresh_val = np.copy(val)
    thresh_val = np.greater(val, thresh, out=thresh_val, where=where_thresh)
    return thresh_val


def _dual_threshold(val, low_thresh, high_thresh):
    low_thresh_val = threshold(val, low_thresh)
    high_thresh_val = threshold(val, high_thresh)
    ambiguous = np.logical_xor(low_thresh_val, high_thresh_val)
    dual_thresh_val = np.where(ambiguous, np.nan, high_thresh_val)
    return dual_thresh_val


def nan_corr_coef(arr):
    """
    Version of numpy's correlation coefficient that respects nan by just throwing
    out any pairs of measurements where either value is nan
    """
    arr = np.array(arr)
    num_rows = arr.shape[0]
    corr_coef_arr = np.empty((num_rows, num_rows))
    for ind in range(num_rows):
        for jnd in range(num_rows):
            if ind == 5 and jnd == 6:
                pass
            if jnd < ind:
                corr_coef_arr[ind, jnd] = corr_coef_arr[jnd, ind]
                continue
            if jnd == ind:
                corr_coef_arr[ind, jnd] = 1
                continue
            i_counts = arr[ind]
            j_counts = arr[jnd]
            i_counts_m = ma.masked_invalid(i_counts)
            j_counts_m = ma.masked_invalid(j_counts)
            mask = ~i_counts_m.mask & ~j_counts_m.mask
            corr_coef_arr[ind, jnd] = np.corrcoef(i_counts[mask], j_counts[mask])[0, 1]
    return corr_coef_arr


def moving_average(x, w):
    return np.convolve(x, np.ones(w), "valid") / w


def get_pi_pulse_dur(rabi_period):
    return round(rabi_period / 2)


def get_pi_on_2_pulse_dur(rabi_period):
    return round(rabi_period / 4)


def iq_comps(phase, amp):
    """Given the phase and amplitude of the IQ vector, calculate the I (real) and
    Q (imaginary) components
    """
    if isinstance(phase, list):
        ret_vals = []
        for val in phase:
            ret_vals.append(np.round(amp * np.exp((0 + 1j) * val), 5))
        return (np.real(ret_vals).tolist(), np.imag(ret_vals).tolist())
    else:
        ret_val = np.round(amp * np.exp((0 + 1j) * phase), 5)
        return (np.real(ret_val), np.imag(ret_val))


def lorentzian(x, x0, A, L, offset):
    """Calculates the value of a lorentzian for the given input and parameters

    Params:
        x: float
            Input value
        params: tuple
            The parameters that define the lorentzian
            0: x0, mean postiion in x
            1: A, amplitude of curve
            2: L, related to width of curve
            3: offset, constant y value offset
    """

    x_center = x - x0
    return offset + A * 0.5 * L / (x_center**2 + (0.5 * L) ** 2)


def exp_decay(x, amp, decay, offset):
    return offset + amp * np.exp(-x / decay)


def linear(x, slope, y_offset):
    return slope * x + y_offset


def quadratic(x, a, b, c, x_offset):
    x_ = x - x_offset
    return a * (x_) ** 2 + b * x_ + c


def exp_stretch_decay(x, amp, decay, offset, B):
    return offset + amp * np.exp(-((x / decay) ** B))


def exp_t2(x, amp, decay, offset):
    return exp_stretch_decay(x, amp, decay, offset, 3)


# def gaussian(x, *params):
def gaussian(x, coeff, mean, stdev, offset):
    """Calculates the value of a gaussian for the given input and parameters

    Params:
        x: float
            Input value
        params: tuple
            The parameters that define the Gaussian
            0: coefficient that defines the peak height
            1: mean, defines the center of the Gaussian
            2: standard deviation, defines the width of the Gaussian
            3: constant y value to account for background
    """

    # coeff, mean, stdev, offset = params
    var = stdev**2  # variance
    centDist = x - mean  # distance from the center
    return offset + coeff * np.exp(-(centDist**2) / (2 * var))


def sinexp(t, offset, amp, freq, decay):
    two_pi = 2 * np.pi
    half_pi = np.pi / 2
    return offset + (amp * np.sin((two_pi * freq * t) + half_pi)) * exp(-(decay**2) * t)


def cosexp(t, offset, amp, freq, decay):
    two_pi = 2 * np.pi
    return offset + (np.exp(-t / abs(decay)) * abs(amp) * np.cos((two_pi * freq * t)))


def inverted_cosexp(t, offset, freq, decay):
    two_pi = 2 * np.pi
    amp = offset - 1
    return offset - (np.exp(-t / abs(decay)) * abs(amp) * np.cos((two_pi * freq * t)))


def cosexp_1_at_0(t, offset, freq, decay):
    two_pi = 2 * np.pi
    amp = 1 - offset
    return offset + (np.exp(-t / abs(decay)) * abs(amp) * np.cos((two_pi * freq * t)))


def sin_1_at_0_phase(t, amp, offset, freq, phase):
    return offset + (abs(amp) * np.sin((freq * t - np.pi / 2 + phase)))


def sin_phase(t, amp, offset, freq, phase):
    return offset + (abs(amp) * np.sin((freq * t + phase)))


def cosine_sum(t, offset, decay, amp_1, freq_1, amp_2, freq_2, amp_3, freq_3):
    two_pi = 2 * np.pi

    return offset + np.exp(-t / abs(decay)) * (
        amp_1 * np.cos(two_pi * freq_1 * t)
        + amp_2 * np.cos(two_pi * freq_2 * t)
        + amp_3 * np.cos(two_pi * freq_3 * t)
    )


def cosine_double_sum(t, offset, decay, amp_1, freq_1, amp_2, freq_2):
    two_pi = 2 * np.pi

    return offset + np.exp(-t / abs(decay)) * (
        amp_1 * np.cos(two_pi * freq_1 * t) + amp_2 * np.cos(two_pi * freq_2 * t)
        # + amp_3 * np.cos(two_pi * freq_3 * t)
    )


def cosine_one(t, offset, decay, amp_1, freq_1):
    two_pi = 2 * np.pi

    return offset + np.exp(-t / abs(decay)) * (amp_1 * np.cos(two_pi * freq_1 * t))


def t2_func(t, amplitude, offset, t2):
    n = 3
    return amplitude * np.exp(-((t / t2) ** n)) + offset


def poiss_snr(sig, ref):
    """Take a list of signal and reference counts, and take their average,
    then calculate a snr.
    inputs:
        sig_count = list
        ref_counts = list
    outputs:
        snr = list
    """

    # Assume Poisson statistics on each count value
    # sig_noise = np.sqrt(sig)
    # ref_noise = np.sqrt(ref)
    # snr = (ref - sig) / np.sqrt(sig_noise**2 + ref_noise**2)
    # snr_per_readout = (snr / np.sqrt(num_reps))

    ref_count = np.array(ref)
    sig_count = np.array(sig)
    num_reps, num_points = ref_count.shape

    sig_count_avg = np.average(sig_count)
    ref_count_avg = np.average(ref_count)
    dif = sig_count_avg - ref_count_avg
    sig_noise = np.sqrt(sig_count_avg)
    ref_noise = np.sqrt(ref_count_avg)
    noise = np.sqrt(sig_noise**2 + ref_noise**2)
    snr = dif / noise

    N = sig_count_avg - ref_count_avg
    d = np.sqrt(sig_noise**2 + ref_noise**2)
    D = np.sqrt(sig_count_avg + ref_count_avg)
    d_d = 0.5 * d / D

    snr_unc = snr * np.sqrt((N / d) ** 2 + (d_d / D) ** 2)

    return snr, snr_unc


def get_scan_vals(center, scan_range, num_steps, dtype=float):
    """
    Returns a linspace for a scan centered about specified point
    """

    half_scan_range = scan_range / 2
    low = center - half_scan_range
    high = center + half_scan_range
    scan_vals = np.linspace(low, high, num_steps, dtype=dtype)
    # Deduplicate - may be necessary for ints and low scan ranges
    scan_vals = np.unique(scan_vals)
    return scan_vals


def bose(energy, temp):
    """Calculate Bose Einstein occupation number

    Parameters
    ----------
    energy : numeric
        Mode energy in meV
    temp : numeric
        Temperature in K

    Returns
    -------
    numeric
        Occupation number
    """
    # For very low temps we can get divide by zero and overflow warnings.
    # Fortunately, numpy is smart enough to know what we mean when this
    # happens, so let's let numpy figure it out and suppress the warnings.
    old_settings = np.seterr(divide="ignore", over="ignore")
    # print(energy / (Boltzmann * temp))
    val = 1 / (np.exp(energy / (Boltzmann * temp)) - 1)
    # Return error handling to default state for other functions
    np.seterr(**old_settings)
    return val


def process_counts(
    sig_counts, ref_counts, num_reps, readout, norm_mode=NormMode.SINGLE_VALUED
):
    """Extract the normalized average signal at each data point.
    Since we sometimes don't do many runs (<10), we often will have an
    insufficient sample size to run stats on for norm_avg_sig calculation.
    We assume Poisson statistics instead.

    Parameters
    ----------
    sig_counts : 2D array
        Signal counts from the experiment
    ref_counts : 2D array
        Reference counts from the experiment
    num_reps : int
        Number of experiment repetitions summed over for each point in sig or ref counts
    readout : numeric
        Readout duration in ns
    norm_mode : NormMode(enum), optional
        By default NormMode.SINGLE_VALUED

    Returns
    -------
    1D array
        Signal count rate averaged across runs
    1D array
        Reference count rate averaged across runs
    1D array
        Normalized average signal
    1D array
        Standard error of the normalized average signal
    """

    ref_counts = np.array(ref_counts)
    sig_counts = np.array(sig_counts)
    num_runs, num_points = ref_counts.shape
    readout_sec = readout * 1e-9

    # Find the averages across runs
    sig_counts_avg = np.average(sig_counts, axis=0)
    single_ref_avg = np.average(ref_counts)
    ref_counts_avg = np.average(ref_counts, axis=0)

    sig_counts_ste = np.sqrt(sig_counts_avg) / np.sqrt(num_runs)
    single_ref_ste = np.sqrt(single_ref_avg) / np.sqrt(num_runs * num_points)
    ref_counts_ste = np.sqrt(ref_counts_avg) / np.sqrt(num_runs)

    if norm_mode == NormMode.SINGLE_VALUED:
        norm_avg_sig = sig_counts_avg / single_ref_avg
        norm_avg_sig_ste = norm_avg_sig * np.sqrt(
            (sig_counts_ste / sig_counts_avg) ** 2
            + (single_ref_ste / single_ref_avg) ** 2
        )
    elif norm_mode == NormMode.POINT_TO_POINT:
        norm_avg_sig = sig_counts_avg / ref_counts_avg
        norm_avg_sig_ste = norm_avg_sig * np.sqrt(
            (sig_counts_ste / sig_counts_avg) ** 2
            + (ref_counts_ste / ref_counts_avg) ** 2
        )

    sig_counts_avg_kcps = (sig_counts_avg / (num_reps * 1000)) / readout_sec
    ref_counts_avg_kcps = (ref_counts_avg / (num_reps * 1000)) / readout_sec

    return (
        sig_counts_avg_kcps,
        ref_counts_avg_kcps,
        norm_avg_sig,
        norm_avg_sig_ste,
    )


# endregion
# region Config getters


@cache
def get_ref_img_array():
    config = common.get_config_module()
    ref_img_array = config.ref_img_array
    return ref_img_array


@cache
def get_apd_indices():
    "Get a list of the APD indices in use from the config"
    config_dict = common.get_config_dict()
    return config_dict["apd_indices"]


@cache
def get_apd_gate_channel():
    config_dict = common.get_config_dict()
    return config_dict["Wiring"]["Tagger"]["di_apd_gate"]


@cache
def get_common_duration(key):
    config = common.get_config_dict()
    common_duration = config["CommonDurations"][key]
    return common_duration


@cache
def get_virtual_laser_dict(virtual_laser_key):
    config = common.get_config_dict()
    return config["Optics"]["VirtualLasers"][virtual_laser_key]


@cache
def get_physical_laser_dict(physical_laser_name):
    config = common.get_config_dict()
    return config["Optics"]["PhysicalLasers"][physical_laser_name]


@cache
def get_physical_sig_gen_dict(physical_sig_gen_name):
    config = common.get_config_dict()
    return config["Microwaves"]["PhysicalSigGens"][physical_sig_gen_name]


@cache
def get_virtual_sig_gen_dict(sig_gen_ind):
    config = common.get_config_dict()
    return config["Microwaves"]["VirtualSigGens"][sig_gen_ind]


@cache
def get_physical_laser_name(laser_key):
    return get_virtual_laser_dict(laser_key)["physical_name"]


@cache
def get_uwave_dict(uwave_ind):
    config = common.get_config_dict()
    return config["Microwaves"][f"sig_gen_{uwave_ind}"]


# Server getters
# Each getter looks up the requested server from the config and
# returns a usable reference to the requested server (i.e. cxn.<server>)


def get_server_pulse_gen():
    """Get the pulse gen server for this setup, e.g. opx or swabian"""
    return common.get_server("pulse_gen")


def get_server_charge_readout_laser():
    """Get the laser for charge readout"""
    return common.get_server("charge_readout_laser")


def get_server_arb_wave_gen():
    """Get the arbitrary waveform generator server for this setup, e.g. opx or keysight"""
    return common.get_server("arb_wave_gen")


def get_server_camera():
    """Get the camera server"""
    return common.get_server("camera")


def get_server_counter():
    """Get the photon counter server for this setup, e.g. opx or swabian"""
    return common.get_server("counter")


def get_server_tagger():
    """Get the photon time tagger server for this setup, e.g. opx or swabian"""
    return common.get_server("tagger")


def get_server_temp_controller():
    return common.get_server("temp_controller")


def get_server_temp_monitor():
    return common.get_server("temp_monitor")


def get_server_power_supply():
    return common.get_server("power_supply")


def get_server_sig_gen(virtual_sig_gen_ind):
    """Retrieve the signal generator server based on the physical signal generator name."""
    # Fetch server connection based on the physical signal generator name
    virtual_sig_gen_dict = get_virtual_sig_gen_dict(virtual_sig_gen_ind)
    physical_sig_gen_name = virtual_sig_gen_dict["physical_name"]
    return common.get_server_by_name(physical_sig_gen_name)


def get_server_magnet_rotation():
    """Get the signal generator that controls magnet rotation angle"""
    return common.get_server("magnet_rotation")


@cache
def get_server_thorslm():
    """Get the Thorslm server."""
    return common.get_server("thorslm")


@cache
def get_server_thorcam():
    """Get the Thorslm server."""
    return common.get_server("thorcam")


# endregion
# region Email utils


def send_exception_email(email_from=None, email_to=None):
    default_email = common.get_default_email()
    if email_from is None:
        email_from = default_email
    if email_to is None:
        email_to = default_email
    # format_exc extracts the stack and error message from
    # the exception currently being handled.
    now = time.localtime()
    date = time.strftime("%A, %B %d, %Y", now)
    timex = time.strftime("%I:%M:%S %p", now)
    exc_info = traceback.format_exc()
    content = f"An unhandled exception occurred on {date} at {timex}.\n{exc_info}"
    send_email(content, email_from=email_from, email_to=email_to)


def send_email(content, email_from=None, email_to=None):
    default_email = common.get_default_email()
    if email_from is None:
        email_from = default_email
    if email_to is None:
        email_to = default_email
    pc_name = socket.gethostname()
    msg = MIMEText(content)
    msg["Subject"] = f"Alert from {pc_name}"
    msg["From"] = email_from
    msg["To"] = email_to

    pw = keyring.get_password("system", email_from)

    server = smtplib.SMTP("smtp.gmail.com", 587)  # port 465 or 587
    server.ehlo()
    server.starttls()
    server.ehlo()
    server.login(email_from, pw)
    server.sendmail(email_from, email_to, msg.as_string())
    server.close()


# endregion
# region Miscellaneous


def single_conversion(single_func, freq, *args):
    if type(freq) in [list, np.ndarray]:
        line = np.array([single_func(f, *args) for f in freq])
        return line
    else:
        return single_func(freq, *args)


# endregion
# region Rounding
"""
Various rounding tools, including several for presenting data with errors (round_for_print).
Relies on the decimals package for accurate arithmetic w/o binary rounding errors.
"""


def round_sig_figs(val, num_sig_figs):
    """Round a value to the passed number of sig figs

    Parameters
    ----------
    val : numeric
        Value to round
    num_sig_figs : int
        Number of sig figs to round to

    Returns
    -------
    numeric
        Rounded value
    """

    # All the work is done here
    def sub_fn(val, num_sig_figs):
        return round(val, -int(math.floor(math.log10(abs(val))) - num_sig_figs + 1))

    # Check for list/array/single value
    if isinstance(val, list):
        return [sub_fn(el, num_sig_figs) for el in val]
    elif type(val) is np.ndarray:
        rounded_val_list = [sub_fn(el, num_sig_figs) for el in val.tolist()]
        return np.array(rounded_val_list)
    else:
        return sub_fn(val, num_sig_figs)


def round_for_print_sci(val, err):
    """Round a value and associated error to the appropriate level given the
    magnitude of the error. The error will be rounded to 1 or 2 sig figs depending
    on whether the first sig fig is >1 or =1 respectively. Returned in a form
    suitable for scientific notation

    Parameters
    ----------
    val : numeric
        Value to round
    err : numeric
        Associated error

    Returns
    -------
    Decimal
        Rounded value as a string
    Decimal
        Rounded error as a string
    int
        Order of magnitude
    """

    val = Decimal(val)
    err = Decimal(err)

    try:
        err_mag = math.floor(math.log10(err))
    except Exception:
        return [0, 0, 0]
    sci_err = err / (Decimal(10) ** err_mag)
    first_err_digit = int(str(sci_err)[0])
    if first_err_digit == 1:
        err_sig_figs = 2
    else:
        err_sig_figs = 1

    try:
        power_of_10 = math.floor(math.log10(abs(val)))
    except Exception:
        power_of_10 = None
    if power_of_10 is None or power_of_10 < err_mag:
        power_of_10 = err_mag + err_sig_figs
    mag = Decimal(10) ** power_of_10
    rounded_err = round_sig_figs(err / mag, err_sig_figs)
    rounded_val = round(val / mag, (power_of_10 - err_mag) + err_sig_figs - 1)

    # Check for corner case where the value is e.g. 0.999 and rounds up to another decimal place
    if rounded_val >= 10 and err < val:
        power_of_10 += 1
        # Just shift the decimal over and recast to Decimal to ensure proper rounding
        rounded_err = Decimal(_shift_decimal_left(str(rounded_err)))
        rounded_val = Decimal(_shift_decimal_left(str(rounded_val)))

    return [rounded_val, rounded_err, power_of_10]


def round_for_print_sci_latex(val, err):
    """Round a value and associated error to the appropriate level given the
    magnitude of the error. The error will be rounded to 1 or 2 sig figs depending
    on whether the first sig fig is >1 or =1 respectively. Returned as a string
    to be put directly into LaTeX - the printed result will be in scientific notation

    Parameters
    ----------
    val : numeric
        Value to round
    err : numeric
        Associated error

    Returns
    -------
    str
        Rounded value including error and order of magnitude to be put directly into LaTeX
    """

    rounded_val, rounded_err, power_of_10 = round_for_print_sci(val, err)
    err_str = _strip_err(rounded_err)
    return r"\num{{{}({})e{}}}".format(rounded_val, err_str, power_of_10)


def round_for_print(val, err):
    """Round a value and associated error to the appropriate level given the
    magnitude of the error. The error will be rounded to 1 or 2 sig figs depending
    on whether the first sig fig is >1 or =1 respectively. Returned as a string
    to be printed directly in standard (not scientific) notation. As such, it is
    assumed that err < 1, otherwise the number of sig figs will be unclear

    Parameters
    ----------
    val : numeric
        Value to round
    err : numeric
        Associated error

    Returns
    -------
    str
        Rounded value including error to be printed directly
    """

    # If err > 10, this presentation style becomes unclear
    if err > 10:
        return None

    # Start from the scientific presentation
    rounded_val, rounded_err, power_of_10 = round_for_print_sci(val, err)

    # Get the representation of the actual value, where min_digits
    # ensures the last digit lines up with the error
    mag = Decimal(10) ** power_of_10
    str_rounded_err = str(rounded_err)
    val_str = np.format_float_positional(
        rounded_val * mag, min_digits=max(len(str_rounded_err) - 2 - power_of_10, 1)
    )

    # Trim possible trailing decimal point
    if val_str[-1] == ".":
        val_str = val_str[:-1]

    # Get the representation of the error, which is alway just the trailing non-zero digits
    err_str = _strip_err(rounded_err)

    return f"{val_str}({err_str})"


def _shift_decimal_left(val_str):
    """Finds the . character in a string and moves it one place to the left"""

    decimal_pos = val_str.find(".")
    # No decimal
    if decimal_pos == -1:
        last_char = val_str[-1]
        val_str = val_str.replace(last_char, f".{last_char}")
    else:
        left_char = val_str[decimal_pos - 1]
        val_str = val_str.replace(f"{left_char}.", f".{left_char}")
    return val_str


def _strip_err(err):
    """Get the representation of the error, which is alway just the trailing non-zero digits

    Parameters
    ----------
    err : str
        Error to process

    Returns
    -------
    str
        Trailing non-zero digits of err
    """
    stripped_err = ""
    trailing = False
    for char in str(err):
        if char == ".":
            continue
        elif char != "0":
            trailing = True
        if trailing:
            stripped_err += char
    return stripped_err


# endregion
# region Safe Stop
"""Use this to safely stop experiments without risking data loss or weird state.
Works by reassigning CTRL + C to set a global variable rather than raise a
KeyboardInterrupt exception. That way we can check on the global variable
whenever we like and stop the experiment appropriately. It's up to you (the
routine author) to place this in your routine appropriately.
"""


def init_safe_stop():
    """Call this at the beginning of a loop or other section which you may
    want to interrupt
    """
    global SAFESTOPFLAG
    # Tell the user safe stop has started if it was stopped or just not started
    try:
        if SAFESTOPFLAG:
            print("\nPress CTRL + C to stop...\n")
    except Exception:
        print("\nPress CTRL + C to  stop...\n")
    SAFESTOPFLAG = False
    signal.signal(signal.SIGINT, _safe_stop_handler)
    return


def _safe_stop_handler(sig, frame):
    """This should never need to be called directly"""
    global SAFESTOPFLAG
    SAFESTOPFLAG = True


def safe_stop():
    """Call this to check whether the user asked us to stop"""
    global SAFESTOPFLAG
    try:
        time.sleep(0.1)  # Pause execution to allow safe_stop_handler to run
        return SAFESTOPFLAG
    except Exception:
        return False


def reset_safe_stop():
    """Reset the Safe Stop flag, but don't remove the handler in case we
    want to reuse it.
    """
    global SAFESTOPFLAG
    SAFESTOPFLAG = False


def poll_safe_stop():
    """Blocking version of safe stop"""
    init_safe_stop()
    while not safe_stop():
        time.sleep(0.1)


# endregion
# region Reset hardware


def reset_cfm():
    """Reset our cfm so that it's ready to go for a new experiment. Avoids
    unnecessarily resetting components that may suffer hysteresis (ie the
    components that control xyz since these need to be reset in any
    routine where they matter anyway).
    """
    cxn = common.labrad_connect()
    cxn_server_names = cxn.servers
    for name in cxn_server_names:
        server = cxn[name]
        # Check for servers that ask not to be reset automatically
        if hasattr(server, "reset_cfm_opt_out"):
            continue
        if hasattr(server, "reset"):
            server.reset()


# endregion


# Testing
if __name__ == "__main__":
    test_a = _dual_threshold(
        [1, 2, 3, 4.5, 5, 6], [3, 3, 3, 4, 4, 4], [5, 5, 5, 5, 5, 5]
    )
    test_b = _dual_threshold([1, 2, 3, 2, 5, 6], [3, 3, 3, 4, 4, 4], [5, 5, 5, 5, 5, 5])
    test_c = _dual_threshold([1, 2, 3, 6, 5, 6], [3, 3, 3, 4, 4, 4], [5, 5, 5, 5, 5, 5])
    print(nan_corr_coef([test_a, test_b, test_c]))
