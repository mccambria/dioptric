# -*- coding: utf-8 -*-
"""
Analysis functions for bimodal histograms, the kind you get with single-shot readout.
Includes fitting functions and threshold determination

Created on November 11th, 2024

@author: mccambria
"""

import inspect
import sys
from enum import Enum, auto
from inspect import signature

import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import quad
from scipy.optimize import curve_fit
from scipy.special import factorial, gammainc
from scipy.stats import norm, skewnorm

from utils import kplotlib as kpl

# region Probability distributions


class ProbDist(Enum):
    POISSON = auto()
    BROADENED_POISSON = auto()
    GAUSSIAN = auto()
    SKEW_GAUSSIAN = auto()


def get_single_mode_num_params(prob_dist: ProbDist):
    single_mode_pdf = get_single_mode_pdf(prob_dist)
    sig = signature(single_mode_pdf)
    # Loop through params, count only non-optional
    num_params = 0
    for param in sig.parameters.values():
        if param.default is param.empty:
            num_params += 1

    # Exclude first param, x, the point to evaluate at
    return num_params - 1


def get_single_mode_pdf(prob_dist: ProbDist):
    if prob_dist is ProbDist.POISSON:
        return poisson_pdf
    if prob_dist is ProbDist.GAUSSIAN:
        return gaussian_pdf
    if prob_dist is ProbDist.SKEW_GAUSSIAN:
        return skew_gaussian_pdf
    if prob_dist is ProbDist.BROADENED_POISSON:
        return broadened_poisson_pdf


def get_single_mode_cdf(prob_dist: ProbDist):
    if prob_dist is ProbDist.POISSON:
        return poisson_cdf
    if prob_dist is ProbDist.GAUSSIAN:
        return gaussian_cdf
    if prob_dist is ProbDist.SKEW_GAUSSIAN:
        return skew_gaussian_cdf
    if prob_dist is ProbDist.BROADENED_POISSON:
        return broadened_poisson_cdf


def get_bimodal_pdf(prob_dist: ProbDist):
    single_mode_fn = get_single_mode_pdf(prob_dist)
    return _get_bimodal_fn(single_mode_fn)


def get_bimodal_cdf(prob_dist: ProbDist):
    single_mode_fn = get_single_mode_cdf(prob_dist)
    return _get_bimodal_fn(single_mode_fn)


def _get_bimodal_fn(single_mode_fn):
    def bimodal_fn(x, first_mode_weight, *params):
        second_mode_weight = 1 - first_mode_weight
        half_num_params = len(params) // 2
        first_mode_val = single_mode_fn(x, *params[:half_num_params])
        second_mode_val = single_mode_fn(x, *params[half_num_params:])
        return first_mode_weight * first_mode_val + second_mode_weight * second_mode_val

    return bimodal_fn


def poisson_pdf(x, rate):
    return (rate**x) * np.exp(-rate) / factorial(x)


def poisson_cdf(x, rate):
    return _calc_cdf(ProbDist.POISSON, x, rate)


def _calc_cdf(prob_dist, x, *params):
    """Cumulative distribution function for poisson pdf. Integrates
    up to and including x"""
    pdf = get_single_mode_pdf(prob_dist)
    x_floor = int(np.floor(x))
    val = 0
    for ind in range(x_floor):
        val += pdf(ind, *params)
    return val


def broadened_poisson_pdf(x, rate, sigma, do_norm=True):
    if isinstance(x, (list, np.ndarray)):
        ret_vals = [broadened_poisson_pdf(el, rate, sigma) for el in x]
        return np.array(ret_vals)

    def integrand(y):
        return poisson_pdf(y, rate) * gaussian_pdf(x - y, 0, sigma)

    lower_lim = round(max(0, rate - 5 * np.sqrt(rate)))
    upper_lim = round(rate + 5 * np.sqrt(rate))

    ret_val = np.sum(integrand(np.arange(lower_lim, upper_lim, 1, dtype=np.float64)))
    return ret_val

    adj_rate = rate * sigma
    if do_norm:
        x_vals = np.array(range(100), dtype=int)
        norm = np.sum(broadened_poisson_pdf(x_vals, rate, sigma, do_norm=False))
        return (1 / norm) * poisson_pdf(x / sigma, rate)
    else:
        return poisson_pdf(x / sigma, rate)
    # norm = np.exp(-rate) * exponential_integral(1 / shape, rate ** (1 / shape))

    # # norm = 1
    # return (1 / norm) * poisson_pdf(x / shape, rate)


def broadened_poisson_cdf(x, rate, stretch_factor):
    return _calc_cdf(ProbDist.BROADENED_POISSON, x, rate, stretch_factor)


def gaussian_pdf(x, mean, std):
    return norm(loc=mean, scale=std).pdf(x)


def gaussian_cdf(x, mean, std):
    return norm(loc=mean, scale=std).cdf(x)


def skew_gaussian_pdf(x, mean, std, skew):
    return skewnorm(a=skew, loc=mean, scale=std).pdf(x)


def skew_gaussian_cdf(x, mean, std, skew):
    return skewnorm(a=skew, loc=mean, scale=std).cdf(x)


def exponential_integral(nu, z):
    return (z ** (nu - 1)) * gammainc(1 - nu, z)


# endregion


def fit_bimodal_histogram(
    counts_list, no_print=False, prob_dist: ProbDist = ProbDist.SKEW_GAUSSIAN
):
    """counts_list should have some population in both modes"""

    counts_list = counts_list.flatten()

    # Remove outliers
    median = np.median(counts_list)
    std = np.std(counts_list)
    counts_list = counts_list[counts_list < median + 10 * std]

    # Histogram the counts
    # counts_list = np.array([round(el) for el in counts_list])
    max_count = round(max(counts_list))
    x_vals = np.linspace(0, max_count, max_count + 1)
    hist, bin_edges = np.histogram(
        counts_list, bins=max_count + 1, range=(0, max_count), density=True
    )

    ### Fit the histogram

    # Get guess params
    mean_dark_guess = round(np.quantile(counts_list, 0.2))
    mean_bright_guess = round(np.quantile(counts_list, 0.98))
    mean_dark_min = round(np.quantile(counts_list, 0.02))
    mean_bright_max = round(np.quantile(counts_list, 0.98))
    bounds = (-np.inf, np.inf)  # Default bounds
    if prob_dist is ProbDist.SKEW_GAUSSIAN:
        guess_params = [0.7]
        guess_params.extend([mean_dark_guess, 2 * np.sqrt(mean_dark_guess), 2])
        guess_params.extend([mean_bright_guess, 2 * np.sqrt(mean_bright_guess), -2])
        skew_lim = 5
        bounds = (
            (0, mean_dark_min, 0, -skew_lim, mean_dark_min, 0, -skew_lim),
            (1, mean_bright_max, np.inf, skew_lim, mean_bright_max, np.inf, skew_lim),
        )
    elif prob_dist is ProbDist.POISSON:
        guess_params = (0.7, mean_dark_guess, mean_bright_guess)
    elif prob_dist is ProbDist.BROADENED_POISSON:
        guess_params = (0.7, mean_dark_guess, 2, mean_bright_guess, 2)
        bounds = (
            (0, mean_dark_min, 0, mean_dark_min, 0),
            (1, mean_bright_max, np.inf, mean_bright_max, np.inf),
        )

    # Fit
    fit_fn = get_bimodal_pdf(prob_dist)
    try:
        popt, _ = curve_fit(fit_fn, x_vals, hist, p0=guess_params, bounds=bounds)
        if not no_print:
            print(popt)
        return popt
    except Exception as exc:
        return None


def determine_threshold(
    counts_list,
    bright_ratio=None,
    no_print=False,
    ret_fidelity=False,
    prob_dist: ProbDist = ProbDist.SKEW_GAUSSIAN,
):
    popt = fit_bimodal_histogram(counts_list, no_print, prob_dist)

    # Popt will be None
    if popt is None:
        if ret_fidelity:
            return None, None
        else:
            return None

    if bright_ratio is None:
        bright_ratio = 1 - popt[0]
    dark_ratio = 1 - bright_ratio

    num_single_mode_params = get_single_mode_num_params(prob_dist)

    # Calculate fidelity (probability of calling state correctly) for given threshold
    mean_counts_dark, mean_counts_bright = popt[1], popt[1 + num_single_mode_params]
    mean_counts_dark = round(mean_counts_dark)
    mean_counts_bright = round(mean_counts_bright)
    thresh_options = np.arange(0.5, np.max(counts_list) + 0.5, 1)
    fidelities = []
    left_fidelities = []
    right_fidelities = []
    single_mode_cdf = get_single_mode_cdf(prob_dist)
    for val in thresh_options:
        dark_left_prob = single_mode_cdf(val, *popt[1 : 1 + num_single_mode_params])
        dark_right_prob = 1 - dark_left_prob
        bright_left_prob = single_mode_cdf(val, *popt[1 + num_single_mode_params :])
        bright_right_prob = 1 - bright_left_prob

        fidelity = dark_ratio * dark_left_prob + bright_ratio * bright_right_prob
        fidelities.append(fidelity)

        # Two-sided
        # left_fidelity = (dark_ratio * dark_left_prob) / (
        #     dark_ratio * dark_left_prob + bright_ratio * bright_left_prob
        # )
        # right_fidelity = (bright_ratio * bright_right_prob) / (
        #     bright_ratio * bright_right_prob + dark_ratio * dark_right_prob
        # )
        # left_fidelities.append(left_fidelity)
        # right_fidelities.append(right_fidelity)
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
    bright_ratio=None,
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
    fit_fn = bimodal_skew_gaussian_pdf
    num_single_dist_params = 3
    mean_dark_guess = round(np.quantile(counts_list, 0.2))
    mean_bright_guess = round(np.quantile(counts_list, 0.98))
    guess_params = (
        0.7,
        mean_dark_guess,
        2 * np.sqrt(mean_dark_guess),  # 1.5 factor for broadening
        2,
        mean_bright_guess,
        2 * np.sqrt(mean_bright_guess),
        -2,
    )
    popt, _ = curve_fit(fit_fn, x_vals, hist, p0=guess_params)
    if not no_print:
        print(popt)

    if bright_ratio is None:
        bright_ratio = 1 - popt[0]
    dark_ratio = 1 - bright_ratio

    # Calculate fidelities for given threshold
    mean_counts_dark, mean_counts_bright = popt[1], popt[1 + num_single_dist_params]
    mean_counts_dark = round(mean_counts_dark)
    mean_counts_bright = round(mean_counts_bright)
    thresh_options = np.arange(0.5, np.max(counts_list) + 0.5, 1)
    num_options = len(thresh_options)
    fidelities = []
    left_fidelities = []
    right_fidelities = []
    for val in thresh_options:
        dark_left_prob = skew_gaussian_cdf(val, *popt[1 : 1 + num_single_dist_params])
        bright_left_prob = skew_gaussian_cdf(val, *popt[1 + num_single_dist_params :])
        dark_right_prob = 1 - dark_left_prob
        bright_right_prob = 1 - bright_left_prob
        fidelity = dark_ratio * dark_left_prob + bright_ratio * bright_right_prob
        left_fidelity = (dark_ratio * dark_left_prob) / (
            dark_ratio * dark_left_prob + bright_ratio * bright_left_prob
        )
        right_fidelity = (bright_ratio * bright_right_prob) / (
            bright_ratio * bright_right_prob + dark_ratio * dark_right_prob
        )
        fidelities.append(fidelity)
        left_fidelities.append(left_fidelity)
        right_fidelities.append(right_fidelity)
    single_threshold = thresh_options[np.argmax(fidelities)]
    best_fidelity = np.max(fidelities)

    # Calculate normalized probabilities for given integrated counts value
    norm_dark_probs = []
    norm_bright_probs = []
    for val in x_vals:
        dark_prob = skew_gaussian_pdf(val, *popt[1 : 1 + num_single_dist_params])
        bright_prob = skew_gaussian_pdf(val, *popt[1 + num_single_dist_params :])
        norm_dark_prob = (
            dark_ratio
            * dark_prob
            / (dark_ratio * dark_prob + bright_ratio * bright_prob)
        )
        norm_bright_prob = (
            bright_ratio
            * bright_prob
            / (dark_ratio * dark_prob + bright_ratio * bright_prob)
        )
        norm_dark_probs.append(norm_dark_prob)
        norm_bright_probs.append(norm_bright_prob)
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
        norm_dark_probs = np.array(norm_dark_probs)
        norm_bright_probs = np.array(norm_bright_probs)
        adj_norm_dark_probs = np.where(
            norm_dark_probs > min_fidelity, norm_dark_probs, 1
        )
        adj_norm_bright_probs = np.where(
            norm_bright_probs > min_fidelity, norm_bright_probs, 1
        )
        threshold = [
            x_vals[np.argmin(adj_norm_dark_probs)] + 0.5,
            x_vals[np.argmin(adj_norm_bright_probs)] - 0.5,
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
        # popt: prob_dark, mean_dark, std_dark, skew_dark, mean_bright, std_bright, skew_bright
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


if __name__ == "__main__":
    # print(get_single_mode_num_params(ProbDist.BROADENED_POISSON))
    # sys.exit()

    kpl.init_kplotlib()
    fig, ax = plt.subplots()
    x_vals = np.linspace(0, 50, 1000)
    kpl.plot_line(ax, x_vals, poisson_pdf(x_vals, 20), label="Poisson")
    kpl.plot_line(
        ax, x_vals, broadened_poisson_pdf(x_vals, 20, 5), label="Broadened Poisson"
    )
    kpl.show(block=True)
