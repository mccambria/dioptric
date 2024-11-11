# -*- coding: utf-8 -*-
"""
Analysis functions for bimodal histograms, the kind you get with single-shot readout.
Includes fitting functions and threshold determination

Created on November 11th, 2024

@author: mccambria
"""

from enum import Enum, auto

import numpy as np
from scipy.optimize import curve_fit
from scipy.special import factorial
from scipy.stats import norm, skewnorm

# region Probability distributions


class ProbabilityDistribution(Enum):
    POISSON = auto()
    GAUSSIAN = auto()
    SKEW_GAUSSIAN = auto()


def get_single_mode_pdf(dist: ProbabilityDistribution):
    if dist is ProbabilityDistribution.POISSON:
        return poisson_pdf
    if dist is ProbabilityDistribution.GAUSSIAN:
        return gaussian_pdf
    if dist is ProbabilityDistribution.SKEW_GAUSSIAN:
        return skew_gaussian_pdf


def get_single_mode_cdf(dist: ProbabilityDistribution):
    if dist is ProbabilityDistribution.POISSON:
        return poisson_cdf
    if dist is ProbabilityDistribution.GAUSSIAN:
        return gaussian_cdf
    if dist is ProbabilityDistribution.SKEW_GAUSSIAN:
        return skew_gaussian_cdf


def get_bimodal_pdf(dist: ProbabilityDistribution):
    single_mode_fn = get_single_mode_pdf(dist)
    return _get_bimodal_fn(single_mode_fn)


def get_bimodal_cdf(dist: ProbabilityDistribution):
    single_mode_fn = get_single_mode_cdf(dist)
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
    """Cumulative distribution function for poisson pdf. Integrates
    up to and including x"""
    x_floor = int(np.floor(x))
    val = 0
    for ind in range(x_floor):
        val += poisson_pdf(ind, rate)
    return val


def gaussian_pdf(x, mean, std):
    return norm(loc=mean, scale=std).pdf(x)


def gaussian_cdf(x, mean, std):
    return norm(loc=mean, scale=std).cdf(x)


def skew_gaussian_pdf(x, mean, std, skew):
    return skewnorm(a=skew, loc=mean, scale=std).pdf(x)


def skew_gaussian_cdf(x, mean, std, skew):
    return skewnorm(a=skew, loc=mean, scale=std).cdf(x)


# endregion


def fit_bimodal_histogram(
    counts_list,
    no_print=False,
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

    # Fit the histogram
    single_mode_pdf = poisson_pdf

    def fit_fn(x, first_mode_weight, *params):
        return bimodal_pdf(x, single_mode_pdf, first_mode_weight, *params)

    mean_nv0_guess = round(np.quantile(counts_list, 0.2))
    mean_nvn_guess = round(np.quantile(counts_list, 0.98))
    guess_params = (
        0.7,
        mean_nv0_guess,
        # 2 * np.sqrt(mean_nv0_guess),
        # 2,
        mean_nvn_guess,
        # 2 * np.sqrt(mean_nvn_guess),
        # -2,
    )
    skew_lim = 5
    mean_nv0_min = round(np.quantile(counts_list, 0.02))
    mean_nvn_max = round(np.quantile(counts_list, 0.98))
    bounds = (
        (0, mean_nv0_min, 0, -skew_lim, mean_nv0_min, 0, -skew_lim),
        (1, mean_nvn_max, np.inf, skew_lim, mean_nvn_max, np.inf, skew_lim),
    )
    bounds = (-np.inf, np.inf)
    try:
        popt, _ = curve_fit(fit_fn, x_vals, hist, p0=guess_params, bounds=bounds)
        if not no_print:
            print(popt)
        return popt
    except Exception as exc:
        return None


def determine_threshold(
    counts_list, nvn_ratio=None, no_print=False, ret_fidelity=False
):
    popt = fit_bimodal_histogram(counts_list, no_print)

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
    prob_errs = []
    # cdf = skew_gaussian_cdf
    cdf = poisson_cdf
    for val in thresh_options:
        nv0_left_prob = cdf(val, *popt[1 : 1 + num_single_dist_params])
        nvn_left_prob = cdf(val, *popt[1 + num_single_dist_params :])
        nv0_right_prob = 1 - nv0_left_prob
        nvn_right_prob = 1 - nvn_left_prob
        prob_errs.append(np.abs(nvn_ratio - nvn_left_prob))
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
    fit_fn = bimodal_skew_gaussian_pdf
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
