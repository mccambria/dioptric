# -*- coding: utf-8 -*-
"""
Analysis functions for bimodal histograms, the kind you get with single-shot readout.
Includes fitting functions and threshold determination

Created on November 11th, 2024

@author: mccambria
"""

import inspect
import sys
import time
from enum import Enum, auto
from functools import cache
from inspect import signature

import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import quad
from scipy.special import comb, factorial, gammainc, gammaincc, gammaln, xlogy
from scipy.stats import norm, poisson, skewnorm

from utils import kplotlib as kpl
from utils.tool_belt import curve_fit

inv_root_2_pi = 1 / np.sqrt(2 * np.pi)

# region Probability distributions


class ProbDist(Enum):
    POISSON = auto()
    BROADENED_POISSON = auto()
    GAUSSIAN = auto()
    SKEW_GAUSSIAN = auto()
    # For the following 4, see Cambria PRX 2025
    COMPOUND_POISSON = auto()
    COMPOUND_POISSON_WITH_IONIZATION = auto()
    NEGATIVE_BINOMIAL = auto()
    NEGATIVE_BINOMIAL_WITH_IONIZATION = auto()
    #


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
    fn_name = f"{prob_dist.name.lower()}_pdf"
    return eval(fn_name)


def get_single_mode_cdf(prob_dist: ProbDist):
    fn_name = f"{prob_dist.name.lower()}_cdf"
    return eval(fn_name)


def get_bimodal_pdf(prob_dist: ProbDist):
    if prob_dist is ProbDist.COMPOUND_POISSON_WITH_IONIZATION:
        dark_mode_fn = get_single_mode_pdf(ProbDist.COMPOUND_POISSON)
        bright_mode_fn = get_single_mode_pdf(ProbDist.COMPOUND_POISSON_WITH_IONIZATION)

        def bimodal_fn(x, dark_mode_weight, *params):
            if params[1] < params[0]:
                return [0] * len(x)
            bright_mode_weight = 1 - dark_mode_weight
            first_mode_val = dark_mode_fn(x, params[0])
            second_mode_val = bright_mode_fn(x, *params)
            return (
                dark_mode_weight * first_mode_val + bright_mode_weight * second_mode_val
            )

    if prob_dist is ProbDist.NEGATIVE_BINOMIAL_WITH_IONIZATION:
        dark_mode_fn = get_single_mode_pdf(ProbDist.NEGATIVE_BINOMIAL)
        bright_mode_fn = get_single_mode_pdf(ProbDist.NEGATIVE_BINOMIAL_WITH_IONIZATION)

        def bimodal_fn(x, dark_mode_weight, *params):
            if params[1] < params[0]:
                return [0] * len(x)
            bright_mode_weight = 1 - dark_mode_weight
            first_mode_val = dark_mode_fn(x, params[0])
            second_mode_val = bright_mode_fn(x, *params)
            return (
                dark_mode_weight * first_mode_val + bright_mode_weight * second_mode_val
            )

    else:
        single_mode_fn = get_single_mode_pdf(prob_dist)
        bimodal_fn = _get_bimodal_fn(single_mode_fn)

    return bimodal_fn


def get_bimodal_cdf(prob_dist: ProbDist):
    single_mode_fn = get_single_mode_cdf(prob_dist)
    return _get_bimodal_fn(single_mode_fn)


def _get_bimodal_fn(single_mode_fn):
    def bimodal_fn(x, dark_mode_weight, *params):
        bright_mode_weight = 1 - dark_mode_weight
        half_num_params = len(params) // 2
        first_mode_val = single_mode_fn(x, *params[:half_num_params])
        second_mode_val = single_mode_fn(x, *params[half_num_params:])
        return dark_mode_weight * first_mode_val + bright_mode_weight * second_mode_val

    return bimodal_fn


# @cache
def poisson_pdf(x, rate):
    # return poisson(mu=rate).pmf(x)
    # return (rate**x) * np.exp(-rate) / factorial(x)
    # Computing the pdf directly tends to overflow. Compute exp(ln(pdf)) instead
    return np.exp(xlogy(x, rate) - rate - gammaln(x + 1))


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


def compound_poisson_pdf(z, rate):
    if isinstance(z, list):
        z = np.array(z)
    z_not_array = not isinstance(z, np.ndarray)
    # If z is not an array, turn it into one so we can use the same code.
    # Convert back at the end.
    if z_not_array:
        z = np.array([z])
    z = z[np.newaxis, :]

    lower_lim = 0
    upper_lim = round(rate + 5 * np.sqrt(rate))
    integral_points = np.arange(lower_lim, upper_lim, 1, dtype=np.float64)
    integral_points = integral_points[:, np.newaxis]

    integrand = poisson_pdf(z, integral_points) * poisson_pdf(integral_points, rate)
    ret_val = np.sum(integrand, axis=0)
    if z_not_array:
        return ret_val[0]
    else:
        return ret_val


def negative_binomial_pdf(z, rate):
    if isinstance(z, list):
        z = np.array(z)
    z_not_array = not isinstance(z, np.ndarray)
    # If z is not an array, turn it into one so we can use the same code.
    # Convert back at the end.
    if z_not_array:
        z = np.array([z])

    p = 1 / 2
    ret_val = comb(z + rate - 1, z) * (1 - p) ** z * p**rate

    if z_not_array:
        return ret_val[0]
    else:
        return ret_val


def negative_binomial_with_ionization_pdf(z, lambda_0, lambda_m, ion):
    if isinstance(z, list):
        z = np.array(z)
    z_not_array = not isinstance(z, np.ndarray)
    # If z is not an array, turn it into one so we can use the same code.
    # Convert back at the end.
    if z_not_array:
        z = np.array([z])

    def integrand(tp, z_val):
        return (
            ion
            * np.exp(-ion * tp)
            * negative_binomial_pdf(z_val, lambda_m * tp + lambda_0 * (1 - tp))
        )

    def integrate(z_val):
        return quad(integrand, 0, 1, args=(z_val,))[0]

    part1 = np.exp(-ion) * negative_binomial_pdf(z, lambda_m)
    part2 = np.vectorize(integrate)(z)
    ret_val = part1 + part2

    if z_not_array:
        return ret_val[0]
    else:
        return ret_val


def negative_binomial_cdf(x, rate):
    return _calc_cdf(ProbDist.NEGATIVE_BINOMIAL, x, rate)


def negative_binomial_with_ionization_cdf(x, lambda_0, lambda_m, ion):
    return _calc_cdf(
        ProbDist.NEGATIVE_BINOMIAL_WITH_IONIZATION, x, lambda_0, lambda_m, ion
    )


def compound_poisson_with_ionization_pdf(z, lambda_0, lambda_m, ion):
    # ion = 0  # MCC
    if isinstance(z, list):
        z = np.array(z)
    z_not_array = not isinstance(z, np.ndarray)
    # If z is not an array, turn it into one so we can use the same code.
    # Convert back at the end.
    if z_not_array:
        z = np.array([z])
    z = z[np.newaxis, :]

    lower_lim = 0
    upper_lim = round(lambda_m + 5 * np.sqrt(lambda_m))
    integral_points = np.arange(lower_lim, upper_lim, 1, dtype=np.float64)
    integral_points = integral_points[:, np.newaxis]

    lambda_diff = lambda_m - lambda_0
    term_1 = poisson_pdf(integral_points, lambda_m) * (1 - ion + (1 / 2) * ion**2)
    coeff_23 = ion * (lambda_diff + ion * lambda_0) / (lambda_diff**2)
    term_2 = gammaincc(integral_points + 1, lambda_0)
    term_3 = gammaincc(integral_points + 1, lambda_m)
    coeff_45 = (ion**2) / (lambda_diff**2)
    term_4 = (integral_points + 1) * gammaincc(integral_points + 2, lambda_m)
    term_5 = (integral_points + 1) * gammaincc(integral_points + 2, lambda_0)
    integrand = poisson_pdf(z, integral_points) * (
        term_1 + coeff_23 * (term_2 - term_3) + coeff_45 * (term_4 - term_5)
    )

    ret_val = np.sum(integrand, axis=0)
    if z_not_array:
        return ret_val[0]
    else:
        return ret_val


def compound_poisson_cdf(x, rate):
    return _calc_cdf(ProbDist.COMPOUND_POISSON, x, rate)


def compound_poisson_with_ionization_cdf(x, lambda_0, lambda_m, ion):
    return _calc_cdf(
        ProbDist.COMPOUND_POISSON_WITH_IONIZATION, x, lambda_0, lambda_m, ion
    )


def broadened_poisson_pdf(x, rate, sigma, do_norm=True):
    if isinstance(x, (list, np.ndarray)):
        ret_vals = [broadened_poisson_pdf(el, rate, sigma) for el in x]
        return np.array(ret_vals)

    def integrand(y):
        return poisson_pdf(y, rate) * gaussian_pdf(x - y, 0, sigma)

    lower_lim = round(max(0, x - 4 * sigma))
    upper_lim = round(x + 4 * sigma)

    integral_points = np.arange(lower_lim, upper_lim, 1, dtype=np.float64)
    ret_val = np.sum(integrand(integral_points))
    return ret_val


def broadened_poisson_cdf(x, rate, sigma):
    return _calc_cdf(ProbDist.BROADENED_POISSON, x, rate, sigma)


# @cache
def gaussian_pdf(x, mean, std):
    return inv_root_2_pi * (1 / std) * np.exp(-0.5 * ((x - mean) / std) ** 2)
    # return norm(loc=mean, scale=std).pdf(x)


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
    counts_list, prob_dist: ProbDist, no_print=True, no_plot=True
):
    """Fit the passed probability distribution to a histogram of the passed counts_list.
    counts_list should have some population in both modes

    Parameters
    ----------
    counts_list : list | np.ndarray
        Array-like of recorded counts from an NV
    prob_dist : ProbDist
        Probability distribution to use for the fit
    no_print : bool, optional
        Whether to skip printing out the results of the fit, by default True
    no_plot : bool, optional
        Whether to skip plotting out the histogram and fit, by default True

    Returns
    -------
    np.ndarray(float)
        popt, the optimized fit parameters
    """

    counts_list = counts_list.flatten()

    # Remove outliers
    median = np.median(counts_list)
    std = np.std(counts_list)
    counts_list = counts_list[counts_list < median + 10 * std]
    counts_list = counts_list[counts_list > 0]
    num_samples = len(counts_list)

    # Histogram the counts
    # counts_list = np.array([round(el) for el in counts_list])
    max_count = round(max(counts_list))
    x_vals = np.linspace(0, max_count, max_count + 1)
    hist, bin_edges = np.histogram(
        counts_list, bins=max_count + 1, range=(0, max_count), density=True
    )
    mode = x_vals[np.argmax(hist)]

    # Histogram error bars - assume poisson statistics for each bin's distribution
    hist_errs = np.sqrt(hist / num_samples)
    min_err = 1 / num_samples  # Error we would calculate for bin with one occurrence
    hist_errs = np.where(hist_errs > min_err, hist_errs, min_err)  # Enforce no zeros

    ### Fit the histogram
    # Get guess params
    mean_dark_guess = round(np.quantile(counts_list, 0.15))
    mean_bright_guess = round(np.quantile(counts_list, 0.65))
    mean_dark_min = round(np.quantile(counts_list, 0.02))
    mean_bright_max = round(np.quantile(counts_list, 0.98))
    ratio_guess = 0.7 if mode < median else 0.3
    bounds = (-np.inf, np.inf)  # Default bounds
    if prob_dist is ProbDist.SKEW_GAUSSIAN:
        guess_params = [ratio_guess]
        guess_params.extend([mean_dark_guess, 2 * np.sqrt(mean_dark_guess), 2])
        guess_params.extend([mean_bright_guess, 2 * np.sqrt(mean_bright_guess), -2])
        skew_lim = 5
        bounds = (
            (0, mean_dark_min, 0, -skew_lim, mean_dark_min, 0, -skew_lim),
            (1, mean_bright_max, np.inf, skew_lim, mean_bright_max, np.inf, skew_lim),
        )
    elif prob_dist is ProbDist.POISSON:
        guess_params = (ratio_guess, mean_dark_guess, mean_bright_guess)
    elif prob_dist is ProbDist.BROADENED_POISSON:
        guess_params = (ratio_guess, mean_dark_guess, 3, mean_bright_guess, 3)
        bounds = (
            (0, mean_dark_min, 1, mean_dark_min, 1),
            (1, mean_bright_max, mean_dark_guess, mean_bright_max, mean_dark_guess),
        )
    elif prob_dist is ProbDist.COMPOUND_POISSON:
        guess_params = (ratio_guess, mean_dark_guess, mean_bright_guess)
        bounds = (
            (0, mean_dark_min, mean_dark_min),
            (1, mean_bright_max, mean_bright_max),
        )
    elif prob_dist is ProbDist.COMPOUND_POISSON_WITH_IONIZATION:
        guess_params = (ratio_guess, mean_dark_guess, mean_bright_guess, 0.0)
        bounds = (
            (0, mean_dark_min, mean_dark_min, 0.0),
            (1, mean_bright_max, mean_bright_max, 1.0),
        )
    elif prob_dist is ProbDist.NEGATIVE_BINOMIAL:
        guess_params = (ratio_guess, mean_dark_guess, mean_bright_guess)
        bounds = (
            (0, mean_dark_min, mean_dark_min),
            (1, mean_bright_max, mean_bright_max),
        )
    elif prob_dist is ProbDist.NEGATIVE_BINOMIAL_WITH_IONIZATION:
        guess_params = (ratio_guess, mean_dark_guess, mean_bright_guess, 0.0)
        bounds = (
            (0, mean_dark_min, mean_dark_min, 0.0),
            (1, mean_bright_max, mean_bright_max, 1.0),
        )
    # With p as free paramter
    # elif prob_dist is ProbDist.NEGATIVE_BINOMIAL:
    #     guess_params = (ratio_guess, mean_dark_guess, mean_bright_guess, 0.5)
    #     bounds = (
    #         (0, mean_dark_min, mean_dark_min, 0.25),
    #         (1, mean_bright_max, mean_bright_max, 0.75),
    #     )
    # elif prob_dist is ProbDist.NEGATIVE_BINOMIAL_WITH_IONIZATION:
    #     guess_params = (ratio_guess, mean_dark_guess, mean_bright_guess, 0.0, 0.5)
    #     bounds = (
    #         (0, mean_dark_min, mean_dark_min, 0.0, 0.25),
    #         (1, mean_bright_max, mean_bright_max, 10.0, 0.75),
    #     )

    # return guess_params

    # Fit
    fit_fn = get_bimodal_pdf(prob_dist)
    try:
        popt, pcov, red_chi_sq = curve_fit(
            fit_fn,
            x_vals,
            hist,
            guess_params,
            hist_errs,
            bounds=bounds,
            # ftol=1e-6,
            # xtol=1e-6,
        )
        if not no_print:
            print(f"Fit Parameters: {popt}")
            print(f"Reduced chi squared: {red_chi_sq}")

        if not no_plot:
            fig, ax = plt.subplots()
            ax.set_xlabel("Integrated counts")
            ax.set_ylabel("Probability")
            kpl.histogram(ax, counts_list, density=True)
            x_vals = np.linspace(0, np.max(counts_list), 1000)

            # Dark mode
            dark_ratio = popt[0]
            # single_mode_fn = get_single_mode_pdf(prob_dist)
            # num_params = get_single_mode_num_params(prob_dist)
            # line = dark_ratio * single_mode_fn(x_vals, *popt[1 : 1 + num_params])
            # line = dark_ratio * negative_binomial_pdf(x_vals, popt[1])
            line = dark_ratio * negative_binomial_pdf(x_vals, popt[1])
            kpl.plot_line(
                ax, x_vals, line, color=kpl.KplColors.RED, label=r"NV$^{0}$ mode"
            )

            # Bright mode
            bright_ratio = 1 - dark_ratio
            # num_params = get_single_mode_num_params(prob_dist)
            # line = (1 - dark_ratio) * single_mode_fn(x_vals, *popt[1 + num_params :])
            line = bright_ratio * negative_binomial_with_ionization_pdf(
                x_vals, *popt[1:]
            )
            kpl.plot_line(
                ax, x_vals, line, color=kpl.KplColors.GREEN, label=r"NV$^{-}$ mode"
            )

            # Both modes
            line = fit_fn(x_vals, *popt)
            kpl.plot_line(ax, x_vals, line, color=kpl.KplColors.BLUE, label="Combined")

            ax.legend(loc=kpl.Loc.UPPER_RIGHT)
            kpl.show(block=True)
        return popt, pcov, red_chi_sq
    except Exception as exc:
        raise exc
        return None, None, None


def determine_threshold(
    popt, prob_dist: ProbDist, dark_mode_weight=None, do_print=False, ret_fidelity=False
):
    """Determine the optimal threshold for assigning a state based on a measured number of counts

    Parameters
    ----------
    popt : np.ndarray
        Optimized fit parameters
    prob_dist : ProbDist
        Probability distribution uses in the fit
    dark_mode_weight : float, optional
        Portion of measurements that project into the dark mode, by default popt[0],
        the dark mode weight parameter from the fit
    no_print : bool, optional
        Whether to skip printing out the results of the determination, by default False
    ret_fidelity : bool, optional
        Whether to return the readout fidelity expected under the optimal threshold, by default False

    Returns
    -------
    float | list(float)
        The threshold or the threshold and the expected readout fidelity
    """
    if popt is None:
        if ret_fidelity:
            return None, None
        else:
            return None

    if dark_mode_weight is None:
        dark_mode_weight = popt[0]
    bright_mode_weight = 1 - dark_mode_weight

    num_single_mode_params = get_single_mode_num_params(prob_dist)

    # Calculate fidelity (probability of calling state correctly) for given threshold
    # mean_counts_dark, mean_counts_bright = popt[1], popt[1 + num_single_mode_params]
    # MCC hack for including ionization
    mean_counts_dark, mean_counts_bright = popt[1], popt[2]
    mean_counts_dark = round(mean_counts_dark)
    mean_counts_bright = round(mean_counts_bright)
    thresh_options = np.arange(mean_counts_dark - 0.5, mean_counts_bright + 0.5, 1)
    fidelities = []
    snrs = []
    left_fidelities = []
    right_fidelities = []
    single_mode_cdf = get_single_mode_cdf(prob_dist)
    for val in thresh_options:
        # dark_left_prob = single_mode_cdf(val, *popt[1 : 1 + num_single_mode_params])
        # bright_left_prob = single_mode_cdf(val, *popt[1 + num_single_mode_params :])
        # MCC hack for including ionization
        dark_mode_cdf = get_single_mode_cdf(ProbDist.NEGATIVE_BINOMIAL)
        dark_left_prob = dark_mode_cdf(val, popt[1])
        bright_mode_cdf = get_single_mode_cdf(
            ProbDist.NEGATIVE_BINOMIAL_WITH_IONIZATION
        )
        bright_left_prob = bright_mode_cdf(val, *popt[1:])
        # Pass lambda_0, lambda_m, ion
        # bright_left_prob = bright_mode_cdf(val, popt[1], popt[2], popt[3])

        dark_right_prob = 1 - dark_left_prob
        bright_right_prob = 1 - bright_left_prob

        fidelity = (
            dark_mode_weight * dark_left_prob + bright_mode_weight * bright_right_prob
        )
        fidelities.append(fidelity)

        signal = (
            dark_mode_weight * dark_right_prob + bright_mode_weight * bright_right_prob
        )
        dark_var = dark_mode_weight**2 * dark_left_prob * dark_right_prob
        bright_var = bright_mode_weight**2 * bright_left_prob * bright_right_prob
        noise = np.sqrt(dark_var + bright_var)
        snrs.append(signal / noise)

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
    print(f"Threshold from fidelity: {threshold}")
    snr = np.max(snrs)
    threshold = thresh_options[np.argmax(snrs)]
    print(f"Threshold from SNR: {threshold}")

    if do_print:
        print(
            f"Optimum readout fidelity {round(fidelity, 3)} achieved at threshold {threshold}"
        )

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
    kpl.init_kplotlib()
    # (z, lambda_0, lambda_m, ion)
    fig, ax = plt.subplots()
    x_vals = np.linspace(0, 65, 1000)
    # line_vals = compound_poisson_with_ionization_pdf(x_vals, 20, 40, 0.0)
    line_vals = compound_poisson_pdf(x_vals, 10)
    # print(np.sum(line_vals) * x_vals[1] - x_vals[0])
    kpl.plot_line(ax, x_vals, line_vals)
    line_vals = negative_binomial_pdf(x_vals, 10)
    kpl.plot_line(ax, x_vals, line_vals, color=kpl.KplColors.RED)
    line_vals = negative_binomial_with_ionization_pdf(x_vals, 10, 30, 10)
    kpl.plot_line(ax, x_vals, line_vals, color=kpl.KplColors.GREEN)
    kpl.show(block=True)
