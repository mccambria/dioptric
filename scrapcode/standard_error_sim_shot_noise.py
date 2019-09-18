# -*- coding: utf-8 -*-
"""The standard error should tell us the standard deviation of whatever 
distribution we're pulling from. In the case of the mean, we're pulling from
the sampling distribution with N = however many samples we have in total. In
the more complex case of the standard error on extracted parameters, we're
pulling from the distribution of repeated curve fits to some random data. To
make sure our standard error is telling us what we think it is in this case,
let's just simulate this distribution and see what happens.

Created on Sun Sep 16 2019

@author: mccambria
"""


# %% Imports


import numpy
from numpy import exp
from numpy import sqrt
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from scipy import stats


# %% Constants


# %% Functions


def exp_decay(x, coeff, rate):
    return coeff*exp(-rate*x)+(1-coeff)

def exp_inc

def simulate_shot_noisy_exp_decay(x_linspace, coeff, rate, num_runs, num_reps):
    num_steps = len(x_linspace)
    data = numpy.zeros((num_runs, num_steps))
    for run_ind in range(num_runs):
        for step_ind in range(num_steps):
            x_val = x_linspace[step_ind]
            brightness = exp_decay(x_val, coeff, rate)
            counts = numpy.sum(numpy.random.poisson(brightness, num_reps))
            data[run_ind, step_ind] = counts / num_reps
    return data


# %% Main


def main(num_samples, num_runs, num_steps, num_reps, time_range, coeff, rate):
    """When you run the file, we'll call into main, which should contain the
    body of the script.
    """

    x_vals = numpy.linspace(time_range[0], time_range[1], num_steps)
    x_smooth = numpy.linspace(time_range[0], time_range[1], 101)
    rate_ste_list = []
    rate_avg_list = []

    for sample_ind in range(num_samples):

        # Simulate the data

        data = simulate_shot_noisy_exp_decay(x_vals, coeff, rate,
                                             num_runs, num_reps)
        data_avg = numpy.mean(data, axis=0)
        data_std = numpy.std(data, axis=0, ddof=1)
        data_ste = data_std / numpy.sqrt(num_runs)
        # data_ste = [numpy.mean(data_ste)]*num_steps
        # sigmas = data_avg - exp_decay(x_vals, coeff, rate)
        # data_ste = stats.sem(data, axis=0)
        # data_stv = numpy.var(data, axis=0, ddof=1) / numpy.sqrt(num_runs)

        # Fit

        popt, pcov = curve_fit(exp_decay, x_vals, data_avg,
                      p0=(coeff, rate), sigma=data_ste, absolute_sigma=True)

        # Populate our lists

        rate_ste = pcov[1,1]
        rate_ste = sqrt(rate_ste)
        rate_ste_list.append(rate_ste)
        rate_avg_list.append(popt[1])

        # Plot an example

        if sample_ind == 0:
            fig, ax = plt.subplots(figsize=(8.5, 8.5))
            fig.set_tight_layout(True)
            ax.errorbar(x_vals, data_avg, yerr=data_ste,
                        label='data', fmt='o', color='blue')
            ax.plot(x_smooth, exp_decay(x_smooth, *popt),
                    color='red', label = 'fit')
            ax.legend()
            temp = [el for el in popt]
            temp[1] += rate_ste
            ax.plot(x_smooth, exp_decay(x_smooth, *temp))
            temp = [el for el in popt]
            temp[1] -= rate_ste
            ax.plot(x_smooth, exp_decay(x_smooth, *temp))

    # Get the standard devation of the distibution

    mean_ste = numpy.mean(rate_ste_list)
    print(mean_ste)
    fit_std = numpy.std(rate_avg_list, ddof=1)
    print(fit_std)
    return mean_ste / fit_std


# %% Run the file


# The __name__ variable will only be '__main__' if you run this file directly.
# This allows a file's functions, classes, etc to be imported without running
# the script that you set up here.
if __name__ == '__main__':

    # Set up your parameters to be passed to main here
    num_runs = 50
    num_samples = 100
    num_reps = 10**3
    # num_runs = 100
    # num_samples = num_runs // 10
    # num_samples = 100
    # num_runs = num_samples // 10
    num_steps = 26
    time_range = [0, 6]
    coeff = 0.75
    rate = 0.9

    # Run the script
    main(num_samples, num_runs, num_steps, num_reps, time_range, coeff, rate)
    # test = []
    # num_runs_vals = numpy.linspace(5,25,11, dtype=int)
    # for val in num_runs_vals:
    #     print(val)
    #     test.append(main(num_samples, val, num_steps, time_range, coeff, rate))
    # plt.plot(num_runs_vals, test)
    # test = []
    # num_steps_vals = numpy.linspace(5,25,11, dtype=int)
    # for val in num_steps_vals:
    #     print(val)
    #     test.append(main(num_samples, num_runs, val, time_range, coeff, rate))
    # plt.plot(num_steps_vals, test)
