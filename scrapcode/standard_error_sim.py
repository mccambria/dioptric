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
    return coeff*exp(-rate*x)

def simulate_noisy_exp_decay(x_linspace, coeff, rate, num_runs, st_dev_factor=300):
    num_steps = len(x_linspace)
    data = numpy.zeros((num_runs, num_steps))
    st_dev = sqrt(num_runs/(sqrt(rate)*st_dev_factor))*coeff  # Chosen for a rough quantitative match
    # st_dev = sqrt(num_runs/10000)*coeff
    noise_list = numpy.random.normal(0, st_dev, num_runs*num_steps)
    for run_ind in range(num_runs):
        for step_ind in range(num_steps):
            x_val = x_linspace[step_ind]
            y_val = exp_decay(x_val, coeff, rate)
            noise = noise_list[(run_ind*num_steps)+step_ind]
            data[run_ind, step_ind] = y_val+noise
    return data


# %% Main


def main(num_samples, num_runs, num_steps, time_range, coeff, rate):
    """When you run the file, we'll call into main, which should contain the
    body of the script.
    """

    multi = False
    if (type(num_runs) is list):
        multi = True
        num_multi = len(num_runs)

    if not multi:
        x_vals = numpy.linspace(time_range[0], time_range[1], num_steps)
        x_smooth = numpy.linspace(time_range[0], time_range[1], 101)
    rate_ste_list = []
    rate_avg_list = []

    for sample_ind in range(num_samples):

        # Simulate the data
        if multi:
            x_vals_cum = []
            data_avg_cum = []
            data_ste_cum = []
            st_dev_factors = [500, 300]
            for ind in range(num_multi):
                tr = time_range[ind]
                x_vals = numpy.linspace(tr[0], tr[1], num_steps[ind])
                data = simulate_noisy_exp_decay(x_vals, coeff, rate,
                                        num_runs[ind], st_dev_factors[ind])
                data_avg = numpy.mean(data, axis=0)
                data_std = numpy.std(data, axis=0, ddof=1)
                data_ste = data_std / numpy.sqrt(num_runs[ind])
                x_vals_cum.extend(x_vals.tolist())
                data_avg_cum.extend(data_avg.tolist())
                data_ste_cum.extend(data_ste.tolist())
            data_avg = data_avg_cum
            data_ste = data_ste_cum
            x_vals = x_vals_cum
            x_smooth = numpy.linspace(0, max(x_vals), 101)

        else:
            data = simulate_noisy_exp_decay(x_vals, coeff, rate, num_runs)
            data_avg = numpy.mean(data, axis=0)
            data_std = numpy.std(data, axis=0, ddof=1)
            data_ste = data_std / numpy.sqrt(num_runs)

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

    # print(numpy.mean(rate_avg_list) / 3)
    # mean_ste = numpy.mean(rate_ste_list)
    # print(mean_ste / 3)
    # fit_std = numpy.std(rate_avg_list, ddof=1)
    # print(fit_std / 3)

    print((numpy.mean(rate_avg_list) - 0.286) / 2)
    mean_ste = numpy.mean(rate_ste_list)
    gamma_mean_ste = numpy.sqrt(mean_ste**2 + 0.244**2) / 2
    print(gamma_mean_ste)
    fit_std = numpy.std(rate_avg_list, ddof=1)
    gamma_fit_std = numpy.sqrt(fit_std**2 + 0.244**2) / 2
    print(gamma_fit_std)

    return mean_ste / fit_std


# %% Run the file


# The __name__ variable will only be '__main__' if you run this file directly.
# This allows a file's functions, classes, etc to be imported without running
# the script that you set up here.
if __name__ == '__main__':

    # Omega
    num_runs = 100
    num_samples = 1000
    num_steps = 26
    time_range = [0, 6]
    coeff = 0.30
    rate = 0.285*3

    # Gamma
    # num_runs = [20, 30]
    # num_steps = [51, 26]
    # time_range = [[0, 0.05], [0, 0.5]]
    # num_samples = 1000
    # coeff = 0.30
    # rate = 0.285+2*4.075

    # Run the script
    main(num_samples, num_runs, num_steps, time_range, coeff, rate)
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
