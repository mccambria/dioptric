# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 10:06:12 2019

This file plots the data we've taken on NV2_2019_04_30 at the same splitting.

Main will plot a data point corresponding to the time that the experiment
finished.

Tmp is an experimnetal plotting, trying to plot the range of the experiment with
error bars.

@author: Aedan
"""
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime
import pandas as pd
import random
import numpy
from scipy.optimize import curve_fit
import utils.tool_belt as tool_belt

# %%
nv2_rates = [33.0, 32.3, 35.0, 28.9, 30, 33, 32.9, 28.9, 30.4, 34.8, 30.3,
            29, 29.1, 30.5, 31.1, 33.9, 35.5, 34.5, 35.1, 36.6, 33.0, 33,
            33.3, 33.9, 32.1, 34.3]
nv2_rates_bi = [33.5, 33.5, 33.5, 29.5, 29.5, 33.5, 33.5, 29.5, 29.5, 33.5, 29.5,
                 29.5, 29.5, 29.5, 29.5, 33.5, 33.5, 33.5, 33.5, 33.5, 33.5, 33.5,
                 33.5, 33.5, 33.5, 33.5]
nv2_error = [0.7, 0.9, 1.1, 1.0, 2, 1, 0.7, 0.7, 0.9, 1.3, 0.7, 1, 0.6, 1.1,
             1.1, 1.6, 1.3, 1.2, 1.0, 1.0, 0.7, 2, 1.1, 1.0, 0.8, 1.1]
# the time of the start of the experiment (when the pESSR and rabi data was saved

start_datetimes = [datetime.datetime(2019,8,13,14,13,52),
                       datetime.datetime(2019,8,13,19,8,32),
                       datetime.datetime(2019,8,14,0,4,19),
                       datetime.datetime(2019,8,14,4,59,48),
                       datetime.datetime(2019,8,14,10,12,28),
                       datetime.datetime(2019,8,14,15,7,55),
                       datetime.datetime(2019,8,14,20,3,21),
                       datetime.datetime(2019,8,15,0,59,7),
                       datetime.datetime(2019,8,15,6,0,1),
                       datetime.datetime(2019,8,15,11,14,42),
                       datetime.datetime(2019,8,15,16,15,58),
                       datetime.datetime(2019,8,15,21,17,3),
                       datetime.datetime(2019,8,16,2,18,37),
                       datetime.datetime(2019,8,16,7,20,15),
                       datetime.datetime(2019,8,16,14,53,13),
                       datetime.datetime(2019,8,16,19,54,43),
                       datetime.datetime(2019,8,17,0,56,17),
                       datetime.datetime(2019,8,17,5,57,36),
                       datetime.datetime(2019,8,17,10,58,47),
                       datetime.datetime(2019,8,17,16,0,11),
                       datetime.datetime(2019,8,17,21,1,38),
                       datetime.datetime(2019,8,18,9,28,58),
                       datetime.datetime(2019,8,18,14,29,51),
                       datetime.datetime(2019,8,18,19,31,3),
                       datetime.datetime(2019,8,19,0,32,6)
                       ,datetime.datetime(2019,8,19,5,33,27)
                       ]
# The time of the end of the experiment
end_datetimes = [datetime.datetime(2019,8,13,18,44,32),
                     datetime.datetime(2019,8,13,23,39,48),
                     datetime.datetime(2019,8,14,4,35,43),
                     datetime.datetime(2019,8,14,9,31,10),
                     datetime.datetime(2019,8,14,14,43,33),
                     datetime.datetime(2019,8,14,19,38,53),
                     datetime.datetime(2019,8,15,0,34,47),
                     datetime.datetime(2019,8,15,5,34,56),
                     datetime.datetime(2019,8,15,10,35,42),
                     datetime.datetime(2019,8,15,15,50,32),
                     datetime.datetime(2019,8,15,20,51,51),
                     datetime.datetime(2019,8,16,1,53,13),
                     datetime.datetime(2019,8,16,6,54,48),
                     datetime.datetime(2019,8,16,11,56,7),
                     datetime.datetime(2019,8,16,19,29,27),
                     datetime.datetime(2019,8,17,0,30,46),
                     datetime.datetime(2019,8,17,5,32,16),
                     datetime.datetime(2019,8,17,10,33,30),
                     datetime.datetime(2019,8,17,15,34,51),
                     datetime.datetime(2019,8,17,20,36,19),
                     datetime.datetime(2019,8,18,1,38,1),
                     datetime.datetime(2019,8,18,14,4,38),
                     datetime.datetime(2019,8,18,19,5,39),
                     datetime.datetime(2019,8,19,0,6,50),
                     datetime.datetime(2019,8,19,5,8,3),
                     datetime.datetime(2019,8,19,10,9,37)]

#%%

def gaussian(freq, constrast, sigma, center):
    return constrast * numpy.exp(-((freq-center)**2) / (2 * (sigma**2)))


def double_gaussian(x, amp_1, sigma_1, center_1,
                        amp_2, sigma_2, center_2):
    low_gauss = gaussian(x, amp_1, sigma_1, center_1)
    high_gauss = gaussian(x, amp_2, sigma_2, center_2)
    return low_gauss + high_gauss

def exp(t, lifetime, amp):
    return amp * numpy.exp(-2*t/lifetime)
#%%

def white_noise(mean, std, num_samples, num_avg):
    '''
    Produces a list, num_samples long, of white noise around some mean, with
    some standard deviation
    '''
    data_wn_2D = []
    for i in range(num_avg):
        data_wn = numpy.random.normal(mean, std, size=num_samples)
        data_wn_2D.append(data_wn)
    return data_wn_2D

def white_telegraph_noise(mean, std, num_samples, num_avg):
    '''
    Produces a list, num_samples long, of telegraphic noise in the specifc case
    when each point has a 50% chance of changing value. This produces a two
    value white noise
    '''
    data_wt_2D = []
    low_value = mean - std
    high_value = mean + std

    for avg in range(num_avg):
        data_wt_single = []

        for i in range(num_samples):
            rand = numpy.random.randint(0, high=2)
            if rand == 0:
                data_wt_2D.append(low_value)
            else:
                data_wt_2D.append(high_value)
        data_wt_2D.append(data_wt_single)

    return data_wt_2D

def telegraph_noise(prob_of_flipping, mean, std, num_samples, num_avg):
    '''
    Produces a list, num_samples long, of telegraphic noise. At each point,
    there is a probablity (which is passed into the function) of switching to
    the other state
    http://mathworld.wolfram.com/ExponentialDistribution.html
    https://math.stackexchange.com/questions/1875135/how-to-calculate-rate-parameter-in-exponential-distribution
    '''
    data_t_2D = []
    low_value = mean - std
    high_value = mean + std
    for avg in range(num_avg):

        data_t = []

        rand = numpy.random.randint(0, high=2)
        if rand == 0:
            current_value = low_value
        else:
            current_value = high_value
        data_t.append(current_value)

        for i in range(num_samples - 1):
            rand = numpy.random.random()
            if rand <= prob_of_flipping:
                # The state changes
                if current_value == low_value:
                    current_value = high_value
                elif current_value == high_value:
                    current_value = low_value

            data_t.append(current_value)
        data_t_2D.append(data_t)
    return data_t_2D

def telegraph_noise_rate(rate, mean, std, num_samples, num_avg):
    '''
    Produces a list, num_samples long, of telegraphic noise. The time spent in
    each state is set by an exponential distribution of a given rate.

    http://mathworld.wolfram.com/ExponentialDistribution.html
    https://math.stackexchange.com/questions/1875135/how-to-calculate-rate-parameter-in-exponential-distribution
    '''
    data_tr_2D = []
    low_value = mean - std
    high_value = mean + std

    for avg in range(num_avg):
        # start randomly in one of the two states
        rand = numpy.random.randint(0, high=2)
        if rand == 0:
            current_value = low_value
        else:
            current_value = high_value

        time_intervals = []
        while sum(time_intervals) <= num_samples:
#            random_exp_distribution = numpy.log(1- random.random()) / (-rate)
            random_exp_distribution = numpy.random.exponential(1/rate)
            random_exp_distribution_int = round(random_exp_distribution)
            # If rounding causes the time to be 0, then set the value to 1
            if random_exp_distribution_int == 0:
                random_exp_distribution_int = 1
            time_intervals.append(random_exp_distribution_int)

        # change the last value so that it fits within the num_samples
        diff = sum(time_intervals) - num_samples
        time_intervals[-1] = time_intervals[-1] - diff

        data_tr = numpy.empty([num_samples])
        data_tr[:] = numpy.nan
        i = 0
        for t in range(len(time_intervals)):
            data_tr[i:i+time_intervals[t]] = current_value
            if current_value == low_value:
                next_value = high_value
            elif current_value == high_value:
                next_value = low_value
            i = i+time_intervals[t]
            current_value = next_value

        plt.plot(data_tr)
        data_tr_2D.append(data_tr)

    return(data_tr_2D)


# %%

def time_plot():
    '''
    Basic function to plot the data we collected on this NV. Data represented
    as points.
    '''
    time = mdates.date2num(end_datetimes)

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    fig.autofmt_xdate()
    ax.xaxis_date()
    ax.errorbar(time, nv2_rates, yerr = nv2_error,
                label = r'$\gamma$', fmt='o', markersize = 10,color='blue')

    ax.tick_params(which = 'both', length=6, width=2, colors='k',
                    grid_alpha=0.7, labelsize = 18)

    ax.tick_params(which = 'major', length=12, width=2)

    ax.grid()

    xfmt = mdates.DateFormatter('%m-%d-%y %H:%M')
    ax.xaxis.set_major_formatter(xfmt)

    plt.xlabel('Date (mm-dd-yy hh:mm)', fontsize=18)
    plt.ylabel('Relaxation Rate (kHz)', fontsize=18)
    plt.title(r'NV2', fontsize=18)
    ax.legend(fontsize=18)
   # %%
def time_plot_formal():
    '''
    This function also plots the data we collected, however it represents the
    data as horizontal lines over the course of the experiment
    '''
    # convert the datetimes ito python time
    start_time = mdates.date2num(start_datetimes).tolist()
    end_time = mdates.date2num(end_datetimes).tolist()

    # create the figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    fig.autofmt_xdate()
    ax.xaxis_date()

    # for each data "line", plot the hline and error
    for i in range(len(nv2_rates)):
        ax.hlines(nv2_rates[i], start_time[i], end_time[i], linewidth=5, colors = 'blue')
        ax.hlines(nv2_rates_bi[i], start_time[i], end_time[i], linewidth=5, colors = 'black')
        time_space = numpy.linspace(start_time[i], end_time[i], 1000)
        ax.fill_between(time_space, nv2_rates[i] + nv2_error[i],
                        nv2_rates[i] - nv2_error[i],
                        color='blue', alpha=0.2)

    ax.tick_params(which = 'both', length=6, width=2, colors='k',
                    grid_alpha=0.7, labelsize = 18)

    ax.tick_params(which = 'major', length=12, width=2)

    ax.grid()

    xfmt = mdates.DateFormatter('%m-%d-%y %H:%M')
    ax.xaxis.set_major_formatter(xfmt)

    plt.xlabel('Date (mm-dd-yy hh:mm)', fontsize=18)
    plt.ylabel('Relaxation Rate (kHz)', fontsize=18)
    plt.title(r'NV2, $\gamma$ rate', fontsize=18)
#    ax.legend(fontsize=18)

def histogram(x, bins):
    '''
    Produces a histogram of the data passed
    '''
#    numpy.histogram(nv2_rates, bins = 10)
    plt.hist(x, bins = bins)
    plt.xlabel('Gamma (kHz)')
    plt.ylabel('Occurances')


def kde_sklearn(x, bandwidth=0.2):
    '''
    Produces a kernel density estimation of the data passed. It also plots it.
    https://jakevdp.github.io/blog/2013/12/01/kernel-density-estimation/
    '''
    from sklearn.neighbors import KernelDensity
    """Kernel Density Estimation with Scikit-learn"""

    kde_skl = KernelDensity(bandwidth=bandwidth)
    x = numpy.array(x)
    kde_skl.fit(x[:, numpy.newaxis])
    # score_samples() returns the log-likelihood of the samples
    x_grid = numpy.linspace(min(x), max(x), 1000)
    log_pdf = kde_skl.score_samples(x_grid[:, numpy.newaxis])

    pdf = numpy.exp(log_pdf)
    fig,ax = plt.subplots(1,1)
    ax.plot(x_grid, pdf, color='blue', alpha=0.5)
    ax.set_xlabel('Gamma (kHz)')
    ax.set_ylabel('Density')
    ax.set_title('Kernal Density Estimation')

#    print(numpy.exp(log_pdf))
    return numpy.exp(log_pdf), x_grid

def estimated_autocorrelation(array, do_plot = False):
    """
    Calculates the autocorrelation for discrete points.

    x must be a 2D list. The points are averaged along the 0 axis

    Documentation:
    http://stackoverflow.com/q/14297012/190597
    http://en.wikipedia.org/wiki/Autocorrelation#Estimation
    https://dsp.stackexchange.com/questions/16596/autocorrelation-of-a-telegraph-process-constant-signal?newreg=4516667dd4964cab94c278bde3b417ea
    """
    result_list = []
    print(array)
    for row in array:
        x = numpy.array(row)
        n = len(x)
        variance = numpy.var(x)
        x = x-x.mean()
        r = numpy.correlate(x, x, mode = 'full')[-n:]
    #    assert numpy.allclose(r, numpy.array([(x[:n-k]*x[-(n-k):]).sum() for k in range(n)]))
        row_result = r/(variance*(numpy.arange(n, 0, -1)))
        result_list.append(row_result)

    avg_result = numpy.average(result_list, axis = 0)

    if do_plot:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    #    ax.plot(numpy.linspace(0, 5*25, 26), result)
        ax.plot(avg_result, 'r', label = 'kde')
        ax.set_xlabel('Lag')
        ax.set_ylabel('Autocorrelation')
        ax.axhline(y=0, color='k')
        ax.grid()

    return avg_result

def lag_plot(x):
    '''
    Creates a lag plot
    '''
    data = pd.Series(x)
#    print(data)
    pd.plotting.lag_plot(data, 1)

#%%
if __name__ == "__main__":
    time_plot_formal()

#     KDE Estimating Two Values
#    kde_points, x_grid = kde_sklearn(nv2_rates, bandwidth=1.1)
#
#    init_guess = [0.1, 1, 29, 0.1, 1, 34]
#
#    dbl_gssn_popt, pcov = curve_fit(double_gaussian, x_grid, kde_points, p0 = init_guess)
#
#    plt.plot(x_grid, double_gaussian(x_grid, *dbl_gssn_popt), 'b--', label = 'fit')
#    plt.legend()
#
#    print(dbl_gssn_popt)

    #Data simulations
    num_samples = 26
    num_avg = 1
    telegraph_prob = 0.30
    rate = 1/3.5

#    telegraph_noise_rate(rate, numpy.average(nv2_rates), numpy.std(nv2_rates),num_samples, num_avg)
#     averaging

    # simulate Data
#    data_wn = white_noise(0, numpy.std(nv2_rates), num_samples, num_avg)
#    data_wt = white_telegraph_noise(numpy.average(nv2_rates), numpy.std(nv2_rates),num_samples, num_avg)
#    data_t = telegraph_noise(telegraph_prob, numpy.average(nv2_rates), numpy.std(nv2_rates),num_samples, num_avg)
#    data_tr = telegraph_noise_rate(rate, numpy.average(nv2_rates), numpy.std(nv2_rates),num_samples, num_avg)    

    # Correlation Plot
#    telegraph_noise_rate(rate, 0, 1, num_samples, num_avg)

#    result_wn = estimated_autocorrelation(data_wn)
#    result_wt = estimated_autocorrelation(data_wt)
#    result_t = estimated_autocorrelation(data_t)
#    result_tr = estimated_autocorrelation(data_tr)
#    real_data = estimated_autocorrelation([nv2_rates])

    
    # Plot    
#    linspace = numpy.linspace(0,25, 26)
#    init_params = (0, 1, 1/25, 0.1) # sinexp
#    init_params = (1, 1) # sinexp
#    popt, pcov = curve_fit(exp, linspace, result_tr, p0 = init_params)
#    lifetime = 2 / (popt[2]**2 * 5)
#    print(popt[0])
#    fig, ax = plt.subplots(1, 1, figsize=(17, 8))
#    ax.plot(result_wn, label = 'white noise')
##    ax.plot(result_wt, label = 'telegraph noise, p = 0.5')
##    ax.plot(result_t, label = 'telegraph noise, p = {}'.format(telegraph_prob))
#    ax.plot(real_data, label = 'actual data')
#    ax.plot(result_tr, label = 'telegraph simulation')
#    ax.plot(linspace, exp(linspace, *popt), label = 'fit')
#    ax.set_xlabel('Lag')
#    ax.set_ylabel('Autocorrelation')
##    ax.axhline(y = 2/numpy.sqrt(26), color = 'gray')
##    ax.axhline(y = -2/numpy.sqrt(26), color = 'gray')
#    ax.axhline(y=0, color='k')
#    ax.grid()
#    ax.legend()
    
    # Plot multiple telegraph noise functions
#    fig, ax = plt.subplots(1, 1, figsize=(17, 8))
#    for i in [ 0.1, 0.2, 0.25, 0.3]:
#        prob = i * 100
#        data_t_2 = telegraph_noise_2(i, numpy.average(nv2_rates), numpy.std(nv2_rates),num_samples)
#        result_t_2 = estimated_autocorrelation(data_t_2)
#        ax.plot(result_t_2, label = '{}% prob of flipping states'.format(prob))
#    ax.set_xlabel('Lag')
#    ax.set_ylabel('Autocorrelation')
#    ax.axhline(y=0, color='k')
#    ax.grid()
#    ax.legend()
