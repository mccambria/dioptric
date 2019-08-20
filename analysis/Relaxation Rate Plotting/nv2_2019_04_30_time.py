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

# %%
nv2_rates = [33.0, 32.3, 35.0, 28.9, 30, 33, 32.9, 28.9, 30.4, 34.8, 30.3,
            29, 29.1, 30.5, 31.1, 33.9, 35.5, 34.5, 35.1, 36.6, 33.0, 33,
            33.3, 33.9, 32.1, 34.3]
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

def white_noise(mean, std, num_samples):
    '''
    Produces a list, num_samples long, of white noise around some mean, with 
    some standard deviation
    '''
    data_wn = numpy.random.normal(mean, std, size=num_samples)
    return data_wn
    
def white_telegraph_noise(mean, std, num_samples):
    '''
    Produces a list, num_samples long, of telegraphic noise in the specifc case 
    when each point has a 50% chance of changing value. This produces a two
    value white noise
    '''
    data_telegraph = []
    
    low_value = mean - std
    high_value = mean + std
    
    for i in range(num_samples):
        rand = numpy.random.randint(0, high=2)
        if rand == 0:
            data_telegraph.append(low_value)
        else:
            data_telegraph.append(high_value)
    
    return(data_telegraph)
    
def telegraph_noise(prob_of_flipping, mean, std, num_samples):
    '''
    Produces a list, num_samples long, of telegraphic noise. At each point, 
    there is a probablity (which is passed into the function) of switching to
    the other state
    '''
    data_telegraph = []    
    low_value = mean - std
    high_value = mean + std
    
    rand = numpy.random.randint(0, high=2)
    if rand == 0:
        current_value = low_value
    else:
        current_value = high_value
    data_telegraph.append(current_value)
    
    for i in range(num_samples - 1):
        rand = numpy.random.random()
        if rand <= prob_of_flipping:
            # The state changes
            if current_value == low_value:
                current_value = high_value
            elif current_value == high_value:
                current_value = low_value

        data_telegraph.append(current_value)
        
    return data_telegraph
    
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
    kde_skl.fit(x[:, numpy.newaxis])
    # score_samples() returns the log-likelihood of the samples
    x_grid = numpy.linspace(min(x), max(x), 1000)
    log_pdf = kde_skl.score_samples(x_grid[:, numpy.newaxis])
        
    pdf = kde_sklearn(x, x_grid, bandwidth=1.1)
    fig,ax = plt.subplots(1,1)
    ax.plot(x_grid, pdf, color='blue', alpha=0.5)
    ax.set_xlabel('Gamma (kHz)')
    ax.set_ylabel('Density')
    ax.set_title('Kernal Density Estimation')
    
    return numpy.exp(log_pdf)
    
def stacked_gaussian():
    def gaussian(t, i):
        return numpy.exp( (t - nv2_rates[i])**2 / (2 * nv2_error[i]**2))
    
    def summed_gaussian(t):
        eq = 0
        for i in range(len(nv2_rates)):
            eq += gaussian(t, i=i)
        print(eq)
        return eq
    
    linspace = numpy.linspace(min(nv2_rates), max(nv2_rates), 1000)
    plt.plot(linspace, summed_gaussian(linspace))
    
def correlation_1st_attempt():
    # not very successful
    auto_corr = numpy.array(numpy.correlate(nv2_rates, nv2_rates, mode='full'))
    auto_corr_half = auto_corr[int(auto_corr.size/2):] / max(auto_corr)
    print(auto_corr_half )
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.plot(auto_corr_half)
    
def pd_corelations():
    # The horizontal lines displayed in the plot correspond to 95% and 99% confidence bands. The dashed line is 99% confidence band.
    # https://stackoverflow.com/questions/643699/how-can-i-use-numpy-correlate-to-do-autocorrelation/676302    
    data = pd.Series(nv2_rates, index = end_datetimes)
    print(data)
    pd.plotting.autocorrelation_plot(data)
    
def autocorr(series):
    n = len(series)
    data = numpy.asarray(series)
    mean = numpy.mean(data)
    c0 = numpy.sum((data - mean) ** 2) / float(n)

    def r(h):
        acf_lag = ((data[:n - h] - mean) * (data[h:] - mean)).sum() / float(n) / c0
        return round(acf_lag, 3)
    x = numpy.arange(n) # Avoiding lag 0 calculation
    acf_coeffs = map(r, x)
    return acf_coeffs
    
def estimated_autocorrelation(x):
    """
    http://stackoverflow.com/q/14297012/190597
    http://en.wikipedia.org/wiki/Autocorrelation#Estimation
    """
    x = numpy.array(x)
    n = len(x)
    variance = x.var()
    x = x-x.mean()
    r = numpy.correlate(x, x, mode = 'full')[-n:]
#    assert numpy.allclose(r, numpy.array([(x[:n-k]*x[-(n-k):]).sum() for k in range(n)]))
    result = r/(variance*(numpy.arange(n, 0, -1)))
#    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
##    ax.plot(numpy.linspace(0, 5*25, 26), result)
#    ax.plot(result)
#    ax.set_xlabel('Lag')
#    ax.set_ylabel('Autocorrelation')
#    ax.axhline(y=0, color='k')
##    ax.set_xlim(0, 5*26)
##    ax.set_ylim(-1, 1)
#    ax.grid()
    
    return result

def statsmodel_acf(x):
    #un biased autocorrelation?
    # http://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.acf.html
    import statsmodels.tsa.stattools.acf as acfunction
    
    
def lag_plot():
    data = pd.Series(nv2_rates, index = end_datetimes)
#    print(data)
    pd.plotting.lag_plot(data, 1)
    
def run_multiple_times(num_samples, num_avg):
    acf_w_multi = []
    acf_t_multi = []
    acf_t_2_multi = []
    for i in range(num_avg):
        data_w_ind = white_noise(0, numpy.std(nv2_rates), num_samples)
        result_w = estimated_autocorrelation(data_w_ind)
        acf_w_multi.append(result_w)
        
        data_t_ind = telegraph_noise(numpy.average(nv2_rates), numpy.std(nv2_rates),num_samples)
        result_t = estimated_autocorrelation(data_t_ind)
        acf_t_multi.append(result_t)
        
        data_t_2_ind = telegraph_noise_2(.2, numpy.average(nv2_rates), numpy.std(nv2_rates),num_samples)
        result_t_2 = estimated_autocorrelation(data_t_2_ind)
        acf_t_2_multi.append(result_t_2)
    
    avg_acf_w = numpy.average(acf_w_multi, axis = 0)
    avg_acf_t = numpy.average(acf_t_multi, axis = 0)
    avg_acf_t_2 = numpy.average(acf_t_2_multi, axis = 0)
    
    return avg_acf_w, avg_acf_t, avg_acf_t_2
    
#%%
if __name__ == "__main__":
    
#    main()
#    tmp()
#    histogram()
#    stacked_gaussian()
#    
    #Data simulations
#    num_samples = 26
#    data_w = white_noise(0, numpy.std(nv2_rates), num_samples)
#    data_t = telegraph_noise(numpy.average(nv2_rates), numpy.std(nv2_rates),num_samples)
#    data_t_2 = telegraph_noise_2(.2, numpy.average(nv2_rates), numpy.std(nv2_rates),num_samples)

    # averaging
    results = run_multiple_times(26, 20)
    result_w = results[0]
    result_t = results[1]
    result_t_2 = results[2]

#    plt.plot(data_t)
#    plt.plot(data_t_2)

    
#     Correlation Plot
#    result_w = estimated_autocorrelation(data_w)
#    result_t = estimated_autocorrelation(data_t)
    real_data = estimated_autocorrelation(nv2_rates)
##    
    fig, ax = plt.subplots(1, 1, figsize=(17, 8))
    ax.plot(result_w, label = 'white noise')
    ax.plot(result_t, label = 'telegraph noise, p = 0.5')
    ax.plot(result_t_2, label = 'telegraph noise, p = 0.2')
    ax.plot(real_data, label = 'actual data')
    ax.set_xlabel('Lag')
    ax.set_ylabel('Autocorrelation')
    ax.axhline(y = 2/numpy.sqrt(26), color = 'gray')
    ax.axhline(y = -2/numpy.sqrt(26), color = 'gray')
    ax.axhline(y=0, color='k')
    ax.grid()
    ax.legend()

#    lag_plot()
    
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