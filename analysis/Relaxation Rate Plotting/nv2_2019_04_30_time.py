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

import numpy

# %%
nv2_rates = [33.0, 32.3, 35.0, 28.9, 30, 33, 32.9, 28.9, 30.4, 34.8, 30.3,
            29, 29.1, 30.5, 31.1, 33.9, 35.5, 34.5, 35.1, 36.6, 33.0, 33,
            33.3, 33.9, 32.1]
nv2_error = [0.7, 0.9, 1.1, 1.0, 2, 1, 0.7, 0.7, 0.9, 1.3, 0.7, 1, 0.6, 1.1,
             1.1, 1.6, 1.3, 1.2, 1.0, 1.0, 0.7, 2, 1.1, 1.0, 0.8]
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
                     datetime.datetime(2019,8,19,5,8,3)]
    
 #%%
   
def main():

    
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
def tmp():
    
    #Taken when the auto_pESR_and_rabi file saves (the timestamp)
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
#                       ,datetime.datetime(2019,8,19,5,33,27)
                       ]
    
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
    
def histogram():
#    numpy.histogram(nv2_rates, bins = 10)
    plt.hist(nv2_rates, bins = 6)
    plt.xlabel('Gamma (kHz)')
    plt.ylabel('Occurances')
    
    
def kde_sklearn(x, x_grid, bandwidth=0.2):
    from sklearn.neighbors import KernelDensity
    """Kernel Density Estimation with Scikit-learn"""
    kde_skl = KernelDensity(bandwidth=bandwidth)
    kde_skl.fit(x[:, numpy.newaxis])
    # score_samples() returns the log-likelihood of the samples
    log_pdf = kde_skl.score_samples(x_grid[:, numpy.newaxis])
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
    
        
def corelations():
    import pandas as pd
    # https://stackoverflow.com/questions/643699/how-can-i-use-numpy-correlate-to-do-autocorrelation/676302
    auto_corr = numpy.array(numpy.correlate(nv2_rates, nv2_rates, mode='full'))
    auto_corr_half = auto_corr[int(auto_corr.size/2):] / max(auto_corr)
    print(auto_corr_half )
    
#    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
#    ax.plot(auto_corr_half)
#    stat_cor=numpy.corrcoef(numpy.array([nv2_rates[:-1], nv2_rates[1:]]))
#    print(stat_cor)
    
    pd.plotting.autocorrelation_plot(nv2_rates)


#%%
if __name__ == "__main__":
    
#    main()
#    tmp()
#    histogram()
#    stacked_gaussian()
#    
#    # KDE method: https://jakevdp.github.io/blog/2013/12/01/kernel-density-estimation/
#    x_grid = numpy.linspace(min(nv2_rates), max(nv2_rates), 1000)
#    x = numpy.array(nv2_rates)
#        
#    pdf = kde_sklearn(x, x_grid, bandwidth=0.7)
#    fig,ax = plt.subplots(1,1)
#    ax.plot(x_grid, pdf, color='blue', alpha=0.5)
#    ax.set_xlabel('Gamma (kHz)')
#    ax.set_ylabel('Density')
#    ax.set_title('Kernal Density Estimation')
    
#     Correlation Plot
    corelations()