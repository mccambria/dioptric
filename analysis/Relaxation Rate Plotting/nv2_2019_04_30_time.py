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

 #%%
   
def main():
    nv2_rates = [33.0, 32.3, 35.0, 28.9, 30, 33, 32.9, 28.9, 30.4, 34.8, 30.3,
                                         29, 29.1]
    nv2_error = [0.7, 0.9, 1.1, 1.0, 2, 1, 0.7, 0.7, 0.9, 1.3, 0.7, 1, 0.6]
    # The time of the end of the experiment
    list_of_datetimes = [datetime.datetime(2019,8,13,18,44,32),
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
                         datetime.datetime(2019,8,16,6,54,48)
                         ]
    
    time = mdates.date2num(list_of_datetimes)
#    time = ['13-08-2019 18:44']
    
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
    nv2_rates = [33.0, 32.3, 35.0, 28.9, 30, 33, 32.9, 28.9, 30.4, 34.8, 30.3,
                                         29, 29.1]
    nv2_error = [0.7, 0.9, 1.1, 1.0, 2, 1, 0.7, 0.7, 0.9, 1.3, 0.7, 1, 0.6]
    
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
                       datetime.datetime(2019,8,16,2,18,37)]
    
    # Taken when the experiment ends (last t1 timestamp)
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
                         datetime.datetime(2019,8,16,6,54,48)]
    
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
    
#%%
if __name__ == "__main__":
    
    main()
#    tmp()