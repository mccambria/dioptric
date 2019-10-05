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
import utils.tool_belt as tool_belt
from scipy.optimize import curve_fit

purple = '#87479b'

# %%

def gaussian(freq, constrast, sigma, center):
    return constrast * numpy.exp(-((freq-center)**2) / (2 * (sigma**2)))


def double_gaussian(x, amp_1, sigma_1, center_1,
                        amp_2, sigma_2, center_2):
    low_gauss = gaussian(x, amp_1, sigma_1, center_1)
    high_gauss = gaussian(x, amp_2, sigma_2, center_2)
    return low_gauss + high_gauss

# %%
# These rates were retaken with new analysis method (9/3)
#nv2_rates = [32.70770316064578, 32.35792873748028, 35.60840175962606, 29.076817363540858, 29.68546343026479, 32.22330577990145, 32.387345860666684, 27.780385383768593, 30.00007693747414, 33.09897897375375, 30.773285182687147, 27.97443572316569, 29.065815743119092, 29.36234609205914, 30.662448026051702, 32.01335672237963, 34.71088090216354, 33.84862495884125, 35.384631508491054, 35.77689947969005, 31.34895513001851, 34.057308895650294, 33.233369742643895, 34.19121868682558, 32.32305688825209, 33.35789514617837]
#nv2_rates_bi = [32.6, 32.6, 32.6, 28.2, 28.2, 32.6, 32.6, 28.2, 28.2, 32.6, 32.6,
#                 28.2, 28.2, 28.2, 32.6, 32.6, 32.6, 32.6, 32.6, 32.6, 32.6, 32.6,
#                 32.6, 32.6, 32.6, 32.6]
#nv2_error = [0.7369762116347649, 0.7562560021631601, 0.7625914980081805, 0.6574314474315747, 0.919359268283852, 0.731682094646155, 0.6947147322987086, 0.616051484886374, 0.6780672107044817, 0.8163823109181038, 0.6770579087023205, 0.6029035459656106, 0.6382539810559575, 0.6652562354500725, 0.6922682555827679, 0.7336262328354057, 0.8238800269399206, 0.7897512310371543, 0.7605036639707671, 0.8105764122736127, 0.7538842164929711, 0.8121492389404094, 0.75566606260391, 0.7553684059638512, 0.7000456778038091, 0.7725893647614507]

# Rates after new analysis
nv2_rates = [27.621071292343977, 28.00467103154353, 29.469965453079418, 25.10316827931753, 27.384306752689977, 26.950826695652587, 27.57847277904147, 25.813643197609753, 25.545459621923523, 27.73919964386004, 24.93512434869353, 25.341877755575116, 25.03634200036785, 25.81743330146043, 27.664173722079138, 26.182779332824943, 29.769336662684722, 29.004513033512666, 28.668977681142096, 30.34822157435222, 27.712413403059614, 27.39954530909307, 29.030026517648455, 28.11304948821951, 27.691511309720386, 26.623793229361375]
nv2_error = numpy.array([0.48665910284728936, 0.5136437039949685, 0.5229472135927203, 0.43261891664090063, 0.6756646634091575, 0.46467072686726413, 0.47143103043938, 0.423587832606777, 0.4446911497352184, 0.5366201978752652, 0.4412212934696276, 0.43465587142873024, 0.42048005502218033, 0.4462953842115663, 0.4822572573604373, 0.4403568078813716, 0.5657279159827925, 0.543961952271339, 0.49386476574863025, 0.5473851659364642, 0.5002106035777926, 0.5016894491027237, 0.5087576703572062, 0.4840832910081832, 0.4797090724802973, 0.47941746597351753])

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

   # %%
def time_main_plot():
    '''
    This function also plots the data we collected, however it represents the
    data as horizontal lines over the course of the experiment
    '''
    
    # convert the datetimes ito python time
    zero_time = mdates.date2num(datetime.datetime(2019,8,13,14,13,52))
    start_time_list = mdates.date2num(start_datetimes).tolist()
    end_time_list = mdates.date2num(end_datetimes).tolist()

    start_time_list_1 = []
    end_time_list_1 = []
    for i in range(len(nv2_rates)):
        start_time = start_time_list[i]-zero_time
        start_time_h = start_time * 24
        end_time = end_time_list[i]-zero_time
        end_time_h = end_time * 24
        
        start_time_list_1.append(start_time_h)
        end_time_list_1.append(end_time_h)



    # create the figure
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    for i in range(len(start_time_list_1)):
        ax.hlines(nv2_rates[i], start_time_list_1[i], end_time_list_1[i], linewidth=5, colors = purple)
#        ax.hlines(nv2_rates_bi[i], start_time[i], end_time[i], linewidth=5, colors = 'black')
        time_space = numpy.linspace(start_time_list_1[i], end_time_list_1[i], 10)
        ax.fill_between(time_space, nv2_rates[i] + nv2_error[i],
                        nv2_rates[i] - nv2_error[i],
                        color=purple, alpha=0.2)
    
    
    ax.spines['right'].set_visible(False)

#    for i in [0, 1,2,5,6,9,10, 14, 15,16,17,18,19,20,21,22,23,24,25]:
#        ax.hlines(nv2_rates[i], start_time_list_1[i], end_time_list_1[i], linewidth=5, colors = '#453fff')
##        ax.hlines(nv2_rates_bi[i], start_time[i], end_time[i], linewidth=5, colors = 'black')
#        time_space = numpy.linspace(start_time_list_1[i], end_time_list_1[i], 1000)
#        ax.fill_between(time_space, nv2_rates[i] + nv2_error[i],
#                        nv2_rates[i] - nv2_error[i],
#                        color='#453fff', alpha=0.2)
#        
#    for i in [3,4,7,8,11,12,13]:
#        ax.hlines(nv2_rates[i], start_time_list_1[i], end_time_list_1[i], linewidth=5, colors = '#c91600')
#        time_space = numpy.linspace(start_time_list_1[i], end_time_list_1[i], 1000)
#        ax.fill_between(time_space, nv2_rates[i] + nv2_error[i],
#                        nv2_rates[i] - nv2_error[i],
#                        color='#c91600', alpha=0.2)
        
    time_points = [start_time_list_1[0], end_time_list_1[2],  
                   start_time_list_1[3], end_time_list_1[4],
                   start_time_list_1[5], end_time_list_1[6],
                   start_time_list_1[7], end_time_list_1[8],
                   start_time_list_1[9], end_time_list_1[9],
                   start_time_list_1[10], end_time_list_1[14],
                   start_time_list_1[15], end_time_list_1[25]+10
                   ]
    values = [32.6, 32.6,
              28.2, 28.2,
              32.6, 32.6,
              28.2, 28.2,
              32.6, 32.6,
              28.2, 28.2,
              32.6, 32.6]
#    ax.plot(time_points, values, '--', color = 'gray')
    
    ax.tick_params(which = 'both', length=6, width=2, colors='k',
                    grid_alpha=0.7, labelsize = 18)

    ax.tick_params(which = 'major', length=12, width=2)

#    ax.grid(axis='y')


#    xfmt = mdates.DateFormatter('%m-%d-%y %H:%M')
#    ax.xaxis.set_major_formatter(xfmt)
    ax.set_ylim([23, 31.5])
    ax.set_xlim([-5, 145])
    plt.xlabel('Time (hours)', fontsize=18)
    plt.ylabel(r'Relaxation Rate, $\gamma$ (kHz)', fontsize=18)
#    plt.title(r'NV2, $\gamma$ rate', fontsize=18)
    
    fig.savefig("C:/Users/Aedan/Creative Cloud Files/Paper Illustrations/Magnetically Forbidden Rate/fig_4aLEFT.pdf", bbox_inches='tight')

def time_plot_inc():
    file29 = '29.5_MHz_splitting_5_bins_error'
    folder29 = 'nv2_2019_04_30_29MHz_29'
    data29 = tool_belt.get_raw_data('t1_double_quantum', file29, folder29)

    file30 = '29.8_MHz_splitting_2_bins_error'
    folder30 = 'nv2_2019_04_30_29MHz_30'
    data30 = tool_belt.get_raw_data('t1_double_quantum', file30, folder30)
    
    gamma_list = data29['gamma_list'] + data30['gamma_list']
    gamma_ste_list = data29['gamma_ste_list'] + data30['gamma_ste_list']
    gamma_ste_list = numpy.array(gamma_ste_list)
    time_inc = 5.5 # hr
    
    start_time_list_2 = []
    end_time_list_2 = []
    
    for i in range(len(gamma_list)):
        time = i*time_inc + 640
        start_time_list_2.append(time)
        
        time = i*time_inc + time_inc + 640
        end_time_list_2.append(time)
        
    fig, ax = plt.subplots(1, 1, figsize=(2, 8))
    
    for i in range(len(start_time_list_2)):
        ax.hlines(gamma_list[i], start_time_list_2[i], end_time_list_2[i], linewidth=5, colors = purple)
        time_space = numpy.linspace(start_time_list_2[i], end_time_list_2[i], 10)
        ax.fill_between(time_space, gamma_list[i] + gamma_ste_list[i],
                        gamma_list[i] - gamma_ste_list[i],
                        color=purple, alpha=0.2)
                        
#    for i in [3,4,5,6]:
#        ax.hlines(gamma_list[i], start_time_list_2[i], end_time_list_2[i], linewidth=5, colors = '#453fff')
#        time_space = numpy.linspace(start_time_list_2[i], end_time_list_2[i], 10)
#        ax.fill_between(time_space, gamma_list[i] + gamma_ste_list[i],
#                        gamma_list[i] - gamma_ste_list[i],
#                        color='#453fff', alpha=0.2)
#                        
#    for i in [0,1,2]:
#        ax.hlines(gamma_list[i], start_time_list_2[i], end_time_list_2[i], linewidth=5, colors = '#c91600')
#        time_space = numpy.linspace(start_time_list_2[i], end_time_list_2[i], 10)
#        ax.fill_between(time_space, gamma_list[i] + gamma_ste_list[i],
#                        gamma_list[i] - gamma_ste_list[i],
#                        color='#c91600', alpha=0.2)
    time_points = [start_time_list_2[0], end_time_list_2[2],  
                   start_time_list_2[3], end_time_list_2[6]
                   ]
    values = [28.2, 28.2,
              32.6, 32.6
              ]
#    ax.plot(time_points, values, '--', color = 'gray')
    
    
    ax.tick_params(which = 'both', length=6, width=2, colors='k',
                    grid_alpha=0.7, labelsize = 18)

    ax.tick_params(which = 'major', length=12, width=2)

#    ax.grid(axis='y')
    ax.set_ylim([23, 31.5])
    ax.spines['left'].set_visible(False)
    
#    ax.set_xlabel('Time (hour)', fontsize=18)
#    ax.set_ylabel('Relaxation Rate (kHz)', fontsize=18)
#    ax.set_title(r'NV2', fontsize=18)
#    ax.legend(fontsize=18)
    fig.canvas.draw()
    fig.canvas.flush_events()

    fig.savefig("C:/Users/Aedan/Creative Cloud Files/Paper Illustrations/Magnetically Forbidden Rate/fig_4aRIGHT.pdf", bbox_inches='tight')

def time_plot_zoom():
    file29 = '29.5_MHz_splitting_25_bins_error'
    folder29 = 'nv2_2019_04_30_29MHz_29'
    data29 = tool_belt.get_raw_data('t1_double_quantum', file29, folder29)

    file30 = '29.8_MHz_splitting_10_bins_error'
    folder30 = 'nv2_2019_04_30_29MHz_30'
    data30 = tool_belt.get_raw_data('t1_double_quantum', file30, folder30)
    
    gamma_list = data29['gamma_list'] + data30['gamma_list']
    gamma_ste_list = data29['gamma_ste_list'] + data30['gamma_ste_list']
    gamma_ste_list = numpy.array(gamma_ste_list)*2
    
    time_inc = 1.0 # hr
#    time_inc = 5.5 # hr
    
    start_time_list_2 = []
    end_time_list_2 = []
    
    for i in range(len(gamma_list)):
        time = i*time_inc + 640
        start_time_list_2.append(time)
        
        time = i*time_inc + time_inc + 640
        end_time_list_2.append(time)
        
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    for i in range(15, 35):
        ax.hlines(gamma_list[i], start_time_list_2[i], end_time_list_2[i], linewidth=5, colors = purple)
    #        ax.hlines(nv2_rates_bi[i], start_time[i], end_time[i], linewidth=5, colors = 'black')
        time_space = numpy.linspace(start_time_list_2[i], end_time_list_2[i], 10)
        ax.fill_between(time_space, gamma_list[i] + gamma_ste_list[i],
                        gamma_list[i] - gamma_ste_list[i],
                        color=purple, alpha=0.2)
    
#    for i in range(15, 31):
#        ax.hlines(gamma_list[i], start_time_list_2[i], end_time_list_2[i], linewidth=5, colors = '#453fff')
#        time_space = numpy.linspace(start_time_list_2[i], end_time_list_2[i], 10)
#        ax.fill_between(time_space, gamma_list[i] + gamma_ste_list[i],
#                        gamma_list[i] - gamma_ste_list[i],
#                        color='#453fff', alpha=0.2)
#                
#    index = 31
#    ax.hlines(gamma_list[index], start_time_list_2[index], end_time_list_2[index], linewidth=5, colors = '#c91600')
#    time_space = numpy.linspace(start_time_list_2[index], end_time_list_2[index], 10)
#    ax.fill_between(time_space, gamma_list[index] + gamma_ste_list[index],
#                    gamma_list[index] - gamma_ste_list[index],
#                    color='#c91600', alpha=0.2)
#                    
#    for i in range(32, 35):
#        ax.hlines(gamma_list[i], start_time_list_2[i], end_time_list_2[i], linewidth=5, colors = '#453fff')
#        time_space = numpy.linspace(start_time_list_2[i], end_time_list_2[i], 10)
#        ax.fill_between(time_space, gamma_list[i] + gamma_ste_list[i],
#                        gamma_list[i] - gamma_ste_list[i],
#                        color='#453fff', alpha=0.2)

    
    ax.tick_params(which = 'both', length=6, width=2, colors='k',
                    grid_alpha=0.7, labelsize = 18)

    ax.tick_params(which = 'major', length=12, width=2)

    ax.grid()
    ax.set_ylim([24, 36])

    ax.set_xlim([653, 677])
    plt.xlabel('Time (hours)', fontsize=18)
    plt.ylabel(r'Relaxation Rate, $\gamma$ (kHz)', fontsize=18)
    fig.canvas.draw()
    fig.canvas.flush_events()

#    fig.savefig("C:/Users/Aedan/Creative Cloud Files/Paper Illustrations/Magnetically Forbidden Rate/fig_4aZOOM.pdf", bbox_inches='tight')


def histogram(bins = 9, fit_gaussian = False):
    '''
    Produces a histogram of the data passed
    '''
    text = 62
    blue = '#2e3192'
    red = '#ed1c24'
    # The data from the econd take
    file29 = '29.5_MHz_splitting_5_bins_error'
    folder29 = 'nv2_2019_04_30_29MHz_29'
    data29 = tool_belt.get_raw_data('t1_double_quantum', file29, folder29)

    file30 = '29.8_MHz_splitting_2_bins_error'
    folder30 = 'nv2_2019_04_30_29MHz_30'
    data30 = tool_belt.get_raw_data('t1_double_quantum', file30, folder30)

    # Combine all the data
    
    gamma_list = data29['gamma_list'] + data30['gamma_list'] + nv2_rates


    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ret_vals= ax.hist(gamma_list, bins = bins, color = purple)
    ax.set_xlabel(r'$\gamma$ (kHz)', fontsize=text)
    ax.set_ylabel('Occurrences', fontsize=text)
    ax.tick_params(which = 'both', length=10, width=20, colors='k',
                    grid_alpha=1.2, labelsize = text)

    ax.tick_params(which = 'major', length=20, width=5)
    ax.set_yticks([0,2, 4,6])
    
        
    gssn_1_popt = [4, 0.8, 25.5]
    gssn_2_popt = [7,  0.8, 27.9]

    x_linspace = numpy.linspace(20, 35, 1000)
#    ax.plot(x_linspace, gaussian(x_linspace, *gssn_1_popt), '--',color = red,  lw = 6, label = 'fit')
#    ax.plot(x_linspace, gaussian(x_linspace, *gssn_2_popt), '--', color = blue,  lw = 6, label = 'fit')
    ax.set_xlim([23,31.2])

    # Fit double gaussian
    x_grid_endpoints = ret_vals[1]
    bin_width = (x_grid_endpoints[1] - x_grid_endpoints[0])/2
    x_grid = numpy.array(x_grid_endpoints) + bin_width
    hist_points = ret_vals[0]
    print(x_grid, hist_points)
    init_guesses = gssn_1_popt + gssn_2_popt
    popt, pcov = curve_fit(double_gaussian, hist_points, x_grid[:-1], p0= init_guesses)
    print(popt)
    
#    ax.plot(x_linspace, double_gaussian(x_linspace, *popt), 'k--', lw = 6, label = 'fit')

    fig.savefig("C:/Users/Aedan/Creative Cloud Files/Paper Illustrations/Magnetically Forbidden Rate/fig_4b.pdf", bbox_inches='tight')
    
def kde_sklearn(x, bandwidth=0.5):
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

#%%
if __name__ == "__main__":
    # The data from the econd take
    file29 = '29.5_MHz_splitting_5_bins_error'
    folder29 = 'nv2_2019_04_30_29MHz_29'
    data29 = tool_belt.get_raw_data('t1_double_quantum', file29, folder29)

    file30 = '29.8_MHz_splitting_2_bins_error'
    folder30 = 'nv2_2019_04_30_29MHz_30'
    data30 = tool_belt.get_raw_data('t1_double_quantum', file30, folder30)

    # Combine all the data
    
    gamma_list = data29['gamma_list'] + data30['gamma_list'] + nv2_rates

#    time_plot_inc()
#    time_main_plot()
#    time_plot_zoom()
    histogram(bins = 10)
    
#    init_guesses = [.16, 0.8, 25.5, .24, 0.8, 27.9]
#    x_linspace = numpy.linspace(20, 35, 1000)
#    y,x = kde_sklearn(gamma_list, bandwidth=0.8)
#    popt, pcov = curve_fit(double_gaussian, y, x, p0= init_guesses)
#    print(popt)
#    plt.plot(x_linspace, double_gaussian(x_linspace, *popt), 'k--', lw = 1, label = 'fit')
    
    


