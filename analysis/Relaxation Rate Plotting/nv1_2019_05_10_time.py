# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 14:43:51 2019

Plotting time for the NV1_2019_05_10

@author: agardill
"""
import matplotlib.pyplot as plt
import utils.tool_belt as tool_belt

gamma_list=  [
    52.59543496298523,
    56.82144940136519,
    55.3126329163876,
    45.44249034570042,
    59.4835139228308,
    52.6799045438338,
    51.92665707887074,
    53.424913600568075,
    42.44866500386536
  ]

gamma_ste_list= [
    9.097236428097386,
    7.531331732369077,
    6.5618760509982605,
    6.244061098490613,
    6.647872869875284,
    7.542957405241219,
    7.483242654100141,
    7.115469961688277,
    5.158527405140316
  ]


def time_plot(file_name):
    '''
    Basic function to plot the data we collected on this NV. Data represented
    as points.
    '''
    data = tool_belt.get_raw_data('t1_double_quantum', file_name, 'nv1_2019_05_10_28MHz_4')
    
    gamma_list = data['gamma_list']
    gamma_ste_list = data['gamma_ste_list']
    
    time_inc = 1380 / len(gamma_list) # min
    time_list = []
    for i in range(len(gamma_list)):
        time = i*time_inc
        time_list.append(time)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
#    ax.errorbar(time_list, gamma_list, yerr = gamma_ste_list,
#                label = r'$\gamma$', fmt='o', markersize = 10,color='blue')
    ax.errorbar(time_list, gamma_list,
                label = r'$\gamma$', fmt='-o', markersize = 10,color='blue')

    ax.tick_params(which = 'both', length=6, width=2, colors='k',
                    grid_alpha=0.7, labelsize = 18)

    ax.tick_params(which = 'major', length=12, width=2)

    ax.grid()

    ax.set_xlabel('Time (minutes)', fontsize=18)
    ax.set_ylabel('Relaxation Rate (kHz)', fontsize=18)
    ax.set_title(r'NV1', fontsize=18)
    ax.legend(fontsize=18)
    fig.canvas.draw()
    fig.canvas.flush_events()
    
def histogram(file_name, bins):
    '''
    Produces a histogram of the data passed
    '''
    data = tool_belt.get_raw_data('t1_double_quantum', file_name, 'nv1_2019_05_10_28MHz_4')
    
    gamma_list = data['gamma_list']
    
    plt.hist(gamma_list, bins = bins)
    plt.xlabel('Gamma (kHz)')
    plt.ylabel('Occurances')
    
    
# %%
file_name = '26.3_MHz_splitting_30_bins_v2'
    
#time_plot(file_name)
histogram(file_name, 12)

    