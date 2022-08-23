# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 21:43:30 2021

@author: samli
"""

import scipy.stats
import scipy.special
import math  
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
#import Data_green_photo_rate as data
import random
import photonstatistics as model
import utils.tool_belt as tool_belt

#%% import data 
# time in unit of ns
#time_list = data.test_pulse_list
#
#green_array7 = data.Data_4_29_green_array
#red_array7 = data.Data_4_29_red_array

def get_ave_count(array):  
    mean_list = []
    for i in range(0,len(time_list)):
        mean_list.append(np.mean(array[i]))
    return np.array(mean_list)

def get_ave_pop(array,n_th,time_list):
    pop_list = []
    for i in range(0,np.size(time_list)):
        counter = 0
        for j in range(0,np.size(array[i])):
             if array[i][j] >= n_th:
                 counter +=1 
        pop_list.append(counter/np.size(array[i]))
    return np.array(pop_list)

def fit_model(x_data,A,B):
    result = []
    B = 0.7
    for i in range(np.size(x_data)):
        result.append( B*(1 - np.exp(-A *x_data[i])))
    return result 

def quadratic_model(x_data,A):
    return A*(x_data)**2 

#plot and fit to the raw data of the green illumination measurement
def get_NVm_pop_plot(time_list,count_array,n_th):
    time_scale = np.array(time_list)
    t_data = np.linspace(0,max(time_list),1000)
    red_int = get_ave_pop(count_array,n_th,time_list)
    para,var = curve_fit(fit_model,time_scale,red_int,p0 = [0.0051,0.7],bounds = (0,1))
    A,B = para
    fit_curve = fit_model(t_data,A,B)
    fig,ax = plt.subplots()
    ax.plot(time_scale,red_int,'.')
    ax.plot(t_data,fit_curve)
    textstr = '\n'.join((
        r'$r_{tot}(10^6 s^{-1}) =%.2f$'% (A*10**3, ),
        r'$P_{steady state} =%.2f$'% (B, )))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.60, 0.40, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)  
    plt.xlabel('Green illumination time (ns)')
    plt.ylabel('NV- population')
    plt.title("Time resolved NV charge states transition under green")
    plt.show()
    
    return A*10**3
 
folder = 'pc_rabi/branch_Spin_to_charge/determine_photoionization_rates/2021_05'
file_07 = '2021_05_04-11_05_54-goeppert-mayer-nv5_2021_04_15' # 0.7 mW
file_09 = '2021_05_04-12_41_58-goeppert-mayer-nv5_2021_04_15' # 0.9 mW
file_19 = '2021_05_04-14_11_32-goeppert-mayer-nv5_2021_04_15' # 1.9 mW
file_0016 = '2021_05_04-15_46_27-goeppert-mayer-nv5_2021_04_15' # 0.016 mW
file_014 = '2021_05_04-19_41_37-goeppert-mayer-nv5_2021_04_15' # 0.14 mW
file_024 = '2021_05_05-09_08_19-goeppert-mayer-nv5_2021_04_15' # 0.24 mW
file_058 = '2021_05_05-10_39_20-goeppert-mayer-nv5_2021_04_15' # 0.58 mW
file_124 = '2021_05_05-12_10_30-goeppert-mayer-nv5_2021_04_15' # 1.24 mW

file_list = [file_07,file_09,  file_014, file_024, file_058, file_124]

rate_list = []
power_list = []
for file in file_list:
    data = tool_belt.get_raw_data(folder, file)   
    time_list = data['test_pulse_dur_list'] 
    red_array = data['red_count_raw']
    green_array = data['green_count_raw']
    power = data['test_power']
    rate = get_NVm_pop_plot(time_list,red_array,8)
    rate_list.append(rate)
    power_list.append(power)
power_list.append(0)
rate_list.append(0)

fig,ax = plt.subplots()
ax.plot(power_list,rate_list,'o')
popt, pcov = curve_fit(quadratic_model, power_list, rate_list, p0=[10])
print(pcov[0][0])
print(popt)
lin_power = np.linspace(0, 2, 100)

ax.plot(lin_power,quadratic_model(lin_power, *popt),'-')
ax.set_xlabel('Power (mW)')
ax.set_ylabel(r'$r_{tot}(10^6 s^{-1}$)')


    
    