# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 14:43:51 2019

Plotting time for the NV1_2019_05_10

@author: agardill
"""
import matplotlib.pyplot as plt
import utils.tool_belt as tool_belt
import numpy
from scipy.optimize import curve_fit

purple = '#87479b'

# %%

def gaussian(freq, constrast, sigma, center):
    return constrast * numpy.exp(-((freq-center)**2) / (2 * (sigma**2)))


def triple_gaussian(x, amp_1, sigma_1, center_1,
                        amp_2, sigma_2, center_2,
                        amp_3, sigma_3, center_3):
    low_gauss = gaussian(x, amp_1, sigma_1, center_1)
    mid_gauss = gaussian(x, amp_2, sigma_2, center_2)
    high_gauss = gaussian(x, amp_3, sigma_3, center_3)
    return low_gauss + mid_gauss + high_gauss

# %%

# 45 min incr
#file4 = '26.3_MHz_splitting_30_bins_error'
#folder4 = 'nv1_2019_05_10_28MHz_4'
#data4 = tool_belt.get_raw_data('t1_double_quantum', file4, folder4)
#
#file5 = '26.5_MHz_splitting_25_bins_error'
#folder5 = 'nv1_2019_05_10_28MHz_5'
#data5 = tool_belt.get_raw_data('t1_double_quantum', file5, folder5)
#
#file6 = '26.2_MHz_splitting_25_bins_error'
#folder6 = 'nv1_2019_05_10_28MHz_6'
#data6 = tool_belt.get_raw_data('t1_double_quantum', file6, folder6)

# 1.25 hour incr
#file4 = '26.3_MHz_splitting_18_bins_error'
#folder4 = 'nv1_2019_05_10_28MHz_4'
#data4 = tool_belt.get_raw_data('t1_double_quantum', file4, folder4)
#
#file5 = '26.5_MHz_splitting_15_bins_error'
#folder5 = 'nv1_2019_05_10_28MHz_5'
#data5 = tool_belt.get_raw_data('t1_double_quantum', file5, folder5)
#
#file6 = '26.2_MHz_splitting_15_bins_error'
#folder6 = 'nv1_2019_05_10_28MHz_6'
#data6 = tool_belt.get_raw_data('t1_double_quantum', file6, folder6)


# 3.75 hour incr
#file4 = '26.3_MHz_splitting_6_bins_error'
#folder4 = 'nv1_2019_05_10_28MHz_4'
#data4 = tool_belt.get_raw_data('t1_double_quantum', file4, folder4)
#
#file5 = '26.5_MHz_splitting_5_bins_error'
#folder5 = 'nv1_2019_05_10_28MHz_5'
#data5 = tool_belt.get_raw_data('t1_double_quantum', file5, folder5)
#
#file6 = '26.2_MHz_splitting_5_bins_error'
#folder6 = 'nv1_2019_05_10_28MHz_6'
#data6 = tool_belt.get_raw_data('t1_double_quantum', file6, folder6)

#gamma_list = data4['gamma_list'] + data5['gamma_list'] + data6['gamma_list'] 
#gamma_ste_list = data4['gamma_ste_list'] + data5['gamma_ste_list'] \
#                + data6['gamma_ste_list']
                    
# %%

def time_main_plot(folder_name, file_name):
    '''
    Basic function to plot the data we collected on this NV. Data represented
    as points.
    '''
    font_size = 36
    # 3.75 hour incr
    file4 = '26.3_MHz_splitting_6_bins_error'
    folder4 = 'nv1_2019_05_10_28MHz_4'
    data4 = tool_belt.get_raw_data('t1_double_quantum', file4, folder4)
    
    file5 = '26.5_MHz_splitting_5_bins_error'
    folder5 = 'nv1_2019_05_10_28MHz_5'
    data5 = tool_belt.get_raw_data('t1_double_quantum', file5, folder5)
    
    file6 = '26.2_MHz_splitting_5_bins_error'
    folder6 = 'nv1_2019_05_10_28MHz_6'
    data6 = tool_belt.get_raw_data('t1_double_quantum', file6, folder6)
    
    gamma_list = data4['gamma_list'] + data5['gamma_list'] + data6['gamma_list'] 
    gamma_ste_list = data4['gamma_ste_list'] + data5['gamma_ste_list'] \
                    + data6['gamma_ste_list']
    gamma_ste_list = numpy.array(gamma_ste_list)
    
    time_inc = 3.75 # hr
    
    time_start_list_4 = []
    for i in range(len(data4['gamma_list'] )):
        time = i*time_inc
        time_start_list_4.append(time)
    time_end_list_4 = []
    for i in range(len(data4['gamma_list']) ):
        time = i*time_inc+ time_inc
        time_end_list_4.append(time)
        
    time_start_list_5 = []
    for i in range(len(data5['gamma_list'] )):
        time = i*time_inc + 3 + time_end_list_4[-1]
        time_start_list_5.append(time)
    time_end_list_5 = []
    for i in range(len(data5['gamma_list'] )):
        time = i*time_inc+ time_inc + 3 + time_end_list_4[-1]
        time_end_list_5.append(time)
        
    time_start_list_6 = []
    for i in range(len(data6['gamma_list'] )):
        time = i*time_inc + 1 + time_end_list_5[-1]
        time_start_list_6.append(time)
    time_end_list_6 = []
    for i in range(len(data6['gamma_list'] )):
        time = i*time_inc+ time_inc + 1 + time_end_list_5[-1]
        time_end_list_6.append(time)
        
    time_start_list = time_start_list_4 + time_start_list_5 + time_start_list_6
    time_end_list = time_end_list_4 + time_end_list_5 + time_end_list_6
        
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
#    for i in range(0,15):
    for i in range(len(time_start_list)):
        ax.hlines(gamma_list[i], time_start_list[i], time_end_list[i], linewidth=5, colors = purple)
#        ax.hlines(nv2_rates_bi[i], start_time[i], end_time[i], linewidth=5, colors = 'black')
        time_space = numpy.linspace(time_start_list[i], time_end_list[i], 1000)
        ax.fill_between(time_space, gamma_list[i] + gamma_ste_list[i],
                        gamma_list[i] - gamma_ste_list[i],
                        color=purple, alpha=0.2)

#    for i in [0,1,2,3,4,5, 8, 9, 10, 11,12,13,14,15]:
#        ax.hlines(gamma_list[i], time_start_list[i], time_end_list[i], linewidth=5, colors = '#453fff')
##        ax.hlines(nv2_rates_bi[i], start_time[i], end_time[i], linewidth=5, colors = 'black')
#        time_space = numpy.linspace(time_start_list[i], time_end_list[i], 1000)
#        ax.fill_between(time_space, gamma_list[i] + gamma_ste_list[i],
#                        gamma_list[i] - gamma_ste_list[i],
#                        color='#453fff', alpha=0.2)
#                        
#    for i in [6,7]:
#        ax.hlines(gamma_list[i], time_start_list[i], time_end_list[i], linewidth=5, colors = '#c91600')
##        ax.hlines(nv2_rates_bi[i], start_time[i], end_time[i], linewidth=5, colors = 'black')
#        time_space = numpy.linspace(time_start_list[i], time_end_list[i], 1000)
#        ax.fill_between(time_space, gamma_list[i] + gamma_ste_list[i],
#                        gamma_list[i] - gamma_ste_list[i],
#                        color='#c91600', alpha=0.2)
                        

    ax.tick_params(which = 'both', length=6, width=2, colors='k',
                    direction='in',grid_alpha=0.7, labelsize = font_size)

    ax.tick_params(which = 'major', length=12, width=2)


    ax.set_xlabel('Time (hours)', fontsize=font_size)
    ax.set_ylabel(r'Relaxation Rate, $\gamma$ (kHz)', fontsize=font_size)
    ax.set_ylim(39.5,59.5)
#    ax.set_title(r'NV1', fontsize=18)
#    ax.legend(fontsize=18)
    fig.canvas.draw()
    fig.canvas.flush_events()
    
    fig.savefig("C:/Users/Aedan/Creative Cloud Files/Paper Illustrations/Magnetically Forbidden Rate/fig_4c.pdf", bbox_inches='tight')

def time_zoom_plot(folder_name, file_name):
    '''
    Basic function to plot the data we collected on this NV. Data represented
    as points.
    '''
    file4 = '26.3_MHz_splitting_18_bins_error'
    folder4 = 'nv1_2019_05_10_28MHz_4'
    data4 = tool_belt.get_raw_data('t1_double_quantum', file4, folder4)
    
    file5 = '26.5_MHz_splitting_15_bins_error'
    folder5 = 'nv1_2019_05_10_28MHz_5'
    data5 = tool_belt.get_raw_data('t1_double_quantum', file5, folder5)
    
    file6 = '26.2_MHz_splitting_15_bins_error'
    folder6 = 'nv1_2019_05_10_28MHz_6'
    data6 = tool_belt.get_raw_data('t1_double_quantum', file6, folder6)
#    
    gamma_list = data4['gamma_list'] + data5['gamma_list'] + data6['gamma_list'] 
    gamma_ste_list = data4['gamma_ste_list'] + data5['gamma_ste_list'] \
                    + data6['gamma_ste_list']
    gamma_ste_list = numpy.array(gamma_ste_list)
    
    time_inc = 1.25 # hr
    
    time_start_list_4 = []
    for i in range(len(data4['gamma_list'] )):
        time = i*time_inc
        time_start_list_4.append(time)
    time_end_list_4 = []
    for i in range(len(data4['gamma_list']) ):
        time = i*time_inc+ time_inc
        time_end_list_4.append(time)
        
    time_start_list_5 = []
    for i in range(len(data5['gamma_list'] )):
        time = i*time_inc + 3 + time_end_list_4[-1]
        time_start_list_5.append(time)
    time_end_list_5 = []
    for i in range(len(data5['gamma_list'] )):
        time = i*time_inc+ time_inc + 3 + time_end_list_4[-1]
        time_end_list_5.append(time)
        
    time_start_list_6 = []
    for i in range(len(data6['gamma_list'] )):
        time = i*time_inc + 1 + time_end_list_5[-1]
        time_start_list_6.append(time)
    time_end_list_6 = []
    for i in range(len(data6['gamma_list'] )):
        time = i*time_inc+ time_inc + 1 + time_end_list_5[-1]
        time_end_list_6.append(time)
        
    time_start_list = time_start_list_4 + time_start_list_5 + time_start_list_6
    time_end_list = time_end_list_4 + time_end_list_5 + time_end_list_6
        
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
#    for i in range(0,15):
    for i in range(len(gamma_list)):
        ax.hlines(gamma_list[i], time_start_list[i], time_end_list[i], linewidth=5, colors = purple)
        time_space = numpy.linspace(time_start_list[i], time_end_list[i], 1000)
        ax.fill_between(time_space, gamma_list[i] + gamma_ste_list[i],
                        gamma_list[i] - gamma_ste_list[i],
                        color=purple, alpha=0.2)

    ax.tick_params(which = 'both', length=6, width=2, colors='k',
                    direction='in',grid_alpha=0.7, labelsize = 18)

    ax.tick_params(which = 'major', length=12, width=2)


    ax.set_xlabel('Time (hours)', fontsize=18)
    ax.set_ylabel(r'Relaxation Rate, $\gamma$ (kHz)', fontsize=18)
    ax.set_ylim(35,68)
#    ax.set_title(r'NV1', fontsize=18)
#    ax.legend(fontsize=18)
    fig.canvas.draw()
    fig.canvas.flush_events()
    
    fig.savefig("C:/Users/Aedan/Creative Cloud Files/Paper Illustrations/Magnetically Forbidden Rate/fig_4c2.pdf", bbox_inches='tight')

    
def histogram(bins = 7, fit_gaussian = False):
    '''
    Produces a histogram of the data passed
    '''
    
    light_purple = '#CE7FFF'
    # 3.75 hour incr
    file4 = '26.3_MHz_splitting_6_bins_error'
    folder4 = 'nv1_2019_05_10_28MHz_4'
    data4 = tool_belt.get_raw_data('t1_double_quantum', file4, folder4)
    
    file5 = '26.5_MHz_splitting_5_bins_error'
    folder5 = 'nv1_2019_05_10_28MHz_5'
    data5 = tool_belt.get_raw_data('t1_double_quantum', file5, folder5)
    
    file6 = '26.2_MHz_splitting_5_bins_error'
    folder6 = 'nv1_2019_05_10_28MHz_6'
    data6 = tool_belt.get_raw_data('t1_double_quantum', file6, folder6)
    
    gamma_list = data4['gamma_list'] + data5['gamma_list'] + data6['gamma_list']
    gamma_ste_list = data4['gamma_ste_list'] + data5['gamma_ste_list'] \
                    + data6['gamma_ste_list']
                    
        

    text = 115
#    numpy.histogram(nv2_rates, bins = 10)
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ret_vals= ax.hist(gamma_list, bins = bins, color = light_purple)
    ax.set_xlabel(r'$\gamma$ (kHz)', fontsize=text)
    ax.set_ylabel('Occur.', fontsize=text)
    ax.tick_params(which = 'both', length=10, width=20, colors='k',
                    direction='in',grid_alpha=1.2, labelsize = text)
    ax.set_yticks([0,2, 4])
    ax.tick_params(which = 'major', length=25, width=10)
  
    gaus_params = [4, numpy.average(gamma_ste_list), 49.3]
    
    x_linspace = numpy.linspace(40, 58, 1000)
    ax.plot(x_linspace, gaussian(x_linspace, *gaus_params), '--',color = 'k',  lw = 10, label = 'fit')
    ax.set_xlim([42.5,56.5])

    fig.savefig("C:/Users/Aedan/Creative Cloud Files/Paper Illustrations/Magnetically Forbidden Rate/fig_4d.pdf", bbox_inches='tight')

# %%
file_name = '26.2_MHz_splitting_25_bins_error'
folder_name = 'nv1_2019_05_10_28MHz_6'    

time_main_plot(folder_name, file_name)
#time_zoom_plot(folder_name, file_name)
histogram(bins = 7, fit_gaussian = True)
#histogram_1hr(bins = 14)

#file4 = '26.3_MHz_splitting_6_bins_error'
#folder4 = 'nv1_2019_05_10_28MHz_4'
#data4 = tool_belt.get_raw_data('t1_double_quantum', file4, folder4)
#
#file5 = '26.5_MHz_splitting_5_bins_error'
#folder5 = 'nv1_2019_05_10_28MHz_5'
#data5 = tool_belt.get_raw_data('t1_double_quantum', file5, folder5)
#
#file6 = '26.2_MHz_splitting_5_bins_error'
#folder6 = 'nv1_2019_05_10_28MHz_6'
#data6 = tool_belt.get_raw_data('t1_double_quantum', file6, folder6)
#
#gamma_list = data4['gamma_list'] + data5['gamma_list'] + data6['gamma_list']
#    
#    
#kde_points, x_grid = kde_sklearn(gamma_list, bandwidth=1.1)
#
#init_guess = [.1, 1, 47, .3, 1, 53, .3, 1, 61]
#
#trp_gssn_popt, pcov = curve_fit(triple_gaussian, x_grid, kde_points, p0 = init_guess)
#
#plt.plot(x_grid, triple_gaussian(x_grid, *trp_gssn_popt), 'b--', label = 'fit')
#plt.legend()
#
#print(trp_gssn_popt)
    