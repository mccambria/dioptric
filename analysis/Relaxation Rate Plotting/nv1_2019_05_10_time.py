# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 14:43:51 2019

Plotting time for the NV1_2019_05_10

@author: agardill
"""
import matplotlib.pyplot as plt
import utils.tool_belt as tool_belt

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
                    
# %%

def time_plot(folder_name, file_name):
    '''
    Basic function to plot the data we collected on this NV. Data represented
    as points.
    '''
#    file4 = '26.3_MHz_splitting_6_bins_error'
#    folder4 = 'nv1_2019_05_10_28MHz_4'
#    data4 = tool_belt.get_raw_data('t1_double_quantum', file4, folder4)
#    
#    file5 = '26.5_MHz_splitting_5_bins_error'
#    folder5 = 'nv1_2019_05_10_28MHz_5'
#    data5 = tool_belt.get_raw_data('t1_double_quantum', file5, folder5)
#    
#    file6 = '26.2_MHz_splitting_5_bins_error'
#    folder6 = 'nv1_2019_05_10_28MHz_6'
#    data6 = tool_belt.get_raw_data('t1_double_quantum', file6, folder6)
#    
#    gamma_list = data4['gamma_list'] + data5['gamma_list'] + data6['gamma_list'] 
#    gamma_ste_list = data4['gamma_ste_list'] + data5['gamma_ste_list'] \
#                    + data6['gamma_ste_list']
    
#    time_inc = 1380 / len(gamma_list) # min (4)
#    time_inc = 1147 / len(gamma_list) # min (5)
#    time_inc = 0.75 # hr
#    time_inc = 1.25 # hr
    time_inc = 3.75 # hr
    
    time_list = []
    for i in range(len(gamma_list)):
        time = i*time_inc
        time_list.append(time)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.errorbar(time_list, gamma_list, yerr = gamma_ste_list,
                label = r'$\gamma$', fmt='o', markersize = 10,color='blue')
#    ax.plot(gamma_list,'b-',
#                label = r'$\gamma$',  markersize = 10)

    ax.tick_params(which = 'both', length=6, width=2, colors='k',
                    grid_alpha=0.7, labelsize = 18)

    ax.tick_params(which = 'major', length=12, width=2)

    ax.grid()

    ax.set_xlabel('Time (hour)', fontsize=18)
    ax.set_ylabel('Relaxation Rate (kHz)', fontsize=18)
    ax.set_title(r'NV1', fontsize=18)
    ax.legend(fontsize=18)
    fig.canvas.draw()
    fig.canvas.flush_events()
    
def histogram(folder_name, file_name, bins):
    '''
    Produces a histogram of the data passed
    '''
#    data = tool_belt.get_raw_data('t1_double_quantum', file_name, folder_name)
    
#    gamma_list = data['gamma_list']
    
    plt.hist(gamma_list, bins = bins)
    plt.xlabel('Gamma (kHz)')
    plt.ylabel('Occurances')
    
    
# %%
file_name = '26.2_MHz_splitting_25_bins_error'
folder_name = 'nv1_2019_05_10_28MHz_6'    

time_plot(folder_name, file_name)
#histogram(folder_name, file_name, 7)

    