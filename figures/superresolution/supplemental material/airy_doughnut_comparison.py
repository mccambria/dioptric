# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 10:32:39 2022

@author: agard
"""

import utils.tool_belt as tool_belt
import numpy
import matplotlib.pyplot as plt
import csv
# %%
fig_tick_l = 3
fig_tick_w = 0.75

f_size = 8
f_size_legend = 5

mu = u"\u03BC"
superscript_minus = u"\u207B" 

nvdata_dir = 'E:/Shared drives/Kolkowitz Lab Group/nvdata'
folder = 'paper_materials/super_resolution/supplemental_figs/mathematica_doughnut_airy_comp'

# %% Import data from Mathematica csv files

def import_csv_data(file):
    x_data_list = []
    y_data_list =[]
    file_loc = nvdata_dir + '/' + folder + '/' + file + '.csv'
    
    with open(file_loc, 'r') as file:
        csvreader = csv.reader(file)
        for row in csvreader:
            x = float(row[0])
            if row[1] == 'Indeterminate': #for airy pattern, x = 0 is indeterminant, but approaches 1
                y = 1
            else:
                y = float(row[1])
            x_data_list.append(x)
            y_data_list.append(y)

    return x_data_list, y_data_list

# %%

def full_range_comp():
    file_airy = 'airy_full_range'
    file_doug = 'doughnut_full_range'
    x_airy, y_airy = import_csv_data(file_airy)
    x_doug, y_doug  = import_csv_data(file_doug)
    
    max_doughnut = max(y_doug)
    print(max_doughnut)
    
    fig_l = 2
    fig_w =fig_l * 1.8
    fig, ax = plt.subplots()
    fig.set_figwidth(fig_w)
    fig.set_figheight(fig_l)
    
    ax.set_xlabel(r"Radial position, $r$ (nm)", fontsize = f_size)
    ax.set_ylabel(r'Normalized intensity, $I / I_0$', fontsize = f_size)
    ax.tick_params(which = 'both', length=fig_tick_l, width=fig_tick_w,
                    direction='in',grid_alpha=0.7, labelsize = f_size)
    
    # ax.axvline(x=0, color='grey', linestyle = '-', linewidth = 0.7)
    # ax.axhline(y=0, color='grey', linestyle = '-', linewidth = 0.7)
    
    ax.axhline(y=max_doughnut, color='grey', linestyle = '--', linewidth = 0.7)
    
    ax.plot(x_airy, y_airy,
                '-',  color= 'red',  linewidth = 0.75, label = 'Airy profile')
    ax.plot(x_doug, y_doug,
                '-',  color= 'blue',  linewidth = 0.75, label = 'Doughnut profile')
    ax.legend(fontsize = f_size_legend)
    ax.set_ylim([0, 1.05])


def expand_comp():
    file_airy_exact_1= 'airy_n1_exact_zoom'
    file_airy_exact_2= 'airy_n2_exact_zoom'
    file_doughnut_exact = 'doughnut_exact_zoom'
    x_airy_exact_1, y_airy_exact_1 = import_csv_data(file_airy_exact_1)
    x_airy_exact_2, y_airy_exact_2 = import_csv_data(file_airy_exact_2)
    x_doug_exact, y_doug_exact  = import_csv_data(file_doughnut_exact)
    
    file_airy_expand_1= 'airy_n1_expand'
    file_airy_expand_2= 'airy_n2_expand'
    file_doughnut_expand = 'doughnut_expand'
    x_airy_expand_1, y_airy_expand_1 = import_csv_data(file_airy_expand_1)
    x_airy_expand_2, y_airy_expand_2 = import_csv_data(file_airy_expand_2)
    x_doug_expand, y_doug_expand  = import_csv_data(file_doughnut_expand)
    
    fig_l = 2
    fig_w =fig_l * 1.8
    fig, ax = plt.subplots()
    fig.set_figwidth(fig_w)
    fig.set_figheight(fig_l)
    ax.set_xlabel(r"Radial distance from zero intensity point, $r '$ (nm)", fontsize = f_size)
    ax.set_ylabel(r'Normalized intensity, $I / I_0$', fontsize = f_size)
    ax.tick_params(which = 'both', length=fig_tick_l, width=fig_tick_w,
                    direction='in',grid_alpha=0.7, labelsize = f_size)
    
    ax.plot(x_airy_expand_1, y_airy_expand_1,
                '-',  color= 'red',  linewidth = 0.75, 
                label = r'Airy profile, centered at $n_1$ (approx.)')
    ax.plot(x_airy_exact_1, y_airy_exact_1,
                '--',  color= 'red',  linewidth = 0.75, 
                label = r'Airy profile, centered at $n_1$ (exact)')
    ax.plot(x_airy_expand_2, y_airy_expand_2,
                '-',  color= 'orange',  linewidth = 0.75, 
                label = r'Airy profile, centered at $n_2$ (approx.)')
    ax.plot(x_airy_exact_2, y_airy_exact_2,
                '--',  color= 'orange',  linewidth = 0.75, 
                label = r'Airy profile, centered at $n_2$ (exact)')
    ax.plot(x_doug_expand, y_doug_expand,
                '-',  color= 'blue',  linewidth = 0.75, label = 'Doughnut profile (approx.)')
    ax.plot(x_doug_exact, y_doug_exact,
                '--',  color= 'blue',  linewidth = 0.75, label = 'Doughnut profile (exact)')
    ax.legend(fontsize = f_size_legend)
    
    # # set the x-spine (see below for more info on `set_position`)
    # ax.spines['left'].set_position('zero')
    
    # turn off the right spine/ticks
    # ax.spines['right'].set_color('none')
    # ax.yaxis.tick_left()
    
    # # set the y-spine
    # ax.spines['bottom'].set_position('zero')
    
    # # turn off the top spine/ticks
    # ax.spines['top'].set_color('none')
    # ax.xaxis.tick_bottom()
    
    ax.set_ylim([0, 0.018635257923212713])
    
# %%

full_range_comp()

expand_comp()
