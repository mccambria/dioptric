# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 11:47:55 2022

@author: agard
"""


import matplotlib.pyplot as plt 
import numpy
import math
import utils.tool_belt as tool_belt
# import majorroutines.image_sample as image_sample
import csv

f_size = 8
tick_f_size = 8
clb_f_size = 8
fig_w = 3
fig_l =3
mu = u"\u03BC" 

fig_tick_l = 3
fig_tick_w = 0.75

clb_tick_1 = 3
clb_tick_w = 0.75
    
def create_space_figure(imgArray, imgExtent):
    

    
    # Tell matplotlib to generate a figure with just one plot in it
    fig, ax = plt.subplots(dpi=300)
    fig.set_figwidth(fig_w)
    fig.set_figheight(fig_l)
    
    ax.set_xticks([-250, 0, 250])
    ax.set_yticks([-250, 0, 250])
    # ax.set_xticks([-500, 0, 500])
    # ax.set_yticks([-500, 0, 500])
    ax.tick_params(which = 'both', length=fig_tick_l, width=fig_tick_w, colors='k',
                    direction='in',grid_alpha=0.7, labelsize = tick_f_size)
    # ax.set_xlim([580, -620])
    # ax.set_ylim([-600, 600])
        
    # Tell the axes to show an image
    img = ax.imshow(imgArray, cmap='inferno',# vmax = 0.6,
                    extent=tuple(imgExtent))

    # Add a colorbar
    clb = plt.colorbar(img)
    clb.set_label("", rotation=270, fontsize = f_size)
    clb.ax.tick_params( length=clb_tick_1, width=clb_tick_w, grid_alpha=0.7, labelsize = clb_f_size)
    
    # clb.set_ticks([0, 0.2, 0.4, 0.6,0.8,1])
    
    
    # Label axes
    # ax.set_xlabel('x (nm)', fontsize = f_size)
    # ax.set_ylabel('y (nm)', fontsize = f_size)
        


    # Draw the canvas and flush the events to the backend
    fig.canvas.draw()
    plt.tight_layout()
    fig.canvas.flush_events()

    return fig


def do_plot_space_figure(file_name,  sub_folder, scale, threshold):
    data = tool_belt.get_raw_data(file_name, sub_folder)
    nv_sig = data['nv_sig']
    CPG_laser_dur = nv_sig['CPG_laser_dur']
    # readout_counts_avg = numpy.array(data['readout_counts_avg'])
    readout_image_array = numpy.array(data['readout_image_array'])
    img_extent = numpy.array(data['img_extent'])*scale
    num_steps_b = data['num_steps_b']    
    # a_voltages_1d = data['a_voltages_1d']
    # b_voltages_1d = data['b_voltages_1d']
    # img_range_2D= data['img_range_2D']
    # offset_2D = data["offset_2D"]
    # drift_list = data['drift_list_master']
    axes = [0,1]
    
    
    # convert single shot measurements to NV- population
    raw_counts = numpy.array(data['readout_counts_array'])
    for r in range(len(raw_counts)):
        row = raw_counts[r]
        for c in range(len(row)):
            current_val = raw_counts[r][c]
            if current_val < threshold:
                set_val = 0
            elif current_val >= threshold:
                set_val = 1
            raw_counts[r][c] = set_val
    readout_counts_avg = numpy.average(raw_counts, axis = 1)

    split_counts = numpy.split(readout_counts_avg, num_steps_b)
    readout_image_array = numpy.vstack(split_counts)
    r = 0
    
    for i in reversed(range(len(readout_image_array))):
        if r % 2 == 0:
            readout_image_array[i] = list(reversed(readout_image_array[i]))
        r += 1
    
    readout_image_array = numpy.flipud(readout_image_array)
    



    fig = create_space_figure(readout_image_array, img_extent)
    # Redraw the canvas and flush the changes to the backend
    fig.canvas.draw()
    fig.canvas.flush_events()
        
    # file_path = 'E:/Shared drives/Kolkowitz Lab Group/nvdata/paper_materials/charge siv/Figures/Fig 1 - E6 and SiV comparison/'
    # file_name = timestamp + '-' + nv_sig['name']
#    tool_belt.save_figure(fig, file_path + file_name +'-charge')
    

    return fig  


# vertical polarization
folder = 'pc_rabi/branch_CFMIII/SPaCE_digital/2021_12'
file = '2021_12_13-09_56_42-johnson-nv0_2021_12_10' 
threshold = 5
do_plot_space_figure(file,  folder, 0.97e3, threshold)


# horizontal polarization
folder = 'pc_rabi/branch_master/SPaCE_digital/2022_01'
file = '2022_01_27-03_41_50-johnson-nv5_2022_01_24' 
threshold = 5
do_plot_space_figure(file,  folder, 0.97e3, threshold)


# circular polarization
folder = 'pc_rabi/branch_master/SPaCE_digital/2022_01'
file = '2022_01_30-14_58_08-johnson-nv1_2022_01_28' 
threshold = 7
do_plot_space_figure(file,  folder, 0.97e3, threshold)