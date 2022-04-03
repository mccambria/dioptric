# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 14:56:11 2021

@author: agardill
"""

import matplotlib.pyplot as plt 
import numpy
import math
import utils.tool_belt as tool_belt
# import majorroutines.image_sample as image_sample
import csv

min_counts = 0
max_counts = 7

def create_image_figure(imgArray, imgExtent):
    f_size = 8
    tick_f_size = 8
    clb_f_size = 8
    fig_w = 1.3
    fig_l =1.3
    
    mu = u"\u03BC" 
    
    fig_tick_l = 3
    fig_tick_w = 0.75
    
    clb_tick_1 = 3
    clb_tick_w = 0.75

    
    # Tell matplotlib to generate a figure with just one plot in it
    fig, ax = plt.subplots(dpi=300)
    fig.set_figwidth(fig_w)
    fig.set_figheight(fig_l)
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(which = 'both', length=fig_tick_l, width=fig_tick_w, colors='k',
                    direction='in',grid_alpha=0.7, labelsize = tick_f_size)

        
    # Tell the axes to show an image
    img = ax.imshow(imgArray, cmap='inferno', #vmin = min_counts, vmax = max_counts,
                    extent=tuple(imgExtent))

    # Add a colorbar
    # clb = plt.colorbar(img)
    # clb.set_label("", rotation=270, fontsize = f_size)
    # clb.ax.tick_params( length=clb_tick_1, width=clb_tick_w, grid_alpha=0.7, labelsize = clb_f_size)

    
    
    # Label axes
    # ax.set_xlabel('x (nm)', fontsize = f_size)
    # ax.set_ylabel('y (nm)', fontsize = f_size)
        


    # Draw the canvas and flush the events to the backend
    fig.canvas.draw()
    plt.tight_layout()
    fig.canvas.flush_events()

    return fig


def save_csv(file_name,  sub_folder):
    data = tool_belt.get_raw_data(file_name, sub_folder)
    nv_sig = data['nv_sig']
    CPG_laser_dur = nv_sig['CPG_laser_dur']
    readout_counts_avg = numpy.array(data['readout_counts_avg'])
    # readout_counts_array = numpy.array(data['readout_counts_array'])
    num_steps_b = data['num_steps']    
    a_voltages_1d = data['a_voltages_1d']
    # b_voltages_1d = data['b_voltages_1d']
    # img_range_2D= data['img_range_2D']
    # offset_2D = data["offset_2D"]
    # drift_list = data['drift_list_master']
    axes = [0,1]
    
    print((a_voltages_1d[1] - a_voltages_1d[0])*35000)


    csv_data = []
    for el in readout_counts_avg:
        row=[]
        row.append(el)
        csv_data.append(row)

    csv_file_name = file    
    file_path = 'E:/Shared drives/Kolkowitz Lab Group/nvdata/paper_materials/super resolution/Fig 1/'


    with open('{}/{}.csv'.format(file_path, csv_file_name),
              'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',',
                                quoting=csv.QUOTE_NONE)
        csv_writer.writerows(csv_data)
        
    # file_path = 'E:/Shared drives/Kolkowitz Lab Group/nvdata/paper_materials/charge siv/Figures/Fig 1 - E6 and SiV comparison/'
    # file_name = timestamp + '-' + nv_sig['name']
#    tool_belt.save_figure(fig, file_path + file_name +'-charge')
    

    return #fig  

    
# %%
if __name__ == '__main__':
    
    folder = 'pc_rabi/branch_master/SPaCE/2021_09'
    
    file = '2021_09_06-01_46_43-johnson-nv1_2021_09_03'
    save_csv(file,  folder)
