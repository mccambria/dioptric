# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 10:31:28 2021

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
fig_w = 2
fig_l =2
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
    
    # ax.set_xticks([-250, 0, 250])
    # ax.set_yticks([-250, 0, 250])
    # ax.set_xticks([-500, 0, 500])
    # ax.set_yticks([-500, 0, 500])
    ax.tick_params(which = 'both', length=fig_tick_l, width=fig_tick_w, colors='k',
                    direction='in',grid_alpha=0.7, labelsize = tick_f_size)
    # ax.set_xlim([580, -620])
    # ax.set_ylim([-600, 600])
        
    # Tell the axes to show an image
    img = ax.imshow(imgArray, cmap='inferno', vmax = 0.6,
                    extent=tuple(imgExtent))

    # Add a colorbar
    # clb = plt.colorbar(img)
    # clb.set_label("", rotation=270, fontsize = f_size)
    # clb.ax.tick_params( length=clb_tick_1, width=clb_tick_w, grid_alpha=0.7, labelsize = clb_f_size)
    
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
# %%

def create_confocal_figure(imgArray, imgExtent, clickHandler=None, title = None, 
                        color_bar_label = 'Counts', min_value=None, 
                        um_scaled = False, color_map = 'inferno'):
    
    fig, ax = plt.subplots(dpi=300)
    fig.set_figwidth(fig_w)
    fig.set_figheight(fig_l)
    # ax.set_xticks([-250, 0, 250])
    # ax.set_yticks([-250, 0, 250])
    # ax.set_xticks([-500, 0, 500])
    # ax.set_yticks([-500, 0, 500])
    # ax.set_xlim([-600, 600])
    # ax.set_ylim([-600, 600])
    # ax.set_xlabel('x (nm)', fontsize = f_size)
    # ax.set_ylabel('y (nm)', fontsize = f_size)
        
    ax.tick_params(which = 'both', length=fig_tick_l, width=fig_tick_w, colors='k',
                    direction='in',grid_alpha=0.7, labelsize = tick_f_size)
    # Tell the axes to show a grayscale image
    img = ax.imshow(imgArray, cmap=color_map, vmin = 0, vmax = 55,
                    extent=tuple(imgExtent))

#    if min_value == None:
#        img.autoscale()

    # Add a colorbar
    # clb = plt.colorbar(img)
    # clb.set_label(color_bar_label, fontsize = clb_f_size, rotation=180)
    # clb.ax.tick_params( length=clb_tick_1, width=clb_tick_w,grid_alpha=0.7, labelsize = clb_f_size)
    # clb.set_ticks([20, 40, 60])
    # clb.set_label('kcounts/sec', rotation=270)
    
    if title:
        plt.title(title)

    # Wire up the click handler to print the coordinates
    if clickHandler is not None:
        fig.canvas.mpl_connect('button_press_event', clickHandler)

    # Draw the canvas and flush the events to the backend
    fig.canvas.draw()
    plt.tight_layout()
    fig.canvas.flush_events()

    return fig
        # %%
def do_plot_confocal_figure(file_name, folder, scale):
    
    data = tool_belt.get_raw_data(file_name, folder)
    x_range = data['x_range']
    y_range = data['y_range']
    try:
        x_voltages = data['x_voltages']
    except Exception:
        x_voltages = data['x_positions_1d']
    timestamp = data['timestamp']
    nv_sig = data['nv_sig']
    
    coords = [0,0,5.0]
    img_array = numpy.array(data['img_array'])
    print(numpy.average(img_array))
    readout = data['readout']

    x_coord = coords[0]
    half_x_range = x_range / 2
    x_low = x_coord - half_x_range
    x_high = x_coord + half_x_range
    y_coord = coords[1]
    half_y_range = y_range / 2
    y_low = y_coord - half_y_range
    y_high = y_coord + half_y_range

    img_array_kcps = (img_array / 1000) / (readout / 10**9)

    pixel_size = x_voltages[1] - x_voltages[0]
    half_pixel_size = pixel_size / 2
    img_extent = [x_low - half_pixel_size, x_high + half_pixel_size,
                  y_low - half_pixel_size, y_high + half_pixel_size]

    fig = create_confocal_figure(img_array_kcps, numpy.array(img_extent)*scale,
                                        color_bar_label = 'kcps',
                                        um_scaled = True,
                                        color_map = 'YlGnBu_r'
                                        )
    # Redraw the canvas and flush the changes to the backend
    fig.canvas.draw()
    fig.canvas.flush_events()

    file_path = 'E:/Shared drives/Kolkowitz Lab Group/nvdata/paper_materials/charge siv/Figures/Fig 1 - E6 and SiV comparison/'
    file_name = timestamp + '-' + nv_sig['name']
    # tool_belt.save_figure(fig, file_path + file_name +'-YlGnBu_r')

    return fig


    
# %%
if __name__ == '__main__':
        

    
    # 2nd airy ring
    folder = 'pc_rabi/branch_CFMIII/SPaCE_digital/2022_01'
    file=  '2022_01_04-10_01_08-johnson-nv0_2021_12_22' 
    threshold = 14
    # do_plot_space_figure(file,  folder, 0.99e3, threshold)
    
    folder = 'pc_rabi/branch_CFMIII/image_sample_digital/2021_12' 
    file = '2021_12_28-00_12_34-johnson-nv0_2021_12_22'
    # do_plot_confocal_figure(file, folder, 0.99e3)
    
    #1st airy ring
    folder = 'pc_rabi/branch_CFMIII/SPaCE_digital/2021_12'
    file = '2021_12_13-09_56_42-johnson-nv0_2021_12_10' 
    threshold = 5
    # do_plot_space_figure(file,  folder, 0.99e3, threshold)
    
    
    folder = 'pc_rabi/branch_CFMIII/image_sample_digital/2021_12' 
    file = '2021_12_13-00_17_25-johnson-nv0_2021_12_10'
            
    do_plot_confocal_figure(file, folder, 0.99e3)
    
    
    # folder = 'pc_rabi/branch_CFMIII/SPaCE_digital_annulus/2021_11'
    # file = '2021_11_30-08_45_30-johnson-nv1_2021_11_17'
    
    # 20 ms
    # folder = 'pc_rabi/branch_CFMIII/SPaCE_digital/2021_12'
    # file=  '2021_12_29-13_52_13-johnson-nv0_2021_12_22'
    # threshold = 9
    
    # # 5 ms
    # folder = 'pc_rabi/branch_CFMIII/SPaCE_digital/2022_01'
    # file=  '2022_01_03-08_49_15-johnson-nv0_2021_12_22'
    # threshold = 9