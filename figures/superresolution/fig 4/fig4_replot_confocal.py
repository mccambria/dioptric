# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 13:45:12 2021

@author: agard
"""

import numpy
import utils.tool_belt as tool_belt 
import matplotlib.pyplot as plt

f_size = 8
tick_f_size = 8
clb_f_size = 8
fig_w = 1.3
fig_l = 1.3

fig_tick_l = 3
fig_tick_w = 0.75

clb_tick_1 = 3
clb_tick_w = 0.75
scale = 34500
def create_image_figure(imgArray, imgExtent, clickHandler=None, title = None, 
                        color_bar_label = 'Counts', min_value=None, 
                        um_scaled = False, color_map = 'inferno'):
    
    fig, ax = plt.subplots(dpi=300)
    fig.set_figwidth(fig_w)
    fig.set_figheight(fig_l)
    ax.set_xticks([-400, -200, 0, 200, 400])
    ax.set_yticks([-400, -200, 0, 200, 400])
    # ax.set_xlabel('x (nm)', fontsize = f_size)
    # ax.set_ylabel('y (nm)', fontsize = f_size)
        
    ax.tick_params(which = 'both', length=fig_tick_l, width=fig_tick_w, colors='k',
                    direction='in',grid_alpha=0.7, labelsize = tick_f_size)
    # Tell the axes to show a grayscale image
    img = ax.imshow(imgArray, cmap=color_map, #vmin = 0, vmax = 65,
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
def create_figure(file_name, folder, colormap, sub_folder = None):
    
    data = tool_belt.get_raw_data(file_name, folder)
    x_range = data['x_range']
    y_range = data['y_range']
    x_voltages = data['x_voltages']
    timestamp = data['timestamp']
    nv_sig = data['nv_sig']
    
    coords = [0,0,5.0]
    img_array = numpy.array(data['img_array'])
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

    fig = create_image_figure(img_array_kcps, numpy.array(img_extent)*scale,
                                        color_bar_label = 'kcps',
                                        um_scaled = True,
                                        color_map = colormap
                                        )
    # Redraw the canvas and flush the changes to the backend
    fig.canvas.draw()
    fig.canvas.flush_events()

    file_path = 'E:/Shared drives/Kolkowitz Lab Group/nvdata/paper_materials/charge siv/Figures/Fig 1 - E6 and SiV comparison/'
    file_name = timestamp + '-' + nv_sig['name']
    # tool_belt.save_figure(fig, file_path + file_name +'-YlGnBu_r')

    return fig



def create_zoomed_figure(file_name, folder, colormap, sub_folder = None):
    zoomed_half_range = 14#10
    
    data = tool_belt.get_raw_data(file_name, folder)
    x_range = data['x_range']
    y_range = data['y_range']
    x_voltages = data['x_voltages']
    timestamp = data['timestamp']
    nv_sig = data['nv_sig']
    
    coords = [0,0,5.0]
    img_array = numpy.array(data['img_array'])
    readout = data['readout']
    
    pixel_size = x_voltages[1] - x_voltages[0]
    half_pixel_size = pixel_size / 2

    num_px = len(x_voltages)
    # print(num_px)
    half_num_px = (num_px-1)//2
    start_px = half_num_px-zoomed_half_range
    end_px = half_num_px+zoomed_half_range +1
    # print(start_px)
    # print(end_px)
    z_image_array = img_array[start_px:end_px, start_px:end_px]

    # print(len(z_image_array[0]))
    
    # return
    # x_coord = coords[0]
    half_x_range = x_range / 2
    x_low = -zoomed_half_range*pixel_size
    x_high = zoomed_half_range*pixel_size
    # y_coord = coords[1]
    half_y_range = y_range / 2
    y_low = -zoomed_half_range*pixel_size
    y_high = zoomed_half_range*pixel_size

    img_array_kcps = (z_image_array / 1000) / (readout / 10**9)

    img_extent = [x_low - half_pixel_size, x_high + half_pixel_size,
                  y_low - half_pixel_size, y_high + half_pixel_size]
    # img_extent = [x_low, x_high,
    #               y_low, y_high]
    
    print((x_high + half_pixel_size - (x_low - half_pixel_size))*34500)

    fig = create_image_figure(img_array_kcps, numpy.array(img_extent)*scale,
                                        color_bar_label = 'kcps',
                                        um_scaled = True,
                                        color_map = colormap
                                        )
    # Redraw the canvas and flush the changes to the backend
    fig.canvas.draw()
    fig.canvas.flush_events()

    # file_path = 'E:/Shared drives/Kolkowitz Lab Group/nvdata/paper_materials/charge siv/Figures/Fig 1 - E6 and SiV comparison/'
    # file_name = timestamp + '-' + nv_sig['name']
    # tool_belt.save_figure(fig, file_path + file_name +'-YlGnBu_r')

    return fig



# %% Run the file


if __name__ == '__main__':
    folder = 'pc_rabi/branch_master/image_sample/2021_10'
    file = '2021_10_08-11_14_22-johnson-dnv5_2021_09_23'
    create_zoomed_figure(file, folder, colormap = 'YlGnBu_r')


