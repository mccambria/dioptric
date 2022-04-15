# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 13:45:12 2021

@author: agard
"""

import numpy
import json
import matplotlib.pyplot as plt
from pathlib import PurePath

#some parameters for the file size and font size
f_size = 8
tick_f_size = 8
clb_f_size = 8
fig_w = 4
fig_l = 4

fig_tick_l = 3
fig_tick_w = 0.75

clb_tick_1 = 3
clb_tick_w = 0.75

#scaling of X/Y voltage to um
scale = 33.700

datadir = 'E:/Shared drives/Kolkowitz Lab Group/nvdata'
def get_raw_data(path_from_datadir, file_name,
                 data_dir=datadir):
    """Returns a dictionary containing the json object from the specified
    raw data file.
    """

    data_dir = PurePath(data_dir, path_from_datadir)
    file_name_ext = '{}.txt'.format(file_name)
    file_path = data_dir / file_name_ext

    with open(file_path) as file:
        return json.load(file)
    
def create_image_figure(imgArray, imgExtent,  title = None, 
                        color_map = 'inferno'):
    
    fig, ax = plt.subplots(dpi=300)
    fig.set_figwidth(fig_w)
    fig.set_figheight(fig_l)
    ax.set_xlabel('x (um)', fontsize = f_size)
    ax.set_ylabel('y (um)', fontsize = f_size)
        
    ax.tick_params(which = 'both', length=fig_tick_l, width=fig_tick_w, colors='k',
                    direction='in',grid_alpha=0.7, labelsize = tick_f_size)

    img = ax.imshow(imgArray, cmap=color_map, #vmin = 0, vmax = 65,
                    extent=tuple(imgExtent))


    # Add a colorbar
    clb = plt.colorbar(img)
    clb.set_label('kcounts/sec', fontsize = clb_f_size, rotation=270)
    clb.ax.tick_params( length=clb_tick_1, width=clb_tick_w,grid_alpha=0.7, labelsize = clb_f_size)
    
    if title:
        plt.title(title)

    # Draw the canvas and flush the events to the backend
    fig.canvas.draw()
    plt.tight_layout()
    fig.canvas.flush_events()

    return fig
        # %%
def plot_figure(file_name, folder, colormap, sub_folder = None):
    
    data = get_raw_data(folder, file_name)
    x_range = data['x_range']
    y_range = data['y_range']
    x_voltages = data['x_voltages']
    img_array = numpy.array(data['img_array']) #get the raw data of the image
    readout = data['readout'] #get the readout time for each point
    
    coords = [0,0,5.0] #set the center to 0,0
    ###if you want the actual coordinates, then use: 
    #nv_sig = data['nv_sig']
    #coords = nv_sig['coords']

    #Some calculations for the size and resolution of the image
    x_coord = coords[0]
    half_x_range = x_range / 2
    x_low = x_coord - half_x_range
    x_high = x_coord + half_x_range
    y_coord = coords[1]
    half_y_range = y_range / 2
    y_low = y_coord - half_y_range
    y_high = y_coord + half_y_range

    pixel_size = x_voltages[1] - x_voltages[0]
    half_pixel_size = pixel_size / 2
    img_extent = [x_low - half_pixel_size, x_high + half_pixel_size,
                  y_low - half_pixel_size, y_high + half_pixel_size]

    #convert the raw data to kcps
    img_array_kcps = (img_array / 1000) / (readout / 10**9)

    # create the figure using the function create_image_figure
    fig = create_image_figure(img_array_kcps, numpy.array(img_extent)*scale,
                                        color_map = colormap
                                        )
    # Redraw the canvas and flush the changes to the backend
    fig.canvas.draw()
    fig.canvas.flush_events()


    return fig




# %% Run the file


if __name__ == '__main__':
    folder = 'pc_rabi/branch_master/image_sample/2021_10'
    file = '2021_10_08-11_14_22-johnson-dnv5_2021_09_23'
    color_map = 'inferno' # You can change the color map of the image.
    plot_figure(file, folder, colormap = color_map)


