# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 15:34:03 2021

plotting a 2D image array from data

@author: agard
"""
import matplotlib.pyplot as plt 
import numpy
import json
from pathlib import PurePath
# import utils.tool_belt as tool_belt

def get_raw_data(path_from_nvdata, file_name,
                 nvdata_dir='E:/Shared drives/Kolkowitz Lab Group/nvdata'):
    """Returns a dictionary containing the json object from the specified
    raw data file.
    """

    data_dir = PurePath(nvdata_dir, path_from_nvdata)
    file_name_ext = '{}.txt'.format(file_name)
    file_path = data_dir / file_name_ext

    with open(file_path) as file:
        return json.load(file)
    
def on_click_image(event):
    """
    Click handler for images. Prints the click coordinates to the console.

    Params:
        event: dictionary
            Dictionary containing event details
    """

    try:
        print('{:.3f}, {:.3f}'.format(event.xdata, event.ydata))
#        print('[{:.3f}, {:.3f}, 50.0],'.format(event.xdata, event.ydata))
    except TypeError:
        # Ignore TypeError if you click in the figure but out of the image
        pass

def create_image_figure(imgArray, imgExtent, clickHandler=None, title = None, color_bar_label = 'Counts', min_value=None, um_scaled = False):
    """
    Creates a figure containing a single grayscale image and a colorbar.

    Params:
        imgArray: numpy.ndarray
            Rectangular numpy array containing the image data.
            Just zeros if you're going to be writing the image live.
        imgExtent: list(float)
            The extent of the image in the form [left, right, bottom, top]
        clickHandler: function
            Function that fires on clicking in the image
        title: str
            Indicate the title you want for the figure
        color_bar_label: str
            Indicate a label for the color bar. 
        min_value: flt
            Can indicate if you want the plot to exclude counts below some 
            minimum value
        um_scaled: boolean
            If True, the image will automaticaly be scaled to um isntead of V
        

    Returns:
        matplotlib.figure.Figure
    """
    axes_label = 'V'
    # Tell matplotlib to generate a figure with just one plot in it
    fig, ax = plt.subplots()
    if um_scaled:
        axes_label = 'um'
        
    # Tell the axes to show a grayscale image
    img = ax.imshow(imgArray, cmap='inferno',
                    extent=tuple(imgExtent), vmin = min_value)

#    if min_value == None:
#        img.autoscale()

    # Add a colorbar
    clb = plt.colorbar(img)
    clb.set_label(color_bar_label, rotation=270)
#    clb.set_label('kcounts/sec', rotation=270)
    
    # Label axes
    plt.xlabel(axes_label)
    plt.ylabel(axes_label)
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

def plot_2D_figure(file_name, sub_folder = None):
    data = get_raw_data('', file_name)
    try:
        x_range = data['x_range']
        y_range = data['y_range']
    except Exception as e:
        print(e)
        x_range = data['image_range']
        y_range = x_range
    x_voltages = data['x_voltages']
    try:
        nv_sig = data['nv_sig']
        coords = nv_sig['coords']
    except Exception as e:
        print(e)
        coords = data['coords']
    try:
        img_array = numpy.array(data['img_array'])
    except Exception as e:
        print(e)
        img_array = numpy.array(data['dif_img_array'])
    
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
    img_extent = [x_high + half_pixel_size, x_low - half_pixel_size,
                  y_low - half_pixel_size, y_high + half_pixel_size]

    readout_us = readout / 10**3
    title = 'Confocal scan.\nReadout {} us'.format(readout_us)
    fig = create_image_figure(img_array_kcps, img_extent,
                                        clickHandler=on_click_image,
                                        title = title,
                                        color_bar_label = 'kcps',
                                        min_value = 0,
                                        um_scaled = True # False if you want the iamge scaled in galvo Volts
                                        )
    # Redraw the canvas and flush the changes to the backend
    fig.canvas.draw()
    fig.canvas.flush_events()

    return fig

    
# %% 
if __name__ == '__main__':
    file_name = 'pc_rabi/branch_Spin_to_charge/image_sample/2020_06/2020_06_26-13_01_44-Hopper-ensemble_dif'
    plot_2D_figure(file_name)
