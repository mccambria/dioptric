# -*- coding: utf-8 -*-
"""
Scan the galvos over the designated area, collecting counts at each point.
Generate an image of the sample.

Includes a replotting routine to show the data with axes in um instead of V.

Includes a replotting routine to replot rw data to manipulate again.

Created on Tue Apr  9 15:18:53 2019

@author: mccambria
"""

import numpy
import utils.tool_belt as tool_belt
import matplotlib.pyplot as plt

    
def create_image_figure(imgArray, imgExtent, clickHandler=None, title = None, 
                        color_bar_label = 'Counts', min_value=None, 
                        um_scaled = False, color_map = 'inferno'):
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

    Returns:
        matplotlib.figure.Figure
    """
    axes_label = 'V'
    # Tell matplotlib to generate a figure with just one plot in it
    fig, ax = plt.subplots()
    if um_scaled:
        axes_label = r'$\mu$m'
        
    # Tell the axes to show a grayscale image
    img = ax.imshow(imgArray, cmap=color_map,
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
def create_figure(file_name, colormap, sub_folder = None):
    
    data = tool_belt.get_raw_data('', file_name)
    x_range = data['x_range']
    y_range = data['y_range']
    x_voltages = data['x_voltages']
    
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

    fig = create_image_figure(img_array_kcps, numpy.array(img_extent)*35,
                                        color_bar_label = 'kcps',
                                        um_scaled = True,
                                        color_map = colormap
                                        )
    # Redraw the canvas and flush the changes to the backend
    fig.canvas.draw()
    fig.canvas.flush_events()

    return fig




# %% Run the file


if __name__ == '__main__':
    # NV confocal scans
    file_name = 'pc_rabi/branch_Spin_to_charge/image_sample/2021_04/2021_04_07-09_47_41-goeppert-mayer-nv13_2021_04_02' # NV13 GP
    file_name = 'pc_rabi/branch_Spin_to_charge/image_sample/2021_04/2021_04_14-09_28_46-johnson-nv0_2021_04_13' # NV0 J
    create_figure(file_name, colormap = 'YlGnBu_r')
    
    # SiV confocal scans
    file_name = 'pc_rabi/branch_Spin_to_charge/image_sample/2021_04/2021_04_16-09_39_51-goeppert-mayer' # bright
    file_name = 'pc_rabi/branch_Spin_to_charge/image_sample/2021_04/2021_04_16-09_43_06-goeppert-mayer' # dark
    
    

#    create_figure(file_name, colormap = 'afmhot')

