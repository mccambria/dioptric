# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 14:56:11 2021

@author: agardill
"""

import matplotlib.pyplot as plt 
import numpy
import utils.tool_belt as tool_belt
import majorroutines.image_sample as image_sample

def create_image_figure(imgArray, imgExtent, clickHandler=None, title = None, color_bar_label = 'Counts',  um_scaled = False):
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
    axes_label = r'Remote Pulse Position, Relative to NV ($\mu$m)'#'V'
    # Tell matplotlib to generate a figure with just one plot in it
    fig, ax = plt.subplots()
    if um_scaled:
        axes_label = r'$mu$m'
        
    # Tell the axes to show a grayscale image
    img = ax.imshow(imgArray, cmap='inferno', vmin = 0, vmax = 1,
                    extent=tuple(imgExtent))

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
def create_figure(file_name, threshold, sub_folder = None):
#    if sub_folder:
    data = tool_belt.get_raw_data('', file_name)
#    else:
#        data = tool_belt.get_raw_data('image_sample', file_name)
    x_range = data['img_range']
    y_range = data['img_range']
    x_voltages = data['x_voltages_1d']
    num_steps = data['num_steps']
    nv_sig = data['nv_sig']
    readout = nv_sig['pulsed_SCC_readout_dur']
    coords = [0,0,5.0]#data['start_coords']
#    img_array = numpy.array(data['readout_image_array'])
    raw_counts = numpy.array(data['readout_counts_array'])
    


    # charge state information
    cut_off = threshold
    
    # for each individual measurement, determine if the NV was in NV0 or NV- by threshold.
    # Then average the measurements for each pixel to gain mean charge state.
    for r in range(len(raw_counts)):
        row = raw_counts[r]
        for c in range(len(row)):
            current_val = raw_counts[r][c]
            if current_val < cut_off:
                set_val = 0
            elif current_val >= cut_off:
                set_val = 1
            raw_counts[r][c] = set_val
    charge_counts_avg = numpy.average(raw_counts, axis = 1)
    
    
    # create the img arrays
    readout_image_array = numpy.empty([num_steps, num_steps])
    readout_image_array[:] = numpy.nan
    writePos = []
    img_array = image_sample.populate_img_array(charge_counts_avg, readout_image_array, writePos)
    
    x_coord = coords[0]
    half_x_range = x_range / 2
    x_low = x_coord - half_x_range
    x_high = x_coord + half_x_range
    y_coord = coords[1]
    half_y_range = y_range / 2
    y_low = y_coord - half_y_range
    y_high = y_coord + half_y_range

#    img_array_chrg = (img_array - nv0_avg) / (nvm_avg - nv0_avg)

#    img_array_cps = (img_array_chrg) / (readout / 10**9)

    pixel_size = x_voltages[1] - x_voltages[0]
    half_pixel_size = pixel_size / 2
    img_extent = [(x_high + half_pixel_size)*35, (x_low - half_pixel_size)*35,
                  (y_low - half_pixel_size)*35, (y_high + half_pixel_size)*35]

    readout_us = readout / 10**3
    title = 'Confocal scan.\nReadout {} us'.format(readout_us)
    fig = create_image_figure(img_array, img_extent,
                                        clickHandler=None,
                                        title = title,
                                        color_bar_label = 'NV- population (arb)',
                                        um_scaled = False
                                        )
    # Redraw the canvas and flush the changes to the backend
    fig.canvas.draw()
    fig.canvas.flush_events()

    return fig    
# %%
if __name__ == '__main__':
    
#    image_file = 'pc_rabi/branch_Spin_to_charge/moving_target_siv_init/2021_03/2021_03_23-13_19_10-goeppert-mayer-nv1-2021_03_17' #dark
    image_file = 'pc_rabi/branch_Spin_to_charge/moving_target_siv_init/2021_03/2021_03_23-13_37_37-goeppert-mayer-nv1-2021_03_17' #bright
    threshold = 6
    create_figure(image_file, threshold)