# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 17:35:45 2020

@author: matth
"""


# %% Imports


import numpy
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar as scale_bar
import json


# %% Main


def main(file_names):
    """
    2 x 2 figure. Top left ax blank for level structure, next 3 for sample
    scans. file_names should be a list with 3 paths to the appropriate
    raw data files.
    """
    
    plt.rcParams.update({'font.size': 18})  # Increase font size
    fig, axes_pack = plt.subplots(2, 2, figsize=(10,10))
    fig.set_tight_layout(True)
    ticks = [[10,20,30,40,50,60],
             [20,40,60,80,100],
             [15,20,25,30]]
    
    axes_pack[0,0].set_axis_off()
    
    for ind in range(3):
        
        ax_ind = ind+1
        ax = axes_pack[int(numpy.floor(ax_ind/2)), (ax_ind % 2)]
        name = file_names[ind]
        
        # This next bit is jacked from image_sample.reformat_plot 2/29/2020
        with open(name) as file:

            # Load the data from the file
            data = json.load(file)

            # Build the image array from the data
            # Not sure why we're doing it this way...
            img_array = []
            try:
                file_img_array = data['img_array']
            except:
                file_img_array = data['imgArray']
            for line in file_img_array:
                img_array.append(line)

            # Get the readout in s
            readout = float(data['readout']) / 10**9

            try:
                xScanRange = data['x_range']
                yScanRange = data['y_range']
            except:
                xScanRange = data['xScanRange']
                yScanRange = data['yScanRange']
            
        kcps_array = (numpy.array(img_array) / 1000) / readout

        # ax.set_xlabel('Position ($\mu$m)')
        # ax.set_ylabel('Position ($\mu$m)')
        
        scale = 35  # galvo scaling in microns / volt
        
        # Plot!
        img = ax.imshow(kcps_array, cmap='inferno', interpolation='none')
        ax.set_axis_off()
        
        # Scale bar
        # Find the number of pixels in a micron
        num_steps = kcps_array.shape[0]
        v_resolution = xScanRange / num_steps  # resolution in volts / pixel
        resolution = v_resolution * scale  # resolution in microns / pixel
        px_per_micron = int(1/resolution)
        trans = ax.transData
        bar = scale_bar(trans, 5*px_per_micron, '5 $\mu$m', 'upper right',
                        size_vertical=int(num_steps/100))
        ax.add_artist(bar)

        # Add the color bar
        cbar = fig.colorbar(img, ax=ax, ticks=ticks[ind])
        cbar.ax.set_title('kcps')


# %% Run


if __name__ == '__main__':

    path = 'E:/Shared drives/Kolkowitz Lab Group/nvdata/image_sample/'
    file_names = ['2019_07/2019-07-23_17-39-48_johnson1.txt',
                  '2019_10/2019-10-02-15_12_01-goeppert_mayer-nv_search.txt',
                  '2019_04/2019-04-15_16-42-08_Hopper.txt']
    file_names = [path+name for name in file_names]
    main(file_names)
