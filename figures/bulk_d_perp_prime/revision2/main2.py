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
import matplotlib.font_manager


# %% Main


def main(file_names):
    """
    2 x 2 figure. Top left ax blank for level structure, next 3 for sample
    scans. file_names should be a list with 3 paths to the appropriate
    raw data files.
    """
    
    # plt.rcParams.update({'font.size': 18})  # Increase font size
    fig, axes_pack = plt.subplots(1, 3, figsize=(6.75, 2.23))
    ticks = [[10,20,30,40,50,60],
             [20,40,60,80,100],
             [15,20,25,30]]
    centers = [[96, 31], [100, 100], [50,50]]
    fig_labels = [r'(a)', r'(b)', r'(c)']
    
    for ind in range(3):
        
        ax = axes_pack[ind]
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
        
        # Scaling
        scale = 35  # galvo scaling in microns / volt
        num_steps = kcps_array.shape[0]
        v_resolution = xScanRange / num_steps  # resolution in volts / pixel
        resolution = v_resolution * scale  # resolution in microns / pixel
        px_per_micron = 1/resolution
        
        # Plot! 5 um out from center in any direction
        center = centers[ind]
        clip_range = 5 * px_per_micron 
        x_clip = [center[0] - clip_range, center[0] + clip_range]
        x_clip = [int(el) for el in x_clip]
        y_clip = [center[1] - clip_range, center[1] + clip_range]
        y_clip = [int(el) for el in y_clip]
        # print((x_clip[1] - x_clip[0]) * v_resolution)
        clip_array = kcps_array[x_clip[0]: x_clip[1], 
                                y_clip[0]: y_clip[1]]
        img = ax.imshow(clip_array, cmap='inferno', interpolation='none')
        ax.set_axis_off()
        
        # Scale bar
        trans = ax.transData
        bar_text = r'2 $\upmu$m '  # Insufficient left padding...
        bar = scale_bar(trans, 2*px_per_micron, bar_text, 'upper right',
                        size_vertical=int(num_steps/100),
                        pad=0.2, borderpad=0.5, sep=3.0,
                        # frameon=False, color='white',
                        )
        ax.add_artist(bar)
        
        # Fig label
        fig_label = fig_labels[ind]
        ax.text(0.035, 0.88, fig_label, transform=ax.transAxes,
                color='white', fontsize=16)

        # Add the color bar
        # cbar = fig.colorbar(img, ax=ax, ticks=ticks[ind])
        # cbar.ax.set_title('kcps')
        
    fig.tight_layout(pad=0.1, w_pad=0.5)


# %% Run


if __name__ == '__main__':
    
    plt.rc('text', usetex=True)
    plt.rcParams['text.latex.preamble'] = [
        r'\usepackage{physics}',
        r'\usepackage{sfmath}',
        r'\usepackage{upgreek}',
        r'\usepackage{helvet}',
       ]  
    plt.rcParams.update({'font.size': 13})
    plt.rcParams.update({'font.family': 'sans-serif'})
    plt.rcParams.update({'font.sans-serif': ['Helvetica']})

    path = 'E:/Shared drives/Kolkowitz Lab Group/nvdata/image_sample/'
    file_names = ['2019_07/2019-07-23_17-39-48_johnson1.txt',
                  '2019_10/2019-10-02-15_12_01-goeppert_mayer-nv_search.txt',
                  '2019_04/2019-04-15_16-42-08_Hopper.txt']
    file_names = [path+name for name in file_names]
    
    # NV data in three samples at B perp ~ 0
    nv_data = [
        {
            'name': 'NVA3',
            'gamma': 0.114,
            'gamma_err': 0.01,
            'omega': 0.059,
            'omega_err': 0.004,
            'ratio': 1.9322,
            'ratio_err': 0.2142,
            'perp_B': 0.5536,
        },
        {
            'name': 'NVB1',
            'gamma': 0.121,
            'gamma_err': 0.01,
            'omega': 0.062,
            'omega_err': 0.005,
            'ratio': 1.9516,
            'ratio_err': 0.2254,
            'perp_B': 0.0,
        },
        {
            'name': 'NVE',
            'gamma': 0.12,
            'gamma_err': 0.011,
            'omega': 0.061,
            'omega_err': 0.004,
            'ratio': 1.9672,
            'ratio_err': 0.2217,
            'perp_B': 0.0,
        }
    ]
    
    main(file_names, nv_data)
