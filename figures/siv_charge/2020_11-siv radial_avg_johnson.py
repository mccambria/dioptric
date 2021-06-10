# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 14:49:46 2021
File to create figures for SiV charge paper

@author: agardill
"""

import matplotlib.pyplot as plt 
from scipy.optimize import curve_fit
import numpy
import math
import utils.tool_belt as tool_belt
import majorroutines.image_sample as image_sample
     
# %%
def gaussian(x, *params):
    """
    Calculates the value of a gaussian for the given input and parameters

    Params:
        x: float
            Input value
        params: tuple
            The parameters that define the Gaussian
            0: coefficient that defines the peak height
            1: mean, defines the center of the Gaussian
            2: standard deviation, defines the width of the Gaussian
            3: constant y value to account for background
    """

    coeff,  stdev, offset = params
    var = stdev**2  # variance
    centDist = x  # distance from the center
    return offset + coeff**2*numpy.exp(-(centDist**2)/(2*var))

def nv_pop(r, w, pop_ss):
    # w is in um 
    # pop_ss = 0.7
    p = 3 # mW
    a_tot = 7.39 # MHz/mW^2
    t = 1 # us
    
    return pop_ss*(1 - numpy.exp(-p**2*a_tot*t*numpy.exp(-4*r**2/w**2)))

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
    img = ax.imshow(imgArray, cmap='inferno',
                    extent=tuple(imgExtent))

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
def radial_distrbution_data(center_coords, x_voltages, y_voltages, num_steps, img_range, img_array):
    # Initial calculations
    x_coord = center_coords[0]          
    y_coord = center_coords[1]
    
    # subtract the center coords from the x and y voltages so that we are working from the "origin"
    x_voltages = numpy.array(x_voltages) - x_coord
    y_voltages = numpy.array(y_voltages) - y_coord
    
    half_x_range = img_range / 2
    # x_high = x_coord + half_x_range
    
    # Having trouble with very small differences in pixel values. (10**-15 V)
    # Let's round to a relatively safe value and see if that helps
    pixel_size = round(x_voltages[1] - x_voltages[0], 10)
    half_pixel_size = pixel_size / 2
    
    # List to hold the values of each pixel within the ring
    counts_r = []
    # New 2D array to put the radial values of each pixel
    r_array = numpy.empty((num_steps, num_steps))
    
    # Calculate the radial distance from each point to center
    for i in range(num_steps):
        x_pos = x_voltages[i]
        for j in range(num_steps):
            y_pos = y_voltages[j]
            r = numpy.sqrt(x_pos**2 + y_pos**2)
            r_array[i][j] = r
    
    # define bounds on each ring radial values, which will be one pixel in size
    low_r = 0
    high_r = pixel_size
    # step throguh the radial ranges for each ring, add pixel within ring to list
    while high_r <= (half_x_range + pixel_size + 10**-9):
        ring_counts = []
        for i in range(num_steps):
            for j in range(num_steps): 
                radius = r_array[i][j]
                if radius >= low_r and radius < high_r:
                    ring_counts.append(img_array[i][j])
        # average the counts of all counts in a ring
        counts_r.append(numpy.average(ring_counts))
        # advance the radial bounds
        low_r = high_r
        high_r = round(high_r + pixel_size, 10)
    
    # define the radial values as center values of pixels along x, convert to um
    # we need to subtract the center value from the x voltages
    radii = numpy.array(x_voltages[(math.ceil(num_steps/2)):])*35
    
    return radii, counts_r

# %% data from moving target
def plot_radial_avg_moving_target(file_img,  pc_name, branch_name, data_folder, 
                                  sub_folder, threshold, do_plot = True, save_plot = True):

    folder = pc_name + '/' + branch_name + '/' + data_folder + '/' + sub_folder
    
    # Get data from the file
    data = tool_belt.get_raw_data(folder, file_img) 
    timestamp = data['timestamp']
    init_color = data['init_color']
    pulse_color = data['pulse_color']
    nv_sig = data['nv_sig']
    coords = numpy.array(data['start_coords']) #+ [-0.004, 0.00123, 0]#[0.0043,-0.0031,5.0]
    try:
        x_voltages = data['x_voltages_1d']
        y_voltages = data['y_voltages_1d']
    except Exception:
        coords_voltages = data['coords_voltages']
        x_voltages, y_voltages = zip(*coords_voltages)
    num_steps = data['num_steps']
    img_range= data['img_range']
#    readout_image_array = numpy.array(data['readout_image_array'])
    # img_extent = data['img_extent']
    img_array = numpy.array(data['readout_image_array'])
    # print(img_array[20])
    slice_img = img_array[20]
    
    

    if pulse_color == 532:
        opt_power = data['green_optical_power_mW']
        pulse_time = nv_sig['pulsed_reionization_dur']
    if pulse_color == 638:
        opt_power = data['red_optical_power_mW']
        pulse_time = nv_sig['pulsed_ionization_dur']
    
    readout = nv_sig['pulsed_SCC_readout_dur']
    
    ### produce a 2D plot ###
    # coord_mod = [0,0, 0]
    x_coord = coords[0]
    half_x_range = img_range / 2
    x_low = x_coord - half_x_range
    x_high = x_coord + half_x_range
    y_coord = coords[1]
    half_y_range = img_range / 2
    y_low = y_coord - half_y_range
    y_high = y_coord + half_y_range

    pixel_size = x_voltages[1] - x_voltages[0]
    half_pixel_size = pixel_size / 2
    img_extent = [(x_high + half_pixel_size)*35, (x_low - half_pixel_size)*35,
                  (y_low - half_pixel_size)*35, (y_high + half_pixel_size)*35]

    readout_us = readout / 10**3
    title = 'Confocal scan.\nReadout {} us'.format(readout_us)
    fig_2d = create_image_figure(img_array, img_extent,
                                        clickHandler=None,
                                        title = title,
                                        color_bar_label = 'Counts',
                                        um_scaled = False
                                        )
    ####
    # plot radial average
    radii, counts_r = radial_distrbution_data(coords, x_voltages,
                              y_voltages, num_steps, img_range, img_array)
    
    ################
    x_coord = 0.17371429         
    y_coord = 0.24971429
    
    # subtract the center coords from the x and y voltages so that we are working from the "origin"
    x_voltages = numpy.array(x_voltages) - x_coord
    y_voltages = numpy.array(y_voltages) - y_coord
    
    r_array = numpy.empty((num_steps, num_steps))
    
    # Calculate the radial distance from each point to center
    for i in range(num_steps):
        x_pos = x_voltages[i]
        for j in range(num_steps):
            y_pos = y_voltages[j]
            r = numpy.sqrt(x_pos**2 + y_pos**2)
            r_array[i][j] = r
            
    # r_array = r_array.flatten()*35
    # img_array = (numpy.array(img_array).flatten() - 6.5)/6.5
    counts_r = (numpy.array(counts_r) - 6.5) / 6.5
    ################
    # fit to a gaussian
    norm_slice = (numpy.array(slice_img) - 6.5)/6.5
    fit_func = nv_pop
    # popt, _ = curve_fit(fit_func, radii, norm_counts, p0 = [0.1, 0.7])
    x_voltages_um =( numpy.array(x_voltages) - x_coord - 0.004)*35
    # popt, _ = curve_fit(fit_func, r_array, img_array, p0 = [0.1, 0.7])
    popt, _ = curve_fit(fit_func, radii, counts_r, p0 = [0.1, 0.7])

    if do_plot:
        
        fig_1d, ax = plt.subplots(1,1, figsize = (8, 8))
        ax.plot(radii, counts_r)
        lin_radii = numpy.linspace(0, 2, 100)
        ax.plot(lin_radii, fit_func(lin_radii, *popt))
        print(popt)
        
        ax.set_xlabel('Radial distance (um)')
        ax.set_ylabel('Azimuthal avg counts')
        ax.set_title('Radial plot of moving target\n{} nm init pulse\n{} s {} nm pulse at {:.1f} mW'.format(init_color, 
                                                                          pulse_time/10**9, pulse_color,opt_power))
        text = r'$\sigma$={:.3f} um'.format(*popt)
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.6, 0.95, text, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=props)
            
        if save_plot:
            # save data from this file
            rawData = {'timestamp': timestamp,
                        'init_color': init_color,
                        'pulse_color': pulse_color,
                        'pulse_time': pulse_time,
                        'pulse_time-units': 'ns',
                        'opt_power': opt_power,
                        'opt_power-units': 'mW',
                        'readout': readout,
                        'readout-units': 'ns',
                        'nv_sig': nv_sig,
                        'nv_sig-units': tool_belt.get_nv_sig_units(),
                        'num_steps': num_steps,
                        'radii': radii.tolist(),
                        'radii-units': 'um',
                        'counts_r': counts_r,
                        'counts_r-units': 'kcps'}
                
            file_path = 'E:/Shared drives/Kolkowitz Lab Group/nvdata/pc_rabi/branch_Spin_to_charge/isolate_nv_charge_dynamics_moving_target/2020_11'
            # tool_belt.save_raw_data(rawData, filePath + '_radial_dist')
            file_name = timestamp + '-' + nv_sig['name']
            # tool_belt.save_figure(fig_1d, file_path + '/' + file_name +'-radial_avg')
            # tool_belt.save_raw_data(rawData, file_path + '/' + file_name +'-radial_avg')


    return radii, counts_r

# %%
if __name__ == '__main__':

    pc_name = 'pc_rabi'
    branch_name = 'branch_Spin_to_charge'
    # file_img = '2020_11_26-01_26_16-johnson-nv18_2020_11_10-img' # 0.5 mW 200 us
    # file_img = '2020_11_25-22_33_28-johnson-nv18_2020_11_10-img' # 1 mW 200 us
    # file_img = '2020_11_25-19_03_06-johnson-nv18_2020_11_10-img' # 3 mW 200 us
    
    file_img = '2020_11_26-03_49_44-johnson-nv18_2020_11_10-img' # 3 mW, 1 us
    
    # file_img = '2020_12_08-18_04_02-goeppert-mayer-nv1_2020_12_02-img'
    data_folder = 'isolate_nv_charge_dynamics_moving_target'
    sub_folder = '2020_11'
    plot_radial_avg_moving_target(file_img, pc_name, branch_name, data_folder, 
                                  sub_folder, None, do_plot = True, save_plot = True)
