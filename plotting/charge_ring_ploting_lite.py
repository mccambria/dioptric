# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 09:35:53 2020

@author: Aedan 
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
 
# %%

def radial_distrbution_data(center_coords, x_voltages, y_voltages, num_steps, img_range, img_array):
    '''
    Base function to take 2D image and average azimuthally

    Parameters
    ----------
    center_coords : list (float)
        Coordinates that the image is centered around [x,y,z]
    x_voltages : list (float)
        List of x_voltages used for the image range.
    y_voltages : list (float)
        List of y_voltages used for the image range.
    num_steps : int
        resolution of image.
    img_range : float
        range of image, in V, both x and y.
    img_array : list (float)
        2D array of the data contained in the image.

    Returns
    -------
    radii : numpy array
        The radii of each radial point, in units of um.
    counts_r : list
        The averaged azimuthal counts at each radial distance from the center.

    '''
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
    # Note, I was runnign into rounding issue with the uper bound, probably a 
    # poor fix of just adding a small bit to the bound
    while high_r <= (half_x_range + half_pixel_size + 10**-9):
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
    radii = numpy.array(x_voltages[int(num_steps/2):])*35
    
    return radii, counts_r
    
# %% data from moving readout
def plot_radial_avg_moving_readout(file, folder_name, do_plot = True):
    '''
    Use this function to plot the azimuthal averaged counts as a function of radius 
    from the moving readout (charge ring) data. 
    
    You need to pass the daa file ending with "dif", which has it's data stored
    under "dif_img_array"

    Parameters
    ----------
    file : str
        file name, excluding the .txt
    pc_name : str
        name of the pc, either 'pc_rbi' or 'pc_hahn'
    branch_anme: str
        the name of the branch that the data is saved under
    data_folder: str
        data folder that file is saved in, ex: 'moving_target'
    sub_folder : str
        the path from the parent folder to the folder containing the file.
    do_plot : Boolean, optional
        If True, the radial coutns will be plotted. The default is True.
    save_plot : Boolean, optional
        If True, the figure and data will be saved. The default is True.

    Returns
    -------
    radii : list
        The radii of each radial point, in units of um.
    counts_r : numpy array
        The averaged azimuthal counts at each radial distance from the center.

    '''
    folder = folder_name
    
    # Get data from the file
    data = get_raw_data(folder, file) 
    timestamp = data['timestamp']
    nv_sig = data['nv_sig']
    coords = nv_sig['coords']
    x_voltages = data['x_voltages']
    y_voltages = data['y_voltages']
    num_steps = data['num_steps']
    img_range= data['image_range']
    dif_img_array = numpy.array(data['dif_img_array'])
    try:
        pulse_time = data['pulse_time']
    except Exception:
        pulse_time = data['green_pulse_time']
    readout = data['readout']
    try:
        opt_volt = data['green_optical_voltage']
        opt_power = data['green_opt_power']
    except Exception:
        opt_volt = data['green_optical_power_pd']
        opt_power = data['green_optical_power_mW']
        
    
    # plot
    
    radii, counts_r = radial_distrbution_data(coords, x_voltages,
                              y_voltages, num_steps, img_range, dif_img_array)
    # convert ring counts to kcps
    counts_r = numpy.array(counts_r)/ 1000 / (readout / 10**9)
    
    if do_plot:
    
        fig, ax = plt.subplots(1,1, figsize = (8, 8))
        radii = radii[:-1]
        counts_r = counts_r[:-1]
        ax.plot(radii, counts_r)
        ax.set_xlabel('Radial distance (um)')
        ax.set_ylabel('Azimuthal avg counts (kcps)')
        ax.set_title('Radial plot of subtracted confocal scan\n{} s pulse at {:.1f} mW'.format(pulse_time, opt_power))

        

    return radii, counts_r, opt_power, pulse_time
    
# %% 
if __name__ == '__main__':
    # # Changing power data
    # file_list = ['1', '3', '4', '5', '6', '8', '10', '12', '25', '60']
    
    # pc_name = 'pc_rabi'
    # branch_name = 'branch_Spin_to_charge'
    # data_folder = 'image_sample'
    # sub_folder = '2020_06/hopper_1s_power/select_data'
    
    # folder = pc_name + '/' + branch_name + '/' + data_folder + '/' + sub_folder
    
    # # for changing power data
    # fig, ax = plt.subplots(1,1, figsize = (8, 8))
    # for i in range(len(file_list)):
    #     file = file_list[i]
    #     ret_vals =plot_radial_avg_moving_readout(file, folder, do_plot = False)
    #     radii, counts_r, opt_power, pulse_time  = ret_vals
    #     ax.plot(radii, counts_r, label = '{:.1f} mW'.format(opt_power))
        
    # ax.set_xlabel('Radial distance (um)')
    # ax.set_ylabel('Azimuthal avg counts (kcps)')
    # ax.set_title('Azimuthal average for {} s pulse length'.format(pulse_time/10**9))
    # ax.legend()
    
    # Changing time data
    file_list = ['0.1', '1', '2', '5', '10', '25', '50', '100', '250' , '1000']
    
    pc_name = 'pc_rabi'
    branch_name = 'branch_Spin_to_charge'
    data_folder = 'image_sample'
    sub_folder = '2020_06/hopper_4mw_time/select_data'
    folder = pc_name + '/' + branch_name + '/' + data_folder + '/' + sub_folder
    # for changing power data
    fig, ax = plt.subplots(1,1, figsize = (8, 8))
    for i in range(len(file_list)):
        file = file_list[i]
        ret_vals =plot_radial_avg_moving_readout(file, folder, do_plot = False)
        radii, counts_r, opt_power, pulse_time  = ret_vals
        ax.plot(radii, counts_r, label = '{} s'.format(pulse_time/10**9))
        
    ax.set_xlabel('Radial distance (um)')
    ax.set_ylabel('Azimuthal avg counts (kcps)')
    ax.set_title('Azimuthal average for {:.2f} mW pulse'.format(opt_power))
    ax.legend()
    
