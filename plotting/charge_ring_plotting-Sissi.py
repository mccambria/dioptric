# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 09:05:07 2020

This file was written to analyze the data taken in exploring charge rings.

This program takes a subtracted scan and averages the counts in pixel-wide 
rings around the center of the image. It plots the average radial counts, and 
fits to the peaks.

@author: agardill
"""

import matplotlib.pyplot as plt 
import numpy
import os
import json
import utils.tool_belt as tool_belt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from pathlib import Path
from pathlib import PurePath

data_dir = 'PUT/THE/FILE/PATH/TO/FOLDER/HERE'

# %%

def gaussian(r, constrast, sigma, center, offset):
    return offset+ constrast * numpy.exp(-((r-center)**2) / (2 * (sigma**2)))

def double_gaussian_dip(freq, low_constrast, low_sigma, low_center, low_offset,
                        high_constrast, high_sigma, high_center, high_offset):
    low_gauss = gaussian(freq, low_constrast, low_sigma, low_center, low_offset)
    high_gauss = gaussian(freq, high_constrast, high_sigma, high_center, high_offset)
    return low_gauss + high_gauss
              
def get_file_list(path_from_nvdata, file_ends_with,
                 data_dir='E:/Shared drives/Kolkowitz Lab Group/nvdata'):
    '''
    Creates a list of all the files in the folder for one experiment, based on
    the ending file name
    '''
    
    data_dir = Path(data_dir)
    file_path = data_dir / path_from_nvdata 
    
    file_list = []
    
    for file in os.listdir(file_path):
        if file.endswith(file_ends_with):
            file_list.append(file)

    return file_list  

def get_raw_data(path_from_nvdata, file_name,
                 data_dir='E:/Shared drives/Kolkowitz Lab Group/nvdata'):
    """Returns a dictionary containing the json object from the specified
    raw data file.
    """

    data_dir = PurePath(data_dir, path_from_nvdata)
    file_name_ext = '{}.txt'.format(file_name)
    file_path = data_dir / file_name_ext

    with open(file_path) as file:
        return json.load(file)
# %%

def radial_distrbution_time(folder_name, sub_folder):
    # create a file list of the files to analyze
    file_list  = get_file_list(folder_name, '.txt', data_dir = data_dir)
    # create lists to fill with data
    green_time_list = []
    ring_radius_list = []
    
    for file in file_list:
        try:
            data = tool_belt.get_raw_data(folder_name, file[:-4], data_dir = data_dir)
            # Get info from file
            nv_sig = data['nv_sig']
            coords = nv_sig['coords']
            x_voltages = data['x_voltages']
            y_voltages = data['y_voltages']
            num_steps = data['num_steps']
            img_range= data['image_range']
            dif_img_array = numpy.array(data['dif_img_array'])
            green_pulse_time = data['green_pulse_time']
            opt_power = data['green_opt_power']
            
            # Initial calculations
            x_coord = coords[0]          
            y_coord = coords[1]
            half_x_range = img_range / 2
            x_high = x_coord + half_x_range

            pixel_size = x_voltages[1] - x_voltages[0]
            half_pixel_size = pixel_size / 2
            
            # List to hold the values of each pixel within the ring
            counts_r = []
            # New 2D array to put the radial values of each pixel
            r_array = numpy.empty((num_steps, num_steps))
            
            # Calculate the radial distance from each point to center
            for i in range(num_steps):
                x_pos = x_voltages[i] - x_coord
                for j in range(num_steps):
                    y_pos = y_voltages[j]  - y_coord
                    r = numpy.sqrt(x_pos**2 + y_pos**2)
                    r_array[i][j] = r
            
            # define bound on each ring radius, which will be one pixel in size
            low_r = 0
            high_r = pixel_size
            
            # step throguh the radial ranges for each ring, add pixel within ring to list
            while high_r <= (x_high + half_pixel_size):
                ring_counts = []
                for i in range(num_steps):
                    for j in range(num_steps): 
                        radius = r_array[i][j]
                        if radius >= low_r and radius < high_r:
                            ring_counts.append(dif_img_array[i][j])
                # average the counts of all counts in a ring
                counts_r.append(numpy.average(ring_counts))
                # advance the radial bounds
                low_r = high_r
                high_r = high_r + pixel_size
            
            # define the radial values as center values of pizels along x, convert to um
            radii = numpy.array(x_voltages[int(num_steps/2):])*35
            # plot
            fig, ax = plt.subplots(1,1, figsize = (8, 8))
            ax.plot(radii, counts_r)
            ax.set_xlabel('Radius (um)')
            ax.set_ylabel('Avg counts around ring (kcps)')
            ax.set_title('{} s, {} mW green pulse'.format(green_pulse_time/10**9, opt_power))
  
#%%############################################################################       
            # try to fit the radial distribution to a double gaussian(work in prog)    
            try:
                contrast_low = 500
                sigma_low = 5
                center_low = -5
                offset_low = 100
                contrast_high = 300
                sigma_high = 5
                center_high = 20
                offset_high = 100            
                guess_params = (contrast_low, sigma_low, center_low, offset_low,
                                contrast_high, sigma_high, center_high, offset_high)
                
                popt, pcov = curve_fit(double_gaussian_dip, radii[1:], counts_r[1:], p0=guess_params)
                radii_linspace = numpy.linspace(radii[0], radii[-1], 1000)
                
                ax.plot(radii_linspace, double_gaussian_dip(radii_linspace, *popt))
                print('fit succeeded')
                
                green_time_list.append(green_pulse_time)
                ring_radius_list.append(popt[6])
 
#%%############################################################################               
            except Exception:
                print('fit failed' )
                
        except Exception:
            continue
    
    return ring_radius_list, green_time_list
    
# %% 
if __name__ == '__main__':

    folder_name = "hopper_8mw_time"
    
    # extract the redii for each data set
    ring_radius_list, green_time_list = radial_distrbution_time(folder_name, sub_folder)
    
    print(ring_radius_list)
    print(green_time_list)
    




