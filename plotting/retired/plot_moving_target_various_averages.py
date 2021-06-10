# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 16:13:28 2020

@author: Aedan
"""
import numpy
import majorroutines.image_sample as image_sample
import utils.tool_belt as tool_belt

folder_name = 'isolate_nv_charge_dynamics_moving_target/branch_Spin_to_charge/2020_12'

# %%
def plot_averages_moving_tager(num_to_avg, file):
    # Get data from the file
    data = tool_belt.get_raw_data(folder_name, file) 
    readout_counts_array=numpy.array(data['readout_counts_array_unsh'])
    # coords_voltages = data['coords_voltages']
    num_steps = data['num_steps']
    x_voltages_1d = data['x_voltages_1d']
    y_voltages_1d = data['y_voltages_1d']
    
    readout_counts_array = numpy.transpose(readout_counts_array)
    
    readout_image_array = numpy.empty([num_steps, num_steps])
    # print(readout_counts_array.shape)
    
    readout_counts_avg = numpy.average(readout_counts_array[:num_to_avg], axis=0)
    # print(readout_counts_avg.shape)
    
    # create the img arrays
    writePos = []
    readout_image_array = image_sample.populate_img_array(readout_counts_avg, readout_image_array, writePos)
    
    # image extent
    x_low = x_voltages_1d[0]
    x_high = x_voltages_1d[num_steps-1]
    y_low = y_voltages_1d[0]
    y_high = y_voltages_1d[num_steps-1]

    pixel_size = (x_voltages_1d[1] - x_voltages_1d[0])
    
    half_pixel_size = pixel_size / 2
    img_extent = [x_high + half_pixel_size, x_low - half_pixel_size,
                  y_low - half_pixel_size, y_high + half_pixel_size]
    

    
    title = 'Moving target, {} of {} runs averaged'.format(num_to_avg, 200)
    tool_belt.create_image_figure(readout_image_array, img_extent,
                                                title = title)
    
    
    return

# %% 
if __name__ == '__main__':
    file = '2020_12_08-goeppert-mayer_10us_NV-unshuffled'
    for i in [20]:
        plot_averages_moving_tager(i, file)