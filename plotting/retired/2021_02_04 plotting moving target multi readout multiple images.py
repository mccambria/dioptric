# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 14:52:01 2021

a file just to stich together multiple images into one larger image.

All the smaller images must be the same size and dimensions, and can't overlap

@author: agardill
"""

import numpy
import utils.tool_belt as tool_belt
import time

import matplotlib.pyplot as plt
def chunks(lst, n):
    for i in range(0,len(lst),n):
        yield lst[i:i + n]
#    return
# %%
def main(folder_path):
    file_list = tool_belt.get_file_list(folder_path, 'txt')
    
    # edit file names to exclude '.txt', and sort files
    file_list =  [el[:-4] for el in file_list]
    file_list = sorted(file_list)
    
    # calc the number of images in x and y direction
    n = int(numpy.sqrt(len(file_list)))
    
    # get the image range form the first file
    data = tool_belt.get_raw_data(folder_path, file_list[0])
    num_steps = data['num_steps']
    
    # seperate the file list into n number of sublists
    seperated_folder_list = list(chunks(file_list, n))
    
    # Take each row in the fields and append them
    complete_image_array = []
    
    # Might be kind of inneficient, but I am going to save the x and y values 
    # in all the img extenets, and then take the extreme values for the min and max
    x_img_extent = []
    y_img_extent = []
    
    # start with the first file sublist
    for f in range(n):
        for i in range(num_steps):
            for c in range(n):
                data = tool_belt.get_raw_data(folder_path, seperated_folder_list[f][c])
                readout_image_array = data['readout_image_array']
                img_extent = data['img_extent']
                for el in readout_image_array[i]:
                    complete_image_array.append(el)
                x_img_extent.append(img_extent[0])
                x_img_extent.append(img_extent[1])
                y_img_extent.append(img_extent[2])
                y_img_extent.append(img_extent[3])
                
    image_array = numpy.array(list(chunks(complete_image_array, num_steps*n)))
    
    x_low = min(x_img_extent)
    x_high = max(x_img_extent)
    y_low = min(y_img_extent)
    y_high = max(y_img_extent)
    
    total_img_extent = [x_high, x_low, y_low, y_high]
    
    
#    title = '{}\nwith {} nm {} ms pulse'.format(title_list[i], pulse_color, pulse_time/10**6)
    title = 'goeppert-mayer_NV13_2021_01_26\nwith 515 nm 10 ms pulse'
    tool_belt.create_image_figure(image_array, numpy.array(total_img_extent)*35,
                                                title = title, um_scaled = True)
        
    return

# %% Run the files

if __name__ == '__main__':
    
    
    pc = 'pc_rabi'
    branch = 'branch_Spin_to_charge'
    data_folder = 'moving_target_multi_readout'
    date_folder = '2021_02'
    folder = '2021_02_04-goeppert-mayer-nv13-2021_01_26'
   
    folder_path = pc + '/' + branch + '/' + data_folder + '/' + date_folder + '/' + folder
    
    main(folder_path)
    