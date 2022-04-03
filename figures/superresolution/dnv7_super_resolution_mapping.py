# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 15:13:16 2021

@author: agard
"""

import utils.tool_belt as tool_belt
import numpy
import matplotlib.pyplot as plt

#dnv7
A = [ -0.0318, 0.275, 4.85]
B = [-0.035, 0.276, 4.85]
C = [-0.022,0.288, 4.85]

#dnv5
# A = [0.159, 0.139, 4.79]
# B = [0.152, 0.134,  #[0.153, 0.139, 
#      4.79]
# C = [0.157, 0.126, 4.79]

def plot_2D_space(file, path, true_position = False):
        data = tool_belt.get_raw_data(file, path)
        # try:
        nv_sig = data['nv_sig']
        CPG_laser_dur = nv_sig['CPG_laser_dur']
        readout_counts_avg = numpy.array(data['readout_counts_avg'])
        readout_counts_array = numpy.array(data['readout_counts_array'])
        num_steps_b = data['num_steps_b']    
        a_voltages_1d = data['a_voltages_1d']
        b_voltages_1d = data['b_voltages_1d']
        img_range_2D= data['img_range_2D']
        drift_list = data['drift_list_master']
        offset_2D = data["offset_2D"]
        center = numpy.array(nv_sig['coords'])
        axes = [0,1]
        
        half_range_a = img_range_2D[axes[0]]/2
        half_range_b = img_range_2D[axes[1]]/2
        a_low = -half_range_a
        a_high = half_range_a
        b_low = -half_range_b
        b_high = half_range_b
        
        # a_low = -half_range_a + offset_2D[axes[0]]
        # a_high = half_range_a + offset_2D[axes[0]]
        # b_low = -half_range_b + offset_2D[axes[1]]
        # b_high = half_range_b + offset_2D[axes[1]]


        pixel_size_a = (a_voltages_1d[1] - a_voltages_1d[0])
        pixel_size_b = (b_voltages_1d[1] - b_voltages_1d[0])

        half_pixel_size_a = pixel_size_a / 2
        half_pixel_size_b = pixel_size_b / 2
        
        img_extent = [(a_low - half_pixel_size_a)*35,
                      (a_high + half_pixel_size_a)*35, 
                     
                     (b_low - half_pixel_size_b)*35, 
                     (b_high + half_pixel_size_b)*35 ]
        um_scaled = True
        
        

        
        split_counts = numpy.split(readout_counts_avg, num_steps_b)
        readout_image_array = numpy.vstack(split_counts)
        r = 0
        for i in reversed(range(len(readout_image_array))):
            if r % 2 == 0:
                readout_image_array[i] = list(reversed(readout_image_array[i]))
            r += 1
        
        readout_image_array = numpy.flipud(readout_image_array)
        title = 'SPaCE - {} ms depletion pulse'.format(CPG_laser_dur)
        
        

        fig = tool_belt.create_image_figure(readout_image_array, img_extent, clickHandler=None,
                            title='', color_bar_label='Counts',
                            min_value=None, um_scaled=um_scaled)
        
        
        # Plot points for A, B, and C
        dx, dy, dz = offset_2D
        offset_2D = [dx, -dy, dz]
        adj_A = (A - center - offset_2D)*35
        adj_B = (B - center - offset_2D)*35
        # adj_C = (C - center - offset_2D)*35
        
        plt.plot(-adj_A[0], adj_A[1]+0.2, 'bo', markersize = 10)
        plt.plot(-adj_B[0], adj_B[1]+0.2, 'ro', markersize = 10)
        # plt.plot(-adj_C[0], adj_C[1] - 0.05, 'go', markersize = 10)
        plt.plot(-adj_A[0]-0.07, adj_A[1]+0.1, marker = '$A$', color = 'b', markersize = 20)
        plt.plot(-adj_B[0]-0.07, adj_B[1]+0.1, marker = '$B$', color = 'r', markersize = 20)
        # plt.plot(-adj_C[0]-0.07, adj_C[1]+0.1, marker = '$C$', color = 'g', markersize = 20)
        
if __name__ == '__main__':
    
    file = '2021_09_29-14_08_01-johnson-dnv7_2021_09_23'
    # file = '2021_09_24-19_30_38-johnson-dnv7_2021_09_23'
    path = 'pc_rabi/branch_master/SPaCE/2021_09'
    
    # file = '2021_10_05-11_09_19-johnson-dnv5_2021_09_23'
    # path = 'pc_rabi/branch_master/SPaCE/2021_10'
        
    plot_2D_space(file, path, True)