# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 15:13:16 2021

@author: agard
"""

import fig3_replot_SpaCE
import numpy
import matplotlib.pyplot as plt
import utils.tool_belt as tool_belt
import canny_and_houghes_fig_3 as circle_fit

color_1 = '#00a651'
color_2 = '#f7941d'
#dnv7
A = [ -0.0318, 0.275, 4.85]
B = [-0.035, 0.276, 4.85]
C = [-0.022,0.288, 4.85]


def plot_2D_space(file, path, true_position = False):
        data = tool_belt.get_raw_data(file, path)
        # try:
        nv_sig = data['nv_sig']
        CPG_laser_dur = nv_sig['CPG_laser_dur']
        readout = nv_sig['charge_readout_dur']
        readout_s = readout/10**9
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
        
        img_extent = [(a_low - half_pixel_size_a)*34.8,
                      (a_high + half_pixel_size_a)*34.8, 
                     
                     (b_low - half_pixel_size_b)*34.8, 
                     (b_high + half_pixel_size_b)*34.8 ]
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
        
        

        fig, ax = fig3_replot_SpaCE.create_image_figure(readout_image_array / 1000 / readout_s, img_extent)
        
        
        # Plot points for A, B, and C
        dx, dy, dz = offset_2D
        offset_2D = [dx-0.05/34.8, dy, dz]
        adj_A = (A - center - offset_2D)*34.8
        adj_B = (B - center - offset_2D)*34.8
        adj_C = (C - center - offset_2D)*34.8
        
        dot_size = 6
        letter_size = 6
        plt.plot(-adj_A[0], adj_A[1], 'b^', color = color_1, markersize = dot_size)
        plt.plot(-adj_B[0], adj_B[1], 'rs',color = color_2, markersize = dot_size)
        
        ret_vals=circle_fit.find_centers(file + '.txt')
        a_x, a_y, a_r, b_x, b_y, b_r = ret_vals
        
        ax.plot(-(a_x- center[0])*34.8, (a_y- center[1])*34.8, 'b.', color = color_1, markersize = 4.5)
        ax.plot(-(b_x- center[0])*34.8, (b_y- center[1])*34.8, 'r.',color = color_2, markersize = 4.5)
        
if __name__ == '__main__':
    
    # file = '2021_09_29-14_08_01-johnson-dnv7_2021_09_23'
    # file = '2021_09_24-19_30_38-johnson-dnv7_2021_09_23'
    file = '2021_09_30-13_18_47-johnson-dnv7_2021_09_23'
    path = 'pc_rabi/branch_master/SPaCE/2021_09'
    
        
    plot_2D_space(file, path, True)