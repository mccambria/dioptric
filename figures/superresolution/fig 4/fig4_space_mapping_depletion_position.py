# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 15:13:16 2021

@author: agard
"""

import utils.tool_belt as tool_belt
import numpy
import matplotlib.pyplot as plt
import canny_and_houghes_fig_4 as circle_fit

color_1 = '#00aeef'
color_2 = '#ee3542'
scale = 34800
#dnv5
A = [-0.001,
      -0.008,
      4.08457485]
coords = [0.00949217,
      -0.00614178,
      4.08457485]
B = [
      -0.007,
      -0.008,
      4.08457485
    ]

A = [0.00849217, -0.01414178,  8.1691497]
B = [2.4921700e-03, -1.4141780e-02,  8.1691497e+00]
C = [0.157, 0.126, 4.79]

def create_image_figure(imgArray, imgExtent):
    f_size = 8
    tick_f_size = 8
    clb_f_size = 8
    # fig_w = 1.617 
    # fig_l =1.617 
    fig_w = 3
    fig_l =3
    
    mu = u"\u03BC" 
    
    fig_tick_l = 3
    fig_tick_w = 0.75
    
    clb_tick_1 = 3
    clb_tick_w = 0.75

    
    # Tell matplotlib to generate a figure with just one plot in it
    fig, ax = plt.subplots(dpi=300)
    fig.set_figwidth(fig_w)
    fig.set_figheight(fig_l)
    
    ax.set_xticks([-0.5, 0, 0.5])
    ax.set_yticks([-0.5, 0, 0.5])
    ax.tick_params(which = 'both', length=fig_tick_l, width=fig_tick_w, colors='k',
                    direction='in',grid_alpha=0.7, labelsize = tick_f_size)

        
    # Tell the axes to show an image
    img = ax.imshow(imgArray, cmap='inferno', #vmin = 0, vmax = 1,
                    extent=tuple(imgExtent))

    # Add a colorbar
    clb = plt.colorbar(img)
    clb.set_label("", rotation=270, fontsize = f_size)
    clb.ax.tick_params( length=clb_tick_1, width=clb_tick_w, grid_alpha=0.7, labelsize = clb_f_size)

    
    
    # Label axes
    # ax.set_xlabel('x (nm)', fontsize = f_size)
    # ax.set_ylabel('y (nm)', fontsize = f_size)
        


    # Draw the canvas and flush the events to the backend
    fig.canvas.draw()
    plt.tight_layout()
    fig.canvas.flush_events()

    return fig, ax

def plot_2D_space(file, path, threshold, true_position = False):
        data = tool_belt.get_raw_data(file, path)
        # try:
        nv_sig = data['nv_sig']
        CPG_laser_dur = nv_sig['CPG_laser_dur']
        readout = nv_sig['charge_readout_dur']
        readout_s = readout/10**9
        readout_counts_avg = numpy.array(data['readout_counts_avg'])
        # readout_counts_array = numpy.array(data['readout_counts_array'])
        num_steps_b = data['num_steps_b']    
        a_voltages_1d = data['a_voltages_1d']
        b_voltages_1d = data['b_voltages_1d']
        img_range_2D= data['img_range_2D']
        drift_list = data['drift_list_master']
        offset_2D = data["offset_2D"]
        center = numpy.array(nv_sig['coords'])
        axes = [0,1]
        
        # convert single shot measurements to NV- population
        # bright_counts = [] #let's try to recreate the charge state populations
        # dark_counts = []
        # raw_counts = numpy.array(data['readout_counts_array'])
        # for r in range(len(raw_counts)):
        #     row = raw_counts[r]
        #     for c in range(len(row)):
        #         current_val = raw_counts[r][c]
        #         if current_val < threshold:
        #             set_val = 0
        #         elif current_val >= threshold:
        #             set_val = 1
        #         raw_counts[r][c] = set_val
                
                
            # collect data for histogram at top left and in the middle
            # half=int((len(raw_counts)-1)/2)
            # step = int(numpy.sqrt(len(raw_counts)))
            # for i in range(1,21):
            #     b_point = len(raw_counts)-i
            #     if r == b_point:
            #         print(b_point)
            #         bright_counts.append(numpy.array(data['readout_counts_array'])[b_point])
            # for i in range(-10,10):
            #     d_point = half + step*i
            #     if r == d_point:
            #         print(d_point)
            #         dark_counts.append(numpy.array(data['readout_counts_array'])[d_point])
                    
        # plot histogram
        # nv0 = [int(item) for sublist in dark_counts for item in sublist]
        # nvm = [int(item) for sublist in bright_counts for item in sublist]
        # print(nvm)
        # print(nv0)
        # fig_hist, ax = plt.subplots(1, 1)
        # max_0 = max(nv0)
        # max_m = max(nvm)
        # occur_0, x_vals_0 = numpy.histogram(nv0, numpy.linspace(0,max_0, max_0+1))
        # occur_m, x_vals_m = numpy.histogram(nvm, numpy.linspace(0,max_m, max_m+1))
        # ax.plot(x_vals_0[:-1],occur_0,  'r-o', label = 'Initial red pulse' )
        # ax.plot(x_vals_m[:-1],occur_m,  'g-o', label = 'Initial green pulse' )
        # ax.set_xlabel('Counts')
        # ax.set_ylabel('Occur.')
        # ax.legend()
        # return
        
        # readout_counts_avg = numpy.average(raw_counts, axis = 1)
        
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
        
        img_extent = [(a_low - half_pixel_size_a)*scale/1e3,
                      (a_high + half_pixel_size_a)*scale/1e3, 
                     
                     (b_low - half_pixel_size_b)*scale/1e3, 
                     (b_high + half_pixel_size_b)*scale/1e3 ]
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
        
        
        

        fig, ax = create_image_figure(readout_image_array / 1000 / readout_s, img_extent)
        
        
        # Plot points for A, B, and C
        dx, dy, dz = offset_2D
        offset_2D = [dx+ 0.276/35, dy -0.146/35 , dz]
        adj_A = (A - center- offset_2D)*35
        adj_B = (B - center - offset_2D)*35
        # adj_C = (C - center - offset_2D)*35
        
        dot_size = 6
        letter_size = 6
        plt.plot(-adj_A[0], adj_A[1], 'b^', color = color_1, markersize = dot_size)
        plt.plot(-adj_B[0], adj_B[1], 'rs',color = color_2, markersize = dot_size)
        
        ret_vals=circle_fit.find_centers(file + '.txt')
        c_x, c_y, c_r, d_x, d_y, d_r = ret_vals
        
        ax.plot(-(c_x- center[0])*35, (c_y- center[1])*35, 'b.', color = color_1, markersize = 4.5)
        ax.plot(-(d_x- center[0])*35, (d_y- center[1])*35, 'r.',color = color_2, markersize = 4.5)
        
        # plt.plot(-adj_C[0], adj_C[1] , 'go', markersize = dot_size)
        # plt.plot(-adj_A[0]-0.07, adj_A[1]+0.1, marker = '$A$', color = 'b', markersize = letter_size)
        # plt.plot(-adj_B[0]-0.07, adj_B[1]+0.1, marker = '$B$', color = 'r', markersize = letter_size)
        # plt.plot(-adj_C[0]-0.07, adj_C[1]+0.1, marker = '$C$', color = 'g', markersize = letter_size)
        
if __name__ == '__main__':
    # path = 'pc_rabi/branch_master/SPaCE/2021_09'
    
    file = '2021_10_17-19_02_22-johnson-dnv5_2021_09_23'
    path = 'pc_rabi/branch_master/SPaCE/2021_10'
    threshold = 1 #don't have actual data, but was able to estimate based on this data
    plot_2D_space(file, path,threshold,  True)