# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 10:14:06 2022

@author: agard
"""

import matplotlib.pyplot as plt 
import numpy
import utils.tool_belt as tool_belt
from scipy.optimize import curve_fit
import sympy

f_size = 8
tick_f_size = 8
clb_f_size = 8
fig_w = 2.6
fig_l =fig_w
mu = u"\u03BC" 

fig_tick_l = 3
fig_tick_w = 0.75

clb_tick_1 = 3
clb_tick_w = 0.75

             
def triple_gaussian(x, *params):
    """
    Calculates the value of a gaussian for the given input and parameters

    Params:
        x: float
            Input value
        params: tuple
            The parameters that define the Gaussian
            0: coefficient that defines the peak height of all three peaks
            1: mean, defines the distance between the centers of the identical Gaussians
            2: x_offset, defines the center of the pattern of identical Gaussians
            3: standard deviation, defines the width of the Gaussians
            4: constant y value to account for background
    """
    
    coeff, mean, x_offset, stdev, offset = params
    var = stdev ** 2  # variance
    centDist_1 = x - mean + x_offset  # distance from the center
    centDist_2 = x + x_offset  # distance from the center
    centDist_3 = x + mean + x_offset  # distance from the center
    return offset + coeff ** 2  *( numpy.exp(-(centDist_1 ** 2) / (2 * var)) \
             + numpy.exp(-(centDist_2 ** 2) / (2 * var)) \
             +  numpy.exp(-(centDist_3 ** 2) / (2 * var)) )

def create_confocal_figure(imgArray, imgExtent, min_val=None, max_val = None, 
                           color_map = 'inferno'):
    
    fig, ax = plt.subplots(dpi=300)
    fig.set_figwidth(fig_w)
    fig.set_figheight(fig_l)
    # ax.set_xticks([-250, 0, 250])
    # ax.set_yticks([-250, 0, 250])
    # ax.set_xticks([-500, 0, 500])
    # ax.set_yticks([-500, 0, 500])
    # ax.set_xlim([-600, 600])
    # ax.set_ylim([-600, 600])
    ax.set_xlabel('x (V)', fontsize = f_size)
    ax.set_ylabel('y (V)', fontsize = f_size)
        
    ax.tick_params(which = 'both', length=fig_tick_l, width=fig_tick_w, colors='k',
                    direction='in',grid_alpha=0.7, labelsize = tick_f_size)
    # Tell the axes to show a grayscale image
    img = ax.imshow(imgArray, cmap=color_map, vmin = min_val, vmax = max_val,
                    extent=tuple(imgExtent))

#    if min_value == None:
#        img.autoscale()

    # Add a colorbar
    # clb = plt.colorbar(img)
    # clb.set_label('kcps', fontsize = clb_f_size, rotation=270)
    # clb.ax.tick_params( length=clb_tick_1, width=clb_tick_w,grid_alpha=0.7, labelsize = clb_f_size)
    # clb.set_ticks([20, 40, 60])
    # clb.set_label('kcounts/sec', rotation=270)
    
    # Draw the canvas and flush the events to the backend
    fig.canvas.draw()
    plt.tight_layout()
    fig.canvas.flush_events()

    return fig

def do_plot_confocal_figure(file_name, folder, max_val=None, min_val = None, ):
    
    data = tool_belt.get_raw_data(file_name, folder)
    x_range = data['x_range']
    y_range = data['y_range']
    try:
        x_voltages = data['x_voltages']
    except Exception:
        x_voltages = data['x_positions_1d']
    
    coords = [0,0,5.0]
    img_array = numpy.array(data['img_array'])
    readout = data['readout']

    x_coord = coords[0]
    half_x_range = x_range / 2
    x_low = x_coord - half_x_range
    x_high = x_coord + half_x_range
    y_coord = coords[1]
    half_y_range = y_range / 2
    y_low = y_coord - half_y_range
    y_high = y_coord + half_y_range

    img_array_kcps = (img_array / 1000) / (readout / 10**9)

    pixel_size = x_voltages[1] - x_voltages[0]
    half_pixel_size = pixel_size / 2
    img_extent = [x_low - half_pixel_size, x_high + half_pixel_size,
                  y_low - half_pixel_size, y_high + half_pixel_size]

    fig = create_confocal_figure(img_array_kcps, numpy.array(img_extent),
                                         min_val=min_val, max_val = max_val, 
                                        color_map = 'YlGnBu_r'
                                        )
    # Redraw the canvas and flush the changes to the backend
    fig.canvas.draw()
    fig.canvas.flush_events()
    return fig

def dx(fn, x, delta=0.001):
    return (fn(x+delta) - fn(x))/delta

def solve(fn, value, x=0.5, maxtries=1000, maxerr=0.00001):
    for tries in range(maxtries):
        err = fn(x) - value
        if abs(err) < maxerr:
            return x
        slope = dx(fn, x)
        x -= err/slope
    raise ValueError('no solution found')

def fit_triple_gaus(pos_vals, counts):    
    f_size = 8
    tick_f_size = 8
    clb_f_size = 8
    fig_w = 1.8
    fig_l = 1
    linewidth=1
    
    fig_tick_l = 3
    fig_tick_w = 0.75
    
    fig, ax = plt.subplots()
    fig.set_figwidth(fig_w)
    fig.set_figheight(fig_l)
    
    
    
    ax.tick_params(which = 'both', length=fig_tick_l, width=fig_tick_w,
                            direction='in',grid_alpha=0.7, labelsize = f_size)
    ax.set_ylabel('kcps', fontsize = f_size)
    ax.set_xlabel('y(V)', fontsize = f_size)
    
    fit_func = triple_gaussian
    

            
    init_params = [numpy.sqrt(max(counts)), 
                   pos_vals[-1]/1.5, 0, pos_vals[-1]*0.175, 
                    0.7] 
    #fit a sum of two gaussians to the histogram to estimate the max and min counts
    popt, pcov = curve_fit(
        fit_func,
        pos_vals,
        counts ,
        p0=init_params,
        # bounds = ([-numpy.infty, 
        #             pos_vals[0]*1.1, -numpy.infty,
        #             pos_vals[0], -numpy.infty,
        #             0, -numpy.infty,
        #             -0.1,],
        #           [numpy.infty, 
        #             0, numpy.infty,
        #             pos_vals[-1], numpy.infty,
        #             pos_vals[-1]*1.1, numpy.infty,
        #             numpy.infty,])
        )    
    
    if popt[-1] < -0.1:
        popt, pcov = curve_fit(
        fit_func,
        pos_vals,
        counts ,
        p0=init_params,
        bounds = ([-numpy.infty, 
                    0, -numpy.infty, -numpy.infty,
                    0,],
                  [numpy.infty, 
                    pos_vals[-1]*1, numpy.infty,numpy.infty,
                    numpy.infty,])
        )    
        
        
    ax.plot(pos_vals, counts, 'bo',   linewidth = linewidth, 
            markersize = 3 , mfc='white')
    
    
    pos_vals_cont = numpy.linspace(pos_vals[0], pos_vals[-1],100)
    ax.plot(pos_vals_cont, fit_func(pos_vals_cont, *popt), 'r-',   linewidth = linewidth)
    ax.set_ylim([-1.5, 55])
    
    # print(popt)
    
    # hm = (popt[0]**2)/2 + popt[-1]
    # ax.plot(pos_vals_cont,[ hm]*len(pos_vals_cont), 'r-',   linewidth = linewidth)
    
    # print((popt[0]**2) / (popt[0]**2)/2)
    # print(popt)
    
    # now find the intersection between the half max and the Guasian curves.
    # fn = lambda x: fit_func(x, *popt)
    
    # l_center = popt[2] - popt[1]
    # r_center = popt[2] + popt[1]
    # lsol = solve(fn, hm, l_center-abs(popt[3]*1.5))
    # hsol = solve(fn, hm, r_center-abs(popt[3]*1.5)) ##########
    # print(lsol)
    # print(hsol)

    
    # dist = hsol - lsol
    
    # um = 10.965 *4/5
    # conversion = um / dist
    # # print(conversion)
    
    #use the spacing between gaussians to define length
    delta = popt[1]
    um_ = 10.965 *2/5
    print(um_)
    conversion = um_ / delta
    
    print(conversion)
    
    
    return conversion

def plot_1D_cut_for_horizontal_lines(file_name, folder):
    data = tool_belt.get_raw_data(file_name, folder)
    x_range = data['x_range']
    y_range = data['y_range']
    try:
        x_voltages = data['x_voltages']
    except Exception:
        x_voltages = data['x_positions_1d']
    num_steps = len(x_voltages)
    # timestamp = data['timestamp']
    # nv_sig = data['nv_sig']
    
    coords = [0,0,5.0]
    img_array = numpy.array(data['img_array'])
    readout = data['readout']

    y_coord = coords[1]
    half_y_range = y_range / 2
    y_low = y_coord - half_y_range
    y_high = y_coord + half_y_range
    
    y_vals = numpy.linspace(y_low, y_high, num_steps)

    img_array_kcps = (img_array / 1000) / (readout / 10**9)

    ###### Y slice
    c_list = []
    for sl in [0.5]:#[0.25, 0.5, 0.75]:
        counts = []
        x_ind = int(num_steps*sl)
        for i in reversed(range(num_steps)):
            counts.append(img_array_kcps[i][x_ind])
    
        conversion = fit_triple_gaus(y_vals, counts)
        c_list.append(conversion)
    
    print('average conversion: {}'.format(numpy.average(c_list)))
       
   
        
    return 

def plot_1D_cut_for_vertical_lines(file_name, folder, cfm):    
    data = tool_belt.get_raw_data(file_name, folder)
    x_range = data['x_range']
    y_range = data['y_range']
    try:
        x_voltages = data['x_voltages']
    except Exception:
        x_voltages = data['x_positions_1d']
    num_steps = len(x_voltages)
    # timestamp = data['timestamp']
    # nv_sig = data['nv_sig']
    
    coords = [0,0,5.0]
    img_array = numpy.array(data['img_array'])
    readout = data['readout']

    x_coord = coords[0]
    half_x_range = x_range / 2
    x_low = x_coord - half_x_range
    x_high = x_coord + half_x_range
    y_coord = coords[1]
    half_y_range = y_range / 2
    y_low = y_coord - half_y_range
    y_high = y_coord + half_y_range
    
    y_vals = numpy.linspace(y_low, y_high, num_steps)

    img_array_kcps = (img_array / 1000) / (readout / 10**9)


    ###### X slice
    # if cfm == 'A':
    #     sl = 0.5
    # else:
    #     sl = 0.8
    c_list = []
    for sl in [0.5]:#[0.25, 0.5, 0.75]:
        y_ind = int(num_steps*sl) #0.8
        # print(y_vals[y_ind])
        counts = img_array_kcps[y_ind]
        # print(counts)
    
        conversion = fit_triple_gaus(y_vals, counts)
        c_list.append(conversion)
    
    print('average conversion: {}'.format(numpy.average(c_list)))
        
    return


#%%
#B
folder= 'pc_rabi/branch_master/image_sample/2022_01'
file_name ='2022_01_11-15_44_46-USAF51-search'
# do_plot_confocal_figure(file_name, folder, max_val = 12.6)
# plot_1D_cut_for_horizontal_lines(file_name, folder)

file_name ='2022_01_11-15_50_14-USAF51-search'
# do_plot_confocal_figure(file_name, folder, max_val = 12.6)
# plot_1D_cut_for_vertical_lines(file_name, folder, "B") 

file_name ='2022_01_11-15_44_46-USAF51-search'

#A
folder= 'pc_rabi/branch_master/image_sample_digital/2022_01'
file_name ='2022_01_20-17_14_14-usaf1951'
# do_plot_confocal_figure(file_name, folder, max_val = 51.5)
# plot_1D_cut_for_vertical_lines(file_name, folder,"A")

file_name ='2022_01_20-18_21_41-usaf1951'
# do_plot_confocal_figure(file_name, folder, max_val = 51.5)
plot_1D_cut_for_horizontal_lines(file_name, folder)


# do_plot_confocal_figure(file_name, folder)