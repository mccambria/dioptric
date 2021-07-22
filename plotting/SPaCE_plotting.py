# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 14:56:11 2021

@author: agardill
"""

import matplotlib.pyplot as plt 
import numpy
import utils.tool_belt as tool_belt
import majorroutines.image_sample as image_sample
import math
import fit_gaussians_function
import csv

# siv_min = 441
# siv_max = 3124
# If I based off 25 ms radial avg: 
siv_min = 450
siv_max = 2700


def create_image_figure(imgArray, imgExtent, band):
    f_size = 8
    tick_f_size = 8
    clb_f_size = 8
    fig_w = 2.15
    fig_l = fig_w * 0.75
    
    mu = u"\u03BC" 
    
    fig_tick_l = 3
    fig_tick_w = 0.75
    
    clb_tick_1 = 3
    clb_tick_w = 0.75

    
    # Tell matplotlib to generate a figure with just one plot in it
    fig, ax = plt.subplots()
    fig.set_figwidth(fig_w)
    fig.set_figheight(fig_l)
    
    ax.set_xticks([-10,-5, 0, 5,10])
    ax.set_yticks([-10,-5, 0, 5,10])
    ax.tick_params(which = 'both', length=fig_tick_l, width=fig_tick_w, colors='k',
                    direction='in',grid_alpha=0.7, labelsize = tick_f_size)

        
    # Tell the axes to show an image
    if band == 'nv':
        img = ax.imshow(imgArray, cmap='inferno', vmin = 0, vmax = 1,
                    extent=tuple(imgExtent))
    else:
        img = ax.imshow(imgArray, cmap='inferno', vmin = 0.4, vmax = 3.2,
                    extent=tuple(imgExtent))

    # Add a colorbar
    clb = plt.colorbar(img)
    clb.set_label("", rotation=270, fontsize = f_size)
    clb.ax.tick_params( length=clb_tick_1, width=clb_tick_w, grid_alpha=0.7, labelsize = clb_f_size)
    if band == 'nv':
        clb.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    # if band == 'siv':
    #     clb.set_ticks([2, 4, 6, 8, 10])
    
    
    # Label axes
    ax.set_xlabel('x ({}m)'.format(mu), fontsize = f_size)
    ax.set_ylabel('y ({}m)'.format(mu), fontsize = f_size)
        


    # Draw the canvas and flush the events to the backend
    fig.canvas.draw()
    plt.tight_layout()
    fig.canvas.flush_events()


    return fig

# %%
def create_figure(file_name, threshold, title, band, sub_folder = None):
    data = tool_belt.get_raw_data('', file_name)
    x_range = data['img_range']
    y_range = data['img_range']
    x_voltages = data['x_voltages_1d']
    num_steps = data['num_steps']
    coords = [0,0,5.0]
    timestamp = data['timestamp']
    nv_sig = data['nv_sig']
    

    if band == 'nv':
        raw_counts = numpy.array(data['readout_counts_array'])
        # charge state information
        cut_off = threshold
    
        # for each individual measurement, determine if the NV was in NV0 or NV- by threshold.
        # Then average the measurements for each pixel to gain mean charge state.
        for r in range(len(raw_counts)):
            row = raw_counts[r]
            for c in range(len(row)):
                current_val = raw_counts[r][c]
                if current_val < cut_off:
                    set_val = 0
                elif current_val >= cut_off:
                    set_val = 1
                raw_counts[r][c] = set_val
            
            charge_counts_avg = numpy.average(raw_counts, axis = 1)
    
    elif band == 'siv':
        readout_counts_avg = data['readout_counts_avg']
        charge_counts_avg = numpy.array(readout_counts_avg)/1000
        
    
    print(charge_counts_avg)
    
    # create the img arrays
    readout_image_array = numpy.empty([num_steps, num_steps])
    readout_image_array[:] = numpy.nan
    writePos = []
    img_array = image_sample.populate_img_array(charge_counts_avg, readout_image_array, writePos)
    
    x_coord = coords[0]
    half_x_range = x_range / 2
    x_low = x_coord - half_x_range
    x_high = x_coord + half_x_range
    y_coord = coords[1]
    half_y_range = y_range / 2
    y_low = y_coord - half_y_range
    y_high = y_coord + half_y_range

    pixel_size = x_voltages[1] - x_voltages[0]
    half_pixel_size = pixel_size / 2
    img_extent = [(x_low - half_pixel_size)*35, (x_high + half_pixel_size)*35,
                  (y_low - half_pixel_size)*35, (y_high + half_pixel_size)*35]

    ### SAVE CSV
    csv_data = []
    for ind in range(len(charge_counts_avg)):
        row = []
        row.append(charge_counts_avg[ind])
        csv_data.append(row)

    csv_file_name = '{}_10ms'.format(band)        
    file_path = 'E:/Shared drives/Kolkowitz Lab Group/nvdata/paper_materials/charge siv/Figures/Fig 1 - E6 and SiV comparison/'


    with open('{}/{}.csv'.format(file_path, csv_file_name),
              'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',',
                                quoting=csv.QUOTE_NONE)
        csv_writer.writerows(csv_data)            


    # file_path = 'E:/Shared drives/Kolkowitz Lab Group/nvdata/paper_materials/charge siv/Figures/Supp_Mater/'
    # file_name = timestamp + '-' + nv_sig['name']
    # print(file_path)
    # tool_belt.save_raw_data(raw_data, file_path + file_name +'-radial')

    ###


    fig = create_image_figure(img_array, img_extent, band)
    # Redraw the canvas and flush the changes to the backend
    fig.canvas.draw()
    fig.canvas.flush_events()
        
    file_path = 'E:/Shared drives/Kolkowitz Lab Group/nvdata/paper_materials/charge siv/Figures/Fig 2 - vary remote pusle time/'
    file_name = timestamp + '-' + nv_sig['name']
    tool_belt.save_figure(fig, file_path + file_name +'-charge')
    

    return fig  

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
    radii = numpy.array(x_voltages[(math.ceil(num_steps/2)-1):])*35
    
    return radii, counts_r

def plot_radial_avg_moving_target(file,  folder, threshold, band):
    '''
    Use this function to plot the azimuthal averaged counts as a function of radius 
    from the moving target data. 
    
    You need to pass the daa file ending with "dif", which has it's data stored
    under "readout_image_array"

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
    
    # Get data from the file
    data = tool_belt.get_raw_data(folder, file) 
    coords = numpy.array(data['start_coords'])
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
    
    if band == 'nv':
        raw_counts = numpy.array(data['readout_counts_array'])
        # charge state information
        if threshold:
            cut_off = threshold
        
            # for each individual measurement, determine if the NV was in NV0 or NV- by threshold.
            # Then average the measurements for each pixel to gain mean charge state.
            for r in range(len(raw_counts)):
                row = raw_counts[r]
                for c in range(len(row)):
                    current_val = raw_counts[r][c]
                    if current_val < cut_off:
                        set_val = 0
                    elif current_val >= cut_off:
                        set_val = 1
                    raw_counts[r][c] = set_val
            charge_counts_avg = numpy.average(raw_counts, axis = 1)
    elif band == 'siv':
        readout_counts_avg = data['readout_counts_avg']
        charge_counts_avg = numpy.array(readout_counts_avg)/1000
    
    # create the img arrays
    readout_image_array = numpy.empty([num_steps, num_steps])
    readout_image_array[:] = numpy.nan
    writePos = []
    readout_image_array = image_sample.populate_img_array(charge_counts_avg, readout_image_array, writePos)
    
    # plot
    radii, counts_r = radial_distrbution_data(coords, x_voltages,
                              y_voltages, num_steps, img_range, readout_image_array)
    

        

    return radii, counts_r

def radial_avg_nv_siv_plot(nv_file, siv_file, threshold, folder):
    data = tool_belt.get_raw_data('', folder + '/' + nv_file)
    timestamp = data['timestamp']
    nv_sig = data['nv_sig']
    
    # f_size = 8
    tick_f_size = 8
    label_f_size = 6
    fig_w = 0.9
    fig_l = 1.15
    
    # mu = u"\u03BC" 
    
    fig_tick_l = 3
    fig_tick_w = 0.75

    
    color_blue = '#2278b5'
    color_red = '#c82127'
    
    
    
    radii, counts_r_nv = plot_radial_avg_moving_target(nv_file,  folder, threshold, 'nv') 
    params_nv, cov_arr_nv = fit_gaussians_function.fit_double_gaussian_nv(radii, counts_r_nv)
    print(params_nv)
    radii, counts_r_siv = plot_radial_avg_moving_target(siv_file,  folder, threshold, 'siv') 
    params_siv, cov_arr_siv = fit_gaussians_function.fit_double_gaussian_siv(radii, counts_r_siv)
    print(params_siv)
    
    lin_r = numpy.linspace(radii[0], radii[-1], 100)
    
    superscript_minus = u"\u207B" 
#    superscript_0 = u"\u2070" 
    # nv_label = "NV" + superscript_minus + " Population"
    # siv_label = "Norm. SiV" + superscript_minus + " Population"
    # y_label = 'Norm. Defect Pop.'
    
    fig, ax1 = plt.subplots()
    fig.set_figwidth(fig_w)
    fig.set_figheight(fig_l)
    
    ax1.plot(radii, counts_r_nv, 'bo', color = color_blue, markersize = 1.5,  
             label = "NV data")
    ax1.plot(lin_r, fit_gaussians_function.double_gaussian_nv(lin_r, *params_nv), 
             color = color_blue, linestyle = 'solid' , linewidth = 0.75, label = 'Double gaussian fit')
    
    # ax1.plot(lin_r, fit_gaussians_function.positive_gaussian_nv(lin_r, *params_nv[0:2]),  color = 'black', linestyle = 'dashed' ,linewidth = 0.5, label = 'Ionization dominated')
    # ax1.plot(lin_r, fit_gaussians_function.negative_gaussian_nv(lin_r, *params_nv[2:4]),  color = 'red', linestyle = 'dashdot' , linewidth = 0.5, label = 'Diffusion dominated')
    
    # ax1.set_xlabel('r ({}m)'.format(mu), fontsize = f_size)
    # ax1.set_ylabel(y_label,  fontsize = f_size)
    ax1.set_ylim([0.1,0.9])
    ax1.tick_params(which = 'both', length=fig_tick_l, width=fig_tick_w,
                    direction='in',grid_alpha=0.7, labelsize = tick_f_size)
    ax1.set_xticks([0, 5, 10])
    ax1.set_yticks([0.2, 0.4, 0.6, 0.8])
    
    ax2 = ax1.twinx()
    ax2.plot(radii, counts_r_siv, 'ro', color = color_red, markersize = 1.5, 
             label = "SiV" + superscript_minus)
    ax2.plot(lin_r, fit_gaussians_function.double_gaussian_siv(lin_r, *params_siv), 
             color = color_red, linestyle = 'solid' , linewidth = 0.75, label = 'double gaussian fit')
    
    # ax2.plot(lin_r, fit_gaussians_function.positive_gaussian_siv(lin_r, *params_siv[0:3]), 'g--', label = 'single gaussian from fit')
    # ax2.plot(lin_r, fit_gaussians_function.negative_gaussian_siv(lin_r, *params_siv[3:6]), 'k--', label = 'single gaussian from fit')
    
    ax2.set_ylim([0.2,2.6])
    ax2.tick_params(which = 'both', length=fig_tick_l, width=fig_tick_w,
                    direction='in',grid_alpha=0.7, labelsize = tick_f_size)
    # ax1.legend(fontsize = label_f_size)
    # ax2.legend(fontsize = label_f_size)
    
    file_path = 'E:/Shared drives/Kolkowitz Lab Group/nvdata/paper_materials/charge siv/Figures/Fig 2 - vary remote pusle time/'
    file_name = timestamp + '-' + nv_sig['name']
    tool_belt.save_figure(fig, file_path + file_name +'-radial_avg')
    
    
        

# %%
if __name__ == '__main__':
    
    folder = 'pc_rabi/branch_Spin_to_charge/moving_target/2021_04'
    threshold = 8
    
    # nv13_2021_04_02
    title = '25 ms'
    image_file_nv_25 = '2021_04_05-22_10_54-goeppert-mayer-nv13_2021_04_02'
    # create_figure(folder + '/' + image_file_nv_25, threshold, title, 'nv')
    image_file_siv_25 = '2021_04_06-11_40_09-goeppert-mayer-nv13_2021_04_02'
    # create_figure(folder + '/' + image_file_siv_25, threshold, title, 'siv')
    title = '10 ms'
    image_file_nv_10 ='2021_04_06-00_42_42-goeppert-mayer-nv13_2021_04_02'
    create_figure(folder + '/' + image_file_nv_10, threshold, title, 'nv')
    image_file_siv_10 = '2021_04_06-12_54_37-goeppert-mayer-nv13_2021_04_02'
    # create_figure(folder + '/' + image_file_siv_10, threshold, title, 'siv')
    title = '1 ms'
    image_file_nv_1 = '2021_04_06-05_27_42-goeppert-mayer-nv13_2021_04_02'
    # create_figure(folder + '/' + image_file_nv_1, threshold, title, 'nv')
    image_file_siv_1 = '2021_04_06-15_05_14-goeppert-mayer-nv13_2021_04_02'
    # create_figure(folder + '/' + image_file_siv_1, threshold, title, 'siv')
    
    # radial_avg_nv_siv_plot(image_file_nv_25, image_file_siv_25,threshold, folder)
    # radial_avg_nv_siv_plot(image_file_nv_10, image_file_siv_10, threshold,folder)
    # radial_avg_nv_siv_plot(image_file_nv_1, image_file_siv_1, threshold,folder)
    