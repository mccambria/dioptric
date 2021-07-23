# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 14:56:11 2021

@author: agardill
"""

import matplotlib.pyplot as plt 
import numpy
import utils.tool_belt as tool_belt
import majorroutines.image_sample as image_sample
import csv

# siv_min = 441
# siv_max = 3124
# If I based off 25 ms radial avg: 
siv_min = 450
siv_max = 2700

# %%
def create_image_figure(imgArray, imgExtent, threshold = None):    
    mu = u"\u03BC" 

    
    # Tell matplotlib to generate a figure with just one plot in it
    fig, ax = plt.subplots()

        
    # Tell the axes to show an image
    if threshold:
        img = ax.imshow(imgArray, cmap='inferno', vmin = 0, vmax = 1,
                    extent=tuple(imgExtent))
        color_bar_label = 'NV- state prob.'
    else:
        img = ax.imshow(imgArray, cmap='inferno',
                    extent=tuple(imgExtent))
        color_bar_label = 'kcps'

    # Add a colorbar
    clb = plt.colorbar(img)
    clb.set_label("", rotation=270)
    clb.set_label(color_bar_label)
    
    
    # Label axes
    ax.set_xlabel('x ({}m)'.format(mu))
    ax.set_ylabel('y ({}m)'.format(mu))

    # Draw the canvas and flush the events to the backend
    fig.canvas.draw()
    plt.tight_layout()
    fig.canvas.flush_events()

    return fig
     
# %% data from moving target
def do_plot_SPaCE_radial_avg(file,  pc_name, branch_name, data_folder, 
                                  sub_folder, do_plot = True, save_plot = True):
    '''
    Use this function to plot the azimuthal averaged counts as a function of radius 
    from the SPaCE data
    
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
    folder = pc_name + '/' + branch_name + '/' + data_folder + '/' + sub_folder
    
    # Get data from the file
    data = tool_belt.get_raw_data(file, folder) 
    timestamp = data['timestamp']
    nv_sig = data['nv_sig']
    initialize_laser = nv_sig['initialize_laser']
    CPG_laser = nv_sig['CPG_laser']
    coords = numpy.array(nv_sig['coords'])
    x_voltages = data['x_voltages_1d']
    y_voltages = data['y_voltages_1d']
    num_steps = data['num_steps']
    img_range= data['img_range']
    readout_image_array = numpy.array(data['readout_image_array'])
    
    # if pulse_color == 532:
    #     opt_power = data['green_optical_power_mW']
    #     pulse_time = nv_sig['pulsed_reionization_dur']
    # if pulse_color == 638:
    #     opt_power = data['red_optical_power_mW']
    #     pulse_time = nv_sig['pulsed_ionization_dur']
    
    readout = nv_sig['charge_readout_dur']
    
    # plot
    radii, counts_r = tool_belt.radial_distrbution_data(coords, x_voltages,
                              y_voltages, num_steps, img_range, readout_image_array)
    
    # convert ring counts to kcps
    counts_r = numpy.array(counts_r)/ 1000 / (readout / 10**9)
    
    if do_plot:
    
        fig, ax = plt.subplots(1,1, figsize = (8, 8))
        # radii = radii[:-1]
        # counts_r = counts_r[:-1]
        ax.plot(radii, counts_r)
        ax.set_xlabel('Radial distance (um)')
        ax.set_ylabel('Azimuthal avg counts (kcps)')
        # ax.set_title('Radial plot of moving target\n{} nm init pulse\n{} s {} nm pulse at {:.1f} mW'.format(init_color, 
                                                                          # pulse_time/10**9, pulse_color,opt_power))
    
        if save_plot:
            # save data from this file
            # rawData = {'timestamp': timestamp,
            #             'init_color': init_color,
            #             'pulse_color': pulse_color,
            #             'pulse_time': pulse_time,
            #             'pulse_time-units': 'ns',
            #             'opt_power': opt_power,
            #             'opt_power-units': 'mW',
            #             'readout': readout,
            #             'readout-units': 'ns',
            #             'nv_sig': nv_sig,
            #             'nv_sig-units': tool_belt.get_nv_sig_units(),
            #             'num_steps': num_steps,
            #             'radii': radii.tolist(),
            #             'radii-units': 'um',
            #             'counts_r': counts_r,
            #             'counts_r-units': 'kcps'}
                
            filePath = tool_belt.get_file_path(data_folder, timestamp, nv_sig['name'])
            # tool_belt.save_raw_data(rawData, filePath + '_radial_dist')
            tool_belt.save_figure(fig, filePath + '_radial_dist')
        

    return radii, counts_r

# %%
def do_plot_SPaCE_image(pc_name, branch_name, data_folder, sub_folder, file, 
                  threshold = None, save_csv = False):
    
    folder_name = pc_name + '/' + branch_name + '/' + data_folder + '/' + sub_folder
    
    data = tool_belt.get_raw_data(file, folder_name)
    x_range = data['img_range']
    y_range = data['img_range']
    x_voltages = data['x_voltages_1d']
    num_steps = data['num_steps']
    coords = [0,0,5.0]
    timestamp = data['timestamp']
    nv_sig = data['nv_sig']
    readout_dur = nv_sig['charge_readout_dur']
    
    
    

    raw_counts = numpy.array(data['readout_counts_array'])
    # charge state information
    if threshold:
    
        # for each individual measurement, determine if the NV was in NV0 or NV- by threshold.
        # Then average the measurements for each pixel to gain mean charge state.
        for r in range(len(raw_counts)):
            row = raw_counts[r]
            for c in range(len(row)):
                current_val = raw_counts[r][c]
                if current_val < threshold:
                    set_val = 0
                elif current_val >= threshold:
                    set_val = 1
                raw_counts[r][c] = set_val
            
            charge_counts_avg = numpy.average(raw_counts, axis = 1)
    else:
        # kcps
        readout_dur_s = readout_dur * 10**-9
        charge_counts = raw_counts / 1000 / readout_dur_s
        charge_counts_avg = numpy.average(charge_counts, axis = 1)
        
        # only plot first run
        charge_counts_avg = []
        for r in charge_counts:
            charge_counts_avg.append(r[0])
    
    
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

    ############################# SAVE CSV #############################
    if save_csv:
        csv_data = []
        for ind in range(len(charge_counts_avg)):
            row = []
            row.append(charge_counts_avg[ind])
            csv_data.append(row)
    
        csv_file_name = '{}_{}_ns'.format(timestamp, readout_dur) 
        file_path = tool_belt.get_file_path(data_folder, timestamp, nv_sig['name'])
    
        with open('{}/{}.csv'.format(file_path, csv_file_name),
                  'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',',
                                    quoting=csv.QUOTE_NONE)
            csv_writer.writerows(csv_data)            


    fig = create_image_figure(img_array, img_extent, threshold)
    # Redraw the canvas and flush the changes to the backend
    fig.canvas.draw()
    fig.canvas.flush_events()
        
    
    file_path = tool_belt.get_file_path(data_folder, timestamp, nv_sig['name'])
    file_name = timestamp + '-' + nv_sig['name']
    tool_belt.save_figure(fig, file_path + file_name +'-replot')
    

    return fig  

        

# %%
if __name__ == '__main__':
    
    pc_name = 'pc_rabi'
    branch_name = 'branch_master'
    data_folder = 'SPaCE'
    sub_folder = '2021_07'
    
    file = '2021_07_23-09_32_56-johnson-nv1_2021_07_21'
    
    threshold = 20
    # do_plot_SPaCE_radial_avg(file,  pc_name, branch_name, data_folder, 
    #                               sub_folder)
    
    do_plot_SPaCE_image(pc_name, branch_name, data_folder, sub_folder, file)
                                                            # , threshold)
    
    # Plot radial average of files together:
        
    # 33 mW
    file_33_1m = '2021_07_23-01_14_21-johnson-nv1_2021_07_21' # 1 ms
    file_33_500u = '2021_07_23-09_32_56-johnson-nv1_2021_07_21' # 500 us
    file_33_100u = '2021_07_23-07_28_07-johnson-nv1_2021_07_21' # 100 us
    file_33_50u = '2021_07_23-05_23_30-johnson-nv1_2021_07_21' # 50 us
    file_33_10u = '2021_07_23-03_18_55-johnson-nv1_2021_07_21' # 10 us
    
    # 22 mW
    file_22_100 = '2021_07_22-01_05_19-johnson-nv1_2021_07_21' # 100 us
    file_22_50 = '2021_07_22-06_54_02-johnson-nv1_2021_07_21' # 50 us
    file_22_10 = '2021_07_21-23_18_30-johnson-nv1_2021_07_21' # 10 us
    file_22_5 = '2021_07_22-05_07_14-johnson-nv1_2021_07_21' # 5 us
    file_22_1 = '2021_07_21-21_31_43-johnson-nv1_2021_07_21' # 1 us

    file_list = [file_33_10u, file_33_50u, file_33_100u, file_33_500u,file_33_1m ]
    title = '33 mW, 638 nm CPG pulse'
    
    label_list = ['10 us', '50 us', '100 us', '500 us', '1 ms']
    
    # Start plotting
    fig, ax = plt.subplots(1,1, figsize = (8, 8))
    for f in range(len(file_list)):
        file = file_list[f]
        radii, counts_r= do_plot_SPaCE_radial_avg(file,  pc_name, branch_name, 
                                                  data_folder, sub_folder,
                                  do_plot = False, save_plot = False)
        # Been having trouble with these lists lining up... should fix that
        if len(radii) == len(counts_r) + 1:
            radii = radii[:-1]
        elif len(radii) == len(counts_r) - 1:
            counts_r = counts_r[:-1]
            
        ax.plot(radii, counts_r, label = label_list[f])
        ax.set_xlabel(r'r ($\mu$m)')
        ax.set_ylabel('kcps')
        ax.set_title(title)
        ax.legend()
            
        
    
    
    