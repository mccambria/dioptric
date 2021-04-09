# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 14:49:46 2021
File to create figures for SiV charge paper

@author: agardill
"""

import matplotlib.pyplot as plt 
import numpy
import math
import utils.tool_belt as tool_belt
import majorroutines.image_sample as image_sample
     
# %%
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

# %% data from moving target
def plot_radial_avg_moving_target(file,  pc_name, branch_name, data_folder, 
                                  sub_folder, threshold, do_plot = True, save_plot = True):
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
    folder = pc_name + '/' + branch_name + '/' + data_folder + '/' + sub_folder
    
    # Get data from the file
    data = tool_belt.get_raw_data(folder, file) 
    timestamp = data['timestamp']
    init_color = data['init_color']
    pulse_color = data['pulse_color']
    nv_sig = data['nv_sig']
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
    else:
        charge_counts_avg = data['readout_counts_avg']
    
    # create the img arrays
    readout_image_array = numpy.empty([num_steps, num_steps])
    readout_image_array[:] = numpy.nan
    writePos = []
    readout_image_array = image_sample.populate_img_array(charge_counts_avg, readout_image_array, writePos)
        
    if pulse_color == 532:
        opt_power = data['green_optical_power_mW']
        pulse_time = nv_sig['pulsed_reionization_dur']
    if pulse_color == 638:
        opt_power = data['red_optical_power_mW']
        pulse_time = nv_sig['pulsed_ionization_dur']
    
    readout = nv_sig['pulsed_SCC_readout_dur']
    
    # plot
    radii, counts_r = radial_distrbution_data(coords, x_voltages,
                              y_voltages, num_steps, img_range, readout_image_array)
    
    if do_plot:
    
        fig, ax = plt.subplots(1,1, figsize = (8, 8))
        # radii = radii[:-1]
        # counts_r = counts_r[:-1]
        ax.plot(radii, counts_r)
        ax.set_xlabel('Radial distance (um)')
        ax.set_ylabel('Azimuthal avg counts (kcps)')
        ax.set_title('Radial plot of moving target\n{} nm init pulse\n{} s {} nm pulse at {:.1f} mW'.format(init_color, 
                                                                          pulse_time/10**9, pulse_color,opt_power))
    
        if save_plot:
            # save data from this file
            rawData = {'timestamp': timestamp,
                        'init_color': init_color,
                        'pulse_color': pulse_color,
                        'pulse_time': pulse_time,
                        'pulse_time-units': 'ns',
                        'opt_power': opt_power,
                        'opt_power-units': 'mW',
                        'readout': readout,
                        'readout-units': 'ns',
                        'nv_sig': nv_sig,
                        'nv_sig-units': tool_belt.get_nv_sig_units(),
                        'num_steps': num_steps,
                        'radii': radii.tolist(),
                        'radii-units': 'um',
                        'counts_r': counts_r,
                        'counts_r-units': 'kcps'}
                
            filePath = tool_belt.get_file_path(data_folder, timestamp, nv_sig['name'])
            tool_belt.save_raw_data(rawData, filePath + '_radial_dist')
            tool_belt.save_figure(fig, filePath + '_radial_dist')
        

    return radii, counts_r

def siv_reset(dark_file, bright_file, threshold):
    nv_file_list = [dark_file, bright_file]
    pc_name = 'pc_rabi'
    branch_name = 'branch_Spin_to_charge'
    data_folder = 'moving_target_siv_init'
    sub_folder = '2021_03'
    
#    color = 'tab:blue'
    labels = ['SiV dark reset', 'SiV bright reset', ]
    fig, ax1 = plt.subplots(1,1, figsize = (10,8))#(12, 6))
    for i in [0,1]:
            file = nv_file_list[i]
            # file_9 = nv_file_list_9[i]
            radii, counts_r=plot_radial_avg_moving_target(file, pc_name, branch_name, data_folder, 
                                                          sub_folder, threshold, do_plot = False, save_plot =  False)
            
            ax1.plot(radii, counts_r, label = labels[i])
            # ax1.plot(radii_9, counts_r_9, label = '1/9')
    ax1.set_xlabel(r'Remote Pulse Position, Relative to NV ($\mu$m)')
    ax1.set_ylabel('NV- population (arb)')
    ax1.set_title('Radial plot of moving target w/ SiV reset\n50 ms 515 nm remote pulse')
    ax1.legend()
    
    return

def siv_and_nv_band(NV_file, SiV_file, threshold, title = None):
     file_list = [NV_file, SiV_file]
     pc_name = 'pc_rabi'
     branch_name = 'branch_Spin_to_charge'
     data_folder = 'moving_target'
     sub_folder = '2021_04'
    
     label_list = ['NV band', 'SiV band']
     fig, ax1 = plt.subplots(1,1, figsize = (8, 8))
    
     color = 'tab:blue'
     file = file_list[0]
     radii, counts_r=plot_radial_avg_moving_target(file, pc_name, branch_name, data_folder, 
                                                          sub_folder, threshold, do_plot = False, save_plot =  False)
     ax1.plot(radii, counts_r, color = color)
     ax1.set_xlabel(r'Remote Pulse Position, Relative to NV ($\mu$m)')
     ax1.set_ylabel('NV- population', color = color)
#     ax1.set_title('Radial plot of moving target\n532 nm init pulse\n1 ms 532 nm pulse')
     ax1.tick_params(axis = 'y', labelcolor=color)

     color = 'tab:red'
     ax2 = ax1.twinx()
     file = file_list[1]
     radii, counts_r=plot_radial_avg_moving_target(file, pc_name, branch_name, data_folder, 
                                                          sub_folder, None, do_plot = False, save_plot =  False)
     ax2.plot(radii, counts_r, color = color)
     ax2.set_xlabel(r'Remote Pulse Position, Relative to NV ($\mu$m)')
     ax2.set_ylabel('Azimuthal avg counts (kcps) SiV', color = color)
     ax2.set_title(title)
     ax2.tick_params(axis = 'y', labelcolor=color)
     
     return
# %%
if __name__ == '__main__':
#     dark_file = ''
#     bright_file = ''
#     threshold = 6
#     siv_reset(dark_file, bright_file, threshold)
     
    
    
    image_file_nv = '2021_04_06-07_59_32-goeppert-mayer-nv13_2021_04_02'
    image_file_siv = '2021_04_06-16_19_38-goeppert-mayer-nv13_2021_04_02'
    threshold = 8
    title = '10 ms (zoomed)'
#    siv_and_nv_band(image_file_nv, image_file_siv, threshold, title)
     
    # nv13_2021_04_02
    # 50 ms
#    image_file_nv = '2021_04_05-19_19_19-goeppert-mayer-nv13_2021_04_02'
#    image_file_siv = '2021_04_06-10_06_12-goeppert-mayer-nv13_2021_04_02'
    # 25 ms
#    image_file_nv = '2021_04_05-22_10_54-goeppert-mayer-nv13_2021_04_02'
#    image_file_siv = '2021_04_06-11_40_09-goeppert-mayer-nv13_2021_04_02'
    # 10 ms
#    image_file_nv = '2021_04_06-00_42_42-goeppert-mayer-nv13_2021_04_02'
#    image_file_siv = '2021_04_06-12_54_37-goeppert-mayer-nv13_2021_04_02'
    # 5 ms
#    image_file_nv = '2021_04_06-03_07_50-goeppert-mayer-nv13_2021_04_02'
#    image_file_siv = '2021_04_06-14_02_31-goeppert-mayer-nv13_2021_04_02'
    # 1 ms
#    image_file_nv = '2021_04_06-05_27_42-goeppert-mayer-nv13_2021_04_02'
#    image_file_siv = '2021_04_06-15_05_14-goeppert-mayer-nv13_2021_04_02'
    # 10 ms zoom
#    image_file_nv = '2021_04_06-07_59_32-goeppert-mayer-nv13_2021_04_02'
#    image_file_siv = '2021_04_06-16_19_38-goeppert-mayer-nv13_2021_04_02'
    
    nv_file_list = ['2021_04_06-05_27_42-goeppert-mayer-nv13_2021_04_02', '2021_04_06-03_07_50-goeppert-mayer-nv13_2021_04_02',
                    '2021_04_06-00_42_42-goeppert-mayer-nv13_2021_04_02', '2021_04_05-22_10_54-goeppert-mayer-nv13_2021_04_02',
                    '2021_04_05-19_19_19-goeppert-mayer-nv13_2021_04_02']
    siv_file_list = ['2021_04_06-15_05_14-goeppert-mayer-nv13_2021_04_02', '2021_04_06-14_02_31-goeppert-mayer-nv13_2021_04_02',
                     '2021_04_06-12_54_37-goeppert-mayer-nv13_2021_04_02', '2021_04_06-11_40_09-goeppert-mayer-nv13_2021_04_02',
                     '2021_04_06-10_06_12-goeppert-mayer-nv13_2021_04_02']
    labels = ['1 ms' , '5 ms', '10 ms', '25 ms', '50 ms']
    pc_name = 'pc_rabi'
    branch_name = 'branch_Spin_to_charge'
    data_folder = 'moving_target'
    sub_folder = '2021_04'
    
#    color = 'tab:blue'
    fig, ax1 = plt.subplots(1,1, figsize = (10,8))#(12, 6))
    for i in range(len(nv_file_list)):
            file = siv_file_list[i]
            threshold = None
            radii, counts_r=plot_radial_avg_moving_target(file, pc_name, branch_name, data_folder, 
                                                          sub_folder, threshold, do_plot = False, save_plot =  False)
            
            ax1.plot(radii, counts_r, label = labels[i])
            # ax1.plot(radii_9, counts_r_9, label = '1/9')
    ax1.set_xlabel(r'Remote Pulse Position, Relative to NV ($\mu$m)')
    ax1.set_ylabel('SiV counts (kcps)')
    ax1.set_title('Radial average of moving target measurement')
    ax1.legend()
