# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 09:35:53 2020

@author: Aedan 
"""

import matplotlib.pyplot as plt 
import numpy
import utils.tool_belt as tool_belt
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
    
    pixel_size = x_voltages[1] - x_voltages[0]
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
    while high_r <= (half_x_range + half_pixel_size):
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
        high_r = high_r + pixel_size
    
    # define the radial values as center values of pixels along x, convert to um
    # we need to subtract the center value from the x voltages
    radii = numpy.array(x_voltages[int(num_steps/2):])*35
    
    return radii, counts_r
    
# %% data from moving readout
def plot_radial_avg_moving_readout(file, parent_folder, sub_folder, do_plot = True, save_plot = True):
    '''
    Use this function to plot the azimuthal averaged counts as a function of radius 
    from the moving readout (charge ring) data. 
    
    You need to pass the daa file ending with "dif", which has it's data stored
    under "dif_img_array"

    Parameters
    ----------
    file : str
        file name, excluding the .txt
    parent_folder : str
        The file directly under nv_data. For this, it should be "image_sample".
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
    folder = parent_folder + '/' + sub_folder
    
    # Get data from the file
    data = tool_belt.get_raw_data(folder, file) 
    timestamp = data['timestamp']
    nv_sig = data['nv_sig']
    coords = nv_sig['coords']
    x_voltages = data['x_voltages']
    y_voltages = data['y_voltages']
    num_steps = data['num_steps']
    img_range= data['image_range']
    dif_img_array = numpy.array(data['dif_img_array'])
    try:
        pulse_time = data['pulse_time']
    except Exception:
        pulse_time = data['green_pulse_time']
    readout = data['readout']
    opt_volt = data['green_optical_voltage']
    opt_power = data['green_opt_power']
        
    
    # plot
    
    radii, counts_r = radial_distrbution_data(coords, x_voltages,
                              y_voltages, num_steps, img_range, dif_img_array)
    # convert ring counts to kcps
    counts_r = numpy.array(counts_r)/ 1000 / (readout / 10**9)
    
    if do_plot:
    
        fig, ax = plt.subplots(1,1, figsize = (8, 8))
        radii = radii[:-1]
        counts_r = counts_r[:-1]
        ax.plot(radii, counts_r)
        ax.set_xlabel('Radial distance (um)')
        ax.set_ylabel('Azimuthal avg counts (kcps)')
        ax.set_title('Radial plot of subtracted confocal scan\n{} s pulse at {:.1f} mW'.format(pulse_time, opt_power))
    
        if save_plot:
            # save data from this file
            rawData = {'timestamp': timestamp,
                       'nv_sig': nv_sig,
                       'nv_sig-units': tool_belt.get_nv_sig_units(),
                       'num_steps': num_steps,
                       'green_optical_voltage': opt_volt,
                       'green_optical_voltage-units': 'V',
                       'green_opt_power': opt_power,
                       'green_opt_power-units': 'mW',
                       'readout': readout,
                       'readout-units': 'ns',
                       'pulse_time': pulse_time,
                       'pulse_time-units':'s',
                       'radii': radii.tolist(),
                       'radii-units': 'um',
                       'counts_r': counts_r,
                       'counts_r-units': 'kcps'}
                
            filePath = tool_belt.get_file_path(parent_folder, timestamp, nv_sig['name'], subfolder = sub_folder)
            tool_belt.save_raw_data(rawData, filePath + '_radial_dist')
            tool_belt.save_figure(fig, filePath + '_radial_dist')
        

    return radii, counts_r
     
# %% data from moving target
def plot_radial_avg_moving_target(file, parent_folder, sub_folder, do_plot = True, save_plot = True):
    '''
    Use this function to plot the azimuthal averaged counts as a function of radius 
    from the moving target data. 
    
    You need to pass the daa file ending with "dif", which has it's data stored
    under "readout_image_array"

    Parameters
    ----------
    file : str
        file name, excluding the .txt
    parent_folder : str
        The file directly under nv_data. For this, 
        it should be "isolate_nv_charge_dynamics_moving_target".
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
    folder = parent_folder + '/' + sub_folder
    
    # Get data from the file
    data = tool_belt.get_raw_data(folder, file) 
    timestamp = data['timestamp']
    init_color = data['init_color']
    pulse_color = data['pulse_color']
    nv_sig = data['nv_sig']
    coords = numpy.array(data['start_coords']) #-[0.005,0,0]
    try:
        x_voltages = data['x_voltages_1d']
        y_voltages = data['y_voltages_1d']
    except Exception:
        coords_voltages = data['coords_voltages']
        x_voltages, y_voltages = zip(*coords_voltages)
    num_steps = data['num_steps']
    img_range= data['img_range']
    readout_image_array = numpy.array(data['readout_image_array'])
    # img_extent = data['img_extent']
    
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
    
    # convert ring counts to kcps
    counts_r = numpy.array(counts_r)/ 1000 / (readout / 10**9)
    
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
                
            filePath = tool_belt.get_file_path(parent_folder, timestamp, nv_sig['name'], subfolder = sub_folder)
            tool_belt.save_raw_data(rawData, filePath + '_radial_dist')
            tool_belt.save_figure(fig, filePath + '_radial_dist')
        

    return radii, counts_r
       

def plot_moving_target_1D_line(file_list, pulse_time_list):
    folder = 'moving_target/branch_Spin_to_charge/2020_12'
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    for i in range(len(file_list)):
        file = file_list[i]
        data = tool_belt.get_raw_data(folder, file)
        readout_counts_avg = data['readout_counts_avg']
        rad_dist = numpy.array(data['rad_dist'])
        init_color = data['init_color']
        pulse_color = data['pulse_color']
        pulse_length = pulse_time_list[i]
        nv_sig = data['nv_sig']
        color_filter = nv_sig['color_filter']
        
        ax.plot(rad_dist*35,readout_counts_avg, label = '{} pulse length'.format(pulse_length))
        
        
    ax.set_xlabel('Distance from readout NV (um)')
    ax.set_ylabel('Average counts')
    ax.set_title('1D moving target data ({} init, {} pulse) {} filter'.\
                                    format(init_color, pulse_color, color_filter))
    ax.legend()
    return

def plot_moving_target_1D_line_2_data(file_list, pulse_length):
    folder = 'moving_target/branch_Spin_to_charge/2020_12'
    
    fig, ax1 = plt.subplots(1, 1, figsize=(10, 10))
    
    # NV
    color = 'tab:blue'
    file = file_list[0]
    data = tool_belt.get_raw_data(folder, file)
    readout_counts_avg = data['readout_counts_avg']
    rad_dist = numpy.array(data['rad_dist'])
    init_color = data['init_color']
    pulse_color = data['pulse_color']
    nv_sig = data['nv_sig']
    color_filter = nv_sig['color_filter']
    
    ax1.plot(rad_dist*35,readout_counts_avg, color = color)
    ax1.set_xlabel('Distance from readout NV (um)')
    ax1.set_ylabel('Average counts ({} filter)'.format(color_filter), color = color)
    ax1.tick_params(axis = 'y', labelcolor=color)
    ax1.set_ylim([1,13])
    # ax1.set_xlim([-1,53])
    ax1.set_title('1D moving target data ({} init, {} {} pulse)'.\
                                    format(init_color, pulse_length, pulse_color))
    
    # SiV
    color = 'tab:red'
    file = file_list[1]
    data = tool_belt.get_raw_data(folder, file)
    readout_counts_avg = data['readout_counts_avg']
    rad_dist = numpy.array(data['rad_dist'])
    init_color = data['init_color']
    pulse_color = data['pulse_color']
    nv_sig = data['nv_sig']
    color_filter = nv_sig['color_filter']
    
    ax2 = ax1.twinx()
    ax2.plot(rad_dist*35,readout_counts_avg, color = color)
    ax2.set_ylabel('Average counts ({} filter)'.format(color_filter), color = color)
    ax2.tick_params(axis = 'y', labelcolor=color)
    ax2.set_ylim([2,62])
    
    return
    
# %% 
if __name__ == '__main__':
    
    # file = '2020_12_08-18_04_02-goeppert-mayer-nv1_2020_12_02-img'
    # file_list = ['2020_11_27-13_50_28-johnson-nv18_2020_11_10-img', # NV in johnson
    #              '2020_12_07-20_35_40-goeppert-mayer-nv1_2020_12_02-img'] # NV in goeppert mayer
    # # file = '2020_12_08-03_01_48-goeppert-mayer-nv1_2020_12_02-img'
    # sub_folder_list = ['branch_Spin_to_charge/2020_11', 'branch_Spin_to_charge/2020_12']
    
    # label_list = ['E6 sample (Johnson)', 'Indian sample (Goeppert Mayer)']
    # fig, ax = plt.subplots(1,1, figsize = (8, 8))
    
    # for i in range(len(file_list)):
    #     file = file_list[i]
    #     sub_folder = sub_folder_list[i]
    #     radii, counts_r=plot_radial_avg_moving_target(file, 'isolate_nv_charge_dynamics_moving_target', sub_folder, save_plot =  False)

    #     ax.plot(radii, counts_r, label = label_list[i])
    #     ax.set_xlabel('Radial distance (um)')
    #     ax.set_ylabel('Azimuthal avg counts (kcps)')
    # ax.set_title('Radial plot of moving target\n532 nm init pulse\n0.01 s 638 nm pulse')#' at 3 mW')
    # ax.legend()
    
    # filePath = tool_belt.get_file_path('isolate_nv_charge_dynamics_moving_target', timestamp, nv_sig['name'], subfolder = sub_folder)
    # tool_belt.save_figure(fig, filePath + '_radial_dist_comp')
  
    master_pulse_time_list = [
                        # '1 us', '10 us', '100 us', 
                        # '1 ms', '10 ms', '100 ms', 
                        # '1 s', '10 s', 
                        '100 s', 
                        ]   
                       
    # g/g NV band
    # file_list = ['2020_12_11-17_42_22-goeppert-mayer-nv2_2020_12_10',
    #              '2020_12_11-17_45_26-goeppert-mayer-nv2_2020_12_10',
    #              '2020_12_11-17_48_33-goeppert-mayer-nv2_2020_12_10',
    #              '2020_12_11-17_51_43-goeppert-mayer-nv2_2020_12_10',
    #              '2020_12_11-17_55_08-goeppert-mayer-nv2_2020_12_10',
    #              '2020_12_11-18_01_04-goeppert-mayer-nv2_2020_12_10',
    #              '2020_12_11-13_40_46-goeppert-mayer-nv2_2020_12_10',
    #              '2020_12_11-22_42_38-goeppert-mayer-nv2_2020_12_10',
    #              '2020_12_11-10_08_11-goeppert-mayer-nv2_2020_12_10',]    
    
    # pulse_time_list = ['1 us', '10 us', '100 us', 
    #                    '1 ms', '10 ms', '100 ms', 
    #                    '1 s', '10 s', '100 s', 
                       # ]         
    # g/g SiV band
    # file_list = ['2020_12_11-17_43_54-goeppert-mayer-nv2_2020_12_10',
    #               '2020_12_11-17_46_59-goeppert-mayer-nv2_2020_12_10',
    #               '2020_12_11-17_50_06-goeppert-mayer-nv2_2020_12_10',
    #               '2020_12_11-17_53_18-goeppert-mayer-nv2_2020_12_10',
    #               '2020_12_11-17_56_57-goeppert-mayer-nv2_2020_12_10',
    #               '2020_12_11-18_05_12-goeppert-mayer-nv2_2020_12_10',
    #               '2020_12_14-15_57_55-goeppert-mayer-nv2_2020_12_10',
    #               '2020_12_12-03_20_56-goeppert-mayer-nv2_2020_12_10',
    #                 '2020_12_15-07_26_11-goeppert-mayer-nv2_2020_12_10'
    #         ]

    # pulse_time_list = ['1 us', '10 us', '100 us', 
    #                     '1 ms', '10 ms', '100 ms', 
    #                     '1 s', '10 s', '100 s'
    #                     ]     

    master_file_list = [
                        # ['2020_12_11-17_42_22-goeppert-mayer-nv2_2020_12_10', 
                        #   '2020_12_11-17_43_54-goeppert-mayer-nv2_2020_12_10'] ,
                        # ['2020_12_11-17_45_26-goeppert-mayer-nv2_2020_12_10',
                        #   '2020_12_11-17_46_59-goeppert-mayer-nv2_2020_12_10'],
                        # ['2020_12_11-17_48_33-goeppert-mayer-nv2_2020_12_10',
                        #   '2020_12_11-17_50_06-goeppert-mayer-nv2_2020_12_10'],
                        # ['2020_12_11-17_51_43-goeppert-mayer-nv2_2020_12_10',
                        #   '2020_12_11-17_53_18-goeppert-mayer-nv2_2020_12_10'],
                        # ['2020_12_11-17_55_08-goeppert-mayer-nv2_2020_12_10',
                        #   '2020_12_11-17_56_57-goeppert-mayer-nv2_2020_12_10'],
                        # ['2020_12_11-18_01_04-goeppert-mayer-nv2_2020_12_10',
                        #   '2020_12_11-18_05_12-goeppert-mayer-nv2_2020_12_10'],
                        # ['2020_12_11-13_40_46-goeppert-mayer-nv2_2020_12_10',
                        #   '2020_12_14-15_57_55-goeppert-mayer-nv2_2020_12_10'],
                        # ['2020_12_11-22_42_38-goeppert-mayer-nv2_2020_12_10',
                        #   '2020_12_12-03_20_56-goeppert-mayer-nv2_2020_12_10'],
                        ['2020_12_11-10_08_11-goeppert-mayer-nv2_2020_12_10',
                          '2020_12_15-07_26_11-goeppert-mayer-nv2_2020_12_10']
                        ]
    
    # r/g NV band
    # file_list = ['2020_12_12-03_22_30-goeppert-mayer-nv2_2020_12_10',
    #              '2020_12_12-03_25_40-goeppert-mayer-nv2_2020_12_10',
    #              '2020_12_12-03_28_50-goeppert-mayer-nv2_2020_12_10',
    #              '2020_12_12-03_32_05-goeppert-mayer-nv2_2020_12_10',
    #              '2020_12_12-03_35_34-goeppert-mayer-nv2_2020_12_10',
    #              '2020_12_12-03_41_34-goeppert-mayer-nv2_2020_12_10',
    #              '2020_12_11-16_09_52-goeppert-mayer-nv2_2020_12_10',
    #              '2020_12_12-08_22_16-goeppert-mayer-nv2_2020_12_10',
    #              '2020_12_13-10_04_13-goeppert-mayer-nv2_2020_12_10',
    #     ]
    # pulse_time_list = ['1 us', '10 us', '100 us', 
    #                     '1 ms', '10 ms', '100 ms', 
    #                     '1 s', '10 s', '100 s', 
    #                     ]    
    
    # r/g SiV band
    # file_list = ['2020_12_12-03_24_06-goeppert-mayer-nv2_2020_12_10',
    #               '2020_12_12-03_27_15-goeppert-mayer-nv2_2020_12_10',
    #               '2020_12_12-03_30_27-goeppert-mayer-nv2_2020_12_10',
    #               '2020_12_12-03_33_43-goeppert-mayer-nv2_2020_12_10',
    #               '2020_12_12-03_37_26-goeppert-mayer-nv2_2020_12_10',
    #               '2020_12_12-03_45_42-goeppert-mayer-nv2_2020_12_10',
    #               '2020_12_14-16_26_34-goeppert-mayer-nv2_2020_12_10',
    #               '2020_12_12-13_00_36-goeppert-mayer-nv2_2020_12_10',
    #               '2020_12_14-13_01_52-goeppert-mayer-nv2_2020_12_10'
    #     ]
    # pulse_time_list = ['1 us', '10 us', '100 us', 
    #                     '1 ms', '10 ms', '100 ms', 
    #                     '1 s', '10 s', '100 s', 
    #                     ]    
    # master_file_list = [['2020_12_12-03_22_30-goeppert-mayer-nv2_2020_12_10',
    #                       '2020_12_12-03_24_06-goeppert-mayer-nv2_2020_12_10'],
    #                     ['2020_12_12-03_25_40-goeppert-mayer-nv2_2020_12_10',
    #                       '2020_12_12-03_27_15-goeppert-mayer-nv2_2020_12_10'],
    #                     ['2020_12_12-03_28_50-goeppert-mayer-nv2_2020_12_10',
    #                       '2020_12_12-03_30_27-goeppert-mayer-nv2_2020_12_10'],
    #                     ['2020_12_12-03_32_05-goeppert-mayer-nv2_2020_12_10',
    #                       '2020_12_12-03_33_43-goeppert-mayer-nv2_2020_12_10'],
    #                     ['2020_12_12-03_35_34-goeppert-mayer-nv2_2020_12_10',
    #                       '2020_12_12-03_37_26-goeppert-mayer-nv2_2020_12_10'],
    #                     ['2020_12_12-03_41_34-goeppert-mayer-nv2_2020_12_10',
    #                       '2020_12_12-03_45_42-goeppert-mayer-nv2_2020_12_10'],
    #                     ['2020_12_11-16_09_52-goeppert-mayer-nv2_2020_12_10',
    #                       '2020_12_14-16_26_34-goeppert-mayer-nv2_2020_12_10'],
    #                     ['2020_12_12-08_22_16-goeppert-mayer-nv2_2020_12_10',
    #                       '2020_12_12-13_00_36-goeppert-mayer-nv2_2020_12_10'],
    #                     ['2020_12_13-10_04_13-goeppert-mayer-nv2_2020_12_10',
    #                       '2020_12_14-13_01_52-goeppert-mayer-nv2_2020_12_10']
    #                 ]
    
    for i in range(len(master_file_list)):
        file_list = master_file_list[i]
        pulse_time = master_pulse_time_list[i]
    
        plot_moving_target_1D_line_2_data(file_list, pulse_time)         
    

    
