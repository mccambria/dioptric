# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 09:05:07 2020

@author: agardill
"""

import matplotlib.pyplot as plt 
import numpy
import utils.tool_belt as tool_belt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy import integrate

# %%

def gaussian(r, constrast, sigma, center, offset):
    return offset+ constrast * numpy.exp(-((r-center)**2) / (2 * (sigma**2)))

def double_gaussian_dip(freq, low_constrast, low_sigma, low_center, low_offset,
                        high_constrast, high_sigma, high_center, high_offset):
    low_gauss = gaussian(freq, low_constrast, low_sigma, low_center, low_offset)
    high_gauss = gaussian(freq, high_constrast, high_sigma, high_center, high_offset)
    return low_gauss + high_gauss

def exponential(x, a, d, c):
    return c+a*(1-numpy.exp(-x/d))

def exponential_power(x, a, d, c, f):
    return c+a*(1-numpy.exp(-(x/d)**f))

def power_law(x, c, a, p):
    return c - a/x**p

def sqrt(x, a):
    return a*x**0.5


# %%
    
def r_vs_power_plot(nv_sig, ring_radius_list, ring_err_list, power_list, 
                    power_err_list, sub_folder, 
                    img_range, num_steps, green_pulse_time, readout):
    power_fig, ax = plt.subplots(1,1, figsize = (8, 8))
    ax.errorbar(power_list, ring_radius_list, xerr = power_err_list, yerr = ring_err_list, fmt = 'o')
    ax.set_xlabel('Green optical power (mW)')
    ax.set_ylabel('Charge ring radius (um)')
    ax.legend()
            
    timestamp = tool_belt.get_time_stamp()
    
    # save this dile 
    rawData = {'timestamp': timestamp,
               'nv_sig': nv_sig,
               'nv_sig-units': tool_belt.get_nv_sig_units(),
               'image_range': img_range,
               'image_range-units': 'V',
               'num_steps': num_steps,
               'green_pulse_time': green_pulse_time,
               'green_pulse_time-units': 'ns',
               'readout': readout,
               'readout-units': 'ns',
               'power_list': power_list,
               'power_list-units': 'mW',
               'power_err_list': power_err_list.tolist(),
               'power_err_list-units': 'mW',
               'ring_radius_list':ring_radius_list,
               'ring_radius_list-units': 'um',
               'ring_err_list': ring_err_list,
               'ring_err_list-units': 'um'}
    
    filePath = tool_belt.get_file_path("image_sample", timestamp, nv_sig['name'], subfolder = sub_folder)
    tool_belt.save_raw_data(rawData, filePath + '_radius_vs_power')
    
    tool_belt.save_figure(power_fig, filePath + '_radius_vs_power')
    
def r_vs_time_plot(nv_sig, ring_radius_list, ring_err_list, green_time_list, 
                     sub_folder, 
                    img_range, num_steps, green_pulse_time, readout):
    power_fig, ax = plt.subplots(1,1, figsize = (8, 8))
    ax.errorbar(numpy.array(green_time_list)/10**9, ring_radius_list, yerr = ring_err_list, fmt = 'o')
    ax.set_xlabel('Green pulse time (s)')
    ax.set_ylabel('Charge ring radius (um)')
    ax.legend()
            
    timestamp = tool_belt.get_time_stamp()
    
    # save this dile 
    rawData = {'timestamp': timestamp,
               'nv_sig': nv_sig,
               'nv_sig-units': tool_belt.get_nv_sig_units(),
               'image_range': img_range,
               'image_range-units': 'V',
               'num_steps': num_steps,
               'green_pulse_time': green_pulse_time,
               'green_pulse_time-units': 'ns',
               'readout': readout,
               'readout-units': 'ns',
               'green_time_list': green_time_list,
               'green_time_list-units': 'ns',
               'ring_radius_list':ring_radius_list,
               'ring_radius_list-units': 'um',
               'ring_err_list': ring_err_list,
               'ring_err_list-units': 'um'}
    
    filePath = tool_belt.get_file_path("image_sample", timestamp, nv_sig['name'], subfolder = sub_folder)
    tool_belt.save_raw_data(rawData, filePath + '_radius_vs_time')
    
    tool_belt.save_figure(power_fig, filePath + '_radius_vs_time')
# %%

def radial_distrbution_power(folder_name, sub_folder):
#    labls = ['Area A', 'Area B', 'Area C']
   # create a file list of the files to analyze
    file_list  = tool_belt.get_file_list(folder_name, '.txt')
    file_list = [ 'A.txt','B.txt', 'C.txt']
    
    # create lists to fill with data
    power_list = []
    radii_array = []
    counts_r_array = []
    fig, ax = plt.subplots(1,1, figsize = (8, 8))
    l = 0
    for file in file_list:
#        try:
            data = tool_belt.get_raw_data(folder_name, file[:-4])
            # Get info from file
            timestamp = data['timestamp']
            nv_sig = data['nv_sig']
            coords = nv_sig['coords']
            x_voltages = data['x_voltages']
            y_voltages = data['y_voltages']
            num_steps = data['num_steps']
            img_range= data['image_range']
            dif_img_array = numpy.array(data['dif_img_array'])
            green_pulse_time = data['green_pulse_time']
            readout = data['readout']
#            opt_volt = data['green_optical_voltage']
            opt_power = data['green_opt_power']
            
            # Initial calculations
            x_coord = coords[0]          
            y_coord = coords[1]
            half_x_range = img_range / 2
            x_high = x_coord + half_x_range

            pixel_size = x_voltages[1] - x_voltages[0]
            half_pixel_size = pixel_size / 2
            
            # List to hold the values of each pixel within the ring
            counts_r = []
            # New 2D array to put the radial values of each pixel
            r_array = numpy.empty((num_steps, num_steps))
            
            # Calculate the radial distance from each point to center
            for i in range(num_steps):
                x_pos = x_voltages[i] - x_coord
                for j in range(num_steps):
                    y_pos = y_voltages[j]  - y_coord
                    r = numpy.sqrt(x_pos**2 + y_pos**2)
                    r_array[i][j] = r
            
            # define bound on each ring radius, which will be one pixel in size
            low_r = 0
            high_r = pixel_size
            
            # step throguh the radial ranges for each ring, add pixel within ring to list
            while high_r <= (x_high + half_pixel_size):
                ring_counts = []
                for i in range(num_steps):
                    for j in range(num_steps): 
                        radius = r_array[i][j]
                        if radius >= low_r and radius < high_r:
                            ring_counts.append(dif_img_array[i][j])
                # average the counts of all counts in a ring
                counts_r.append(numpy.average(ring_counts))
                # advance the radial bounds
                low_r = high_r
                high_r = high_r + pixel_size
            
            # define the radial values as center values of pizels along x, convert to um
            radii = numpy.array(x_voltages[int(num_steps/2):])*35
            # plot
            ax.plot(radii, counts_r, label  = labls[l])#'{} mW green pulse'.format('%.2f'%opt_power))
            power_list.append(opt_power)
            radii_array.append(radii.tolist())
            counts_r_array.append(counts_r)
            l = l+1
            
            integrated_counts = integrate.simps(counts_r, x = radii)
            print(integrated_counts)
            # try to fit the radial distribution to a double gaussian(work in prog)    
#            try:
#                contrast_low = 500
#                sigma_low = 5
#                center_low = -5
#                offset_low = 100
#                contrast_high = 300
#                sigma_high = 5
#                center_high = 20
#                offset_high = 100            
#                guess_params = (contrast_low, sigma_low, center_low, offset_low,
#                                contrast_high, sigma_high, center_high, offset_high)
#                
#                popt, pcov = curve_fit(double_gaussian_dip, radii[1:], counts_r[1:], p0=guess_params)
#                radii_linspace = numpy.linspace(radii[0], radii[-1], 1000)
#                
#                ax.plot(radii_linspace, double_gaussian_dip(radii_linspace, *popt))
#                print('fit succeeded')
#                
#                power_list.append(opt_power)
#                ring_radius_list.append(popt[6])
#                ring_err_list.append(pcov[6][6])
#                
#            except Exception:
#                print('fit failed' )
                
            
#        except Exception:
#            continue
        
    ax.set_xlabel('Radius (um)')
    ax.set_ylabel('Avg counts around ring (kcps)')
    ax.set_title('Varying position in diamond, similar depth\n50 s, 2 mW green pulse')
            #'Varying green pulse power, {} s'.format(green_pulse_time / 10**9))
    ax.legend()
 
    # save data from this file
    rawData = {'timestamp': timestamp,
               'file_list': file_list,
               'nv_sig': nv_sig,
               'nv_sig-units': tool_belt.get_nv_sig_units(),
               'num_steps': num_steps,
               'green_pulse_time': green_pulse_time,
               'green_pulse_time-units': 'ns',
               'readout': readout,
               'readout-units': 'ns',
               'power_list': power_list,
               'power_list-units':'mW',
               'radii_array': radii_array,
               'radii_array-units': 'um',
               'counts_r_array': counts_r_array,
               'counts_r_array-units': 'kcps'}
    
    filePath = tool_belt.get_file_path("image_sample", timestamp, nv_sig['name'], subfolder = sub_folder)
#    print(filePath)
    tool_belt.save_raw_data(rawData, filePath + '_radius')
    
    tool_belt.save_figure(fig, filePath + '_radius')
    
    return
                
# %%

def radial_distrbution_time(folder_name, sub_folder):
    # create a file list of the files to analyze
    file_list  = tool_belt.get_file_list(folder_name, '.txt')
    file_list = ['0.1.txt', '1.txt', '5.txt', '10.txt', '25.txt', '50.txt', '75.txt', '100.txt', '250.txt', '1000.txt' ]
    # create lists to fill with data
    green_time_list = []
    radii_array = []
    counts_r_array = []
    
    fig, ax = plt.subplots(1,1, figsize = (8, 8))
    for file in file_list:
        try:
            data = tool_belt.get_raw_data(folder_name, file[:-4])
            # Get info from file
            timestamp = data['timestamp']
            nv_sig = data['nv_sig']
            coords = nv_sig['coords']
            x_voltages = data['x_voltages']
            y_voltages = data['y_voltages']
            num_steps = data['num_steps']
            img_range= data['image_range']
            dif_img_array = numpy.array(data['dif_img_array'])
            green_pulse_time = data['green_pulse_time']
            readout = data['readout']
            opt_volt = data['green_optical_voltage']
            opt_power = data['green_opt_power']
            
            # Initial calculations
            x_coord = coords[0]          
            y_coord = coords[1]
            half_x_range = img_range / 2
            x_high = x_coord + half_x_range

            pixel_size = x_voltages[1] - x_voltages[0]
            half_pixel_size = pixel_size / 2
            
            # List to hold the values of each pixel within the ring
            counts_r = []
            # New 2D array to put the radial values of each pixel
            r_array = numpy.empty((num_steps, num_steps))
            
            # Calculate the radial distance from each point to center
            for i in range(num_steps):
                x_pos = x_voltages[i] - x_coord
                for j in range(num_steps):
                    y_pos = y_voltages[j]  - y_coord
                    r = numpy.sqrt(x_pos**2 + y_pos**2)
                    r_array[i][j] = r
            
            # define bound on each ring radius, which will be one pixel in size
            low_r = 0
            high_r = pixel_size
            
            # step throguh the radial ranges for each ring, add pixel within ring to list
            while high_r <= (x_high + half_pixel_size):
                ring_counts = []
                for i in range(num_steps):
                    for j in range(num_steps): 
                        radius = r_array[i][j]
                        if radius >= low_r and radius < high_r:
                            ring_counts.append(dif_img_array[i][j])
#                print(ring_counts)
                # convert ring counts to kcps
                ring_counts = numpy.array(ring_counts)/ 1000 / (readout / 10**9)
                # average the counts of all counts in a ring
                counts_r.append(numpy.average(ring_counts))
#                print(counts_r)
                # advance the radial bounds
                low_r = high_r
                high_r = high_r + pixel_size
            
            # define the radial values as center values of pizels along x, convert to um
            radii = numpy.array(x_voltages[int(num_steps/2):])*35
            # plot
#            fig, ax = plt.subplots(1,1, figsize = (8, 8))
            ax.plot(radii, counts_r, label  = '{} s green pulse'.format(green_pulse_time/10**9))
            green_time_list.append(green_pulse_time)
            radii_array.append(radii.tolist())
            counts_r_array.append(counts_r)
            
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
                       'green_time_list': green_pulse_time,
                       'green_time_list-units':'ns',
                       'radii': radii.tolist(),
                       'radii-units': 'um',
                       'counts_r': counts_r,
                       'counts_r-units': 'kcps'}
            
            filePath = tool_belt.get_file_path("image_sample", timestamp, nv_sig['name'], subfolder = sub_folder)
#            print(filePath)
            tool_belt.save_raw_data(rawData, filePath + '_radial_dist')
            
            tool_belt.save_figure(fig, filePath + '_radial_dist')
            
#            integrated_counts = integrate.simps(counts_r, x = radii)
#            print(integrated_counts)
                
        except Exception:
            continue
    
    ax.set_xlabel('Radius (um)')
    ax.set_ylabel('Avg counts around ring (kcps)')
    ax.set_title('Varying green pulse time, {} mW'.format('%.2f'%opt_power))
    ax.legend()
 
    # save data from this file
    rawData = {'timestamp': timestamp,
               'file_list': file_list,
               'nv_sig': nv_sig,
               'nv_sig-units': tool_belt.get_nv_sig_units(),
               'num_steps': num_steps,
               'green_optical_voltage': opt_volt,
               'green_optical_voltage-units': 'V',
               'green_opt_power': opt_power,
               'green_opt_power-units': 'mW',
               'readout': readout,
               'readout-units': 'ns',
               'green_time_list': green_time_list,
               'green_time_list-units':'ns',
               'radii_array': radii_array,
               'radii_array-units': 'um',
               'counts_r_array': counts_r_array,
               'counts_r_array-units': 'kcps'}
    
    filePath = tool_belt.get_file_path("image_sample", timestamp, nv_sig['name'], subfolder = sub_folder)
    tool_belt.save_raw_data(rawData, filePath + '_radius')
    
    tool_belt.save_figure(fig, filePath + '_radius')
            
    return
    
# %%

def radial_distrbution_wait_time(folder_name, sub_folder):
    # create a file list of the files to analyze
    file_list  = tool_belt.get_file_list(folder_name, '.txt')
#    file_list = [ '0.txt', '1.txt','100.txt', '1000.txt', '10000.txt', '100000.txt']
    # create lists to fill with data
    wait_time_list = []
    radial_counts_list = []
    
    fig, ax = plt.subplots(1,1, figsize = (8, 8))
              
    for file in file_list:
        try:
            data = tool_belt.get_raw_data(folder_name, file[:-4])
            # Get info from file
            timestamp = data['timestamp']
            nv_sig = data['nv_sig']
            coords = nv_sig['coords']
            x_voltages = data['x_voltages']
            y_voltages = data['y_voltages']
            num_steps = data['num_steps']
            img_range= data['image_range']
            dif_img_array = numpy.array(data['dif_img_array'])
            green_pulse_time = data['green_pulse_time']
            wait_time = data['wait_time']
            readout = data['readout']
            opt_volt = data['green_optical_voltage']
            opt_power = data['green_opt_power']
            
            # Initial calculations
            x_coord = coords[0]          
            y_coord = coords[1]
            half_x_range = img_range / 2
            x_high = x_coord + half_x_range

            pixel_size = x_voltages[1] - x_voltages[0]
            half_pixel_size = pixel_size / 2
            
            # List to hold the values of each pixel within the ring
            counts_r = []
            # New 2D array to put the radial values of each pixel
            r_array = numpy.empty((num_steps, num_steps))
            
            # Calculate the radial distance from each point to center
            for i in range(num_steps):
                x_pos = x_voltages[i] - x_coord
                for j in range(num_steps):
                    y_pos = y_voltages[j]  - y_coord
                    r = numpy.sqrt(x_pos**2 + y_pos**2)
                    r_array[i][j] = r
            
            # define bound on each ring radius, which will be one pixel in size
            low_r = 0
            high_r = pixel_size
            
            # step throguh the radial ranges for each ring, add pixel within ring to list
            while high_r <= (x_high + half_pixel_size):
                ring_counts = []
                for i in range(num_steps):
                    for j in range(num_steps): 
                        radius = r_array[i][j]
                        if radius >= low_r and radius < high_r:
                            ring_counts.append(dif_img_array[i][j])
                # average the counts of all counts in a ring
                counts_r.append(numpy.average(ring_counts))
                # advance the radial bounds
                low_r = high_r
                high_r = high_r + pixel_size
            
            # define the radial values as center values of pizels along x, convert to um
            radii = numpy.array(x_voltages[int(num_steps/2):])*35
            radial_counts_list.append(counts_r)
            wait_time_list.append(wait_time)
            # add to the plot
            ax.plot(radii, counts_r, label = '{} s wait time'.format(wait_time))
      
        except Exception:
            continue

            
    ax.set_xlabel('Radius (um)')
    ax.set_ylabel('Avg counts around ring (kcps)')
    ax.set_title('{} s, {} mW green pulse'.format(green_pulse_time/10**9, '%.2f'%opt_power))
    ax.legend()
    
    # save data from this file
    rawData = {'timestamp': timestamp,
               'nv_sig': nv_sig,
               'nv_sig-units': tool_belt.get_nv_sig_units(),
               'num_steps': num_steps,
               'green_pulse_time': green_pulse_time,
               'green_pulse_time-units': 'ns',
               'green_optical_voltage': opt_volt,
               'green_optical_voltage-units': 'V',
               'green_opt_power': opt_power,
               'green_opt_power-units': 'mW',
               'readout': readout,
               'readout-units': 'ns',
               'wait_time_list': wait_time_list,
               'wait_time_list-units': 's',
               'radii': radii.tolist(),
               'radii-units': 'um',
               'radial_counts_list': radial_counts_list,
               'radial_counts_list-units': 'kcps'}
    
    filePath = tool_belt.get_file_path("image_sample", timestamp, nv_sig['name'], subfolder = sub_folder)
    tool_belt.save_raw_data(rawData, filePath + '_radial_dist_vs_wait_time')
    
    tool_belt.save_figure(fig, filePath + '_radial_dist_vs_wait_time')

# %% 
if __name__ == '__main__':
    parent_folder = "image_sample/branch_Spin_to_charge/2020_10/"
    
#    sub_folder = "hopper_50s_power"
#    sub_folder = "hopper_10s_power"
#    sub_folder = "hopper_1s_power"
#    sub_folder = "spatial variation"
#    folder_name = parent_folder + sub_folder
#    
#    radial_distrbution_power(folder_name, sub_folder)
    
#    sub_folder = "hopper_0.8mw_green_init"
    sub_folder = "goeppert_mayer_3mw_time"
#    sub_folder = "hopper_0.8mw_green_init/shorter_times/dif_scans"
    folder_name = parent_folder + sub_folder 
    
    radial_distrbution_time(folder_name, sub_folder)
#
    
#    radial_distrbution_wait_time(folder_name, sub_folder)



    # %% Manual data fitting for power
#    parent_folder = 'image_sample/branch_Spin_to_charge'
#    data = tool_belt.get_raw_data(parent_folder + '/2020_09/hopper_1.9mw_time', 
#                                  '2020_09_16-03_23_25-hopper-ensemble_radial_dist')
#    radii_array_air = data['radii']
#    counts_r_air = data['counts_r']
# 
#    data = tool_belt.get_raw_data(parent_folder + '/2020_07/hopper_2mw_time', 
#                                    '2020_07_11-22_50_13-hopper-ensemble_radial_dist') #2 mw
#    radii_array_oil = data['radii']
#    counts_r_oil = data['counts_r']
#    
#    
#    fig, ax = plt.subplots(1,1, figsize = (8, 8))
#    ax.plot(radii_array_air, counts_r_air, 'r-', label = 'air objective (2 mW)')
#    ax.plot(radii_array_oil, counts_r_oil, 'b-', label = 'oil objective (2 mW)')
#
#    ax.set_xlabel('Radius (um)')
#    ax.set_ylabel('Avg counts (kcps)')
#    ax.legend()
#    ax.set_title('Charge ring, air vs oil objective (1000 s green pulse)')

    
