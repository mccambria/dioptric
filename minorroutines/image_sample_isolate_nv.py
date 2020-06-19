# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 10:59:44 2020

@author: agardill
"""
# %%
import majorroutines.image_sample as image_sample
import utils.tool_belt as tool_belt
import numpy
import matplotlib.pyplot as plt
import labrad
import time
import copy
# %%

reset_range = 1.0
image_range = 1.0
num_steps = 120
num_steps_reset = 120
apd_indices = [0]
# %%

def write_pos(prev_pos, num_steps):
    yDim = num_steps
    xDim = num_steps

    if len(prev_pos) == 0:
        prev_pos[:] = [xDim, yDim - 1]

    xPos = prev_pos[0]
    yPos = prev_pos[1]

    # Figure out what direction we're heading
    headingLeft = ((yDim - 1 - yPos) % 2 == 0)

    if headingLeft:
        # Determine if we're at the left x edge
        if (xPos == 0):
            yPos = yPos - 1
        else:
            xPos = xPos - 1
    else:
        # Determine if we're at the right x edge
        if (xPos == xDim - 1):
            yPos = yPos - 1
        else:
            xPos = xPos + 1
    return xPos, yPos

def red_scan(x_voltages, y_voltages, z_center):
    with labrad.connect() as cxn:       
        # Define the previous position as null
        prev_pos = []
        # step thru every "pixel" of the reset area
        for i in range(num_steps_reset**2):
            # get the new index of the x and y positions
            x_ind, y_ind = write_pos(prev_pos, num_steps_reset)
            # update the prev_pos
            prev_pos = [x_ind, y_ind]
            # Move to the specified x and y position
#            print(x_voltages[x_ind], y_voltages[y_ind])
            tool_belt.set_xyz(cxn, [x_voltages[x_ind], y_voltages[y_ind], z_center])
            # Shine red light for 0.1 s
            cxn.pulse_streamer.constant([7], 0.0, 0.0)
            time.sleep(0.01)
            cxn.pulse_streamer.constant([], 0.0, 0.0)
  
def green_scan(x_voltages, y_voltages, z_center):
    with labrad.connect() as cxn:       
        # Define the previous position as null
        prev_pos = []
        # step thru every "pixel" of the reset area
        for i in range(num_steps_reset**2):
            # get the new index of the x and y positions
            x_ind, y_ind = write_pos(prev_pos, num_steps_reset)
            # update the prev_pos
            prev_pos = [x_ind, y_ind]
            # Move to the specified x and y position
#            print(x_voltages[x_ind], y_voltages[y_ind])
            tool_belt.set_xyz(cxn, [x_voltages[x_ind], y_voltages[y_ind], z_center])
            # Shine red light for 0.1 s
            cxn.pulse_streamer.constant([3], 0.0, 0.0)
            time.sleep(0.01)
            cxn.pulse_streamer.constant([], 0.0, 0.0)  
            
def main(nv_sig, green_pulse_time):
    aom_ao_589_pwr = nv_sig['am_589_power']
    coords = nv_sig['coords']
    readout = nv_sig['pulsed_SCC_readout_dur']
    
    adj_coords = (numpy.array(nv_sig['coords']) + \
                  numpy.array(tool_belt.get_drift())).tolist()
    x_center, y_center, z_center = adj_coords
    # get a list of x and y voltages for the galvo to position the red light at
    with labrad.connect() as cxn:    
        x_voltages, y_voltages = cxn.galvo.load_sweep_scan(x_center, y_center,
                                                       reset_range, reset_range,
                                                       num_steps_reset, 10**6)
    print('Resetting with red light\n...')
#    print('Scanning with red light\n...')
    red_scan(x_voltages, y_voltages, z_center)
        
    # collect an image under yellow right after ionization
    print('Scanning yellow light\n...')
    ref_img_array, x_voltages, y_voltages = image_sample.main(nv_sig, image_range, image_range, num_steps, 
                      aom_ao_589_pwr, apd_indices, 589, save_data=True, plot_data=True) 

    print('Resetting with red light\n...')
    red_scan(x_voltages, y_voltages, z_center)
#    print('Scanning with green light\n...')
#    green_scan(x_voltages, y_voltages, z_center)
 
    # now pulse the green at the center of the scan for a short time    
    with labrad.connect() as cxn:        
        print('Pulsing green light for {} us'.format(green_pulse_time/10**3))
        print(x_center, y_center, z_center)
        tool_belt.set_xyz(cxn, [x_center, y_center, z_center])
        cxn.pulse_streamer.constant([3], 0.0, 0.0)
        time.sleep(green_pulse_time/ 10**9)
        cxn.pulse_streamer.constant([], 0.0, 0.0)
#        seq_args = [green_pulse_time, 100, 532]           
#        seq_args_string = tool_belt.encode_seq_args(seq_args)            
#        cxn.pulse_streamer.stream_immediate('analog_sequence_test.py', 1, seq_args_string)
        
    # collect an image under yellow after green pulse
    print('Scanning yellow light\n...')
    sig_img_array, x_voltages, y_voltages = image_sample.main(nv_sig, image_range, image_range, num_steps, 
                      aom_ao_589_pwr, apd_indices, 589, save_data=True, plot_data=True) 
    
    dif_img_array = sig_img_array - ref_img_array
    x_coord = coords[0]
    half_x_range = image_range / 2
    x_low = x_coord - half_x_range
    x_high = x_coord + half_x_range
    y_coord = coords[1]
    half_y_range = image_range / 2
    y_low = y_coord - half_y_range
    y_high = y_coord + half_y_range
    
    dif_img_array_kcps = (dif_img_array / 1000) / (readout / 10**9)
    
    pixel_size = x_voltages[1] - x_voltages[0]
    half_pixel_size = pixel_size / 2
    img_extent = [x_high + half_pixel_size, x_low - half_pixel_size,
                  y_low - half_pixel_size, y_high + half_pixel_size]

#    title = 'Yellow scan (after green scan vs after red scan)'
    title = 'Yellow scan (with/without green pulse)\nGreen pulse {} ms'.format(green_pulse_time/10**6) 
    fig = tool_belt.create_image_figure(dif_img_array_kcps, img_extent, clickHandler=None, title = title, color_bar_label = 'Difference (kcps)')
    # Redraw the canvas and flush the changes to the backend
    fig.canvas.draw()
    fig.canvas.flush_events()  
    
    # Save data
    timestamp = tool_belt.get_time_stamp()

    rawData = {'timestamp': timestamp,
               'nv_sig': nv_sig,
               'nv_sig-units': tool_belt.get_nv_sig_units(),
               'image_range': image_range,
               'image_range-units': 'V',
               'num_steps': num_steps,
               'green_pulse_time': green_pulse_time,
               'green_pulse_time-units': 'ns',
               'readout': readout,
               'readout-units': 'ns',
               'x_voltages': x_voltages.tolist(),
               'x_voltages-units': 'V',
               'y_voltages': y_voltages.tolist(),
               'y_voltages-units': 'V',
               'ref_img_array': ref_img_array.tolist(),
               'ref_img_array-units': 'counts',
               'sig_img_array': sig_img_array.tolist(),
               'sig_img_array-units': 'counts',
               'dif_img_array': dif_img_array.tolist(),
               'dif_img_array-units': 'counts'}

    filePath = tool_belt.get_file_path('image_sample', timestamp, nv_sig['name'])
    tool_belt.save_raw_data(rawData, filePath + '_dif')

    tool_belt.save_figure(fig, filePath + '_dif')
   
# %%
if __name__ == '__main__':
    sample_name = 'bachman-A1'
    ensemble_B1 = { 'coords':[-2.113, 1.079, 5.06],
            'name': '{}-B1'.format(sample_name),
            'expected_count_rate': 580, 'nd_filter': 'nd_0',
            'pulsed_readout_dur': 300,
            'pulsed_SCC_readout_dur': 1*10**7, 'am_589_power': 0.25, 
            'pulsed_initial_ion_dur': 25*10**3,
            'pulsed_shelf_dur': 200, 
            'am_589_shelf_power': 0.35,
            'pulsed_ionization_dur': 500, 'cobalt_638_power': 160, 
            'pulsed_reionization_dur': 100*10**3, 'cobalt_532_power': 24, 
            'magnet_angle': 0, #60
            "resonance_LOW": 2.8235,"rabi_LOW": 110.4, "uwave_power_LOW": 9.0,
            "resonance_HIGH": 2.9878,"rabi_HIGH": 549.1,"uwave_power_HIGH": 10.0} 
    nv_sig = ensemble_B1
#    
#    green_pulse_time_list = [10**9]
    green_pulse_time_list = [10**6, 10**7, 10**8, 10**9, 10**10, 10**11]

#    green_pulse_time_list = [10**9, 1, 2]
#    green_pulse_time = 5*10**3
    
    for t in green_pulse_time_list:
#        print(i)
          
        main(nv_sig, t)
    
    
        
        
    