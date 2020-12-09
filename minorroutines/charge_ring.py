# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 10:59:44 2020

@author: agardill
"""
# %%
import majorroutines.image_sample as image_sample
import utils.tool_belt as tool_belt
import majorroutines.optimize as optimize
import numpy
import matplotlib.pyplot as plt
import labrad
import time
import copy
# %%

reset_range = 2.5
image_range = 2.5
num_steps = 200
num_steps_reset = 120
apd_indices = [0]

# %%

def plot_dif_fig(coords, x_voltages,range, dif_img_array, readout, title ):
    x_coord = coords[0]
    half_x_range = range / 2
    x_low = x_coord - half_x_range
    x_high = x_coord + half_x_range
    y_coord = coords[1]
    half_y_range = range / 2
    y_low = y_coord - half_y_range
    y_high = y_coord + half_y_range
    
    dif_img_array_kcps = (dif_img_array / 1000) / (readout / 10**9)
    
    pixel_size = x_voltages[1] - x_voltages[0]
    half_pixel_size = pixel_size / 2
    img_extent = [x_high + half_pixel_size, x_low - half_pixel_size,
                  y_low - half_pixel_size, y_high + half_pixel_size]
 
    fig = tool_belt.create_image_figure(dif_img_array_kcps, img_extent, 
                                        clickHandler=None, title = title, 
                                        color_bar_label = 'Difference (kcps)', 
#                                        min_value = 0
                                        )
    fig.canvas.draw()
    fig.canvas.flush_events()  
    return fig

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
        # get the wiring for the red
        wiring = tool_belt.get_pulse_streamer_wiring(cxn)
        pulser_wiring_red = wiring['do_638_laser']
        # Define the previous position as null
        prev_pos = []
        # step thru every "pixel" of the reset area
        for i in range(red_num_steps_reset**2):
            # get the new index of the x and y positions
            x_ind, y_ind = write_pos(prev_pos, red_num_steps_reset)
            # update the prev_pos
            prev_pos = [x_ind, y_ind]
            # Move to the specified x and y position
            tool_belt.set_xyz(cxn, [x_voltages[x_ind], y_voltages[y_ind], z_center])
            # Shine red light for 0.01 s
            cxn.pulse_streamer.constant([pulser_wiring_red], 0.0, 0.0)
            time.sleep(0.01)
            cxn.pulse_streamer.constant([], 0.0, 0.0)
  
def green_scan(x_voltages, y_voltages, z_center):
    with labrad.connect() as cxn:      
        # get the wiring for the green
        wiring = tool_belt.get_pulse_streamer_wiring(cxn)
        pulser_wiring_green = wiring['do_532_aom']
        # Define the previous position as null
        prev_pos = []
        # step thru every "pixel" of the reset area
        for i in range(num_steps_reset**2):
            # get the new index of the x and y positions
            x_ind, y_ind = write_pos(prev_pos, num_steps_reset)
            # update the prev_pos
            prev_pos = [x_ind, y_ind]
            # Move to the specified x and y position
            tool_belt.set_xyz(cxn, [x_voltages[x_ind], y_voltages[y_ind], z_center])
            # Shine red light for 0.01 s
            cxn.pulse_streamer.constant([pulser_wiring_green], 0.0, 0.0)
            time.sleep(0.01)
            cxn.pulse_streamer.constant([], 0.0, 0.0)  
  
def yellow_scan(x_voltages, y_voltages, z_center):
    with labrad.connect() as cxn:      
        # get the wiring for the green
#        wiring = tool_belt.get_pulse_streamer_wiring(cxn)
#        pulser_wiring_yellow = wiring['ao_589_aom']
        # Define the previous position as null
        prev_pos = []
        # step thru every "pixel" of the reset area
        for i in range(num_steps_reset**2):
            # get the new index of the x and y positions
            x_ind, y_ind = write_pos(prev_pos, num_steps_reset)
            # update the prev_pos
            prev_pos = [x_ind, y_ind]
            # Move to the specified x and y position
            tool_belt.set_xyz(cxn, [x_voltages[x_ind], y_voltages[y_ind], z_center])
            # Shine red light for 0.01 s
            cxn.pulse_streamer.constant([], 0.0, 1.0)
            time.sleep(0.01)
            cxn.pulse_streamer.constant([], 0.0, 0.0)  
# %%          
def main_red_green(cxn, nv_sig, green_pulse_time, wait_time = 0):
#    aom_ao_589_pwr = nv_sig['am_589_power']
    coords = nv_sig['coords']
#    print(coords)
    readout = nv_sig['pulsed_SCC_readout_dur']
#    green_pulse_time = int(green_pulse_time)
    color_filter = nv_sig['color_filter'] 
    

    optimize.main_with_cxn(cxn, nv_sig, apd_indices, 532, disable=False)  
    
    adj_coords = (numpy.array(nv_sig['coords']) + \
                  numpy.array(tool_belt.get_drift())).tolist()
    x_center, y_center, z_center = adj_coords
    
    # Get a list of x and y voltages for the red scan
    x_voltages_r, y_voltages_r = cxn.galvo.load_sweep_scan(x_center, y_center,
                                                   reset_range, reset_range,
                                                   red_num_steps_reset, 10**6)
    print('Resetting with red light\n...')
    red_scan(x_voltages_r, y_voltages_r, z_center)
         
    print('Waiting for {} s, during green pulse'.format(green_pulse_time))
    tool_belt.set_xyz(cxn, [x_center, y_center, z_center])
    # Use two methods to pulse the green light, depending on pulse length
    if green_pulse_time < 1:
        seq_args = [int(green_pulse_time*10**9), 0, 0.0, 532]           
        seq_args_string = tool_belt.encode_seq_args(seq_args)            
        cxn.pulse_streamer.stream_immediate('simple_pulse.py', 1, seq_args_string)   
    else:
        cxn.pulse_streamer.constant([], 0.0, 0.0)
        time.sleep(green_pulse_time)
    cxn.pulse_streamer.constant([], 0.0, 0.0) 
    
    if wait_time:
        print('Waiting for {} s, after green pulse'.format(wait_time))
        tool_belt.set_xyz(cxn, [x_center, y_center, z_center])  
        cxn.pulse_streamer.constant([], 0.0, 0.0)
        time.sleep(wait_time)
        cxn.pulse_streamer.constant([], 0.0, 0.0) 
        
    # collect an image under yellow right after ionization
    print('Scanning yellow light\n...')
    ref_img_array, x_voltages, y_voltages = image_sample.main(nv_sig, image_range, image_range, num_steps, 
                      apd_indices, 589,readout = readout, save_data=True, plot_data=True) 


    optimize.main_with_cxn(cxn, nv_sig, apd_indices, 532, disable=False)  
    
    print('Resetting with red light\n...')
    red_scan(x_voltages_r, y_voltages_r, z_center)
 
    # now pulse the green at the center of the scan for a short time         
    print('Pulsing green light for {} s'.format(green_pulse_time))
    tool_belt.set_xyz(cxn, [x_center, y_center, z_center])
    # Use two methods to pulse the green light, depending on pulse length
    if green_pulse_time < 1.1:
        shared_params = tool_belt.get_shared_parameters_dict(cxn)
        laser_515_delay = shared_params['515_laser_delay']
        seq_args = [laser_515_delay, int(green_pulse_time*10**9), 0.0, 532]           
        seq_args_string = tool_belt.encode_seq_args(seq_args)            
        cxn.pulse_streamer.stream_immediate('simple_pulse.py', 1, seq_args_string)   
    else:
        cxn.pulse_streamer.constant([3], 0.0, 0.0)
        time.sleep(green_pulse_time)
    cxn.pulse_streamer.constant([], 0.0, 0.0)
    
    if wait_time:
        print('Waiting for {} s, after green pulse'.format(wait_time))
        tool_belt.set_xyz(cxn, [x_center, y_center, z_center])  
        cxn.pulse_streamer.constant([], 0.0, 0.0)
        time.sleep(wait_time)
        cxn.pulse_streamer.constant([], 0.0, 0.0) 
        
    # collect an image under yellow after green pulse
    print('Scanning yellow light\n...')
    sig_img_array, x_voltages, y_voltages = image_sample.main(nv_sig, image_range, image_range, num_steps, 
                      apd_indices, 589, readout = readout, save_data=True, plot_data=True) 

    # Measure the green power  
    wiring = tool_belt.get_pulse_streamer_wiring(cxn)
    pulser_wiring_green = wiring['do_532_aom']
    cxn.pulse_streamer.constant([pulser_wiring_green], 0.0, 0.0)
    opt_volt = tool_belt.opt_power_via_photodiode(532)
    opt_power = tool_belt.calc_optical_power_mW(532, opt_volt)
    time.sleep(0.1)
    cxn.pulse_streamer.constant([], 0.0, 0.0)

    # Subtract and plot
    dif_img_array = sig_img_array - ref_img_array
    
    if wait_time:
        title = 'Yellow scan (with/without green pulse)\nGreen pulse 10 s\n{} s wait'.format(wait_time) 
    else:
        title = 'Yellow scan (with/without green pulse)\nGreen pulse {} s'.format(green_pulse_time) 
    fig = plot_dif_fig(coords, x_voltages,image_range,  dif_img_array, readout, title )
    
    # Save data
    timestamp = tool_belt.get_time_stamp()

    rawData = {'timestamp': timestamp,
               'nv_sig': nv_sig,
               'nv_sig-units': tool_belt.get_nv_sig_units(),
               'color_filter' : color_filter,
               'image_range': image_range,
               'image_range-units': 'V',
               'num_steps': num_steps,
               'reset_range': reset_range,
               'reset_range-units': 'V',
               'red_num_steps_reset': red_num_steps_reset,
               'green_pulse_time': float(green_pulse_time),
               'green_pulse_time-units': 's',
               'wait_time': int(wait_time),
               'wait_time-units': 's',
               'green_optical_voltage': opt_volt,
               'green_optical_voltage-units': 'V',
               'green_opt_power': opt_power,
               'green_opt_power-units': 'mW',
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
    
    return coords, x_voltages,image_range , dif_img_array, readout
# %%         
def main_green_red(cxn, nv_sig, green_pulse_time, wait_time = 0):
#    aom_ao_589_pwr = nv_sig['am_589_power']
    coords = nv_sig['coords']
#    print(coords)
    readout = nv_sig['pulsed_SCC_readout_dur']
#    green_pulse_time = int(green_pulse_time)
    

#    optimize.main(nv_sig, apd_indices, 532)    
    
    adj_coords = (numpy.array(nv_sig['coords']) + \
                  numpy.array(tool_belt.get_drift())).tolist()
    x_center, y_center, z_center = adj_coords
    
    # Get a list of x and y voltages for the red scan
    x_voltages_r, y_voltages_r = cxn.galvo.load_sweep_scan(x_center, y_center,
                                                   reset_range, reset_range,
                                                   num_steps_reset, 10**6)
    print('Resetting with green light\n...')
    green_scan(x_voltages_r, y_voltages_r, z_center)
         
    print('Waiting for {} s, during red pulse'.format(green_pulse_time))
    tool_belt.set_xyz(cxn, [x_center, y_center, z_center])
    # Use two methods to pulse the green light, depending on pulse length
    if green_pulse_time < 1:
        seq_args = [int(green_pulse_time*10**9), 0, 0.0, 638]           
        seq_args_string = tool_belt.encode_seq_args(seq_args)            
        cxn.pulse_streamer.stream_immediate('simple_pulse.py', 1, seq_args_string)   
    else:
        cxn.pulse_streamer.constant([], 0.0, 0.0)
        time.sleep(green_pulse_time)
    cxn.pulse_streamer.constant([], 0.0, 0.0) 
    
    if wait_time:
        print('Waiting for {} s, after green pulse'.format(wait_time))
        tool_belt.set_xyz(cxn, [x_center, y_center, z_center])  
        cxn.pulse_streamer.constant([], 0.0, 0.0)
        time.sleep(wait_time)
        cxn.pulse_streamer.constant([], 0.0, 0.0) 
        
    # collect an image under yellow right after ionization
    print('Scanning yellow light\n...')
    ref_img_array, x_voltages, y_voltages = image_sample.main(nv_sig, image_range, image_range, num_steps, 
                      apd_indices, 589,readout = readout, save_data=True, plot_data=True) 


#    optimize.main(nv_sig, apd_indices, 532)    
    print('Resetting with green light\n...')
    green_scan(x_voltages_r, y_voltages_r, z_center)
 
    # now pulse the red at the center of the scan for a short time         
    print('Pulsing red light for {} s'.format(green_pulse_time))
    tool_belt.set_xyz(cxn, [x_center, y_center, z_center])
    # Use two methods to pulse the green light, depending on pulse length
    if green_pulse_time < 1:
        shared_params = tool_belt.get_shared_parameters_dict(cxn)
        laser_638_delay = shared_params['638_DM_laser_delay']
        seq_args = [laser_638_delay, int(green_pulse_time*10**9), 0.0, 638]           
        seq_args_string = tool_belt.encode_seq_args(seq_args)            
        cxn.pulse_streamer.stream_immediate('simple_pulse.py', 1, seq_args_string)   
    else:
        cxn.pulse_streamer.constant([7], 0.0, 0.0)
        time.sleep(green_pulse_time)
    cxn.pulse_streamer.constant([], 0.0, 0.0)
    
    if wait_time:
        print('Waiting for {} s, after green pulse'.format(wait_time))
        tool_belt.set_xyz(cxn, [x_center, y_center, z_center])  
        cxn.pulse_streamer.constant([], 0.0, 0.0)
        time.sleep(wait_time)
        cxn.pulse_streamer.constant([], 0.0, 0.0) 
        
    # collect an image under yellow after green pulse
    print('Scanning yellow light\n...')
    sig_img_array, x_voltages, y_voltages = image_sample.main(nv_sig, image_range, image_range, num_steps, 
                      apd_indices, 589, readout = readout, save_data=True, plot_data=True) 

    # Measure the green power  
    wiring = tool_belt.get_pulse_streamer_wiring(cxn)
    pulser_wiring_green = wiring['do_532_aom']
    cxn.pulse_streamer.constant([pulser_wiring_green], 0.0, 0.0)
    opt_volt = tool_belt.opt_power_via_photodiode(532)
    opt_power = tool_belt.calc_optical_power_mW(532, opt_volt)
    time.sleep(0.1)
    cxn.pulse_streamer.constant([], 0.0, 0.0)
    
    # Measure the red power  
    wiring = tool_belt.get_pulse_streamer_wiring(cxn)
    pulser_wiring_green = wiring['do_638_laser']
    cxn.pulse_streamer.constant([pulser_wiring_green], 0.0, 0.0)
    opt_volt_red = tool_belt.opt_power_via_photodiode(638)
    opt_power_red = tool_belt.calc_optical_power_mW(638, opt_volt_red)
    time.sleep(0.1)
    cxn.pulse_streamer.constant([], 0.0, 0.0)

    # Subtract and plot
    dif_img_array = sig_img_array - ref_img_array
    
    if wait_time:
        title = 'Yellow scan (with/without red pulse)\nGreen pulse 10 s\n{} s wait'.format(wait_time) 
    else:
        title = 'Yellow scan (with/without red pulse)\nGreen pulse {} s'.format(green_pulse_time) 
    fig = plot_dif_fig(coords, x_voltages,image_range,  dif_img_array, readout, title )
    
    # Save data
    timestamp = tool_belt.get_time_stamp()

    rawData = {'timestamp': timestamp,
               'nv_sig': nv_sig,
               'nv_sig-units': tool_belt.get_nv_sig_units(),
               'image_range': image_range,
               'image_range-units': 'V',
               'num_steps': num_steps,
               'reset_range': reset_range,
               'reset_range-units': 'V',
               'num_steps_reset': num_steps_reset,
               'green_pulse_time': float(green_pulse_time),
               'green_pulse_time-units': 's',
               'wait_time': int(wait_time),
               'wait_time-units': 's',
               'green_optical_voltage': opt_volt,
               'green_optical_voltage-units': 'V',
               'green_opt_power': opt_power,
               'green_opt_power-units': 'mW',
               'red_optical_voltage': opt_volt_red,
               'red_optical_voltage-units': 'V',
               'red_opt_power': opt_power_red,
               'red_opt_power-units': 'mW',
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
    
    return coords, x_voltages,image_range , dif_img_array, readout

         
def main_green_green_red_scan(cxn, nv_sig, pulse_time):
    coords = nv_sig['coords']
#    print(coords)
    readout = nv_sig['pulsed_SCC_readout_dur']
    color_filter = nv_sig['color_filter'] 
    am_589_power = nv_sig['am_589_power']
    

    optimize.main_with_cxn(cxn, nv_sig, apd_indices, 532, disable=False)  
    
    adj_coords = (numpy.array(nv_sig['coords']) + \
                  numpy.array(tool_belt.get_drift())).tolist()
    x_center, y_center, z_center = adj_coords
    
    # Get a list of x and y voltages for the red scan
    x_voltages_r, y_voltages_r = cxn.galvo.load_sweep_scan(x_center, y_center,
                                                   reset_range, reset_range,
                                                   red_num_steps_reset, 10**6)
    print('Resetting with green light\n...')
    green_scan(x_voltages_r, y_voltages_r, z_center)
         
    print('Waiting for {} s, during green pulse'.format(pulse_time))
    tool_belt.set_xyz(cxn, [x_center, y_center, z_center])
    # Use two methods to pulse the green light, depending on pulse length
    if pulse_time < 1:
        seq_args = [int(pulse_time*10**9), 0, 0.0, 532]           
        seq_args_string = tool_belt.encode_seq_args(seq_args)            
        cxn.pulse_streamer.stream_immediate('simple_pulse.py', 1, seq_args_string)   
    else:
        cxn.pulse_streamer.constant([], 0.0, 0.0)
        time.sleep(pulse_time)
    cxn.pulse_streamer.constant([], 0.0, 0.0) 
        
    # collect an image under yellow right after ionization
    print('Scanning red light\n...')
    ref_img_array, x_voltages, y_voltages = image_sample.main(nv_sig, image_range, image_range, num_steps, 
                      apd_indices, 638,readout = readout, save_data=True, plot_data=True) 


    optimize.main_with_cxn(cxn, nv_sig, apd_indices, 532, disable=False)  
    
    print('Resetting with green light\n...')
    green_scan(x_voltages_r, y_voltages_r, z_center)
 
    # now pulse the yellow at the center of the scan for a short time         
    print('Pulsing green light for {} s'.format(pulse_time))
    tool_belt.set_xyz(cxn, [x_center, y_center, z_center])
    # Use two methods to pulse the green light, depending on pulse length
    if pulse_time < 1.1:
        shared_params = tool_belt.get_shared_parameters_dict(cxn)
        laser_589_delay = shared_params['515_laser_delay']
        seq_args = [laser_589_delay, int(pulse_time*10**9), 0.0, 532]           
        seq_args_string = tool_belt.encode_seq_args(seq_args)            
        cxn.pulse_streamer.stream_immediate('simple_pulse.py', 1, seq_args_string)   
    else:
        cxn.pulse_streamer.constant([3], 0.0)
        time.sleep(pulse_time)
    cxn.pulse_streamer.constant([], 0.0, 0.0)
    

        
    # collect an image under yellow after green pulse
    print('Scanning red light\n...')
    sig_img_array, x_voltages, y_voltages = image_sample.main(nv_sig, image_range, image_range, num_steps, 
                      apd_indices, 638, readout = readout, save_data=True, plot_data=True) 


    # measure laser powers:
    green_optical_power_pd, green_optical_power_mW, \
            red_optical_power_pd, red_optical_power_mW, \
            yellow_optical_power_pd, yellow_optical_power_mW = \
            tool_belt.measure_g_r_y_power(
                              nv_sig['am_589_power'], nv_sig['nd_filter'])

    # Subtract and plot
    dif_img_array = sig_img_array - ref_img_array
    
 
    title = 'Red scan (with/without green pulse)\Green pulse {} s'.format(pulse_time) 
    fig = plot_dif_fig(coords, x_voltages,image_range,  dif_img_array, readout, title )
    
    # Save data
    timestamp = tool_belt.get_time_stamp()

    rawData = {'timestamp': timestamp,
               'nv_sig': nv_sig,
               'nv_sig-units': tool_belt.get_nv_sig_units(),
               'color_filter' : color_filter,
               'image_range': image_range,
               'image_range-units': 'V',
               'num_steps': num_steps,
               'reset_range': reset_range,
               'reset_range-units': 'V',
               'red_num_steps_reset': red_num_steps_reset,
               'pulse_time': float(pulse_time),
               'green_pulse_time-units': 's',
            'green_optical_power_pd': green_optical_power_pd,
            'green_optical_power_pd-units': 'V',
            'green_optical_power_mW': green_optical_power_mW,
            'green_optical_power_mW-units': 'mW',
            'red_optical_power_pd': red_optical_power_pd,
            'red_optical_power_pd-units': 'V',
            'red_optical_power_mW': red_optical_power_mW,
            'red_optical_power_mW-units': 'mW',
            'yellow_optical_power_pd': yellow_optical_power_pd,
            'yellow_optical_power_pd-units': 'V',
            'yellow_optical_power_mW': yellow_optical_power_mW,
            'yellow_optical_power_mW-units': 'mW',
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
    
    return coords, x_voltages,image_range , dif_img_array, readout
  
# %%      
def main_moving_target(cxn, nv_sig, pulse_time, init_time, init_color, pulse_color, readout_color, disable_optimize=False,wait_time = 0):
    am_589_power = nv_sig['am_589_power']
    coords = nv_sig['coords']
    readout = nv_sig['pulsed_SCC_readout_dur']
    color_filter = nv_sig['color_filter'] 
    
    wiring = tool_belt.get_pulse_streamer_wiring(cxn)
    pulser_wiring_green = wiring['do_532_aom']
#    pulser_wiring_yellow = wiring['ao_589_aom']
    pulser_wiring_red = wiring['do_638_laser']

    shared_params = tool_belt.get_shared_parameters_dict(cxn)    
    laser_515_delay = shared_params['515_laser_delay']
    laser_589_delay = shared_params['589_aom_delay']
    laser_638_delay = shared_params['638_DM_laser_delay']
    if pulse_color == 532:
        direct_wiring = pulser_wiring_green
        laser_delay = laser_515_delay
    elif pulse_color == 589:
#        direct_wiring = pulser_wiring_yellow
        laser_delay = laser_589_delay
    elif pulse_color == 638:
        direct_wiring = pulser_wiring_red
        laser_delay = laser_638_delay
    
    

    optimize.main_with_cxn(cxn, nv_sig, apd_indices, 532, disable=disable_optimize)  
    
    adj_coords = (numpy.array(nv_sig['coords']) + \
                  numpy.array(tool_belt.get_drift())).tolist()
    x_center, y_center, z_center = adj_coords
    
    print('Resetting with {} nm light\n...'.format(init_color))
    _,_,_ = image_sample.main(nv_sig, reset_range, reset_range, num_steps_reset, 
                      apd_indices, init_color,readout = init_time, save_data=False, plot_data=False) 
         
    print('Waiting for {} s, during {} nm pulse'.format(pulse_time, pulse_color))
    tool_belt.set_xyz(cxn, [x_center, y_center, z_center])
    # Use two methods to pulse the green light, depending on pulse length
    if pulse_time < 1:
        seq_args = [int(pulse_time*10**9), 0, 0.0, 532]           
        seq_args_string = tool_belt.encode_seq_args(seq_args)            
        cxn.pulse_streamer.stream_immediate('simple_pulse.py', 1, seq_args_string)   
    else:
        cxn.pulse_streamer.constant([], 0.0, 0.0)
        time.sleep(pulse_time)
    cxn.pulse_streamer.constant([], 0.0, 0.0) 
    
    if wait_time:
        print('Waiting for {} s, after {} nm pulse'.format(wait_time, pulse_color))
        tool_belt.set_xyz(cxn, [x_center, y_center, z_center])  
        cxn.pulse_streamer.constant([], 0.0, 0.0)
        time.sleep(wait_time)
        cxn.pulse_streamer.constant([], 0.0, 0.0) 
        
    # collect an image
    print('Imaging {} nm light\n...'.format(readout_color))
    ref_img_array, x_voltages, y_voltages = image_sample.main(nv_sig, image_range, image_range, num_steps, 
                      apd_indices, readout_color,readout = readout, save_data=True, plot_data=True) 


    optimize.main_with_cxn(cxn, nv_sig, apd_indices, 532, disable=disable_optimize)    
    adj_coords = (numpy.array(nv_sig['coords']) + \
                  numpy.array(tool_belt.get_drift())).tolist()
    x_center, y_center, z_center = adj_coords  
    
    print('Resetting with {} nm light\n...'.format(init_color))
    _,_,_ = image_sample.main(nv_sig, reset_range, reset_range, num_steps_reset, 
                      apd_indices, init_color,readout = init_time, save_data=False, plot_data=False) 
 
    # now pulse at the center of the scan for a short time         
    print('Pulsing {} nm light for {} s'.format(pulse_color, pulse_time))
    tool_belt.set_xyz(cxn, [x_center, y_center, z_center])
    # Use two methods to pulse the light, depending on pulse length
    if pulse_time < 1.1:
        seq_args = [laser_delay, int(pulse_time*10**9), 0.0, pulse_color]           
        seq_args_string = tool_belt.encode_seq_args(seq_args)            
        cxn.pulse_streamer.stream_immediate('simple_pulse.py', 1, seq_args_string)   
    else:
        if pulse_color == 532 or pulse_color==638:
            cxn.pulse_streamer.constant([direct_wiring], 0.0, 0.0)
        elif pulse_color == 589:
            cxn.pulse_streamer.constant([], 0.0, am_589_power)
        time.sleep(pulse_time)
    cxn.pulse_streamer.constant([], 0.0, 0.0)

    if wait_time:
        print('Waiting for {} s, after {} nm pulse'.format(wait_time, pulse_color))
        tool_belt.set_xyz(cxn, [x_center, y_center, z_center])  
        cxn.pulse_streamer.constant([], 0.0, 0.0)
        time.sleep(wait_time)
        cxn.pulse_streamer.constant([], 0.0, 0.0) 
        
    # collect an image under yellow after green pulse
    print('Imaging {} nm light\n...'.format(readout_color))
    sig_img_array, x_voltages, y_voltages = image_sample.main(nv_sig, image_range, image_range, num_steps, 
                      apd_indices, readout_color,readout = readout, save_data=True, plot_data=True) 

    # measure laser powers:
    green_optical_power_pd, green_optical_power_mW, \
            red_optical_power_pd, red_optical_power_mW, \
            yellow_optical_power_pd, yellow_optical_power_mW = \
            tool_belt.measure_g_r_y_power(
                              nv_sig['am_589_power'], nv_sig['nd_filter'])

    # Subtract and plot
    dif_img_array = sig_img_array - ref_img_array
    
    if pulse_color == 532:
        pulse_power = green_optical_power_mW
    if pulse_color == 589:
        pulse_power = red_optical_power_mW
    if pulse_color == 638:
        pulse_power = yellow_optical_power_mW

    title = 'Moving readout subtracted image, {} nm init, {} nm readout\n{} nm pulse {} s, {:.1f} mW'.format(init_color, readout_color, pulse_color, pulse_time, pulse_power) 
    fig = plot_dif_fig(coords, x_voltages,image_range,  dif_img_array, readout, title )
    
    # Save data
    timestamp = tool_belt.get_time_stamp()

    rawData = {'timestamp': timestamp,
               'init_color': init_color,
               'pulse_color': pulse_color,
               'readout_color': readout_color,
               'nv_sig': nv_sig,
               'nv_sig-units': tool_belt.get_nv_sig_units(),
               'color_filter' : color_filter,
               'image_range': image_range,
               'image_range-units': 'V',
               'num_steps': num_steps,
               'reset_range': reset_range,
               'reset_range-units': 'V',
               'num_steps_reset': num_steps_reset,
               'init_time': init_time,
               'init_time-units': 'ns',
               'pulse_time': float(pulse_time),
               'pulse_time-units': 's',
               
               'wait_time': int(wait_time),
               'wait_time-units': 's',
                'green_optical_power_pd': green_optical_power_pd,
                'green_optical_power_pd-units': 'V',
                'green_optical_power_mW': green_optical_power_mW,
                'green_optical_power_mW-units': 'mW',
                'red_optical_power_pd': red_optical_power_pd,
                'red_optical_power_pd-units': 'V',
                'red_optical_power_mW': red_optical_power_mW,
                'red_optical_power_mW-units': 'mW',
                'yellow_optical_power_pd': yellow_optical_power_pd,
                'yellow_optical_power_pd-units': 'V',
                'yellow_optical_power_mW': yellow_optical_power_mW,
                'yellow_optical_power_mW-units': 'mW',
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
    
    return coords, x_voltages,image_range , dif_img_array, readout  
# %%
if __name__ == '__main__':
    sample_name = 'goeppert-mayer'
    
    nv1_2020_12_02 = { 'coords':[0.225, 0.242, 5.20], 
            'name': '{}-nv1_2020_12_02'.format(sample_name),
            'expected_count_rate': 50, 'nd_filter': 'nd_1.0',
            'color_filter': '635-715 bp',
            'pulsed_readout_dur': 300,
            'pulsed_SCC_readout_dur': 20000000, 'am_589_power': 0.7, 
            'pulsed_ionization_dur': 10**3, 'cobalt_638_power': 120, 
            'pulsed_reionization_dur': 100*10**3, 'cobalt_532_power':20, 
            'magnet_angle': 0,
            "resonance_LOW": 2.7,"rabi_LOW": 146.2, "uwave_power_LOW": 9.0,
            "resonance_HIGH": 2.9774,"rabi_HIGH": 95.2,"uwave_power_HIGH": 10.0} 
    
    init_time = 3*10**7
    init_color = 638
    pulse_color = 532
    readout_color = 589
    
    green_pulse_time_list = numpy.array([10])
#    green_pulse_time_list = numpy.array([0.1, 1, 5, 10, 25, 50, 75, 100, 250 ,1000]) #  0.6 mW, 2 mW,  4?
#    green_pulse_time_list = numpy.array([0.1]) # 60 mW, 16 mW, 4 mW
  
#    green_pulse_time_list = numpy.array([0.001, 0.005,0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1 ]) # 60 mW, 16 mW, 4 mW
        
#    green_pulse_time_list = [100]

    with labrad.connect() as cxn:
        for t in green_pulse_time_list:
            nv_sig = copy.deepcopy(nv1_2020_12_02)
            nv_sig['color_filter'] = '715 lp' 
#            main_red_green(cxn, nv_sig, t)
            main_moving_target(cxn, nv_sig, t, init_time, init_color, pulse_color, readout_color)
            
#            nv_sig = copy.deepcopy(nv1_2020_12_02) 
#            nv_sig['color_filter'] = '635-715 bp'
#            main_red_green(cxn, nv_sig, t)
#            main_moving_target(cxn, nv_sig, t, init_time, init_color, pulse_color, readout_color)
            
#    folder = 'image_sample/branch_Spin_to_charge/2020_10'
#    file_dark_dif = '2020_10_23-20_32_23-goeppert-mayer-nv2_dif'
#    file_nv_dif = '2020_10_23-21_07_25-goeppert-mayer-nv2_dif'
#    data = tool_belt.get_raw_data(folder, file_dark_dif)
#    
#    dif_img_array_dark = data['dif_img_array']   
#    data = tool_belt.get_raw_data(folder, file_nv_dif)
#    
#    dif_img_array_nv = data['dif_img_array'] 
#    x_voltages = data['x_voltages']
#    readout= 10**7
#    image_range = 0.5
#    coords = [
#      -0.079,
#      -0.019,
#      5.0]
#    
#     
#    dif_dif_img_array = numpy.array(dif_img_array_nv) - numpy.array(dif_img_array_dark) 
#    title = 'Subtraction of charge ring measurement on nv vs off nv'
#    plot_dif_fig(coords, x_voltages,image_range,  dif_dif_img_array, readout, title )



        
        