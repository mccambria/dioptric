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

#reset_range = 2.5
#image_range = 2.5
reset_range = 0.5
image_range = 0.5
num_steps = 120
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
        for i in range(num_steps_reset**2):
            # get the new index of the x and y positions
            x_ind, y_ind = write_pos(prev_pos, num_steps_reset)
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
  
# %%          
def main(cxn, nv_sig, green_pulse_time, wait_time = 0):
    aom_ao_589_pwr = nv_sig['am_589_power']
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
                      aom_ao_589_pwr, apd_indices, 589, save_data=True, plot_data=True) 


#    optimize.main(nv_sig, apd_indices, 532)    
    print('Resetting with red light\n...')
    red_scan(x_voltages_r, y_voltages_r, z_center)
 
    # now pulse the green at the center of the scan for a short time         
    print('Pulsing green light for {} s'.format(green_pulse_time))
    tool_belt.set_xyz(cxn, [x_center, y_center, z_center])
    # Use two methods to pulse the green light, depending on pulse length
    if green_pulse_time < 1:
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
                      aom_ao_589_pwr, apd_indices, 589, save_data=True, plot_data=True) 

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
               'image_range': image_range,
               'image_range-units': 'V',
               'num_steps': num_steps,
               'reset_range': reset_range,
               'reset_range-units': 'V',
               'num_steps_reset': num_steps_reset,
               'green_pulse_time': int(green_pulse_time*10**9),
               'green_pulse_time-units': 'ns',
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
  
def do_green_then_yellow_scan(cxn, nv_sig, green_pulse_time, wait_time = 0):
    aom_ao_589_pwr = nv_sig['am_589_power']
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
    
    green_then_yellow_scan(x_voltages_r, y_voltages_r, z_center)   
    #park red beam, then measure with yellow. 
    # Set up a sequence to do this?
    
# %%
if __name__ == '__main__':
    sample_name = 'goeppert-mayer'
    
    nv2 = { 'coords':[ 0.055, 0.170,  5.6],
            'name': '{}-nv2'.format(sample_name),
            'expected_count_rate': 152, 'nd_filter': 'nd_0',
            'pulsed_readout_dur': 300,
            'pulsed_SCC_readout_dur': 50*10**6, 'am_589_power': 0.45, 
            'pulsed_initial_ion_dur': 25*10**3,
            'pulsed_shelf_dur': 200, 
            'am_589_shelf_power': 0.35,
            'pulsed_ionization_dur': 500*10**3, 'cobalt_638_power': 160, 
            'pulsed_reionization_dur': 100*10**3, 'cobalt_532_power': 19, 
            'magnet_angle': 0,
            "resonance_LOW": 2.7666,"rabi_LOW": 146.2, "uwave_power_LOW": 9.0,
            "resonance_HIGH": 2.9774,"rabi_HIGH": 95.2,"uwave_power_HIGH": 10.0} 
    
    nv_sig = nv2
#    green_pulse_time_list = numpy.array([0.1, 1, 5, 10, 25, 50, 75, 100, 250 ,1000]) # 60 mW, 16 mW, 4 mW
    green_pulse_time_list = numpy.array([100, 250 ,1000]) # 60 mW, 16 mW, 4 mW
  
#    green_pulse_time_list = numpy.array([0.001, 0.005,0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1 ]) # 60 mW, 16 mW, 4 mW
        
#    green_pulse_time_list = [100]
    
    for t in green_pulse_time_list:
        with labrad.connect() as cxn:         
            main(cxn, nv_sig, t)


# %%
            
#    ref_file_list = ['2020_09_14-11_15_28-hopper-ensemble', '2020_09_14-11_36_08-hopper-ensemble',
#                     '2020_09_14-11_56_30-hopper-ensemble', '2020_09_14-12_17_11-hopper-ensemble',
#                     '2020_09_14-12_38_09-hopper-ensemble', '2020_09_14-13_00_02-hopper-ensemble',
#                     '2020_09_14-13_22_32-hopper-ensemble', '2020_09_14-13_46_00-hopper-ensemble',
#                     '2020_09_14-14_12_10-hopper-ensemble', '2020_09_14-14_53_38-hopper-ensemble']
#    
#    sig_file_list = ['2020_09_14-11_25_45-hopper-ensemble', '2020_09_14-11_46_20-hopper-ensemble',
#                     '2020_09_14-12_06_45-hopper-ensemble', '2020_09_14-12_27_32-hopper-ensemble',
#                     '2020_09_14-12_48_50-hopper-ensemble', '2020_09_14-13_11_10-hopper-ensemble',
#                     '2020_09_14-13_34_00-hopper-ensemble', '2020_09_14-13_57_54-hopper-ensemble',
#                     '2020_09_14-14_26_37-hopper-ensemble', '2020_09_14-15_20_19-hopper-ensemble']
#    
#    dif_file_list = ['2020_09_14-11_25_47-hopper-ensemble_dif', '2020_09_14-11_46_22-hopper-ensemble_dif',
#                     '2020_09_14-12_06_47-hopper-ensemble_dif', '2020_09_14-12_27_34-hopper-ensemble_dif',
#                     '2020_09_14-12_48_52-hopper-ensemble_dif', '2020_09_14-13_11_12-hopper-ensemble_dif',
#                     '2020_09_14-13_34_02-hopper-ensemble_dif', '2020_09_14-13_57_57-hopper-ensemble_dif',
#                     '2020_09_14-14_26_40-hopper-ensemble_dif', '2020_09_14-15_20_22-hopper-ensemble_dif']
#
#    for i in range(len(ref_file_list)) :
#        ref_file = ref_file_list[i]
#        sig_file = sig_file_list[i]
#        dif_file = dif_file_list[i]
#        green_pulse_time = green_pulse_time_list[i]
#    
#        data = tool_belt.get_raw_data('image_sample/branch_Spin_to_charge/2020_09/sig', sig_file)
#        sig_img_array =numpy.array(data['img_array'])
#        
#        
#        data = tool_belt.get_raw_data('image_sample/branch_Spin_to_charge/2020_09/ref', ref_file)
#        ref_img_array = numpy.array(data['img_array'])
#        
#        data = tool_belt.get_raw_data('image_sample/branch_Spin_to_charge/2020_09/dif', dif_file)
#        green_pulse_time = data['green_pulse_time']/10**9
#        image_range = data['image_range']
#        x_voltages = data['x_voltages']
#        y_voltages = data['y_voltages']
#        readout = data['readout']
#        wait_time = data['wait_time']
#        opt_volt = data['green_optical_voltage']
#        opt_power = data['green_opt_power']
#        
#    
#        dif_img_array = sig_img_array - ref_img_array
#        
#        title = 'Yellow scan (with/without green pulse)\nGreen pulse {} s'.format(green_pulse_time) 
#        fig = plot_dif_fig([0,0,5.0], x_voltages,image_range,  dif_img_array, readout, title )
#        
#        # Save data
#        timestamp = tool_belt.get_time_stamp()
#    
#        rawData = {'timestamp': timestamp,
#                   'nv_sig': nv_sig,
#                   'nv_sig-units': tool_belt.get_nv_sig_units(),
#                   'image_range': image_range,
#                   'image_range-units': 'V',
#                   'num_steps': num_steps,
#                   'reset_range': reset_range,
#                   'reset_range-units': 'V',
#                   'num_steps_reset': num_steps_reset,
#                   'green_pulse_time': green_pulse_time*10**9,
#                   'green_pulse_time-units': 'ns',
#                   'wait_time': int(wait_time),
#                   'wait_time-units': 's',
#                   'green_optical_voltage': opt_volt,
#                   'green_optical_voltage-units': 'V',
#                   'green_opt_power': opt_power,
#                   'green_opt_power-units': 'mW',
#                   'readout': readout,
#                   'readout-units': 'ns',
#                   'x_voltages': x_voltages,
#                   'x_voltages-units': 'V',
#                   'y_voltages': y_voltages,
#                   'y_voltages-units': 'V',
#                   'ref_img_array': ref_img_array.tolist(),
#                   'ref_img_array-units': 'counts',
#                   'sig_img_array': sig_img_array.tolist(),
#                   'sig_img_array-units': 'counts',
#                   'dif_img_array': dif_img_array.tolist(),
#                   'dif_img_array-units': 'counts'}
#    
#        filePath = tool_belt.get_file_path('image_sample', timestamp, nv_sig['name'])
#        tool_belt.save_raw_data(rawData, filePath + '_dif_alt')
#    
#        tool_belt.save_figure(fig, filePath + '_dif_alt')   
        
        