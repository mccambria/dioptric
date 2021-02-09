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
#    sample_name = 'goeppert-mayer'
    
    nv19_2021_01_26 = { 'coords':[0.292, -0.370, 5.01], 
            'name': 'goeppert-mayer-nv19_2021_01_26',
            'expected_count_rate': 45,'nd_filter': 'nd_1.0',
            'color_filter': '635-715 bp', 
#            'color_filter': '715 lp',
            'pulsed_readout_dur': 300,
            'pulsed_SCC_readout_dur': 4*10**7,  'am_589_power': 0.3, 
            'pulsed_initial_ion_dur': 25*10**3,
            'pulsed_shelf_dur': 200, 
            'am_589_shelf_power': 0.35,
            'pulsed_ionization_dur': 10**3, 'cobalt_638_power': 130, 
            'pulsed_reionization_dur': 100*10**3, 'cobalt_532_power':10, 
            'magnet_angle': 0,
            "resonance_LOW": 2.7,"rabi_LOW": 146.2, "uwave_power_LOW": 9.0,
            "resonance_HIGH": 2.9774,"rabi_HIGH": 95.2,"uwave_power_HIGH": 10.0}     
    
    init_time = 10**4
    
    init_color = 638
    pulse_color = 532
    readout_color = 589
    
    green_pulse_time_list = numpy.array([10])
#    green_pulse_time_list = numpy.array([250, 5,  10, 25, 50, 75, 100 ,1000]) #  0.6 mW, 2 mW,  4?
#    green_pulse_time_list = numpy.array([0.1]) # 60 mW, 16 mW, 4 mW
  
#    green_pulse_time_list = numpy.array([0.001, 0.005,0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1 ]) # 60 mW, 16 mW, 4 mW
        
#    green_pulse_time_list = [100]

    with labrad.connect() as cxn:
        for t in green_pulse_time_list:
            nv_sig = copy.deepcopy(nv19_2021_01_26)
            nv_sig['color_filter'] = '715 lp' 
            main_moving_target(cxn, nv_sig, t, init_time, init_color, pulse_color, readout_color)
            
            nv_sig = copy.deepcopy(nv19_2021_01_26) 
            nv_sig['color_filter'] = '635-715 bp'
            main_moving_target(cxn, nv_sig, t, init_time, init_color, pulse_color, readout_color)


        
        