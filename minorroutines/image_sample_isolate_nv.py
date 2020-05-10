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
# %%

scan_range = 0.1
num_steps = 90
apd_indices = [0]
# %%

def main(nv_sig, green_pulse_time):
    aom_ao_589_pwr = nv_sig['am_589_power']
    coords = nv_sig['coords']
    readout = nv_sig['pulsed_SCC_readout_dur']
    
    adj_coords = (numpy.array(nv_sig['coords']) + \
                  numpy.array(tool_belt.get_drift())).tolist()
    x_center, y_center, z_center = adj_coords

    print('Resetting with red light\n...')
    image_sample.main(nv_sig, scan_range, scan_range, num_steps, 
                      aom_ao_589_pwr, apd_indices, 638, save_data=False, plot_data=False) 
    # shine red light
    with labrad.connect() as cxn:       
        cxn.filter_slider_ell9k.set_filter('nd_0.5') 
        tool_belt.set_xyz(cxn, [x_center, y_center, z_center])
        seq_args = [10**3, 100, 638]           
        seq_args_string = tool_belt.encode_seq_args(seq_args)            
        cxn.pulse_streamer.stream_immediate('analog_sequence_test.py', 1, seq_args_string)
        
    # collect an image under yellow right after ionization
    print('Scanning yelow light\n...')
    ref_img_array, x_voltages, y_voltages = image_sample.main(nv_sig, scan_range, scan_range, num_steps, 
                      aom_ao_589_pwr, apd_indices, 589, save_data=True, plot_data=True) 
    
    # shine green light
    print('Pulsing green light for {} us'.format(green_pulse_time/10**3))
    with labrad.connect() as cxn:
#        cxn.pulse_streamer.constant([3], 0.0, 0.0)
#        time.sleep(green_pulse_time/ 10**9)
#        cxn.pulse_streamer.constant([], 0.0, 0.0)
        tool_belt.set_xyz(cxn, [x_center, y_center, z_center])
        seq_args = [green_pulse_time, 100, 532]           
        seq_args_string = tool_belt.encode_seq_args(seq_args)            
        cxn.pulse_streamer.stream_immediate('analog_sequence_test.py', 1, seq_args_string)
        
    # collect an image under yellow after green pulse
    print('Scanning yelow light\n...')
    sig_img_array, x_voltages, y_voltages = image_sample.main(nv_sig, scan_range, scan_range, num_steps, 
                      aom_ao_589_pwr, apd_indices, 589, save_data=True, plot_data=True) 
    
    dif_img_array = sig_img_array - ref_img_array
    x_coord = coords[0]
    half_x_range = scan_range / 2
    x_low = x_coord - half_x_range
    x_high = x_coord + half_x_range
    y_coord = coords[1]
    half_y_range = scan_range / 2
    y_low = y_coord - half_y_range
    y_high = y_coord + half_y_range
    
    dif_img_array_kcps = (dif_img_array / 1000) / (readout / 10**9)
    
    pixel_size = x_voltages[1] - x_voltages[0]
    half_pixel_size = pixel_size / 2
    img_extent = [x_high + half_pixel_size, x_low - half_pixel_size,
                  y_low - half_pixel_size, y_high + half_pixel_size]

    title = 'Yellow scan (green pulse vs after ionization)\nGreen pulse {} us'.format(green_pulse_time/10**3) 
    fig = tool_belt.create_image_figure(dif_img_array_kcps, img_extent, clickHandler=None, title = title)
    # Redraw the canvas and flush the changes to the backend
    fig.canvas.draw()
    fig.canvas.flush_events()  
    
    # Save data
    timestamp = tool_belt.get_time_stamp()

    rawData = {'timestamp': timestamp,
               'nv_sig': nv_sig,
               'nv_sig-units': tool_belt.get_nv_sig_units(),
               'scan_range': scan_range,
               'scan_range-units': 'V',
               'num_steps': num_steps,
               'green_pulse_time': green_pulse_time,
               'green_pulse_time-units': 'ns',
               'readout': readout,
               'readout-units': 'ns',
               'x_voltages': x_voltages.tolist(),
               'x_voltages-units': 'V',
               'y_voltages': y_voltages.tolist(),
               'y_voltages-units': 'V',
               'ref_img_array': ref_img_array.astype(int).tolist(),
               'ref_img_array-units': 'counts',
               'sig_img_array': sig_img_array.astype(int).tolist(),
               'sig_img_array-units': 'counts',
               'dif_img_array': dif_img_array.astype(int).tolist(),
               'dif_img_array-units': 'counts'}

    filePath = tool_belt.get_file_path('image_sample', timestamp, nv_sig['name'])
    tool_belt.save_raw_data(rawData, filePath + '_dif')

    tool_belt.save_figure(fig, filePath + '_dif')
   
# %%
if __name__ == '__main__':
    sample_name = 'hopper'
    ensemble = { 'coords': [0.0, 0.0, 5.00],
            'name': '{}-ensemble'.format(sample_name),
            'expected_count_rate': 1000, 'nd_filter': 'nd_0.5',
            'pulsed_readout_dur': 300,
            'pulsed_SCC_readout_dur': 1*10**7, 'am_589_power': 0.3, 
            'yellow_pol_dur': 2*10**3, 'am_589_pol_power': 0.3,
            'pulsed_initial_ion_dur': 200*10**3,
            'pulsed_shelf_dur': 50, 'am_589_shelf_power': 0.3,
            'pulsed_ionization_dur': 450, 'cobalt_638_power': 160, 
            'pulsed_reionization_dur': 200*10**3, 'cobalt_532_power': 8,
            'ionization_rep': 7,
            'magnet_angle': 0,
            'resonance_LOW': 2.8059, 'rabi_LOW': 187.8, 'uwave_power_LOW': 9.0, 
            'resonance_HIGH': 2.9366, 'rabi_HIGH': 247.4, 'uwave_power_HIGH': 10.0}    
    nv_sig = ensemble
    
#    green_pulse_time_list = [100, 500, 10**3, 5*10**3, 10**4, 5*10**4, 10**5, 5*10**5, 10**6]
    green_pulse_time = 10**6 * 50
    
#    for t in green_pulse_time_list:
        
    main(nv_sig, green_pulse_time)
    
    
        
        
    