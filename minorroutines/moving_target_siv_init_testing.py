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

reset_range = 0.3#2.5
num_steps_reset = int(75 * reset_range)
image_range = 0.45#2.5
num_steps = int(225 * image_range) 
#num_steps = 90
apd_indices = [0]

#green_reset_power = 0.65#0.6075
green_pulse_power = 0.65
green_image_power = 0.65

# %%

# %%      
def main(cxn, base_sig, optimize_coords, center_coords, reset_coords, pulse_coord_list, 
         center_pulse_time, pulse_time, init_color, pulse_color, readout_color,
         siv_state, boo = False):
    am_589_power = base_sig['am_589_power']
    readout = base_sig['pulsed_SCC_readout_dur']
#    color_filter = base_sig['color_filter'] 
    
    image_sig = copy.deepcopy(base_sig)
    image_sig['coords'] = center_coords
    image_sig['ao_515_pwr'] = green_image_power
    
    optimize_sig = copy.deepcopy(base_sig)
    optimize_sig['coords'] = optimize_coords
    
    
    
    wiring = tool_belt.get_pulse_streamer_wiring(cxn)
#    pulser_wiring_green = wiring['do_532_aom']
    pulser_wiring_green = wiring['ao_515_laser']
#    pulser_wiring_yellow = wiring['ao_589_aom']
    pulser_wiring_red = wiring['do_638_laser']
#    pulser_wiring_red = wiring['ao_638_laser']

    shared_params = tool_belt.get_shared_parameters_dict(cxn)    
    laser_515_delay = shared_params['515_laser_delay']
    laser_589_delay = shared_params['589_aom_delay']
    laser_638_delay = shared_params['638_DM_laser_delay']
#    laser_638_delay = shared_params['638_AM_laser_delay']
    
    if pulse_color == 532 or pulse_color == '515a':
        direct_wiring = pulser_wiring_green
        laser_delay = laser_515_delay
    elif pulse_color == 589:
#        direct_wiring = pulser_wiring_yellow
        laser_delay = laser_589_delay
    elif pulse_color == 638:
        direct_wiring = pulser_wiring_red
        laser_delay = laser_638_delay
        

    optimize.main_with_cxn(cxn, optimize_sig, apd_indices, '515a', disable=False) 
    start=time.time()   
    if siv_state == 'bright':
        
        print('Resetting with {} nm light for bright state\n...'.format(init_color))
        reset_sig = copy.deepcopy(base_sig)
        reset_sig['coords'] = reset_coords
        reset_sig['ao_515_pwr'] = 0.6240
        _,_,_ = image_sample.main(reset_sig, 0.5, 0.5, int(70 * 0.5), 
                          apd_indices, init_color,readout = 2*10**7,   save_data=False, plot_data=False)
    elif siv_state == 'dark':
        print('Resetting with {} nm light for dark state\n...'.format(init_color))
        reset_sig = copy.deepcopy(base_sig)
        reset_sig['coords'] = reset_coords
        reset_sig['ao_515_pwr'] = 0.6035
        _,_,_ = image_sample.main(reset_sig, 0.5, 0.5, int(70 * 0.5), 
                          apd_indices, init_color,readout = 2*10**7, save_data=False, plot_data=False) 
        _,_,_ = image_sample.main(reset_sig, 0.5, 0.5, int(70 * 0.5), 
                          apd_indices, init_color,readout = 2*10**7, save_data=False, plot_data=False)  
        _,_,_ = image_sample.main(reset_sig, 0.5, 0.5, int(70 * 0.5), 
                          apd_indices, init_color,readout = 2*10**7, save_data=False, plot_data=False) 
        
    end = time.time()
    print('Reset {:.1f} s'.format(end-start))
    
    for i in range(len(pulse_coord_list)):
        pulse_coords = pulse_coord_list[i]
        adj_coords = (numpy.array(pulse_coords) + \
                  numpy.array(tool_belt.get_drift())).tolist()
        x_pulse, y_pulse, z_pulse = adj_coords  
        
        pulse_sig = copy.deepcopy(base_sig)
        pulse_sig['coords'] = pulse_coords
        pulse_sig['ao_515_pwr'] = green_pulse_power
    
        # now pulse at the center of the scan for a short time         
        print('Pulsing {} nm light for {} s (SiV reset)'.format(pulse_color, pulse_time))
        tool_belt.set_xyz(cxn, [x_pulse, y_pulse, z_pulse])
        # Use two methods to pulse the light, depending on pulse length
        if pulse_time < 1.1:
            seq_args = [laser_delay, int(pulse_time*10**9),am_589_power,green_pulse_power,  pulse_color]   
            seq_args_string = tool_belt.encode_seq_args(seq_args)            
            cxn.pulse_streamer.stream_immediate('simple_pulse.py', 1, seq_args_string)   
        else:
            if pulse_color == 532 or pulse_color==638:
                cxn.pulse_streamer.constant([direct_wiring], 0.0, 0.0)
            elif pulse_color == 589:
                cxn.pulse_streamer.constant([], 0.0, am_589_power)
            elif pulse_color =='515a':
                cxn.pulse_streamer.constant([], green_pulse_power, 0)
            time.sleep(pulse_time)
        cxn.pulse_streamer.constant([], 0.0, 0.0)     
        
        adj_coords = (numpy.array(center_coords) + \
                      numpy.array(tool_belt.get_drift())).tolist()
        x_center, y_center, z_center = adj_coords  
        
        # now pulse at the center of the scan for the nv reset       
        print('Pulsing {} nm light for {} us on center (NV init)'.format(pulse_color, center_pulse_time*10**6))
        tool_belt.set_xyz(cxn, [x_center, y_center, z_center])
        # Use two methods to pulse the light, depending on pulse length
        if pulse_time < 1.1:
            seq_args = [laser_delay, int(center_pulse_time*10**9),am_589_power,green_pulse_power,  pulse_color]   
            seq_args_string = tool_belt.encode_seq_args(seq_args)            
            cxn.pulse_streamer.stream_immediate('simple_pulse.py', 1, seq_args_string)   
        else:
            if pulse_color == 532 or pulse_color==638:
                cxn.pulse_streamer.constant([direct_wiring], 0.0, 0.0)
            elif pulse_color == 589:
                cxn.pulse_streamer.constant([], 0.0, am_589_power)
            elif pulse_color =='515a':
                cxn.pulse_streamer.constant([], green_pulse_power, 0)
            time.sleep(center_pulse_time)
        cxn.pulse_streamer.constant([], 0.0, 0.0)  
        
        remote_pulse_time = 0*10**6/10**9
        # now pulse at the center of the scan for the nv rese        
        print('Pulsing {} nm light for {} ms (remote pulse)'.format(pulse_color, remote_pulse_time))
        tool_belt.set_xyz(cxn, [x_pulse, y_pulse, z_pulse])
        # Use two methods to pulse the light, depending on pulse length
        if pulse_time < 1.1:
            seq_args = [laser_delay, int(remote_pulse_time*10**9),am_589_power,green_pulse_power,  pulse_color]   
            seq_args_string = tool_belt.encode_seq_args(seq_args)            
            cxn.pulse_streamer.stream_immediate('simple_pulse.py', 1, seq_args_string)   
        else:
            if pulse_color == 532 or pulse_color==638:
                cxn.pulse_streamer.constant([direct_wiring], 0.0, 0.0)
            elif pulse_color == 589:
                cxn.pulse_streamer.constant([], 0.0, am_589_power)
            elif pulse_color =='515a':
                cxn.pulse_streamer.constant([], green_pulse_power, 0)
            time.sleep(center_pulse_time)
        cxn.pulse_streamer.constant([], 0.0, 0.0)  
        
    if boo:
        print('Resetting with {} nm light for bright state\n...'.format(init_color))
        reset_sig = copy.deepcopy(base_sig)
        reset_sig['coords'] = reset_coords
        reset_sig['ao_515_pwr'] = 0.6240
        _,_,_ = image_sample.main(reset_sig, 0.5, 0.5, int(70 * 0.5), 
                          apd_indices, init_color,readout = 10**7,   save_data=False, plot_data=False)
        
    # collect an image under yellow after green pulse
    print('Imaging {} nm light\n...'.format(readout_color))
    sig_img_array, x_voltages, y_voltages = image_sample.main(image_sig, image_range, image_range, num_steps, 
                      apd_indices, readout_color,readout = readout,save_data=True, plot_data=True) 
    avg_counts = numpy.average(sig_img_array)
    print(avg_counts)

    
    return  
# %%
if __name__ == '__main__':
    sample_name = 'goeppert-mayer'
    
    base_sig = { 'coords':[],
            'name': '{}'.format(sample_name),
            'expected_count_rate': 50, 'nd_filter': 'nd_1.0',
#            'color_filter': '635-715 bp',
            'color_filter': '715 lp',
            'pulsed_readout_dur': 300,
            'pulsed_SCC_readout_dur': 2*10**7, 'am_589_power': 0.25, 
            'pulsed_initial_ion_dur': 25*10**3,
            'pulsed_shelf_dur': 200, 
            'am_589_shelf_power': 0.35,
            'pulsed_ionization_dur': 10**3, 'cobalt_638_power': 10, 
            'pulsed_reionization_dur': 100*10**3, 'cobalt_532_power':10, 
            'ao_515_pwr': 0.6350,
            'magnet_angle': 0,
            "resonance_LOW": 2.7,"rabi_LOW": 146.2, "uwave_power_LOW": 9.0,
            "resonance_HIGH": 2.9774,"rabi_HIGH": 95.2,"uwave_power_HIGH": 10.0}   
    expected_count_list = [55, 45, 50, 45, 45, 45, 50, 50, 50, 60] # 3/30/21
    start_coords_list = [
[0.046, 0.019, 5.20],
[-0.057, 0.002, 5.14],
[0.051, 0.077, 5.18],
[-0.043, -0.047, 5.15],
[0.047, 0.023, 5.21],
[0.098, -0.105, 5.15],
[0.019, 0.053, 5.19],
[0.015, -0.134, 5.19],
[0.077, -0.167, 5.19],
]
    
#    init_time = 10**7
    
    init_color = '515a'
    pulse_color = '515a'
    readout_color = 589
    
    
    center_coords =[0,0, 5.18]
    optimize_coords =[0.051, 0.077, 5.18]
#    center_coords = pulse_coords
    
#    reset_list = [5]
    
    pulse_coords_list = [[0.1,  0.1, 5.18]]
    pulse_time = 0*10**6/10**9
    center_pulse_time = 0*10**3/10**9
    with labrad.connect() as cxn:
#        for z in numpy.linspace(4.98, 5.38, 3):
            reset_coords = [0,0, 5.18]
#            center_coords = [0,0, 5.18]
            main(cxn, base_sig, optimize_coords, center_coords, reset_coords,
                           pulse_coords_list, center_pulse_time, pulse_time, init_color, 
                           pulse_color, readout_color,  siv_state = 'bright', boo = False)
            main(cxn, base_sig, optimize_coords, center_coords, reset_coords,
                       pulse_coords_list, center_pulse_time, pulse_time, init_color, 
                       pulse_color, readout_color,  siv_state = 'dark', boo = False)
#            main(cxn, base_sig, optimize_coords, center_coords, reset_coords,
#                           pulse_coords_list, center_pulse_time, pulse_time, init_color, 
#                           pulse_color, readout_color,  siv_state = 'bright')
        
        

        

            

            


        
        