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

reset_range = 0.5#2.5
num_steps_reset = int(75 * reset_range)
image_range = 0.4#2.5
num_steps = int(225 * image_range) 
apd_indices = [0]

#green_reset_power = 0.65#0.6075
green_pulse_power = 0.65
green_image_power = 0.65

# %%

# %%      
def main(cxn, base_sig, optimize_coords, center_coords, reset_coords, pulse_coords, 
         pulse_time, init_time, init_color, pulse_color, readout_color, 
         pulse_repeat, siv_state):
    am_589_power = base_sig['am_589_power']
    readout = base_sig['pulsed_SCC_readout_dur']
    color_filter = base_sig['color_filter'] 
    
    image_sig = copy.deepcopy(base_sig)
    image_sig['coords'] = center_coords
    image_sig['ao_515_pwr'] = green_image_power
    
    optimize_sig = copy.deepcopy(base_sig)
    optimize_sig['coords'] = optimize_coords
    
    pulse_sig = copy.deepcopy(base_sig)
    pulse_sig['coords'] = pulse_coords
    pulse_sig['ao_515_pwr'] = green_pulse_power
    
    
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
        

#    optimize.main_xy_with_cxn(cxn, optimize_sig, apd_indices, 532, disable=disable_optimize)    
    adj_coords = (numpy.array(pulse_coords) + \
                  numpy.array(tool_belt.get_drift())).tolist()
    x_center, y_center, z_center = adj_coords  
     
    if siv_state == 'bright':
        
        print('Resetting with {} nm light for bright state\n...'.format(init_color))
        reset_sig = copy.deepcopy(base_sig)
        reset_sig['coords'] = reset_coords
        reset_sig['ao_515_pwr'] = 0.6358
        _,_,_ = image_sample.main(reset_sig, 0.5, 0.5, int(70 * 0.5), 
                          apd_indices, init_color,readout = 10**7,   save_data=False, plot_data=False) 
    elif siv_state == 'dark':
        print('Resetting with {} nm light for dark state\n...'.format(init_color))
        reset_sig = copy.deepcopy(base_sig)
        reset_sig['coords'] = reset_coords
        reset_sig['ao_515_pwr'] = 0.6085
        _,_,_ = image_sample.main(reset_sig, 0.5, 0.5, int(70 * 0.3), 
                          apd_indices, init_color,readout = 10**7,   save_data=False, plot_data=False) 
    
    for i in range(pulse_repeat):
        # now pulse at the center of the scan for a short time         
        print('Pulsing {} nm light for {} s'.format(pulse_color, pulse_time))
        tool_belt.set_xyz(cxn, [x_center, y_center, z_center])
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
        
        time.sleep(0.1)
     


    # collect an image under yellow after green pulse
    print('Imaging {} nm light\n...'.format(readout_color))
    sig_img_array, x_voltages, y_voltages = image_sample.main(image_sig, image_range, image_range, num_steps, 
                      apd_indices, readout_color,readout = readout,save_data=True, plot_data=True) 

    
    return  
# %%
if __name__ == '__main__':
    sample_name = 'goeppert-mayer'
    
    base_sig = { 'coords':[],
            'name': '{}'.format(sample_name),
            'expected_count_rate': 38, 'nd_filter': 'nd_1.0',
#            'color_filter': '635-715 bp',
            'color_filter': '715 lp',
            'pulsed_readout_dur': 300,
            'pulsed_SCC_readout_dur': 2*10**7, 'am_589_power': 0.3, 
            'pulsed_initial_ion_dur': 25*10**3,
            'pulsed_shelf_dur': 200, 
            'am_589_shelf_power': 0.35,
            'pulsed_ionization_dur': 10**3, 'cobalt_638_power': 10, 
            'pulsed_reionization_dur': 100*10**3, 'cobalt_532_power':80, 
            'magnet_angle': 0,
            "resonance_LOW": 2.7,"rabi_LOW": 146.2, "uwave_power_LOW": 9.0,
            "resonance_HIGH": 2.9774,"rabi_HIGH": 95.2,"uwave_power_HIGH": 10.0}   
    start_coords_list =[    
    [0.309, 0.334, 4.90],
[0.184, 0.342, 4.79],
[-0.038, 0.294, 4.72],
[-0.048, 0.260, 4.80],
[-0.074, 0.264, 4.83],
[0.325, 0.272, 4.75],
[0.322, 0.203, 4.79],
[-0.067, 0.173, 4.73],
[0.194, 0.123, 4.88],
[0.181, 0.122, 4.78],
[0.025, 0.080, 4.79],
[0.120, 0.045, 4.79],
[-0.018, 0.046, 4.76],
[0.055, -0.126, 4.79],
[0.396, -0.195, 4.83],
[0.128, -0.197, 4.75],
[0.392, -0.304, 4.80],
[0.392, -0.303, 4.81],
[-0.203, -0.330, 4.81],
[0.251, -0.385, 4.76],
]
    expected_count_list = [60, 40, 38, 36, 38, 40, 55, 48, 42, 36, 38, 30, 
                           46, 53, 44, 70, 50, 40, 40, 32] # 2/11/21
    
    
#    init_time = 10**7
    
    init_color = '515a'
    pulse_color = '515a'
    readout_color = 589
    
#    center_coords = [0.315, 0.222, 4.79]
    center_coords = [-0.25, -0.25, 4.79]
    reset_coords =center_coords
    optimize_coords =center_coords
#    center_coords = pulse_coords
    
    reset_list = [5]
    with labrad.connect() as cxn:
        pulse_time = 10**5/10**9
        pulse_coords =    center_coords
        for i in [0]:
            init_time = int(i*10**7)
            main(cxn, base_sig, optimize_coords, center_coords, reset_coords,
                           pulse_coords, pulse_time, init_time, init_color, 
                           pulse_color, readout_color, i, siv_state = 'dark')
            main(cxn, base_sig, optimize_coords, center_coords, reset_coords,
                           pulse_coords, pulse_time, init_time, init_color, 
                           pulse_color, readout_color, i, siv_state = 'bright')
#            main(cxn, base_sig, optimize_coords, center_coords, reset_coords,
#                           pulse_coords, pulse_time, init_time, init_color, 
#                           pulse_color, readout_color, 1, siv_state = None)

        

            

            


        
        