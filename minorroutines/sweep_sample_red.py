# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 16:26:33 2020

Scan over a sample area of the diamond with red light. No need to record 
the counts

@author: agardill
"""
# %%
import utils.tool_belt as tool_belt
import labrad
import numpy

# %%
def main(nv_sig, x_range, num_steps):
    with labrad.connect() as cxn:
        apd_indices = [0]
        delay = int(0.5 * 10**6)
        shared_params = tool_belt.get_shared_parameters_dict(cxn)
        readout = shared_params['continuous_readout_dur']
        aom_ao_589_pwr = nv_sig['am_589_power']
    
    
        adj_coords = (numpy.array(nv_sig['coords']) + \
                      numpy.array(tool_belt.get_drift())).tolist()
        x_center, y_center, z_center = adj_coords
 
        y_range = x_range

        total_num_samples = num_steps**2  
        
        seq_args = [delay, readout, aom_ao_589_pwr, apd_indices[0], 638]
        seq_args_string = tool_belt.encode_seq_args(seq_args)
        ret_vals = cxn.pulse_streamer.stream_load('simple_readout.py',
                                                  seq_args_string)
        period = ret_vals[0]
    
        # Initialize at the passed coordinates
    
        tool_belt.set_xyz(cxn, [x_center, y_center, z_center])
    
        # Set up the galvo
    
        x_voltages, y_voltages = cxn.galvo.load_sweep_scan(x_center, y_center,
                                                           x_range, y_range,
                                                           num_steps, period)
    
        x_num_steps = len(x_voltages)
        x_low = x_voltages[0]
        x_high = x_voltages[x_num_steps-1]
        y_num_steps = len(y_voltages)
        y_low = y_voltages[0]
        y_high = y_voltages[y_num_steps-1]
    
        pixel_size = x_voltages[1] - x_voltages[0]
    
        # Scan over the sample with red
    
        cxn.pulse_streamer.stream_start(total_num_samples)
        
# %%
    
if __name__ == '__main__':
    sample_name = 'hopper'
    ensemble = { 'coords': [0.183, 0.043, 5.00],
            'name': '{}-ensemble'.format(sample_name),
            'expected_count_rate': 1000, 'nd_filter': 'nd_0.5',
            'pulsed_readout_dur': 300,
            'pulsed_SCC_readout_dur': 1*10**7, 'am_589_power': 0.3, 
            'pulsed_initial_ion_dur': 50*10**3,
            'pulsed_shelf_dur': 0, 'am_589_shelf_power': 0,#50/0.3,
            'pulsed_ionization_dur': 450, 'cobalt_638_power': 160, 
            'pulsed_reionization_dur': 10*10**3, 'cobalt_532_power': 8, 
            'magnet_angle': 0,
            'resonance_LOW': 2.8059, 'rabi_LOW': 187.8, 'uwave_power_LOW': 9.0, 
            'resonance_HIGH': 2.9366, 'rabi_HIGH': 247.4, 'uwave_power_HIGH': 10.0}   
    nv_sig = ensemble
    
    main(nv_sig, 0.1, 60)