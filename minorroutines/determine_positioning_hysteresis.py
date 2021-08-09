# -*- coding: utf-8 -*-
"""

A routine to measure the hystersis of position equipment. 

Routine starts on a single NV. Routine moves a fixed distance in x, y, and z, 
then returns by that amount plus some amount that is changed.
Then plots the counts collected after moving back versus delta to determine 
hysteresis.

Created on Wed Jul 28 15:56:01

@author: agardill
"""


import utils.tool_belt as tool_belt
import majorroutines.optimize as optimize
import numpy
import time
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import labrad

def linear_line(x, m ,b):
    return m*x + b
# %% Main


def main ( nv_sig, movement_displ, axis_ind,  apd_indices,  plot = True):

    with labrad.connect() as cxn:
        opti_delta = main_with_cxn(cxn,
                    nv_sig, movement_displ, axis_ind, apd_indices,   plot)

    return opti_delta

def main_with_cxn(cxn, nv_sig, movement_displ, axis_ind,  apd_indices,  plot):
    

    if len(apd_indices) > 1:
        msg = 'Currently lifetime only supports single APDs!!'
        raise NotImplementedError(msg)
        
    tool_belt.reset_cfm(cxn)
    
    # Define the parameters to be used in the sequence
    num_steps =25
    config = tool_belt.get_config_dict(cxn)
    
    xy_opti_scan_range =  config['Positioning']['xy_optimize_range']/2
    z_opti_scan_range = config['Positioning']['z_optimize_range']/2
    
    x_units = config['Positioning']['xy_units']
    y_units = config['Positioning']['xy_units']
    z_units = config['Positioning']['z_units']
        
    start_coords = nv_sig['coords']
    
    scan_range_list = [xy_opti_scan_range,
                       xy_opti_scan_range,
                       z_opti_scan_range]
    axis_units = [x_units,
                       y_units,
                       z_units]
    
    # create list of shifts to test in move back to NV
    half_scan_range = scan_range_list[axis_ind]/2
    delta_list = numpy.linspace(-half_scan_range, half_scan_range, num_steps)
    count_ratio_list = []
    
    
    tool_belt.init_safe_stop()
    # Loop
    for n in range(num_steps):
        print('Run: {}'.format(n))
        # Break out of the while if the user says stop
        if tool_belt.safe_stop():
            break
        
        delta = delta_list[n]
        # optimzie on NV, save that exact position
        optimize.main_with_cxn(cxn, nv_sig, apd_indices)
        time.sleep(1)
        
        drift = tool_belt.get_drift()
        
        init_coords = numpy.array(start_coords) + drift
        
        # print(init_coords)
        # record the counts on this spot
        ref = optimize.stationary_count_lite(cxn, nv_sig,  init_coords, config, apd_indices)
        time.sleep(1)
    
    
        shift_coords = init_coords
        shift_coords[axis_ind] = shift_coords[axis_ind] + movement_displ
        # print(shift_coords)
        
        # move in the position some set amount
        tool_belt.set_xyz(cxn, shift_coords)
        time.sleep(1)
        
        # move back same displacement + delta
        return_coords = shift_coords
        return_coords[axis_ind] = return_coords[axis_ind] - (movement_displ + delta)
        
        # print(return_coords)
        # move in the position some set amount
        tool_belt.set_xyz(cxn, return_coords)
        time.sleep(1)
        
        # measure under green light
        sig = optimize.stationary_count_lite(cxn, nv_sig,  return_coords, config, apd_indices)
        
        count_ratio_list.append(sig/ref)
        
        
    tool_belt.reset_cfm(cxn)
    # Plot the counts versus delta
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.plot(delta_list, count_ratio_list, 'b.')
    ax.set_xlabel('delta ({})'.format(axis_units[axis_ind]))
    ax.set_ylabel('sig / ref')
    ax.set_title('Movement {} V in axis {}'.format(movement_displ, axis_ind))
    
    # Fit to a gaussian
    init_fit = [1.0, 0.0, scan_range_list[axis_ind]/2.5, 0.0]
    print(init_fit)
    
    try:
        opti_params, cov_arr = curve_fit(tool_belt.gaussian, 
                  delta_list,count_ratio_list, p0=init_fit)
    except Exception as e:
        print(e)
        return None
    
    lin_radii = numpy.linspace(delta_list[0],
                        delta_list[-1], 100)
    ax.plot(lin_radii,
           tool_belt.gaussian(lin_radii, *opti_params), 'r-')
    text = 'A={:.3f}\n$r_0$={:.5f} {}\n ' \
        '$\sigma$={:.3f} {}\nC={:.3f}'.format(opti_params[0], opti_params[1],
                                              axis_units[axis_ind], opti_params[2],
                                              axis_units[axis_ind], opti_params[3])
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.5, 0.1, text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)
    print(opti_params)
    
    # save data
    timestamp = tool_belt.get_time_stamp()
    raw_data = {'timestamp': timestamp,
                'axis_ind': axis_ind,
            'nv_sig': nv_sig,
            'movement_displ': movement_displ,
            'movement_displ-units': axis_units[axis_ind],
            'delta_list': delta_list.tolist(),
            'delta_list-units': axis_units[axis_ind],
            'count_ratio_list': count_ratio_list,
            'num_steps': num_steps,
            'optimum_delta': opti_params[1],
            'opti_params': opti_params.tolist()
            }

    file_path = tool_belt.get_file_path(__file__, timestamp, nv_sig['name'])
    tool_belt.save_figure(fig, file_path)
    tool_belt.save_raw_data(raw_data, file_path)
    
    
    return opti_params[1]
    

# %% Run the file


if __name__ == '__main__':

    apd_indices = [0]
    sample_name = 'johnson'
    
    # movement_displ = 0.2
    # displacement_list = [0.12]
    displacement_list = numpy.linspace(-0.48, 0.48, 41)
    displacement_list = displacement_list[12:]
    
    nv_sig = { 'coords': [0.021, -0.058, 4.77],
            'name': '{}-nv2_2021_08_04'.format(sample_name),
            'disable_opt': False, 'expected_count_rate': 50,
            'imaging_laser': 'laserglow_532', 'imaging_laser_filter': 'nd_0.5', 'imaging_readout_dur': 1E7,
            'collection_filter': '630_lp', 'magnet_angle': None,
            'resonance_LOW': 2.8012, 'rabi_LOW': 141.5, 'uwave_power_LOW': 15.5,  # 15.5 max
            'resonance_HIGH': 2.9445, 'rabi_HIGH': 191.9, 'uwave_power_HIGH': 14.5}   # 14.5 max 
    
    try:
        for axis_ind in [2]:
            opti_delta_list = []
            for movement_displ in displacement_list:
                opti_delta = None
                # Just keep trying until we get the fit succeeds
                while opti_delta is None:
                    opti_delta = main( nv_sig, movement_displ, axis_ind,  apd_indices)
                opti_delta_list.append(opti_delta)
            print(opti_delta_list)
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            ax.plot(displacement_list, opti_delta_list, 'ro')
            ax.set_xlabel('Displacement in axis {} (V)'.format(axis_ind))
            ax.set_ylabel('Added adjustment to return to original position (V)')
            ax.set_title('Movement in axis {}'.format(axis_ind))
        
            time.sleep(0.01)
            # save data
            timestamp = tool_belt.get_time_stamp()
            raw_data = {'timestamp': timestamp,
                        'axis_ind': axis_ind,
                    'nv_sig': nv_sig,
                    'displacement_list': displacement_list.tolist(),
                    'opti_delta_list':opti_delta_list
                    }
        
            file_path = tool_belt.get_file_path(__file__, timestamp, nv_sig['name'])
            tool_belt.save_figure(fig, file_path)
            tool_belt.save_raw_data(raw_data, file_path)
    finally:
        # Reset our hardware - this should be done in each routine, but
        # let's double check here
        tool_belt.reset_cfm()
        # Kill safe stop
        if tool_belt.check_safe_stop_alive():
            print('\n\nRoutine complete. Press enter to exit.')
            tool_belt.poll_safe_stop()


    ###################### Plotting ##########################
    # file = '2021_07_30-01_20_05-johnson-nv1_2021_07_27'
    # data_path = 'pc_rabi/branch_master/determine_positioning_hysteresis/2021_07'
    # data = tool_belt.get_raw_data(file,data_path )
    # displacement_list = data['displacement_list']
    # opti_delta_list = data['opti_delta_list']
    
    # opti_params, cov_arr = curve_fit(linear_line,
    #               displacement_list,opti_delta_list)
    
    # fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    # ax.plot(displacement_list, opti_delta_list, 'bo')
    
    # lin_x_vals = numpy.linspace(displacement_list[0], displacement_list[-1], 100)
    # ax.plot(lin_x_vals, linear_line(lin_x_vals, *opti_params), 'r-')
    # text = 'y = {:.4f} x + {:.5f}'.format(opti_params[0], opti_params[1])
    # props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    # ax.text(0.5, 0.1, text, transform=ax.transAxes, fontsize=12,
    #         verticalalignment='top', bbox=props)
            
    # ax.set_xlabel('Displacement in axis {} (V)'.format(2))
    # ax.set_ylabel('Added adjustment to return to original position (V)')
    # ax.set_title('Movement in axis {}'.format(2))
    
    