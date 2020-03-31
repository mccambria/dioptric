# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 16:30:14 2020

A file to optimize both the power and readout duration for the 589 laser

@author: agardill
"""

import utils.tool_belt as tool_belt
import numpy
import matplotlib.pyplot as plt
import minorroutines.time_resolved_readout as time_resolved_readout

# %%
    
def optimize_readout_power(nv_sig, apd_indices, num_reps, num_runs,
                           num_bins, yellow_power_list):
    # Create some lists to fill
    power_list = []
    g_y_counts_s_list = []
    r_y_counts_s_list = []
    # Get the starting timestamp for incriment data
    start_timestamp = tool_belt.get_time_stamp()
    # get the yellow readout time from the nv_sig
    illumination_time = nv_sig['pulsed_SCC_readout_dur']
    
    for p in range(len(yellow_power_list)):
        # set the yellow power via AOM
        nv_sig['am_589_power'] = yellow_power_list[p]
        # Set the init pulse (green) to the reionization pulse in nv_sig
        init_pulse_duration = nv_sig['pulsed_reionization_dur']
        
        print('589 AM set to: ' + str(nv_sig['am_589_power']))
        print('Measuring Green/Yellow')
        
        # Run time resolved readout of G/Y
        bin_centers, binned_samples, illum_optical_power_mW = \
                  time_resolved_readout.main(nv_sig, apd_indices, 
                  illumination_time, init_pulse_duration, 532, 589,
                  num_reps, num_runs, num_bins, plot= True)
        
        integrated_counts = numpy.trapz(binned_samples, bin_centers)
        
        power_list.append(illum_optical_power_mW)
        g_y_counts_s_list.append(integrated_counts)
        
        # Set the init pulse (red) to the ionization pulse in nv_sig
        init_pulse_duration = nv_sig['pulsed_ionization_dur']
        print('Measuring Red/Yellow')
        bin_centers, binned_samples, illum_optical_power_mW = time_resolved_readout.main(nv_sig, 
                  apd_indices, illumination_time, init_pulse_duration,
                  638, 589,
                  num_reps, num_runs, num_bins, plot = True)
        integrated_counts = numpy.trapz(binned_samples, bin_centers)
        
        r_y_counts_s_list.append(integrated_counts)
        
        #save data incrimentally
        raw_data = {'start_timestamp': start_timestamp,
                    'nv_sig': nv_sig,
                    'nv_sig-units': tool_belt.get_nv_sig_units(),
                    '589_power_list': yellow_power_list,
                    'num_bins': num_bins,
                    'num_reps': num_reps,
                    'num_runs': num_runs,
                    'g_y_counts_s_list': g_y_counts_s_list,
                    'g_y_counts_s_list-units': 'counts*ns',
                    'r_y_counts_s_list': r_y_counts_s_list,
                    'r_y_counts_s_list-units': 'counts*ns',
                    'power_list': power_list,
                    'power_list-units': 'mW'
                    }

        # This will continuously be the same file path so we will overwrite
        # the existing file with the latest version
        file_path = tool_belt.get_file_path(__file__, start_timestamp,
                                            nv_sig['name'], 'incremental')
        tool_belt.save_raw_data(raw_data, file_path + 'opt_power')
        
    difference = numpy.array(g_y_counts_s_list) - numpy.array(r_y_counts_s_list)
    text = 'Illumnation time: ' + '%.1f'%(illumination_time/10**3) + ' us'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    
    ind_fig, ax = plt.subplots(1, 1, figsize=(10, 8.5))

    ax.text(0.55, 0.85, text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)
    ax.plot(power_list, g_y_counts_s_list, 'g', label = 'Green/Yellow')
    ax.plot(power_list, r_y_counts_s_list, 'r', label = 'Red/Yellow')
    ax.set_xlabel('589 power (mW)')
    ax.set_ylabel('Area under time_resolved_readout curves (count*ns)')
    ax.legend()
    
    dif_fig, ax = plt.subplots(1, 1, figsize=(10, 8.5))
    ax.text(0.55, 0.85, text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)
    ax.plot(power_list, difference)
    ax.set_xlabel('589 power (mW)')
    ax.set_ylabel('Subtracted area under time_resolved_readout curves (counts*ns)')
    
    
    timestamp = tool_belt.get_time_stamp()
    raw_data = {'timestamp': timestamp,
                'nv_sig': nv_sig,
                'nv_sig-units': tool_belt.get_nv_sig_units(),
                '589_power_list': yellow_power_list,
                'num_bins': num_bins,
                'num_reps': num_reps,
                'num_runs': num_runs,
                'g_y_counts_s_list': g_y_counts_s_list,
                'g_y_counts_s_list-units': 'counts*ns',
                'r_y_counts_s_list': r_y_counts_s_list,
                'r_y_counts_s_list-units': 'counts*ns',
                'power_list': power_list,
                'power_list-units': 'mW'
                }
    
    file_path = tool_belt.get_file_path(__file__, timestamp, nv_sig['name'])
    tool_belt.save_figure(ind_fig, str(file_path + 'opt_power_ind_fig'))
    tool_belt.save_figure(dif_fig, str(file_path + 'opt_power_dif_fig'))
    tool_belt.save_raw_data(raw_data, str(file_path + 'opt_power'))
    print('Run complete!')
    
# %% Untested 3/27
    
def optimize_readout_time(nv_sig, apd_indices, num_reps, num_runs,
                           num_bins, readout_time_list):
    g_y_counts_s_list = []
    r_y_counts_s_list = []
    start_timestamp = tool_belt.get_time_stamp()
    
    for p in range(len(readout_time_list)):
        illumination_time = readout_time_list[p]
        init_pulse_duration = nv_sig['pulsed_reionization_dur']
        print('589 pulse length set to: ' + str(illumination_time/10**3) + ' us')
        print('Measuring Green/Yellow')
        bin_centers, binned_samples, illum_optical_power_mW = \
                  time_resolved_readout.main(nv_sig, apd_indices, 
                  illumination_time, init_pulse_duration,
                  532, 589,
                  num_reps, num_runs, num_bins, plot= False)
        
        integrated_counts = numpy.trapz(binned_samples, bin_centers)
        g_y_counts_s_list.append(integrated_counts)
        
        init_pulse_duration = nv_sig['pulsed_reionization_dur']
        print('Measuring Red/Yellow')
        bin_centers, binned_samples, illum_optical_power_mW = time_resolved_readout.main(nv_sig, 
                  apd_indices, illumination_time, init_pulse_duration,
                  638, 589,
                  num_reps, num_runs, num_bins, plot = False)
        integrated_counts = numpy.trapz(binned_samples, bin_centers)
        
        r_y_counts_s_list.append(integrated_counts)
        
        #save data incrimentally
        raw_data = {'start_timestamp': start_timestamp,
                    'nv_sig': nv_sig,
                    'nv_sig-units': tool_belt.get_nv_sig_units(),
                    'readout_time_list': readout_time_list,
                    'readout_time_list-units': 'ns',
                    'num_bins': num_bins,
                    'num_reps': num_reps,
                    'num_runs': num_runs,
                    'g_y_counts_s_list': g_y_counts_s_list,
                    'g_y_counts_s_list-units': 'counts*ns',
                    'r_y_counts_s_list': r_y_counts_s_list,
                    'r_y_counts_s_list-units': 'counts*ns'
                    }
        # This will continuously be the same file path so we will overwrite
        # the existing file with the latest version
        file_path = tool_belt.get_file_path(__file__, start_timestamp,
                                            nv_sig['name'], 'incremental')
        tool_belt.save_raw_data(raw_data, file_path + 'opt_power')
        
    difference = numpy.array(g_y_counts_s_list) - numpy.array(r_y_counts_s_list)
    text = 'Illumnation time: ' + '%.1f'%(illumination_time/10**3) + ' us'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    
    ind_fig, ax = plt.subplots(1, 1, figsize=(10, 8.5))

    ax.text(0.55, 0.85, text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)
    ax.plot(numpy.array(readout_time_list)/10**3, g_y_counts_s_list, 'g', label = 'Green/Yellow')
    ax.plot(numpy.array(readout_time_list)/10**3, r_y_counts_s_list, 'r', label = 'Red/Yellow')
    ax.set_xlabel('589 nm pulse duration (us)')
    ax.set_ylabel('Area under time_resolved_readout curves (count*ns)')
    ax.legend()
    
    dif_fig, ax = plt.subplots(1, 1, figsize=(10, 8.5))
    ax.text(0.55, 0.85, text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)
    ax.plot(numpy.array(readout_time_list)/10**3, difference)
    ax.set_xlabel('589 nm pulse duration (us)')
    ax.set_ylabel('Subtracted area under time_resolved_readout curves (counts*ns)')
    
    
    timestamp = tool_belt.get_time_stamp()
    raw_data = {'timestamp': timestamp,
                'nv_sig': nv_sig,
                'nv_sig-units': tool_belt.get_nv_sig_units(),
                'readout_time_list': readout_time_list,
                'readout_time_list-units': 'ns',
                'num_bins': num_bins,
                'num_reps': num_reps,
                'num_runs': num_runs,
                'g_y_counts_s_list': g_y_counts_s_list,
                'g_y_counts_s_list-units': 'counts*ns',
                'r_y_counts_s_list': r_y_counts_s_list,
                'r_y_counts_s_list-units': 'counts*ns'
                }
    
    file_path = tool_belt.get_file_path(__file__, timestamp, nv_sig['name'])
    tool_belt.save_figure(ind_fig, str(file_path + 'opt_length_ind_fig'))
    tool_belt.save_figure(dif_fig, str(file_path + 'opt_length_dif_fig'))
    tool_belt.save_raw_data(raw_data, str(file_path + 'opt_length'))
    print('Run complete!')
    
# %%

if __name__ == '__main__':
    apd_indices = [0]
    sample_name = 'hopper'
    ensemble = { 'coords': [0.0, 0.0, 5.00],
            'name': '{}-ensemble'.format(sample_name),
            'expected_count_rate': 1000, 'nd_filter': 'nd_0.5',
            'pulsed_readout_dur': 300,
            'pulsed_SCC_readout_dur': 10**7, 'am_589_power': 0.3, 
            'pulsed_ionization_dur': 500, 'cobalt_638_power': 160, 
            'pulsed_reionization_dur': 10**6, 'cobalt_532_power': 8, 
            'magnet_angle': 0,
            'resonance_LOW': 2.8059, 'rabi_LOW': 173.5, 'uwave_power_LOW': 9.0, 
            'resonance_HIGH': 2.9366, 'rabi_HIGH': 247.4, 'uwave_power_HIGH': 10.0}
    nv_sig = ensemble
    
#    power_list = numpy.linspace(0.1, 0.7, 13).tolist()
    power_list = [0.5, 0.7]
    num_runs = 1
#    init_pulse_duration = 3*10**3
#    init_pulse_duration = 500
    
#    illumination_time = 10*10**6    
    num_reps = 5*10**3
    num_bins = 1000
    optimize_readout_power(nv_sig, apd_indices, num_reps, num_runs,
                           num_bins, power_list)    
  
    