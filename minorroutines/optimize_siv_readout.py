# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 10:58:53 2021

An optimize routine to optimize readout in the SiV band.

The idea of this routine is to compare the counts when shining green light on 
the readout spot and also slightly off the readout spot. We've seen that green 
light will essentially ionize SiVs at the illumination point, but wil create 
SiV- in a ring around it. The difference between those two regions is what we 
want to maximize

@author: gardill
"""
import utils.tool_belt as tool_belt
import majorroutines.optimize as optimize
import numpy
import time
import matplotlib.pyplot as plt
import labrad
import copy

#%%
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

green_reset_power = 0.6085
green_pulse_power = 0.65
green_image_power = 0.65

# %%
def plot_time_sweep(test_pulse_dur_list, sig_counts_avg, ref_counts_avg, snr_list, title, text = None):
    # turn the list into an array, so we can convert into us
    test_pulse_dur_list = numpy.array(test_pulse_dur_list)
    
    fig, axes = plt.subplots(1,2, figsize = (17, 8.5)) 
    ax = axes[0]
    ax.plot(test_pulse_dur_list / 10**3, sig_counts_avg, 'go', 
           label = 'Green pulse off readout spot')
    ax.plot(test_pulse_dur_list / 10**3, ref_counts_avg, 'ko', 
           label = 'Green pulse on readout spot')
    ax.set_xlabel('Test pulse length (us)')
    ax.set_ylabel('Counts (single shot measurement)')
    ax.set_title(title)
    ax.legend()
    
    ax = axes[1]    
    ax.plot(test_pulse_dur_list / 10**3, snr_list, 'ro')
    ax.set_xlabel('Test pulse length (us)')
    ax.set_ylabel('SNR')
    ax.set_title(title)
    if text:
        ax.text(0.50, 0.90, text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)
  
    return fig


def compile_raw_data_length_sweep(nv_sig, green_optical_power_pd, green_optical_power_mW, 
                     red_optical_power_pd, red_optical_power_mW, yellow_optical_power_pd, 
                     yellow_optical_power_mW, test_pulse_dur_list, num_reps,green_pulse_time,
                     sig_count_raw, ref_count_raw, sig_counts_avg, ref_counts_avg, snr_list):

    test_pulse_dur_list = numpy.array(test_pulse_dur_list)
    timestamp = tool_belt.get_time_stamp()
    raw_data = {'timestamp': timestamp,
            'nv_sig': nv_sig,
            'nv_sig-units': tool_belt.get_nv_sig_units(),
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
            'test_pulse_dur_list': test_pulse_dur_list.tolist(),
            'test_pulse_dur_list-units': 'ns',
            'green_pulse_time': green_pulse_time,
            'green_pulse_time-units': 'ns',
            'num_reps':num_reps,
            'sig_count_raw': sig_count_raw,
            'sig_count_raw-units': 'counts',
            'ref_count_raw': ref_count_raw,
            'ref_count_raw-units': 'counts',            
            'sig_counts_avg': sig_counts_avg,
            'sig_counts_avg-units': 'counts',
            'ref_counts_avg': ref_counts_avg,
            'ref_counts_avg-units': 'counts',
            'snr_list': snr_list,
            'snr_list-units': 'arb'
            }
    return timestamp, raw_data

def build_voltage_list(start_coords_drift, signal_coords_drift, num_reps):

    # calculate the x values we want to step thru
    start_x_value = start_coords_drift[0]
    start_y_value = start_coords_drift[1]

    
    # we want this list to have the pattern [[readout], [readout], [readout], [readout], 
    #                                                   [target], [readout], [readout],...]
    # The glavo needs a 0th coord, so we'll pass the readout NV as the "starting" point
    x_points = [start_x_value]
    y_points = [start_y_value]
    
    # now append the coordinates in the following pattern:
    for i in range(num_reps):
        x_points.append(start_x_value)
        x_points.append(start_x_value) 
        x_points.append(start_x_value)
        x_points.append(signal_coords_drift[0])
        x_points.append(start_x_value)
        x_points.append(start_x_value)
        
        y_points.append(start_y_value)
        y_points.append(start_y_value) 
        y_points.append(start_y_value)
        y_points.append(signal_coords_drift[1])
        y_points.append(start_y_value)
        y_points.append(start_y_value) 
        
    return x_points, y_points

#%% Main
# Connect to labrad in this file, as opposed to control panel
def main(nv_sig, apd_indices, num_reps, green_pulse_time, dx, readout_color):

    with labrad.connect() as cxn:
        sig_counts, ref_counts = main_with_cxn(cxn, nv_sig, apd_indices, num_reps, green_pulse_time, dx, readout_color)
        
    return sig_counts, ref_counts
def main_with_cxn(cxn, nv_sig, apd_indices, num_reps, green_pulse_time, dx, readout_color):

    tool_belt.reset_cfm_wout_uwaves(cxn)

    # Initial Calculation and setup
    readout_time = nv_sig['pulsed_SCC_readout_dur']
    aom_ao_589_pwr = nv_sig['am_589_power']
    nd_filter = nv_sig['nd_filter']
    color_filter = nv_sig['color_filter']
    
    shared_params = tool_belt.get_shared_parameters_dict(cxn)
    laser_515_delay = shared_params['515_laser_delay']
    aom_589_delay = shared_params['589_aom_delay']
    laser_638_delay = shared_params['638_DM_laser_delay']
    galvo_delay = shared_params['large_angle_galvo_delay']
    
    # Set up our data lists
    opti_coords_list = []
    
    # Optimize
    opti_coords = optimize.main_with_cxn(cxn, nv_sig, apd_indices, 532, disable=False)
    opti_coords_list.append(opti_coords)
    
    # set the nd_filter for yellow
    cxn.filter_slider_ell9k.set_filter(nd_filter)
    cxn.filter_slider_ell9k_color.set_filter(color_filter)

    # Estimate the lenth of the sequance , load the sequence          
    file_name = 'isolate_nv_charge_dynamics_moving_target.py'
    seq_args = [10**5, green_pulse_time, readout_time, 
            laser_515_delay, aom_589_delay, laser_638_delay, galvo_delay,
            aom_ao_589_pwr, 
            green_pulse_power, green_pulse_power, green_image_power,             
            apd_indices[0], 532, 532, readout_color]
    seq_args_string = tool_belt.encode_seq_args(seq_args)
    ret_vals = cxn.pulse_streamer.stream_load(file_name, seq_args_string)
    seq_dur = ret_vals[0]
    period = seq_dur

    # Set up the voltages to step thru  
    # get the drift and add it to the start coordinates      
    drift = numpy.array(tool_belt.get_drift())
    start_coords = numpy.array(nv_sig['coords'])
    start_coords_drift = start_coords + drift
    # define the signal coords as start + dx.
    signal_coords_drift = start_coords_drift + [dx, 0, 0]
    
    x_voltages, y_voltages = build_voltage_list(start_coords_drift, signal_coords_drift, num_reps)
    
    # Collect data
    # start on the readout NV
    tool_belt.set_xyz(cxn, start_coords_drift)
    
    # Load the galvo
    cxn.galvo.load_arb_points_scan(x_voltages, y_voltages, int(period))
    
    # Set up the APD
    cxn.apd_tagger.start_tag_stream(apd_indices)
    
    #Run the sequence double the amount of time, one for the sig and one for the ref
    cxn.pulse_streamer.stream_start(num_reps*2)
    
    # We'll be lookign for three samples each repetition, and double that for 
    # the ref and sig
    total_num_reps = 3*2*num_reps
    
    # Read the counts
    new_samples = cxn.apd_tagger.read_counter_simple(total_num_reps)
    # The last of the triplet of readout windows is the counts we are interested in
    ref_counts = new_samples[2::6]
#    print(ref_counts)
    ref_counts = [int(el) for el in ref_counts]
    sig_counts = new_samples[5::6]
    sig_counts = [int(el) for el in sig_counts]
    
    cxn.apd_tagger.stop_tag_stream()

    return sig_counts, ref_counts

# %%

def optimize_readout_pulse_length(nv_sig,readout_color,  test_pulse_dur_list  = [10*10**3, 
                               50*10**3, 100*10**3,500*10**3, 
                               1*10**6, 2*10**6, 3*10**6, 4*10**6, 5*10**6, 
                               6*10**6, 7*10**6, 8*10**6, 9*10**6, 1*10**7,
                               2*10**7,3*10**7,4*10**7,5*10**7]
                                ):
    apd_indices = [0]
    num_reps = 50
    # Make sure that we are in the SiV band
    nv_sig['color_filter'] = '715 lp'
#    nv_sig['color_filter'] = '635-715 bp'
    green_pulse_time = 10**6
    dx = 0.056
#    dx = 0.25
        
    # create some lists for data
    # signal will be green pulse off readout spot
    # reference will be green pulse on readout spot
    sig_count_raw = []
    ref_count_raw = []
    sig_counts_avg = []
    ref_counts_avg = []
    snr_list = []
    
    # Step through the pulse lengths for the test laser
    for test_pulse_length in test_pulse_dur_list:
        nv_sig['pulsed_SCC_readout_dur'] = int(test_pulse_length)
        print('Readout set to {} ms'.format(test_pulse_length/10**6))
        sig_count, ref_count = main(nv_sig, apd_indices, num_reps, green_pulse_time, dx, readout_color)
        
        sig_count = [int(el) for el in sig_count]
        ref_count = [int(el) for el in ref_count]
        
        sig_count_raw.append(sig_count)
        ref_count_raw.append(ref_count)
        
        snr = tool_belt.calc_snr(ref_count, sig_count)
        sig_counts_avg.append(numpy.average(sig_count))
        ref_counts_avg.append(numpy.average(ref_count))
        snr_list.append(-snr)

    # measure laser powers:
    green_optical_power_pd, green_optical_power_mW, \
            red_optical_power_pd, red_optical_power_mW, \
            yellow_optical_power_pd, yellow_optical_power_mW = \
            tool_belt.measure_g_r_y_power( 
                                  nv_sig['am_589_power'], nv_sig['nd_filter'])
            
    # Plot
    if readout_color == 638:
        title = 'Sweep pulse length for 638 nm readout, SiV band'
        text = 'Red readout pulse power set to ' + '%.0f'%(red_optical_power_mW) + ' mW'
        
    if readout_color == 589:
        title = 'Sweep pulse length for 589 nm readout, SiV band'
        text = 'Yellow readout pulse power set to ' + '%.0f'%(yellow_optical_power_mW*10**3) + ' uW'
    fig = plot_time_sweep(test_pulse_dur_list, sig_counts_avg, ref_counts_avg, 
                          snr_list, title, text = text)
    
    test_pulse_dur_list = numpy.array(test_pulse_dur_list)
    timestamp = tool_belt.get_time_stamp()
    raw_data = {'timestamp': timestamp,
            'nv_sig': nv_sig,
            'nv_sig-units': tool_belt.get_nv_sig_units(),
            'dx': dx,
            'dx-units': 'V',
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
            'test_pulse_dur_list': test_pulse_dur_list.tolist(),
            'test_pulse_dur_list-units': 'ns',
            'green_pulse_time': green_pulse_time,
            'green_pulse_time-units': 'ns',
            'num_reps':num_reps,
            'sig_count_raw': sig_count_raw,
            'sig_count_raw-units': 'counts',
            'ref_count_raw': ref_count_raw,
            'ref_count_raw-units': 'counts',            
            'sig_counts_avg': sig_counts_avg,
            'sig_counts_avg-units': 'counts',
            'ref_counts_avg': ref_counts_avg,
            'ref_counts_avg-units': 'counts',
            'snr_list': snr_list,
            'snr_list-units': 'arb'
            }

    file_path = tool_belt.get_file_path(__file__, timestamp, nv_sig['name'])
    tool_belt.save_raw_data(raw_data, file_path + '-readout_pulse_dur_siv')

    tool_belt.save_figure(fig, file_path + '-readout_pulse_dur_siv')
    
    print(' \nRoutine complete!')
    return

# %% Run the files
    
if __name__ == '__main__':
    sample_name = 'goepert-mayer'
    
    
    nv3_2021_03_01 = { 'coords': [-0.081, 0.096, 5.48], 
            'name': '{}-nv3_2021_03_01'.format(sample_name),
            'expected_count_rate': 40, 'nd_filter': 'nd_0',
#            'color_filter': '635-715 bp', 
            'color_filter': '715 lp',
            'pulsed_readout_dur': 300,
            'pulsed_SCC_readout_dur': 3*10**7, 'am_589_power': 0.6, 
            'pulsed_initial_ion_dur': 25*10**3,
            'pulsed_shelf_dur': 200, 
            'am_589_shelf_power': 0.35,
            'pulsed_ionization_dur': 10**3, 'cobalt_638_power': 10, 
            'pulsed_reionization_dur': 100*10**3, 'cobalt_532_power':10,  
            'ao_515_pwr': 0.65,
            'magnet_angle': 0,
            "resonance_LOW": 2.7,"rabi_LOW": 146.2, "uwave_power_LOW": 9.0,
            "resonance_HIGH": 2.9774,"rabi_HIGH": 95.2,"uwave_power_HIGH": 10.0}  
       
#    for nd in ['nd_1.0']:
#        for p in [0.3]:
#    for nd in ['nd_1.0']:
#        for p in [ 0.3]:
#            nv_sig = copy.deepcopy(nv0_2021_01_26)
#            nv_sig['nd_filter'] = nd
#            nv_sig['am_589_power'] = p
    optimize_readout_pulse_length(nv3_2021_03_01, 589) 