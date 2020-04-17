# -*- coding: utf-8 -*-
"""
Created on mon Apr 8 10:45:09 2020

This file runs a sequence of R/G/(R)/Y. The second red pulse is applied in
the signal and not applied for the reference. The difference in these two 
cases relates to our ability to distinguish the charge state.

@author: agardill
"""
import utils.tool_belt as tool_belt
import majorroutines.optimize as optimize
import numpy
import matplotlib.pyplot as plt
import labrad

#%%
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

# %%
def plot_time_sweep(test_pulse_dur_list, sig_counts_avg, ref_counts_avg, snr_list, title, text = None):
    # turn the list into an array, so we can convert into us
    test_pulse_dur_list = numpy.array(test_pulse_dur_list)
    
    fig, axes = plt.subplots(1,2, figsize = (17, 8.5)) 
    ax = axes[0]
    ax.plot(test_pulse_dur_list / 10**3, sig_counts_avg, 'ro', 
           label = 'W/ 638 nm pulse')
    ax.plot(test_pulse_dur_list / 10**3, ref_counts_avg, 'ko', 
           label = 'W/out 638 nm pulse')
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
        ax.text(0.55, 0.90, text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)
  
    return fig

def plot_power_sweep(power_list, sig_counts_avg, ref_counts_avg, snr_list, title, text = None):
    power_list = numpy.array(power_list)
    
    fig, axes = plt.subplots(1,2, figsize = (17, 8.5)) 
    ax = axes[0]
    ax.plot(power_list * 10**3, sig_counts_avg, 'ro', 
           label = 'W/ 638 nm pulse')
    ax.plot(power_list * 10**3, ref_counts_avg, 'ko', 
           label = 'W/out 638 nm pulse')
    ax.set_xlabel('Test pulse power (uW)')
    ax.set_ylabel('Counts (single shot measurement)')
    ax.set_title(title)
    ax.legend()
    
    ax = axes[1]    
    ax.plot(power_list * 10**3, snr_list, 'ro')
    ax.set_xlabel('Test pulse power (uW)')
    ax.set_ylabel('SNR')
    ax.set_title(title)
    if text:
        ax.text(0.55, 0.90, text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)
  
    return fig

def compile_raw_data_length_sweep(nv_sig, green_optical_power_pd, green_optical_power_mW, 
                     red_optical_power_pd, red_optical_power_mW, yellow_optical_power_pd, 
                     yellow_optical_power_mW, test_pulse_dur_list, num_reps, 
                     sig_count_raw, ref_count_raw, sig_counts_avg, snr_list):
    if type(test_pulse_dur_list) == list:
        test_pulse_dur_list.tolist()
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
            'test_pulse_dur_list': test_pulse_dur_list,
            'test_pulse_dur_list-units': 'ns',
            'num_reps':num_reps,
            'sig_count_raw': sig_count_raw,
            'sig_count_raw-units': 'counts',
            'ref_count_raw': ref_count_raw,
            'ref_count_raw-units': 'counts',            
            'sig_counts_avg': sig_counts_avg,
            'sig_counts_avg-units': 'counts',
            'sig_counts_avg': sig_counts_avg,
            'sig_counts_avg-units': 'counts',
            'snr_list': snr_list,
            'snr_list-units': 'arb'
            }
    return timestamp, raw_data

def compile_raw_data_power_sweep(nv_sig, green_optical_power_pd, green_optical_power_mW, 
                     red_optical_power_pd, red_optical_power_mW, yellow_optical_power_pd, 
                     yellow_optical_power_mW, power_list, optical_power_list, num_reps, 
                     sig_count_raw, ref_count_raw, sig_counts_avg, snr_list):
    if type(power_list) == list:
        power_list.tolist()
        
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
            'power_list': power_list,
            'power_list-units': '0-1 V',
            'optical_power_list': optical_power_list,
            'optical_power_list-units': 'mW',
            'num_reps':num_reps,
            'sig_count_raw': sig_count_raw,
            'sig_count_raw-units': 'counts',
            'ref_count_raw': ref_count_raw,
            'ref_count_raw-units': 'counts',            
            'sig_counts_avg': sig_counts_avg,
            'sig_counts_avg-units': 'counts',
            'sig_counts_avg': sig_counts_avg,
            'sig_counts_avg-units': 'counts',
            'snr_list': snr_list,
            'snr_list-units': 'arb'
            }
    return timestamp, raw_data
#%% Main
# Connect to labrad in this file, as opposed to control panel
def main(nv_sig, apd_indices, num_reps):

    with labrad.connect() as cxn:
        sig_counts, ref_counts = main_with_cxn(cxn, nv_sig, apd_indices, num_reps)
        
    return sig_counts, ref_counts
def main_with_cxn(cxn, nv_sig, apd_indices, num_reps):

    tool_belt.reset_cfm(cxn)

# Initial Calculation and setup
    readout_time = nv_sig['pulsed_SCC_readout_dur']
    ion_time = nv_sig['pulsed_ionization_dur']
    reion_time = nv_sig['pulsed_reionization_dur']
    init_ion_time = nv_sig['pulsed_initial_ion_dur']
    aom_ao_589_pwr = nv_sig['am_589_power']
    nd_filter = nv_sig['nd_filter']
        
    # set the nd_filter for yellow
    cxn.filter_slider_ell9k.set_filter(nd_filter)
    
    shared_params = tool_belt.get_shared_parameters_dict(cxn)

    #delay of aoms and laser
    laser_515_delay = shared_params['515_laser_delay']
    aom_589_delay = shared_params['589_aom_delay']
    laser_638_delay = shared_params['638_DM_laser_delay']
    
    wait_time = shared_params['post_polarization_wait_dur']

    # Set up our data lists
    opti_coords_list = []

    # Estimate the lenth of the sequance            
    file_name = 'SCC_optimize_pulses_wout_uwaves.py'
    seq_args = [readout_time,init_ion_time, reion_time, ion_time,
            wait_time, laser_515_delay, aom_589_delay, laser_638_delay, 
            apd_indices[0], aom_ao_589_pwr]
    print(seq_args)
    seq_args_string = tool_belt.encode_seq_args(seq_args)
    ret_vals = cxn.pulse_streamer.stream_load(file_name, seq_args_string)

    seq_time = ret_vals[0]

    seq_time_s = seq_time / (10**9)  # s
    expected_run_time = num_reps * seq_time_s  #s
    expected_run_time_m = expected_run_time / 60 # m

    # Ask to continue and timeout if no response in 2 seconds?

    print(' \nExpected run time: {:.1f} minutes. '.format(expected_run_time_m))

    # Collect data

    # Optimize
    opti_coords = optimize.main_with_cxn(cxn, nv_sig, apd_indices, 532, disable=True)
    opti_coords_list.append(opti_coords)

    # Load the APD
    cxn.apd_tagger.start_tag_stream(apd_indices)
    # Run the sequence
    cxn.pulse_streamer.stream_immediate(file_name, num_reps, seq_args_string)

    new_counts = cxn.apd_tagger.read_counter_separate_gates(1)
    sample_counts = new_counts[0]

    # signal counts are even - get every second element starting from 0
    sig_counts = sample_counts[0::2]

    # ref counts are odd - sample_counts every second element starting from 1
    ref_counts = sample_counts[1::2]
    
    cxn.apd_tagger.stop_tag_stream()
    
    return sig_counts, ref_counts

# %%

def optimize_reion_pulse_length(nv_sig, test_pulse_dur_list = None):
    apd_indices = [0]
    num_reps = 1000
    if not test_pulse_dur_list:
        test_pulse_dur_list = [0,5*10**3, 10*10**3, 20*10**3, 30*10**3, 
                        40*10**3, 50*10**3, 100*10**3,200*10**3,300*10**3,
                        400*10**3,500*10**3, 600*10**3, 700*10**3, 800*10**3, 
                        900*10**3, 1*10**6, 2*10**6, 3*10**6 ]
    
    # measure laser powers:
    green_optical_power_pd, green_optical_power_mW, \
            red_optical_power_pd, red_optical_power_mW, \
            yellow_optical_power_pd, yellow_optical_power_mW = \
            tool_belt.measure_g_r_y_power( 
                                  nv_sig['am_589_power'], nv_sig['nd_filter'])
    
    # make sure the list is a list, for saving purposes
    if type(test_pulse_dur_list) != list:
        test_pulse_dur_list = test_pulse_dur_list.tolist()
    
    # create some lists for data
    sig_count_raw = []
    ref_count_raw = []
    sig_counts_avg = []
    ref_counts_avg = []
    snr_list = []
    
    # Step through the pulse lengths for the test laser
    for test_pulse_length in test_pulse_dur_list:
        nv_sig['pulsed_reionization_dur'] = test_pulse_length
        sig_count, ref_count = main(nv_sig, apd_indices, num_reps)
        
        sig_count = [int(el) for el in sig_count]
        ref_count = [int(el) for el in ref_count]
        
        sig_count_raw.append(sig_count)
        ref_count_raw.append(ref_count)
        
        snr = tool_belt.calc_snr(sig_count, ref_count)
        sig_counts_avg.append(numpy.average(sig_count))
        ref_counts_avg.append(numpy.average(ref_count))
        snr_list.append(-snr)
        
    # Plot
    title = 'Sweep pusle length for 515 nm'
    fig = plot_time_sweep(test_pulse_dur_list, sig_counts_avg, ref_counts_avg, 
                          snr_list, title)
    # Save
    timestamp, raw_data = compile_raw_data_length_sweep(nv_sig, 
                     green_optical_power_pd, green_optical_power_mW, 
                     red_optical_power_pd, red_optical_power_mW, 
                     yellow_optical_power_pd, yellow_optical_power_mW, 
                     test_pulse_dur_list, num_reps, 
                     sig_count_raw, ref_count_raw, sig_counts_avg, snr_list)

    file_path = tool_belt.get_file_path(__file__, timestamp, nv_sig['name'])
    tool_belt.save_raw_data(raw_data, file_path + '-reion_pulse_dur')

    tool_belt.save_figure(fig, file_path + '-reion_pulse_dur')
    
    print(' \nRoutine complete!')
    return

# %%
    
def optimize_init_ion_and_reion_pulse_length(nv_sig, test_pulse_dur_list = None):
    '''
    This function will test init red pulse and green lengths w/ and w/out 
    second red ion pulse.
    
    Both pulse lengths are set to the same value
    '''
    apd_indices = [0]
    num_reps = 1000
    if not test_pulse_dur_list:
        test_pulse_dur_list = [0,5*10**3, 10*10**3, 20*10**3, 30*10**3, 
                        40*10**3, 50*10**3, 100*10**3,200*10**3,300*10**3,
                        400*10**3,500*10**3, 600*10**3, 700*10**3, 800*10**3, 
                        900*10**3, 1*10**6, 2*10**6, 3*10**6 ]
    
    # measure laser powers:
    green_optical_power_pd, green_optical_power_mW, \
            red_optical_power_pd, red_optical_power_mW, \
            yellow_optical_power_pd, yellow_optical_power_mW = \
            tool_belt.measure_g_r_y_power( 
                                  nv_sig['am_589_power'], nv_sig['nd_filter'])
    
    # make sure the list is a list, for saving purposes
    if type(test_pulse_dur_list) != list:
        test_pulse_dur_list = test_pulse_dur_list.tolist()
    
    # create some lists for data
    sig_count_raw = []
    ref_count_raw = []
    sig_counts_avg = []
    ref_counts_avg = []
    snr_list = []
    
    # Step through the pulse lengths for the test laser
    for test_pulse_length in test_pulse_dur_list:
        nv_sig['pulsed_reionization_dur'] = test_pulse_length
        nv_sig['pulsed_initial_ion_dur'] = test_pulse_length
        sig_count, ref_count = main(nv_sig, apd_indices, num_reps)
        
        sig_count = [int(el) for el in sig_count]
        ref_count = [int(el) for el in ref_count]
        
        sig_count_raw.append(sig_count)
        ref_count_raw.append(ref_count)
        
        snr = tool_belt.calc_snr(sig_count, ref_count)
        sig_counts_avg.append(numpy.average(sig_count))
        ref_counts_avg.append(numpy.average(ref_count))
        snr_list.append(-snr)
    # Plot
    title = 'Sweep pusle length for initial 638 nm and 515 nm'
    fig = plot_time_sweep(test_pulse_dur_list, sig_counts_avg, ref_counts_avg, 
                          snr_list, title)

    # Save
    timestamp, raw_data = compile_raw_data_length_sweep(nv_sig, 
                     green_optical_power_pd, green_optical_power_mW, 
                     red_optical_power_pd, red_optical_power_mW, 
                     yellow_optical_power_pd, yellow_optical_power_mW, 
                     test_pulse_dur_list, num_reps, 
                     sig_count_raw, ref_count_raw, sig_counts_avg, snr_list)

    file_path = tool_belt.get_file_path(__file__, timestamp, nv_sig['name'])
    tool_belt.save_raw_data(raw_data, file_path + '-init_ion_and_reion_pulse_dur')

    tool_belt.save_figure(fig, file_path + '-init_ion_and_reion_pulse_dur')
    
    print(' \nRoutine complete!')
    return

# %%

def optimize_ion_pulse_length(nv_sig, test_pulse_dur_list = None):
    apd_indices = [0]
    num_reps = 1000
    if not test_pulse_dur_list:
        test_pulse_dur_list = numpy.linspace(0, 1000, 11)
    
    # measure laser powers:
    green_optical_power_pd, green_optical_power_mW, \
            red_optical_power_pd, red_optical_power_mW, \
            yellow_optical_power_pd, yellow_optical_power_mW = \
            tool_belt.measure_g_r_y_power( 
                                  nv_sig['am_589_power'], nv_sig['nd_filter'])
    
    # make sure the list is a list, for saving purposes
    if type(test_pulse_dur_list) != list:
        test_pulse_dur_list = test_pulse_dur_list.tolist()
    
    # create some lists for data
    sig_count_raw = []
    ref_count_raw = []
    sig_counts_avg = []
    ref_counts_avg = []
    snr_list = []
    
    # Step through the pulse lengths for the test laser
    for test_pulse_length in test_pulse_dur_list:
        nv_sig['pulsed_ionization_dur'] = test_pulse_length
        sig_count, ref_count = main(nv_sig, apd_indices, num_reps)
        
        sig_count = [int(el) for el in sig_count]
        ref_count = [int(el) for el in ref_count]
        
        sig_count_raw.append(sig_count)
        ref_count_raw.append(ref_count)
        
        snr = tool_belt.calc_snr(sig_count, ref_count)
        sig_counts_avg.append(numpy.average(sig_count))
        ref_counts_avg.append(numpy.average(ref_count))
        snr_list.append(-snr)
    
    # Plot
    title = 'Sweep pusle length for 638 nm'
    fig = plot_time_sweep(test_pulse_dur_list, sig_counts_avg, ref_counts_avg, 
                          snr_list, title)

    # Save
    timestamp, raw_data = compile_raw_data_length_sweep(nv_sig, 
                     green_optical_power_pd, green_optical_power_mW, 
                     red_optical_power_pd, red_optical_power_mW, 
                     yellow_optical_power_pd, yellow_optical_power_mW, 
                     test_pulse_dur_list, num_reps, 
                     sig_count_raw, ref_count_raw, sig_counts_avg, snr_list)

    file_path = tool_belt.get_file_path(__file__, timestamp, nv_sig['name'])
    tool_belt.save_raw_data(raw_data, file_path + '-ion_pulse_dur')

    tool_belt.save_figure(fig, file_path + '-ion_pulse_dur')
    
    print(' \nRoutine complete!')
    return

# %%

def optimize_readout_pulse_length(nv_sig, test_pulse_dur_list = None):
    apd_indices = [0]
    num_reps = 1000
    if not test_pulse_dur_list:
        test_pulse_dur_list = [10*10**3, 50*10**3, 100*10**3,500*10**3, 
                               1*10**6, 2*10**6, 3*10**6, 4*10**6, 5*10**6, 
                               6*10**6, 7*10**6, 8*10**6, 9*10**6, 1*10**7,
                               2*10**7,3*10**7,4*10**7,5*10**7]
    
    # measure laser powers:
    green_optical_power_pd, green_optical_power_mW, \
            red_optical_power_pd, red_optical_power_mW, \
            yellow_optical_power_pd, yellow_optical_power_mW = \
            tool_belt.measure_g_r_y_power( 
                                  nv_sig['am_589_power'], nv_sig['nd_filter'])
    
    # make sure the list is a list, for saving purposes
    if type(test_pulse_dur_list) != list:
        test_pulse_dur_list = test_pulse_dur_list.tolist()
    
    # create some lists for data
    sig_count_raw = []
    ref_count_raw = []
    sig_counts_avg = []
    ref_counts_avg = []
    snr_list = []
    
    # Step through the pulse lengths for the test laser
    for test_pulse_length in test_pulse_dur_list:
        nv_sig['pulsed_SCC_readout_dur'] = test_pulse_length
        sig_count, ref_count = main(nv_sig, apd_indices, num_reps)
        
        sig_count = [int(el) for el in sig_count]
        ref_count = [int(el) for el in ref_count]
        
        sig_count_raw.append(sig_count)
        ref_count_raw.append(ref_count)
        
        snr = tool_belt.calc_snr(sig_count, ref_count)
        sig_counts_avg.append(numpy.average(sig_count))
        ref_counts_avg.append(numpy.average(ref_count))
        snr_list.append(-snr)
    
    # Plot
    title = 'Sweep pusle length for 589 nm'
    text = 'Yellow pusle power set to ' + '%.0f'%(yellow_optical_power_mW*10**3) + ' uW'
    fig = plot_time_sweep(test_pulse_dur_list, sig_counts_avg, ref_counts_avg, 
                          snr_list, title, text = text)
    
    # Save
    timestamp, raw_data = compile_raw_data_length_sweep(nv_sig, 
                     green_optical_power_pd, green_optical_power_mW, 
                     red_optical_power_pd, red_optical_power_mW, 
                     yellow_optical_power_pd, yellow_optical_power_mW, 
                     test_pulse_dur_list, num_reps, 
                     sig_count_raw, ref_count_raw, sig_counts_avg, snr_list)

    file_path = tool_belt.get_file_path(__file__, timestamp, nv_sig['name'])
    tool_belt.save_raw_data(raw_data, file_path + '-readout_pulse_dur')

    tool_belt.save_figure(fig, file_path + '-readout_pulse_dur')
    
    print(' \nRoutine complete!')
    return

# %%

def optimize_readout_pulse_power(nv_sig, power_list = None):
    apd_indices = [0]
    num_reps = 1000

    if not power_list:
        power_list = numpy.linspace(0.1,0.8,15).tolist()
       
    # create some lists for data
    optical_power_list = []
    sig_count_raw = []
    ref_count_raw = []
    sig_counts_avg = []
    ref_counts_avg = []
    snr_list = []
    
    # Step through the pulse lengths for the test laser
    for power in power_list:
        nv_sig['am_589_power'] = power
        
        # measure laser powers:
        green_optical_power_pd, green_optical_power_mW, \
                red_optical_power_pd, red_optical_power_mW, \
                yellow_optical_power_pd, yellow_optical_power_mW = \
                tool_belt.measure_g_r_y_power( 
                                  nv_sig['am_589_power'], nv_sig['nd_filter'])
        optical_power_list.append(yellow_optical_power_mW)
        sig_count, ref_count = main(nv_sig, apd_indices, num_reps)
        
        sig_count = [int(el) for el in sig_count]
        ref_count = [int(el) for el in ref_count]
        
        sig_count_raw.append(sig_count)
        ref_count_raw.append(ref_count)
        
        snr = tool_belt.calc_snr(sig_count, ref_count)
        sig_counts_avg.append(numpy.average(sig_count))
        ref_counts_avg.append(numpy.average(ref_count))
        snr_list.append(-snr)
    
    # Plot
    title = 'Sweep pusle power for 589 nm'
    text = 'Yellow pusle length set to ' + str(nv_sig['pulsed_SCC_readout_dur']/10**6) + ' ms'
    fig = plot_power_sweep(optical_power_list, sig_counts_avg, ref_counts_avg, 
                          snr_list, title, text = text)
    # Save
    timestamp, raw_data = compile_raw_data_power_sweep(nv_sig, 
                     green_optical_power_pd, green_optical_power_mW, 
                     red_optical_power_pd, red_optical_power_mW, 
                     yellow_optical_power_pd, yellow_optical_power_mW, 
                     power_list, optical_power_list, num_reps, 
                     sig_count_raw, ref_count_raw, sig_counts_avg, snr_list)

    file_path = tool_belt.get_file_path(__file__, timestamp, nv_sig['name'])
    tool_belt.save_raw_data(raw_data, file_path + '-readout_pulse_pwr')

    tool_belt.save_figure(fig, file_path + '-readout_pulse_pwr')
    
    print(' \nRoutine complete!')
    return
# %% Run the files
    
if __name__ == '__main__':
    sample_name = 'hopper'
    ensemble = { 'coords': [0.0, 0.0, 5.00],
            'name': '{}-ensemble'.format(sample_name),
            'expected_count_rate': 1000, 'nd_filter': 'nd_0.5',
            'pulsed_readout_dur': 300,
            'pulsed_SCC_readout_dur': 1*10**7, 'am_589_power': 0.3, 
            'pulsed_initial_ion_dur': 200*10**3,
            'pulsed_shelf_dur': 0, 'am_589_shelf_power': 0,#50/0.3,
            'pulsed_ionization_dur': 800, 'cobalt_638_power': 160, 
            'pulsed_reionization_dur': 200*10**3, 'cobalt_532_power': 8, 
            'magnet_angle': 0,
            'resonance_LOW': 2.8059, 'rabi_LOW': 187.8, 'uwave_power_LOW': 9.0, 
            'resonance_HIGH': 2.9366, 'rabi_HIGH': 247.4, 'uwave_power_HIGH': 10.0}   
    nv_sig = ensemble
    
    # Run the program
#    optimize_reion_pulse_length(nv_sig)
#    optimize_ion_pulse_length(nv_sig)
#    optimize_init_ion_and_reion_pulse_length(nv_sig)
#    optimize_readout_pulse_length(nv_sig)
#    optimize_readout_pulse_power(nv_sig)
    
# %% Temp test: manual test
    import time
    apd_indices = [0]
    num_reps = 10**3
    # shine the green laser for a few seconds before the sequence
    with labrad.connect() as cxn:
        cxn.pulse_streamer.constant([3], 0.0, 0.0)
        time.sleep(5)  
        
    # measure red power
    optical_power = tool_belt.opt_power_via_photodiode(638)
    red_power = tool_belt.calc_optical_power_mW(638, optical_power)
    
    nv_sig['pulsed_initial_ion_dur'] = 0
    nv_sig['pulsed_ionization_dur'] = 450
    sig_counts, ref_counts = main(nv_sig, apd_indices,num_reps)
    
    avg_sig_counts = numpy.average(sig_counts)
    avg_ref_counts = numpy.average(ref_counts)
    snr = tool_belt.calc_snr(sig_counts, ref_counts)
    
    print('Red power = ' + '%.5f'%(optical_power) + ' mW')
    print('Signal count: ' + str(avg_sig_counts))
    print('Reference count: ' + str(avg_ref_counts))
    print('snr = ' + str(-snr))
    
    temp = [#21.0, 20.8, 20.6, 20.4, 
            20.2, 20.0, 19.8,
            19.6, 20.4, 19.4, 19.2, 19.0, 18.5, 18.0,
            20.6, 20.8, 21.0, 21.2, 21.4, 21.6, 21.8, 
            22.0, 22.2, 22.4, 22.6, 22.8, 23.0, 23.2,
            23.4, 23.6, 23.8, 24.0, 24.2, 24.4, 24.6, 
            24.8, 25.0]
    
    power = [#4.7, 4.7, 4.7, 4.7, 
             4.70764, 4.70136, 4.69524, 
             4.69299, 4.71972, 4.69653, 4.70426, 4.70442, 4.70329, 4.70587, 
             4.73195, 4.73550, 4.73501, 4.73646, 4.73405, 4.74129, 4.74483, 
             4.76963, 4.78042, 4.79942, 4.81198, 4.83049, 4.84933, 4.85029,
             4.78267, 4.75788, 4.81970, 4.72133, 4.59107, 4.52828, 4.40913, 
             4.29176, 4.01966]
    
    sig = [365.342, 367.935, 367.295, 370.865, 372.961, 376.993, 374.302, 
           375.517, 375.776, 377.025, 375.988, 377.689, 384.095, 383.085,
           369.935, 371.498, 367.518, 366.297, 367.721, 364.146, 362.111, 
           365.094, 357.569, 358.812, 354.496, 355.893, 357.438, 356.491, 
           356.761, 354.928, 353.55, 354.331, 352.036, 350.535, 350.264, 
           350.248, 345.612]   
    ref = [382.344, 382.212, 380.804, 382.405, 383.984, 385.332, 382.396, 
           384.018, 387.805, 383.579, 382.97, 383.148, 387.275, 384.719,
           381.728, 385.951, 382.018, 383.615, 385.893, 383.549, 383.486, 
           386.137, 379.213, 379.211, 377.962, 378.432, 381.828, 381.679, 
           381.18, 379.431, 379.473, 381.067, 381.765, 379.307, 379.629, 
           380.299, 377.16]
    snr = [0.869, 0.730, 0.692,0.590, 0.563, 0.425, 0.414, 
           0.434, 0.611, 0.335, 0.357, 0.279, 0.162, 0.083,
           0.604, 0.736, 0.742, 0.884, 0.925, 0.991, 1.092, 
           1.071, 1.111, 1.045, 1.207, 1.159, 1.248, 1.289,
           1.251, 1.2579, 1.331, 1.370, 1.522, 1.477, 1.507, 
           1.541, 1.624]
 
#    fig, axes = plt.subplots(1,2, figsize = (17, 8.5)) 
#    ax = axes[0]
#    ax.plot(temp, sig, 'ro')
#    ax.plot(temp, ref, 'go')
#    ax.set_xlabel('TEC Temperature (C)')
#    ax.set_ylabel('Counts (single shot)')
#    ax.set_title('Changing 638 nm TEC temperature') 
#    
#    ax = axes[1]
#    ax.plot(temp, snr, 'ro')
#    ax.set_xlabel('TEC Temperature (C)')
#    ax.set_ylabel('snr')
#    ax.set_title('Changing 638 nm TEC temperature') 

#    fig, ax = plt.subplots(1,1, figsize = (10, 8.5))     
#    ax.plot(temp, power, 'ro')
#    ax.set_xlabel('TEC Temperatur (C)')
#    ax.set_ylabel('Power (mW)')
#    ax.set_title('Changing 638 nm TEC temperature') 
    
    # %% If you need to combine multiple data sets, it's easy enoguh to do it below
#    pulse_list = [0,5*10**3, 10*10**3, 20*10**3, 30*10**3, 40*10**3, 50*10**3, 
#                           100*10**3,200*10**3,300*10**3,400*10**3,500*10**3,
#                           600*10**3, 700*10**3, 800*10**3, 900*10**3, 
#                           1*10**6, 2*10**6, 3*10**6 ] + \
#        [4*10**6, 5*10**6, 6*10**6, 7*10**6, 8*10**6, 9*10**6, 10*10**6]
#            
#    snr = [
#    -0.3073715396069574,
#    -1.5337386682446503,
#    -1.7521599720926855,
#    -2.0182458269201744,
#    -2.14992057457769,
#    -2.183234296139274,
#    -2.258685530408988,
#    -2.3858208219808876,
#    -2.3430010695587815,
#    -2.390784360141965,
#    -2.460223811983043,
#    -2.386856667340553,
#    -2.383557909765004,
#    -2.4293828395619292,
#    -2.4444565551633457,
#    -2.4657687514687594,
#    -2.408608602356511,
#    -2.531233834247444,
#    -2.564933291607767
#  ] + [
#    -2.5909010202572973,
#    -2.5454938071779867,
#    -2.6295350165068845,
#    -2.5678510102514998,
#    -2.6684628831371584,
#    -2.6178349514005452,
#    -2.7136979075414684
#  ]
    
#    pulse_list = numpy.array(pulse_list)
#    fig, ax = plt.subplots(1,1, figsize = (10, 8.5)) 
#    ax.plot(pulse_list / 10**3, snr, 'ro')
#    ax.set_xlabel('Test pulse length (us)')
#    ax.set_ylabel('SNR')
#    ax.set_title('Sweep pusle length for 515 nm and initial 638 nm')
    