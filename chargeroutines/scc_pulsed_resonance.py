# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 10:52:28 2021

@author: agardill
"""

# %% Imports


import utils.tool_belt as tool_belt
import majorroutines.optimize as optimize
import numpy
import matplotlib.pyplot as plt
import time
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import labrad
from utils.tool_belt import States
import majorroutines.pulsed_resonance as pulsed_resonance
from random import shuffle

# %%

def plot_esr(ref_counts, sig_counts, num_runs, freqs = None, freq_center = None, freq_range = None, num_steps = None):
    
    # if all.freqs() == None:
    #     half_freq_range = freq_range / 2
    #     freq_low = freq_center - half_freq_range
    #     freq_high = freq_center + half_freq_range
    #     freqs = numpy.linspace(freq_low, freq_high, num_steps)
    

    ret_vals = pulsed_resonance.process_counts(ref_counts, sig_counts, num_runs)
    avg_ref_counts, avg_sig_counts, norm_avg_sig, ste_ref_counts, ste_sig_counts, norm_avg_sig_ste = ret_vals

    # Convert to kilocounts per second
    # readout_sec = depletion_time / 1e9
    cts_uwave_off_avg = (avg_ref_counts / (num_runs))# * 1000)) / readout_sec
    cts_uwave_on_avg = (avg_sig_counts / (num_runs))# * 1000)) / readout_sec

    # Create an image with 2 plots on one row, with a specified size
    # Then draw the canvas and flush all the previous plots from the canvas
    fig, axes_pack = plt.subplots(1, 2, figsize=(17, 8.5))

    if len(freqs) == 1:
        marker = 'o'
    else:
        marker = '-'
    # The first plot will display both the uwave_off and uwave_off counts
    ax = axes_pack[0]
    ax.plot(freqs, cts_uwave_off_avg, 'r{}'.format(marker), label = 'Reference')
    ax.plot(freqs, cts_uwave_on_avg, 'g{}'.format(marker), label = 'Signal')
    ax.set_title('Non-normalized Count Rate Versus Frequency')
    ax.set_xlabel('Frequency (GHz)')
    ax.set_ylabel('NV fluorescence (counts)')
    ax.legend()
    # The second plot will show their subtracted values
    ax = axes_pack[1]
    ax.plot(freqs, norm_avg_sig, 'b{}'.format(marker))
    ax.set_title('Normalized Count Rate vs Frequency')
    ax.set_xlabel('Frequency (GHz)')
    ax.set_ylabel('Contrast (arb. units)')

    fig.canvas.draw()
    fig.tight_layout()
    fig.canvas.flush_events()
    
    return fig, norm_avg_sig, norm_avg_sig_ste 
# %% Main


def main(nv_sig, opti_nv_sig, apd_indices, freq_center, freq_range,
         num_steps, num_reps, num_runs, uwave_power, uwave_pulse_dur, do_plot = True,
         state=States.LOW):

    with labrad.connect() as cxn:
        main_with_cxn(cxn, nv_sig,opti_nv_sig, apd_indices, freq_center, freq_range,
                  num_steps, num_reps, num_runs, uwave_power, uwave_pulse_dur,do_plot,
                  state)
    return 


def main_with_cxn(cxn, nv_sig, opti_nv_sig,apd_indices, freq_center, freq_range,
              num_steps, num_reps, num_runs, uwave_power, uwave_pulse_dur, do_plot = True,
              state=States.LOW):

    # %% Initial calculations and setup

    tool_belt.reset_cfm(cxn)
    file_name = 'SCC_optimize_pulses_w_uwaves.py'

    # Calculate the frequencies we need to set
    half_freq_range = freq_range / 2
    freq_low = freq_center - half_freq_range
    freq_high = freq_center + half_freq_range
    freqs = numpy.linspace(freq_low, freq_high, num_steps)
    # print(freqs)
    # return
    if num_steps == 1:
        freqs = numpy.array([freq_center])
    freq_ind_list = list(range(num_steps))
    
    opti_interval = 4 # min
    
    nv_coords = nv_sig['coords']

    # Set up our data structure, an array of NaNs that we'll fill
    # incrementally. NaNs are ignored by matplotlib, which is why they're
    # useful for us here.
    # We define 2D arrays, with the horizontal dimension for the frequency and
    # the veritical dimension for the index of the run.
    ref_counts = numpy.empty([num_runs, num_steps])
    ref_counts[:] = numpy.nan
    sig_counts = numpy.copy(ref_counts)
    
    # imaging_laser_key = 'imaging_laser'
    # imaging_laser_name = nv_sig[imaging_laser_key]
    # imaging_laser_power = tool_belt.set_laser_power(cxn, nv_sig, imaging_laser_key)
    
    # init_color = tool_belt.get_registry_entry_no_cxn('wavelength',
    #                   ['Config', 'Optics', nv_sig['initialize_laser']])
    # depletion_color = tool_belt.get_registry_entry_no_cxn('wavelength',
    #                   ['Config', 'Optics', nv_sig['CPG_laser']])
    
    
    # Set the charge readout (assumed to be yellow here) to the correct filter
    if 'charge_readout_laser_filter' in nv_sig:
        tool_belt.set_filter(cxn, nv_sig, 'charge_readout_laser')
    
    
    
    readout_time = nv_sig['charge_readout_dur']
    readout_power = nv_sig['charge_readout_laser_power']
    ionization_time = nv_sig['nv0_ionization_dur']
    reionization_time = nv_sig['nv-_reionization_dur']
    shelf_time = nv_sig['spin_shelf_dur']
    shelf_power = nv_sig['spin_shelf_laser_power']
    
    
    green_laser_name = nv_sig['nv-_reionization_laser']
    red_laser_name = nv_sig['nv0_ionization_laser']
    yellow_laser_name = nv_sig['charge_readout_laser']
    sig_gen_name = tool_belt.get_signal_generator_name_no_cxn(state)    
            
    
    seq_args = [readout_time, reionization_time, ionization_time, uwave_pulse_dur,
        shelf_time ,  uwave_pulse_dur, green_laser_name, yellow_laser_name, red_laser_name, sig_gen_name,
        apd_indices[0], readout_power, shelf_power]
    
    # print(seq_args)
    # return
    seq_args_string = tool_belt.encode_seq_args(seq_args)

    drift_list = []

    # %% Get the starting time of the function

    start_timestamp = tool_belt.get_time_stamp()

    # %% Collect the data

    # Start 'Press enter to stop...'
    tool_belt.init_safe_stop()
    
    start_time = time.time()
    start_function_time = start_time
    
    
    for run_ind in range(num_runs):
        print('Run index: {}'. format(run_ind))

        # Break out of the while if the user says stop
        if tool_belt.safe_stop():
            break

        # Optimize and save the coords we found
        optimize.main_with_cxn(cxn, opti_nv_sig, apd_indices)
        drift = tool_belt.get_drift()
        drift_list.append(drift)
        
        adjusted_nv_coords = numpy.array(nv_coords) + drift
        
        
        # Set up the microwaves and laser. Then load the pulse streamer 
        # (must happen after optimize and iq_switch since run their
        # own sequences)
        sig_gen_cxn = tool_belt.get_signal_generator_cxn(cxn, state)
        sig_gen_cxn.set_amp(uwave_power)
        ret_vals = cxn.pulse_streamer.stream_load(file_name, seq_args_string)
        
        period = ret_vals[0]
        
        period_s = period/10**9
        period_s_total = (period_s*num_reps*num_steps + 1)
        period_m_total = period_s_total/60
        print('Expected time for this run: {:.1f} m'.format(period_m_total))
        # return

        # Shuffle the freqs we step thru
        shuffle(freq_ind_list)
        
        # Take a sample and increment the frequency
        for step_ind in range(num_steps):
            
            # Break out of the while if the user says stop
            if tool_belt.safe_stop():
                break
            
            freq_ind = freq_ind_list[step_ind]
            print(freqs[freq_ind])
            
            time_current = time.time()
            if time_current - start_time > opti_interval * 60:
                optimize.main_with_cxn(cxn, opti_nv_sig, apd_indices)
                drift = tool_belt.get_drift()
                drift_list.append(drift)
                
                adjusted_nv_coords = numpy.array(nv_coords) + drift
                
                start_time = time_current
                
            tool_belt.set_xyz(cxn, adjusted_nv_coords)
            
            # print(freqs[freq_ind])
            sig_gen_cxn.set_freq(freqs[freq_ind])
            sig_gen_cxn.uwave_on()

            # Start the tagger stream
            cxn.apd_tagger.start_tag_stream(apd_indices)
            cxn.pulse_streamer.stream_load(file_name, seq_args_string)
            

            cxn.pulse_streamer.stream_immediate(file_name, num_reps, seq_args_string)
        
            new_counts = cxn.apd_tagger.read_counter_separate_gates(1)
            sample_counts = new_counts[0]
        
            # signal counts are even - get every second element starting from 0
            sig_counts[run_ind, freq_ind] = sum(sample_counts[0::2])
        
            # ref counts are odd - sample_counts every second element starting from 1
            ref_counts[run_ind, freq_ind] = sum(sample_counts[1::2])
    

        cxn.apd_tagger.stop_tag_stream()

        # %% Save the data we have incrementally for long measurements

        rawData = {'start_timestamp': start_timestamp,
                   'nv_sig': nv_sig,
                   'nv_sig-units': tool_belt.get_nv_sig_units(),
                   'freq_center': freq_center,
                   'freq_center-units': 'GHz',
                   'freq_range': freq_range,
                   'freq_range-units': 'GHz',
                   'uwave_pulse_dur': uwave_pulse_dur,
                   'uwave_pulse_dur-units': 'ns',
                   'state': state.name,
                   'num_steps': num_steps,
                   'run_ind': run_ind,
                   'uwave_power': uwave_power,
                   'uwave_power-units': 'dBm',
                   'freqs': freqs.tolist(),
                   'drift_list': drift_list,
                   'opti_interval': opti_interval,
                   'sig_counts': sig_counts.astype(int).tolist(),
                   'sig_counts-units': 'counts',
                   'ref_counts': ref_counts.astype(int).tolist(),
                   'ref_counts-units': 'counts'}

        # This will continuously be the same file path so we will overwrite
        # the existing file with the latest version
        file_path = tool_belt.get_file_path(__file__, start_timestamp,
                                            nv_sig['name'], 'incremental')
        tool_belt.save_raw_data(rawData, file_path)


    # %% Process and plot the data
    end_function_time = time.time()
    time_elapsed = end_function_time - start_function_time
    # print(time_elapsed)
    
    if do_plot:
        fig, norm_avg_sig,norm_avg_sig_ste  = plot_esr(ref_counts, sig_counts, num_runs, freqs)



    # %% Clean up and save the data

    tool_belt.reset_cfm(cxn)

    timestamp = tool_belt.get_time_stamp()

    rawData = {'timestamp': timestamp,
               'time_elapsed': time_elapsed,
                'nv_sig': nv_sig,
                'nv_sig-units': tool_belt.get_nv_sig_units(),
                'freq_center': freq_center,
                'freq_center-units': 'GHz',
                'freq_range': freq_range,
                'freq_range-units': 'GHz',
                'uwave_pulse_dur': uwave_pulse_dur,
                'uwave_pulse_dur-units': 'ns',
                'state': state.name,
                'num_steps': num_steps,
                'num_reps': num_reps,
                'num_runs': num_runs,
                'uwave_power': uwave_power,
                'uwave_power-units': 'dBm',
                'freqs': freqs.tolist(),
                'drift_list': drift_list,
                'opti_interval': opti_interval,
                'sig_counts': sig_counts.astype(int).tolist(),
                'sig_counts-units': 'counts',
                'ref_counts': ref_counts.astype(int).tolist(),
                'ref_counts-units': 'counts'}

    if do_plot:
        rawData['norm_avg_sig'] = norm_avg_sig.astype(float).tolist()
        rawData['norm_avg_sig-units'] = 'arb'
        rawData['norm_avg_sig_ste'] =norm_avg_sig_ste.astype(float).tolist()
        rawData['norm_avg_sig_ste-units'] = 'arb'
        
    name = nv_sig['name']
    filePath = tool_belt.get_file_path(__file__, timestamp, name)
    if do_plot:
        tool_belt.save_figure(fig, filePath)
    tool_belt.save_raw_data(rawData, filePath)

    return 

    

# %%

if __name__ == '__main__':

    apd_indices = [0]
    sample_name = 'johnson'    
    
    green_laser = "cobolt_515"
    yellow_laser = 'laserglow_589'
    red_laser = 'cobolt_638'
    
    green_power = 7
    red_power = 120
    nd_yellow = "nd_0.5"
    
    opti_nv_sig = {
        "coords": [-0.037, 0.273, 4.85],
        "name": "{}-nv0_2021_09_23".format(sample_name,),
        "disable_opt": False,
        "expected_count_rate": 50,
        "imaging_laser":green_laser,
        "imaging_laser_power": green_power,
        "imaging_readout_dur": 1e7,
        "collection_filter": "630_lp",
        "magnet_angle": None,
    }  # 14.5 max
    
    nv_sig = {
        "coords":[0.1614328 , 0.13376454,4.79 ],
        
        "name": "{}-dnv5_2021_09_23".format(sample_name,),
        "disable_opt": False,
        "expected_count_rate": 75,
            'imaging_laser': green_laser, 'imaging_laser_power': green_power,
            'imaging_readout_dur': 1E7,
            
            'nv-_reionization_laser': green_laser, 'nv-_reionization_laser_power': green_power, 
            'nv-_reionization_dur': 1E5,
        
            'nv0_ionization_laser': red_laser, 'nv0_ionization_laser_power': red_power,
            'nv0_ionization_dur':300,
            
            'spin_shelf_laser': yellow_laser, 'spin_shelf_laser_filter': nd_yellow, 
            'spin_shelf_laser_power': 0.4, 'spin_shelf_dur':0,
            
            'charge_readout_laser': yellow_laser, 'charge_readout_laser_filter': nd_yellow, 
            'charge_readout_laser_power': 0.3, 'charge_readout_dur':0.5e6,
            
            'collection_filter': '630_lp', 'magnet_angle': 114,
            
            "resonance_LOW":2.7897,"rabi_LOW": 139.7,"uwave_power_LOW": 15.5,
            "resonance_HIGH": 2.9496,"rabi_HIGH": 215,"uwave_power_HIGH": 14.5} 
    
    
    freq_range = 0.05
    
    uwave_power = nv_sig['uwave_power_LOW']
    # uwave_pulse_dur =  nv_sig['rabi_LOW'] / 2
    num_steps = 1#51
    num_reps = int(10**4)
    num_runs = 1
    
    do_plot = False
    
    try:
        
        main(nv_sig, opti_nv_sig, apd_indices, nv_sig['resonance_LOW'], freq_range,
                num_steps, num_reps, num_runs, uwave_power, nv_sig['rabi_LOW']/2)
        
        
        
        
        # nv_sig['resonance_LOW'] = 2.80364866
        # nv_sig['rabi_LOW']  = 120
        # main(nv_sig, nv_sig, apd_indices, nv_sig['resonance_LOW'], 0.01945944,
        #         13, num_reps, num_runs, uwave_power, nv_sig['rabi_LOW']/2)
        
        # nv_sig['resonance_LOW'] = 2.94635134
        # nv_sig['rabi_LOW']  = 120
        # main(nv_sig, nv_sig, apd_indices, nv_sig['resonance_LOW'], 0.01945944,
        #         13, num_reps, num_runs, uwave_power, nv_sig['rabi_LOW']/2)
        
        
         
        # freq_high = 2.81337838
        # freq_low = 2.79391894
        # steps = 0.00162162
        # freq_center = 2.80364866
        # freq_range = 0.01945944
        # num_steps = 13
        
        # freq_low = 2.93662162
        # freq_high = 2.95608106
        # freq_center = 2.94635134
        # freq_range = 0.01945944
        # num_steps = 13
        
        if do_plot:
            folder= 'pc_rabi/branch_master/scc_pulsed_resonance/2021_09'
            file_low = '2021_09_28-13_32_45-johnson-dnv7_2021_09_23'
            file_center = '2021_09_28-10_04_05-johnson-dnv7_2021_09_23'
            file_high = '2021_09_28-15_24_19-johnson-dnv7_2021_09_23'
            data_low = tool_belt.get_raw_data(file_low, folder)
            freqs_low = data_low['freqs']
            freq_range_low = data_low['freq_range']
            num_steps_low = data_low['num_steps']
            norm_avg_sig_low = data_low['norm_avg_sig']
            
            data = tool_belt.get_raw_data(file_center, folder)
            freqs_center = data['freqs']
            freq_range_center = data['freq_range']
            num_steps_center = data['num_steps']
            norm_avg_sig_center = data['norm_avg_sig']
            
            data_high = tool_belt.get_raw_data(file_high, folder)
            freqs_high = data_high['freqs']
            freq_range_high = data_high['freq_range']
            num_steps_high = data_high['num_steps']
            norm_avg_sig_high = data_high['norm_avg_sig']
            
            total_freq_range = freq_range_center + freq_range_high
            total_num_steps = num_steps_center + num_steps_high
            total_freqs = freqs_center + freqs_high
            total_norm_sig = norm_avg_sig_center + norm_avg_sig_high
            
            
            
            fig, ax= plt.subplots()
            ax.plot(total_freqs, total_norm_sig, 'b-')
            ax.set_title('Normalized Count Rate vs Frequency')
            ax.set_xlabel('Frequency (GHz)')
            ax.set_ylabel('Contrast (arb. units)')
        
            
            timestamp = data['timestamp']
            nv_sig = data['nv_sig']
            freq_center = data['freq_center']
            uwave_pulse_dur = data['uwave_pulse_dur']
            state = data['state']
            num_reps = data[  'num_reps']
            num_runs = data['num_runs']
            uwave_power = data['uwave_power']
            opti_interval = data[ 'opti_interval']
    
            
            
            rawData = {'timestamp': timestamp,
                    'nv_sig': nv_sig,
                    'nv_sig-units': tool_belt.get_nv_sig_units(),
                    'freq_center': freq_center,
                    'freq_center-units': 'GHz',
                    'freq_range': total_freq_range,
                    'freq_range-units': 'GHz',
                    'uwave_pulse_dur': uwave_pulse_dur,
                    'uwave_pulse_dur-units': 'ns',
                    'state': state,
                    'num_steps': total_num_steps,
                    'num_reps': num_reps,
                    'num_runs': num_runs,
                    'uwave_power': uwave_power,
                    'uwave_power-units': 'dBm',
                    'freqs': total_freqs,
                    'opti_interval': opti_interval,
                    # 'sig_counts': sig_counts.astype(int).tolist(),
                    # 'sig_counts-units': 'counts',
                    # 'ref_counts': ref_counts.astype(int).tolist(),
                    # 'ref_counts-units': 'counts',
                    'norm_avg_sig': total_norm_sig,
                    'norm_avg_sig-units': 'arb',
                    # 'norm_avg_sig_ste':norm_avg_sig_ste.astype(float).tolist(),
                    # 'norm_avg_sig_ste-units': 'arb'
                    }
            
            name = nv_sig['name']
            filePath = tool_belt.get_file_path(__file__, timestamp, name)
            tool_belt.save_figure(fig, filePath + '-compilation')
            tool_belt.save_raw_data(rawData, filePath +'-compilation')
    
    finally:
        # Reset our hardware - this should be done in each routine, but
        # let's double check here
        # tool_belt.reset_cfm()
        # Kill safe stop
        if tool_belt.check_safe_stop_alive():
            print('\n\nRoutine complete. Press enter to exit.')
            tool_belt.poll_safe_stop()
