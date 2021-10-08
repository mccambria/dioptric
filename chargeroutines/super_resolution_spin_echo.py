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

def build_voltages(adjusted_nv_coords, adjusted_depletion_coords, num_reps):
    start_x_value = adjusted_nv_coords[0]
    start_y_value = adjusted_nv_coords[1]
    
    dep_x_value = adjusted_depletion_coords[0]
    dep_y_value = adjusted_depletion_coords[1]
    
    # we want the sequence to look like the following: 
        # [[nv_coords], [dep_coords], [nv_coords], [nv_coords], 
        #               [dep_coords], [nv_coords], [nv_coords], 
        #                       .... ]
   

    seq_x = [dep_x_value, start_x_value, start_x_value]
    seq_y = [dep_y_value, start_y_value, start_y_value]
    
    x_voltages = seq_x*num_reps*4
    y_voltages = seq_y*num_reps*4
    
    # and then add on the initial coordinate
    x_voltages = [start_x_value] + x_voltages
    y_voltages = [start_y_value] + y_voltages
    
    return x_voltages, y_voltages

# %%
def combine_revivals(file_list, folder):
    
    norm_counts_tot = []
    taus_tot = []
    for file in file_list:
        data = tool_belt.get_raw_data(file, folder)
        taus = data['taus']
        norm_avg_sig = data['norm_avg_sig']
        norm_counts_tot = norm_counts_tot + norm_avg_sig
        taus_tot = taus_tot + taus
    nv_sig = data['nv_sig']
    uwave_pi_on_2_pulse = data['uwave_pi_on_2_pulse']
    uwave_pi_pulse = data['uwave_pi_pulse']
    state = data['state']
    num_reps = data['num_reps']
    num_runs = ['num_runs']
    
    timestamp = tool_belt.get_time_stamp()
    rawData = {'timestamp': timestamp,
        'nv_sig': nv_sig,
        'nv_sig-units': tool_belt.get_nv_sig_units(),
        "uwave_pi_pulse": uwave_pi_pulse,
        "uwave_pi_pulse-units": "ns",
        "uwave_pi_on_2_pulse": uwave_pi_on_2_pulse,
        "uwave_pi_on_2_pulse-units": "ns",
        'state': state,
        'num_reps': num_reps,
        'num_runs': num_runs,
        'taus': taus_tot,
        "norm_avg_sig": norm_counts_tot,
        "norm_avg_sig-units": "arb",
        }
    
    fig, ax = plt.subplots(figsize=(8.5, 8.5))
    ax.plot(taus_tot, norm_counts_tot, 'bo')
    ax.set_xlabel('Taus (us)')
    ax.set_ylabel('Contrast (arb. units)')
    ax.legend(loc='lower right')
    
    
    name = nv_sig['name']
    filePath = tool_belt.get_file_path(__file__, timestamp, name)
    tool_belt.save_figure(fig, filePath)
    tool_belt.save_raw_data(rawData, filePath)
            
    return
# %% Main


def main(nv_sig, opti_nv_sig, apd_indices, precession_time_range,
         num_steps, num_reps, num_runs,  
         state=States.LOW):

    with labrad.connect() as cxn:
        sig_gate_counts, ref_gate_counts = main_with_cxn(cxn, nv_sig, opti_nv_sig, 
                             apd_indices, precession_time_range,
                             num_steps, num_reps, num_runs,  
                             state)
    return sig_gate_counts, ref_gate_counts


def main_with_cxn(cxn, nv_sig, opti_nv_sig, apd_indices, precession_time_range,
         num_steps, num_reps, num_runs,  
         state=States.LOW):

    #  Initial calculations and setup

    tool_belt.reset_cfm(cxn)
    seq_file_name = 'super_resolution_spin_echo.py'

    # Create the array of relaxation times

    # Array of times to sweep through
    # Must be ints
    min_precession_time = int(precession_time_range[0])
    max_precession_time = int(precession_time_range[1])

    taus = numpy.linspace(
        min_precession_time,
        max_precession_time,
        num=num_steps,
        dtype=numpy.int32,
    )
    print(taus)
    # return
    # Fix the length of the sequence to account for odd amount of elements

    # Our sequence pairs the longest time with the shortest time, and steps
    # toward the middle. This means we only step through half of the length
    # of the time array.

    # That is a problem if the number of elements is odd. To fix this, we add
    # one to the length of the array. When this number is halfed and turned
    # into an integer, it will step through the middle element.

    if len(taus) % 2 == 0:
        half_length_taus = int(len(taus) / 2)
    elif len(taus) % 2 == 1:
        half_length_taus = int((len(taus) + 1) / 2)
        
    # Then we must use this half length to calculate the list of integers to be
    # shuffled for each run

    tau_ind_list = list(range(0, half_length_taus))
    
    # Set up our data structure, an array of NaNs that we'll fill
    # incrementally. NaNs are ignored by matplotlib, which is why they're
    # useful for us here.
    # We define 2D arrays, with the horizontal dimension for the frequency and
    # the veritical dimension for the index of the run.
    ref_counts = numpy.empty([num_runs, num_steps])
    ref_counts[:] = numpy.nan
    sig_counts = numpy.copy(ref_counts)
    
    drift_list = []
    tau_index_master_list = [[] for i in range(num_runs)]
    opti_interval = 4 # min
    
    nv_coords = nv_sig['coords']
    depletion_coords = nv_sig['depletion_coords']
    rabi_period = nv_sig["rabi_{}".format(state.name)]
    uwave_freq = nv_sig["resonance_{}".format(state.name)]
    uwave_power = nv_sig["uwave_power_{}".format(state.name)]
    
    # Get pulse frequencies
    uwave_pi_pulse = tool_belt.get_pi_pulse_dur(rabi_period)
    uwave_pi_on_2_pulse = tool_belt.get_pi_on_2_pulse_dur(rabi_period)

    readout_time = nv_sig['charge_readout_dur']
    init_time = nv_sig['initialize_dur']
    depletion_time = nv_sig['CPG_laser_dur']
    readout_power = nv_sig['charge_readout_laser_power']
    ionization_time = nv_sig['nv0_ionization_dur']
    shelf_time = nv_sig['spin_shelf_dur']
    shelf_power = nv_sig['spin_shelf_laser_power']
    
    green_laser_name = nv_sig['imaging_laser']
    red_laser_name = nv_sig['nv0_ionization_laser']
    yellow_laser_name = nv_sig['charge_readout_laser']
    sig_gen_name = tool_belt.get_signal_generator_name_no_cxn(state)    
            
    init_color = tool_belt.get_registry_entry_no_cxn('wavelength',
                      ['Config', 'Optics', nv_sig['initialize_laser']])
    depletion_color = tool_belt.get_registry_entry_no_cxn('wavelength',
                      ['Config', 'Optics', nv_sig['CPG_laser']])
    
    
    # Set the charge readout (assumed to be yellow here) to the correct filter
    if 'charge_readout_laser_filter' in nv_sig:
        tool_belt.set_filter(cxn, nv_sig, 'charge_readout_laser')
    
    
    seq_args = [readout_time, init_time, depletion_time, ionization_time, shelf_time ,
          min_precession_time, max_precession_time, uwave_pi_pulse, uwave_pi_on_2_pulse, 
          init_color, depletion_color,
          green_laser_name, yellow_laser_name, red_laser_name, sig_gen_name,
          apd_indices[0], readout_power, shelf_power]
    
    seq_args_string = tool_belt.encode_seq_args(seq_args)
    ret_vals = cxn.pulse_streamer.stream_load(seq_file_name, seq_args_string)
        
    period = ret_vals[0]
    
    period_s = period/10**9
    period_s_total = (period_s*num_reps*num_steps*num_runs/2 + 1)
    period_m_total = period_s_total/60
    print('Expected time: {:.1f} m'.format(period_m_total))
    # return

    # %% Get the starting time of the function

    start_timestamp = tool_belt.get_time_stamp()

    # %% Collect the data

    # Start 'Press enter to stop...'
    tool_belt.init_safe_stop()
    
    start_time = time.time()
    start_function_time = start_time
    
    inc_fig, axes_pack = plt.subplots(1, 2, figsize=(17, 8.5))
    plot_taus = (taus + uwave_pi_pulse) / 1000
    ax = axes_pack[0]
    ax.set_xlabel(r"$\tau + \pi$ ($\mathrm{\mu s}$)")
    ax.set_ylabel("Counts")
    
    ax = axes_pack[1]
    ax.set_xlabel(r"$\tau + \pi$ ($\mathrm{\mu s}$)")
    ax.set_ylabel("Contrast (arb. units)")
    
    print(depletion_coords)
    
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
        adjusted_depletion_coords = numpy.array(depletion_coords) + drift
        
        
        # Set up the microwaves and laser. Then load the pulse streamer 
        # (must happen after optimize and iq_switch since run their
        # own sequences)
        # Set up the microwaves
        sig_gen_cxn = tool_belt.get_signal_generator_cxn(cxn, state)
        sig_gen_cxn.set_freq(uwave_freq)
        sig_gen_cxn.set_amp(uwave_power)
        sig_gen_cxn.uwave_on()
        
        # Shuffle the taus we step thru
        shuffle(tau_ind_list)
        
       
        # Take a sample and increment the frequency
        for tau_ind in tau_ind_list:
             
            # Check if we should optimize
            time_current = time.time()
            if time_current - start_time > opti_interval * 60:
                optimize.main_with_cxn(cxn, opti_nv_sig, apd_indices)
                drift = tool_belt.get_drift()
                drift_list.append(drift)
                
                adjusted_nv_coords = numpy.array(nv_coords) + drift
                adjusted_depletion_coords = numpy.array(depletion_coords) + drift
                
                start_time = time_current
            
            # Break out of the while if the user says stop
            if tool_belt.safe_stop():
                break
            
            tool_belt.set_xyz(cxn, adjusted_nv_coords)
            
            # Build the list to step through the coords on readout NV and targets
            x_voltages, y_voltages = build_voltages(adjusted_nv_coords, 
                                                  adjusted_depletion_coords, num_reps)
            
            # 'Flip a coin' to determine which tau (long/shrt) is used first
            rand_boolean = numpy.random.randint(0, high=2)
            
            if rand_boolean == 1:
                tau_ind_first = tau_ind
                tau_ind_second = -tau_ind - 1
            elif rand_boolean == 0:
                tau_ind_first = -tau_ind - 1
                tau_ind_second = tau_ind
                
            # add the tau indexxes used to a list to save at the end
            tau_index_master_list[run_ind].append(tau_ind_first)
            tau_index_master_list[run_ind].append(tau_ind_second)

            print(" \nFirst relaxation time: {}".format(taus[tau_ind_first]))
            print("Second relaxation time: {}".format(taus[tau_ind_second]))
           
            seq_args = [readout_time, init_time, depletion_time, ionization_time, shelf_time ,
                    taus[tau_ind_first], taus[tau_ind_second], uwave_pi_pulse, uwave_pi_on_2_pulse, 
                    init_color, depletion_color,
                    green_laser_name, yellow_laser_name, red_laser_name, sig_gen_name,
                    apd_indices[0], readout_power, shelf_power]
            seq_args_string = tool_belt.encode_seq_args(seq_args)
            # print(seq_args)
            
            # Start the tagger stream
            cxn.apd_tagger.start_tag_stream(apd_indices)
            cxn.pulse_streamer.stream_load(seq_file_name, seq_args_string)
            
            # Load the galvo
            xy_server = tool_belt.get_xy_server(cxn) 
            xy_server.load_arb_scan_xy(x_voltages, y_voltages, int(period))
            
            # Clear the tagger buffer of any excess counts
            cxn.apd_tagger.clear_buffer()
            
            # Start the timing stream
            cxn.pulse_streamer.stream_start(int(num_reps))

            num_samples = num_reps * 3 * 4
            num_read_so_far = 0
            total_samples_list = []
            while num_read_so_far < num_samples:
        
                if tool_belt.safe_stop():
                    break
        
                # Read the samples and update the image
                new_samples = cxn.apd_tagger.read_counter_simple()
                # print(new_samples)
                num_new_samples = len(new_samples)
        
                if num_new_samples > 0:
                    # print(new_samples)
                    for el in new_samples:
                        total_samples_list.append(int(el))
                    num_read_so_far += num_new_samples
            
            sig_gate_count_1 = total_samples_list[2::12]
            sig_counts[run_ind, tau_ind_first] = sum(sig_gate_count_1)
            print("First signal = " + str(sum(sig_gate_count_1)))

            ref_gate_count_1 = total_samples_list[5::12]
            ref_counts[run_ind, tau_ind_first] = sum(ref_gate_count_1)
            print("First Reference = " + str(sum(ref_gate_count_1)))

            sig_gate_count_2 = total_samples_list[8::12]
            sig_counts[run_ind, tau_ind_second] = sum(sig_gate_count_2)
            print("Second Signal = " + str(sum(sig_gate_count_2)))

            ref_gate_count_2 = total_samples_list[11::12]
            ref_counts[run_ind, tau_ind_second] = sum(ref_gate_count_2)
            print("Second Reference = " + str(sum(ref_gate_count_2)))
            
            # print(total_samples_list)

        cxn.apd_tagger.stop_tag_stream()

        # %% Save the data we have incrementally for long measurements

        rawData = {'start_timestamp': start_timestamp,
                   'nv_sig': nv_sig,
                   'nv_sig-units': tool_belt.get_nv_sig_units(),
                    "uwave_freq": uwave_freq,
                    "uwave_freq-units": "GHz",
                    "uwave_power": uwave_power,
                    "uwave_power-units": "dBm",
                    "rabi_period": rabi_period,
                    "rabi_period-units": "ns",
                    "uwave_pi_pulse": uwave_pi_pulse,
                    "uwave_pi_pulse-units": "ns",
                    "uwave_pi_on_2_pulse": uwave_pi_on_2_pulse,
                    "uwave_pi_on_2_pulse-units": "ns",
                    "precession_time_range": precession_time_range,
                    "precession_time_range-units": "ns",
                   'state': state.name,
                   'num_steps': num_steps,
                   'run_ind': run_ind,
                   'taus': taus.tolist(),
                   'drift_list': drift_list,
                   'opti_interval': opti_interval,
                   'sig_counts': sig_counts.astype(int).tolist(),
                   'sig_counts-units': 'counts',
                   'ref_counts': ref_counts.astype(int).tolist(),
                   'ref_counts-units': 'counts'}
        
        
        avg_sig_counts = numpy.average(sig_counts[:run_ind+1], axis=0)
        avg_ref_counts = numpy.average(ref_counts[:run_ind+1], axis=0)
    
        # Replace x/0=inf with 0
        try:
            norm_avg_sig = avg_sig_counts / avg_ref_counts
        except RuntimeWarning as e:
            print(e)
            inf_mask = numpy.isinf(norm_avg_sig)
            # Assign to 0 based on the passed conditional array
            norm_avg_sig[inf_mask] = 0
            
        #  Plot the data incrementally
    
        ax = axes_pack[0]
        ax.cla()  
        # Account for the pi/2 pulse on each side of a tau
        ax.plot(plot_taus, avg_sig_counts, "r-", label="signal")
        ax.plot(plot_taus, avg_ref_counts, "g-", label="reference")
        ax.legend()
    
        ax = axes_pack[1]
        ax.cla()  
        ax.plot(plot_taus, norm_avg_sig, "b-")
        ax.set_title("Spin Echo Measurement")
        inc_fig.canvas.draw()
        inc_fig.set_tight_layout(True)
        inc_fig.canvas.flush_events()
    

        # This will continuously be the same file path so we will overwrite
        # the existing file with the latest version
        file_path = tool_belt.get_file_path(__file__, start_timestamp,
                                            nv_sig['name'], 'incremental')
        tool_belt.save_raw_data(rawData, file_path)
        tool_belt.save_figure(inc_fig, file_path)



    # %% Process and plot the data
    end_function_time = time.time()
    time_elapsed = end_function_time - start_function_time
    # print(time_elapsed)
    
    # %% Average the counts over the iterations

    avg_sig_counts = numpy.average(sig_counts, axis=0)
    avg_ref_counts = numpy.average(ref_counts, axis=0)

    # %% Calculate the ramsey data, signal / reference over different
    # relaxation times

    # Replace x/0=inf with 0
    try:
        norm_avg_sig = avg_sig_counts / avg_ref_counts
    except RuntimeWarning as e:
        print(e)
        inf_mask = numpy.isinf(norm_avg_sig)
        # Assign to 0 based on the passed conditional array
        norm_avg_sig[inf_mask] = 0
        
    # %% Plot the data

    raw_fig, axes_pack = plt.subplots(1, 2, figsize=(17, 8.5))

    ax = axes_pack[0]
    # Account for the pi/2 pulse on each side of a tau
    plot_taus = (taus + uwave_pi_pulse) / 1000
    ax.plot(plot_taus, avg_sig_counts, "r-", label="signal")
    ax.plot(plot_taus, avg_ref_counts, "g-", label="reference")
    ax.legend()
    ax.set_xlabel(r"$\tau + \pi$ ($\mathrm{\mu s}$)")
    ax.set_ylabel("Counts")

    ax = axes_pack[1]
    ax.plot(plot_taus, norm_avg_sig, "b-")
    ax.set_title("Spin Echo Measurement")
    ax.set_xlabel(r"$\tau + \pi$ ($\mathrm{\mu s}$)")
    ax.set_ylabel("Contrast (arb. units)")

    raw_fig.canvas.draw()
    raw_fig.set_tight_layout(True)
    raw_fig.canvas.flush_events()

    # %% Clean up and save the data

    tool_belt.reset_cfm(cxn)

    timestamp = tool_belt.get_time_stamp()

    rawData = {'timestamp': timestamp,
               'time_elapsed': time_elapsed,
                'nv_sig': nv_sig,
                'nv_sig-units': tool_belt.get_nv_sig_units(),
                "uwave_freq": uwave_freq,
                "uwave_freq-units": "GHz",
                "uwave_power": uwave_power,
                "uwave_power-units": "dBm",
                "rabi_period": rabi_period,
                "rabi_period-units": "ns",
                "uwave_pi_pulse": uwave_pi_pulse,
                "uwave_pi_pulse-units": "ns",
                "uwave_pi_on_2_pulse": uwave_pi_on_2_pulse,
                "uwave_pi_on_2_pulse-units": "ns",
                "precession_time_range": precession_time_range,
                "precession_time_range-units": "ns",
                'state': state.name,
                'num_steps': num_steps,
                'num_reps': num_reps,
                'num_runs': num_runs,
                'taus': taus.tolist(),
                'drift_list': drift_list,
                'opti_interval': opti_interval,
                'sig_counts': sig_counts.astype(int).tolist(),
                'sig_counts-units': 'counts',
                'ref_counts': ref_counts.astype(int).tolist(),
                'ref_counts-units': 'counts',
                "norm_avg_sig": norm_avg_sig.astype(float).tolist(),
                "norm_avg_sig-units": "arb"}
        
    name = nv_sig['name']
    filePath = tool_belt.get_file_path(__file__, timestamp, name)
    tool_belt.save_figure(raw_fig, filePath)
    tool_belt.save_raw_data(rawData, filePath)

    return sig_gate_count_1 + sig_gate_count_2, ref_gate_count_1 + ref_gate_count_2

# %%
def sweep_readout_time(nv_sig, opti_nv_sig, apd_indices, readout_time_list, depletion_coords, 
                       wait_time):
    measurement_dur = 2
    num_runs = 1
    num_steps = 2
    nv_sig['depletion_coords'] = depletion_coords 
    
    precession_time_range = [wait_time, wait_time]
    
    snr_list = []
    signal_list = []
    noise_list = []
    reps_list = []
    for t in readout_time_list:
        num_reps = int(measurement_dur * 60e9 / (2*(t + 1e6)) )
        nv_sig['charge_readout_dur'] = t
        
        sig_counts, ref_counts = main(nv_sig, opti_nv_sig, apd_indices, precession_time_range,
              num_steps, num_reps, num_runs)
        #sig_counts_avg = numpy.average(sig_counts)
        #ref_counts_avg = numpy.average(ref_counts)
        signal = numpy.average(numpy.array(sig_counts) - numpy.array(ref_counts))
        noise = numpy.std(sig_counts, ddof=1) / numpy.sqrt(num_reps)
        print('{} ms readout'.format(t/10**6))
        print('{} num reps'.format(num_reps))
        print('signal: {}'.format(signal))
        print('noise: {}'.format(noise))
        print('')
        signal_list.append(signal)
        noise_list.append(noise)
        snr_list.append(signal/noise)
        reps_list.append(num_reps)
        # return
        

    fig, ax = plt.subplots()
    ax.plot(numpy.array(readout_time_list)/10**6, snr_list)
    ax.set_xlabel('Readout time (ms)')
    ax.set_ylabel('snr')
    
    timestamp = tool_belt.get_time_stamp()
    rawData = {'timestamp': timestamp,
               'measurement_dur': measurement_dur,
               'measurement_dur-units': 'min',
                'nv_sig': nv_sig,
                'nv_sig-units': tool_belt.get_nv_sig_units(),
                'precession_time_range': precession_time_range,
                'precession_time_range-units': 'ns',
                'num_steps': num_steps,
                'num_runs': num_runs,
                'readout_time_list': readout_time_list,
                'readout_time_list-units': 'ns',
                'reps_list': reps_list,
                'signal_list': signal_list,
                'noise_list': noise_list,
                'snr_list': snr_list,
                'sig_counts': sig_counts,
                'sig_counts-units': 'counts',
                'ref_counts': ref_counts,
                'ref_counts-units': 'counts',
                }

        
    name = nv_sig['name']
    filePath = tool_belt.get_file_path(__file__, timestamp, name)
    tool_belt.save_figure(fig, filePath)
    tool_belt.save_raw_data(rawData, filePath)
    
        
    return

def sweep_readout_power(nv_sig, opti_nv_sig, apd_indices, readout_power_list, depletion_coords, 
                       wait_time):
    measurement_dur = 2
    num_runs = 1
    num_steps = 2
    nv_sig['depletion_coords'] = depletion_coords 
    
    precession_time_range = [wait_time, wait_time]
    
    snr_list = []
    signal_list = []
    noise_list = []
    charge_readout_dur = nv_sig['charge_readout_dur']
    num_reps = int(measurement_dur * 60e9 / (2*(charge_readout_dur + 1e6)) )
    for p in readout_power_list:
        nv_sig['charge_readout_laser_power'] = p
        
        sig_counts, ref_counts = main(nv_sig, opti_nv_sig, apd_indices, precession_time_range,
              num_steps, num_reps, num_runs)
        sig_counts_avg = numpy.average(sig_counts)
        ref_counts_avg = numpy.average(ref_counts)
        signal = numpy.average(numpy.array(sig_counts_avg) - numpy.array(ref_counts_avg))
        noise = numpy.std(sig_counts_avg, ddof=1) / numpy.sqrt(num_reps)
        print('{} V AOM setting'.format(p))
        print('{} num reps'.format(num_reps))
        print('signal: {}'.format(signal))
        print('noise: {}'.format(noise))
        print('')
        signal_list.append(signal)
        noise_list.append(noise)
        snr_list.append(signal/noise)
        # return
        
    fig, ax = plt.subplots()
    ax.plot(numpy.array(readout_power_list), snr_list)
    ax.set_xlabel('AOM voltage setting (V)')
    ax.set_ylabel('snr')
    
    timestamp = tool_belt.get_time_stamp()
    rawData = {'timestamp': timestamp,
               'measurement_dur': measurement_dur,
               'measurement_dur-units': 'min',
                'nv_sig': nv_sig,
                'nv_sig-units': tool_belt.get_nv_sig_units(),
                'precession_time_range': precession_time_range,
                'precession_time_range-units': 'ns',
                'num_steps': num_steps,
                'num_runs': num_runs,
                'readout_power_list': readout_power_list,
                'readout_power_list-units': 'V',
                'signal_list': signal_list,
                'noise_list': noise_list,
                'snr_list': snr_list,
                'sig_counts': sig_counts.tolist(),
                'sig_counts-units': 'counts',
                'ref_counts': ref_counts.tolist(),
                'ref_counts-units': 'counts',
                }

        
    name = nv_sig['name']
    filePath = tool_belt.get_file_path(__file__, timestamp, name)
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
        "coords": [0.1614328 , 0.13376454,4.79 ],
        
        "name": "{}-dnv5_2021_09_23".format(sample_name,),
        "disable_opt": False,
        "expected_count_rate": 75,
            'imaging_laser': green_laser, 'imaging_laser_power': green_power,
            'imaging_readout_dur': 1E7,
            
            "initialize_laser": green_laser,
            "initialize_laser_power": green_power,
            "initialize_dur": 1e4,
            
            "CPG_laser": red_laser,
            'CPG_laser_power': red_power,
            "CPG_laser_dur": 1e4,
        
            'nv0_ionization_laser': red_laser, 'nv0_ionization_laser_power': red_power,
            'nv0_ionization_dur':300,
            
            'spin_shelf_laser': yellow_laser, 'spin_shelf_laser_filter': nd_yellow, 
            'spin_shelf_laser_power': 0.4, 'spin_shelf_dur':0,
            
            'charge_readout_laser': yellow_laser, 'charge_readout_laser_filter': nd_yellow, 
            'charge_readout_laser_power': 0.3, 'charge_readout_dur':0.5e6,
            
            'collection_filter': '630_lp',  'magnet_angle': 114,
            
                    "resonance_LOW":2.7897,"rabi_LOW": 139.7,"uwave_power_LOW": 15.5, 
        "resonance_HIGH": 2.9496,"rabi_HIGH": 215,"uwave_power_HIGH": 14.5}  
    
    
    
    # max_time = 100 # us
    # num_steps = int(max_time + 1)  # 1 point per us
    # precession_time_range = [0, max_time * 10 ** 3]
    
    start = 0
    stop =20
    num_steps = int((stop - start)*1 + 1)  # 1 point per us
    precession_time_range = [start *1e3, stop *1e3]
    
    num_reps = int(1e3)
    num_runs = 1 #60
    
    A = [0.153, 0.125, 4.79]
    B = [0.148, 0.125, 4.79]
    
    depletion_point = [A]#, B]
    
    depletion_times = [3e3, 2e3]
    do_plot = False
    
    try:
        
        if not do_plot:
             for p in range(len(depletion_point)):   
                nv_sig['depletion_coords'] = depletion_point[p]
                nv_sig['CPG_laser_dur'] = depletion_times[p]
               
                main(nv_sig, opti_nv_sig, apd_indices,precession_time_range,
                  num_steps, num_reps, num_runs)
            
            # for i in [ 5]:#range(5):
                
                 # for p in range(len(depletion_point)):
                 #     nv_sig['depletion_coords'] = depletion_point[p]
                 #     nv_sig['CPG_laser_dur'] = depletion_times[p]
            
                 #     t = (i ) * 30.5
                 #     start = t-10
                 #     if t < 10:
                 #         start = 0
                 #     stop = t+10
                 #     num_steps = int((stop - start)*2 + 1)  # 1 point per us
                 #     precession_time_range = [start *1e3, stop *1e3]
                    
                 #     main(nv_sig, opti_nv_sig, apd_indices,precession_time_range,
                 #       num_steps, num_reps, num_runs)
                
            # readout_time_list = [1e5, 5e5, 6e5, 7e5, 8e5, 9e5, 1e6,2e6,  2.5e6, 3e6,3.5e6, 4e6, 5e6, 1e7,  2e7]
            # readout_time_list = numpy.linspace(0,3,13)*1e6
            # readout_time_list = readout_time_list.tolist()
            #sweep_readout_time(nv_sig, opti_nv_sig, apd_indices, readout_time_list, A, 
             #          30)
            
            # readout_power_list = [0.05, 0.1,0.15, 0.2,0.25, 0.3,0.35, 0.4, 0.45, 0.5,0.55, 0.6]
            # sweep_readout_power(nv_sig,opti_nv_sig,  apd_indices, readout_power_list, A, 30)
        
        
            
            # nv_sig['depletion_coords'] = B
            # main(nv_sig, opti_nv_sig, apd_indices, precession_time_range,
            #         num_steps, num_reps, num_runs)
            
            # nv_sig['depletion_coords'] = C
            # main(nv_sig, opti_nv_sig, apd_indices, precession_time_range,
            #         num_steps, num_reps, num_runs)
        
        else:
            
            
            folder = 'pc_rabi/branch_master/super_resolution_spin_echo/2021_10'
            # ++++ COMPARE +++++
            
            file_list_0 =[ '',
                        '']
            file_list_30 =[ '2021_10_05-19_18_00-johnson-dnv5_2021_09_23',
                        '2021_10_05-20_31_21-johnson-dnv5_2021_09_23']
            file_list_60 =[ '2021_10_05-23_09_33-johnson-dnv5_2021_09_23',
                        '2021_10_06-00_25_35-johnson-dnv5_2021_09_23']
            file_list_90 =[ '2021_10_06-01_44_05-johnson-dnv5_2021_09_23',
                        '2021_10_06-03_02_34-johnson-dnv5_2021_09_23']
            file_list_120 =[ '2021_10_06-09_31_43-johnson-dnv5_2021_09_23',
                        '2021_10_06-10_52_10-johnson-dnv5_2021_09_23']
            file_list_150 =[ '2021_10_06-13_34_33-johnson-dnv5_2021_09_23',
                        '2021_10_06-14_57_46-johnson-dnv5_2021_09_23']
            fmt_list = ['b-', 'r-']
            label_list = ['A', 'B']
            fig, ax = plt.subplots(figsize=(8.5, 8.5))
            
            file_list = file_list_150
            
            for f in range(len(file_list)):
                file = file_list[f]
                data = tool_belt.get_raw_data(file, folder)
                taus = data['taus']
                norm_avg_sig = numpy.array(data['norm_avg_sig'])
                uwave_pi_pulse = data['uwave_pi_pulse']
                plot_taus = (numpy.array(taus) + uwave_pi_pulse) / 1000
            
                #ax.plot(plot_taus, norm_avg_sig, fmt_list[f], label = label_list[f])
            #ax.set_ylabel('Contrast (arb. units)')
            #ax.set_xlabel('Taus (us)')
            #ax.legend(loc='lower right')
            
            
            file = 'incremental/2021_10_07-00_05_19-johnson-dnv5_2021_09_23'
            data = tool_belt.get_raw_data(file, folder)
            taus = data['taus']
            sig_counts = numpy.array(data['sig_counts'])
            ref_counts = numpy.array(data['ref_counts'])
            run_ind =data['run_ind']
            
            avg_sig_counts = numpy.average(sig_counts[:30], axis=0)
            avg_ref_counts = numpy.average(ref_counts[:30], axis=0)
            norm= avg_sig_counts/avg_ref_counts
            uwave_pi_pulse = data['uwave_pi_pulse']
            plot_taus = (numpy.array(taus) + uwave_pi_pulse) / 1000
                                                                                                                        
            ax.plot(plot_taus, norm)
            ax.set_ylabel('Contrast (arb. units)')
            ax.set_xlabel('Taus (us)')
            ax.legend(loc='lower right')
            
            
            
            combine_revivals(file_list, folder)
            
            
            
            
            
            
            
            # file_list_A =[ '2021_10_05-21_15_13-johnson-dnv5_2021_09_23',
            #             '2021_10_06-07_32_54-johnson-dnv5_2021_09_23']
            
            # file_list_B =[ '2021_10_05-21_53_26-johnson-dnv5_2021_09_23',
            #             '2021_10_06-08_10_59-johnson-dnv5_2021_09_23']
            
            # fmt_list = ['b-', 'r-']
            # label_list = ['A', 'B']
            # fig, ax = plt.subplots(figsize=(8.5, 8.5))
            # norm_avg_sig_A = []
            # norm_avg_sig_B = []
            
            # for f in range(len(file_list_A)):
            #     file = file_list_A[f]
            #     data = tool_belt.get_raw_data(file, folder)
            #     taus = data['taus']
            #     norm_avg_sig = numpy.array(data['norm_avg_sig'])
            #     uwave_pi_pulse = data['uwave_pi_pulse']
            #     plot_taus = (numpy.array(taus) + uwave_pi_pulse) / 1000
            #     if f == 0:
            #         norm_avg_sig_A = norm_avg_sig
            #     else:
            #         norm_avg_sig_A = norm_avg_sig_A + norm_avg_sig
            # for f in range(len(file_list_B)):
            #     file = file_list_B[f]
            #     data = tool_belt.get_raw_data(file, folder)
            #     taus = data['taus']
            #     norm_avg_sig = numpy.array(data['norm_avg_sig'])
            #     uwave_pi_pulse = data['uwave_pi_pulse']
            #     plot_taus = (numpy.array(taus) + uwave_pi_pulse) / 1000
            #     if f == 0:
            #         norm_avg_sig_B = norm_avg_sig
            #     else:
            #         norm_avg_sig_B = norm_avg_sig_B + norm_avg_sig
                
            # f_A = norm_avg_sig_A[0]
            # l_A = norm_avg_sig_A[-1]
            
            # scaled_sig_A = (norm_avg_sig_A - f_A) / (l_A - f_A) 
            
            # f_B = norm_avg_sig_B[0]
            # l_B = norm_avg_sig_B[-1]
            
            # scaled_sig_B = (norm_avg_sig_B - f_B) / (l_B - f_B) 
            
            
            # ax.plot(plot_taus, scaled_sig_A, fmt_list[0], label = label_list[0])
            # ax.plot(plot_taus, scaled_sig_B, fmt_list[1], label = label_list[1])
            # ax.set_ylabel('Scaled contrast (arb. units)')
            
            # # ax.plot(plot_taus, norm_avg_sig_A/len(file_list_A), fmt_list[0], label = label_list[0])
            # # ax.plot(plot_taus, norm_avg_sig_B/len(file_list_A), fmt_list[1], label = label_list[1])
            # # ax.set_ylabel('Contrast (arb. units)')
                
                
            # ax.set_xlabel('Taus (us)')
            # ax.legend(loc='lower right')
                
                
                
            

    
    finally:
        # Reset our hardware - this should be done in each routine, but
        # let's double check here
        # tool_belt.reset_cfm()
        # Kill safe stop
        if tool_belt.check_safe_stop_alive():
            print('\n\nRoutine complete. Press enter to exit.')
            tool_belt.poll_safe_stop()
