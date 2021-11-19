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
import labrad
from utils.tool_belt import States
from random import shuffle
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

# %% fit

def extract_oscillations(norm_avg_sig, precession_time_range, num_steps, detuning):
    # Create an empty array for the frequency arrays
    FreqParams = numpy.empty([3])
    # Perform the fft
    time_step = (precession_time_range[1]/1e3 - precession_time_range[0]/1e3) \
                                                    / (num_steps - 1)

    transform = numpy.fft.rfft(norm_avg_sig)
#    window = max_precession_time - min_precession_time
    freqs = numpy.fft.rfftfreq(num_steps, d=time_step)
    transform_mag = numpy.absolute(transform)

    # Plot the fft
    fig_fft, ax= plt.subplots(1, 1, figsize=(10, 8))
    ax.plot(freqs[1:], transform_mag[1:])  # [1:] excludes frequency 0 (DC component)
    ax.set_xlabel('Frequency (MHz)')
    ax.set_ylabel('FFT magnitude')
    ax.set_title('Ramsey FFT')
    fig_fft.canvas.draw()
    fig_fft.canvas.flush_events()


    # Guess the peaks in the fft. There are parameters that can be used to make
    # this more efficient
    freq_guesses_ind = find_peaks(transform_mag[1:]
                                  , prominence = 0.5
#                                  , height = 0.8
#                                  , distance = 2.2 / freq_step
                                  )

#    print(freq_guesses_ind[0])

    # Check to see if there are three peaks. If not, try the detuning passed in
    if len(freq_guesses_ind[0]) != 3:
        print('Number of frequencies found: {}'.format(len(freq_guesses_ind[0])))
#        detuning = 3 # MHz

        FreqParams[0] = detuning - 2.2
        FreqParams[1] = detuning
        FreqParams[2] = detuning + 2.2
    else:
        FreqParams[0] = freqs[freq_guesses_ind[0][0]]
        FreqParams[1] = freqs[freq_guesses_ind[0][1]]
        FreqParams[2] = freqs[freq_guesses_ind[0][2]]
        
    return fig_fft, FreqParams # In MHz

def fit_ramsey(norm_avg_sig,taus,  precession_time_range, FreqParams):
    
    taus_us = numpy.array(taus)/1e3
    # Guess the other params for fitting
    amp_1 = -0.1
    amp_2 = -amp_1
    amp_3 = -amp_1
    decay = 10
    offset = 1.1

    guess_params = (offset, decay, amp_1, FreqParams[0],
                        amp_2, FreqParams[1],
                        amp_3, FreqParams[2])

    # Try the fit to a sum of three cosines

    try:
        popt,pcov = curve_fit(tool_belt.cosine_sum, taus_us, norm_avg_sig,
                      p0=guess_params,
                       bounds=([0, 0,
                                -numpy.infty, 0,
                                -numpy.infty, 0,
                                -numpy.infty, 0,
                                ], 
                               [numpy.infty, numpy.infty, 
                                numpy.infty, numpy.infty,
                                numpy.infty, numpy.infty,
                                numpy.infty, numpy.infty, ])
                      )
    except Exception:
        print('Something went wrong!')
        popt = guess_params
    print(popt)

    taus_us_linspace = numpy.linspace(precession_time_range[0]/1e3, precession_time_range[1]/1e3,
                          num=1000)

    fig_fit, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.plot(taus_us, norm_avg_sig,'b',label='data')
    ax.plot(taus_us_linspace, tool_belt.cosine_sum(taus_us_linspace,*popt),'r',label='fit')
    ax.set_xlabel('Free precesion time (ns)')
    ax.set_ylabel('Contrast (arb. units)')
    ax.legend()
    text1 = "\n".join((r'$C + e^{-t/d} [a_1 \mathrm{cos}(2 \pi \nu_1 t) + a_2 \mathrm{cos}(2 \pi \nu_2 t) + a_3 \mathrm{cos}(2 \pi \nu_3 t)]$',
                       r'$d = $' + '%.2f'%(abs(popt[1])) + ' us',
                       r'$\nu_1 = $' + '%.2f'%(popt[3]) + ' MHz',
                       r'$\nu_2 = $' + '%.2f'%(popt[5]) + ' MHz',
                       r'$\nu_3 = $' + '%.2f'%(popt[7]) + ' MHz'
                       ))
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)

    ax.text(0.40, 0.25, text1, transform=ax.transAxes, fontsize=12,
                            verticalalignment="top", bbox=props)



#  Plot the data itself and the fitted curve

    fig_fit.canvas.draw()
#    fig.set_tight_layout(True)
    fig_fit.canvas.flush_events()
    
    return fig_fit


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

# %% Main


def main(nv_sig, opti_nv_sig, apd_indices, precession_time_range, detuning, 
         num_steps, num_reps, num_runs,  
         state=States.LOW):

    with labrad.connect() as cxn:
        sig_gate_counts, ref_gate_counts = main_with_cxn(cxn, nv_sig, opti_nv_sig, 
                             apd_indices, precession_time_range, detuning,
                             num_steps, num_reps, num_runs,
                             state)
    return sig_gate_counts, ref_gate_counts


def main_with_cxn(cxn, nv_sig, opti_nv_sig, apd_indices, precession_time_range,
         detuning, num_steps, num_reps, num_runs, 
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
    # print(taus)
    #return
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
    # Detune the pi/2 pulse frequency
    uwave_freq_detuned = uwave_freq + detuning / 10**3
    
    # For Ramsey, run spin echo sequence, but set pi pulse to 0
    uwave_pi_pulse = 0
    uwave_pi_on_2_pulse = tool_belt.get_pi_on_2_pulse_dur(rabi_period)

    readout_time = nv_sig['charge_readout_dur']
    init_time = nv_sig['initialize_dur']
    depletion_time = nv_sig['CPG_laser_dur']
    readout_power = nv_sig['charge_readout_laser_power']
    ionization_time = nv_sig['nv0_ionization_dur']
    shelf_time = nv_sig['spin_shelf_dur']
    shelf_power = nv_sig['spin_shelf_laser_power']
    
    magnet_angle = nv_sig['magnet_angle']
    if (magnet_angle is not None) and hasattr(cxn, "rotation_stage_ell18k"):
        cxn.rotation_stage_ell18k.set_angle(magnet_angle)
    
    green_laser_name = nv_sig['imaging_laser']
    red_laser_name = nv_sig['nv0_ionization_laser']
    yellow_laser_name = nv_sig['charge_readout_laser']
    sig_gen_name = tool_belt.get_signal_generator_name_no_cxn(state) 
    sig_gen_cxn = tool_belt.get_signal_generator_cxn(cxn, state)   
            
    init_color = tool_belt.get_registry_entry_no_cxn('wavelength',
                      ['Config', 'Optics', nv_sig['initialize_laser']])
    depletion_color = tool_belt.get_registry_entry_no_cxn('wavelength',
                      ['Config', 'Optics', nv_sig['CPG_laser']])
    
    
    # Set the charge readout (assumed to be yellow here) to the correct filter
    if 'charge_readout_laser_filter' in nv_sig:
        tool_belt.set_filter(cxn, nv_sig, 'charge_readout_laser')
    
    
    seq_args = [readout_time, init_time, depletion_time, ionization_time, shelf_time ,
          min_precession_time/2, max_precession_time/2, uwave_pi_pulse, uwave_pi_on_2_pulse, 
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
    #return

    # %% Get the starting time of the function

    start_timestamp = tool_belt.get_time_stamp()

    # %% Collect the data

    # Start 'Press enter to stop...'
    tool_belt.init_safe_stop()
    
    start_time = time.time()
    start_function_time = start_time
    
    raw_fig, axes_pack = plt.subplots(1, 2, figsize=(17, 8.5))
    plot_taus = (taus + uwave_pi_pulse) / 1000
    ax = axes_pack[0]
    ax.set_xlabel(r"$\taui$ ($\mathrm{\mu s}$)")
    ax.set_ylabel("Counts")
    
    ax = axes_pack[1]
    ax.set_xlabel(r"$\tau$ ($\mathrm{\mu s}$)")
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
            
            # Set up the microwaves and laser. Then load the pulse streamer 
            # (must happen after optimize and iq_switch since run their
            # own sequences)
            # Set up the microwaves
            sig_gen_cxn.set_freq(uwave_freq_detuned)
            sig_gen_cxn.set_amp(uwave_power)
            sig_gen_cxn.uwave_on()
        
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
                    taus[tau_ind_first]/2, taus[tau_ind_second]/2, uwave_pi_pulse, uwave_pi_on_2_pulse, 
                    init_color, depletion_color,
                    green_laser_name, yellow_laser_name, red_laser_name, sig_gen_name,
                    apd_indices[0], readout_power, shelf_power]
            seq_args_string = tool_belt.encode_seq_args(seq_args)
            # print(seq_args)
            # return
            
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
                   'opti_nv_sig': opti_nv_sig,
                           'detuning': detuning,
                                   'detuning-units': 'MHz',
                    "uwave_freq": uwave_freq_detuned,
                    "uwave_freq-units": "GHz",
                    "uwave_power": uwave_power,
                    "uwave_power-units": "dBm",
                    "rabi_period": rabi_period,
                    "rabi_period-units": "ns",
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
                   'tau_index_master_list': tau_index_master_list,
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
        ax.set_title("Ramsey Measurement")
        raw_fig.canvas.draw()
        raw_fig.set_tight_layout(True)
        raw_fig.canvas.flush_events()
    

        # This will continuously be the same file path so we will overwrite
        # the existing file with the latest version
        file_path = tool_belt.get_file_path(__file__, start_timestamp,
                                            nv_sig['name'], 'incremental')
        tool_belt.save_raw_data(rawData, file_path)
        tool_belt.save_figure(raw_fig, file_path)



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

    #raw_fig, axes_pack = plt.subplots(1, 2, figsize=(17, 8.5))

    ax = axes_pack[0]
    ax.cla() 
    # Account for the pi/2 pulse on each side of a tau
    plot_taus = (taus + uwave_pi_pulse) / 1000
    ax.plot(plot_taus, avg_sig_counts, "r-", label="signal")
    ax.plot(plot_taus, avg_ref_counts, "g-", label="reference")
    ax.legend()
    ax.set_xlabel(r"$\tau$ ($\mathrm{\mu s}$)")
    ax.set_ylabel("Counts")

    ax = axes_pack[1]
    ax.cla() 
    ax.plot(plot_taus, norm_avg_sig, "b-")
    ax.set_title("Ramsey Measurement")
    ax.set_xlabel(r"$\tau$ ($\mathrm{\mu s}$)")
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
                        'detuning': detuning,
                                'detuning-units': 'MHz',
                "uwave_freq": uwave_freq_detuned,
                "uwave_freq-units": "GHz",
                "uwave_power": uwave_power,
                "uwave_power-units": "dBm",
                "rabi_period": rabi_period,
                "rabi_period-units": "ns",
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
                'tau_index_master_list': tau_index_master_list,
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

# %% Run the file


if __name__ == "__main__":


    folder = "pc_rabi/branch_master/super_resolution_ramsey/2021_10"
    file = '2021_10_15-17_39_48-johnson-dnv5_2021_09_23'
    # file = '2021_10_15-20_32_32-johnson-dnv5_2021_09_23'
    
    # detuning = 0
    data = tool_belt.get_raw_data(file, folder)
    detuning= data['detuning']
    norm_avg_sig = data['norm_avg_sig']
    precession_time_range = data['precession_time_range']
    num_steps = data['num_steps']
    try:
        taus = data['taus']
    except Exception:
        
        taus = numpy.linspace(
            precession_time_range[0],
            precession_time_range[1],
            num=num_steps,
        )
        
        
    _, FreqParams = extract_oscillations(norm_avg_sig, precession_time_range, num_steps, detuning)
    print(FreqParams)
    f = 4
    FreqParams = [f-2.2, f+2.2, f]
    fit_ramsey(norm_avg_sig,taus,  precession_time_range, FreqParams)
