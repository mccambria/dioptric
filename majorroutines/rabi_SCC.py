# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 13:40:36 2020

This routine performs Rabi, but readouts with SCC

This routine tests rabi under various readout routines: regular green readout,
regular yellow readout, and SCC readout.

@author: agardill
"""

# %% Imports


import utils.tool_belt as tool_belt
import majorroutines.optimize as optimize
import numpy
import os
import time
import matplotlib.pyplot as plt
from random import shuffle
from scipy.optimize import curve_fit
import labrad
from utils.tool_belt import States


# %% Functions


def fit_data(uwave_time_range, num_steps, norm_avg_sig):

    # %% Set up

    min_uwave_time = uwave_time_range[0]
    max_uwave_time = uwave_time_range[1]
    taus, tau_step = numpy.linspace(min_uwave_time, max_uwave_time,
                            num=num_steps, dtype=numpy.int32, retstep=True)

    fit_func = tool_belt.cosexp_scc

    # %% Estimated fit parameters

    offset = numpy.average(norm_avg_sig)
    amplitude = 1.0 - offset
    frequency = 1/75  # Could take Fourier transform
    decay = 1000

    # To estimate the frequency let's find the highest peak in the FFT
    transform = numpy.fft.rfft(norm_avg_sig)
    freqs = numpy.fft.rfftfreq(num_steps, d=tau_step)
    transform_mag = numpy.absolute(transform)
    # [1:] excludes frequency 0 (DC component)
    max_ind = numpy.argmax(transform_mag[1:])
    frequency = freqs[max_ind + 1]

    # %% Fit

    init_params = [offset, amplitude, frequency, decay]
    init_params=[1.005, -0.005, 1/118.7, 847]

    try:
        popt, _ = curve_fit(fit_func, taus, norm_avg_sig,
                               p0=init_params)
    except Exception as e:
        print(e)
        popt = None

    return fit_func, popt

def create_fit_figure(uwave_time_range, uwave_freq, num_steps, norm_avg_sig,
                      fit_func, popt):

    min_uwave_time = uwave_time_range[0]
    max_uwave_time = uwave_time_range[1]
    taus = numpy.linspace(min_uwave_time, max_uwave_time,
                          num=num_steps, dtype=numpy.int32)
    linspaceTau = numpy.linspace(min_uwave_time, max_uwave_time, num=1000)

    text = 40
    fit_fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(taus, norm_avg_sig,'bo', linewidth = 3,label='data')
    ax.plot(linspaceTau, fit_func(linspaceTau, *popt), 'r-', linewidth = 3, label='fit')
    ax.set_xlabel('Microwave duration (ns)', fontsize = text)
    ax.set_ylabel('Normalized signal', fontsize = text)
    ax.tick_params(which = 'both', length=8, width=2, colors='k',
                direction='in',grid_alpha=0.7, labelsize = text)
#    ax.set_ylim([0.982, 1.021])
    ax.set_yticks([ 0.99, 1.0, 1.01, 1.02])
#    ax.set_title('Rabi Oscillation Of NV Center Electron Spin')
    ax.legend(fontsize = text)
    text_freq = 'Resonant frequency:' + '%.3f'%(uwave_freq) + 'GHz'
    
    text_popt = '\n'.join((r'$C + A_0 e^{-t/d} \mathrm{cos}(2 \pi \nu t + \phi)$',
                      r'$C = $' + '%.3f'%(popt[0]),
                      r'$A_0 = $' + '%.3f'%(popt[1]),
                      r'$\frac{1}{\nu} = $' + '%.1f'%(1/popt[2]) + ' ns',
                      r'$d = $' + '%i'%(popt[3]) + ' ' + r'$ ns$'))

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
#    ax.text(0.55, 0.25, text_popt, transform=ax.transAxes, fontsize=12,
#            verticalalignment='top', bbox=props)
#    ax.text(0.55, 0.3, text_freq, transform=ax.transAxes, fontsize=12,
#            verticalalignment='top', bbox=props)

    fit_fig.canvas.draw()
    fit_fig.set_tight_layout(True)
    fit_fig.canvas.flush_events()

    return fit_fig

# %% Main


def main(nv_sig, apd_indices, uwave_time_range, state,
         num_steps, num_reps, num_runs):

    with labrad.connect() as cxn:
        rabi_per, sig_counts, ref_counts = main_with_cxn(cxn, nv_sig, apd_indices, uwave_time_range, state,
                      num_steps, num_reps, num_runs)
        
        return rabi_per, sig_counts, ref_counts
def main_with_cxn(cxn, nv_sig, apd_indices, uwave_time_range, state,
                  num_steps, num_reps, num_runs):

    tool_belt.reset_cfm(cxn)

    # %% Get the starting time of the function, to be used to calculate run time

    startFunctionTime = time.time()
    start_timestamp = tool_belt.get_time_stamp()

    # %% Initial calculations and setup

    uwave_freq = nv_sig['resonance_{}'.format(state.name)]
    uwave_power = nv_sig['uwave_power_{}'.format(state.name)]

    # parameters from nv_sig
    readout_time = nv_sig['pulsed_SCC_readout_dur']
    readout_power = nv_sig['am_589_power']
    init_ion_time = nv_sig['pulsed_initial_ion_dur']
    reion_time = nv_sig['pulsed_reionization_dur']
    ion_time = nv_sig['pulsed_ionization_dur']
    shelf_time = nv_sig['pulsed_shelf_dur']
    shelf_power = nv_sig['am_589_shelf_power']

    shared_params = tool_belt.get_shared_parameters_dict(cxn)

    #delay of aoms and laser
    laser_515_delay = shared_params['515_laser_delay']
    aom_589_delay = shared_params['589_aom_delay']
    laser_638_delay = shared_params['638_DM_laser_delay']
    rf_delay = shared_params['uwave_delay']   
    # wait time between pulses
    wait_time = shared_params['post_polarization_wait_dur']

    # Array of times to sweep through
    # Must be ints since the pulse streamer only works with int64s
    min_uwave_time = uwave_time_range[0]
    max_uwave_time = uwave_time_range[1]
    taus = numpy.linspace(min_uwave_time, max_uwave_time,
                          num=num_steps)

    # Analyze the sequence
    file_name = 'SCC_optimize_pulses_w_uwaves.py'
    seq_args = [readout_time, init_ion_time, reion_time, ion_time, taus[0],
        shelf_time , wait_time, max_uwave_time, laser_515_delay, aom_589_delay, laser_638_delay, rf_delay,
        apd_indices[0], readout_power, shelf_power, state.value]
#    seq_args = [int(el) for el in seq_args]
#    print(seq_args)
#    return
    seq_args_string = tool_belt.encode_seq_args(seq_args)
    cxn.pulse_streamer.stream_load(file_name, seq_args_string)

    # Set up our data structure, an array of NaNs that we'll fill
    # incrementally. NaNs are ignored by matplotlib, which is why they're
    # useful for us here.
    # We define 2D arrays, with the horizontal dimension for the frequency and
    # the veritical dimension for the index of the run.
    sig_counts = numpy.empty([num_runs, num_steps], dtype=numpy.uint32)
    sig_counts[:] = numpy.nan
    ref_counts = numpy.copy(sig_counts)
    # norm_avg_sig = numpy.empty([num_runs, num_steps])

    # %% Make some lists and variables to save at the end

    opti_coords_list = []
    tau_index_master_list = [[] for i in range(num_runs)]

    # Create a list of indices to step through the taus. This will be shuffled
    tau_ind_list = list(range(0, num_steps))

    # %% Collect the data

    # Start 'Press enter to stop...'
    tool_belt.init_safe_stop()

    for run_ind in range(num_runs):

        print('Run index: {}'. format(run_ind))

        # Break out of the while if the user says stop
        if tool_belt.safe_stop():
            break

        # Optimize
        opti_coords = optimize.main_with_cxn(cxn, nv_sig, apd_indices, 532, disable = True)
        opti_coords_list.append(opti_coords)

        # Apply the microwaves
        sig_gen_cxn = tool_belt.get_signal_generator_cxn(cxn, state)
        sig_gen_cxn.set_freq(uwave_freq)
        sig_gen_cxn.set_amp(uwave_power)
        sig_gen_cxn.uwave_on()

        # Load the APD
        cxn.apd_tagger.start_tag_stream(apd_indices)

        # Shuffle the list of indices to use for stepping through the taus
        shuffle(tau_ind_list)

        for tau_ind in tau_ind_list:
            # Break out of the while if the user says stop
            if tool_belt.safe_stop():
                break
            
            # shine the red laser for a few seconds before the sequence
            cxn.pulse_streamer.constant([7], 0.0, 0.0)
            time.sleep(2)
            
            # Load the sequence
            cxn.pulse_streamer.stream_load(file_name, seq_args_string)
            
            # add the tau indexxes used to a list to save at the end
            tau_index_master_list[run_ind].append(tau_ind)

            # Stream the sequence
            seq_args = [readout_time, init_ion_time, reion_time, ion_time, taus[tau_ind],
                shelf_time , wait_time, max_uwave_time, laser_515_delay, aom_589_delay, laser_638_delay, rf_delay,
                apd_indices[0], readout_power, shelf_power, state.value]
#            print(seq_args)
            seq_args_string = tool_belt.encode_seq_args(seq_args)
            cxn.apd_tagger.clear_buffer()
            cxn.pulse_streamer.stream_immediate(file_name, num_reps,
                                                seq_args_string)

            # Get the counts
#            now = time.time()
            new_counts = cxn.apd_tagger.read_counter_separate_gates(1)
#            print(new_counts)
#            print(time.time() - now)

            sample_counts = new_counts[0]

            # signal counts are even - get every second element starting from 0
            sig_gate_counts = sample_counts[0::2]
            sig_counts[run_ind, tau_ind] = sum(sig_gate_counts)
            if sum(sig_gate_counts) == 0:
                print('Oh no, the signals colleted at run {} and tau {} ns are exactly 0!'.format(run_ind,  taus[tau_ind]))

            # ref counts are odd - sample_counts every second element starting from 1
            ref_gate_counts = sample_counts[1::2]
            ref_counts[run_ind, tau_ind] = sum(ref_gate_counts)
            if sum(ref_gate_counts) == 0:
                print('Oh no, the references colleted at run {} and tau {} ns are exactly 0!'.format(run_ind,  taus[tau_ind]))

        cxn.apd_tagger.stop_tag_stream()

        # %% Save the data we have incrementally for long measurements

        raw_data = {'start_timestamp': start_timestamp,
                    'nv_sig': nv_sig,
                    'nv_sig-units': tool_belt.get_nv_sig_units(),
                    'shelf_power': shelf_power,
                    'shelf_power-units': 'mW',
                    'uwave_freq': uwave_freq,
                    'uwave_freq-units': 'GHz',
                    'uwave_power': uwave_power,
                    'uwave_power-units': 'dBm',
                    'uwave_time_range': uwave_time_range,
                    'uwave_time_range-units': 'ns',
                    'state': state.name,
                    'num_steps': num_steps,
                    'num_reps': num_reps,
                    'num_runs': num_runs,
                    'tau_index_master_list':tau_index_master_list,
                    'opti_coords_list': opti_coords_list,
                    'opti_coords_list-units': 'V',
                    'sig_counts': sig_counts.astype(int).tolist(),
                    'sig_counts-units': 'counts',
                    'ref_counts': ref_counts.astype(int).tolist(),
                    'ref_counts-units': 'counts'}

        # This will continuously be the same file path so we will overwrite
        # the existing file with the latest version
        file_path = tool_belt.get_file_path(__file__, start_timestamp,
                                            nv_sig['name'], 'incremental')
        tool_belt.save_raw_data(raw_data, file_path)

    # %% Average the counts over the iterations

    avg_sig_counts = numpy.average(sig_counts, axis=0)
    avg_ref_counts = numpy.average(ref_counts, axis=0)

    # %% Calculate the Rabi data, signal / reference over different Tau

    # Replace x/0=inf with 0
    try:
        norm_avg_sig = avg_sig_counts / avg_ref_counts
    except RuntimeWarning as e:
        print(e)
        inf_mask = numpy.isinf(norm_avg_sig)
        # Assign to 0 based on the passed conditional array
        norm_avg_sig[inf_mask] = 0
        
    # %% Fit the data and extract piPulse

    fit_func, popt = fit_data(uwave_time_range, num_steps, norm_avg_sig)

    # %% Plot the Rabi signal

    raw_fig, axes_pack = plt.subplots(1, 2, figsize=(17, 8.5))

    ax = axes_pack[0]
    ax.plot(taus, avg_sig_counts, 'r-', label = 'signal')
    ax.plot(taus, avg_ref_counts, 'g-', label = 'refernece')
    ax.set_xlabel('Microwave duration (ns)')
    ax.set_ylabel('Counts')
    ax.legend()

    ax = axes_pack[1]
    ax.plot(taus , norm_avg_sig, 'b-')
    ax.set_title('Rabi measurement with SCC readout')
    ax.set_xlabel('Microwave duration (ns)')
    ax.set_ylabel('Contrast (arb. units)')

    raw_fig.canvas.draw()
    raw_fig.set_tight_layout(True)
    raw_fig.canvas.flush_events()

    # %% Plot the data itself and the fitted curve

    fit_fig = None
    if (fit_func is not None) and (popt is not None):
        fit_fig = create_fit_figure(uwave_time_range, uwave_freq, num_steps,
                                    norm_avg_sig, fit_func, popt)
        rabi_period = 1/popt[2]
        print('Rabi period measured: {} ns\n'.format('%.1f'%rabi_period))


    # %% Measure laser powers
    
#    # measure laser powers:
#    green_optical_power_pd, green_optical_power_mW, \
#            red_optical_power_pd, red_optical_power_mW, \
#            yellow_optical_power_pd, yellow_optical_power_mW = \
#            tool_belt.measure_g_r_y_power( 
#                              nv_sig['am_589_power'], nv_sig['nd_filter'])
#            
#    # measure the power of the shelf pulse
#    optical_power = tool_belt.opt_power_via_photodiode(589, 
#                                    AO_power_settings = nv_sig['am_589_shelf_power'], 
#                                    nd_filter = nv_sig['nd_filter'])
#    shelf_power = tool_belt.calc_optical_power_mW(589, optical_power)
    
    # %% Clean up and save the data

    tool_belt.reset_cfm(cxn)

    endFunctionTime = time.time()

    timeElapsed = endFunctionTime - startFunctionTime

    timestamp = tool_belt.get_time_stamp()

    raw_data = {'timestamp': timestamp,
                'timeElapsed': timeElapsed,
                'timeElapsed-units': 's',
                'nv_sig': nv_sig,
                'nv_sig-units': tool_belt.get_nv_sig_units(),
#                'green_optical_power_pd': green_optical_power_pd,
#                'green_optical_power_pd-units': 'V',
#                'green_optical_power_mW': green_optical_power_mW,
#                'green_optical_power_mW-units': 'mW',
#                'red_optical_power_pd': red_optical_power_pd,
#                'red_optical_power_pd-units': 'V',
#                'red_optical_power_mW': red_optical_power_mW,
#                'red_optical_power_mW-units': 'mW',
#                'yellow_optical_power_pd': yellow_optical_power_pd,
#                'yellow_optical_power_pd-units': 'V',
#                'yellow_optical_power_mW': yellow_optical_power_mW,
#                'yellow_optical_power_mW-units': 'mW',
#                'shelf_power': shelf_power,
#                'shelf_power-units': 'mW',
                'uwave_freq': uwave_freq,
                'uwave_freq-units': 'GHz',
                'uwave_power': uwave_power,
                'uwave_power-units': 'dBm',
                'uwave_time_range': uwave_time_range,
                'uwave_time_range-units': 'ns',
                'state': state.name,
                'num_steps': num_steps,
                'num_reps': num_reps,
                'num_runs': num_runs,
                'tau_index_master_list':tau_index_master_list,
                'opti_coords_list': opti_coords_list,
                'opti_coords_list-units': 'V',
                'sig_counts': sig_counts.astype(int).tolist(),
                'sig_counts-units': 'counts',
                'ref_counts': ref_counts.astype(int).tolist(),
                'ref_counts-units': 'counts',
                'norm_avg_sig': norm_avg_sig.astype(float).tolist(),
                'norm_avg_sig-units': 'arb'}

    file_path = tool_belt.get_file_path(__file__, timestamp, nv_sig['name'])
    tool_belt.save_figure(raw_fig, file_path)
    if fit_fig is not None:
        tool_belt.save_figure(fit_fig, file_path + '-fit')
    tool_belt.save_raw_data(raw_data, file_path)
    
    if (fit_func is not None) and (popt is not None):
        return rabi_period, sig_counts, ref_counts
    else:
        return None, sig_counts, ref_counts
    
    
# %%
if __name__ == '__main__':
    sample_name = 'bachman'
    ensemble = { 'coords': [0.408, -0.118,4.66],
            'name': '{}-B5'.format(sample_name),
            'expected_count_rate': 1000, 'nd_filter': 'nd_0',
            'pulsed_readout_dur': 300,
            'pulsed_SCC_readout_dur': 1*10**7, 'am_589_power': 0.25, 
            'pulsed_initial_ion_dur': 25*10**3,
            'pulsed_shelf_dur': 200, 
            'am_589_shelf_power': 0.35,
            'pulsed_ionization_dur': 500, 'cobalt_638_power': 160, 
            'pulsed_reionization_dur': 100*10**3, 'cobalt_532_power': 8, 
            'magnet_angle': 0,
            "resonance_LOW": 2.8030,"rabi_LOW": 123.8, "uwave_power_LOW": 9.0,
            "resonance_HIGH": 2.9479,"rabi_HIGH": 130.1,"uwave_power_HIGH": 10.0}   
    nv_sig = ensemble

    apd_indices = [0]
    num_steps = 3
    num_reps = 2*10**2
    num_runs = 5
    state = States.LOW
    uwave_time_range = [0, 200]
    
    # Run rabi with SCC readout
    main(nv_sig, apd_indices, uwave_time_range, state,
         num_steps, num_reps, num_runs)
    
   
    
    # replotting data
    file = '2020_05_19-15_54_50-bachman-B5'
    data = tool_belt.get_raw_data('rabi_SCC/branch_Spin_to_charge/2020_05', file)
#   
    norm_avg_sig = data['norm_avg_sig']
    uwave_time_range = data['uwave_time_range']
    num_steps = data['num_steps']
    nv_sig = data['nv_sig']
    state = data['state']
    uwave_freq = nv_sig['resonance_{}'.format(state)]
    
#    
#    fit_func, popt = fit_data(uwave_time_range, num_steps, norm_avg_sig)
#    if (fit_func is not None) and (popt is not None):
#        create_fit_figure(uwave_time_range, uwave_freq, num_steps, norm_avg_sig,
#                  fit_func, popt)
    
                  
#    sig_counts = data['sig_counts']
#    ref_counts = data['ref_counts']
#    tau_index_master_list = data['tau_index_master_list'] 
#    sig_counts_sorted = []
#    ref_counts_sorted = []
#     
#    for i in range(len(sig_counts)):
#         zipped_list_sig = zip(tau_index_master_list[i], sig_counts[i])
#         zipped_list_ref = zip(tau_index_master_list[i], ref_counts[i])
#         
#         sorted_zipped_sig = sorted(zipped_list_sig)
#         sorted_zipped_ref = sorted(zipped_list_ref)
#         
#         sig_sorted = [element for _, element in sorted_zipped_sig]
#         ref_sorted = [element for _, element in sorted_zipped_ref]
#         
#         sig_counts_sorted.append(sig_sorted)
#         ref_counts_sorted.append(ref_sorted)
#         
#         fig, ax = plt.subplots(figsize=(8.5, 8.5))
#         ax.plot(sig_sorted, label = 'sig')
#         ax.plot(ref_sorted, label = 'ref')
#         ax.set_xlabel('Measurement index') 
#         ax.set_ylabel('Counts (arb. units)')
#         ax.legend()
#         ax.set_title('Run # {}'.format(i))

    