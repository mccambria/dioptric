# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 13:40:36 2020

This routine performs Rabi, after repeatedly performing a pi pulse (resonant 
with the target NVs) and ionization of non-resonant NVs. Then a short yellow 
readout is perfromed.

Must specify the two sig generators to use for the two different uwave pulses:
    -Shelf refers to the pusle used inthe repeated ionization sub sequence
    -Test refers to the uwaves being tested (toggles on/off in the sequence)

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

    fit_func = tool_belt.cosexp

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

    fit_fig, ax = plt.subplots(figsize=(8.5, 8.5))
    ax.plot(taus, norm_avg_sig,'bo',label='data')
    ax.plot(linspaceTau, fit_func(linspaceTau, *popt), 'r-', label='fit')
    ax.set_xlabel('Microwave duration (ns)')
    ax.set_ylabel('Contrast (arb. units)')
    ax.set_title('Rabi Oscillation Of NV Center Electron Spin')
    ax.legend()
    text_freq = 'Resonant frequency:' + '%.3f'%(uwave_freq) + 'GHz'
    
    text_popt = '\n'.join((r'$C + A_0 e^{-t/d} \mathrm{cos}(2 \pi \nu t + \phi)$',
                      r'$C = $' + '%.3f'%(popt[0]),
                      r'$A_0 = $' + '%.3f'%(popt[1]),
                      r'$\frac{1}{\nu} = $' + '%.1f'%(1/popt[2]) + ' ns',
                      r'$d = $' + '%i'%(popt[3]) + ' ' + r'$ ns$'))

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.55, 0.25, text_popt, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)
    ax.text(0.55, 0.3, text_freq, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)

    fit_fig.canvas.draw()
    fit_fig.set_tight_layout(True)
    fit_fig.canvas.flush_events()

    return fit_fig

# %% Main


def main(nv_sig, apd_indices, uwave_time_range, shelf_state, test_state,
         num_steps, num_reps, num_runs):

    with labrad.connect() as cxn:
        rabi_per, sig_counts, ref_counts = main_with_cxn(cxn, nv_sig, apd_indices, uwave_time_range, 
                     shelf_state, test_state,
                      num_steps, num_reps, num_runs)
        
        return rabi_per, sig_counts, ref_counts
def main_with_cxn(cxn, nv_sig, apd_indices, uwave_time_range, shelf_state, test_state,
                  num_steps, num_reps, num_runs):

    tool_belt.reset_cfm(cxn)

    # %% Get the starting time of the function, to be used to calculate run time

    startFunctionTime = time.time()
    start_timestamp = tool_belt.get_time_stamp()

    # %% Initial calculations and setup

    shelf_uwave_freq = nv_sig['resonance_{}'.format(shelf_state.name)]
    shelf_uwave_power = nv_sig['uwave_power_{}'.format(shelf_state.name)]
    target_pi_pulse = nv_sig['rabi_{}'.format(shelf_state.name)]
    
    test_uwave_freq = nv_sig['resonance_{}'.format(test_state.name)]
    test_uwave_power = nv_sig['uwave_power_{}'.format(test_state.name)]

    # parameters from nv_sig
    readout_time = nv_sig['pulsed_SCC_readout_dur']
    readout_power = nv_sig['am_589_power'] #0.9
    init_ion_time = nv_sig['pulsed_initial_ion_dur']
    reion_time = nv_sig['pulsed_reionization_dur']
    ion_time = nv_sig['pulsed_ionization_dur']
    shelf_time = nv_sig['pulsed_shelf_dur']
    yellow_pol_time = nv_sig['yellow_pol_dur']
    shelf_power = nv_sig['am_589_shelf_power']
    yellow_pol_pwr = nv_sig['am_589_pol_power']
    num_ionizations = nv_sig['ionization_rep']

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
    file_name = 'rabi_isolate_orientation.py'
            
    seq_args = [taus[0], readout_time, yellow_pol_time, shelf_time, init_ion_time, reion_time, ion_time, target_pi_pulse,
         wait_time, num_ionizations, laser_515_delay, aom_589_delay, laser_638_delay, rf_delay,
        apd_indices[0], readout_power, yellow_pol_pwr, shelf_power, shelf_state.value, test_state.value]
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

        # Apply the microwaves for the shelf pulse (repeated one)
        shelf_sig_gen_cxn = tool_belt.get_signal_generator_cxn(cxn, shelf_state)
        shelf_sig_gen_cxn.set_freq(shelf_uwave_freq)
        shelf_sig_gen_cxn.set_amp(shelf_uwave_power)
        shelf_sig_gen_cxn.uwave_on()
        # Apply the microwaves for the testpulse (toggled on/off for sig/ref)
        test_sig_gen_cxn = tool_belt.get_signal_generator_cxn(cxn, test_state)
        test_sig_gen_cxn.set_freq(test_uwave_freq)
        test_sig_gen_cxn.set_amp(test_uwave_power)
        test_sig_gen_cxn.uwave_on()
        
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
            seq_args = [taus[tau_ind], readout_time, yellow_pol_time, shelf_time, init_ion_time, reion_time, ion_time, target_pi_pulse,
                        wait_time, num_ionizations, laser_515_delay, aom_589_delay, laser_638_delay, rf_delay,
                        apd_indices[0], readout_power, yellow_pol_pwr, shelf_power, shelf_state.value, test_state.value]
            seq_args_string = tool_belt.encode_seq_args(seq_args)
            cxn.pulse_streamer.stream_immediate(file_name, num_reps,
                                                seq_args_string)

            # Get the counts
            new_counts = cxn.apd_tagger.read_counter_separate_gates(1)

            sample_counts = new_counts[0]

            # signal counts are even - get every second element starting from 0
            sig_gate_counts = sample_counts[0::2]
            sig_counts[run_ind, tau_ind] = sum(sig_gate_counts)

            # ref counts are odd - sample_counts every second element starting from 1
            ref_gate_counts = sample_counts[1::2]
            ref_counts[run_ind, tau_ind] = sum(ref_gate_counts)

        cxn.apd_tagger.stop_tag_stream()

        # %% Save the data we have incrementally for long measurements

        raw_data = {'start_timestamp': start_timestamp,
                    'nv_sig': nv_sig,
                    'nv_sig-units': tool_belt.get_nv_sig_units(),
                    'shelf_power': shelf_power,
                    'shelf_power-units': 'mW',
                    'uwave_time_range': uwave_time_range,
                    'uwave_time_range-units': 'ns',
                    'shelf state': shelf_state.name,
                    'test state': test_state.name,
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
    ax.set_title('Rabi measurement after repeated ionizations, yellow readout')
    ax.set_xlabel('Microwave duration (ns)')
    ax.set_ylabel('Contrast (arb. units)')

    raw_fig.canvas.draw()
    raw_fig.set_tight_layout(True)
    raw_fig.canvas.flush_events()

    # %% Plot the data itself and the fitted curve

    fit_fig = None
    if (fit_func is not None) and (popt is not None):
        fit_fig = create_fit_figure(uwave_time_range, test_uwave_freq, num_steps,
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
                'uwave_time_range': uwave_time_range,
                'uwave_time_range-units': 'ns',
                'shelf state': shelf_state.name,
                'test state': test_state.name,
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
    sample_name = 'hopper'
    ensemble = { 'coords': [0.0, 0.0, 5.00],
            'name': '{}-ensemble'.format(sample_name),
            'expected_count_rate': 1000, 'nd_filter': 'nd_0',
            'pulsed_readout_dur': 300,
            'pulsed_SCC_readout_dur': 1*10**7, 'am_589_power': 0.2, 
            'yellow_pol_dur': 2*10**3, 'am_589_pol_power': 0.20,
            'pulsed_initial_ion_dur': 50*10**3,
            'pulsed_shelf_dur': 100, 'am_589_shelf_power': 0.20,
            'pulsed_ionization_dur': 450, 'cobalt_638_power': 160, 
            'pulsed_reionization_dur': 10*10**3, 'cobalt_532_power': 8,
            'ionization_rep': 7,
            'magnet_angle': 0,
            'resonance_LOW': 2.8059, 'rabi_LOW': 164.2, 'uwave_power_LOW': 9.0, 
            'resonance_HIGH': 2.8059, 'rabi_HIGH': 247.4, 'uwave_power_HIGH': 10.0}    
    nv_sig = ensemble

    apd_indices = [0]
    num_steps = 51
    num_reps = int(10**3)
    num_runs = 1
    shelf_state = States.LOW
    test_state = States.HIGH
    uwave_time_range = [0, 200]
    
    # Run rabi with SCC readout
    main(nv_sig, apd_indices, uwave_time_range, shelf_state, test_state,
         num_steps, num_reps, num_runs)
    