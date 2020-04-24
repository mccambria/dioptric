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


def main(nv_sig, apd_indices, uwave_time_range, state,
         num_steps, num_reps, num_runs):

    with labrad.connect() as cxn:
        rabi_per, sig_counts, ref_counts = main_with_cxn(cxn, nv_sig, apd_indices, uwave_time_range, state,
                      num_steps, num_reps, num_runs)
        
        return rabi_per, sig_counts, ref_counts
def main_with_cxn(cxn, nv_sig, apd_indices, uwave_time_range, state,
                  num_steps, num_reps, num_runs):

    tool_belt.reset_cfm(cxn)

    # Initial Calculation and setup
    readout_time = nv_sig['pulsed_SCC_readout_dur']
    aom_ao_589_pwr = nv_sig['am_589_power']
    nd_filter = nv_sig['nd_filter']
    init_ion_time = nv_sig['pulsed_initial_ion_dur']
    ionization_time = nv_sig['pulsed_ionization_dur']
    reionization_time = nv_sig['pulsed_reionization_dur']
    shelf_time = nv_sig['pulsed_shelf_dur']
    shelf_power = nv_sig['am_589_shelf_power']
    
    uwave_freq = nv_sig['resonance_{}'.format(state.name)]
    uwave_power = nv_sig['uwave_power_{}'.format(state.name)]
    
    # set the nd_filter for yellow
    cxn.filter_slider_ell9k.set_filter(nd_filter)
    
    shared_params = tool_belt.get_shared_parameters_dict(cxn)

    #delay of aoms and laser
    laser_515_delay = shared_params['515_laser_delay']
    aom_589_delay = shared_params['589_aom_delay']
    laser_638_delay = shared_params['638_DM_laser_delay']
    rf_delay = shared_params['uwave_delay']
    
    wait_time = shared_params['post_polarization_wait_dur']

    num_reps = int(num_reps)
    # %% Set up our data lists and tau lists
    opti_coords_list = []
    
    min_uwave_time = uwave_time_range[0]
    max_uwave_time = uwave_time_range[1]    
    taus = numpy.linspace(min_uwave_time, max_uwave_time, num=num_steps)
     
    sig_counts = [[] for i in range(num_steps)]
    ref_counts = [[] for i in range(num_steps)]
    avg_sig_counts = []
    avg_ref_counts = []
    
    #%% Estimate the lenth of the sequance            
    file_name = 'SCC_optimize_pulses_w_uwaves.py'
    seq_args = [readout_time, init_ion_time, reionization_time, ionization_time, max_uwave_time,
        shelf_time , wait_time, max_uwave_time,  laser_515_delay, aom_589_delay, laser_638_delay, rf_delay,
        apd_indices[0], aom_ao_589_pwr, shelf_power, state.value]
    seq_args_string = tool_belt.encode_seq_args(seq_args)
    ret_vals = cxn.pulse_streamer.stream_load(file_name, seq_args_string)

    seq_time = ret_vals[0]

    seq_time_s = seq_time / (10**9)  # s
    expected_run_time = num_steps * num_reps * seq_time_s  #s
    expected_run_time_m = expected_run_time / 60 # m

    print(' \nExpected run time: {:.1f} minutes. '.format(expected_run_time_m))

    # Collect data

    shuffle(taus)
    for t in taus:
        # Optimize
        opti_coords = optimize.main_with_cxn(cxn, nv_sig, apd_indices, 532, disable=True)
        opti_coords_list.append(opti_coords)
        
        # Turn on the microwaves
        sig_gen_cxn = tool_belt.get_signal_generator_cxn(cxn, state)
        sig_gen_cxn.set_freq(uwave_freq)
        sig_gen_cxn.set_amp(uwave_power)
        sig_gen_cxn.uwave_on()
        
        # Load the APD
        cxn.apd_tagger.start_tag_stream(apd_indices)
        seq_args = [readout_time, init_ion_time, reionization_time, ionization_time, max_uwave_time,
            shelf_time , wait_time, max_uwave_time,  laser_515_delay, aom_589_delay, laser_638_delay, rf_delay,
            apd_indices[0], aom_ao_589_pwr, shelf_power, state.value]
        seq_args_string = tool_belt.encode_seq_args(seq_args)    
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
if __name__ == '__main__':
    sample_name = 'hopper'
    ensemble = { 'coords': [0.183, 0.043, 5.00],
            'name': '{}-ensemble'.format(sample_name),
            'expected_count_rate': 1000, 'nd_filter': 'nd_0',
            'pulsed_readout_dur': 300,
            'pulsed_SCC_readout_dur': 1*10**7, 'am_589_power': 0.2, 
            'pulsed_initial_ion_dur': 50*10**3,
            'pulsed_shelf_dur': 100, 'am_589_shelf_power': 0.2,
            'pulsed_ionization_dur': 450, 'cobalt_638_power': 160, 
            'pulsed_reionization_dur': 10*10**3, 'cobalt_532_power': 8, 
            'magnet_angle': 0,
            'resonance_LOW': 2.8059, 'rabi_LOW': 187.8, 'uwave_power_LOW': 9.0, 
            'resonance_HIGH': 2.9366, 'rabi_HIGH': 247.4, 'uwave_power_HIGH': 10.0}   
    nv_sig = ensemble

    apd_indices = [0]
    num_steps = 31
    num_reps = 10**3
    num_runs = 1
    state = States.LOW
    uwave_time_range = [0, 200]
    
    # Run rabi with SCC readout
    main(nv_sig, apd_indices, uwave_time_range, state,
         num_steps, num_reps, num_runs)
    