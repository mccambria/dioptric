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
import utils.kplotlib as kpl
from utils.kplotlib import KplColors
import utils.positioning as positioning
import numpy
import os
import time
import matplotlib.pyplot as plt
from random import shuffle
from scipy.optimize import curve_fit
import labrad
import majorroutines.optimize as optimize
from majorroutines.rabi import fit_data, create_fit_figure, create_raw_data_figure, simulate


# %% Main


def main(nv_sig, state, pre_init_laser_key, pre_init_time, pre_init_power, total_wait_time, num_reps):

    with labrad.connect() as cxn:
        sig_counts, ref_counts = main_with_cxn(cxn, nv_sig, state, 
                                               pre_init_laser_key, pre_init_time, pre_init_power, total_wait_time, num_reps)

    return sig_counts, ref_counts
        
        
def main_with_cxn(cxn, nv_sig, state, pre_init_laser_key, pre_init_time, pre_init_power, total_wait_time, num_reps):
    
    counter_server = tool_belt.get_server_counter(cxn)
    pulsegen_server = tool_belt.get_server_pulse_gen(cxn)
    arbwavegen_server = tool_belt.get_server_arb_wave_gen(cxn)

    tool_belt.reset_cfm(cxn)
    kpl.init_kplotlib()


    # %% Get the starting time of the function, to be used to calculate run time

    startFunctionTime = time.time()
    start_timestamp = tool_belt.get_time_stamp()

    # %% Initial calculations and setup

    uwave_freq = nv_sig['resonance_{}'.format(state.name)]
    uwave_power = nv_sig['uwave_power_{}'.format(state.name)]
    pi_pulse = tool_belt.get_pi_pulse_dur(nv_sig['rabi_{}'.format(state.name)])

    norm_style = nv_sig["norm_style"]
    
    readout_laser_key = ['charge_readout_laser']
    readout_time = nv_sig['charge_readout_dur']
    readout_power = tool_belt.set_laser_power(cxn, nv_sig, "charge_readout_laser")
    
    #inputed to function:
    #pre_init_laser_key 
    #pre_init_time 
    #pre_init_power
    #total_wait_time
    
    second_init_laser_key = nv_sig['spin_laser']
    second_init_time = nv_sig['spin_pol_dur']
    second_init_power = tool_belt.set_laser_power(cxn, nv_sig, "spin_laser")
    
    ion_laser_key = nv_sig['nv0_ionization_laser']
    ion_time = nv_sig['nv0_ionization_dur']
    ion_power = tool_belt.set_laser_power(cxn, nv_sig, "nv0_ionization_laser")
        
    sig_gen_cxn = tool_belt.get_server_sig_gen(cxn, state) 
    sig_gen_name = sig_gen_cxn.name
    
    # Analyze the sequence
    
    file_name = 'test_spin_repolarization.py'

    seq_args = [pre_init_laser_key, second_init_laser_key, ion_laser_key, readout_laser_key,
                pre_init_time, second_init_time, ion_time, readout_time,
                pre_init_power, second_init_power, ion_power, readout_power,
                sig_gen_name, pi_pulse, total_wait_time]
    print(seq_args)
    
    seq_args_string = tool_belt.encode_seq_args(seq_args)
    pulsegen_server.stream_load(file_name, seq_args_string)

    # Set up our data structure, an array of NaNs that we'll fill
    #
    sig_counts = numpy.empty([num_reps])
    sig_counts[:] = numpy.nan
    ref_counts = numpy.copy(sig_counts)

    # %% Make some lists and variables to save at the end

    opti_coords_list = []
    
    # %% Collect the data

    # Optimize
    opti_coords = optimize.main_with_cxn(cxn, nv_sig)
    opti_coords_list.append(opti_coords)

    # Apply the microwaves
    sig_gen_cxn.set_freq(uwave_freq)
    sig_gen_cxn.set_amp(uwave_power)
    sig_gen_cxn.uwave_on()

    # Load the APD
    counter_server.start_tag_stream()

    # Load the sequence
    pulsegen_server.stream_load(file_name, seq_args_string)

    # Stream the sequence
    # Clear the tagger buffer of any excess counts
    counter_server.clear_buffer()
    pulsegen_server.stream_immediate(file_name, num_reps,seq_args_string)

    # Get the counts
    new_counts = counter_server.read_counter_separate_gates(1)

    sample_counts = new_counts[0]
    sig_counts = sample_counts[0::2]
    ref_counts = sample_counts[1::2]
        
    counter_server.stop_tag_stream()
    
    
    # %% Clean up and save the data

    tool_belt.reset_cfm(cxn)

    endFunctionTime = time.time()

    timeElapsed = endFunctionTime - startFunctionTime

    timestamp = tool_belt.get_time_stamp()

    raw_data = {'timestamp': timestamp,
                'timeElapsed': timeElapsed,
                'timeElapsed-units': 's',
                'nv_sig': nv_sig,
                'uwave_freq': uwave_freq,
                'uwave_freq-units': 'GHz',
                'uwave_power': uwave_power,
                'uwave_power-units': 'dBm',
                'state': state.name,
                'num_reps': num_reps,
                'sig_counts': sig_counts.astype(int).tolist(),
                'sig_counts-units': 'counts',
                'ref_counts': ref_counts.astype(int).tolist(),
                'ref_counts-units': 'counts'}

    nv_name = nv_sig["name"]
    file_path = tool_belt.get_file_path(__file__, timestamp, nv_name)
    tool_belt.save_raw_data(raw_data, file_path)
    
    return sig_counts, ref_counts
    
    
# %%
if __name__ == '__main__':
    import numpy as np
    
    # replotting data
    file = '2022_12_12-19_45_53-johnson-search'
    data = tool_belt.get_raw_data(file)
    
    
    # num_reps = data['num_reps']
    # uwave_time_range = data['uwave_time_range']
    # num_steps = data['num_steps']
    # nv_sig = data['nv_sig']
    # norm_style = tool_belt.NormStyle.SINGLE_VALUED
    # state = data['state']
    # uwave_freq = nv_sig['resonance_{}'.format(state)]
    # readout_time = nv_sig['charge_readout_dur']
    
    
    
#     kpl.init_kplotlib()
#     ret_vals = tool_belt.process_counts(sig_counts, ref_counts, num_reps, readout_time, norm_style)
#     (
#         sig_counts_avg_kcps,
#         ref_counts_avg_kcps,
#         norm_avg_sig,
#         norm_avg_sig_ste,
#     ) = ret_vals
# #    
#     fit_func = tool_belt.inverted_cosexp
#     fit_fig, ax, fit_func, popt, pcov = create_fit_figure(
#         uwave_time_range, num_steps, uwave_freq, norm_avg_sig, norm_avg_sig_ste,
#         fit_func 
#     )
    
                  