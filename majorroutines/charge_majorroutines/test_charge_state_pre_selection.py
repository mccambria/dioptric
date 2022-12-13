# -*- coding: utf-8 -*-
"""
Routine to test rabi_scc with and without an extra charge state readout after initialization. 
Still need to add analysis that goes and makes plots with the counts it gets back.
Those plots should compare the scc histograms with and without throwing away the experiments where the charge readout gave a count rate below the threhsold

Created on Sat Dec 10 11:49:23 2022

@author: cdfox
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



# %% Main


def main(nv_sig, state, num_reps, opti_nv_sig = None):
    
    # run the experiment both with and without charge state checking. 
    # We could do it all from the one with the checking, but it's good to compare to one that doesn't even do the charge state readouts
    
    ret_vals_nocheck = measure(nv_sig, state, num_reps, opti_nv_sig, check_charge_state=False)
    (pre_sig_counts_nocheck, 
     sig_counts_nocheck, 
     pre_ref_counts_nocheck, 
     ref_counts_nocheck) = ret_vals_nocheck

    ret_vals_check = measure(nv_sig, state, num_reps, opti_nv_sig, check_charge_state=True)
    (pre_sig_counts_check, 
     sig_counts_check, 
     pre_ref_counts_check, 
     ref_countsc_check) = ret_vals_check

    
def measure(nv_sig, state, num_reps, opti_nv_sig, check_charge_state):
    
    with labrad.connect() as cxn:
        ret_vals = measure_with_cxn(cxn, nv_sig, state, num_reps, opti_nv_sig, check_charge_state)
    
    return ret_vals        
    

def measure_with_cxn(cxn, nv_sig, state, num_reps, opti_nv_sig, check_charge_state):    

    counter_server = tool_belt.get_server_counter(cxn)
    pulsegen_server = tool_belt.get_server_pulse_gen(cxn)
    arbwavegen_server = tool_belt.get_server_arb_wave_gen(cxn)

    tool_belt.reset_cfm(cxn)
    kpl.init_kplotlib()

    # %% Get the starting time of the function, to be used to calculate run time

    startFunctionTime = time.time()
    start_timestamp = tool_belt.get_time_stamp()

    # Initial Calculation and setup
    
    tool_belt.set_filter(cxn, nv_sig, "charge_readout_laser")
    tool_belt.set_filter(cxn, nv_sig, "nv-_reionization_laser")
    tool_belt.set_filter(cxn, nv_sig, "nv0_ionization_laser")
        
    readout_time = nv_sig['charge_readout_dur']
    ionization_time = nv_sig['nv0_ionization_dur']
    reionization_time = nv_sig['nv-_reionization_dur']
    
    readout_power = tool_belt.set_laser_power(cxn, nv_sig, "charge_readout_laser")
    reion_power = tool_belt.set_laser_power(cxn, nv_sig, "nv-_reionization_laser")
    ion_power = tool_belt.set_laser_power(cxn, nv_sig, "nv0_ionization_laser")
    
    uwave_freq = nv_sig['resonance_{}'.format(state.name)]
    uwave_power = nv_sig['uwave_power_{}'.format(state.name)]
    rabi_period = float(nv_sig['rabi_{}'.format(state.name)])
    uwave_pi_pulse = tool_belt.get_pi_pulse_dur(rabi_period)
    
    green_laser_name = nv_sig['nv-_reionization_laser']
    red_laser_name = nv_sig['nv0_ionization_laser']
    yellow_laser_name = nv_sig['charge_readout_laser']
    sig_gen_cxn = tool_belt.get_server_sig_gen(cxn, state)   
    sig_gen_name = sig_gen_cxn.name
    
    num_reps = int(num_reps)
    opti_coords_list = []
    
    # Optimize
    opti_coords = optimize.main_with_cxn(cxn, nv_sig)
    opti_coords_list.append(opti_coords)
    
    # Turn on the microwaves
    sig_gen_cxn.set_freq(uwave_freq)
    sig_gen_cxn.set_amp(uwave_power)
    sig_gen_cxn.uwave_on()
    
    file_name = 'rabi_scc_check_charge_state.py'   
    
    if check_charge_state:
        check_charge_readout_time = readout_time
        check_charge_readout_power = readout_power
        
    elif not check_charge_state:
        check_charge_readout_time = readout_time
        check_charge_readout_power = 0
    
    seq_args = [check_charge_readout_time, readout_time, reionization_time, ionization_time, uwave_pi_pulse, 0, uwave_pi_pulse, 
                green_laser_name, yellow_laser_name, red_laser_name, sig_gen_name,
                reion_power, ion_power, 0, check_charge_readout_power, readout_power]
    seq_args_string = tool_belt.encode_seq_args(seq_args)
    ret_vals = pulsegen_server.stream_load(file_name, seq_args_string)
    print(seq_args)
    
    counter_server.start_tag_stream()
    pulsegen_server.stream_immediate(file_name, num_reps, seq_args_string)
    
    new_counts = counter_server.read_counter_separate_gates(1)
    sample_counts = new_counts[0]
    print('sample counts: ',sample_counts)

    pre_sig_counts = sample_counts[0::4]
    sig_counts = sample_counts[1::4]
    pre_ref_counts = sample_counts[2::4]
    ref_counts = sample_counts[3::4]

    counter_server.stop_tag_stream()
    
    # %% Clean up and save the data

    tool_belt.reset_cfm(cxn)

    endFunctionTime = time.time()

    timeElapsed = endFunctionTime - startFunctionTime

    timestamp = tool_belt.get_time_stamp()

    raw_data = {'timestamp': timestamp,
                'timeElapsed': timeElapsed,
                'timeElapsed-units': 's',
                'check_charge_state': check_charge_state,
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
                'ref_counts-units': 'counts',
                'pre_sig_counts': pre_sig_counts.astype(int).tolist(),
                'pre_sig_counts-units': 'counts',
                'pre_ref_counts': pre_ref_counts.astype(int).tolist(),
                'pre_ref_counts-units': 'counts'}

    nv_name = nv_sig["name"]
    file_path = tool_belt.get_file_path(__file__, timestamp, nv_name)
    tool_belt.save_raw_data(raw_data, file_path)
    
    return pre_sig_counts, sig_counts, pre_ref_counts, ref_counts
    
    
    

# %% Run the file


if __name__ == '__main__':

    path = 'pc_carr/branch_master/test_charge_state_pre_selection/2022_12'
    file = '2022_12_12-11_56_01-johnson-search'
    data = tool_belt.get_raw_data(file, path)
    
    nv_sig = data['nv_sig']
    check_charge_state = data['check_charge_state']
    print('check?: ',check_charge_state)
    
    sig_counts = data['sig_counts']
    ref_counts = data['ref_counts']
    pre_sig_counts = data['pre_sig_counts']
    pref_ref_counts = data['pre_ref_counts']
    
    snr = tool_belt.calc_snr(sig_counts, ref_counts)
    print(snr)
    
    
    
    
    
    
    
    
    
    
    
    
    
