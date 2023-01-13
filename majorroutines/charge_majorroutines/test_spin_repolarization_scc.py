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
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from random import shuffle
from scipy.optimize import curve_fit
import labrad
import majorroutines.optimize as optimize
from majorroutines.rabi import fit_data, create_fit_figure, create_raw_data_figure, simulate


# %% Main


def main(nv_sig, state, second_init_laser_key, second_init_power, 
         num_reps,num_runs,min_wait_time,max_wait_time,num_steps,threshold,do_ion_pulse,do_pi_pulse):
    
    for ion in do_ion_pulse:
        for pi in do_pi_pulse:

            with labrad.connect() as cxn:
                main_with_cxn(cxn, nv_sig, state,second_init_laser_key, second_init_power, 
                              num_reps,num_runs,min_wait_time,max_wait_time,num_steps,threshold,ion,pi)

        
def main_with_cxn(cxn, nv_sig, state, second_init_laser_key, 
                   second_init_power, num_reps,num_runs,
                  min_wait_time,max_wait_time,num_steps,threshold,do_ion_pulse,do_pi_pulse):
    
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

    readout_laser_key = nv_sig['charge_readout_laser']
    readout_time = nv_sig['charge_readout_dur']
    readout_power = tool_belt.set_laser_power(cxn, nv_sig, "charge_readout_laser")
    
    green_init_laser_key = nv_sig['spin_laser']
    green_init_time = nv_sig['spin_pol_dur']
    green_init_power = 1# tool_belt.set_laser_power(cxn, nv_sig, "spin_laser")
    
    ion_laser_key = nv_sig['nv0_ionization_laser']
    ion_time = nv_sig['nv0_ionization_dur']
    
    if do_ion_pulse:
        ion_power = 1
    else:
        ion_power = 0
    
    if do_pi_pulse:
        pi_pulse = tool_belt.get_pi_pulse_dur(nv_sig['rabi_{}'.format(state.name)])
    else:
        pi_pulse = 0
        
    sig_gen_cxn = tool_belt.get_server_sig_gen(cxn, state) 
    sig_gen_name = sig_gen_cxn.name
    
    # Analyze the sequence
    
    file_name = 'test_spin_repolarization.py'

    wait_times = np.linspace(min_wait_time,max_wait_time,num_steps)
    wait_times_ind_list = list(range(0, num_steps))
    shuffle(wait_times_ind_list)

    counts = np.empty([num_steps,int(num_runs*num_reps)])
    
    # %% Make some lists and variables to save at the end
    
    opti_coords_list = []
    
    # %% Collect the data
    tool_belt.init_safe_stop()
    
    for i in range(num_runs):
        if tool_belt.safe_stop():
            break

        for ind in wait_times_ind_list:
            if tool_belt.safe_stop():
                break
            
            wait_time = wait_times[ind]
            
            seq_args = [green_init_laser_key, second_init_laser_key, ion_laser_key, readout_laser_key,
                        green_init_time, wait_time, ion_time, readout_time,
                        green_init_power, second_init_power, ion_power, readout_power,
                        sig_gen_name, pi_pulse]
            
            print(seq_args)
            print('run: {} of {}'.format(i+1,num_runs))
            
            seq_args_string = tool_belt.encode_seq_args(seq_args)
            pulsegen_server.stream_load(file_name, seq_args_string)
        
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
            # print(num_reps)
            pulsegen_server.stream_immediate(file_name, num_reps,seq_args_string)
        
            # Get the counts
            new_counts = counter_server.read_counter_separate_gates(1)
        
            sample_counts = new_counts[0]
            cur_counts = sample_counts
            
            steps_start = int(i*num_reps)
            steps_end = int((i+1)*num_reps)
            counts[ind][steps_start : steps_end] = cur_counts
            
            counter_server.stop_tag_stream()
            
        timestamp = tool_belt.get_time_stamp()
        raw_data = {'start_timestamp': start_timestamp,
                    'nv_sig': nv_sig,
                    'uwave_freq': uwave_freq,
                    'uwave_freq-units': 'GHz',
                    'uwave_power': uwave_power,
                    'uwave_power-units': 'dBm',
                    'state': state.name,
                    'num_reps': num_reps,
                    'num_runs': num_runs,
                    'current_run': i,
                    'wait_times': wait_times.astype(int).tolist(),
                    'second_pulse_laser': second_init_laser_key,
                    'second_pulse_power': second_init_power,
                    'do_pi_pulse': do_pi_pulse,
                    'do_ion_pulse': do_ion_pulse,
                    'counts': counts.astype(int).tolist()}

        nv_name = nv_sig["name"]
        file_path = tool_belt.get_file_path(__file__, start_timestamp, nv_sig['name'], 'incremental')
        tool_belt.save_raw_data(raw_data, file_path)
    
    
    # %% Clean up and save the data

    tool_belt.reset_cfm(cxn)

    endFunctionTime = time.time()

    timeElapsed = endFunctionTime - startFunctionTime

    timestamp = tool_belt.get_time_stamp()
    
    states=np.copy(counts)
    states[np.where(counts < threshold)] = 0
    states[np.where(counts >= threshold)] = 1
    
    avg_states = np.average(states,axis=1)
    ste_states = np.std(states,axis=1)/np.sqrt(num_reps*num_runs)

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
                'num_runs': num_runs,
                'threshold': threshold,
                'wait_times': wait_times.astype(int).tolist(),
                'second_pulse_laser': second_init_laser_key,
                'do_pi_pulse': do_pi_pulse,
                'do_ion_pulse': do_ion_pulse,
                'second_pulse_power': second_init_power,
                'counts': counts.astype(int).tolist(),
                'states': states.astype(int).tolist(),
                'avg_states':avg_states.tolist(),
                'ste_states':ste_states.tolist()}

    nv_name = nv_sig["name"]
    if do_ion_pulse:
        ion_text = 'ion'
    else:
        ion_text = 'no-ion'
    if do_pi_pulse:
        pi_text = 'pi'
    else:
        pi_text = 'no-pi'
        
    added_text = '_'+second_init_laser_key+'_'+str(round(second_init_power*100))+'_'+ion_text+'_'+pi_text
    file_path = tool_belt.get_file_path(__file__, timestamp, nv_name+added_text)
    tool_belt.save_raw_data(raw_data, file_path)


# %%
if __name__ == '__main__':
    
    def extract_files(data_folder):
        fp='pc_Carr/branch_master/test_spin_repolarization_scc_v3/'
        filelist = os.listdir('E:/Shared drives/Kolkowitz Lab Group/nvdata/pc_Carr/branch_master/test_spin_repolarization_scc_v3/'+
                       data_folder)
        for file in filelist:
            if '_no-ion_no-pi' in file:
                noion_nopi = tool_belt.get_raw_data(os.path.splitext(file)[0],path_from_nvdata=fp+data_folder)
                
            elif '_no-ion_pi' in file:
                noion_pi = tool_belt.get_raw_data(os.path.splitext(file)[0],path_from_nvdata=fp+data_folder)

            elif '_ion_no-pi' in file:
                ion_nopi = tool_belt.get_raw_data(os.path.splitext(file)[0],path_from_nvdata=fp+data_folder)
            elif '_ion_pi' in file:
                ion_pi = tool_belt.get_raw_data(os.path.splitext(file)[0],path_from_nvdata=fp+data_folder)
            else:
                a = 2
        
            
        return noion_nopi, noion_pi, ion_nopi, ion_pi
    
    kpl.init_kplotlib()
    # replotting data
    folders_dict = {
        # 'no_yellow': '2022_12/no_yellow',
                    'new_no_yellow': '2022_12/new_no_yellow',
                    # 'yellow': '2022_12/yellow',
                    'yellow': '2023_01/yellow_1_9',
                    'no_yellow_ms1': '2023_01/no_yellow_ms1_1_10',
                    'yellow_ms1': '2023_01/yellow_ms1_1_10',
                    'no_yellow': '2023_01/no_yellow_1_9',
                    'green': '2022_12/green',
                    'lower_yellow': '2022_12/lower_yellow',
                    'red': '2022_12/red',
                    }
    
    def calc_rho_NVm_ms0_before_2(rho_NVm_after,rho_NVm_before,I0,I1):
        
        ret = ( (1/(I0-I1)) * (rho_NVm_before*(1-I1) - rho_NVm_after) )
        
        return ret
    
    # def calc_rho_NVm_ms_pm1_before_2(rho_NVm_after,rho_NVm_before,I0,I1):    
    #     ret = ( (1/(I1-I0)) * (rho_NVm_before*(1-I0) - rho_NVm_after) )
    #     return ret
    
    
    def plot_spin_probs(data_name):
        data_folder = folders_dict[data_name]
        data_noion_nopi, data_noion_pi, data_ion_nopi, data_ion_pi = extract_files(data_folder)
        
        avg_states_ion_pi = np.array(data_ion_pi['avg_states'])
        avg_states_noion_nopi = np.array(data_noion_nopi['avg_states'])
        avg_states_ion_nopi = np.array(data_ion_nopi['avg_states'])
        avg_states_noion_pi = np.array(data_noion_pi['avg_states'])
        
        wait_times = np.array(data_noion_nopi['wait_times'])/1e6
        laser_power = data_noion_nopi['second_pulse_power']
        
        I0 = 1*((avg_states_noion_nopi[0] - avg_states_ion_nopi[0])/avg_states_noion_nopi[0])
        I1 = 1*((avg_states_noion_pi[0] - avg_states_ion_pi[0])/avg_states_noion_pi[0])
        print(I0,I1)
        nvm_ms0_prob = calc_rho_NVm_ms0_before_2(avg_states_ion_nopi,avg_states_noion_nopi,I0,I1)/avg_states_noion_nopi
        nvm_msm1_prob = calc_rho_NVm_ms0_before_2(avg_states_ion_pi,avg_states_noion_pi,I0,I1)/avg_states_noion_pi
        
        if True:        
            fig,( ax0,ax1) = plt.subplots(1, 2, figsize=kpl.double_figsize)
            kpl.plot_points(ax0, wait_times,nvm_ms0_prob)
            kpl.plot_points(ax1, wait_times,nvm_msm1_prob)
            ax0.set_xlabel('Time [ms]')
            ax1.set_xlabel('Time [ms]')
            ax0.set_ylabel(r'$P(NV^-_0|NV^-)$')
            ax1.set_ylabel(r'$P(NV^-_{-1}|NV^-)$')
            fig.suptitle('laser power = '+str(laser_power))
        
        if False:
            
            fig,( ax0,ax1) = plt.subplots(1, 2, figsize=kpl.double_figsize)
            kpl.plot_points(ax1, wait_times,avg_states_ion_pi)
            kpl.plot_points(ax0, wait_times,avg_states_noion_nopi)
            ax0.set_xlabel('Time [ms]')
            ax1.set_xlabel('Time [ms]')
            ax1.set_ylabel(r'$\rho_(NV^-,a)$')
            ax0.set_ylabel(r'$\rho_(NV^-,b)$')
            fig.suptitle('laser power = '+str(laser_power))
        
        if False:
            counts_ion_pi = np.array(data_ion_pi['counts'])
            counts_noion_nopi = np.array(data_noion_nopi['counts'])
            counts_ion_nopi = np.array(data_ion_nopi['counts'])
            counts_noion_pi = np.array(data_noion_pi['counts'])
            threshold=4
            states_ion_pi = np.copy(counts_ion_pi)
            states_ion_pi[np.where(counts_ion_pi < threshold)] = 0
            states_ion_pi[np.where(counts_ion_pi >= threshold)] = 1
            avg_states_ion_pi = np.average(states_ion_pi,1)

            states_noion_nopi = np.copy(counts_noion_nopi)
            states_noion_nopi[np.where(counts_noion_nopi < threshold)] = 0
            states_noion_nopi[np.where(counts_noion_nopi >= threshold)] = 1
            avg_states_noion_nopi = np.average(states_noion_nopi,1)

            states_ion_nopi = np.copy(counts_ion_nopi)
            states_ion_nopi[np.where(counts_ion_nopi < threshold)] = 0
            states_ion_nopi[np.where(counts_ion_nopi >= threshold)] = 1
            avg_states_ion_nopi = np.average(states_ion_nopi,1)

            states_noion_pi = np.copy(counts_noion_pi)
            states_noion_pi[np.where(counts_noion_pi < threshold)] = 0
            states_noion_pi[np.where(counts_noion_pi >= threshold)] = 1
            avg_states_noion_pi = np.average(states_noion_pi,1)
            
            fig,( ax0,ax1) = plt.subplots(1, 2, figsize=kpl.double_figsize)
            kpl.histogram(ax0, counts_noion_nopi[0])
            kpl.histogram(ax1, counts_noion_nopi[2])
            
        # print(wait_times)
    # plot_spin_probs('no_yellow')
    # plot_spin_probs('new_no_yellow')
    # plot_spin_probs('yellow')
    # plot_spin_probs('yellow_ms1')
    plot_spin_probs('no_yellow_ms1')
    # plot_spin_probs('lower_yellow')
    # plot_spin_probs('green')
    
        
