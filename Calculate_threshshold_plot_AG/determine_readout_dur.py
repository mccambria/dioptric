# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 14:09:39 2021

@author: samli
"""

# import labrad
import scipy.stats
import scipy.special
import numpy
import math  
import copy
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import random
import photonstatistics as model
import labrad

import utils.tool_belt as tool_belt
import majorroutines.optimize as optimize

#%% import your data in the data_file 
import NV_data as data
#%% functions 

#fit the NV- and NV0 counts to the rate model 
def get_photon_dis_curve_fit_plot(readout_time,NV0,NVm, do_plot = True): # NV0, NVm in array
    trail_nv0_30ms = numpy.array(NV0)
    trail_nvm_30ms = numpy.array(NVm)
    tR = readout_time 
    combined_count = trail_nvm_30ms.tolist() + trail_nv0_30ms.tolist()
    random.shuffle(combined_count)
    # fit = [g0,g1,y1,y0]
    
    # if it is the SiV sample, use this guess 
    # guess = [ 0.01 ,0.02, 0.57, 0.08] 
    
    # if it is the E6 sample, use this guess 
    guess = [ 10*10**-4,100*10**-4, 1000*10**-4, 500*10**-4]
    # fit gives g0, g1, y1, y0
    fit,dev = model.get_curve_fit(tR,0,combined_count,guess)
#    print(fit,np.diag(dev))
    
    if do_plot:
        u_value0, freq0 = model.get_Probability_distribution(trail_nv0_30ms.tolist())
        u_valuem, freqm = model.get_Probability_distribution(trail_nvm_30ms.tolist())
        u_value2, freq2 = model.get_Probability_distribution(combined_count)
        curve = model.get_photon_distribution_curve(tR,u_value2, fit[0] ,fit[1], fit[2] ,fit[3])
        
        A1, A1pcov = model.get_curve_fit_to_weight(tR,0,trail_nv0_30ms.tolist(),[0.5],fit)
        A2, A2pcov = model.get_curve_fit_to_weight(tR,0,trail_nvm_30ms.tolist(),[0.5],fit)
    
        nv0_curve = model.get_photon_distribution_curve_weight(u_value0,tR, fit[0] ,fit[1], fit[2] ,fit[3],A1[0])    
        nvm_curve = model.get_photon_distribution_curve_weight(u_valuem,tR, fit[0] ,fit[1], fit[2] ,fit[3],A2[0])
        fig4, ax = plt.subplots()
        ax.plot(u_value0,0.5*np.array(freq0),"-ro")
        ax.plot(u_valuem,0.5*np.array(freqm),"-go")
        ax.plot(u_value2,freq2,"-bo")
        ax.plot(u_value2,curve)
        ax.plot(u_valuem,0.5*np.array(nvm_curve),"green")
        ax.plot(u_value0,0.5*np.array(nv0_curve),"red")
        textstr = '\n'.join((
        r'$g_0(s^{-1}) =%.2f$'% (fit[0]*10**3, ),
        r'$g_1(s^{-1})  =%.2f$'% (fit[1]*10**3, ),
        r'$y_0 =%.2f$'% (fit[3], ),
        r'$y_1 =%.2f$'% (fit[2], )))
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.6, 0.95, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)
        plt.xlabel("Number of counts")
        plt.ylabel("Probability Density")
        plt.show()
    return fit 

# calculate the threshold of given NV0 and NV- data 
# tR in units of ms
def calculate_threshold_plot(readout_time,nv0_array,nvm_array, nd_filter, aom_power):
    tR = readout_time
    fit_rate = get_photon_dis_curve_fit_plot(readout_time,nv0_array,nvm_array)
    x_data = np.linspace(0,500,501)
    thresh_para = model.calculate_threshold(tR,x_data,fit_rate )
    print(thresh_para)
    
    x_data = np.linspace(0,60,61)
    fig3,ax = plt.subplots()
    ax.plot(x_data,model.get_PhotonNV0_list(x_data,tR,fit_rate,0.5),"-o")
    ax.plot(x_data,model.get_PhotonNVm_list(x_data,tR,fit_rate,0.5),"-o")
    plt.axvline(x=thresh_para[0],color = "red")
    textstr = '\n'.join((
        r'$\mu_1=%.2f$' % (fit_rate[3]*tR, ),
        r'$\mu_2=%.2f$'% (fit_rate[2]*tR, ),
        r'$fidelity =%.2f$'% (thresh_para[1] ),
        r'$threshold = %.1f$'% (thresh_para[0], )))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.65, 0.95, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)
    plt.xlabel("Number of counts")
    plt.ylabel("Probability Density")
    plt.title('{} ms readout, {}, {} V'.format(readout_time, nd_filter, aom_power))
    
    fidelity = thresh_para[1]
    threshold =thresh_para[0]
    mu_1 = fit_rate[3]*tR
    mu_2 = fit_rate[2]*tR
    return threshold, fidelity, mu_1, mu_2, fig3
    
# %%
# Apply a gren or red pulse, then measure the counts under yellow illumination. 
# Repeat num_reps number of times and returns the list of counts after red illumination, then green illumination
# Use with DM on red and green
def measure(nv_sig, apd_indices, num_reps):

    with labrad.connect() as cxn:
        sig_counts, ref_counts = measure_with_cxn(cxn, nv_sig, apd_indices, num_reps)
        
    return sig_counts, ref_counts
def measure_with_cxn(cxn, nv_sig, apd_indices, num_reps):

    tool_belt.reset_cfm(cxn)

    # Initial Calculation and setup
    
    tool_belt.set_filter(cxn, nv_sig, 'charge_readout_laser')
    tool_belt.set_filter(cxn, nv_sig, 'nv-_prep_laser')
    
    readout_laser_power = tool_belt.set_laser_power(cxn, nv_sig, 'charge_readout_laser')
    nvm_laser_power = tool_belt.set_laser_power(cxn, nv_sig, "nv-_prep_laser")
    nv0_laser_power = tool_belt.set_laser_power(cxn,nv_sig,"nv0_prep_laser")
    
    
    readout_pulse_time = nv_sig['charge_readout_dur']
    
    reionization_time = nv_sig['nv-_prep_laser_dur']
    ionization_time = nv_sig['nv0_prep_laser_dur']
      

    # Set up our data lists
    opti_coords_list = []
    
    # Optimize
    opti_coords = optimize.main_with_cxn(cxn, nv_sig, apd_indices)
    opti_coords_list.append(opti_coords)

    # Pulse sequence to do a single pulse followed by readout           
    seq_file = 'simple_readout_two_pulse.py'
        
    ################## Load the measuremnt with green laser ##################
      
    seq_args = [reionization_time, readout_pulse_time, nv_sig["nv-_prep_laser"], 
                nv_sig["charge_readout_laser"], nv0_laser_power, 
                readout_laser_power, apd_indices[0]]
    seq_args_string = tool_belt.encode_seq_args(seq_args)
    cxn.pulse_streamer.stream_load(seq_file, seq_args_string)

    # Load the APD
    cxn.apd_tagger.start_tag_stream(apd_indices)
    # Clear the buffer
    cxn.apd_tagger.clear_buffer()
    # Run the sequence
    cxn.pulse_streamer.stream_immediate(seq_file, num_reps, seq_args_string)

    nvm = cxn.apd_tagger.read_counter_simple(num_reps)
    
    ################## Load the measuremnt with red laser ##################
    seq_args = [ionization_time, readout_pulse_time, nv_sig["nv0_prep_laser"], 
                nv_sig["charge_readout_laser"], nvm_laser_power, 
                readout_laser_power, apd_indices[0]]
    seq_args_string = tool_belt.encode_seq_args(seq_args)
    cxn.pulse_streamer.stream_load(seq_file, seq_args_string)

    # Load the APD
    cxn.apd_tagger.start_tag_stream(apd_indices)
    # Clear the buffer
    cxn.apd_tagger.clear_buffer()
    # Run the sequence
    cxn.pulse_streamer.stream_immediate(seq_file, num_reps, seq_args_string)

    nv0 = cxn.apd_tagger.read_counter_simple(num_reps)
    
    
    tool_belt.reset_cfm(cxn)
    
    return nv0, nvm

def determine_readout_dur(nv_sig, readout_times = None, readout_yellow_powers = None,
                          nd_filter = 'nd_0.5'):
    num_reps = 500
    apd_indices =[0]
    
    if not readout_times:
        readout_times = [50*10**6, 100*10**6, 250*10**6]
    if not readout_yellow_powers:
        readout_yellow_powers = [0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6]
        
    # first index will be power, second will be time, third wil be individual points
    nv0_master = []
    nvm_master =[]
    
    tool_belt.init_safe_stop()
    
    for p in readout_yellow_powers:
        nv0_power =[]
        nvm_power =[]
        for t in readout_times:
            # Break out of the while if the user says stop
            if tool_belt.safe_stop():
                break
            
            nv_sig_copy = copy.deepcopy(nv_sig)
            nv_sig_copy['charge_readout_laser_filter'] = nd_filter
            nv_sig_copy['charge_readout_dur'] = t
            nv_sig_copy['charge_readout_laser_power'] = p
            
            nv0, nvm = measure(nv_sig_copy, apd_indices, num_reps)
            nv0_power.append(nv0)
            nvm_power.append(nvm)
            
            print('data collected!')
            print('{} ms readout, {}, {} V'.format(t/10**6, nd_filter, p))
            threshold, fidelity, mu_1, mu_2, fig = calculate_threshold_plot(t/10**6, nv0, nvm, nd_filter, p)
            
            timestamp = tool_belt.get_time_stamp()
            raw_data = {'timestamp': timestamp,
                    'nv_sig': nv_sig_copy,
                    'nv_sig-units': tool_belt.get_nv_sig_units(),
                    'num_runs':num_reps,
                    'nv0': nv0.tolist(),
                    'nv0_list-units': 'counts',
                    'nvm': nvm.tolist(),
                    'nvmt-units': 'counts',
                    'mu_1': mu_1,
                    'mu_2': mu_2,
                    'fidelity': fidelity,
                    'threshold': threshold
                    }
            
            file_path = tool_belt.get_file_path(__file__, timestamp, nv_sig['name'])
            tool_belt.save_raw_data(raw_data, file_path)
            tool_belt.save_figure(fig, file_path)
    
            
        nv0_master.append(nv0_power)
        nvm_master.append(nvm_power)
    
            
    
        

    
    return
#%%     
if __name__ == '__main__':
    # load the data here 
    sample_name = 'johnson'    
    
    green_laser = 'laserglow_532'
    yellow_laser = 'laserglow_589'
    red_laser = 'cobolt_638'
    nd_green = 'nd_0.5'
    
    nv_sig = { 'coords': [0.056, -0.098, 5.0],
            'name': '{}-nv1_2021_07_27'.format(sample_name),
            'disable_opt': False, 'expected_count_rate': 42,
            'imaging_laser': green_laser, 'imaging_laser_filter': nd_green, 'imaging_readout_dur': 1E7,
            # 'initialize_laser': red_laser, 'initialize_laser_power': 130, 'initialize_dur': 1E3,
            'initialize_laser': green_laser, 'initialize_laser_filter': nd_green, 'initialize_dur': 1E3,
            'CPG_laser': red_laser, 'CPG_laser_power': 40, 'CPG_laser_dur': 1E4,
            # 'CPG_laser': green_laser, 'CPG_laser_filter': nd_green, 'CPG_laser_dur': 1E4,
            'charge_readout_laser': yellow_laser, 'charge_readout_laser_filter': None, 
            'charge_readout_laser_power': None, 'charge_readout_dur':250*10**6,
            'dir_1D': 'y',
            'collection_filter': '630_lp', 'magnet_angle': None,
            'resonance_LOW': 2.8012, 'rabi_LOW': 141.5, 'uwave_power_LOW': 15.5,  # 15.5 max
            'resonance_HIGH': 2.9445, 'rabi_HIGH': 191.9, 'uwave_power_HIGH': 14.5}   # 14.5 max 


    try:
        determine_readout_dur(nv_sig, readout_times =[ 250*10**6, 100*10**6],
                          readout_yellow_powers = [0.1],
                           nd_filter = 'nd_0.5')
    finally:
        # Reset our hardware - this should be done in each routine, but
        # let's double check here
        tool_belt.reset_cfm()
        # Kill safe stop
        if tool_belt.check_safe_stop_alive():
            print('\n\nRoutine complete. Press enter to exit.')
            tool_belt.poll_safe_stop()

#    file_name = 'pc_rabi/branch_Spin_to_charge/collect_charge_counts/2021_04/2021_04_28-17_18_59-goeppert-mayer-nv5_2021_04_15-nv_list'
#    data_in = tool_belt.get_raw_data('', file_name)
#    nv0_list = data_in['nv0_list']
#    nvm_list = data_in['nvm_list']
#    nv_sig = data_in['parameters_sig']
#    readout_time = nv_sig['pulsed_SCC_readout_dur']
#    calculate_threshold_plot(readout_time/10**6, nv0_list[0],nvm_list[0], '', '')


# use 60 ms, 0.15 ND filter



