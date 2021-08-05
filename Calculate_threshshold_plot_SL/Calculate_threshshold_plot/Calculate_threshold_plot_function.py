# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 14:09:39 2021

@author: samli
"""

import labrad
import scipy.stats
import scipy.special
import math  
import copy
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import random
import photonstatistics as model

import utils.tool_belt as tool_belt
import majorroutines.optimize as optimize

import os

#%% import your data in the data_file 
import NV_data as data
#%% functions 

#fit the NV- and NV0 counts to the rate model 
def get_photon_dis_curve_fit_plot(readout_time,NV0,NVm, do_plot = True): # NV0, NVm in array
    trail_nv0_30ms = NV0
    trail_nvm_30ms = NVm
    tR = readout_time 
    combined_count = trail_nvm_30ms.tolist() + trail_nv0_30ms.tolist()
    random.shuffle(combined_count)
    # fit = [g0,g1,y1,y0]
    
    # if it is the SiV sample, use this guess 
#    guess = [ 0.01 ,0.02, 0.57, 0.08] 
    
    # if it is the E6 sample, use this guess 
    guess = [ 10*10**-4,100*10**-4, 1000*10**-4, 500*10**-4]
    
    # fit gives g0, g1, y1, y0
    fit,dev = model.get_curve_fit(tR,0,combined_count,guess)
    print(fit,np.diag(dev))
    
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
    return fit, fig4

# calculate the threshold of given NV0 and NV- data 
# tR in units of ms
def calculate_threshold_plot(readout_time,nv0_array,nvm_array):
    tR = readout_time
    fit_rate, _ = get_photon_dis_curve_fit_plot(readout_time,nv0_array,nvm_array)
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

# power should be in unit of uW
def fit_to_rates_g(power_list, g0_list,g1_list):    
    popt, pcov = curve_fit(rate_model,power_list,g0_list,p0 = [0.05,0.1],bounds = ([0,-np.inf],[np.inf,np.inf]))
    A0,B0 = popt
    popt,pcov = curve_fit(rate_model,power_list,g1_list,p0 = [0.2,0.5],bounds = ([0,-np.inf],[np.inf,np.inf]))
    A1,B1 = popt
    A0 = round(A0,3)
    A1 = round(A1,3)
    B0 = round(B0,3)
    B1 = round(B1,3)
    power_array = np.linspace(0,max(power_list),100)
    g0_curve = rate_model(power_array,A0,B0)
    g1_curve = rate_model(power_array,A1,B1)
    fig, ax = plt.subplots()
    ax.plot(power_array,g0_curve,'r')
    ax.plot(power_array,g1_curve,'g')
    ax.plot(power_list,g0_list,'ro')
    ax.plot(power_list,g1_list,'go')
    plt.xlabel('power ($\mu$  W)')
    plt.ylabel('rate ($s^{-1}$)')
    plt.title('NV Ionization and Recombination Rates')
    B1 = round(B1,3)
    print("g0(P) = "+ str(A0) +"*P^2 + "+str(B0)+'*P')
    print("g1(P) = "+ str(A1) +"*P^2 + "+str(B1)+'*P')
    return [A0,B0,A1,B1], fig
    
 # power should be in unit of uW
def fit_to_rates_y(power_list, y0_list,y1_list):    
    popt, pcov = curve_fit(FL_model,power_list,y0_list,bounds = ([0,-np.inf],[np.inf,np.inf]))
    A0,B0 = popt
    popt,pcov = curve_fit(FL_model,power_list,y1_list,bounds = ([0,-np.inf],[np.inf,np.inf]))
    A1,B1 = popt
    power_array = np.linspace(0,max(power_list),100)
    y0_curve = FL_model(power_array,A0,B0)
    y1_curve = FL_model(power_array,A1,B1)
    fig, ax = plt.subplots()
    ax.plot(power_array,y0_curve,'r')
    ax.plot(power_array,y1_curve,'g')
    ax.plot(power_list,y0_list,'ro')
    ax.plot(power_list,y1_list,'go')
    plt.xlabel('power ($\mu$  W)')
    plt.ylabel('PL rate ($kcps$)')
    plt.title('NV PL Rates ')
    A0 = round(A0,3)
    A1 = round(A1,3)
    B0 = round(B0,3)
    B1 = round(B1,3)
    print("y0(P) = "+ str(A0) +"*P + "+str(B0))
    print("y1(P) = "+ str(A1) +"*P + "+str(B1))
    return [A0,B0,A1,B1], fig
    
def rate_model(x,A,B):
    return A*x**2 + B*x

def FL_model(x,A, B):
    return A*x + B

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
    
    return nv0, nvm

def determine_readout_dur(nv_sig, readout_times = None, readout_yellow_powers = None,
                          nd_filter = 'nd_1.0'):
    num_reps = 200
    apd_indices =[0]
    
    if len(readout_times) != len(readout_yellow_powers):
        print('Readout durations and powers will be paired together. Please make lists the same size')
        return
    
    parameter_list = list(zip(readout_yellow_powers, readout_times))
    # if not readout_times:
    #     readout_times = [50*10**6, 100*10**6, 250*10*6]
    # if not readout_yellow_powers:
    #     readout_yellow_powers = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        
    tool_belt.init_safe_stop()
        
    # first index will be power, second will be time, third wil be individual points
    nv0_list= []
    nvm_list =[]
    
    g0_list = []
    g1_list = []
    y1_list = []
    y0_list = []
    
    for i in range(len(parameter_list)):
        # nv0_power =[]
        # nvm_power =[]    
        
        # add in some fit parameters for sample 
        # temp_g0 = []
        # temp_g1 = []
        # temp_y1 = []
        # temp_y0 = []
        p = parameter_list[i][0]
        
        # Break out of the while if the user says stop
        if tool_belt.safe_stop():
            break
        
        t = parameter_list[i][1]
        
        nv_sig_copy = copy.deepcopy(nv_sig)
        nv_sig_copy['charge_readout_laser_filter'] = nd_filter
        nv_sig_copy['charge_readout_dur'] = t
        nv_sig_copy['charge_readout_laser_power'] = p
        
        print('Measuring  {} ms, at AOM voltage {} V'.format(t/10**6, p))
        nv0, nvm = measure(nv_sig_copy, apd_indices, num_reps)       
        
        timestamp = tool_belt.get_time_stamp()  
        file_path = tool_belt.get_file_path(__file__, timestamp, nv_sig['name']) 
        
        nv0_list.append(nv0)
        nvm_list.append(nvm)
        
        print("calculating optimum readout params ...")
        #readout time should be in units of ms
        t_ms = t*10**-6
        fit, fig_phton_dist = get_photon_dis_curve_fit_plot(t_ms, nv0, nvm, do_plot = True)
        tool_belt.save_figure(fig_phton_dist, file_path)
        
        # avoid diverging results
        if fit[0] < 0.1:
            if fit[1] < 1:
                #g0,g1 in units of s^-1
                g0_list.append(fit[0]*10**3)
                g1_list.append(fit[1]*10**3)
                #y1,y0 in units of kcps
                y1_list.append(fit[2])
                y0_list.append(fit[3])            
        
        raw_data = {'timestamp': timestamp,
            'readout_time': t,
            'readout_time-units': 'ns',
            'nd_filter': nd_filter,
            'readout_aom_voltage': p,
            'nv_sig': nv_sig,
            'nv_sig-units': tool_belt.get_nv_sig_units(),
            'num_runs':num_reps,    
            'g0': fit[0],   
            'g1': fit[1],   
            'y0': fit[2],   
            'y1': fit[3],
            'readout_yellow_powers': readout_yellow_powers,
            'readout_yellow_powers-units': 'V',
            'nv0': nv0.tolist(),
            'nv0-units': 'counts',
            'nvm': nvm.tolist(),
            'nvm-units': 'counts',
            }
    
        tool_belt.save_raw_data(raw_data, file_path)
        #############
        
        # nv0_master.append(nv0_power)
        # nvm_master.append(nvm_power)
        
        # fit_time_array = np.array([np.mean(g0_list),np.mean(g1_list),np.mean(y1_list),np.mean(y0_list)])
        #g0,g1 in units of s^-1
        # g0_list.append(fit_time_array[0]*10**3)
        # g1_list.append(fit_time_array[1]*10**3)
        # #y1,y0 in units of kcps
        # y1_list.append(fit_time_array[2])
        # y0_list.append(fit_time_array[3])
            
    

    ### Modeling 
    # print("calculating optimum readout params ...")
    # #readout time should be in units of ms
    # readout_times = np.array(readout_times)*10**-6
    # g0_list = []
    # g1_list = []
    # y1_list = []
    # y0_list = []
    
    # for pi in range(len(nv0_master)):
    # # add in some fit parameters for sample 
    #     temp_g0 = []
    #     temp_g1 = []
    #     temp_y1 = []
    #     temp_y0 = []
    #     for ti in range(np.size(readout_times)):
    #         fit, fig_phton_dist = get_photon_dis_curve_fit_plot(readout_times[ti],nv0_master[pi][ti],nvm_master[pi][ti], do_plot = True)
    #         tool_belt.save_figure(fig_phton_dist, file_path)
    #         # avoid diverging results
    #         if fit[0] < 0.1:
    #             if fit[1] < 1:
    #                 temp_g0.append(fit[0])
    #                 temp_g1.append(fit[1])
    #                 temp_y1.append(fit[2])
    #                 temp_y0.append(fit[3])
    #     fit_time_array = np.array([np.mean(temp_g0),np.mean(temp_g1),np.mean(temp_y1),np.mean(temp_y0)])
    #     #g0,g1 in units of s^-1
    #     g0_list.append(fit_time_array[0]*10**3)
    #     g1_list.append(fit_time_array[1]*10**3)
    #     #y1,y0 in units of kcps
    #     y1_list.append(fit_time_array[2])
    #     y0_list.append(fit_time_array[3])
        

    # need to define this measured_589_power model if the laser power is changed; it converts power unit AOM to uW
    # this can be input manually 
    power_list = []
    for p in readout_yellow_powers:
        power_uw = model.measured_589_power(p, nd_filter)
        power_list.append(power_uw)
    
   # fit each rate parameters to the power dependence model 
   # g0,g1 are quadratic, y1,y0 are linear w.r.t power 
         
    timestamp = tool_belt.get_time_stamp()
    file_path = tool_belt.get_file_path(__file__, timestamp, nv_sig['name'])
    
    ret_vals = fit_to_rates_g(power_list, g0_list,g1_list)
    # tool_belt.save_figure(ret_vals[0], file_path)
    nv_para = ret_vals[0]
    ret_vals = fit_to_rates_y(power_list, y0_list,y1_list)
    # tool_belt.save_figure(ret_vals[0], file_path)
    nv_para = nv_para + ret_vals[0]
    
    #this range can be changed case by case
    power_range = [0.01,max(power_list)]
    time_range = [10,200]
    optimize_steps = 10
    
    result = model.optimize_single_shot_readout(power_range,time_range,nv_para,optimize_steps)
    print("optimized readout time ="+str(result[0]) +' ms')
    print("optimized readout power = {} nW".format(result[1]))
    print("optimized threshold ="+str(result[2]))
    print("optimized fidelity ="+str(result[3]))
    return result
    
#%% 
if __name__ == '__main__':
    sample_name = 'johnson'    
    
    green_laser = 'laserglow_532'
    yellow_laser = 'laserglow_589'
    red_laser = 'cobolt_638'
    nd_green = 'nd_0.5'
    
    nv_sig = { 'coords': [0.056, -0.098, 5.0],
            'name': '{}-nv1_2021_07_27'.format(sample_name),
            'disable_opt': False, 'expected_count_rate': 42,
            'imaging_laser': green_laser, 'imaging_laser_filter': nd_green, 'imaging_readout_dur': 1E7,
            'nv-_prep_laser': green_laser, 'nv-_prep_laser_filter': nd_green, 'nv-_prep_laser_dur': 1E3,
            'nv0_prep_laser': red_laser, 'nv0_prep_laser_value': 130, 'nv0_prep_laser_dur': 1E3,
            'charge_readout_laser': yellow_laser, 'charge_readout_laser_filter': None, 
            'charge_readout_laser_power': None, 'charge_readout_dur':None,
            'collection_filter': '630_lp', 'magnet_angle': None,
            'resonance_LOW': 2.8012, 'rabi_LOW': 141.5, 'uwave_power_LOW': 15.5,  # 15.5 max
            'resonance_HIGH': 2.9445, 'rabi_HIGH': 191.9, 'uwave_power_HIGH': 14.5}   # 14.5 max
    
    try:
        determine_readout_dur(nv_sig, readout_times =[250*10**6, 150*10**6, 150*10**6, 60*10**6],
                          readout_yellow_powers = [0.05, 0.1, 0.15, 0.2], 
                          nd_filter = 'nd_0.5')
        
    finally:
        # Reset our hardware - this should be done in each routine, but
        # let's double check here
        tool_belt.reset_cfm()
        # Kill safe stop
        if tool_belt.check_safe_stop_alive():
            print('\n\nRoutine complete. Press enter to exit.')
            tool_belt.poll_safe_stop()
    
    # file = '2021_04_13-19_14_05-johnson-nv0_2021_04_13-readout_pulse_pwr'
    # folder = 'pc_rabi/branch_Spin_to_charge/SCC_optimize_pulses_wout_uwaves/2021_04'
    # data_f = tool_belt.get_raw_data(folder, file)
    # nv_sig = data_f['nv_sig']
    # readout_time = nv_sig['pulsed_SCC_readout_dur']
    # sig_count_raw = data_f['sig_count_raw']
    # NV0 = np.array(sig_count_raw[3])
    # ref_count_raw = data_f['ref_count_raw']
    # NVm = np.array(ref_count_raw[3])
    # get_photon_dis_curve_fit_plot(readout_time,NV0,NVm, do_plot = True)
    
    # source_name = 'E:/Shared drives/Kolkowitz Lab Group/nvdata'
    # path_from_nvdata = 'pc_rabi/branch_master/Calculate_threshold_plot_function'
    # subfolder = '2021_07/2021_07_22-johnson-nv1_2021_07_16'
    # folder_dir = tool_belt.get_folder_dir(source_name + '/' + path_from_nvdata, subfolder)
    # file_list = tool_belt.get_files_in_folder(folder_dir, filetype = 'txt')
    # print(file_list)
    
    # temp_g0 = []
    # temp_g1 = []
    # temp_y1 = []
    # temp_y0 = []
    
    # power_list = []
        
    # for file in file_list:
    #     file = file[:-4]
    #     data = tool_belt.get_raw_data(file, path_from_nvdata + '/' + subfolder)
    #     t = data['readout_time']
    #     nv0 = np.array(data['nv0'])
    #     nvm = np.array(data['nvm'])
    #     power = data['readout_aom_voltage']
        
    #     t_ms = t*10**-6
    #     fit, fig_phton_dist = get_photon_dis_curve_fit_plot(t_ms, nv0, nvm, do_plot = True)
        
    #     if fit[0] < 0.1:
    #             if fit[1] < 1:
    #                 temp_g0.append(fit[0])
    #                 temp_g1.append(fit[1])
    #                 temp_y1.append(fit[2])
    #                 temp_y0.append(fit[3])  
        
    #%% ======================================================================
    # g0_100 = [0.25, 0.42, 0.38]
    # g1_100 = [0, 1.06, 1.19]
    # y0_100 = [0.03, 0.03, 0.03]
    # y1_100 = [0.16, 0.16, 0.16]
    
    # g0_250 = [7.06, 5.8, 3.88]
    # g1_250 = [15.56, 4.09, 2.23]
    # y0_250 = [0.04, 0.04, 0.05]
    # y1_250 = [0.3, 0.19, 0.2]
    
    
    # g0_400 = [17.03, 7.04, 4.81]
    # g1_400 = [3.16, 2.01, 0.77]
    # y0_400 = [0.0, 0.12, 0.14]
    # y1_400 = [0.4, 0.4, 0.38]
    
    # g0_temp = [g0_100, g0_250, g0_400]
    # g1_temp = [g1_100, g1_250, g1_400]
    # y0_temp = [y0_100, y0_250, y0_400]
    # y1_temp = [y1_100, y1_250, y1_400]
    
    # g0_list = [2.7E-6, 1.28, 7.39, 34.0]
    # g1_list = [63.35, 1.17, 7.74, 30.04]
    # y0_list= [0.03, 0.02, 0.04, 0.10]
    # y1_list = [0.12, 0.18, 0.25, 0.56]
    
    # # for i in range(len(g0_temp)):
    # #     fit_time_array = np.array([np.mean(g0_temp[i]),np.mean(g0_temp[i])
    # #                                ,np.mean(y0_temp[i]),np.mean(y1_temp[i])])
    # #     #g0,g1 in units of s^-1
    # #     g0_list.append(fit_time_array[0]*10**3)
    # #     g1_list.append(fit_time_array[1]*10**3)
    # #     #y1,y0 in units of kcps
    # #     y1_list.append(fit_time_array[2])
    # #     y0_list.append(fit_time_array[3])
    
    # power_list = [0.1, 1, 3.4, 10.7]
    
    # ret_vals = fit_to_rates_g(power_list, g0_list,g1_list)
    # # tool_belt.save_figure(ret_vals[0], file_path)
    # nv_para = ret_vals[0]
    # ret_vals = fit_to_rates_y(power_list, y0_list,y1_list)
    # # tool_belt.save_figure(ret_vals[0], file_path)
    # nv_para = nv_para + ret_vals[0]
    
    # #this range can be changed case by case
    # power_range = [0,max(power_list)]
    # time_range = [10,200]
    # optimize_steps = 10
    
    # result = model.optimize_single_shot_readout(power_range,time_range,nv_para,optimize_steps)
    # print("optimized readout time ="+str(result[0]) +' ms')
    # print("optimized readout power ="+str(result[1])+' uW')
    # print("optimized threshold ="+str(result[2]))
    # print("optimized fidelity ="+str(result[3]))



