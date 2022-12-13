# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 14:44:30 2022

File to run SRT Rabi measurements, based off this report 
https://journals.aps.org/prb/pdf/10.1103/PhysRevB.104.035201

@author: agardill
"""

import utils.tool_belt as tool_belt
import utils.kplotlib as kpl
from utils.kplotlib import KplColors
import majorroutines.optimize as optimize
import numpy
import os
import time
import matplotlib.pyplot as plt
from random import shuffle
from scipy.optimize import curve_fit
from utils.tool_belt import States
import labrad


def create_raw_data_figure(
    taus,
    avg_sig_counts=None,
    avg_ref_counts=None,
    norm_avg_sig=None,
    norm_avg_sig_ste = None,
    title = None
):
    num_steps = len(taus)
    # Plot setup
    fig, axes_pack = plt.subplots(1, 2, figsize=kpl.double_figsize)
    ax_sig_ref, ax_norm = axes_pack
    ax_sig_ref.set_xlabel('Microwave duration (high = low) (ns)')
    ax_sig_ref.set_ylabel("Count rate (kcps)")
    ax_norm.set_xlabel('Microwave duration (high = low) (ns)')
    ax_norm.set_ylabel("Normalized fluorescence")
    if title is not None:
        ax_norm.set_title(title)

    # Plotting
    if avg_sig_counts is None:
        avg_sig_counts = numpy.empty(num_steps)
        avg_sig_counts[:] = numpy.nan
    kpl.plot_line(
        ax_sig_ref, taus, avg_sig_counts, label="Signal", color=KplColors.GREEN
    )
    if avg_ref_counts is None:
        avg_ref_counts = numpy.empty(num_steps)
        avg_ref_counts[:] = numpy.nan
    kpl.plot_line(
        ax_sig_ref, taus, avg_ref_counts, label="Reference", color=KplColors.RED
    )
    ax_sig_ref.legend(loc=kpl.Loc.LOWER_RIGHT)
    if norm_avg_sig is None:
        norm_avg_sig = numpy.empty(num_steps)
        norm_avg_sig[:] = numpy.nan
    if norm_avg_sig_ste is not None:
        kpl.plot_points(ax_norm, taus, norm_avg_sig, yerr=norm_avg_sig_ste)
    else:
        kpl.plot_line(ax_norm, taus, norm_avg_sig, color=KplColors.BLUE)

    return fig, ax_sig_ref, ax_norm



def create_err_figure(
    taus,
    norm_avg_sig=None,
    norm_avg_sig_ste = None,
    title = None
):


    # Plot setup
    fig, ax = plt.subplots()
    ax.set_xlabel('Microwave duration (high = low) (ns)')
    ax.set_ylabel("Normalized fluorescence")
    if title is not None:
        ax.set_title(title)

    # Plotting
    if norm_avg_sig_ste is not None:
        kpl.plot_points(ax, taus, norm_avg_sig, yerr=norm_avg_sig_ste)

    return fig

# %% Main


              
def main(nv_sig, 
         num_steps, num_reps, num_runs,
         uwave_time_range_LOW, 
         uwave_time_range_HIGH, 
         readout_state = States.HIGH,
         initial_state = States.HIGH,
         opti_nv_sig = None,
         ):
        #Right now, make sure SRS is set as State HIGH
   

    with labrad.connect() as cxn:
        main_with_cxn(cxn, nv_sig,  
                 num_steps, num_reps, num_runs,
                 uwave_time_range_LOW, 
                 uwave_time_range_HIGH, 
                 readout_state,
                 initial_state,
                 opti_nv_sig,)


def main_with_cxn(cxn, nv_sig,  
                     num_steps, num_reps, num_runs,
                     uwave_time_range_LOW, 
                     uwave_time_range_HIGH = [], 
                     readout_state = States.HIGH,
                     initial_state = States.HIGH,
                     opti_nv_sig = None,):

    counter_server = tool_belt.get_server_counter(cxn)
    pulsegen_server = tool_belt.get_server_pulse_gen(cxn)
    tool_belt.reset_cfm(cxn)
    kpl.init_kplotlib()

    # %% Get the starting time of the function, to be used to calculate run time

    startFunctionTime = time.time()
    start_timestamp = tool_belt.get_time_stamp()

    # %% Initial calculations and setup
    
    state_high = States.HIGH
    state_low = States.LOW
    uwave_freq_high = nv_sig['resonance_{}'.format(state_high.name)]
    uwave_freq_low = nv_sig['resonance_{}'.format(state_low.name)]
    
    uwave_power_high = nv_sig['uwave_power_{}'.format(state_high.name)]
    uwave_power_low = nv_sig['uwave_power_{}'.format(state_low.name)]
    rabi_high = nv_sig['rabi_{}'.format(state_high.name)]
    rabi_low = nv_sig['rabi_{}'.format(state_low.name)]

    pi_pulse_high = tool_belt.get_pi_pulse_dur(rabi_high)
    pi_pulse_low = tool_belt.get_pi_pulse_dur(rabi_low)

    laser_key = 'spin_laser'
    laser_name = nv_sig[laser_key]
    tool_belt.set_filter(cxn, nv_sig, laser_key)
    laser_power = tool_belt.set_laser_power(cxn, nv_sig, laser_key)

    polarization_time = nv_sig['spin_pol_dur']
    readout = nv_sig['spin_readout_dur']
    readout_sec = readout / (10**9)
    
    norm_style = nv_sig["norm_style"]

    # Array of times to sweep through
    # Must be ints since the pulse streamer only works with int64s
    min_uwave_time_LOW = uwave_time_range_LOW[0]
    max_uwave_time_LOW = uwave_time_range_LOW[1]    
    t_LOW_list = numpy.linspace(min_uwave_time_LOW,max_uwave_time_LOW,num_steps)
    
    min_uwave_time_HIGH = uwave_time_range_HIGH[0]
    max_uwave_time_HIGH = uwave_time_range_HIGH[1]    
    t_HIGH_list = numpy.linspace(min_uwave_time_HIGH,max_uwave_time_HIGH,num_steps)

                
    # Analyze the sequence
    num_reps = int(num_reps)
    file_name = 'rabi_consec.py'
    seq_args = [t_LOW_list[0],t_HIGH_list[0], polarization_time,
                readout, pi_pulse_low, pi_pulse_high, t_LOW_list[0],  t_HIGH_list[0],
                initial_state.value, readout_state.value, 
                laser_name, laser_power] 
    
                
                
#    for arg in seq_args:
#        print(type(arg))
    # print(seq_args)
    # return
    seq_args_string = tool_belt.encode_seq_args(seq_args)
    ret_vals = pulsegen_server.stream_load(file_name, seq_args_string)
    seq_time = ret_vals[0]

    seq_time_s = seq_time / (10 ** 9)  # to seconds
    expected_run_time_s = (
        (num_steps / 2) * num_reps * num_runs * seq_time_s# * 6 #taking slower than expected
    )  # s
    expected_run_time_m = expected_run_time_s / 60  # to minutes

    print(" \nExpected run time: {:.1f} minutes. ".format(expected_run_time_m))
    
    # Set up our data structure, 
    
    
    
    # %% Make some lists and variables to save at the end

    opti_coords_list = []


    # create figure
    img_array = numpy.empty([num_steps, num_steps])
    img_array[:] = numpy.nan

    ### Signal generators servers
    low_sig_gen_cxn = tool_belt.get_server_sig_gen(
        cxn, States.LOW
    )
    high_sig_gen_cxn = tool_belt.get_server_sig_gen(
        cxn, States.HIGH
    )
    
    # %% Collect the data

    # Start 'Press enter to stop...'
    tool_belt.init_safe_stop()
    
            
    # for run_ind in range(num_runs):
    #     print('Run index: {}'. format(run_ind))



    
    for t_L_i in range(len(t_LOW_list)):
        for t_H_i in range(len(t_HIGH_list)):
            t_LOW = t_LOW_list[t_L_i]
            t_HIGH = t_HIGH_list[t_H_i]
            print("t_LOW {} ns, t_HIGH {} ns".format(t_LOW, t_HIGH))
            
            
            # Optimize and save the coords we found
            if opti_nv_sig:
                opti_coords = optimize.main_with_cxn(cxn, opti_nv_sig)
                drift = tool_belt.get_drift()
                adj_coords = nv_sig['coords'] + numpy.array(drift)
                tool_belt.set_xyz(cxn, adj_coords)
            else:
                opti_coords = optimize.main_with_cxn(cxn, nv_sig)
            opti_coords_list.append(opti_coords)
        
            tool_belt.set_filter(cxn, nv_sig, "spin_laser")
            laser_power = tool_belt.set_laser_power(cxn, nv_sig, laser_key)
        
            # Set up the microwaves for the low and high states
            low_sig_gen_cxn.set_freq(uwave_freq_low)
            low_sig_gen_cxn.set_amp(uwave_power_low)
            low_sig_gen_cxn.uwave_on()
        
            high_sig_gen_cxn.set_freq(uwave_freq_high)
            high_sig_gen_cxn.set_amp(uwave_power_high)
            high_sig_gen_cxn.uwave_on()
            
            # Load the APD
            counter_server.start_tag_stream()

            # Stream the sequence
            
            seq_args = [t_LOW_list[t_L_i], t_HIGH_list[t_H_i],polarization_time,
                readout, pi_pulse_low, pi_pulse_high, t_LOW_list[t_L_i], t_HIGH_list[t_H_i],
                initial_state.value, readout_state.value, 
                laser_name, laser_power]
    
            seq_args_string = tool_belt.encode_seq_args(seq_args)
            # print(seq_args)
            # Clear the tagger buffer of any excess counts
            counter_server.clear_buffer()
            pulsegen_server.stream_immediate(file_name, num_reps,
                                                seq_args_string)

            # Get the counts
            new_counts = counter_server.read_counter_separate_gates(1)

            sample_counts = new_counts[0]

            
            count = sum(sample_counts[0::2])
            sig_counts = count
            # print("First signal = " + str(count))

            count = sum(sample_counts[1::2])
            ref_counts = count
            
            norm_avg_sig = sig_counts / ref_counts


            img_array[t_L_i][t_H_i] = norm_avg_sig
                
    counter_server.stop_tag_stream()
    
    img_array=numpy.flipud(img_array)
    kpl.imshow
    fig, ax = plt.subplots()
    axes_labels = ['HIGH MW pulse dur (ns)', 'LOW MW pulse dur (ns)']
    half_pixel_size_x = (t_HIGH_list[1] - t_HIGH_list[0])/2
    half_pixel_size_y = (t_LOW_list[1] - t_LOW_list[0])/2
    img_extent = [
        min_uwave_time_HIGH-half_pixel_size_x,
        max_uwave_time_HIGH+half_pixel_size_x,
        min_uwave_time_LOW-half_pixel_size_y,
        max_uwave_time_LOW+half_pixel_size_y,
    ]
    kpl.imshow(
        ax,
        img_array,
        axes_labels=axes_labels,
        cbar_label="Norm. fluor.",
        extent=img_extent,
    )
    title='{} initial state, {} readout state'.format(initial_state.name, 
                            readout_state.name)
    ax.set_title(title)
    print(list(img_array))
                
    #     # %% incremental plotting
        
    #     # Update the run indicator
        
    #     inc_sig_counts = sig_counts[: run_ind + 1]
    #     inc_ref_counts = ref_counts[: run_ind + 1]
    #     ret_vals = tool_belt.process_counts(
    #         inc_sig_counts, inc_ref_counts, num_reps, readout, norm_style
    #     )
    #     (
    #         sig_counts_avg_kcps,
    #         ref_counts_avg_kcps,
    #         norm_avg_sig,
    #         norm_avg_sig_ste,
    #     ) = ret_vals
        
    #     if do_plot:
    #         text = run_indicator_text.format(run_ind + 1, num_runs)
    #         run_indicator_obj.txt.set_text(text)
    #         kpl.plot_line_update(ax_sig_ref, line_ind=0, y=sig_counts_avg_kcps)
    #         kpl.plot_line_update(ax_sig_ref, line_ind=1, y=ref_counts_avg_kcps)
    #         kpl.plot_line_update(ax_norm, y=norm_avg_sig)
        


    #     # %% Save the data we have incrementally for long measurements

    #     raw_data = {'start_timestamp': start_timestamp,
    #                 'nv_sig': nv_sig,
    #                 'nv_sig-units': tool_belt.get_nv_sig_units(cxn),
    #                 'uwave_time_range_LOW': uwave_time_range_LOW,
    #                 'uwave_time_range_HIGH': uwave_time_range_HIGH,
    #                 'uwave_time_range-units': 'ns',
    #                 'taus_LOW': taus_LOW.tolist(),
    #                 'taus_HIGH': taus_HIGH.tolist(),
    #                 'initial_state': initial_state.name,
    #                 'readout_state': readout_state.name,
    #                 'num_steps': num_steps,
    #                 'num_reps': num_reps,
    #                 'num_runs': num_runs,
    #                 'tau_index_master_list':tau_index_master_list,
    #                 'opti_coords_list': opti_coords_list,
    #                 'opti_coords_list-units': 'V',
    #                 'sig_counts': sig_counts.astype(int).tolist(),
    #                 'sig_counts-units': 'counts',
    #                 'ref_counts': ref_counts.astype(int).tolist(),
    #                 'ref_counts-units': 'counts'}

    #     # This will continuously be the same file path so we will overwrite
    #     # the existing file with the latest version
    #     file_path = tool_belt.get_file_path(__file__, start_timestamp,
    #                                         nv_sig['name'], 'incremental')
    #     tool_belt.save_raw_data(raw_data, file_path)
    #     if do_plot:
    #         tool_belt.save_figure(raw_fig, file_path)

    # # # %% Fit the data and extract piPulse

    # # fit_func, popt = fit_data(uwave_time_range, num_steps, norm_avg_sig)

    # # %% Plot the Rabi signal

    # ### Process and plot the data

    # ret_vals = tool_belt.process_counts(sig_counts, ref_counts, num_reps, readout, norm_style)
    # (
    #     sig_counts_avg_kcps,
    #     ref_counts_avg_kcps,
    #     norm_avg_sig,
    #     norm_avg_sig_ste,
    # ) = ret_vals

    # # Raw data
    # if do_plot:
    #     kpl.plot_line_update(ax_sig_ref, line_ind=0, y=sig_counts_avg_kcps)
    #     kpl.plot_line_update(ax_sig_ref, line_ind=1, y=ref_counts_avg_kcps)
    #     kpl.plot_line_update(ax_norm, y=norm_avg_sig)
    #     run_indicator_obj.remove()
    

    # # %% Plot the data itself and the fitted curve

    # # fit_fig = None
    # # if (fit_func is not None) and (popt is not None):
    # #     fit_fig = create_fit_figure(uwave_time_range, uwave_freq, num_steps,
    # #                                 norm_avg_sig, fit_func, popt)
    # #     rabi_period = 1/popt[1]
    # #     print('Rabi period measured: {} ns\n'.format('%.1f'%rabi_period))

    # # %% Clean up and save the data

    # tool_belt.reset_cfm(cxn)
    # # turn off FM
    # if hasattr(low_sig_gen_cxn, "fm_off"):
    #     low_sig_gen_cxn.fm_off() 
    # if hasattr(high_sig_gen_cxn, "fm_off"):
    #     high_sig_gen_cxn.fm_off() 

    # endFunctionTime = time.time()

    # timeElapsed = endFunctionTime - startFunctionTime

    # timestamp = tool_belt.get_time_stamp()

    # raw_data = {'timestamp': timestamp,
    #             'timeElapsed': timeElapsed,
    #             'timeElapsed-units': 's',
    #             'nv_sig': nv_sig,
    #             'nv_sig-units': tool_belt.get_nv_sig_units(cxn),
    #             'initial_state': initial_state.name,
    #             'readout_state': readout_state.name,
    #             'num_steps': num_steps,
    #             'num_reps': num_reps,
    #             'num_runs': num_runs,
    #             'uwave_time_range_LOW': uwave_time_range_LOW,
    #             'uwave_time_range_HIGH': uwave_time_range_HIGH,
    #             'uwave_time_range-units': 'ns',
    #             'taus_LOW': taus_LOW.tolist(),
    #             'taus_HIGH': taus_HIGH.tolist(),
    #             'opti_coords_list': opti_coords_list,
    #             'opti_coords_list-units': 'V',
    #             'sig_counts': sig_counts.astype(int).tolist(),
    #             'sig_counts-units': 'counts',
    #             'ref_counts': ref_counts.astype(int).tolist(),
    #             'ref_counts-units': 'counts',
    #             'norm_avg_sig': norm_avg_sig.astype(float).tolist(),
    #             'norm_avg_sig-units': 'arb',
    #             'norm_avg_sig_ste': norm_avg_sig_ste.astype(float).tolist(),}

    # nv_name = nv_sig["name"]
    # file_path = tool_belt.get_file_path(__file__, timestamp, nv_name)
    # if do_plot:
    #     tool_belt.save_figure(raw_fig, file_path)
    # # if fit_fig is not None:
    # #     file_path_fit = tool_belt.get_file_path(__file__, timestamp, nv_name + "-fit")
    # #     tool_belt.save_figure(fit_fig, file_path_fit)
    # tool_belt.save_raw_data(raw_data, file_path)
    
    
    # # if (fit_func is not None) and (popt is not None):
    # #     return rabi_period, sig_counts, ref_counts, popt
    # # else:
    # #     return None, sig_counts, ref_counts, []

    return 

# %%
def plot_pop_consec(taus, m_pop, z_pop, p_pop,
                    m_err = None,
                    z_err = None,
                    p_err = None):
    
    fig, ax = plt.subplots()
    ax.set_title('Rabi double quantum')
    ax.set_xlabel('SRT length (ns)')
    ax.set_ylabel('Population')
    ax.set_title('Rabi with consec. pulses')
    
    # Plotting
    if m_err is not None:
        kpl.plot_points(ax, taus, m_pop, yerr=m_err, color=KplColors.RED,
                        label = '-1 population')
    else:
        kpl.plot_line(ax, taus, m_pop, color=KplColors.RED,
                        label = '-1 population')
        
    if z_err is not None:
        kpl.plot_points(ax, taus, z_pop, yerr=z_err, color=KplColors.GREEN, 
                        label = '0 population')
    else:
        kpl.plot_line(ax, taus, z_pop, color=KplColors.GREEN,
                        label = '0 population')
        
    if p_err is not None:
        kpl.plot_points(ax, taus, p_pop, yerr=p_err, color=KplColors.BLUE, 
                        label = '+1 population')
    else:
        kpl.plot_line(ax, taus, p_pop, color=KplColors.BLUE,
                        label = '+1 population')
    
    ax.legend()
    
    return fig
    
    
def full_pop_consec(nv_sig, uwave_time_range,
         num_steps, num_reps, num_runs):
    
    contrast = 0.11*2
    min_pop = 1-contrast
    
    min_uwave_time = uwave_time_range[0]
    max_uwave_time = uwave_time_range[1]
    taus = numpy.linspace(min_uwave_time, max_uwave_time,
                          num=num_steps)
    
    init=States.LOW
    p_sig, p_ste = main(nv_sig, 
             num_steps, num_reps, num_runs,
             uwave_time_range,
             readout_state = States.HIGH,
             initial_state = init,
             do_err_plot = False,
             )
    p_pop = (numpy.array(p_sig) - min_pop) / (1 - min_pop)
    p_err = numpy.array(p_ste)/ (1 - min_pop)
        
    m_sig, m_ste = main(nv_sig, 
        uwave_time_range,
        num_steps, num_reps, num_runs,
        readout_state = States.LOW,
        initial_state = init,
        do_err_plot = False,
        )
    m_pop = (numpy.array(m_sig) - min_pop) / (1 - min_pop)
    m_err = numpy.array(m_ste)/ (1 - min_pop)
    
    z_sig, z_ste = main(nv_sig, 
            uwave_time_range,
            num_steps, num_reps, num_runs,
            readout_state = States.ZERO,
            initial_state = init,
            do_err_plot = False,
            )
    z_pop = (numpy.array(z_sig) - min_pop) / (1 - min_pop)
    z_err = numpy.array(z_ste)/ (1 - min_pop)
    

    
    fig = plot_pop_consec(taus, m_pop, z_pop, p_pop,
                            m_err, z_err, p_err)

    
    timestamp = tool_belt.get_time_stamp()
    nv_name = nv_sig["name"]
    file_path = tool_belt.get_file_path(__file__, timestamp, nv_name)
    tool_belt.save_figure(fig, file_path)
    
def fit_data(taus,  norm_avg_sig):

        # %% Set up

        fit_func = lambda t, off, freq: tool_belt.cosexp_1_at_0(t, off, freq, 1e3)

        # %% Estimated fit parameters

        offset = 0.2#numpy.average(norm_avg_sig)
        decay = 10
        frequency = 0.8

        # %% Fit

        init_params = [offset, frequency, decay]
        init_params = [offset, frequency]

        try:
            popt, _ = curve_fit(fit_func, taus, norm_avg_sig,
                                p0=init_params,
                                bounds=(0, numpy.infty))
        except Exception as e:
            print(e)
            popt = None

        return fit_func, popt
# %% Run the file


if __name__ == '__main__':

    path = 'pc_rabi/branch_master/rabi_consec/2022_12'
    file_p = '2022_12_08-13_44_01-siena-nv1_2022_10_27'
    file_m = '2022_12_08-13_52_38-siena-nv1_2022_10_27'
    file_z = '2022_12_08-14_01_09-siena-nv1_2022_10_27'
    
    data = tool_belt.get_raw_data(file_p, path)
    p_sig = data['norm_avg_sig']
    p_ste = data['norm_avg_sig_ste']
    data = tool_belt.get_raw_data(file_z, path)
    z_sig = data['norm_avg_sig']
    z_ste = data['norm_avg_sig_ste']
    data = tool_belt.get_raw_data(file_m, path)
    m_sig = data['norm_avg_sig']
    m_ste = data['norm_avg_sig_ste']
    taus= numpy.array(data['taus'])/1e3
    
    contrast = 0.220
    low_pop = 1-contrast
    
    p_pop = (numpy.array(p_sig) - low_pop) / (1 - low_pop)
    z_pop = (numpy.array(z_sig) - low_pop) / (1 - low_pop)
    m_pop = (numpy.array(m_sig) - low_pop) / (1 - low_pop)
    
    p_err = numpy.array(p_ste) / (1 - low_pop)
    z_err = numpy.array(z_ste) / (1 - low_pop)
    m_err = numpy.array(m_ste) / (1 - low_pop)
    
    
    plot_pop_consec(taus,  m_pop, z_pop, p_pop, m_err, z_err,p_err )