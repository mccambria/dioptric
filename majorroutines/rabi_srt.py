# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 14:44:30 2022

File to run SRT Rabi measurements, based off this report 
https://journals.aps.org/prb/pdf/10.1103/PhysRevB.104.035201

@author: agardill
"""

import utils.tool_belt as tool_belt
import majorroutines.optimize as optimize
import numpy
import os
import time
import matplotlib.pyplot as plt
from random import shuffle
from scipy.optimize import curve_fit
from utils.tool_belt import States
import labrad


# %% Main


def main(nv_sig, uwave_time_range, deviation_high, deviation_low, 
         num_steps, num_reps, num_runs,
         readout_state = States.HIGH,
         initial_state = States.HIGH,
         opti_nv_sig = None,
         ):
        #Right now, make sure SRS is set as State HIGH
   

    with labrad.connect() as cxn:
        norm_avg_sig = main_with_cxn(cxn, nv_sig, uwave_time_range,  deviation_high, deviation_low, 
                 num_steps, num_reps, num_runs,
                 readout_state,
                 initial_state,
                 opti_nv_sig)



    return norm_avg_sig
def main_with_cxn(cxn, nv_sig, uwave_time_range, deviation_high, deviation_low, 
                     num_steps, num_reps, num_runs,
                     readout_state = States.HIGH,
                     initial_state = States.HIGH,
                     opti_nv_sig = None):

    counter_server = tool_belt.get_counter_server(cxn)
    pulsegen_server = tool_belt.get_pulsegen_server(cxn)
    tool_belt.reset_cfm(cxn)

    # %% Get the starting time of the function, to be used to calculate run time

    startFunctionTime = time.time()
    start_timestamp = tool_belt.get_time_stamp()

    # %% Initial calculations and setup
    dev_high_sign = 1
    dev_low_sign = 1
    if deviation_high < 0:
        dev_high_sign = -1
        
    if deviation_low < 0:
        dev_low_sign = -1
    
    state_high = States.HIGH
    state_low = States.LOW
    uwave_freq_high = nv_sig['resonance_{}'.format(state_high.name)]
    uwave_freq_low = nv_sig['resonance_{}'.format(state_low.name)]
    
    # uwave_freq_high_detune = uwave_freq_high + deviation_high / 1e3
    #uwave_freq_low_detune = uwave_freq_low + deviation_low / 1e3
    
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

    # Array of times to sweep through
    # Must be ints since the pulse streamer only works with int64s
    min_uwave_time = uwave_time_range[0]
    max_uwave_time = uwave_time_range[1]
    taus = numpy.linspace(min_uwave_time, max_uwave_time,
                          num=num_steps, dtype=numpy.int32)

    # Analyze the sequence
    num_reps = int(num_reps)
    file_name = os.path.basename(__file__)
    seq_args = [taus[0], polarization_time,
                readout, pi_pulse_low, pi_pulse_high, max_uwave_time, 
                dev_high_sign, dev_low_sign,
                initial_state.value, readout_state.value, 
                laser_name, laser_power]
#    for arg in seq_args:
#        print(type(arg))
    print(seq_args)
    # return
    seq_args_string = tool_belt.encode_seq_args(seq_args)
    ret_vals = pulsegen_server.stream_load(file_name, seq_args_string)
    seq_time = ret_vals[0]

    seq_time_s = seq_time / (10 ** 9)  # to seconds
    expected_run_time_s = (
        (num_steps / 2) * num_reps * num_runs * seq_time_s * 6 #taking slower than expected
    )  # s
    expected_run_time_m = expected_run_time_s / 60  # to minutes

    print(" \nExpected run time: {:.1f} minutes. ".format(expected_run_time_m))
    
    # Set up our data structure, an array of NaNs that we'll fill
    # incrementally. NaNs are ignored by matplotlib, which is why they're
    # useful for us here.
    # We define 2D arrays, with the horizontal dimension for the frequency and
    # the veritical dimension for the index of the run.
    sig_counts = numpy.empty([num_runs, num_steps], dtype=numpy.float32)
    sig_counts[:] = numpy.nan
    ref_counts = numpy.copy(sig_counts)
    # norm_avg_sig = numpy.empty([num_runs, num_steps])

    # %% Make some lists and variables to save at the end

    opti_coords_list = []
    tau_index_master_list = [[] for i in range(num_runs)]

    # Create a list of indices to step through the taus. This will be shuffled
    tau_ind_list = list(range(0, num_steps))

    # create figure
    raw_fig, axes_pack = plt.subplots(1, 2, figsize=(17, 8.5))
    ax = axes_pack[0]
    ax.plot([], [])
    ax.set_xlabel('rf time (ns)')
    ax.set_ylabel('Counts')

    ax = axes_pack[1]
    ax.plot([], [])
    ax.set_xlabel('Microwave duration (ns)')
    ax.set_ylabel('Normalized signal')

    # %% Collect the data

    # Start 'Press enter to stop...'
    tool_belt.init_safe_stop()
    for run_ind in range(num_runs):
        print('Run index: {}'. format(run_ind))

        # Break out of the while if the user says stop
        if tool_belt.safe_stop():
            break

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
        low_sig_gen_cxn = tool_belt.get_signal_generator_cxn(
            cxn, States.LOW
        )
        low_sig_gen_cxn.set_freq(uwave_freq_low)
        low_sig_gen_cxn.set_amp(uwave_power_low)
        # low_sig_gen_cxn.load_fm(abs(deviation_low))
        # low_sig_gen_cxn.uwave_on()

        high_sig_gen_cxn = tool_belt.get_signal_generator_cxn(
            cxn, States.HIGH
        )
        high_sig_gen_cxn.set_freq(uwave_freq_high)
        # high_sig_gen_cxn.set_freq(uwave_freq_high_detune)
        high_sig_gen_cxn.set_amp(uwave_power_high)
        high_sig_gen_cxn.load_fm(abs(deviation_high))
        high_sig_gen_cxn.uwave_on()
        

        # Load the APD
        counter_server.start_tag_stream()

        # Shuffle the list of indices to use for stepping through the taus
        shuffle(tau_ind_list)

#        start_time = time.time()
        for tau_ind in tau_ind_list:
#        for tau_ind in range(len(taus)):
            # print('Tau: {} ns'. format(taus[tau_ind]))
            # Break out of the while if the user says stop
            if tool_belt.safe_stop():
                break
#            print(taus[tau_ind])

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
            # Stream the sequence
            
            seq_args = [taus[tau_ind_first], polarization_time,
                readout, pi_pulse_low, pi_pulse_high, taus[tau_ind_second], 
                dev_high_sign, dev_low_sign,
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

            
            count = sum(sample_counts[0::4])
            sig_counts[run_ind, tau_ind_first] = count
            # print("First signal = " + str(count))

            count = sum(sample_counts[1::4])
            ref_counts[run_ind, tau_ind_first] = count
            # print("First Reference = " + str(count))

            count = sum(sample_counts[2::4])
            sig_counts[run_ind, tau_ind_second] = count
            # print("Second Signal = " + str(count))

            count = sum(sample_counts[3::4])
            ref_counts[run_ind, tau_ind_second] = count
            # print("Second Reference = " + str(count))

#            run_time = time.time()
#            run_elapsed_time = run_time - start_time
#            start_time = run_time
#            print('Tau: {} ns'.format(taus[tau_ind]))
#            print('Elapsed time {}'.format(run_elapsed_time))
        counter_server.stop_tag_stream()

        # %% incremental plotting

        #Average the counts over the iterations
        avg_sig_counts = numpy.average(sig_counts[:(run_ind+1)], axis=0)
        avg_ref_counts = numpy.average(ref_counts[:(run_ind+1)], axis=0)

        norm_avg_sig = avg_sig_counts / numpy.average(avg_ref_counts)


        ax = axes_pack[0]
        ax.cla()
        ax.plot(taus, avg_sig_counts, 'r-', label = 'signal')
        ax.plot(taus, avg_ref_counts, 'g-', label = 'reference')

        ax.set_xlabel('rf time (ns)')
        ax.set_ylabel('Counts')
        ax.legend()

        ax = axes_pack[1]
        ax.cla()
        ax.plot(taus , norm_avg_sig, 'b-')
        ax.set_title('{} initial state, {} readout state,\n{} MHz deviation on HIGH, {} MHz deviation on LOW'.format(initial_state.name, 
                                   readout_state.name, deviation_high, 
                                   deviation_low))
        ax.set_xlabel('Microwave duration (ns)')
        ax.set_ylabel('Normalized signal')

        text_popt = 'Run # {}/{}'.format(run_ind+1,num_runs)

        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.8, 0.9, text_popt,transform=ax.transAxes,
                verticalalignment='top', bbox=props)

        raw_fig.canvas.draw()
        raw_fig.set_tight_layout(True)
        raw_fig.canvas.flush_events()


        # %% Save the data we have incrementally for long measurements

        raw_data = {'start_timestamp': start_timestamp,
                    'nv_sig': nv_sig,
                    'nv_sig-units': tool_belt.get_nv_sig_units(),
                    'deviation_low': deviation_low,
                    'deviation_low-units': 'MHz',
                    'deviation_high': deviation_high,
                    'deviation_high-units': 'MHz',
                    'uwave_time_range': uwave_time_range,
                    'uwave_time_range-units': 'ns',
                    'taus': taus.tolist(),
                    'initial_state': initial_state.name,
                    'readout_state': readout_state.name,
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
        tool_belt.save_figure(raw_fig, file_path)

    # # %% Fit the data and extract piPulse

    # fit_func, popt = fit_data(uwave_time_range, num_steps, norm_avg_sig)

    # %% Plot the Rabi signal

    ax = axes_pack[0]
    ax.cla()
    ax.plot(taus, avg_sig_counts, 'r-', label = 'signal')
    ax.plot(taus, avg_ref_counts, 'g-', label = 'refernece')

    # ax.plot(tauArray, countsBackground, 'o-')
    ax.set_xlabel('rf time (ns)')
    ax.set_ylabel('Counts')
    ax.legend()
    ax.set_title('Shift to LOW freq {:.2} MHz'.format((uwave_freq_low - 2.7813)*1e3 ))

    ax = axes_pack[1]
    ax.cla()
    ax.plot(taus , norm_avg_sig, 'b-')
    ax.set_title('{} initial state, {} readout state,\n{} MHz deviation on HIGH, {} MHz deviation on LOW'.format(initial_state.name, 
                                   readout_state.name, deviation_high, 
                                   deviation_low))
    ax.set_xlabel('Microwave duration (ns)')
    ax.set_ylabel('Normalized signal')

    raw_fig.canvas.draw()
    raw_fig.set_tight_layout(True)
    raw_fig.canvas.flush_events()

    # %% Plot the data itself and the fitted curve

    # fit_fig = None
    # if (fit_func is not None) and (popt is not None):
    #     fit_fig = create_fit_figure(uwave_time_range, uwave_freq, num_steps,
    #                                 norm_avg_sig, fit_func, popt)
    #     rabi_period = 1/popt[1]
    #     print('Rabi period measured: {} ns\n'.format('%.1f'%rabi_period))

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
                'deviation_low': deviation_low,
                'deviation_low-units': 'MHz',
                'deviation_high': deviation_high,
                'deviation_high-units': 'MHz',
                'initial_state': initial_state.name,
                'readout_state': readout_state.name,
                'num_steps': num_steps,
                'num_reps': num_reps,
                'num_runs': num_runs,
                'taus': taus.tolist(),
                'tau_index_master_list':tau_index_master_list,
                'opti_coords_list': opti_coords_list,
                'opti_coords_list-units': 'V',
                'sig_counts': sig_counts.astype(int).tolist(),
                'sig_counts-units': 'counts',
                'ref_counts': ref_counts.astype(int).tolist(),
                'ref_counts-units': 'counts',
                'norm_avg_sig': norm_avg_sig.astype(float).tolist(),
                'norm_avg_sig-units': 'arb'}

    nv_name = nv_sig["name"]
    file_path = tool_belt.get_file_path(__file__, timestamp, nv_name)
    tool_belt.save_figure(raw_fig, file_path)
    # if fit_fig is not None:
    #     file_path_fit = tool_belt.get_file_path(__file__, timestamp, nv_name + "-fit")
    #     tool_belt.save_figure(fit_fig, file_path_fit)
    tool_belt.save_raw_data(raw_data, file_path)

    # if (fit_func is not None) and (popt is not None):
    #     return rabi_period, sig_counts, ref_counts, popt
    # else:
    #     return None, sig_counts, ref_counts, []

    return norm_avg_sig

# %%
def plot_pop_srt(taus, p_pop, z_pop, deviation, m_pop = []):
    
    fig, ax = plt.subplots()
    ax.plot(taus, p_pop, 'b-', label = '+1 population')
    ax.plot(taus, z_pop, 'g-', label = '0 population')
    if len(m_pop) != 0:
        ax.plot(taus , m_pop, 'r-', label = '-1 population')
    ax.set_title('Rabi SRT, {} MHz detuning'.format(deviation))
    ax.set_xlabel('SRT length (us)')
    ax.set_ylabel('Population')
    ax.legend()
    
    return fig
    
    
def full_pop_srt(nv_sig, apd_indices, uwave_time_range, deviation, 
         num_steps, num_reps, num_runs):
    
    contrast = 0.108*2
    low_pop = 1-contrast
    
    min_uwave_time = uwave_time_range[0]
    max_uwave_time = uwave_time_range[1]
    taus = numpy.linspace(min_uwave_time, max_uwave_time,
                          num=num_steps)
    taus = taus/1e3
    deviation_high = deviation
    deviation_low = deviation
    
    init=States.HIGH
    if True :
        m_sig = main(nv_sig, apd_indices, uwave_time_range, deviation_high, deviation_low, 
            num_steps, num_reps, num_runs,
            readout_state = States.LOW,
            initial_state = init,
            )
        m_pop = (numpy.array(m_sig) - low_pop) / (1 - low_pop)
 
    p_sig = main(nv_sig, apd_indices, uwave_time_range, deviation_high, deviation_low, 
             num_steps, num_reps, num_runs,
             readout_state = States.HIGH,
             initial_state = init,
             )
    p_pop = (numpy.array(p_sig) - low_pop) / (1 - low_pop)
    
    z_sig = main(nv_sig, apd_indices, uwave_time_range, deviation_high, deviation_low, 
            num_steps, num_reps, num_runs,
            readout_state = States.ZERO,
            initial_state = init,
            )
    z_pop = (numpy.array(z_sig) - low_pop) / (1 - low_pop)
    

    
    fig = plot_pop_srt(taus, p_pop, z_pop, deviation, m_pop)

    
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

    path = 'pc_rabi/branch_master/rabi_srt/2022_11'
    file_1 = '2022_11_29-16_58_39-siena-nv1_2022_10_27'
    file_2 = '2022_11_29-18_14_41-siena-nv1_2022_10_27'
    file_3 = '2022_11_29-19_30_36-siena-nv1_2022_10_27'
    file_4 = '2022_11_29-20_46_28-siena-nv1_2022_10_27'
    file_0 = '2022_11_29-13_22_27-siena-nv1_2022_10_27'
    file_5 = '2022_11_29-23_23_29-siena-nv1_2022_10_27'
    
    file_list = [
                  # file_m4,
                 
                  # file_m3,
                  file_5,
                 file_1,
                 file_2,
                 file_0,
                 file_3,
                 file_4,
                  # file_p3,
                  # file_p4,
                 ]
    
    color_list = ['orange','red', 'blue', 'green' ,'black' ,'purple']
    # data = tool_belt.get_raw_data(file_p4, path)
    # sig_counts = data['sig_counts']
    # ref_counts = data['ref_counts']
    # run_ind = 40
    # avg_sig_counts = numpy.average(sig_counts[:(run_ind+1)], axis=0)
    # avg_ref_counts = numpy.average(ref_counts[:(run_ind+1)], axis=0)

    # norm_avg_sig = avg_sig_counts / numpy.average(avg_ref_counts)
    # print(list(norm_avg_sig))
    
    low_resonance = 2.7813 
    fig, ax = plt.subplots()
    for f in range(len(file_list)):
        file = file_list[f]
        data = tool_belt.get_raw_data(file, path)
        norm_avg_sig = data['norm_avg_sig']
        taus= numpy.array(data['taus'])/1e3
        dev = data['deviation_high']
        nv_sig = data['nv_sig']
        resonance_LOW = nv_sig['resonance_LOW']
        
        df = (resonance_LOW - low_resonance)*1e3 
        # print(df)
        
        
        contrast = 0.108*2
        low_pop = 1-contrast
        pop= (numpy.array(norm_avg_sig) - low_pop) / (1 - low_pop)
        ax.plot(taus, pop, 'o',color = color_list[f],  label = 'LOW resonance shifted {:.2f} MHz'.format(df))
        fit_func, popt = fit_data(taus, pop)
        print(popt)
        # popt = [0.85, 1/2, 3]
        linspaceTau = numpy.linspace(taus[0], taus[-1], 100)
        ax.plot(linspaceTau, fit_func(linspaceTau, *popt), '-',color = color_list[f],  label='fit')
        
            
    ax.set_title('Rabi SRT, {} MHz detuning'.format(dev))
    ax.set_xlabel('SRT length (us)')
    ax.set_ylabel('Population')
    ax.legend()
        
        
    # data = tool_belt.get_raw_data(file_p, path)
    # p_sig = data['norm_avg_sig']
    # data = tool_belt.get_raw_data(file_z, path)
    # z_sig = data['norm_avg_sig']
    # data = tool_belt.get_raw_data(file_m, path)
    # m_sig = data['norm_avg_sig']
    # taus= numpy.array(data['taus'])/1e3
    # dev = data['deviation_LOW']
    
    # contrast = 0.238
    # low_pop = 1-contrast
    
    # p_pop = (numpy.array(p_sig) - low_pop) / (1 - low_pop)
    # z_pop = (numpy.array(z_sig) - low_pop) / (1 - low_pop)
    # m_pop = (numpy.array(m_sig) - low_pop) / (1 - low_pop)
    
    
    # plot_pop_srt(taus, p_pop, z_pop, dev, m_pop)