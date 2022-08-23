# -*- coding: utf-8 -*-
"""
Rabi flopping routine. Sweeps the pulse duration of a fixed uwave frequency.

Created on Tue Apr 23 11:49:23 2019

@author: mccambria
"""


# %% Imports


import utils.tool_belt as tool_belt
import majorroutines.optimize as optimize
import numpy
from numpy import pi
import os
import time
import matplotlib.pyplot as plt
from random import shuffle
from scipy.optimize import curve_fit
import labrad


# %% Constants


pi_on_2 = numpy.pi / 2


# %% Functions


def rotate(state, axis, angle):
    
    return numpy.matmul(rotation_matrix(axis, angle), state)


def rotation_matrix(axis, angle):
    
    if axis == 'x':
        return numpy.array([[numpy.cos(angle/2), -1J*numpy.sin(angle/2)],
                            [-1J*numpy.sin(angle/2), numpy.cos(angle/2)]])
    if axis == 'y':
        return numpy.array([[numpy.cos(angle/2), -numpy.sin(angle/2)],
                            [numpy.sin(angle/2), numpy.cos(angle/2)]])
    if axis == 'z':
        return numpy.array([[numpy.exp(-1J*angle/2), 0],
                            [0, numpy.exp(1J*angle/2)]])


def simulate(drive_res, drive_rabi, nv_res, nv_rabi, num_pulses):
    
    pulses_list = list(range(num_pulses+1))
    pops = []
    
    num_samples = 50
    # nv_rabi_distr = numpy.random.normal(100, 8, num_samples)
    nv_res_distr = numpy.random.normal(2.870, 0.0005, num_samples)
    # drive_rabi_distr = numpy.random.normal(100, 8, num_samples)
    
    for el in pulses_list:
        
        sum_pops = 0
        # for nv_rabi in nv_rabi_distr:
        for nv_res in nv_res_distr:
        # for drive_rabi in drive_rabi_distr:
            val = simulate_single(drive_res, drive_rabi,
                                  nv_res, nv_rabi, el)
            sum_pops += val
            
        avg_pops = sum_pops / num_samples
        pops.append(avg_pops)
    
    fig, ax = plt.subplots(figsize=(5, 5))
    fig.set_tight_layout(True)
    ax.set_ylim([0,1])
    ax.plot(pulses_list, pops)
    


def simulate_single(drive_res, drive_rabi, nv_res, nv_rabi, num_pulses):
    
    detuning = drive_res - nv_res
    nv_rabi = numpy.sqrt(detuning**2 + (1/drive_rabi)**2)
    nv_rabi = 1/nv_rabi
    
    drive_prop = drive_rabi / nv_rabi
    
    # Start at the top of the Bloch sphere
    state = numpy.array([1,0])
    
    # for i in range(num_pulses):
    
    #     state = rotate(state, 'x', drive_prop * pi/2)
    #     state = rotate(state, 'z', drive_rabi * pi/2 * detuning)
        
    #     # state = rotate(state, 'z', drive_rabi * pi/2 * detuning)
    #     state = rotate(state, 'y', drive_prop * pi)
    #     state = rotate(state, 'z', drive_rabi * pi * detuning)
        
    #     state = rotate(state, 'z', drive_rabi * pi/2 * detuning)
    #     state = rotate(state, 'x', drive_prop * pi/2)
    
    for i in range(num_pulses):
    
        
        if i // 2 == 1:
            ax1 = 'x'
            ax2 = 'y'
        else:
            ax1 = 'y'
            ax2 = 'x'
            
        state = rotate(state, ax1, drive_prop * pi/2)
        state = rotate(state, 'z', drive_rabi * pi/2 * detuning)
        
        # state = rotate(state, 'z', drive_rabi * pi/2 * detuning)
        state = rotate(state, ax2, drive_prop * pi)
        state = rotate(state, 'z', drive_rabi * pi * detuning)
        
        state = rotate(state, 'z', drive_rabi * pi/2 * detuning)
        state = rotate(state, ax1, drive_prop * pi/2)
    
    # for i in range(num_pulses):
        
    #     if i // 2 == 1:
    #         state = rotate(state, 'x', drive_prop * pi/2)
    #         state = rotate(state, 'y', drive_prop * pi)
    #         state = rotate(state, 'x', drive_prop * pi/2)
    #     else:
    #         state = rotate(state, 'y', drive_prop * pi/2)
    #         state = rotate(state, 'x', drive_prop * pi)
    #         state = rotate(state, 'y', drive_prop * pi/2)
            
        
    excited_component = state[1]
    excited_projection = numpy.real(numpy.conj(excited_component) * excited_component)
    return 1-excited_projection
    


# %% Main


def main(nv_sig, apd_indices, state,
         max_num_pi_pulses, num_reps, num_runs, iq_delay=None):

    with labrad.connect() as cxn:
        main_with_cxn(cxn, nv_sig, apd_indices, state,
                      max_num_pi_pulses, num_reps, num_runs, iq_delay)
    
    
def main_with_cxn(cxn, nv_sig, apd_indices, state,
                  max_num_pi_pulses, num_reps, num_runs, iq_delay=None):

    tool_belt.reset_cfm(cxn)

    # %% Get the starting time of the function, to be used to calculate run time

    startFunctionTime = time.time()
    start_timestamp = tool_belt.get_time_stamp()

    # %% Initial calculations and setup
    
    num_reps = int(num_reps)

    uwave_freq = nv_sig['resonance_{}'.format(state.name)]
    uwave_power = nv_sig['uwave_power_{}'.format(state.name)]
    uwave_pi_pulse = round(nv_sig['rabi_{}'.format(state.name)] / 2)
    # uwave_pi_pulse = round(3 * nv_sig['rabi_{}'.format(state.name)] / 4)
    # uwave_pi_pulse = round(0.70 * nv_sig['rabi_{}'.format(state.name)])
    # uwave_pi_pulse = 0
    uwave_pi_on_2_pulse = round(nv_sig['rabi_{}'.format(state.name)] / 4)
    
    laser_key = 'spin_laser'
    laser_name = nv_sig[laser_key]
    laser_power = tool_belt.set_laser_power(cxn, nv_sig, laser_key)

    polarization_time = nv_sig['spin_pol_dur'] 
    if iq_delay is None:
        iq_delay = tool_belt.get_registry_entry(cxn, 'iq_delay', ['', 'Config', 'Microwaves'])
    gate_time = nv_sig['spin_readout_dur']
    # uwave_delay_time = 15
    # signal_wait_time = 1000

    # Analyze the sequence
    # file_name = os.path.basename(__file__)
    file_name = 'discrete_rabi2.py'
    seq_args = [polarization_time, iq_delay,
                gate_time, uwave_pi_pulse, uwave_pi_on_2_pulse,
                0, max_num_pi_pulses,
                apd_indices[0], state.value, laser_name, laser_power]
    # print(seq_args)
    # return
    seq_args_string = tool_belt.encode_seq_args(seq_args)
    cxn.pulse_streamer.stream_load(file_name, seq_args_string)

    # Set up our data structure, an array of NaNs that we'll fill
    # incrementally. NaNs are ignored by matplotlib, which is why they're
    # useful for us here.
    # We define 2D arrays, with the horizontal dimension for the frequency and
    # the veritical dimension for the index of the run.
    num_steps = max_num_pi_pulses + 1 
    sig_counts = numpy.empty([num_runs, num_steps], dtype=numpy.float32)
    sig_counts[:] = numpy.nan
    ref_counts = numpy.copy(sig_counts)
    # norm_avg_sig = numpy.empty([num_runs, num_steps])

    # %% Make some lists and variables to save at the end

    opti_coords_list = []
    pi_ind_master_list = [[] for i in range(num_runs)]

    # Create a list of indices to step through the taus. This will be shuffled
    pi_ind_list_ordered = list(range(0, num_steps))
    pi_ind_list = list(range(0, num_steps))

    # %% Collect the data

    # Start 'Press enter to stop...'
    tool_belt.init_safe_stop()

    for run_ind in range(num_runs):

        print('Run index: {}'. format(run_ind))

        # Break out of the while if the user says stop
        if tool_belt.safe_stop():
            break

        # Optimize
        opti_coords = optimize.main_with_cxn(cxn, nv_sig, apd_indices)
        opti_coords_list.append(opti_coords)

        # Apply the microwaves and set up IQ
        sig_gen_cxn = tool_belt.get_signal_generator_cxn(cxn, state)
        sig_gen_cxn.set_freq(uwave_freq)
        sig_gen_cxn.set_amp(uwave_power)
        sig_gen_cxn.load_iq()
        sig_gen_cxn.uwave_on()
        cxn.arbitrary_waveform_generator.load_knill()

        # TEST for split resonance
#        sig_gen_cxn = cxn.signal_generator_bnc835
#        sig_gen_cxn.set_freq(uwave_freq + 0.008)
#        sig_gen_cxn.set_amp(uwave_power)
#        sig_gen_cxn.uwave_on()

        # Load the APD
        cxn.apd_tagger.start_tag_stream(apd_indices)

        # Shuffle the list of indices to use for stepping through the taus
        shuffle(pi_ind_list)

        for pi_ind in pi_ind_list:
#        for tau_ind in range(len(taus)):
#            print('Tau: {} ns'. format(taus[tau_ind]))
            # Break out of the while if the user says stop
            if tool_belt.safe_stop():
                break

            # add the tau indexxes used to a list to save at the end
            pi_ind_master_list[run_ind].append(pi_ind)

            # Stream the sequence
            seq_args = [polarization_time, iq_delay,
                        gate_time, uwave_pi_pulse, uwave_pi_on_2_pulse,
                        pi_ind, max_num_pi_pulses,
                        apd_indices[0], state.value, laser_name, laser_power]
            # print(seq_args)
            # return
            seq_args_string = tool_belt.encode_seq_args(seq_args)
            cxn.pulse_streamer.stream_immediate(file_name, num_reps,
                                                seq_args_string)

            # Get the counts
            new_counts = cxn.apd_tagger.read_counter_separate_gates(1)

            sample_counts = new_counts[0]

            # signal counts are even - get every second element starting from 0
            sig_gate_counts = sample_counts[0::2]
            sum_sig_gate_counts = sum(sig_gate_counts)
            sig_counts[run_ind, pi_ind] = sum_sig_gate_counts

            # ref counts are odd - sample_counts every second element starting from 1
            ref_gate_counts = sample_counts[1::2]
            sum_ref_gate_counts = sum(ref_gate_counts)
            ref_counts[run_ind, pi_ind] = sum_ref_gate_counts

        cxn.apd_tagger.stop_tag_stream()

        # %% Save the data we have incrementally for long measurements

        raw_data = {'start_timestamp': start_timestamp,
                    'nv_sig': nv_sig,
                    'nv_sig-units': tool_belt.get_nv_sig_units(),
                    'uwave_freq': uwave_freq,
                    'uwave_freq-units': 'GHz',
                    'uwave_power': uwave_power,
                    'uwave_power-units': 'dBm',
                    'max_num_pi_pulses': max_num_pi_pulses,
                    'state': state.name,
                    'num_steps': num_steps,
                    'num_reps': num_reps,
                    'num_runs': num_runs,
                    'pi_ind_master_list': pi_ind_master_list,
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
    err_sig_counts = numpy.std(sig_counts, axis=0, ddof = 1) / numpy.sqrt(num_runs)
    avg_ref_counts = numpy.average(ref_counts, axis=0)
    err_ref_counts = numpy.std(ref_counts, axis=0, ddof = 1) / numpy.sqrt(num_runs)

    # %% Calculate the Rabi data, signal / reference over different Tau

    norm_avg_sig = avg_sig_counts / avg_ref_counts
    norm_avg_sig_err = norm_avg_sig * numpy.sqrt((err_sig_counts/avg_sig_counts)**2 + (err_ref_counts/avg_ref_counts)**2)

    # %% Plot the Rabi signal

    raw_fig, axes_pack = plt.subplots(1, 2, figsize=(17, 8.5))

    ax = axes_pack[0]
    ax.errorbar(pi_ind_list_ordered, avg_sig_counts, fmt='r-', yerr=err_sig_counts)
    ax.errorbar(pi_ind_list_ordered, avg_ref_counts, fmt='g-', yerr=err_ref_counts)
    # ax.plot(tauArray, countsBackground, 'o-')
    ax.set_xlabel('Number pi pulses')
    ax.set_ylabel('Counts')

    ax = axes_pack[1]
    ax.errorbar(pi_ind_list_ordered , norm_avg_sig, fmt='b-', yerr=norm_avg_sig_err)
    ax.set_title('Normalized Signal With Varying Microwave Duration')
    ax.set_xlabel('Number pi pulses')
    ax.set_ylabel('Contrast (arb. units)')

    raw_fig.canvas.draw()
    raw_fig.set_tight_layout(True)
    raw_fig.canvas.flush_events()

    # %% Clean up and save the data

    tool_belt.reset_cfm(cxn)

    endFunctionTime = time.time()

    timeElapsed = endFunctionTime - startFunctionTime

    timestamp = tool_belt.get_time_stamp()

    raw_data = {'timestamp': timestamp,
                'timeElapsed': timeElapsed,
                'timeElapsed-units': 's',
                'iq_delay': iq_delay,
                'nv_sig': nv_sig,
                'nv_sig-units': tool_belt.get_nv_sig_units(),
                'uwave_freq': uwave_freq,
                'uwave_freq-units': 'GHz',
                'uwave_power': uwave_power,
                'uwave_power-units': 'dBm',
                'max_num_pi_pulses': max_num_pi_pulses,
                'state': state.name,
                'num_steps': num_steps,
                'num_reps': num_reps,
                'num_runs': num_runs,
                'pi_ind_master_list': pi_ind_master_list,
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
    tool_belt.save_raw_data(raw_data, file_path)


# %% Run the file


if __name__ == '__main__':
    
    # drive_res, drive_rabi, nv_res, nv_rabi, num_pulses
    # print(simulate_single(2.87, 90, 2.87, 100, 1))
    
    simulate(2.9592, 49.5, 2.9592, 49.5, 8)
    #drive_res, drive_rabi, nv_res, nv_rabi, num_pulses