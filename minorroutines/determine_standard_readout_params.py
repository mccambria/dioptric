# -*- coding: utf-8 -*-
"""
Determine optimimum readout duration and power for standard spin state
readout under green illumination. Works by comparing readout after 
intialization into ms=0 vs ms=+/-1. 

Created on July 13th, 2022

@author: mccambria
"""


# %% Imports


import utils.tool_belt as tool_belt
import majorroutines.optimize as optimize
import numpy as np
import os
import time
import labrad
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from random import shuffle
from utils.tool_belt import States
import utils.kplotlib as kpl
import copy


# region Functions


def integrate_tags(tags, integral_lim):
    """Add up the number of tags that are below integral_lim"""


def plot_readout_duration_optimization(max_readout, sig_tags, ref_tags):
    """
    Generate two plots: 1, the total counts vs readout duration for each of 
    the spin states; 2 the SNR vs readout duration
    """
    
    fig, axes_pack = plt.subplots(1, 2, kpl.figsize)
    ax1, ax2 = axes_pack
    
    readouts = np.linspace(0, max_readout, 100)
    integrated_sig_tags = [(sig_tags < readout).sum() for readout in readouts]
    integrated_ref_tags = [(ref_tags < readout).sum() for readout in readouts]

    ax1.plot(readouts, integrated_sig_tags, 'r-')
    ax1.plot(readouts, integrated_sig_tags, 'r-')
    ax.plot(list(range(0, num_steps)), avg_ref_counts, 'g-')
    ax.set_xlabel('Readout duration (ns)')
    ax.set_ylabel('Integrated counts')

    ax = axes_pack[1]
    ax.plot(list(range(0, num_steps)), norm_avg_sig, 'b-')
    ax.set_title('Normalized Signal With Varying Microwave Duration')
    ax.set_xlabel('num_run')
    ax.set_ylabel('Normalized contrast')

    raw_fig.canvas.draw()
    # fig.set_tight_layout(True)
    raw_fig.canvas.flush_events()

    return raw_fig
    
# endregion 

# %% Main, this allows any one of the inputs to be varied, like the readout
#    time or the nd_filter

def snr_measurement(nv_sig, readout_time, 
                    num_steps, num_reps, num_runs, apd_indices,
                    do_plot, save_raw_data):

    with labrad.connect() as cxn:
        sig_to_noise_ratio = snr_measurement_with_cxn(cxn, nv_sig, readout_time,
                                                      num_steps, num_reps, num_runs, 
                                                      apd_indices, do_plot, save_raw_data)
        return sig_to_noise_ratio
    
    
def snr_measurement_with_cxn(cxn, nv_sig, readout_time,
                             num_steps, num_reps, num_runs, apd_indices, 
                             do_plot = False, save_raw_data = False):


    # %% Get the starting time of the function

    startFunctionTime = time.time()

    # %% Initial calculations and setup

    # Assume the low state
    state = States.LOW
    # state = States.HIGH

    config = tool_belt.get_config_dict(cxn)

    # Define some times (in ns)
    exp_dur = 5 * 10**3
    polarization_dur = nv_sig["spin_pol_dur"]
    laser_name = nv_sig["spin_laser"]
    pi_pulse = round(nv_sig['rabi_{}'.format(state.name)] / 2)

    # The two parameters we currently alter
    readout_time = int(readout_time)

    file_name = os.path.basename(__file__)

    # Set up our data structure, an array of NaNs that we'll fill
    # incrementally. NaNs are ignored by matplotlib, which is why they're
    # useful for us here.
    # We define 2D arrays, with the horizontal dimension for the frequency and
    # the veritical dimension for the index of the run.
    sig_counts = np.empty([num_runs, num_steps], dtype=np.uint32)
    ref_counts = np.copy(sig_counts)

    # %% Make some lists and variables to save at the end

    opti_coords_list = []

    # Create a list of indices to step through the taus. This will be shuffled
    ind_list = list(range(0, num_steps))

    # %% Collect the data

    for run_ind in range(num_runs):

#        print('Run index: {}'. format(run_ind))

        # Break out of the while if the user says stop
        if tool_belt.safe_stop():
            break

        # Optimize
        opti_coords = optimize.main_with_cxn(cxn, nv_sig, apd_indices)
        opti_coords_list.append(opti_coords)

        tool_belt.set_filter(cxn, nv_sig, "collection")
        tool_belt.set_filter(cxn, nv_sig, "spin_laser")

        sig_gen_cxn = tool_belt.get_signal_generator_cxn(cxn, state)
        sig_gen_cxn.set_freq(nv_sig['resonance_{}'.format(state.name)])
        sig_gen_cxn.set_amp(nv_sig['uwave_power_{}'.format(state.name)])
        sig_gen_cxn.uwave_on()

        # Load the APD stream
        cxn.apd_tagger.start_tag_stream(apd_indices)

        # Shuffle the list of indices to use for stepping through the taus
        shuffle(ind_list)

        for ind in ind_list:

            # Break out of the while if the user says stop
            if tool_belt.safe_stop():
                break

            # Stream the sequence
            # polarization_dur, exp_dur, aom_delay, uwave_delay,
            # readout_dur, pi_pulse, apd_index, uwave_gate_index
            seq_args = [polarization_dur, exp_dur, readout_time, 
                        pi_pulse, apd_indices[0], state.value, laser_name]
            seq_args_string = tool_belt.encode_seq_args(seq_args)

            cxn.pulse_streamer.stream_immediate(file_name, num_reps, seq_args_string)

            # Get the counts
            new_counts = cxn.apd_tagger.read_counter_separate_gates(1)

            sample_counts = new_counts[0]

            # signal counts are even - get every second element starting from 0
            sig_gate_counts = sample_counts[0::2]
            sig_counts[run_ind, ind] = sum(sig_gate_counts)

            # ref counts are odd - sample_counts every second element starting from 1
            ref_gate_counts = sample_counts[1::2]
            ref_counts[run_ind, ind] = sum(ref_gate_counts)

        tool_belt.reset_cfm(cxn)

    # %% Average the counts over the iterations

    avg_sig_counts = np.average(sig_counts, axis=0)
    avg_ref_counts = np.average(ref_counts, axis=0)

    # %% Calculate the statistics of the run

    norm_avg_sig = avg_ref_counts - avg_sig_counts

    sig_stat = np.average(norm_avg_sig)

    st_dev_stat = np.std(norm_avg_sig)

    sig_to_noise_ratio = sig_stat / st_dev_stat

    print('Readout Time: {} ns \nSignal: {:.3f} \nNoise: {:.3f} \nSNR: {:.1f}\n '.format(readout_time, \
          sig_stat, st_dev_stat, sig_to_noise_ratio))

    # %% Plot the counts
    if do_plot:

        raw_fig = raw_plot(num_steps, avg_sig_counts, avg_ref_counts, norm_avg_sig)

    # %% Save the data
    if save_raw_data:

        endFunctionTime = time.time()

        timeElapsed = endFunctionTime - startFunctionTime

        timestamp = tool_belt.get_time_stamp()

        raw_data = {'timestamp': timestamp,
                    'timeElapsed': timeElapsed,
                    'nv_sig': nv_sig,
                    'nv_sig-units': tool_belt.get_nv_sig_units(),
                    'readout_time': readout_time,
                    'readout_time-unit': 'ns',
                    'sig_stat': sig_stat,
                    'st_dev_stat': st_dev_stat,
                    'sig_to_noise_ratio': sig_to_noise_ratio,
                    'nv_sig': nv_sig,
                    'opti_coords_list': opti_coords_list,
                    'coords-units': 'V',
                    'state': state,
                    'num_steps': num_steps,
                    'num_reps': num_reps,
                    'num_runs': num_runs,
                    'sig_counts': sig_counts.astype(int).tolist(),
                    'sig_counts-units': 'counts',
                    'ref_counts': ref_counts.astype(int).tolist(),
                    'ref_counts-units': 'counts',
                    'norm_avg_sig': norm_avg_sig.astype(float).tolist(),
                    'norm_avg_sig-units': 'arb'}

        file_path = tool_belt.get_file_path(__file__, timestamp, nv_sig['name'])
        tool_belt.save_figure(raw_fig, file_path)
        tool_belt.save_raw_data(raw_data, file_path)

    return sig_to_noise_ratio

# %%

def optimize_readout(nv_sig, apd_indices, num_reps, num_runs, 
                     state=States.LOW):

    # don't plot or save each individual raw data of the snr
    do_plot = False
    save_raw_data = False

    # Create an empty list to fill with snr
    snr_list = []
    
    with labrad.connect() as cxn:
        laser_power = tool_belt.set_laser_power(cxn, nv_sig, "spin_laser")

    # Step thru the readout times and take a snr measurement
    for readout_ind_time in readout_time_list:

        # Break out of the while if the user says stop
        if tool_belt.safe_stop():
            break

        readout_time = readout_ind_time

        snr_value = snr_measurement(nv_sig, readout_time, 
                                    num_steps, num_reps, num_runs,
                                    apd_indices, do_plot, save_raw_data)

        snr_list.append(snr_value)


    # Prepare the plot:
    snr_fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    ax.plot(readout_time_list, snr_list, 'ro', label = 'data')
    ax.set_xlabel('Readout time (ns)')
    ax.set_ylabel('Signal-to-noise ratio')
    ax.set_title('Optimize readout window at {} mW'.format(laser_power))

    # Fit the data to a parabola
    offset = 130
    amplitude = 10
    readout_time_guess = np.average(readout_range)

    init_guess_list = [offset, amplitude, readout_time_guess]
    try:
        popt = fit_parabola(readout_time_list, snr_list, init_guess_list)

        # If the fit works, plot the fitted curve
        linspace_time = np.linspace(readout_range[0], readout_range[-1], num=1000)
        ax.plot(linspace_time, parabola(linspace_time,*popt), 'b-', label = 'fit')

        text = ('Optimal readout time = {:.1f} ns'.format(popt[2]))

        props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
        ax.text(0.70, 0.05, text, transform=ax.transAxes, fontsize=12,
                                verticalalignment="top", bbox=props)

    except Exception as e:
        # Just print the error if the fit failed
        print(e)

    # Plot the figure
    ax.legend()
    snr_fig.canvas.draw()
    snr_fig.canvas.flush_events()

    # Save the data
    timestamp = tool_belt.get_time_stamp()
    snr_data = {'timestamp': timestamp,
                'nv_sig': nv_sig,
                'nv_sig-units': tool_belt.get_nv_sig_units(),
                'num_steps': num_steps,
                'num_reps': num_reps,
                'num_runs': num_runs,
                'snr_list': snr_list,
                'readout_time_list': readout_time_list.tolist(),
                }

    file_path = tool_belt.get_file_path(__file__, timestamp, nv_sig['name'])
    tool_belt.save_figure(snr_fig, file_path)
    tool_belt.save_raw_data(snr_data, file_path)


# %%

def main(nv_sig, apd_indices, num_reps, num_runs, 
         max_readouts, powers=None, filters=None, state=States.LOW):
    
    with labrad.connect() as cxn:
        main_with_cxn(cxn, nv_sig, apd_indices, num_reps, num_runs, 
                      max_readouts, powers, filters, state)

def main_with_cxn(cxn, nv_sig, apd_indices, num_reps, num_runs, 
              max_readouts, powers=None, filters=None, state=States.LOW):
    """
    Determine optimized SNR for each pairing of max_readout, power/filter.
    Ie we'll test max_readout[i] and power[i]/filter[i] at the same time. For 
    each experiment i, we'll just run one data set under the max_readout. Then
    we'll determine the optimized readout in post.
    Either powers or filters should be populated but not both.
    """

    # Start 'Press enter to stop...'
    tool_belt.init_safe_stop()

    num_exps = len(max_readouts)
    
    for ind in range(num_exps): 
        
        # Break out of the while if the user says stop
        if tool_belt.safe_stop():
            break
        
        max_readout = max_readouts[ind]
        if powers is not None:
            power = powers[ind]
        if filters is not None:
            filt = filters[ind]
        
        adjusted_nv_sig = copy.deepcopy(nv_sig)
        adjusted_nv_sig["spin_readout_dur"] = max_readout
        adjusted_nv_sig["spin_laser_power"] = power
        adjusted_nv_sig["spin_laser_filter"] = filt

        optimize_readout(nv_sig, max_readout, num_reps, num_runs, state)

# %%

if __name__ == '__main__':

    pass
