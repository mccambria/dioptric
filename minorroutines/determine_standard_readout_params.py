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
import matplotlib.pyplot as plt


# region Functions


def process_raw_tags(apd_gate_channel, raw_tags, channels):
    """
    Take a raw timetag signal with tags in units of ns since the tagger
    started and convert the tags into relative times from when the gate opened
    """

    sig_tags = []
    ref_tags = []
    gate_open_channel = apd_gate_channel
    gate_close_channel = -gate_open_channel

    channels_array = np.array(channels)
    gate_open_inds = np.where(channels_array == gate_open_channel)[0]
    gate_close_inds = np.where(channels_array == gate_close_channel)[0]

    num_reps = len(gate_open_inds)
    for rep_ind in range(num_reps):
        open_ind = gate_open_inds[rep_ind]
        close_ind = gate_close_inds[rep_ind]
        open_timetag = raw_tags[open_ind]
        rep_processed_timetags = raw_tags[open_ind + 1 : close_ind]
        rep_processed_timetags = [
            val - open_timetag for val in rep_processed_timetags
        ]
        # Every even gate is sig, odd is ref
        if rep_ind % 2 == 0:
            sig_tags.extend(rep_processed_timetags)
        else:
            ref_tags.extend(rep_processed_timetags)
            

    sig_tags_arr = np.array(sig_tags, dtype=int)
    ref_tags_arr = np.array(ref_tags, dtype=int)
    return sig_tags_arr, ref_tags_arr


def plot_readout_duration_optimization(max_readout, num_reps, 
                                       sig_tags, ref_tags):
    """
    Generate two plots: 1, the total counts vs readout duration for each of 
    the spin states; 2 the SNR vs readout duration
    """
    
    fig, axes_pack = plt.subplots(1, 2, figsize=kpl.double_figsize)
    
    num_points = 100
    readouts_with_zero = np.linspace(0, max_readout, num_points)
    readouts = readouts_with_zero[1:]  # Exclude 0 ns
    integrated_sig_tags = [(sig_tags < readout).sum() for readout in readouts]
    integrated_ref_tags = [(ref_tags < readout).sum() for readout in readouts]
    snr_per_readout = []
    for sig, ref in zip(integrated_sig_tags, integrated_ref_tags):
        # Assume Poisson statistics on each count value
        sig_noise = np.sqrt(sig)
        ref_noise = np.sqrt(ref)
        snr = (ref-sig) / np.sqrt(sig_noise**2 + ref_noise**2)
        snr_per_readout.append(snr / np.sqrt(num_reps))

    ax = axes_pack[0]
    # ax.plot(readouts, integrated_sig_tags, label=r"$\ket{m_{s}=\pm 1}$")
    # ax.plot(readouts, integrated_ref_tags, label=r"$\ket{m_{s}=0}$")
    # ax.plot(readouts, integrated_sig_tags, label=r"$m_{s}=\pm 1$")
    # ax.plot(readouts, integrated_ref_tags, label=r"$m_{s}=0$")
    # ax.set_ylabel('Integrated counts')
    sig_hist, bin_edges = np.histogram(sig_tags, bins=readouts_with_zero)
    ref_hist, bin_edges = np.histogram(ref_tags, bins=readouts_with_zero)
    readout_window = round(readouts_with_zero[1] - readouts_with_zero[0])
    readout_window_sec = readout_window ** -9
    sig_rates = sig_hist / (readout_window_sec * num_reps)
    ref_rates = ref_hist / (readout_window_sec * num_reps)
    bin_centers = (readouts_with_zero[:-1] + readouts) / 2
    ax.plot(bin_centers, sig_rates, label=r"$m_{s}=\pm 1$")
    ax.plot(bin_centers, ref_rates, label=r"$m_{s}=0$")
    ax.set_ylabel('Count rate (kcps)')
    ax.set_xlabel('Readout duration (ns)')
    ax.legend()

    ax = axes_pack[1]
    ax.plot(readouts, snr)
    ax.set_xlabel('Readout duration (ns)')
    ax.set_ylabel('SNR')
    max_snr = round(max(snr),2)
    optimum_readout = round(readouts[np.argmax(snr)])
    text = f"Max SNR of {max_snr} at {optimum_readout} ns"
    ax.text(0.05, 0.95, text, transform=ax.transAxes)

    fig.tight_layout()

    return fig
    
# endregion 


def optimize_readout_duration_sub(
    cxn, nv_sig, apd_indices, num_reps, state=States.LOW
):
    
    tool_belt.reset_cfm(cxn)

    seq_file = "rabi.py"
    laser_key = "spin_laser"
    laser_name = nv_sig[laser_key]
    laser_power = tool_belt.set_laser_power(cxn, nv_sig, laser_key)
    polarization_time = nv_sig["spin_pol_dur"]
    readout = nv_sig["spin_readout_dur"]
    pi_pulse_dur = tool_belt.get_pi_pulse_dur(nv_sig[f"rabi_{state.name}"])
    seq_args = [
        pi_pulse_dur,
        polarization_time,
        readout,
        pi_pulse_dur,
        apd_indices[0],
        state.value,
        laser_name,
        laser_power,
    ]
    seq_args_string = tool_belt.encode_seq_args(seq_args)
    ret_vals = cxn.pulse_streamer.stream_load(seq_file, seq_args_string)
    period = ret_vals[0]
    period_sec = period / 10 ** 9
    
    opti_coords_list = []

    sig_gen_cxn = tool_belt.get_signal_generator_cxn(cxn, state)

    # Some initial parameters
    opti_period = 2.5 * 60  # optimize every opti_period seconds
    num_reps_per_cycle = round(opti_period / period_sec)
    print(num_reps_per_cycle)
    num_reps_remaining = num_reps
    timetags = []
    channels = []
    
    while num_reps_remaining > 0:

        # Break out of the while if the user says stop
        if tool_belt.safe_stop():
            break

        # Optimize and save the coords we found
        opti_coords = optimize.main_with_cxn(cxn, nv_sig, apd_indices)
        opti_coords_list.append(opti_coords)

        # Set up the microwaves and laser. Then load the pulse streamer
        # (must happen after optimize and iq_switch since run their
        # own sequences)
        tool_belt.set_filter(cxn, nv_sig, laser_key)
        laser_power = tool_belt.set_laser_power(cxn, nv_sig, laser_key)
        sig_gen_cxn.set_freq(nv_sig[f"resonance_{state.name}"])
        sig_gen_cxn.set_amp(nv_sig[f"uwave_power_{state.name}"])
        sig_gen_cxn.uwave_on()
        
        # Load the APD
        cxn.apd_tagger.start_tag_stream(apd_indices)
        cxn.apd_tagger.clear_buffer()

        # Run the sequence
        if num_reps_remaining > num_reps_per_cycle:
            num_reps_to_run = int(num_reps_per_cycle)
        else:
            num_reps_to_run = int(num_reps_remaining)
        print(num_reps_to_run)
        cxn.pulse_streamer.stream_immediate(
            seq_file, num_reps_to_run, seq_args_string
        )

        # Get the counts
        print("Data coming in")
        ret_vals = cxn.apd_tagger.read_tag_stream(1)
        print("Data collected")
        buffer_timetags, buffer_channels = ret_vals

        cxn.apd_tagger.stop_tag_stream()
        
        # We don't care about picosecond resolution here, so just round to ns
        # We also don't care about the offset value, so subtract that off
        if len(timetags) == 0:
            offset = np.int64(buffer_timetags[0])
        buffer_timetags = [
            int((np.int64(val) - offset) / 1e3) for val in buffer_timetags
        ]
        timetags.extend(buffer_timetags)
        channels.extend(buffer_channels)

        cxn.apd_tagger.stop_tag_stream()

        num_reps_remaining -= num_reps_per_cycle

    return timetags, channels, opti_coords_list
    

def optimize_readout_duration(cxn, nv_sig, apd_indices, num_reps, 
                              state=States.LOW):
    
    max_readout = nv_sig["spin_readout_dur"]

    # Assume a common gate for both APDs
    apd_gate_channel = tool_belt.get_apd_gate_channel(cxn, apd_indices[0])
    timetags, channels, opti_coords_list = optimize_readout_duration_sub(
        cxn, nv_sig, apd_indices, num_reps, state
    )

    # Process the raw tags
    sig_tags, ref_tags = process_raw_tags(apd_gate_channel, timetags, channels)
    
    # Analyze and plot
    fig = plot_readout_duration_optimization(max_readout, num_reps, sig_tags, ref_tags)

    # Save the data
    timestamp = tool_belt.get_time_stamp()
    raw_data = {
        "timestamp": timestamp,
        "nv_sig": nv_sig,
        "nv_sig-units": tool_belt.get_nv_sig_units(),
        "opti_coords_list": opti_coords_list,
        "opti_coords_list-units": "V",
        "state": state.name,
        "num_reps": num_reps,
        "apd_gate_channel": apd_gate_channel,
        "sig_tags": sig_tags.tolist(),
        "ref_tags": ref_tags.tolist(),
    }
    file_path = tool_belt.get_file_path(__file__, timestamp, nv_sig['name'])
    tool_belt.save_figure(fig, file_path)
    tool_belt.save_raw_data(raw_data, file_path)


# region Main

def main(nv_sig, apd_indices, num_reps, 
         max_readouts, powers=None, filters=None, state=States.LOW):
    
    with labrad.connect() as cxn:
        main_with_cxn(cxn, nv_sig, apd_indices, num_reps, 
                      max_readouts, powers, filters, state)

def main_with_cxn(cxn, nv_sig, apd_indices, num_reps, 
              max_readouts, powers=None, filters=None, state=States.LOW):
    """
    Determine optimized SNR for each pairing of max_readout, power/filter.
    Ie we'll test max_readout[i] and power[i]/filter[i] at the same time. For 
    each experiment i, we'll just run one data set under the max_readout. Then
    we'll determine the optimized readout in post.
    Either powers or filters should be populated but not both.
    """
    
    kpl.init_kplotlib

    # Start 'Press enter to stop...'
    tool_belt.init_safe_stop()
    
    num_exps = len(max_readouts)
    
    for ind in range(num_exps): 
        
        # Break out of the while if the user says stop
        if tool_belt.safe_stop():
            break
        
        adjusted_nv_sig = copy.deepcopy(nv_sig)
        if max_readouts is not None:
            adjusted_nv_sig["spin_readout_dur"] = int(max_readouts[ind])
        if powers is not None:
            adjusted_nv_sig["spin_laser_power"] = powers[ind]
        if filters is not None:
            adjusted_nv_sig["spin_laser_filter"] = filters[ind]
            
        optimize_readout_duration(cxn, adjusted_nv_sig, apd_indices, 
                                  num_reps, state)

# endregion

if __name__ == '__main__':
    
    file = ""
    data = tool_belt.get_raw_data(file)
    
    max_readout = data["max_readout"]
    sig_tags = data["sig_tags"]
    ref_tags = data["ref_tags"]
    
    plot_readout_duration_optimization(max_readout, sig_tags, ref_tags)

    plt.show(block=True)
