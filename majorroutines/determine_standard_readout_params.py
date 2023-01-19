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
import numpy as np
import os
import time
import labrad
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from random import shuffle
from utils.tool_belt import States
import utils.kplotlib as kpl
from utils.kplotlib import KplColors
import copy
import matplotlib.pyplot as plt
import majorroutines.optimize as optimize


# region Functions


def process_raw_tags(apd_gate_channel, raw_tags, channels):
    """
    Take a raw timetag signal with tags in units of ns since the tagger
    started and convert the tags into relative times from when the gate opened.
    Assumes the only channels are apds and apd gates
    """
    # at this point it looks like one big samples of all the reps. So we just loop through all the reps

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
        rep_processed_timetags = [val - open_timetag for val in rep_processed_timetags]
        # Every even gate is sig, odd is ref
        if rep_ind % 2 == 0:
            sig_tags.extend(rep_processed_timetags)
        else:
            ref_tags.extend(rep_processed_timetags)

    sig_tags_arr = np.array(sig_tags, dtype=int)
    ref_tags_arr = np.array(ref_tags, dtype=int)
    sorted_sig_tags = np.sort(sig_tags_arr)
    sorted_ref_tags = np.sort(ref_tags_arr)
    return sorted_sig_tags, sorted_ref_tags


def plot_readout_duration_optimization(max_readout, num_reps, sig_tags, ref_tags):
    """Generate two plots: 1, the total counts vs readout duration for each of
    the spin states; 2 the SNR vs readout duration
    """

    kpl.init_kplotlib()

    fig, axes_pack = plt.subplots(1, 2, figsize=kpl.double_figsize)

    num_points = 50
    readouts_with_zero = np.linspace(0, max_readout, num_points + 1)
    readouts = readouts_with_zero[1:]  # Exclude 0 ns

    # Integrate up the tags that fell before each readout duration under test
    integrated_sig_tags = []
    integrated_ref_tags = []
    zip_iter = zip(
        (sig_tags, ref_tags), (integrated_sig_tags.append, integrated_ref_tags.append)
    )
    for sorted_tags, integrated_append in zip_iter:
        current_readout_ind = 0
        current_readout = readouts[current_readout_ind]
        for ind in range(len(sorted_tags)):
            # Cycle through readouts until the tag falls within the readout or
            # we run out of readouts
            while sorted_tags[ind] > current_readout:
                integrated_append(ind)
                current_readout_ind += 1
                if current_readout_ind == num_points:
                    break
                current_readout = readouts[current_readout_ind]
            if current_readout_ind == num_points:
                break
        # If we got to the end of the counts and there's still readouts left
        # (eg the experiment went dark for some reason), pad with the final ind
        while current_readout_ind < num_points:
            integrated_append(ind)
            current_readout_ind += 1

    # Calculate the snr per readout for each readout duration
    snr_per_readouts = []
    for sig, ref in zip(integrated_sig_tags, integrated_ref_tags):
        # Assume Poisson statistics on each count value
        sig_noise = np.sqrt(sig)
        ref_noise = np.sqrt(ref)
        snr = (ref - sig) / np.sqrt(sig_noise**2 + ref_noise**2)
        snr_per_readouts.append(snr / np.sqrt(num_reps))

    ax = axes_pack[0]
    sig_hist, bin_edges = np.histogram(sig_tags, bins=readouts_with_zero)
    ref_hist, bin_edges = np.histogram(ref_tags, bins=readouts_with_zero)
    readout_window = round(readouts_with_zero[1] - readouts_with_zero[0])
    readout_window_sec = readout_window * 10**-9
    sig_rates = sig_hist / (readout_window_sec * num_reps * 1000)
    ref_rates = ref_hist / (readout_window_sec * num_reps * 1000)
    bin_centers = (readouts_with_zero[:-1] + readouts) / 2
    kpl.plot_line(
        ax, bin_centers, sig_rates, color=KplColors.GREEN, label=r"$m_{s}=\pm 1$"
    )
    kpl.plot_line(ax, bin_centers, ref_rates, color=KplColors.RED, label=r"$m_{s}=0$")
    ax.set_ylabel("Count rate (kcps)")
    ax.set_xlabel("Time since readout began (ns)")
    ax.legend()

    ax = axes_pack[1]
    kpl.plot_line(ax, readouts, snr_per_readouts)
    ax.set_xlabel("Readout duration (ns)")
    ax.set_ylabel("SNR per sqrt(readout)")
    max_snr = tool_belt.round_sig_figs(max(snr_per_readouts), 3)
    optimum_readout = round(readouts[np.argmax(snr_per_readouts)])
    text = f"Max SNR: {max_snr} at {optimum_readout} ns"
    kpl.anchored_text(ax, text, kpl.Loc.LOWER_LEFT)

    return fig


# endregion


def optimize_readout_duration_sub(cxn, nv_sig, num_reps, state=States.LOW):

    tool_belt.reset_cfm(cxn)
    tagger_server = tool_belt.get_server_tagger(cxn)
    pulsegen_server = tool_belt.get_server_pulse_gen(cxn)

    seq_file = "rabi.py"

    ### the opx needs a specific rabi time tagging sequence because the normal rabi sequence doesn't record time tags
    if pulsegen_server.name == 'QM_opx':
        seq_file = "rabi_time_tagging.py"

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
        state.value,
        laser_name,
        laser_power,
    ]
    # print(seq_args)
    # return
    seq_args_string = tool_belt.encode_seq_args(seq_args)
    ret_vals = pulsegen_server.stream_load(seq_file, seq_args_string)
    period = ret_vals[0]
    period_sec = period / 10**9

    opti_coords_list = []

    sig_gen_cxn = tool_belt.get_server_sig_gen(cxn, state)

    opti_period = 0.25 * 60  # Optimize every opti_period seconds

    # Some initial parameters
    num_reps_per_cycle = round(opti_period / period_sec)
    num_reps_remaining = num_reps
    timetags = []
    channels = []

    while num_reps_remaining > 0:

        # Break out of the while if the user says stop
        if tool_belt.safe_stop():
            break

        # Optimize and save the coords we found
        opti_coords = optimize.main_with_cxn(cxn, nv_sig)
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
        tagger_server.start_tag_stream()
        tagger_server.clear_buffer()

        # Run the sequence
        if num_reps_remaining > num_reps_per_cycle:
            num_reps_to_run = int(num_reps_per_cycle)
        else:
            num_reps_to_run = int(num_reps_remaining)
        pulsegen_server.stream_immediate(seq_file, num_reps_to_run, seq_args_string)

        # Get the counts
        print("Data coming in")
        ret_vals = tagger_server.read_tag_stream(1)
        print("Data collected")
        buffer_timetags, buffer_channels = ret_vals

        tagger_server.stop_tag_stream()

        # We don't care about picosecond resolution here, so just round to ns
        # We also don't care about the offset value, so subtract that off
        if len(timetags) == 0:
            offset = np.int64(buffer_timetags[0])
        buffer_timetags = [
            int((np.int64(val) - offset) / 1e3) for val in buffer_timetags
        ]
        timetags.extend(buffer_timetags)
        channels.extend(buffer_channels)

        tagger_server.stop_tag_stream()

        num_reps_remaining -= num_reps_per_cycle

    return timetags, channels, opti_coords_list


def optimize_readout_duration(cxn, nv_sig, num_reps, state=States.LOW):

    max_readout = nv_sig["spin_readout_dur"]

    # Get gate channels for apds
    apd_wiring = tool_belt.get_tagger_wiring(cxn)
    apd_gate_channel = apd_wiring['di_apd_gate']
    timetags, channels, opti_coords_list = optimize_readout_duration_sub(
        cxn, nv_sig, num_reps, state
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
        "nv_sig-units": tool_belt.get_nv_sig_units(cxn),
        "opti_coords_list": opti_coords_list,
        "opti_coords_list-units": "V",
        "state": state.name,
        "num_reps": num_reps,
        "apd_gate_channel": apd_gate_channel,
        "sig_tags": sig_tags.tolist(),
        "ref_tags": ref_tags.tolist(),
    }
    file_path = tool_belt.get_file_path(__file__, timestamp, nv_sig["name"])
    tool_belt.save_figure(fig, file_path)
    tool_belt.save_raw_data(raw_data, file_path)


# region Main

def main(nv_sig, num_reps,
         max_readouts, powers=None, filters=None, state=States.LOW):

    with labrad.connect() as cxn:
        main_with_cxn(cxn, nv_sig, num_reps,
                      max_readouts, powers, filters, state)

def main_with_cxn(cxn, nv_sig, num_reps,
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
    kpl.init_kplotlib()

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

        optimize_readout_duration(cxn, adjusted_nv_sig,
                                  num_reps, state)

# endregion

if __name__ == "__main__":

    file_name = "2022_12_05-14_16_25-15micro-nv1_zfs_vs_t"
    data = tool_belt.get_raw_data(file_name)

    nv_sig = data["nv_sig"]
    max_readout = nv_sig["spin_readout_dur"]
    sig_tags = data["sig_tags"]
    ref_tags = data["ref_tags"]
    num_reps = data["num_reps"]

    kpl.init_kplotlib(no_latex=True)
    plot_readout_duration_optimization(max_readout, num_reps, sig_tags, ref_tags)

    plt.show(block=True)
