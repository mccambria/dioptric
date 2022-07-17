# -*- coding: utf-8 -*-
"""
This routine uses a four-point PESR measurement based on Kucsko 2013

Created on Thu Apr 11 15:39:23 2019

@author: mccambria
"""

# %% Imports


import utils.tool_belt as tool_belt
import majorroutines.optimize as optimize
import majorroutines.pulsed_resonance as pulsed_resonance
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import labrad
from utils.tool_belt import States
from random import shuffle
import sys


# region Functions

def calc_resonance(norm_avg_sig, norm_avg_sig_ste, 
                   detuning, d_omega, passed_res):
    
    f1, f2, f3, f4 = norm_avg_sig
    f1_err, f2_err, f3_err, f4_err = norm_avg_sig_ste
    delta_res = ((f1 + f2) - (f3 + f4)) * (d_omega / ((f1 - f2) - (f3 - f4)))
    resonance = passed_res + delta_res
    # Calculate the error
    d_delta_res_df1 = (-2 * d_omega * (f2 - f4)) / ((f1 - f2 - f3 + f4) ** 2)
    d_delta_res_df2 = (2 * d_omega * (f1 - f3)) / ((f1 - f2 - f3 + f4) ** 2)
    d_delta_res_df3 = (2 * d_omega * (f2 - f4)) / ((f1 - f2 - f3 + f4) ** 2)
    d_delta_res_df4 = (-2 * d_omega * (f1 - f3)) / ((f1 - f2 - f3 + f4) ** 2)
    resonance_err = np.sqrt(
        (d_delta_res_df1 * f1_err) ** 2
        + (d_delta_res_df2 * f2_err) ** 2
        + (d_delta_res_df3 * f3_err) ** 2
        + (d_delta_res_df4 * f4_err) ** 2
    )
    
    return resonance, resonance_err
    

# endregion

# region Main


def main(
    nv_sig,
    apd_indices,
    num_reps,
    num_runs,
    state,
    detuning=0.005,
    d_omega=0.002,
    opti_nv_sig=None,
):

    with labrad.connect() as cxn:
        resonance, res_err = main_with_cxn(
            cxn,
            nv_sig,
            apd_indices,
            num_reps,
            num_runs,
            state,
            detuning,
            d_omega,
            opti_nv_sig,
        )
    return resonance, res_err


def main_with_cxn(
    cxn,
    nv_sig,
    apd_indices,
    num_reps,
    num_runs,
    state,
    detuning=0.004,
    d_omega=0.002,
    opti_nv_sig=None,
):

    # %% Initial calculations and setup

    tool_belt.reset_cfm(cxn)

    # Calculate the frequencies we need to set
    passed_res = nv_sig[f"resonance_{state.name}"]
    freq_1 = passed_res - detuning - d_omega
    freq_2 = passed_res - detuning + d_omega
    freq_3 = passed_res + detuning - d_omega
    freq_4 = passed_res + detuning + d_omega
    freqs = [freq_1, freq_2, freq_3, freq_4]
    num_steps = 4

    # Set up our data structure, an array of NaNs that we'll fill
    # incrementally. NaNs are ignored by matplotlib, which is why they're
    # useful for us here.
    # We define 2D arrays, with the horizontal dimension for the frequency and
    # the veritical dimension for the index of the run.
    ref_counts = np.empty([num_runs, num_steps])
    ref_counts[:] = np.nan
    sig_counts = np.copy(ref_counts)

    laser_key = "spin_laser"
    laser_name = nv_sig[laser_key]
    laser_power = tool_belt.set_laser_power(cxn, nv_sig, laser_key)

    polarization_time = nv_sig["spin_pol_dur"]
    readout = nv_sig["spin_readout_dur"]
    readout_sec = readout / (10 ** 9)
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

    opti_coords_list = []

    # %% Get the starting time of the function

    start_timestamp = tool_belt.get_time_stamp()

    # %% Collect the data

    # Create a list of indices to step through the freqs. This will be shuffled
    freq_index_master_list = [[] for i in range(num_runs)]
    freq_ind_list = list(range(0, num_steps))

    # Start 'Press enter to stop...'
    tool_belt.init_safe_stop()

    sig_gen_cxn = tool_belt.get_signal_generator_cxn(cxn, state)

    for run_ind in range(num_runs):
        print("Run index: {}".format(run_ind))

        # Break out of the while if the user says stop
        if tool_belt.safe_stop():
            break

        # Optimize and save the coords we found
        if opti_nv_sig:
            opti_coords = optimize.main_with_cxn(cxn, opti_nv_sig, apd_indices)
            drift = tool_belt.get_drift()
            adj_coords = nv_sig["coords"] + np.array(drift)
            tool_belt.set_xyz(cxn, adj_coords)
        else:
            opti_coords = optimize.main_with_cxn(cxn, nv_sig, apd_indices)
        opti_coords_list.append(opti_coords)

        # Set up the microwaves and laser. Then load the pulse streamer
        # (must happen after optimize and iq_switch since run their
        # own sequences)
        sig_gen_cxn.set_amp(nv_sig[f"uwave_power_{state.name}"])
        tool_belt.set_filter(cxn, nv_sig, laser_key)
        laser_power = tool_belt.set_laser_power(cxn, nv_sig, laser_key)

        # Start the tagger stream
        cxn.apd_tagger.start_tag_stream(apd_indices)

        # Take a sample and step through the shuffled frequencies
        shuffle(freq_ind_list)
        for freq_ind in freq_ind_list:

            # Break out of the while if the user says stop
            if tool_belt.safe_stop():
                break

            ret_vals = cxn.pulse_streamer.stream_load(
                "rabi.py", seq_args_string
            )

            freq_index_master_list[run_ind].append(freq_ind)

            sig_gen_cxn.set_freq(freqs[freq_ind])
            sig_gen_cxn.uwave_on()

            # It takes 400 us from receipt of the command to
            # switch frequencies so allow 1 ms total
            #            time.sleep(0.001)
            # Clear the tagger buffer of any excess counts
            cxn.apd_tagger.clear_buffer()
            # Start the timing stream
            cxn.pulse_streamer.stream_start(int(num_reps))

            # Get the counts
            new_counts = cxn.apd_tagger.read_counter_separate_gates(1)

            sample_counts = new_counts[0]

            # signal counts are even - get every second element starting from 0
            sig_gate_counts = sample_counts[0::2]
            sig_counts[run_ind, freq_ind] = sum(sig_gate_counts)

            # ref counts are odd - sample_counts every second element starting from 1
            ref_gate_counts = sample_counts[1::2]
            ref_counts[run_ind, freq_ind] = sum(ref_gate_counts)

        cxn.apd_tagger.stop_tag_stream()

    # %% Process the data

    ret_vals = pulsed_resonance.process_counts(
        ref_counts, sig_counts, num_runs
    )
    (
        avg_ref_counts,
        avg_sig_counts,
        norm_avg_sig,
        ste_ref_counts,
        ste_sig_counts,
        norm_avg_sig_ste,
    ) = ret_vals
    
    resonance, resonance_err = calc_resonance(norm_avg_sig, norm_avg_sig_ste, 
                                              detuning, d_omega, passed_res)

    # %% Clean up and save the data

    tool_belt.reset_cfm(cxn)

    timestamp = tool_belt.get_time_stamp()

    rawData = {
        "timestamp": timestamp,
        "nv_sig": nv_sig,
        "nv_sig-units": tool_belt.get_nv_sig_units(),
        "opti_coords_list": opti_coords_list,
        "opti_coords_list-units": "V",
        "state": state.name,
        "detuning": detuning, 
        "d_omega": d_omega, 
        "num_steps": num_steps,
        "num_reps": num_reps,
        "num_runs": num_runs,
        "readout": readout,
        "readout-units": "ns",
        "norm_avg_sig": norm_avg_sig.astype(float).tolist(),
        "norm_avg_sig-units": "arb",
        "norm_avg_sig_ste": norm_avg_sig_ste.astype(float).tolist(),
        "norm_avg_sig_ste-units": "arb",
        "freq_index_master_list": freq_index_master_list,
        "opti_coords_list": opti_coords_list,
        "opti_coords_list-units": "V",
        "sig_counts": sig_counts.astype(int).tolist(),
        "sig_counts-units": "counts",
        "ref_counts": ref_counts.astype(int).tolist(),
        "ref_counts-units": "counts",
    }

    nv_name = nv_sig["name"]
    filePath = tool_belt.get_file_path(__file__, timestamp, nv_name)
    tool_belt.save_raw_data(rawData, filePath)

    # %% Return

    return resonance, resonance_err


# %% Run the file


if __name__ == "__main__":
    
    f = ""
    data = tool_belt.get_raw_data(f)
    
    norm_avg_sig = data[""]
    norm_avg_sig_ste = data[""]
    # detuning = data[""]
    # d_omega = data[""]
    detuning = 0.004
    d_omega = 0.002
    nv_sig = data["nv_sig"]
    state = data["state"]
    passed_res = nv_sig[f"resonance_{state}"]

    resonance, resonance_err = calc_resonance(norm_avg_sig, norm_avg_sig_ste, 
                                              detuning, d_omega, passed_res)
