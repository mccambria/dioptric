# -*- coding: utf-8 -*-
"""
Ramsey measruement.

This routine polarizes the nv state into 0, then applies a pi/2 pulse to
put the state into a superposition between the 0 and + or - 1 state. The state
then evolves for a time, tau, of free precesion, and then a second pi/2 pulse
is applied. The amount of population in 0 is read out by collecting the
fluorescence during a readout.

It then takes a fast fourier transform of the time data to attempt to extract
the frequencies in the ramsey experiment. If the funtion can't determine the
peaks in the fft, then a detuning is used.

Lastly, this file curve_fits the data to a triple sum of cosines using the
found frequencies.

Created on Wed Apr 24 15:01:04 2019

@author: agardill
"""

# %% Imports


import utils.tool_belt as tool_belt
from scipy.signal import find_peaks
from numpy import pi
import numpy
import time
import matplotlib.pyplot as plt
from random import shuffle
import labrad
from utils.tool_belt import States
from scipy.optimize import curve_fit
import majorroutines.optimize as optimize



def main(
    nv_sig,
    apd_indices,
    detuning,
    precession_time,
    num_reps,
    state=States.LOW,
):

    with labrad.connect() as cxn:
        main_with_cxn(
            cxn,
            nv_sig,
            apd_indices,
            detuning,
            precession_time,
            num_reps,
            state,
        )


def main_with_cxn(
    cxn,
    nv_sig,
    apd_indices,
    detuning,
    precession_time,
    num_reps,
    state=States.LOW,
):
    
    counter_server = tool_belt.get_counter_server(cxn)
    pulsegen_server = tool_belt.get_pulsegen_server(cxn)
    

    tool_belt.reset_cfm(cxn)

    # %% Sequence setup

    green_laser_key = "nv-_reionization_laser"
    green_laser_name = nv_sig[green_laser_key]
    tool_belt.set_filter(cxn, nv_sig, green_laser_key)
    green_laser_power = tool_belt.set_laser_power(cxn, nv_sig, green_laser_key)
    
    polarization_time = nv_sig["nv-_reionization_dur"]
    gate_time = nv_sig["spin_readout_dur"]

    rabi_period = nv_sig["rabi_{}".format(state.name)]
    uwave_freq = nv_sig["resonance_{}".format(state.name)]
    uwave_power = nv_sig["uwave_power_{}".format(state.name)]
    # Detune the pi/2 pulse frequency
    uwave_freq_detuned = uwave_freq + detuning / 10**3

    # Get pulse frequencies
    uwave_pi_pulse = 0
    uwave_pi_on_2_pulse = tool_belt.get_pi_on_2_pulse_dur(rabi_period)

    seq_file_name = "ramsey_noref_onetau.py"

    precession_time = numpy.int32(precession_time)

    sig_counts = numpy.zeros([num_reps])
    sig_counts[:] = numpy.nan
    # ref_counts = numpy.copy(sig_counts)

    # %% Make some lists and variables to save at the end

    opti_coords_list = []

    # %% Analyze the sequence
    seq_args = [
        precession_time/2,
        polarization_time,
        gate_time,
        uwave_pi_pulse,
        uwave_pi_on_2_pulse,
        apd_indices[0],
        state.value,
        green_laser_name,
        green_laser_power
        ]
    print(seq_args)
    seq_args_string = tool_belt.encode_seq_args(seq_args)
    ret_vals = pulsegen_server.stream_load(seq_file_name, seq_args_string)
    seq_time = ret_vals[0]


    startFunctionTime = time.time()
    start_timestamp = tool_belt.get_time_stamp()

    # %% Collect the data

    # Start 'Press enter to stop...'
    tool_belt.init_safe_stop()


    # Break out of the while if the user says stop
    # Optimize and save the coords we found
    opti_coords = optimize.main_with_cxn(cxn, nv_sig, apd_indices)
    opti_coords_list.append(opti_coords)

    # Set up the microwaves
    sig_gen_cxn = tool_belt.get_signal_generator_cxn(cxn, state)
    sig_gen_cxn.set_freq(uwave_freq_detuned)
    sig_gen_cxn.set_amp(uwave_power)
    sig_gen_cxn.uwave_on()

    # Set up the laser
    tool_belt.set_filter(cxn, nv_sig, green_laser_key)
    green_laser_power = tool_belt.set_laser_power(cxn, nv_sig, green_laser_key)

    # Load the APD
    counter_server.start_tag_stream(apd_indices)

    seq_args = [
        precession_time/2,
        polarization_time,
        gate_time,
        uwave_pi_pulse,
        uwave_pi_on_2_pulse,
        apd_indices[0],
        state.value,
        green_laser_name,
        green_laser_power
        ]
    
    seq_args_string = tool_belt.encode_seq_args(seq_args)
    # Clear the counter/tagger buffer of any excess counts
    counter_server.clear_buffer()
    print(seq_args)
    pulsegen_server.stream_immediate(seq_file_name, num_reps, seq_args_string)

    new_counts = counter_server.read_counter_separate_gates(1)
    sample_counts = new_counts[0]
    sig_counts = sample_counts

    counter_server.stop_tag_stream()
        

    # %% Hardware clean up

    tool_belt.reset_cfm(cxn)


    # %% Plot the final data
    raw_fig, ax = plt.subplots(1, 1, figsize=(17, 8.5))
    ax.cla()
    ax.plot(sig_counts, "r-", label="signal")
    ax.set_xlabel(r"$\tau + \pi$"+" + {}".format(precession_time)+" ($\mathrm{\mu s}$)")
    ax.set_ylabel("Counts")
    ax.legend()
    
    raw_fig.canvas.draw()
    raw_fig.set_tight_layout(True)
    raw_fig.canvas.flush_events()

    # %% Save the data

    endFunctionTime = time.time()

    timeElapsed = endFunctionTime - startFunctionTime

    timestamp = tool_belt.get_time_stamp()

    raw_data = {
        "timestamp": timestamp,
        "timeElapsed": timeElapsed,
        "nv_sig": nv_sig,
        "nv_sig-units": tool_belt.get_nv_sig_units(),
        'detuning': detuning,
        'detuning-units': 'MHz',
        "gate_time": gate_time,
        "gate_time-units": "ns",
        "uwave_freq": uwave_freq_detuned,
        "uwave_freq-units": "GHz",
        "uwave_power": uwave_power,
        "uwave_power-units": "dBm",
        "rabi_period": rabi_period,
        "rabi_period-units": "ns",
        "uwave_pi_on_2_pulse": uwave_pi_on_2_pulse,
        "uwave_pi_on_2_pulse-units": "ns",
        "precession_time": int(precession_time),
        "precession_time-units": "ns",
        "state": state.name,
        "num_reps": num_reps,
        "sig_counts": sig_counts.astype(int).tolist(),
        "sig_counts-units": "counts",
    }

    file_path = tool_belt.get_file_path(__file__, timestamp, nv_sig["name"])
    tool_belt.save_figure(raw_fig, file_path)
    tool_belt.save_raw_data(raw_data, file_path)

    # %% Fit and save figs
    
    return 


# %% Run the file


if __name__ == "__main__":

    file = "2022_11_14-15_41_58-johnson-search"
    
    data = tool_belt.get_raw_data(file)
    
    sig_counts = numpy.array(data['sig_counts'])
    precession_time = data['precession_time']

    width=2000
    binned_data = sig_counts[:(sig_counts.size // width) * width].reshape(-1, width).sum(axis=1)
    
    raw_fig, ax = plt.subplots(1, 1, figsize=(17, 8.5))
    ax.cla()
    ax.plot(binned_data, "r-", label="signal")
    ax.set_xlabel(r"$\tau + \pi$"+" + {}".format(precession_time/1000)+" ($\mathrm{\mu s}$)")
    ax.set_ylabel("Counts")
    ax.legend()
    
    if False:
        plt.figure()
        plt.hist(sig_counts,bins=max(sig_counts))
        plt.show()
    
    
    
    
    
        
