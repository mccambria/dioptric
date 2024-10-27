# -*- coding: utf-8 -*-
"""
T1 measurement routine.

This version of t1 allows the the readout and measurement of all nine possible
combinations of the preparation and readout of the states in relaxation
measurements.

We write the +1 frequency to the Tektronix signal generator, 
and set the BNC signal generator to the -1 freq

To specify the preparation and readout states, pass into the function a list in
the form [preparation state, readout state]. That is passed in as
init_read_state.

Created on Wed Apr 24 15:01:04 2019

@author: agardill
"""

# %% Imports


import json
import os
import time
from random import shuffle

import labrad
import matplotlib.pyplot as plt
import numpy
from scipy.optimize import curve_fit

import majorroutines.targeting as targeting
import utils.tool_belt as tool_belt
from utils.tool_belt import States

# %% Main


def main(
    nv_sig,
    apd_indices,
    relaxation_time_range,
    num_steps,
    num_reps,
    num_runs,
    init_read_list,
    plot_data=True,
    save_data=True,
):
    with labrad.connect() as cxn:
        avg_sig_counts, avg_ref_counts, norm_avg_sig = main_with_cxn(
            cxn,
            nv_sig,
            apd_indices,
            relaxation_time_range,
            num_steps,
            num_reps,
            num_runs,
            init_read_list,
            plot_data,
            save_data,
        )

    return avg_sig_counts, avg_ref_counts, norm_avg_sig


def main_with_cxn(
    cxn,
    nv_sig,
    apd_indices,
    relaxation_time_range,
    num_steps,
    num_reps,
    num_runs,
    init_read_list,
    plot_data,
    save_data,
):
    tool_belt.reset_cfm(cxn)

    # %% Define the times to be used in the sequence

    readout_time = nv_sig["pulsed_SCC_readout_dur"]
    readout_power = nv_sig["am_589_power"]
    init_ion_time = nv_sig["pulsed_initial_ion_dur"]
    reion_time = nv_sig["pulsed_reionization_dur"]
    ion_time = nv_sig["pulsed_ionization_dur"]
    shelf_time = nv_sig["pulsed_shelf_dur"]
    shelf_power = nv_sig["am_589_shelf_power"]

    shared_params = tool_belt.get_shared_parameters_dict(cxn)

    laser_515_delay = shared_params["515_laser_delay"]
    aom_589_delay = shared_params["589_aom_delay"]
    laser_638_delay = shared_params["638_DM_laser_delay"]
    rf_delay = shared_params["uwave_delay"]

    wait_time = shared_params["post_polarization_wait_dur"]

    # %% Unpack the initial and read state

    init_state = init_read_list[0]
    read_state = init_read_list[1]

    # %% Setting initialize and readout states

    uwave_pi_pulse_high = round(nv_sig["rabi_HIGH"] / 2)
    uwave_pi_pulse_low = round(nv_sig["rabi_LOW"] / 2)
    uwave_freq_high = nv_sig["resonance_HIGH"]
    uwave_freq_low = nv_sig["resonance_LOW"]
    uwave_power_high = nv_sig["uwave_power_HIGH"]
    uwave_power_low = nv_sig["uwave_power_LOW"]

    # Default values
    uwave_pi_pulse_init = 0
    uwave_freq_init = 2.87
    uwave_power_init = 9.0
    if init_state.value == States.HIGH.value:
        uwave_pi_pulse_init = uwave_pi_pulse_high
        uwave_freq_init = uwave_freq_high
        uwave_power_init = uwave_power_high
    elif init_state.value == tool_belt.States.LOW.value:
        uwave_pi_pulse_init = uwave_pi_pulse_low
        uwave_freq_init = uwave_freq_low
        uwave_power_init = uwave_power_low

    # Default values
    uwave_pi_pulse_read = 0
    uwave_freq_read = 2.87
    uwave_power_read = 9.0
    if read_state.value == States.HIGH.value:
        uwave_pi_pulse_read = uwave_pi_pulse_high
        uwave_freq_read = uwave_freq_high
        uwave_power_read = uwave_power_high
    elif read_state.value == States.LOW.value:
        uwave_pi_pulse_read = uwave_pi_pulse_low
        uwave_freq_read = uwave_freq_low
        uwave_power_read = uwave_power_low
    if plot_data:
        print("Init state: {}".format(init_state.name))
        print("Init pi pulse: {} ns".format(uwave_pi_pulse_init))
        print("Init frequency: {} GHz".format(uwave_freq_init))
        print("Init power: {} dBm".format(uwave_power_init))
        print("Read state: {}".format(read_state.name))
        print("Read pi pulse: {} ns".format(uwave_pi_pulse_read))
        print("Read frequency: {} GHz".format(uwave_freq_read))
        print("Read power: {} dBm".format(uwave_power_read))

    # %% Create the array of relaxation times

    # Array of times to sweep through
    # Must be ints since the pulse streamer only works with int64s

    min_relaxation_time = int(relaxation_time_range[0])
    max_relaxation_time = int(relaxation_time_range[1])

    taus = numpy.linspace(min_relaxation_time, max_relaxation_time, num=num_steps)

    # %% Fix the length of the sequence to account for odd amount of elements

    # Our sequence pairs the longest time with the shortest time, and steps
    # toward the middle. This means we only step through half of the length
    # of the time array.

    # That is a problem if the number of elements is odd. To fix this, we add
    # one to the length of the array. When this number is halfed and turned
    # into an integer, it will step through the middle element.

    if len(taus) % 2 == 0:
        half_length_taus = int(len(taus) / 2)
    elif len(taus) % 2 == 1:
        half_length_taus = int((len(taus) + 1) / 2)

    # Then we must use this half length to calculate the list of integers to be
    # shuffled for each run

    tau_ind_list = list(range(0, half_length_taus))

    # %% Create data structure to save the counts

    # We create an array of NaNs that we'll fill
    # incrementally for the signal and reference counts.
    # NaNs are ignored by matplotlib, which is why they're useful for us here.
    # We define 2D arrays, with the horizontal dimension for the frequency and
    # the veritical dimension for the index of the run.

    sig_counts = numpy.empty([num_runs, num_steps], dtype=numpy.uint32)
    sig_counts[:] = numpy.nan
    ref_counts = numpy.copy(sig_counts)

    # %% Make some lists and variables to save at the end

    opti_coords_list = []
    tau_index_master_list = [[] for i in range(num_runs)]

    # %% Analyze the sequence

    # pulls the file of the sequence from serves/timing/sequencelibrary
    file_name = os.path.basename(__file__)
    seq_args = [
        readout_time,
        init_ion_time,
        reion_time,
        ion_time,
        shelf_time,
        wait_time,
        uwave_pi_pulse_low,
        uwave_pi_pulse_high,
        min_relaxation_time,
        max_relaxation_time,
        laser_515_delay,
        aom_589_delay,
        laser_638_delay,
        rf_delay,
        apd_indices[0],
        init_state.value,
        read_state.value,
        readout_power,
        shelf_power,
    ]
    seq_args_string = tool_belt.encode_seq_args(seq_args)
    ret_vals = cxn.pulse_streamer.stream_load(file_name, seq_args_string)
    seq_time = ret_vals[0]

    # %% Ask user if they wish to run experiment based on run time

    seq_time_s = seq_time / (10**9)  # s
    expected_run_time = num_steps * num_reps * num_runs * seq_time_s / 2  # s
    expected_run_time_m = expected_run_time / 60  # m

    # Ask to continue and timeout if no response in 2 seconds?
    if plot_data:
        print(" \nExpected run time: {:.1f} minutes. ".format(expected_run_time_m))
    #    return

    # %% Get the starting time of the function, to be used to calculate run time

    startFunctionTime = time.time()
    start_timestamp = tool_belt.get_time_stamp()

    # %% Collect the data

    # Start 'Press enter to stop...'
    tool_belt.init_safe_stop()

    for run_ind in range(num_runs):
        if plot_data:
            print(" \nRun index: {}".format(run_ind))

        # Break out of the while if the user says stop
        if tool_belt.safe_stop():
            break

        # Optimize
        opti_coords = targeting.main_with_cxn(
            cxn, nv_sig, apd_indices, 532, disable=True
        )
        opti_coords_list.append(opti_coords)

        # Set up the microwaves for the low and high states
        low_sig_gen_cxn = tool_belt.get_signal_generator_cxn(cxn, States.LOW)
        low_sig_gen_cxn.set_freq(uwave_freq_low)
        low_sig_gen_cxn.set_amp(uwave_power_low)
        low_sig_gen_cxn.uwave_on()
        high_sig_gen_cxn = tool_belt.get_signal_generator_cxn(cxn, States.HIGH)
        high_sig_gen_cxn.set_freq(uwave_freq_high)
        high_sig_gen_cxn.set_amp(uwave_power_high)
        high_sig_gen_cxn.uwave_on()

        # Load the APD
        cxn.apd_tagger.start_tag_stream(apd_indices)

        # Shuffle the list of tau indices so that it steps thru them randomly
        shuffle(tau_ind_list)

        for tau_ind in tau_ind_list:
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

            # Break out of the while if the user says stop
            if tool_belt.safe_stop():
                break
            if plot_data:
                print(" \nFirst relaxation time: {}".format(taus[tau_ind_first]))
                print("Second relaxation time: {}".format(taus[tau_ind_second]))

            # Stream the sequence
            seq_args = [
                readout_time,
                init_ion_time,
                reion_time,
                ion_time,
                shelf_time,
                wait_time,
                uwave_pi_pulse_low,
                uwave_pi_pulse_high,
                taus[tau_ind_first],
                taus[tau_ind_second],
                laser_515_delay,
                aom_589_delay,
                laser_638_delay,
                rf_delay,
                apd_indices[0],
                init_state.value,
                read_state.value,
                readout_power,
                shelf_power,
            ]
            # Clear the tagger buffer of any excess counts
            cxn.apd_tagger.clear_buffer()
            seq_args_string = tool_belt.encode_seq_args(seq_args)

            cxn.pulse_streamer.stream_immediate(
                file_name, int(num_reps), seq_args_string
            )

            # Each sample is of the form [*(<sig_shrt>, <ref_shrt>, <sig_long>, <ref_long>)]
            # So we can sum on the values for similar index modulus 4 to
            # parse the returned list into what we want.
            new_counts = cxn.apd_tagger.read_counter_separate_gates(1)
            sample_counts = new_counts[0]

            #            sig_gate_counts = sample_counts[::4]
            #            sig_counts[run_ind, tau_ind] = sum(sig_gate_counts)

            count = sum(sample_counts[0::4])
            sig_counts[run_ind, tau_ind_first] = count
            if plot_data:
                print("First signal = " + str(count))

            count = sum(sample_counts[1::4])
            ref_counts[run_ind, tau_ind_first] = count
            if plot_data:
                print("First Reference = " + str(count))

            count = sum(sample_counts[2::4])
            sig_counts[run_ind, tau_ind_second] = count
            if plot_data:
                print("Second Signal = " + str(count))

            count = sum(sample_counts[3::4])
            ref_counts[run_ind, tau_ind_second] = count
            if plot_data:
                print("Second Reference = " + str(count))

        cxn.apd_tagger.stop_tag_stream()

        # %% Save the data we have incrementally for long measurements
        if save_data:
            raw_data = {
                "start_timestamp": start_timestamp,
                "init_state": init_state.name,
                "read_state": read_state.name,
                "nv_sig": nv_sig,
                "nv_sig-units": tool_belt.get_nv_sig_units(),
                "uwave_freq_init": uwave_freq_init,
                "uwave_freq_init-units": "GHz",
                "uwave_freq_read": uwave_freq_read,
                "uwave_freq_read-units": "GHz",
                "uwave_power_high": uwave_power_high,
                "uwave_power_high-units": "dBm",
                "uwave_power_low": uwave_power_low,
                "uwave_power_low-units": "dBm",
                "uwave_pi_pulse_init": uwave_pi_pulse_init,
                "uwave_pi_pulse_init-units": "ns",
                "uwave_pi_pulse_read": uwave_pi_pulse_read,
                "uwave_pi_pulse_read-units": "ns",
                "relaxation_time_range": relaxation_time_range,
                "relaxation_time_range-units": "ns",
                "num_steps": num_steps,
                "num_reps": num_reps,
                "run_ind": run_ind,
                "tau_index_master_list": tau_index_master_list,
                "opti_coords_list": opti_coords_list,
                "opti_coords_list-units": "V",
                "sig_counts": sig_counts.astype(int).tolist(),
                "sig_counts-units": "counts",
                "ref_counts": ref_counts.astype(int).tolist(),
                "ref_counts-units": "counts",
            }

            # This will continuously be the same file path so we will overwrite
            # the existing file with the latest version
            file_path = tool_belt.get_file_path(
                __file__, start_timestamp, nv_sig["name"], "incremental"
            )
            tool_belt.save_raw_data(raw_data, file_path)

    # %% Hardware clean up

    tool_belt.reset_cfm(cxn)

    # %% Average the counts over the iterations

    avg_sig_counts = numpy.average(sig_counts, axis=0)
    avg_ref_counts = numpy.average(ref_counts, axis=0)

    # %% Calculate the t1 data, signal / reference over different relaxation times

    # Replace x/0=inf with 0
    try:
        norm_avg_sig = avg_sig_counts / avg_ref_counts
    except RuntimeWarning as e:
        print(e)
        inf_mask = numpy.isinf(norm_avg_sig)
        # Assign to 0 based on the passed conditional array
        norm_avg_sig[inf_mask] = 0

    # %% Plot the t1 signal
    if plot_data:
        raw_fig, axes_pack = plt.subplots(1, 2, figsize=(17, 8.5))

        ax = axes_pack[0]
        ax.plot(taus / 10**6, avg_sig_counts, "r-", label="signal")
        ax.plot(taus / 10**6, avg_ref_counts, "g-", label="reference")
        ax.set_xlabel("Relaxation time (ms)")
        ax.set_ylabel("Counts")
        ax.legend()

        ax = axes_pack[1]
        ax.plot(taus / 10**6, norm_avg_sig, "b-")
        ax.set_title(
            "T1 SCC readout. Initial state: {}, readout state: {}".format(
                init_state.name, read_state.name
            )
        )
        ax.set_xlabel("Relaxation time (ms)")
        ax.set_ylabel("Contrast (arb. units)")

        raw_fig.canvas.draw()
        # fig.set_tight_layout(True)
        raw_fig.canvas.flush_events()

    # %% Save the data
    if save_data:
        endFunctionTime = time.time()

        timeElapsed = endFunctionTime - startFunctionTime

        timestamp = tool_belt.get_time_stamp()

        raw_data = {
            "timestamp": timestamp,
            "timeElapsed": timeElapsed,
            "init_state": init_state.name,
            "read_state": read_state.name,
            "nv_sig": nv_sig,
            "nv_sig-units": tool_belt.get_nv_sig_units(),
            "uwave_freq_init": uwave_freq_init,
            "uwave_freq_init-units": "GHz",
            "uwave_freq_read": uwave_freq_read,
            "uwave_freq_read-units": "GHz",
            "uwave_power_high": uwave_power_high,
            "uwave_power_high-units": "dBm",
            "uwave_power_low": uwave_power_low,
            "uwave_power_low-units": "dBm",
            "uwave_pi_pulse_init": uwave_pi_pulse_init,
            "uwave_pi_pulse_init-units": "ns",
            "uwave_pi_pulse_read": uwave_pi_pulse_read,
            "uwave_pi_pulse_read-units": "ns",
            "relaxation_time_range": relaxation_time_range,
            "relaxation_time_range-units": "ns",
            "num_steps": num_steps,
            "num_reps": num_reps,
            "num_runs": num_runs,
            "tau_index_master_list": tau_index_master_list,
            "opti_coords_list": opti_coords_list,
            "opti_coords_list-units": "V",
            "sig_counts": sig_counts.astype(int).tolist(),
            "sig_counts-units": "counts",
            "ref_counts": ref_counts.astype(int).tolist(),
            "ref_counts-units": "counts",
            "norm_avg_sig": norm_avg_sig.astype(float).tolist(),
            "norm_avg_sig-units": "arb",
        }

        file_path = tool_belt.get_file_path(__file__, timestamp, nv_sig["name"])
        tool_belt.save_raw_data(raw_data, file_path)

    if plot_data:
        tool_belt.save_figure(raw_fig, file_path)

    return avg_sig_counts, avg_ref_counts, norm_avg_sig


# %%


def decayExp(t, offset, amplitude, decay):
    return offset + amplitude * numpy.exp(-decay * t)


# %% Fitting the data


def t1_exponential_decay(open_file_name, save_file_type):
    directory = "E:/Shared Drives/Kolkowitz Lab Group/nvdata/t1_double_quantum/branch_Spin_to_charge/2020_05/"

    # Open the specified file
    with open(directory + open_file_name + ".txt") as json_file:
        # Load the data from the file
        data = json.load(json_file)
        countsT1 = data["norm_avg_sig"]
        relaxation_time_range = data["relaxation_time_range"]
        num_steps = data["num_steps"]
        init_state = data["init_state"]
        read_state = data["read_state"]

    min_relaxation_time = relaxation_time_range[0]
    max_relaxation_time = relaxation_time_range[1]

    timeArray = numpy.linspace(min_relaxation_time, max_relaxation_time, num=num_steps)

    offset = 0.8
    amplitude = 0.1
    decay = 1 / 10000  # inverse ns

    popt, pcov = curve_fit(decayExp, timeArray, countsT1, p0=[offset, amplitude, decay])

    decay_time = 1 / popt[2]

    first = timeArray[0]
    last = timeArray[len(timeArray) - 1]
    linspaceTime = numpy.linspace(first, last, num=1000)

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.semilogy(timeArray / 10**6, countsT1, "bo", label="data")
    #    ax.plot(linspaceTime / 10**6, decayExp(linspaceTime,*popt),'r-',label='fit')
    ax.set_xlabel("Wait time (ms)")
    ax.set_ylabel("Contrast (arb. units)")
    ax.set_title("Prepared {} state, readout {} state".format(init_state, read_state))
    ax.legend()

    #    text = "\n".join((r'$C + A_0 e^{-t / d}$',
    #                      r'$C = $' + '%.1f'%(popt[0]),
    #                      r'$A_0 = $' + '%.1f'%(popt[1]),
    #                      r'$d = $' + "%.3f"%(decay_time / 10**6) + " ms"))
    #
    #
    #    props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    #    ax.text(0.70, 0.95, text, transform=ax.transAxes, fontsize=12,
    #                            verticalalignment="top", bbox=props)

    fig.canvas.draw()
    fig.canvas.flush_events()

    fig.savefig(open_file_name + "replot." + save_file_type)


if __name__ == "__main__":
    file_name = "2020_05_13-23_17_30-bachman-ensemble"
    t1_exponential_decay(file_name, "svg")
