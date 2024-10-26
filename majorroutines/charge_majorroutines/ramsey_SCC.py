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


import time
from random import shuffle

import labrad
import matplotlib.pyplot as plt
import numpy
from numpy import pi
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

import majorroutines.targeting as targeting
import utils.tool_belt as tool_belt
from majorroutines.ramsey import extract_oscillations, fit_ramsey
from utils.tool_belt import States

# %% Main


def main(
    nv_sig,
    detuning,
    precession_dur_range,
    num_steps,
    num_reps,
    num_runs,
    state=States.LOW,
    opti_nv_sig=None,
    one_precession_time=False,
    do_fm=False,
):
    with labrad.connect() as cxn:
        angle = main_with_cxn(
            cxn,
            nv_sig,
            detuning,
            precession_dur_range,
            num_steps,
            num_reps,
            num_runs,
            state,
            opti_nv_sig,
            one_precession_time,
            do_fm,
        )
        return angle


def main_with_cxn(
    cxn,
    nv_sig,
    detuning,
    precession_time_range,
    num_steps,
    num_reps,
    num_runs,
    state=States.LOW,
    opti_nv_sig=None,
    one_precession_time=False,
    do_fm=False,
):
    counter_server = tool_belt.get_counter_server(cxn)
    pulsegen_server = tool_belt.get_pulsegen_server(cxn)
    arbwavegen_server = tool_belt.get_server_arb_wave_gen(cxn)

    tool_belt.reset_cfm(cxn)

    # %% Sequence setup

    green_laser_key = "nv-_reionization_laser"
    green_laser_name = nv_sig[green_laser_key]
    red_laser_key = "nv0_ionization_laser"
    red_laser_name = nv_sig[red_laser_key]
    yellow_laser_key = "charge_readout_laser"
    yellow_laser_name = nv_sig[yellow_laser_key]
    tool_belt.set_filter(cxn, nv_sig, green_laser_key)
    green_laser_power = tool_belt.set_laser_power(cxn, nv_sig, green_laser_key)
    tool_belt.set_filter(cxn, nv_sig, red_laser_key)
    red_laser_power = tool_belt.set_laser_power(cxn, nv_sig, red_laser_key)
    tool_belt.set_filter(cxn, nv_sig, yellow_laser_key)
    yellow_laser_power = tool_belt.set_laser_power(cxn, nv_sig, yellow_laser_key)

    polarization_time = nv_sig["spin_pol_dur"]
    ion_time = nv_sig["nv0_ionization_dur"]
    gate_time = nv_sig["charge_readout_dur"]

    rabi_period = nv_sig["rabi_{}".format(state.name)]
    uwave_freq = nv_sig["resonance_{}".format(state.name)]
    uwave_power = nv_sig["uwave_power_{}".format(state.name)]
    # Detune the pi/2 pulse frequency
    uwave_freq_detuned = uwave_freq + detuning / 10**3

    # Get pulse frequencies
    uwave_pi_pulse = 0
    uwave_pi_on_2_pulse = tool_belt.get_pi_on_2_pulse_dur(rabi_period)

    seq_file_name = "ramsey_scc.py"

    # %% Create the array of relaxation times

    # Array of times to sweep through
    # Must be ints
    min_precession_time = int(precession_time_range[0])
    max_precession_time = int(precession_time_range[1])

    taus = numpy.linspace(
        min_precession_time,
        max_precession_time,
        num=num_steps,
        dtype=numpy.int32,
    )
    plot_taus = (taus + uwave_pi_pulse) / 1000

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

    sig_counts = numpy.zeros([num_runs, num_steps])
    sig_counts[:] = numpy.nan
    ref_counts = numpy.copy(sig_counts)

    # %% Make some lists and variables to save at the end

    opti_coords_list = []
    tau_index_master_list = [[] for i in range(num_runs)]

    # %% Analyze the sequence
    seq_args = [
        min_precession_time / 2,
        polarization_time,
        ion_time,
        gate_time,
        uwave_pi_pulse,
        uwave_pi_on_2_pulse,
        max_precession_time / 2,
        state.value,
        green_laser_name,
        red_laser_name,
        yellow_laser_name,
        green_laser_power,
        red_laser_power,
        yellow_laser_power,
    ]
    print(seq_args)
    seq_args_string = tool_belt.encode_seq_args(seq_args)
    ret_vals = pulsegen_server.stream_load(seq_file_name, seq_args_string)
    seq_time = ret_vals[0]
    #    print(seq_args)
    #    return
    #    print(seq_time)

    # %% Let the user know how long this will take

    seq_time_s = seq_time / (10**9)  # to seconds
    expected_run_time_s = (num_steps / 2) * num_reps * num_runs * seq_time_s  # s
    expected_run_time_m = expected_run_time_s / 60  # to minutes

    print(" \nExpected run time: {:.1f} minutes. ".format(expected_run_time_m))
    #    return

    # create figure
    raw_fig, axes_pack = plt.subplots(1, 2, figsize=(17, 8.5))
    ax = axes_pack[0]
    ax.plot([], [])
    ax.set_title("Non-normalized Count Rate Versus Frequency")
    ax.set_xlabel(r"$\tau + \pi$ ($\mathrm{\mu s}$)")
    ax.set_ylabel("Counts")

    ax = axes_pack[1]
    ax.plot([], [])
    ax.set_title("Ramsey Measurement")
    ax.set_xlabel(r"$\tau$ ($\mathrm{\mu s}$)")
    ax.set_ylabel("Contrast (arb. units)")

    # %% Get the starting time of the function, to be used to calculate run time

    startFunctionTime = time.time()
    start_timestamp = tool_belt.get_time_stamp()

    # %% Collect the data

    # Start 'Press enter to stop...'
    tool_belt.init_safe_stop()

    for run_ind in range(num_runs):
        print(" \nRun index: {}".format(run_ind))

        # Break out of the while if the user says stop
        if tool_belt.safe_stop():
            break

        # Optimize and save the coords we found
        if opti_nv_sig:
            opti_coords = targeting.main_with_cxn(cxn, opti_nv_sig)
            drift = tool_belt.get_drift()
            adj_coords = nv_sig["coords"] + numpy.array(drift)
            tool_belt.set_xyz(cxn, adj_coords)
        else:
            opti_coords = targeting.main_with_cxn(cxn, nv_sig)
        opti_coords_list.append(opti_coords)

        # Set up the microwaves
        sig_gen_cxn = tool_belt.get_signal_generator_cxn(cxn, state)
        sig_gen_cxn.set_freq(uwave_freq_detuned)
        sig_gen_cxn.set_amp(uwave_power)
        sig_gen_cxn.uwave_on()

        # Set up the laser
        tool_belt.set_filter(cxn, nv_sig, green_laser_key)
        green_laser_power = tool_belt.set_laser_power(cxn, nv_sig, green_laser_key)
        tool_belt.set_filter(cxn, nv_sig, red_laser_key)
        red_laser_power = tool_belt.set_laser_power(cxn, nv_sig, red_laser_key)
        tool_belt.set_filter(cxn, nv_sig, yellow_laser_key)
        yellow_laser_power = tool_belt.set_laser_power(cxn, nv_sig, yellow_laser_key)

        # Load the APD
        counter_server.start_tag_stream()

        # Shuffle the list of tau indices so that it steps thru them randomly
        shuffle(tau_ind_list)

        tau_run_ind = 0

        for tau_ind in tau_ind_list:
            tau_run_ind = tau_run_ind + 1

            # 'Flip a coin' to determine which tau (long/shrt) is used first

            # Optimize between some of the taus because the measurements are long.
            # I had to put the start tag stream in again because it gets reset in the optimize file when we reset the cfm
            # might need to adjust this if you run into problems with the pulse streamer. This works with the opx at least.
            if tau_run_ind % 5 == 0:
                if opti_nv_sig:
                    opti_coords = targeting.main_with_cxn(cxn, opti_nv_sig)
                    drift = tool_belt.get_drift()
                    adj_coords = nv_sig["coords"] + numpy.array(drift)
                    tool_belt.set_xyz(cxn, adj_coords)
                else:
                    opti_coords = targeting.main_with_cxn(cxn, nv_sig)
                counter_server.start_tag_stream()
                # opti_coords_list.append(opti_coords)

            rand_boolean = numpy.random.randint(0, high=2)

            if rand_boolean == 1:
                tau_ind_first = tau_ind
                tau_ind_second = -tau_ind - 1
            elif rand_boolean == 0:
                tau_ind_first = -tau_ind - 1
                tau_ind_second = tau_ind

            if one_precession_time:
                tau_ind_first = 0
                tau_ind_second = 0

            # add the tau indexxes used to a list to save at the end
            tau_index_master_list[run_ind].append(tau_ind_first)
            tau_index_master_list[run_ind].append(tau_ind_second)

            # Break out of the while if the user says stop
            if tool_belt.safe_stop():
                break

            print(" \nFirst relaxation time: {}".format(taus[tau_ind_first]))
            print("Second relaxation time: {}".format(taus[tau_ind_second]))

            seq_args = [
                taus[tau_ind_first] / 2,
                polarization_time,
                ion_time,
                gate_time,
                uwave_pi_pulse,
                uwave_pi_on_2_pulse,
                taus[tau_ind_second] / 2,
                state.value,
                green_laser_name,
                red_laser_name,
                yellow_laser_name,
                green_laser_power,
                red_laser_power,
                yellow_laser_power,
            ]

            seq_args_string = tool_belt.encode_seq_args(seq_args)
            # Clear the counter/tagger buffer of any excess counts
            counter_server.clear_buffer()
            print(seq_args)
            pulsegen_server.stream_immediate(seq_file_name, num_reps, seq_args_string)

            # Each sample is of the form [*(<sig_shrt>, <ref_shrt>, <sig_long>, <ref_long>)]
            # So we can sum on the values for similar index modulus 4 to
            # parse the returned list into what we want.
            new_counts = counter_server.read_counter_separate_gates(1)
            sample_counts = new_counts[0]

            count = sum(sample_counts[0::4])
            sig_counts[run_ind, tau_ind_first] = count
            print("First signal = " + str(count))

            count = sum(sample_counts[1::4])
            ref_counts[run_ind, tau_ind_first] = count
            print("First Reference = " + str(count))

            count = sum(sample_counts[2::4])
            sig_counts[run_ind, tau_ind_second] = count
            print("Second Signal = " + str(count))

            count = sum(sample_counts[3::4])
            ref_counts[run_ind, tau_ind_second] = count
            print("Second Reference = " + str(count))

        counter_server.stop_tag_stream()

        # %% incremental plotting

        # Average the counts over the iterations
        avg_sig_counts = numpy.average(sig_counts[: (run_ind + 1)], axis=0)
        avg_ref_counts = numpy.average(ref_counts[: (run_ind + 1)], axis=0)

        try:
            norm_avg_sig = avg_sig_counts / numpy.average(avg_ref_counts)
        except RuntimeWarning as e:
            print(e)
            inf_mask = numpy.isinf(norm_avg_sig)
            # Assign to 0 based on the passed conditional array
            norm_avg_sig[inf_mask] = 0

        ax = axes_pack[0]
        ax.cla()
        ax.plot(plot_taus, avg_sig_counts, "r-", label="signal")
        ax.plot(plot_taus, avg_ref_counts, "g-", label="reference")
        ax.set_xlabel(r"$\tau + \pi$ ($\mathrm{\mu s}$)")
        ax.set_ylabel("Counts")
        ax.legend()

        ax = axes_pack[1]
        ax.cla()
        ax.plot(plot_taus, norm_avg_sig, "b-")
        ax.set_title("Ramsey Measurement")
        ax.set_xlabel(r"$\tau + \pi$ ($\mathrm{\mu s}$)")
        ax.set_ylabel("Contrast (arb. units)")

        text_popt = "Run # {}/{}".format(run_ind + 1, num_runs)

        props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
        ax.text(
            0.8,
            0.9,
            text_popt,
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=props,
        )

        raw_fig.canvas.draw()
        raw_fig.set_tight_layout(True)
        raw_fig.canvas.flush_events()

        # %% Save the data we have incrementally for long T1s

        raw_data = {
            "start_timestamp": start_timestamp,
            "nv_sig": nv_sig,
            "nv_sig-units": tool_belt.get_nv_sig_units(),
            "detuning": detuning,
            "detuning-units": "MHz",
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
            "precession_time_range": precession_time_range,
            "precession_time_range-units": "ns",
            "state": state.name,
            "num_steps": num_steps,
            "num_reps": num_reps,
            "run_ind": run_ind,
            "tau_index_master_list": tau_index_master_list,
            "opti_coords_list": opti_coords_list,
            "opti_coords_list-units": "V",
            "taus": taus.tolist(),
            "taus-units": "ns",
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
        tool_belt.save_figure(raw_fig, file_path)

    # %% Hardware clean up

    tool_belt.reset_cfm(cxn)

    # %% Plot the final data

    ax = axes_pack[0]
    ax.cla()
    # Account for the pi/2 pulse on each side of a tau
    ax.plot(plot_taus, avg_sig_counts, "r-", label="signal")
    ax.plot(plot_taus, avg_ref_counts, "g-", label="reference")
    ax.set_xlabel(r"$\tau + \pi$ ($\mathrm{\mu s}$)")
    ax.set_ylabel("Counts")
    ax.legend()

    ax = axes_pack[1]
    ax.cla()
    ax.plot(plot_taus, norm_avg_sig, "b-")
    ax.set_title("Ramsey Measurement")
    ax.set_xlabel(r"$\tau + \pi$ ($\mathrm{\mu s}$)")
    ax.set_ylabel("Contrast (arb. units)")

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
        "detuning": detuning,
        "detuning-units": "MHz",
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
        "precession_time_range": precession_time_range,
        "precession_time_range-units": "ns",
        "state": state.name,
        "num_steps": num_steps,
        "num_reps": num_reps,
        "num_runs": num_runs,
        "tau_index_master_list": tau_index_master_list,
        "opti_coords_list": opti_coords_list,
        "opti_coords_list-units": "V",
        "taus": taus.tolist(),
        "taus-units": "ns",
        "sig_counts": sig_counts.astype(int).tolist(),
        "sig_counts-units": "counts",
        "ref_counts": ref_counts.astype(int).tolist(),
        "ref_counts-units": "counts",
        "norm_avg_sig": norm_avg_sig.astype(float).tolist(),
        "norm_avg_sig-units": "arb",
    }

    file_path = tool_belt.get_file_path(__file__, timestamp, nv_sig["name"])
    tool_belt.save_figure(raw_fig, file_path)
    tool_belt.save_raw_data(raw_data, file_path)

    # %% Fit and save figs

    # Fourier transform
    fig_fft, FreqParams = extract_oscillations(
        norm_avg_sig, precession_time_range, num_steps, detuning
    )

    # Save the fft figure
    file_path_fft = tool_belt.get_file_path(
        __file__, timestamp, nv_sig["name"] + "_fft"
    )
    tool_belt.save_figure(fig_fft, file_path_fft)

    # Fit actual data
    fig_fit = fit_ramsey(norm_avg_sig, taus, precession_time_range, FreqParams)

    # Save the file in the same file directory
    file_path_fit = tool_belt.get_file_path(
        __file__, timestamp, nv_sig["name"] + "_fit"
    )
    tool_belt.save_figure(fig_fit, file_path_fit)

    return


# %% Run the file


if __name__ == "__main__":
    analysis = True
    analytics = True
    if analysis:
        folder = "pc_carr/branch_opx-setup/ramsey/2022_11"
        file = "2022_11_11-11_59_10-johnson-search"

        # detuning = 0
        data = tool_belt.get_raw_data(file, folder)
        detuning = data["detuning"]
        nv_sig = data["nv_sig"]
        sig_counts = data["sig_counts"]
        ref_counts = data["ref_counts"]
        norm_avg_sig = numpy.average(sig_counts, axis=0) / numpy.average(ref_counts)
        # norm_avg_sig = data['norm_avg_sig']
        precession_time_range = data["precession_time_range"]
        num_steps = data["num_steps"]
        try:
            taus = data["taus"]
            taus = numpy.array(taus)
        except Exception:
            taus = numpy.linspace(
                precession_time_range[0],
                precession_time_range[1],
                num=num_steps,
            )

        # _, FreqParams = extract_oscillations(norm_avg_sig, precession_time_range, num_steps, detuning)
        # print(FreqParams)

        # fit_ramsey(norm_avg_sig,taus,  precession_time_range, FreqParams)

    if analytics:
        # t = numpy.linspace(.040,1.04,50)
        func = (
            tool_belt.cosine_sum
        )  # (t, offset, decay, amp_1, freq_1, amp_2, freq_2, amp_3, freq_3)
        taus = taus / 1000
        offset = 0.9
        decay = 0.5
        amp_1 = -0.033
        amp_2 = amp_1
        amp_3 = amp_1
        detuning = -0.74
        freq_1 = detuning - 2.2
        freq_2 = detuning
        freq_3 = detuning + 2.2

        fit_func = tool_belt.cosine_sum
        # fit_func = tool_belt.cosine_one
        # fit_func = cosine_sum_fixed_freq
        # init_params = guess_params_fixed_freq

        guess_params = (offset, decay, amp_1, freq_1, amp_2, freq_2, amp_3, freq_3)
        # guess_params = (offset, decay, amp_1, freq_1)
        init_params = guess_params

        popt, pcov = curve_fit(fit_func, taus, norm_avg_sig, p0=init_params)

        theoryvals = func(
            taus, offset, decay, amp_1, freq_1, amp_2, freq_2, amp_3, freq_3
        )
        # print(vals)
        # plt.figure()
        # plt.plot(taus,theoryvals)
        plt.plot(taus, norm_avg_sig)
        # plt.plot(taus,fit_func(taus,popt[0],popt[1],popt[2],popt[3]))
        # plt.plot(taus,fit_func(taus,popt[0],popt[1],popt[2],popt[3],popt[4],popt[5],popt[6],popt[7]))
        # plt.show()
        raw_fig = fit_ramsey(
            norm_avg_sig, taus * 1000, precession_time_range, [freq_1, freq_2, freq_3]
        )

        cur_time = tool_belt.get_time_stamp()
        file_path = tool_belt.get_file_path(
            __file__, cur_time, nv_sig["name"] + "-refit"
        )
        tool_belt.save_figure(raw_fig, file_path)
        # extract_oscillations(vals, t, len(t), detuning)
