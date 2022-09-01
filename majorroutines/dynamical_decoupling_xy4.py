# -*- coding: utf-8 -*-
"""
Dynamical decoupling XY4.

One unit of XY4 is defined as:
    tau - pi_x - tau - tau - pi_y - tau - tau - pi_x - tau - tau - pi_y - tau

Created on Fri Aug 5 2022

@author: agardill
"""

# %% Imports


import utils.tool_belt as tool_belt
import majorroutines.optimize as optimize
from scipy.optimize import minimize_scalar
from numpy import pi
import numpy
import time
import matplotlib.pyplot as plt
from random import shuffle
import labrad
from utils.tool_belt import States
from scipy.optimize import curve_fit
from numpy.linalg import eigvals


# %% Constants


im = 0 + 1j
inv_sqrt_2 = 1 / numpy.sqrt(2)
gmuB = 2.8e-3  # gyromagnetic ratio in GHz / G



# %% Main


def main(
    nv_sig,
    apd_indices,
    precession_dur_range,
    num_xy4_reps,
    num_steps,
    num_reps,
    num_runs,
    taus=[],
    state=States.LOW,
):

    with labrad.connect() as cxn:
        angle = main_with_cxn(
            cxn,
            nv_sig,
            apd_indices,
            precession_dur_range,
            num_xy4_reps,
            num_steps,
            num_reps,
            num_runs,
            taus,
            state,
        )
        return angle


def main_with_cxn(
    cxn,
    nv_sig,
    apd_indices,
    precession_time_range,
    num_xy4_reps,
    num_steps,
    num_reps,
    num_runs,
    taus = [],
    state=States.LOW,
):

    tool_belt.reset_cfm(cxn)

    # %% Sequence setup

    laser_key = "spin_laser"
    laser_name = nv_sig[laser_key]
    tool_belt.set_filter(cxn, nv_sig, laser_key)
    laser_power = tool_belt.set_laser_power(cxn, nv_sig, laser_key)
    polarization_time = nv_sig["spin_pol_dur"]
    gate_time = nv_sig["spin_readout_dur"]

    rabi_period = nv_sig["rabi_{}".format(state.name)]
    uwave_freq = nv_sig["resonance_{}".format(state.name)]
    uwave_power = nv_sig["uwave_power_{}".format(state.name)]

    # Get pulse frequencies
    uwave_pi_pulse = tool_belt.get_pi_pulse_dur(rabi_period)
    uwave_pi_on_2_pulse = tool_belt.get_pi_on_2_pulse_dur(rabi_period)
    # pi pulses are slightly different than just half of the rabi period
    # uwave_pi_pulse = nv_sig["pi_pulse_{}".format(state.name)] 
    # uwave_pi_on_2_pulse = nv_sig["pi_on_2_pulse_{}".format(state.name)] 

    seq_file_name = "dynamical_decoupling.py"

    # %% Create the array of relaxation times

    # Array of times to sweep through
    # Must be ints
    min_precession_time = int(precession_time_range[0])
    max_precession_time = int(precession_time_range[1])
    
    if len(taus) == 0:
        taus = numpy.linspace(
            min_precession_time,
            max_precession_time,
            num=num_steps,
            dtype=numpy.int32,
        )
    # taus = taus + 500
    print(taus)
    # Convert to ms
    #plot_taus = taus / 1000
    plot_taus = (taus * 2 *4 * num_xy4_reps) / 1000

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
    
    num_reps = int(num_reps)

    pi_pulse_reps = num_xy4_reps*4
    seq_args = [
        taus[0],
        polarization_time,
        gate_time,
        uwave_pi_pulse,
        uwave_pi_on_2_pulse,
        taus[-1],
        pi_pulse_reps,
        apd_indices[0],
        state.value,
        laser_name,
        laser_power,
    ]
    seq_args_string = tool_belt.encode_seq_args(seq_args)
    ret_vals = cxn.pulse_streamer.stream_load(seq_file_name, seq_args_string)
    seq_time = ret_vals[0]
    print(seq_args)
    # return
    #    print(seq_time)

    # %% Let the user know how long this will take

    seq_time_s = seq_time / (10 ** 9)  # to seconds
    expected_run_time_s = (
        (num_steps / 2) * num_reps * num_runs * seq_time_s
    )  # s
    expected_run_time_m = expected_run_time_s / 60  # to minutes

    print(" \nExpected run time: {:.1f} minutes. ".format(expected_run_time_m))
    #return
    
    # create figure
    raw_fig, axes_pack = plt.subplots(1, 2, figsize=(17, 8.5))
    
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

        # Optimize
        opti_coords = optimize.main_with_cxn(cxn, nv_sig, apd_indices)
        opti_coords_list.append(opti_coords)

        # Set up the microwaves
        sig_gen_cxn = tool_belt.get_signal_generator_cxn(cxn, state)
        sig_gen_cxn.set_freq(uwave_freq)
        sig_gen_cxn.set_amp(uwave_power)
        sig_gen_cxn.load_iq()
        sig_gen_cxn.uwave_on()
        
        cxn.arbitrary_waveform_generator.load_xy4n(num_xy4_reps)
        # cxn.arbitrary_waveform_generator.load_cpmg(num_dd_reps)
        

        # Set up the laser
        tool_belt.set_filter(cxn, nv_sig, laser_key)
        laser_power = tool_belt.set_laser_power(cxn, nv_sig, laser_key)

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

            print(" \nFirst relaxation time: {}".format(taus[tau_ind_first]))
            print("Second relaxation time: {}".format(taus[tau_ind_second]))

            seq_args = [
                taus[tau_ind_first],
                polarization_time,
                gate_time,
                uwave_pi_pulse,
                uwave_pi_on_2_pulse,
                taus[tau_ind_second],
                pi_pulse_reps,
                apd_indices[0],
                state.value,
                laser_name,
                laser_power,
            ]
            # print(seq_args)
            # return
            seq_args_string = tool_belt.encode_seq_args(seq_args)
            # Clear the tagger buffer of any excess counts
            # cxn.apd_tagger.clear_buffer()
            cxn.pulse_streamer.stream_immediate(
                seq_file_name, num_reps, seq_args_string
            )

            # Each sample is of the form [*(<sig_shrt>, <ref_shrt>, <sig_long>, <ref_long>)]
            # So we can sum on the values for similar index modulus 4 to
            # parse the returned list into what we want.
            new_counts = cxn.apd_tagger.read_counter_separate_gates(1)
            sample_counts = new_counts[0]
            # print(new_counts)

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

        cxn.apd_tagger.stop_tag_stream()

        # %% incremental plotting
        
        #Average the counts over the iterations
        avg_sig_counts = numpy.average(sig_counts[:(run_ind+1)], axis=0)
        avg_ref_counts = numpy.average(ref_counts[:(run_ind+1)], axis=0)
        # print(numpy.average(avg_ref_counts))
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
        ax.set_xlabel(r"Precession time, $T = 2*4*N*\tau (\mathrm{\mu s}$)")
        ax.set_ylabel("Counts")
        ax.legend()
        
        ax = axes_pack[1]
        ax.cla()
        ax.plot(plot_taus, norm_avg_sig, "b-")
        ax.set_title("XY4-{} Measurement".format(num_xy4_reps))
        ax.set_xlabel(r"Precession time, $T = 2*4*N*\tau (\mathrm{\mu s}$)")
        ax.set_ylabel("Contrast (arb. units)")
        
        text_popt = 'Run # {}/{}'.format(run_ind+1,num_runs)

        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.8, 0.9, text_popt,transform=ax.transAxes,
                verticalalignment='top', bbox=props)
        
        raw_fig.canvas.draw()
        raw_fig.set_tight_layout(True)
        raw_fig.canvas.flush_events()
        
        # %% Save the data we have incrementally for long T1s

        raw_data = {
            "start_timestamp": start_timestamp,
            "nv_sig": nv_sig,
            "nv_sig-units": tool_belt.get_nv_sig_units(),
            'num_xy4_reps': num_xy4_reps,
            "gate_time": gate_time,
            "gate_time-units": "ns",
            "uwave_freq": uwave_freq,
            "uwave_freq-units": "GHz",
            "uwave_power": uwave_power,
            "uwave_power-units": "dBm",
            "uwave_pi_pulse": uwave_pi_pulse,
            "uwave_pi_pulse-units": "ns",
            "uwave_pi_on_2_pulse": uwave_pi_on_2_pulse,
            "uwave_pi_on_2_pulse-units": "ns",
            "precession_time_range": precession_time_range,
            "precession_time_range-units": "ns",
            "state": state.name,
            "num_steps": num_steps,
            "num_reps": num_reps,
            "run_ind": run_ind,
            "taus": taus.tolist(),
            "plot_taus":plot_taus.tolist(),
            "taus-units": "ns",
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
        tool_belt.save_figure(raw_fig, file_path)

    # %% Hardware clean up

    tool_belt.reset_cfm(cxn)

    # %% Plot the data

    ax = axes_pack[0]
    ax.cla()
    ax.plot(plot_taus, avg_sig_counts, "r-", label="signal")
    ax.plot(plot_taus, avg_ref_counts, "g-", label="reference")
    ax.set_xlabel(r"Precession time, $T = 2*4*N*\tau (\mathrm{\mu s}$)")
    ax.set_ylabel("Counts")
    ax.legend()

    ax = axes_pack[1]
    ax.cla()
    ax.plot(plot_taus, norm_avg_sig, "b-")
    ax.set_title("XY4-{} Measurement".format(num_xy4_reps))
    ax.set_xlabel(r"Precession time, $T = 2*4*N*\tau (\mathrm{\mu s}$)")
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
        'num_xy4_reps': num_xy4_reps,
        "gate_time": gate_time,
        "gate_time-units": "ns",
        "uwave_freq": uwave_freq,
        "uwave_freq-units": "GHz",
        "uwave_power": uwave_power,
        "uwave_power-units": "dBm",
        "uwave_pi_pulse": uwave_pi_pulse,
        "uwave_pi_pulse-units": "ns",
        "uwave_pi_on_2_pulse": uwave_pi_on_2_pulse,
        "uwave_pi_on_2_pulse-units": "ns",
        "precession_time_range": precession_time_range,
        "precession_time_range-units": "ns",
        "state": state.name,
        "num_steps": num_steps,
        "num_reps": num_reps,
        "num_runs": num_runs,
        "taus": taus.tolist(),
        "plot_taus":plot_taus.tolist(),
        "taus-units": "ns",
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

    nv_name = nv_sig["name"]
    file_path = tool_belt.get_file_path(__file__, timestamp, nv_name)
    tool_belt.save_figure(raw_fig, file_path)
    tool_belt.save_raw_data(raw_data, file_path)

    # %% Fit and save figs


    return 


# %% Run the file


if __name__ == "__main__":
    
    folder= 'pc_rabi/branch_master/dynamical_decoupling_xy4/2022_08'
    file1 = '2022_08_31-01_53_17-rubin-nv1_2022_08_10'
    file4 = '2022_08_31-02_41_31-rubin-nv4_2022_08_10'
    file5 = '2022_08_31-03_29_44-rubin-nv5_2022_08_10'
    file8 = '2022_08_31-04_18_27-rubin-nv8_2022_08_10'
    
    file = '2022_08_31-23_31_16-rubin-nv4_2022_08_10'
    
    file_list = []
    fig, ax = plt.subplots()
    
    # for file in file_list:
    data = tool_belt.get_raw_data(file, folder)
    taus = numpy.array(data['taus'])
    num_xy4_reps = data['num_xy4_reps']
    norm_avg_sig = data['norm_avg_sig']
    num_steps=data['num_steps']
    nv_sig = data['nv_sig']
    
    
    tau_step = taus[1]-taus[0]
    plot_taus = (taus * 2 *4* num_xy4_reps) / 1000
    
    ax.plot(plot_taus, norm_avg_sig, 'bo')
    ax.set_title("XY4-{} Measurement".format(num_xy4_reps))
    ax.set_xlabel(r"Precession time, $T = 2*4*N*\tau (\mathrm{\mu s}$)")
    ax.set_ylabel("Contrast (arb. units)")
    # ax.legend()
    
    revival_t = nv_sig['t2_revival_time']/1e3
    for i in range(6+1):
        rev_t_mod = i * revival_t * 2 * 4 * num_xy4_reps
        ax.axvline(x=rev_t_mod, c='grey', linestyle='--')

    # transform = numpy.fft.rfft(norm_avg_sig)
    # freqs = numpy.fft.rfftfreq(num_steps, d=tau_step/1000)
    # transform_mag = numpy.absolute(transform)
    # fig, ax = plt.subplots()
    # ax.plot(freqs, transform_mag)
