# -*- coding: utf-8 -*-
"""
T1 measurement routine, with experiments interleaved.

We've made a variety of T1 routines, each with different options or variations.
This version streamlines those options into a single file where the options
are passed into the main function. (Interleaving, running each experiment
once then moving onto the next experiment as opposed to completing all the runs
for one experiment then movning onto the next experiment, is not optional since
not interleaving can introduce systematics probably related to charge
dynamics.)

Pass into the function an experiment array, and it will run each experiment
one run at a time. That way, in post processing, we can split the data up by
num_run and see smaller time scales if the values are changing.

The num_runs of each experiment MUST BE THE SAME.

Created on December 16, 2021

@author: mccambria
"""

# %% Imports


import utils.tool_belt as tool_belt
import utils.common as common
import majorroutines.optimize as optimize
import numpy
import os
import time
from random import shuffle
import matplotlib.pyplot as plt
import labrad
from utils.tool_belt import States
import shutil
import analysis.relaxation_rate_analysis as relaxation_rate_analysis
from pathlib import Path


# %% Functions


def collate_incremental(path, folder):

    nvdata_dir = common.get_nvdata_dir()
    path_from_nvdata = "{}/{}/incremental".format(path, folder)
    full_path_to_folder = nvdata_dir / path_from_nvdata

    coll_data = {}
    first_file = True
    coll_sig_counts_master_list = None
    coll_ref_counts_master_list = None
    coll_tau_ind_save_list = None
    coll_opti_coords_master_list = None
    coll_params_master_list = None
    coll_tau_master_list = None
    coll_nv_sig = None
    total_num_runs = 0

    for filename_with_ext in os.listdir(full_path_to_folder):
        filename = os.path.splitext(filename_with_ext)[0]
        # Skip over the collated incremental file if we already built one
        if filename.endswith("collated_incremental"):
            continue
        full_path_to_file = full_path_to_folder / filename_with_ext
        if not os.path.isfile(full_path_to_file):
            continue
        data = tool_belt.get_raw_data(filename, path_from_nvdata, nvdata_dir)

        sig_counts_master_list = data["sig_counts_master_list"]
        ref_counts_master_list = data["ref_counts_master_list"]
        params_master_list = data["params_master_list"]
        tau_ind_save_list = data["tau_ind_save_list"]
        opti_coords_master_list = data["opti_coords_master_list"]
        tau_master_list = data["tau_master_list"]
        nv_sig = data["nv_sig"]
        if "run_ind" in data:
            num_runs = data["run_ind"]
        else:
            num_runs = data["num_runs"]
        # Just skip data sets that ran for less than 5 runs
        if num_runs < 5:
            continue
        # Throw out the last run where we presumably crashed out
        accepted_num_runs = num_runs - 1
        total_num_runs += accepted_num_runs

        # Make sure everything that should match up does
        if first_file:
            coll_params_master_list = params_master_list
            common_params = [el[:4] for el in params_master_list]
            coll_tau_master_list = tau_master_list
            coll_nv_sig = nv_sig
        test_common_params = [el[:4] for el in params_master_list]
        if (
            (common_params != test_common_params)
            or (coll_tau_master_list != tau_master_list)
            or (coll_nv_sig["name"] != nv_sig["name"])
        ):
            raise RuntimeError("Inconsistent experiments being collated.")

        # Add what needs to be added to the collated data
        num_exps = len(params_master_list)
        if first_file:
            coll_sig_counts_master_list = [[] for ind in range(num_exps)]
            coll_ref_counts_master_list = [[] for ind in range(num_exps)]
            coll_tau_ind_save_list = [[] for ind in range(num_exps)]
            coll_opti_coords_master_list = [[] for ind in range(num_exps)]

        for ind in range(num_exps):
            coll_sig_counts_master_list[ind].extend(
                sig_counts_master_list[ind][:accepted_num_runs]
            )
            coll_ref_counts_master_list[ind].extend(
                ref_counts_master_list[ind][:accepted_num_runs]
            )
            coll_tau_ind_save_list[ind].extend(
                tau_ind_save_list[ind][:accepted_num_runs]
            )
            coll_opti_coords_master_list[ind].extend(
                opti_coords_master_list[ind][:accepted_num_runs]
            )
        first_file = False

    coll_data["sig_counts_master_list"] = coll_sig_counts_master_list
    coll_data["ref_counts_master_list"] = coll_ref_counts_master_list
    coll_data["params_master_list"] = coll_params_master_list
    coll_data["tau_ind_save_list"] = coll_tau_ind_save_list
    coll_data["opti_coords_master_list"] = coll_opti_coords_master_list
    coll_data["tau_master_list"] = coll_tau_master_list
    coll_data["nv_sig"] = coll_nv_sig
    coll_data["num_runs"] = total_num_runs
    # coll_data["num_runs"] = 150

    coll_incs_file_name = "collated_incremental"
    full_path_to_coll_incs = full_path_to_folder / coll_incs_file_name
    tool_belt.save_raw_data(coll_data, str(full_path_to_coll_incs))


def unpack_interleave(data, start_run=0, stop_run=None):
    """
    Save the interleaved data into separate files and plots for each
    experiment. Processes the data between start_run: stop_run. If stop_run is
    None, processes all fully completed runs.
    """

    sig_counts_master_list = data["sig_counts_master_list"]
    ref_counts_master_list = data["ref_counts_master_list"]
    params_master_list = data["params_master_list"]
    tau_ind_save_list = data["tau_ind_save_list"]
    opti_coords_master_list = data["opti_coords_master_list"]
    tau_master_list = data["tau_master_list"]
    nv_sig = data["nv_sig"]
    gate_time = nv_sig["spin_readout_dur"]
    if stop_run is None:
        # Assume incremental
        if "run_ind" in data:
            stop_run = data["run_ind"]
        else:
            stop_run = data["num_runs"]
    num_runs = stop_run - start_run
    avg_sig_counts_master_list = []
    avg_ref_counts_master_list = []
    norm_sig_counts_master_list = []

    # List of the generated file names
    file_names = []

    # %% Average the counts over the iterations, for each experiment
    num_exp = len(params_master_list)
    for exp_ind in range(num_exp):
        sig_counts = sig_counts_master_list[exp_ind]
        ref_counts = ref_counts_master_list[exp_ind]

        # Clip according to the number of completed runs
        sig_counts = sig_counts[start_run:stop_run][:]
        ref_counts = ref_counts[start_run:stop_run][:]

        avg_sig_counts = numpy.average(sig_counts, axis=0)
        avg_ref_counts = numpy.average(ref_counts, axis=0)

        avg_sig_counts_master_list.append(avg_sig_counts.tolist())
        avg_ref_counts_master_list.append(avg_ref_counts.tolist())

        # Replace x/0=inf with 0
        try:
            norm_avg_sig = avg_sig_counts / avg_ref_counts
        except RuntimeWarning as e:
            print(e)
            inf_mask = numpy.isinf(norm_avg_sig)
            # Assign to 0 based on the passed conditional array
            norm_avg_sig[inf_mask] = 0

        norm_sig_counts_master_list.append(norm_avg_sig.tolist())

        # %% Plot and save each experiment data

        # Extract the params for each experiment
        init_state_name = params_master_list[exp_ind][0][0]
        read_state_name = params_master_list[exp_ind][0][1]
        relaxation_time_range = params_master_list[exp_ind][1]
        num_steps = params_master_list[exp_ind][2]
        num_reps = params_master_list[exp_ind][3]
        uwave_pi_pulse_init = params_master_list[exp_ind][4]
        uwave_freq_init = params_master_list[exp_ind][5]
        uwave_power_init = params_master_list[exp_ind][6]
        uwave_pi_pulse_read = params_master_list[exp_ind][7]
        uwave_freq_read = params_master_list[exp_ind][8]
        uwave_power_read = params_master_list[exp_ind][9]

        tau_index_master_list = tau_ind_save_list[exp_ind]
        opti_coords_list = opti_coords_master_list[exp_ind]

        #        sig_counts = sig_counts_master_list[exp_ind]
        #        ref_counts = ref_counts_master_list[exp_ind]
        #        norm_avg_sig = norm_sig_counts_master_list[exp_ind]

        taus = tau_master_list[exp_ind]

        # Plot
        individual_fig, axes_pack = plt.subplots(1, 2, figsize=(17, 8.5))

        ax = axes_pack[0]
        ax.plot(
            numpy.array(taus) / 10 ** 6, avg_sig_counts, "r-", label="signal"
        )
        ax.plot(
            numpy.array(taus) / 10 ** 6,
            avg_ref_counts,
            "g-",
            label="reference",
        )
        ax.set_xlabel("Relaxation time (ms)")
        ax.set_ylabel("Counts")
        ax.legend()

        ax = axes_pack[1]
        ax.plot(numpy.array(taus) / 10 ** 6, norm_avg_sig, "b-")
        ax.set_title(
            "T1 Measurement. Initial state: {}, readout state: {}".format(
                init_state_name, read_state_name
            )
        )
        ax.set_xlabel("Relaxation time (ms)")
        ax.set_ylabel("Contrast (arb. units)")

        individual_fig.canvas.draw()
        # fig.set_tight_layout(True)
        individual_fig.canvas.flush_events()

        timestamp = tool_belt.get_time_stamp()

        individual_raw_data = {
            "timestamp": timestamp,
            #            'timeElapsed': timeElapsed,
            "init_state": init_state_name,
            "read_state": read_state_name,
            "nv_sig": nv_sig,
            "nv_sig-units": tool_belt.get_nv_sig_units(),
            "gate_time": gate_time,
            "gate_time-units": "ns",
            "uwave_freq_init": uwave_freq_init,
            "uwave_freq_init-units": "GHz",
            "uwave_freq_read": uwave_freq_read,
            "uwave_freq_read-units": "GHz",
            "uwave_power_init": uwave_power_init,
            "uwave_power_init-units": "dBm",
            "uwave_power_read": uwave_power_read,
            "uwave_power_read-units": "dBm",
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
            "sig_counts": sig_counts,
            "sig_counts-units": "counts",
            "ref_counts": ref_counts,
            "ref_counts-units": "counts",
            "norm_avg_sig": norm_avg_sig.astype(float).tolist(),
            "norm_avg_sig-units": "arb",
        }

        # Save each figure
        dosave = True
        if dosave:
            file_path = tool_belt.get_file_path(
                __file__, timestamp, nv_sig["name"]
            )
            tool_belt.save_raw_data(individual_raw_data, file_path)
            tool_belt.save_figure(individual_fig, file_path)

            file_names.append(Path(file_path).name)

            # Sleep for 1.1 seconds so the files don't save over each other
            time.sleep(1.1)

    return file_names


# %% Main


def main(
    nv_sig,
    apd_indices,
    t1_exp_array,
    num_runs,
    composite_pulses=False,
    scc_readout=False,
):

    with labrad.connect() as cxn:
        main_with_cxn(
            cxn,
            nv_sig,
            apd_indices,
            t1_exp_array,
            num_runs,
            composite_pulses,
            scc_readout,
        )


def main_with_cxn(
    cxn,
    nv_sig,
    apd_indices,
    t1_exp_array,
    num_runs,
    composite_pulses=False,
    scc_readout=False,
):

    tool_belt.reset_cfm(cxn)

    # %% Optical setup
    if scc_readout:

        laser_tag = "nv-_reionization"
        laser_key = "{}_laser".format(laser_tag)
        pol_laser_name = nv_sig[laser_key]
        pol_laser_power = tool_belt.set_laser_power(cxn, nv_sig, laser_key)
        polarization_dur = nv_sig["{}_dur".format(laser_tag)]

        laser_tag = "nv0_ionization"
        laser_key = "{}_laser".format(laser_tag)
        ion_laser_name = nv_sig[laser_key]
        ion_laser_power = tool_belt.set_laser_power(cxn, nv_sig, laser_key)
        ionization_dur = nv_sig["{}_dur".format(laser_tag)]

        laser_tag = "spin_shelf"
        laser_key = "{}_laser".format(laser_tag)
        shelf_laser_name = nv_sig[laser_key]
        shelf_laser_power = tool_belt.set_laser_power(cxn, nv_sig, laser_key)
        shelf_dur = nv_sig["{}_dur".format(laser_tag)]

        laser_tag = "charge_readout"
        laser_key = "{}_laser".format(laser_tag)
        readout_laser_name = nv_sig[laser_key]
        readout_laser_power = tool_belt.set_laser_power(cxn, nv_sig, laser_key)
        readout_dur = nv_sig["{}_dur".format(laser_tag)]

    else:
        laser_key = "spin_laser"
        laser_name = nv_sig[laser_key]
        laser_power = tool_belt.set_laser_power(cxn, nv_sig, laser_key)
        polarization_time = nv_sig["spin_pol_dur"]
        readout = nv_sig["spin_readout_dur"]

    # %% Microwave setup

    uwave_pi_pulse_high = round(nv_sig["rabi_HIGH"] / 2)
    uwave_pi_pulse_low = round(nv_sig["rabi_LOW"] / 2)
    uwave_freq_high = nv_sig["resonance_HIGH"]
    uwave_freq_low = nv_sig["resonance_LOW"]
    uwave_power_high = nv_sig["uwave_power_HIGH"]
    uwave_power_low = nv_sig["uwave_power_LOW"]

    # %% Extract the number of experiments

    num_exp = len(t1_exp_array)

    # %% Create lists, that the first index will refer to the experiment #

    # Lists to fill with taus for each experiment. The index master list is
    # to store the indexes, but is not the one we will save at the end
    tau_master_list = []
    tau_ind_master_list = []

    # Lists to store the sig/ref counts for each experiment
    sig_counts_master_list = []
    ref_counts_master_list = []

    avg_sig_counts_master_list = []
    avg_ref_counts_master_list = []

    norm_sig_counts_master_list = []

    # List for the parameters for each experiment to save at end
    # Format: [[init_state, read_state], relaxation_time_range, num_steps,
    #                       num_reps, init_pi, init_freq, init_pwr,
    #                       read_pi, read_freq, read_pwr]
    params_master_list = [[] for i in range(num_exp)]

    # Nested lists for opticoord and shuffled tau_ind for each experiment
    opti_coords_master_list = [[] for i in range(num_exp)]
    tau_ind_save_list = [[[] for i in range(num_runs)] for i in range(num_exp)]

    # Trivial list to add the seq times to obtain a total run time
    exp_time_list = []

    # %% Extract the needed information for each experiment and add them to lists

    for exp_ind in range(num_exp):
        # Extract the info for each experiment
        init_state = t1_exp_array[exp_ind][0][0]
        read_state = t1_exp_array[exp_ind][0][1]
        min_relaxation_time = int(t1_exp_array[exp_ind][1][0])
        max_relaxation_time = int(t1_exp_array[exp_ind][1][1])
        num_steps = t1_exp_array[exp_ind][2]
        num_reps = t1_exp_array[exp_ind][3]

        # For each experiment, take the time values and create a linspace
        taus = numpy.linspace(
            min_relaxation_time,
            max_relaxation_time,
            num=num_steps,
            dtype=numpy.int64,
        )
        tau_master_list.append(taus.tolist())

        # also calculate the half length for each tau list to step through,
        # and later shuffle
        if len(taus) % 2 == 0:
            half_length_taus = int(len(taus) / 2)
        elif len(taus) % 2 == 1:
            half_length_taus = int((len(taus) + 1) / 2)

        # Then we must use this half length to calculate the list of integers to be
        # shuffled for each run

        tau_ind_list = list(range(0, half_length_taus))
        tau_ind_master_list.append(tau_ind_list)

        # Create empty arrays to fill with data, the indexing will be [exp_ind][num_run][num_steps]

        # append the list with each experiment's specific sized arrays
        sig_count_single = numpy.empty(
            [num_runs, num_steps], dtype=numpy.float32
        )
        sig_count_single[:] = numpy.nan
        ref_count_single = numpy.copy(sig_count_single)

        sig_counts_master_list.append(sig_count_single.tolist())
        ref_counts_master_list.append(ref_count_single.tolist())

        # Create a list of the init and read params per experiment
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

        # Append the values for each experiment into the master list
        # Format: [[init_state, read_state], relaxation_time_range, num_steps,
        #                       num_reps, init_pi, init_freq, init_pwr,
        #                       read_pi, read_freq, read_pwr]
        params_master_list[exp_ind].append([init_state.name, read_state.name])
        params_master_list[exp_ind].append(t1_exp_array[exp_ind][1])
        params_master_list[exp_ind].append(num_steps)
        params_master_list[exp_ind].append(num_reps)
        params_master_list[exp_ind].append(uwave_pi_pulse_init)
        params_master_list[exp_ind].append(uwave_freq_init)
        params_master_list[exp_ind].append(uwave_power_init)
        params_master_list[exp_ind].append(uwave_pi_pulse_read)
        params_master_list[exp_ind].append(uwave_freq_read)
        params_master_list[exp_ind].append(uwave_power_read)

        # %% Analyze the sequence

        # file_name = os.path.basename(__file__)
        if scc_readout and composite_pulses:
            raise (
                RuntimeError(
                    "Composite pulses with scc readout not yet implemented!"
                )
            )
        if scc_readout:
            seq_file = "t1_dq_scc.py"
            seq_args = [
                min_relaxation_time,
                None,
                None,
                uwave_pi_pulse_low,
                uwave_pi_pulse_high,
                max_relaxation_time,
                apd_indices[0],
                init_state.value,
                read_state.value,
                None,
                None,
                pol_laser_name,
                pol_laser_power,
                polarization_dur,
                ion_laser_name,
                ion_laser_power,
                ionization_dur,
                shelf_laser_name,
                shelf_laser_power,
                shelf_dur,
                readout_laser_name,
                readout_laser_power,
                readout_dur,
            ]
        elif composite_pulses:
            seq_file = "t1_dq_knill2.py"
            seq_args = [
                min_relaxation_time,
                polarization_time,
                readout,
                uwave_pi_pulse_low,
                uwave_pi_pulse_high,
                max_relaxation_time,
                apd_indices[0],
                init_state.value,
                read_state.value,
                laser_name,
                laser_power,
            ]
        else:
            seq_file = "t1_double_quantum.py"
            seq_args = [
                min_relaxation_time,
                polarization_time,
                readout,
                uwave_pi_pulse_low,
                uwave_pi_pulse_high,
                max_relaxation_time,
                apd_indices[0],
                init_state.value,
                read_state.value,
                laser_name,
                laser_power,
            ]
        # print(seq_args)
        # continue
        seq_args_string = tool_belt.encode_seq_args(seq_args)
        ret_vals = cxn.pulse_streamer.stream_load(seq_file, seq_args_string)
        seq_time = numpy.int64(ret_vals[0])

        seq_time_s = seq_time / (10 ** 9)  # s
        expected_run_time = (
            num_steps * num_reps * num_runs * seq_time_s / 2
        )  # s
        expected_run_time_m = expected_run_time / 60  # m

        exp_time_list.append(expected_run_time_m)

    # return

    # %% Report the total time for the experiment

    # Add up the total time for the experiment
    total_exp_time = sum(exp_time_list)
    total_exp_time_h = total_exp_time / 60  # h

    print(
        " \nExpected run time for entire experiment: {:.1f} hours. ".format(
            total_exp_time_h
        )
    )
    #    return

    # %% Get the starting time of the function, to be used to calculate run time

    startFunctionTime = time.time()
    start_timestamp = tool_belt.get_time_stamp()

    # %% Collect the data

    # Start 'Press enter to stop...'
    tool_belt.init_safe_stop()

    # %% Start one of the runs

    for run_ind in range(num_runs):

        # Break out of the while if the user says stop
        if tool_belt.safe_stop():
            break

        for exp_ind in range(num_exp):

            # Define the values for this experiment
            init_state = t1_exp_array[exp_ind][0][0]
            read_state = t1_exp_array[exp_ind][0][1]
            num_reps = t1_exp_array[exp_ind][3]
            taus = tau_master_list[exp_ind]

            print(" \nOptimizing...\n")
            # Optimize
            opti_coords = optimize.main_with_cxn(cxn, nv_sig, apd_indices)
            # if opti_coords is None:
            #     return
            opti_coords_master_list[exp_ind].append(opti_coords)

            # Set up the microwaves for the low and high states
            low_sig_gen_cxn = tool_belt.get_signal_generator_cxn(
                cxn, States.LOW
            )
            low_sig_gen_cxn.set_freq(uwave_freq_low)
            low_sig_gen_cxn.set_amp(uwave_power_low)
            if composite_pulses:
                low_sig_gen_cxn.load_iq()
            low_sig_gen_cxn.uwave_on()

            high_sig_gen_cxn = tool_belt.get_signal_generator_cxn(
                cxn, States.HIGH
            )
            high_sig_gen_cxn.set_freq(uwave_freq_high)
            high_sig_gen_cxn.set_amp(uwave_power_high)
            if composite_pulses:
                high_sig_gen_cxn.load_iq()
            high_sig_gen_cxn.uwave_on()

            if composite_pulses:
                cxn.arbitrary_waveform_generator.load_knill()

            # Set up the lasers
            if scc_readout:
                laser_tags = [
                    "nv-_reionization",
                    "nv0_ionization",
                    "spin_shelf",
                    "charge_readout",
                ]
                for tag in laser_tags:
                    laser_key = "{}_laser".format(tag)
                    laser_power = tool_belt.set_laser_power(
                        cxn, nv_sig, laser_key
                    )
                    tool_belt.set_filter(cxn, nv_sig, laser_key)
            else:
                laser_key = "spin_laser"
                laser_power = tool_belt.set_laser_power(cxn, nv_sig, laser_key)
                tool_belt.set_filter(cxn, nv_sig, laser_key)

            print(
                "\nStarting experiment: ({}, {}) on run_ind: {}".format(
                    init_state.name, read_state.name, run_ind
                )
            )

            # Load the APD
            cxn.apd_tagger.start_tag_stream(apd_indices)

            # Shuffle the list of tau indices so that it steps thru them randomly
            shuffle(tau_ind_master_list[exp_ind])

            for tau_ind in tau_ind_master_list[exp_ind]:

                # 'Flip a coin' to determine which tau (long/shrt) is used first
                rand_boolean = numpy.random.randint(0, high=2)

                if rand_boolean == 1:
                    tau_ind_first = tau_ind
                    tau_ind_second = -tau_ind - 1
                elif rand_boolean == 0:
                    tau_ind_first = -tau_ind - 1
                    tau_ind_second = tau_ind

                # add the tau indexxes used to a list to save at the end
                tau_ind_save_list[exp_ind][run_ind].append(tau_ind_first)
                tau_ind_save_list[exp_ind][run_ind].append(tau_ind_second)

                # Break out of the while if the user says stop
                if tool_belt.safe_stop():
                    break

                print(
                    " \nFirst relaxation time: {}".format(taus[tau_ind_first])
                )
                print(
                    "Second relaxation time: {}".format(taus[tau_ind_second])
                )

                # Stream the sequence
                seq_args[0] = taus[tau_ind_first]
                seq_args[5] = taus[tau_ind_second]
                seq_args[7] = init_state.value
                seq_args[8] = read_state.value
                seq_args_string = tool_belt.encode_seq_args(seq_args)

                cxn.pulse_streamer.stream_immediate(
                    seq_file, int(num_reps), seq_args_string
                )

                # Each sample is of the form [*(<sig_shrt>, <ref_shrt>, <sig_long>, <ref_long>)]
                # So we can sum on the values for similar index modulus 4 to
                # parse the returned list into what we want.
                new_counts = cxn.apd_tagger.read_counter_separate_gates(1)
                sample_counts = new_counts[0]
                # print(len(sample_counts))

                count = sum(sample_counts[0::4])
                sig_counts_master_list[exp_ind][run_ind][tau_ind_first] = int(
                    count
                )
                print("First signal = " + str(count))

                count = sum(sample_counts[1::4])
                ref_counts_master_list[exp_ind][run_ind][tau_ind_first] = int(
                    count
                )
                print("First Reference = " + str(count))

                count = sum(sample_counts[2::4])
                sig_counts_master_list[exp_ind][run_ind][tau_ind_second] = int(
                    count
                )
                print("Second Signal = " + str(count))

                count = sum(sample_counts[3::4])
                ref_counts_master_list[exp_ind][run_ind][tau_ind_second] = int(
                    count
                )
                print("Second Reference = " + str(count))

            cxn.apd_tagger.stop_tag_stream()

        # %% Save the data we have incrementally for long measurements

        incr_data = {
            "start_timestamp": start_timestamp,
            "nv_sig": nv_sig,
            "nv_sig-units": tool_belt.get_nv_sig_units(),
            "run_ind": run_ind,
            "params_master_list": params_master_list,
            "params_master_list-format": (
                "[[init_state, read_state],relaxation range, num_steps,"
                " num_reps, uwave_pi_pulse_init, uwave_freq_init,"
                " uwave_power_init, uwave_pi_pulse_read, uwave_freq_read,"
                " uwave_power_read]"
            ),
            "params_master_list-units": (
                "[[null, null], [ns, ns], null, null, ns, GHz, dBm, ns, GHz,"
                " dBm]"
            ),
            "tau_master_list": tau_master_list,
            "tau_master_list-units": "ns",
            "tau_ind_save_list": tau_ind_save_list,
            "opti_coords_master_list": opti_coords_master_list,
            "opti_coords_master_list-units": "V",
            "sig_counts_master_list": sig_counts_master_list,
            "sig_counts_master_list-units": "counts",
            "ref_counts_master_list": ref_counts_master_list,
            "ref_counts_master_list-units": "counts",
        }

        #        # This will continuously be the same file path so we will overwrite
        #        # the existing file with the latest version
        file_path = tool_belt.get_file_path(
            __file__, start_timestamp, nv_sig["name"], "incremental"
        )
        tool_belt.save_raw_data(incr_data, file_path)

    # %% Hardware clean up

    tool_belt.reset_cfm(cxn)

    # %% Save the data

    # At this point incr_data is all the data so just unpack it
    unpack_interleave(incr_data)

    endFunctionTime = time.time()

    timeElapsed = endFunctionTime - startFunctionTime

    timestamp = tool_belt.get_time_stamp()

    full_data = {
        "timestamp": timestamp,
        "timeElapsed": timeElapsed,
        "nv_sig": nv_sig,
        "nv_sig-units": tool_belt.get_nv_sig_units(),
        "num_runs": num_runs,
        "params_master_list": params_master_list,
        "params_master_list-format": (
            "[[init_state, read_state],relaxation range, num_steps, num_reps,"
            " uwave_pi_pulse_init, uwave_freq_init, uwave_power_init,"
            " uwave_pi_pulse_read, uwave_freq_read, uwave_power_read]"
        ),
        "params_master_list-units": (
            "[[null, null], [ns, ns], null, null, ns, GHz, dBm, ns, GHz, dBm]"
        ),
        "tau_master_list": tau_master_list,
        "tau_master_list-units": "ns",
        "tau_ind_save_list": tau_ind_save_list,
        "opti_coords_master_list": opti_coords_master_list,
        "opti_coords_master_list-units": "V",
        "norm_sig_counts_master_list": norm_sig_counts_master_list,
        "norm_sig_counts_master_list-units": "arb",
        "avg_sig_counts_master_list": avg_sig_counts_master_list,
        "avg_sig_counts_master_list-units": "counts",
        "avg_ref_counts_master_list": avg_ref_counts_master_list,
        "avg_ref_counts_master_list-units": "counts",
        "sig_counts_master_list": sig_counts_master_list,
        "sig_counts_master_list-units": "counts",
        "ref_counts_master_list": ref_counts_master_list,
        "ref_counts_master_list-units": "counts",
    }

    file_path = tool_belt.get_file_path(__file__, timestamp, nv_sig["name"])
    tool_belt.save_raw_data(full_data, file_path)


# %% Run the file


if __name__ == "__main__":

    path = "pc_hahn/branch_master/t1_dq_main/data_collections"
    folder = "wu-nv6_2022_04_14-295K"
    # folder = "wu-nv1_2022_03_16-295K"
    collate_incremental(path, folder)
    full_path_to_incremental = "{}/{}/incremental".format(path, folder)
    data = tool_belt.get_raw_data(
        "collated_incremental", full_path_to_incremental
    )

    # start = 75
    # file_names = unpack_interleave(data, start, start + 24)
    file_names = unpack_interleave(data)

    save_folder_path = common.get_nvdata_dir() / path / folder
    for el in file_names:
        path_to_file = tool_belt.get_raw_data_path(el)
        shutil.copy(str(path_to_file), str(save_folder_path))
        shutil.copy(
            str(path_to_file.with_suffix(".svg")), str(save_folder_path)
        )

    # plt.show(block=True)
