# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 13:40:36 2020

This routine performs Rabi, but readouts with SCC

This routine tests rabi under various readout routines: regular green readout,
regular yellow readout, and SCC readout.

@author: agardill
"""

# %% Imports

import os
import time
from random import shuffle

import labrad
import matplotlib.pyplot as plt
import numpy
from scipy.optimize import curve_fit

import majorroutines.targeting as targeting
import utils.kplotlib as kpl
import utils.positioning as positioning
import utils.tool_belt as tool_belt
from majorroutines.rabi import (
    create_fit_figure,
    create_raw_data_figure,
    fit_data,
    simulate,
)
from utils.kplotlib import KplColors

# %% Main


def main(
    nv_sig,
    uwave_time_range,
    state,
    num_steps,
    num_reps,
    num_runs,
    opti_nv_sig=None,
    return_popt=False,
):
    with labrad.connect() as cxn:
        rabi_per, sig_counts, ref_counts, popt = main_with_cxn(
            cxn,
            nv_sig,
            uwave_time_range,
            state,
            num_steps,
            num_reps,
            num_runs,
            opti_nv_sig,
        )

        if return_popt:
            return rabi_per, popt
        if not return_popt:
            return rabi_per


def main_with_cxn(
    cxn,
    nv_sig,
    uwave_time_range,
    state,
    num_steps,
    num_reps,
    num_runs,
    opti_nv_sig=None,
):
    counter_server = tool_belt.get_server_counter(cxn)
    pulsegen_server = tool_belt.get_server_pulse_gen(cxn)
    arbwavegen_server = tool_belt.get_server_arb_wave_gen(cxn)

    tool_belt.reset_cfm(cxn)
    kpl.init_kplotlib()

    # %% Get the starting time of the function, to be used to calculate run time

    startFunctionTime = time.time()
    start_timestamp = tool_belt.get_time_stamp()

    # %% Initial calculations and setup

    uwave_freq = nv_sig["resonance_{}".format(state.name)]
    uwave_power = nv_sig["uwave_power_{}".format(state.name)]

    norm_style = nv_sig["norm_style"]
    readout_time = nv_sig["charge_readout_dur"]
    readout_power = tool_belt.set_laser_power(cxn, nv_sig, "charge_readout_laser")
    ion_time = nv_sig["nv0_ionization_dur"]
    ion_power = 1
    reion_power = 1
    reion_time = nv_sig["nv-_reionization_dur"]
    shelf_time = 0  # nv_sig['spin_shelf_dur']
    shelf_power = (
        nv_sig["spin_shelf_laser_power"] if "spin_shelf_laser_power" in nv_sig else None
    )

    green_laser_name = nv_sig["nv-_reionization_laser"]
    red_laser_name = nv_sig["nv0_ionization_laser"]
    yellow_laser_name = nv_sig["charge_readout_laser"]
    sig_gen_cxn = tool_belt.get_server_sig_gen(cxn, state)
    sig_gen_name = sig_gen_cxn.name

    # Array of times to sweep through
    # Must be ints since the pulse streamer only works with int64s
    min_uwave_time = uwave_time_range[0]
    max_uwave_time = uwave_time_range[1]
    taus = numpy.linspace(
        min_uwave_time, max_uwave_time, num=num_steps, dtype=numpy.int32
    )

    # Analyze the sequence

    file_name = "rabi_scc.py"

    seq_args = [
        readout_time,
        reion_time,
        ion_time,
        max_uwave_time,
        shelf_time,
        max_uwave_time,
        green_laser_name,
        yellow_laser_name,
        red_laser_name,
        sig_gen_name,
        reion_power,
        ion_power,
        shelf_power,
        readout_power,
    ]
    print(seq_args)

    seq_args_string = tool_belt.encode_seq_args(seq_args)
    pulsegen_server.stream_load(file_name, seq_args_string)

    # Set up our data structure, an array of NaNs that we'll fill
    # incrementally. NaNs are ignored by matplotlib, which is why they're
    # useful for us here.
    # We define 2D arrays, with the horizontal dimension for the frequency and
    # the veritical dimension for the index of the run.
    sig_counts = numpy.empty([num_runs, num_steps])
    sig_counts_each_shot = numpy.zeros([num_runs, num_steps, num_reps])
    sig_counts[:] = numpy.nan
    ref_counts = numpy.copy(sig_counts)
    ref_counts_each_shot = numpy.copy(sig_counts_each_shot)
    # norm_avg_sig = numpy.empty([num_runs, num_steps])

    # %% Make some lists and variables to save at the end

    opti_coords_list = []
    tau_index_master_list = [[] for i in range(num_runs)]

    # Create a list of indices to step through the taus. This will be shuffled
    tau_ind_list = list(range(0, num_steps))

    # Create raw data figure for incremental plotting
    raw_fig, ax_sig_ref, ax_norm = create_raw_data_figure(taus)

    # Set up a run indicator for incremental plotting
    run_indicator_text = "Run #{}/{}"
    text = run_indicator_text.format(0, num_runs)
    run_indicator_obj = kpl.anchored_text(ax_norm, text, loc=kpl.Loc.UPPER_RIGHT)

    # %% Collect the data

    # Start 'Press enter to stop...'
    tool_belt.init_safe_stop()

    for run_ind in range(num_runs):
        print("Run index: {}".format(run_ind))

        # Break out of the while if the user says stop
        if tool_belt.safe_stop():
            break

        # Optimize
        if opti_nv_sig:
            opti_coords = targeting.main_with_cxn(cxn, opti_nv_sig)
            drift = positioning.get_drift(cxn)
            adj_coords = nv_sig["coords"] + numpy.array(drift)
            positioning.set_xyz(cxn, adj_coords)
        else:
            opti_coords = targeting.main_with_cxn(cxn, nv_sig)
        opti_coords_list.append(opti_coords)

        # Apply the microwaves
        sig_gen_cxn.set_freq(uwave_freq)
        sig_gen_cxn.set_amp(uwave_power)
        sig_gen_cxn.uwave_on()

        # Load the APD
        counter_server.start_tag_stream()

        # Shuffle the list of indices to use for stepping through the taus
        shuffle(tau_ind_list)

        for tau_ind in tau_ind_list:
            # Break out of the while if the user says stop
            if tool_belt.safe_stop():
                break

            # shine the red laser for a few seconds before the sequence
            # pulsegen_server.constant([7], 0.0, 0.0)
            # time.sleep(2)

            # Load the sequence
            pulsegen_server.stream_load(file_name, seq_args_string)

            # add the tau indexxes used to a list to save at the end
            tau_index_master_list[run_ind].append(tau_ind)

            # Stream the sequence

            seq_args = [
                readout_time,
                reion_time,
                ion_time,
                taus[tau_ind],
                shelf_time,
                max_uwave_time,
                green_laser_name,
                yellow_laser_name,
                red_laser_name,
                sig_gen_name,
                reion_power,
                ion_power,
                shelf_power,
                readout_power,
            ]

            seq_args_string = tool_belt.encode_seq_args(seq_args)
            # Clear the tagger buffer of any excess counts
            counter_server.clear_buffer()
            pulsegen_server.stream_immediate(file_name, num_reps, seq_args_string)

            # Get the counts
            new_counts = counter_server.read_counter_separate_gates(1)

            sample_counts = new_counts[0]
            sig_gate_counts = sample_counts[0::2]
            sig_counts[run_ind, tau_ind] = sum(sig_gate_counts)
            sig_counts_each_shot[run_ind, tau_ind] = sig_gate_counts
            ref_gate_counts = sample_counts[1::2]
            ref_counts[run_ind, tau_ind] = sum(ref_gate_counts)
            ref_counts_each_shot[run_ind, tau_ind] = ref_gate_counts

        counter_server.stop_tag_stream()

        ### Incremental plotting

        # Update the run indicator
        text = run_indicator_text.format(run_ind + 1, num_runs)
        run_indicator_obj.txt.set_text(text)

        # Average the counts over the iterations
        inc_sig_counts = sig_counts[: run_ind + 1]
        inc_ref_counts = ref_counts[: run_ind + 1]
        ret_vals = tool_belt.process_counts(
            inc_sig_counts, inc_ref_counts, num_reps, readout_time, norm_style
        )
        (
            sig_counts_avg_kcps,
            ref_counts_avg_kcps,
            norm_avg_sig,
            norm_avg_sig_ste,
        ) = ret_vals

        kpl.plot_line_update(ax_sig_ref, line_ind=0, y=sig_counts_avg_kcps)
        kpl.plot_line_update(ax_sig_ref, line_ind=1, y=ref_counts_avg_kcps)
        kpl.plot_line_update(ax_norm, y=norm_avg_sig)

        # %% Save the data we have incrementally for long measurements

        raw_data = {
            "start_timestamp": start_timestamp,
            "nv_sig": nv_sig,
            # 'nv_sig-units': tool_belt.get_nv_sig_units(),
            "shelf_power": shelf_power,
            "shelf_power-units": "mW",
            "uwave_freq": uwave_freq,
            "uwave_freq-units": "GHz",
            "uwave_power": uwave_power,
            "uwave_power-units": "dBm",
            "uwave_time_range": uwave_time_range,
            "uwave_time_range-units": "ns",
            "state": state.name,
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
            "sig_counts_each_shot": sig_counts_each_shot.astype(int).tolist(),
            "sig_counts_each_shot-units": "counts",
            "ref_counts_each_shot": ref_counts_each_shot.astype(int).tolist(),
            "ref_counts-units_each_shot": "counts",
        }

        # This will continuously be the same file path so we will overwrite
        # the existing file with the latest version
        file_path = tool_belt.get_file_path(
            __file__, start_timestamp, nv_sig["name"], "incremental"
        )
        tool_belt.save_raw_data(raw_data, file_path)
        tool_belt.save_figure(raw_fig, file_path)

    ### Process and plot the data

    ret_vals = tool_belt.process_counts(
        sig_counts, ref_counts, num_reps, readout_time, norm_style
    )
    (
        sig_counts_avg_kcps,
        ref_counts_avg_kcps,
        norm_avg_sig,
        norm_avg_sig_ste,
    ) = ret_vals

    # Raw data
    kpl.plot_line_update(ax_sig_ref, line_ind=0, y=sig_counts_avg_kcps)
    kpl.plot_line_update(ax_sig_ref, line_ind=1, y=ref_counts_avg_kcps)
    kpl.plot_line_update(ax_norm, y=norm_avg_sig)
    run_indicator_obj.remove()

    #  Plot the data itself and the fitted curve
    fit_func = tool_belt.inverted_cosexp
    fit_fig, ax, fit_func, popt, pcov = create_fit_figure(
        uwave_time_range,
        num_steps,
        uwave_freq,
        norm_avg_sig,
        norm_avg_sig_ste,
        fit_func,
    )
    rabi_period = 1 / popt[1]
    print("Rabi period measured: {} ns\n".format("%.1f" % rabi_period))

    # %% Clean up and save the data

    tool_belt.reset_cfm(cxn)

    endFunctionTime = time.time()

    timeElapsed = endFunctionTime - startFunctionTime

    timestamp = tool_belt.get_time_stamp()

    raw_data = {
        "timestamp": timestamp,
        "timeElapsed": timeElapsed,
        "timeElapsed-units": "s",
        "nv_sig": nv_sig,
        "uwave_freq": uwave_freq,
        "uwave_freq-units": "GHz",
        "uwave_power": uwave_power,
        "uwave_power-units": "dBm",
        "uwave_time_range": uwave_time_range,
        "uwave_time_range-units": "ns",
        "state": state.name,
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
        "sig_counts_each_shot": sig_counts_each_shot.astype(int).tolist(),
        "sig_counts_each_shot-units": "counts",
        "ref_counts_each_shot": ref_counts_each_shot.astype(int).tolist(),
        "ref_counts-units_each_shot": "counts",
    }

    nv_name = nv_sig["name"]
    file_path = tool_belt.get_file_path(__file__, timestamp, nv_name)
    tool_belt.save_figure(raw_fig, file_path)
    if fit_fig is not None:
        file_path_fit = tool_belt.get_file_path(__file__, timestamp, nv_name + "-fit")
        tool_belt.save_figure(fit_fig, file_path_fit)
    tool_belt.save_raw_data(raw_data, file_path)

    if (fit_func is not None) and (popt is not None):
        return rabi_period, sig_counts, ref_counts, popt
    else:
        return None, sig_counts, ref_counts


# %%
if __name__ == "__main__":
    import numpy as np

    # replotting data
    file = "2022_12_12-19_45_53-johnson-search"
    data = tool_belt.get_raw_data(file)

    threshold = 3

    num_steps = data["num_steps"]
    uwave_time_range = data["uwave_time_range"]

    taus = numpy.linspace(
        uwave_time_range[0], uwave_time_range[1], num=num_steps, dtype=numpy.int32
    )
    #
    # norm_avg_sig = data['norm_avg_sig']
    sig_counts_all = np.array(data["sig_counts_each_shot"])
    ref_counts_all = np.array(data["ref_counts_each_shot"])

    states_s = np.copy(sig_counts_all) * 0
    states_s[np.where(sig_counts_all >= threshold)] = 1
    states_r = np.copy(ref_counts_all) * 0
    states_r[np.where(ref_counts_all >= threshold)] = 1

    avg_states_s = np.average(states_s[0], 1)

    avg_states_r = np.average(states_r[0], 1)

    plt.figure()
    plt.plot(taus, avg_states_s, label="sig")
    plt.plot(taus, avg_states_r, label="ref")
    plt.ylabel(r"NV- probability")
    plt.xlabel("t [ns]")
    plt.legend()
    plt.show()

    # num_reps = data['num_reps']
    # uwave_time_range = data['uwave_time_range']
    # num_steps = data['num_steps']
    # nv_sig = data['nv_sig']
    # norm_style = tool_belt.NormStyle.SINGLE_VALUED
    # state = data['state']
    # uwave_freq = nv_sig['resonance_{}'.format(state)]
    # readout_time = nv_sig['charge_readout_dur']


#     kpl.init_kplotlib()
#     ret_vals = tool_belt.process_counts(sig_counts, ref_counts, num_reps, readout_time, norm_style)
#     (
#         sig_counts_avg_kcps,
#         ref_counts_avg_kcps,
#         norm_avg_sig,
#         norm_avg_sig_ste,
#     ) = ret_vals
# #
#     fit_func = tool_belt.inverted_cosexp
#     fit_fig, ax, fit_func, popt, pcov = create_fit_figure(
#         uwave_time_range, num_steps, uwave_freq, norm_avg_sig, norm_avg_sig_ste,
#         fit_func
#     )
