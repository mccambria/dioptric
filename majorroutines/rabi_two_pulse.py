# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 14:44:30 2022

File to run SRT Rabi measurements, based off this report 
https://journals.aps.org/prb/pdf/10.1103/PhysRevB.104.035201

@author: agardill
"""

import os
import time
from random import shuffle

import labrad
import matplotlib.pyplot as plt
import numpy
from scipy.optimize import curve_fit

import majorroutines.targeting as targeting
import utils.kplotlib as kpl
import utils.tool_belt as tool_belt
from utils.kplotlib import KplColors
from utils.tool_belt import States


def create_raw_data_figure(
    taus,
    avg_sig_counts=None,
    avg_ref_counts=None,
    norm_avg_sig=None,
    norm_avg_sig_ste=None,
    title=None,
):
    num_steps = len(taus)
    # Plot setup
    fig, axes_pack = plt.subplots(1, 2, figsize=kpl.double_figsize)
    ax_sig_ref, ax_norm = axes_pack
    ax_sig_ref.set_xlabel("Microwave duration (high = low) (ns)")
    ax_sig_ref.set_ylabel("Count rate (kcps)")
    ax_norm.set_xlabel("Microwave duration (high = low) (ns)")
    ax_norm.set_ylabel("Normalized fluorescence")
    if title is not None:
        ax_norm.set_title(title)

    # Plotting
    if avg_sig_counts is None:
        avg_sig_counts = numpy.empty(num_steps)
        avg_sig_counts[:] = numpy.nan
    kpl.plot_line(
        ax_sig_ref, taus, avg_sig_counts, label="Signal", color=KplColors.GREEN
    )
    if avg_ref_counts is None:
        avg_ref_counts = numpy.empty(num_steps)
        avg_ref_counts[:] = numpy.nan
    kpl.plot_line(
        ax_sig_ref, taus, avg_ref_counts, label="Reference", color=KplColors.RED
    )
    ax_sig_ref.legend(loc=kpl.Loc.LOWER_RIGHT)
    if norm_avg_sig is None:
        norm_avg_sig = numpy.empty(num_steps)
        norm_avg_sig[:] = numpy.nan
    if norm_avg_sig_ste is not None:
        kpl.plot_points(ax_norm, taus, norm_avg_sig, yerr=norm_avg_sig_ste)
    else:
        kpl.plot_line(ax_norm, taus, norm_avg_sig, color=KplColors.BLUE)

    return fig, ax_sig_ref, ax_norm


def create_err_figure(taus, norm_avg_sig=None, norm_avg_sig_ste=None, title=None):
    # Plot setup
    fig, ax = plt.subplots()
    ax.set_xlabel("Microwave duration (high = low) (ns)")
    ax.set_ylabel("Normalized fluorescence")
    if title is not None:
        ax.set_title(title)

    # Plotting
    if norm_avg_sig_ste is not None:
        kpl.plot_points(ax, taus, norm_avg_sig, yerr=norm_avg_sig_ste)

    return fig


# %% Main


def main(
    nv_sig,
    num_steps,
    num_reps,
    num_runs,
    uwave_time_range_LOW,
    uwave_time_range_HIGH,
    readout_state=States.HIGH,
    initial_state=States.HIGH,
    opti_nv_sig=None,
):
    # Right now, make sure SRS is set as State HIGH

    with labrad.connect() as cxn:
        main_with_cxn(
            cxn,
            nv_sig,
            num_steps,
            num_reps,
            num_runs,
            uwave_time_range_LOW,
            uwave_time_range_HIGH,
            readout_state,
            initial_state,
            opti_nv_sig,
        )


def main_with_cxn(
    cxn,
    nv_sig,
    num_steps,
    num_reps,
    num_runs,
    uwave_time_range_LOW,
    uwave_time_range_HIGH=[],
    readout_state=States.HIGH,
    initial_state=States.HIGH,
    opti_nv_sig=None,
):
    counter_server = tool_belt.get_server_counter(cxn)
    pulsegen_server = tool_belt.get_server_pulse_gen(cxn)
    tool_belt.reset_cfm(cxn)
    kpl.init_kplotlib()

    # %% Get the starting time of the function, to be used to calculate run time

    startFunctionTime = time.time()
    start_timestamp = tool_belt.get_time_stamp()

    # %% Initial calculations and setup

    state_high = States.HIGH
    state_low = States.LOW
    uwave_freq_high = nv_sig["resonance_{}".format(state_high.name)]
    uwave_freq_low = nv_sig["resonance_{}".format(state_low.name)]

    uwave_power_high = nv_sig["uwave_power_{}".format(state_high.name)]
    uwave_power_low = nv_sig["uwave_power_{}".format(state_low.name)]
    rabi_high = nv_sig["rabi_{}".format(state_high.name)]
    rabi_low = nv_sig["rabi_{}".format(state_low.name)]

    pi_pulse_high = tool_belt.get_pi_pulse_dur(rabi_high)
    pi_pulse_low = tool_belt.get_pi_pulse_dur(rabi_low)

    laser_key = "spin_laser"
    laser_name = nv_sig[laser_key]
    tool_belt.set_filter(cxn, nv_sig, laser_key)
    laser_power = tool_belt.set_laser_power(cxn, nv_sig, laser_key)

    polarization_time = nv_sig["spin_pol_dur"]
    readout = nv_sig["spin_readout_dur"]
    readout_sec = readout / (10**9)

    norm_style = nv_sig["norm_style"]

    # Array of times to sweep through
    # Must be ints since the pulse streamer only works with int64s
    min_uwave_time_LOW = uwave_time_range_LOW[0]
    max_uwave_time_LOW = uwave_time_range_LOW[1]
    t_LOW_list = numpy.linspace(min_uwave_time_LOW, max_uwave_time_LOW, num_steps)

    min_uwave_time_HIGH = uwave_time_range_HIGH[0]
    max_uwave_time_HIGH = uwave_time_range_HIGH[1]
    t_HIGH_list = numpy.linspace(min_uwave_time_HIGH, max_uwave_time_HIGH, num_steps)

    # Analyze the sequence
    num_reps = int(num_reps)
    file_name = "rabi_consec.py"
    seq_args = [
        t_LOW_list[0],
        t_HIGH_list[0],
        polarization_time,
        readout,
        pi_pulse_low,
        pi_pulse_high,
        t_LOW_list[0],
        t_HIGH_list[0],
        initial_state.value,
        readout_state.value,
        laser_name,
        laser_power,
    ]

    #    for arg in seq_args:
    #        print(type(arg))
    # print(seq_args)
    # return
    seq_args_string = tool_belt.encode_seq_args(seq_args)
    ret_vals = pulsegen_server.stream_load(file_name, seq_args_string)
    seq_time = ret_vals[0]

    seq_time_s = seq_time / (10**9)  # to seconds
    expected_run_time_s = (
        (num_steps / 2)
        * num_reps
        * num_runs
        * seq_time_s  # * 6 #taking slower than expected
    )  # s
    expected_run_time_m = expected_run_time_s / 60  # to minutes

    print(" \nExpected run time: {:.1f} minutes. ".format(expected_run_time_m))

    # Set up our data structure,

    # %% Make some lists and variables to save at the end

    opti_coords_list = []

    # create figure
    img_array = numpy.empty([num_steps, num_steps])
    img_array[:] = numpy.nan

    ### Signal generators servers
    low_sig_gen_cxn = tool_belt.get_server_sig_gen(cxn, States.LOW)
    high_sig_gen_cxn = tool_belt.get_server_sig_gen(cxn, States.HIGH)

    # %% Collect the data

    # Start 'Press enter to stop...'
    tool_belt.init_safe_stop()

    # for run_ind in range(num_runs):
    #     print('Run index: {}'. format(run_ind))

    for t_L_i in range(len(t_LOW_list)):
        for t_H_i in range(len(t_HIGH_list)):
            t_LOW = t_LOW_list[t_L_i]
            t_HIGH = t_HIGH_list[t_H_i]
            print("t_LOW {} ns, t_HIGH {} ns".format(t_LOW, t_HIGH))

            # Optimize and save the coords we found
            if opti_nv_sig:
                opti_coords = targeting.main_with_cxn(cxn, opti_nv_sig)
                drift = tool_belt.get_drift()
                adj_coords = nv_sig["coords"] + numpy.array(drift)
                tool_belt.set_xyz(cxn, adj_coords)
            else:
                opti_coords = targeting.main_with_cxn(cxn, nv_sig)
            opti_coords_list.append(opti_coords)

            tool_belt.set_filter(cxn, nv_sig, "spin_laser")
            laser_power = tool_belt.set_laser_power(cxn, nv_sig, laser_key)

            # Set up the microwaves for the low and high states
            low_sig_gen_cxn.set_freq(uwave_freq_low)
            low_sig_gen_cxn.set_amp(uwave_power_low)
            low_sig_gen_cxn.uwave_on()

            high_sig_gen_cxn.set_freq(uwave_freq_high)
            high_sig_gen_cxn.set_amp(uwave_power_high)
            high_sig_gen_cxn.uwave_on()

            # Load the APD
            counter_server.start_tag_stream()

            # Stream the sequence

            seq_args = [
                t_LOW_list[t_L_i],
                t_HIGH_list[t_H_i],
                polarization_time,
                readout,
                pi_pulse_low,
                pi_pulse_high,
                t_LOW_list[t_L_i],
                t_HIGH_list[t_H_i],
                initial_state.value,
                readout_state.value,
                laser_name,
                laser_power,
            ]

            seq_args_string = tool_belt.encode_seq_args(seq_args)
            # print(seq_args)
            # Clear the tagger buffer of any excess counts
            counter_server.clear_buffer()
            pulsegen_server.stream_immediate(file_name, num_reps, seq_args_string)

            # Get the counts
            new_counts = counter_server.read_counter_separate_gates(1)

            sample_counts = new_counts[0]

            count = sum(sample_counts[0::2])
            sig_counts = count
            # print("First signal = " + str(count))

            count = sum(sample_counts[1::2])
            ref_counts = count

            norm_avg_sig = sig_counts / ref_counts

            img_array[t_L_i][t_H_i] = norm_avg_sig

    counter_server.stop_tag_stream()

    img_array = numpy.flipud(img_array)
    kpl.imshow
    fig, ax = plt.subplots()
    axes_labels = ["HIGH MW pulse dur (ns)", "LOW MW pulse dur (ns)"]
    half_pixel_size_x = (t_HIGH_list[1] - t_HIGH_list[0]) / 2
    half_pixel_size_y = (t_LOW_list[1] - t_LOW_list[0]) / 2
    img_extent = [
        min_uwave_time_HIGH - half_pixel_size_x,
        max_uwave_time_HIGH + half_pixel_size_x,
        min_uwave_time_LOW - half_pixel_size_y,
        max_uwave_time_LOW + half_pixel_size_y,
    ]
    kpl.imshow(
        ax,
        img_array,
        axes_labels=axes_labels,
        cbar_label="Norm. fluor.",
        extent=img_extent,
    )
    title = "{} initial state, {} readout state".format(
        initial_state.name, readout_state.name
    )
    ax.set_title(title)
    print(list(img_array))

    #     # %% incremental plotting

    #     # Update the run indicator

    #     inc_sig_counts = sig_counts[: run_ind + 1]
    #     inc_ref_counts = ref_counts[: run_ind + 1]
    #     ret_vals = tool_belt.process_counts(
    #         inc_sig_counts, inc_ref_counts, num_reps, readout, norm_style
    #     )
    #     (
    #         sig_counts_avg_kcps,
    #         ref_counts_avg_kcps,
    #         norm_avg_sig,
    #         norm_avg_sig_ste,
    #     ) = ret_vals

    #     if do_plot:
    #         text = run_indicator_text.format(run_ind + 1, num_runs)
    #         run_indicator_obj.txt.set_text(text)
    #         kpl.plot_line_update(ax_sig_ref, line_ind=0, y=sig_counts_avg_kcps)
    #         kpl.plot_line_update(ax_sig_ref, line_ind=1, y=ref_counts_avg_kcps)
    #         kpl.plot_line_update(ax_norm, y=norm_avg_sig)

    #     # %% Save the data we have incrementally for long measurements

    #     raw_data = {'start_timestamp': start_timestamp,
    #                 'nv_sig': nv_sig,
    #                 'nv_sig-units': tool_belt.get_nv_sig_units(cxn),
    #                 'uwave_time_range_LOW': uwave_time_range_LOW,
    #                 'uwave_time_range_HIGH': uwave_time_range_HIGH,
    #                 'uwave_time_range-units': 'ns',
    #                 'taus_LOW': taus_LOW.tolist(),
    #                 'taus_HIGH': taus_HIGH.tolist(),
    #                 'initial_state': initial_state.name,
    #                 'readout_state': readout_state.name,
    #                 'num_steps': num_steps,
    #                 'num_reps': num_reps,
    #                 'num_runs': num_runs,
    #                 'tau_index_master_list':tau_index_master_list,
    #                 'opti_coords_list': opti_coords_list,
    #                 'opti_coords_list-units': 'V',
    #                 'sig_counts': sig_counts.astype(int).tolist(),
    #                 'sig_counts-units': 'counts',
    #                 'ref_counts': ref_counts.astype(int).tolist(),
    #                 'ref_counts-units': 'counts'}

    #     # This will continuously be the same file path so we will overwrite
    #     # the existing file with the latest version
    #     file_path = tool_belt.get_file_path(__file__, start_timestamp,
    #                                         nv_sig['name'], 'incremental')
    #     tool_belt.save_raw_data(raw_data, file_path)
    #     if do_plot:
    #         tool_belt.save_figure(raw_fig, file_path)

    # # # %% Fit the data and extract piPulse

    # # fit_func, popt = fit_data(uwave_time_range, num_steps, norm_avg_sig)

    # # %% Plot the Rabi signal

    # ### Process and plot the data

    # ret_vals = tool_belt.process_counts(sig_counts, ref_counts, num_reps, readout, norm_style)
    # (
    #     sig_counts_avg_kcps,
    #     ref_counts_avg_kcps,
    #     norm_avg_sig,
    #     norm_avg_sig_ste,
    # ) = ret_vals

    # # Raw data
    # if do_plot:
    #     kpl.plot_line_update(ax_sig_ref, line_ind=0, y=sig_counts_avg_kcps)
    #     kpl.plot_line_update(ax_sig_ref, line_ind=1, y=ref_counts_avg_kcps)
    #     kpl.plot_line_update(ax_norm, y=norm_avg_sig)
    #     run_indicator_obj.remove()

    # # %% Plot the data itself and the fitted curve

    # # fit_fig = None
    # # if (fit_func is not None) and (popt is not None):
    # #     fit_fig = create_fit_figure(uwave_time_range, uwave_freq, num_steps,
    # #                                 norm_avg_sig, fit_func, popt)
    # #     rabi_period = 1/popt[1]
    # #     print('Rabi period measured: {} ns\n'.format('%.1f'%rabi_period))

    # # %% Clean up and save the data

    # tool_belt.reset_cfm(cxn)
    # # turn off FM
    # if hasattr(low_sig_gen_cxn, "fm_off"):
    #     low_sig_gen_cxn.fm_off()
    # if hasattr(high_sig_gen_cxn, "fm_off"):
    #     high_sig_gen_cxn.fm_off()

    # endFunctionTime = time.time()

    # timeElapsed = endFunctionTime - startFunctionTime

    # timestamp = tool_belt.get_time_stamp()

    # raw_data = {'timestamp': timestamp,
    #             'timeElapsed': timeElapsed,
    #             'timeElapsed-units': 's',
    #             'nv_sig': nv_sig,
    #             'nv_sig-units': tool_belt.get_nv_sig_units(cxn),
    #             'initial_state': initial_state.name,
    #             'readout_state': readout_state.name,
    #             'num_steps': num_steps,
    #             'num_reps': num_reps,
    #             'num_runs': num_runs,
    #             'uwave_time_range_LOW': uwave_time_range_LOW,
    #             'uwave_time_range_HIGH': uwave_time_range_HIGH,
    #             'uwave_time_range-units': 'ns',
    #             'taus_LOW': taus_LOW.tolist(),
    #             'taus_HIGH': taus_HIGH.tolist(),
    #             'opti_coords_list': opti_coords_list,
    #             'opti_coords_list-units': 'V',
    #             'sig_counts': sig_counts.astype(int).tolist(),
    #             'sig_counts-units': 'counts',
    #             'ref_counts': ref_counts.astype(int).tolist(),
    #             'ref_counts-units': 'counts',
    #             'norm_avg_sig': norm_avg_sig.astype(float).tolist(),
    #             'norm_avg_sig-units': 'arb',
    #             'norm_avg_sig_ste': norm_avg_sig_ste.astype(float).tolist(),}

    # nv_name = nv_sig["name"]
    # file_path = tool_belt.get_file_path(__file__, timestamp, nv_name)
    # if do_plot:
    #     tool_belt.save_figure(raw_fig, file_path)
    # # if fit_fig is not None:
    # #     file_path_fit = tool_belt.get_file_path(__file__, timestamp, nv_name + "-fit")
    # #     tool_belt.save_figure(fit_fig, file_path_fit)
    # tool_belt.save_raw_data(raw_data, file_path)

    # # if (fit_func is not None) and (popt is not None):
    # #     return rabi_period, sig_counts, ref_counts, popt
    # # else:
    # #     return None, sig_counts, ref_counts, []

    return


# %%
def plot_pop_consec(taus, m_pop, z_pop, p_pop, m_err=None, z_err=None, p_err=None):
    fig, ax = plt.subplots()
    ax.set_title("Rabi double quantum")
    ax.set_xlabel("SRT length (ns)")
    ax.set_ylabel("Population")
    ax.set_title("Rabi with consec. pulses")

    # Plotting
    if m_err is not None:
        kpl.plot_points(
            ax, taus, m_pop, yerr=m_err, color=KplColors.RED, label="-1 population"
        )
    else:
        kpl.plot_line(ax, taus, m_pop, color=KplColors.RED, label="-1 population")

    if z_err is not None:
        kpl.plot_points(
            ax, taus, z_pop, yerr=z_err, color=KplColors.GREEN, label="0 population"
        )
    else:
        kpl.plot_line(ax, taus, z_pop, color=KplColors.GREEN, label="0 population")

    if p_err is not None:
        kpl.plot_points(
            ax, taus, p_pop, yerr=p_err, color=KplColors.BLUE, label="+1 population"
        )
    else:
        kpl.plot_line(ax, taus, p_pop, color=KplColors.BLUE, label="+1 population")

    ax.legend()

    return fig


def full_pop_consec(nv_sig, uwave_time_range, num_steps, num_reps, num_runs):
    contrast = 0.11 * 2
    min_pop = 1 - contrast

    min_uwave_time = uwave_time_range[0]
    max_uwave_time = uwave_time_range[1]
    taus = numpy.linspace(min_uwave_time, max_uwave_time, num=num_steps)

    init = States.LOW
    p_sig, p_ste = main(
        nv_sig,
        num_steps,
        num_reps,
        num_runs,
        uwave_time_range,
        readout_state=States.HIGH,
        initial_state=init,
        do_err_plot=False,
    )
    p_pop = (numpy.array(p_sig) - min_pop) / (1 - min_pop)
    p_err = numpy.array(p_ste) / (1 - min_pop)

    m_sig, m_ste = main(
        nv_sig,
        uwave_time_range,
        num_steps,
        num_reps,
        num_runs,
        readout_state=States.LOW,
        initial_state=init,
        do_err_plot=False,
    )
    m_pop = (numpy.array(m_sig) - min_pop) / (1 - min_pop)
    m_err = numpy.array(m_ste) / (1 - min_pop)

    z_sig, z_ste = main(
        nv_sig,
        uwave_time_range,
        num_steps,
        num_reps,
        num_runs,
        readout_state=States.ZERO,
        initial_state=init,
        do_err_plot=False,
    )
    z_pop = (numpy.array(z_sig) - min_pop) / (1 - min_pop)
    z_err = numpy.array(z_ste) / (1 - min_pop)

    fig = plot_pop_consec(taus, m_pop, z_pop, p_pop, m_err, z_err, p_err)

    timestamp = tool_belt.get_time_stamp()
    nv_name = nv_sig["name"]
    file_path = tool_belt.get_file_path(__file__, timestamp, nv_name)
    tool_belt.save_figure(fig, file_path)


def fit_data(taus, norm_avg_sig):
    # %% Set up

    fit_func = lambda t, off, freq: tool_belt.cosexp_1_at_0(t, off, freq, 1e3)

    # %% Estimated fit parameters

    offset = 0.2  # numpy.average(norm_avg_sig)
    decay = 10
    frequency = 0.8

    # %% Fit

    init_params = [offset, frequency, decay]
    init_params = [offset, frequency]

    try:
        popt, _ = curve_fit(
            fit_func, taus, norm_avg_sig, p0=init_params, bounds=(0, numpy.infty)
        )
    except Exception as e:
        print(e)
        popt = None

    return fit_func, popt


# %% Run the file


if __name__ == "__main__":
    img_array = numpy.array(
        [
            [
                0.84890941,
                0.83852017,
                0.8628879,
                0.86936508,
                0.85804317,
                0.85777427,
                0.86298536,
                0.87368421,
                0.88972048,
                0.90468963,
                0.91406367,
                0.90202601,
                0.92364305,
                0.89440994,
                0.90979922,
                0.89906714,
                0.94278795,
                0.9108791,
                0.92799635,
                0.89135583,
                0.92084393,
                0.8976378,
                0.9107734,
                0.94086849,
                0.91989625,
                0.91448785,
                0.91282844,
                0.91257576,
                0.89753725,
                0.89208308,
                0.8841344,
            ],
            [
                0.84109421,
                0.84531297,
                0.87915454,
                0.8394523,
                0.89264931,
                0.87261621,
                0.88787325,
                0.87940825,
                0.91945197,
                0.91085097,
                0.92427958,
                0.91978193,
                0.90638298,
                0.93014872,
                0.92333034,
                0.92940473,
                0.9312774,
                0.9056432,
                0.89131728,
                0.91705069,
                0.94547526,
                0.90386443,
                0.90007981,
                0.91920755,
                0.89378514,
                0.93131064,
                0.92588092,
                0.91165153,
                0.90418619,
                0.89121928,
                0.87353778,
            ],
            [
                0.82477204,
                0.85213226,
                0.86048232,
                0.87646352,
                0.87346142,
                0.90638363,
                0.862704,
                0.92230657,
                0.89277089,
                0.89426713,
                0.92281033,
                0.89773934,
                0.90745773,
                0.90335878,
                0.90400828,
                0.91126845,
                0.93001346,
                0.94537564,
                0.92356021,
                0.91513247,
                0.9303401,
                0.93997006,
                0.93486074,
                0.91718427,
                0.91615967,
                0.9120844,
                0.91585127,
                0.90558969,
                0.90538089,
                0.90447185,
                0.88601265,
            ],
            [
                0.86528262,
                0.85567471,
                0.87694128,
                0.88144863,
                0.87650289,
                0.90686881,
                0.88086643,
                0.90020883,
                0.90107692,
                0.90412844,
                0.89601703,
                0.89970997,
                0.9361802,
                0.92865124,
                0.92541392,
                0.92736764,
                0.95748895,
                0.94764001,
                0.92565056,
                0.94088816,
                0.97252238,
                0.94701987,
                0.90876619,
                0.9377954,
                0.95348487,
                0.93177068,
                0.92885846,
                0.91391914,
                0.91878552,
                0.90034211,
                0.9112782,
            ],
            [
                0.85909024,
                0.86558346,
                0.86110692,
                0.90461783,
                0.90349236,
                0.88106343,
                0.89946278,
                0.9100725,
                0.9014318,
                0.92305362,
                0.9381953,
                0.92244348,
                0.92386617,
                0.95131938,
                0.93588948,
                0.92490842,
                0.96149068,
                0.96326027,
                0.96741697,
                0.93990163,
                0.96764009,
                0.95685909,
                0.94921081,
                0.9539575,
                0.92075945,
                0.91938224,
                0.9453125,
                0.94770035,
                0.91161928,
                0.90031574,
                0.88394556,
            ],
            [
                0.84943482,
                0.85896858,
                0.89593174,
                0.86068585,
                0.90715727,
                0.90945002,
                0.92052177,
                0.90367475,
                0.93238547,
                0.96353348,
                0.92074484,
                0.9377954,
                0.92993441,
                0.95721438,
                0.93950491,
                0.91093887,
                0.92639032,
                0.95001543,
                0.95984059,
                0.95521236,
                0.93565135,
                0.96065626,
                0.94592885,
                0.93747062,
                0.9334902,
                0.93643877,
                0.938687,
                0.93604651,
                0.90076448,
                0.92004773,
                0.90242091,
            ],
            [
                0.88856305,
                0.88810869,
                0.87488816,
                0.90708372,
                0.8808352,
                0.90851221,
                0.91339048,
                0.92539092,
                0.93204314,
                0.91323595,
                0.93220339,
                0.949474,
                0.92157434,
                0.9055732,
                0.92372756,
                0.98196798,
                0.9854984,
                0.95646238,
                0.95473122,
                0.93863739,
                0.94074627,
                0.96632946,
                0.98286321,
                0.94859186,
                0.942351,
                0.94510562,
                0.95422327,
                0.92579976,
                0.93281344,
                0.90281856,
                0.9076877,
            ],
            [
                0.86302033,
                0.92356897,
                0.86234128,
                0.85023711,
                0.89440146,
                0.89393038,
                0.92167939,
                0.93281853,
                0.93116922,
                0.91622024,
                0.92463235,
                0.93722042,
                0.94131992,
                0.95360346,
                0.98019199,
                0.94421645,
                0.93953346,
                0.96890958,
                0.96877842,
                0.94899012,
                0.99107143,
                0.97179525,
                0.97761194,
                0.94791512,
                0.95807938,
                0.94251691,
                0.94246783,
                0.93577031,
                0.9336858,
                0.90998464,
                0.90559025,
            ],
            [
                0.87092809,
                0.86006219,
                0.88104034,
                0.86343741,
                0.89202436,
                0.95104247,
                0.90900887,
                0.90560775,
                0.92062793,
                0.98354662,
                0.93871581,
                0.96233608,
                0.93198091,
                0.95065205,
                0.95838886,
                0.98699112,
                0.99294612,
                0.96315555,
                0.97868188,
                0.94907476,
                0.95505111,
                0.94180451,
                0.980881,
                0.95208302,
                0.9886658,
                0.91878827,
                0.94420812,
                0.95336871,
                0.95169735,
                0.90079618,
                0.93930542,
            ],
            [
                0.88188616,
                0.90891089,
                0.87797888,
                0.908645,
                0.92408337,
                0.89907562,
                0.91137506,
                0.90358613,
                0.94827586,
                0.93848776,
                0.94263952,
                0.95378812,
                0.94018663,
                0.97946065,
                0.98317162,
                0.95289747,
                0.9552127,
                0.98464112,
                0.97920461,
                0.96796615,
                0.96288376,
                0.97967914,
                0.96151533,
                0.98479496,
                0.95582207,
                0.93254613,
                0.95892503,
                0.94198426,
                0.96601942,
                0.94270753,
                0.89043765,
            ],
            [
                0.87026945,
                0.92197922,
                0.91518405,
                0.91461935,
                0.89638084,
                0.90759076,
                0.94092002,
                0.92788971,
                0.93806239,
                0.91903306,
                0.95882531,
                0.95076698,
                0.962183,
                0.97275285,
                0.95177515,
                0.98865727,
                0.971461,
                0.96273901,
                0.95284139,
                0.9965304,
                0.99669521,
                0.98051454,
                0.95851111,
                0.94969199,
                0.94271902,
                0.97243446,
                0.9377557,
                0.95194923,
                0.96815382,
                0.92896175,
                0.93966702,
            ],
            [
                0.89059994,
                0.89882957,
                0.89875867,
                0.91207131,
                0.9027611,
                0.93234987,
                0.92460671,
                0.9200534,
                0.94806178,
                0.95076272,
                0.95885268,
                0.953872,
                0.99489085,
                0.96239239,
                0.97649735,
                0.96267886,
                0.99969164,
                0.96962829,
                0.94691831,
                0.97530679,
                0.95794183,
                0.96569212,
                0.98004501,
                0.94988242,
                0.97037583,
                0.9683753,
                0.94921295,
                0.94398217,
                0.93956451,
                0.94265125,
                0.94377696,
            ],
            [
                0.88485588,
                0.86492605,
                0.90343114,
                0.88845694,
                0.87952172,
                0.95002321,
                0.94988864,
                0.91737546,
                0.93233431,
                0.95945309,
                0.95940671,
                0.9484057,
                0.95345892,
                0.99330357,
                0.95987362,
                0.97879177,
                0.97350993,
                0.98815947,
                0.94584253,
                0.9946865,
                1.00688314,
                0.98823709,
                0.9944709,
                0.97759764,
                0.95764741,
                0.95055797,
                0.90513955,
                0.95090881,
                0.94391202,
                0.9161442,
                0.92562652,
            ],
            [
                0.86544801,
                0.88107549,
                0.8836836,
                0.91213578,
                0.89423942,
                0.93414449,
                0.91795642,
                0.93032278,
                0.96582419,
                0.97069653,
                0.93780761,
                0.94990162,
                0.98064319,
                0.96441012,
                0.98978191,
                0.98047755,
                0.9661693,
                0.94199738,
                0.99548261,
                0.98006546,
                0.98554396,
                0.97520784,
                0.94704325,
                0.9767728,
                0.96935283,
                0.96117262,
                0.93735317,
                0.96123666,
                0.91655141,
                0.92568408,
                0.93274945,
            ],
            [
                0.87243492,
                0.86504808,
                0.88592385,
                0.89855305,
                0.87920635,
                0.89453978,
                0.94692344,
                0.98050459,
                0.92714417,
                0.93834772,
                0.9575917,
                0.9904777,
                0.95320048,
                0.97184378,
                0.98847995,
                0.97900961,
                0.93644315,
                0.94459299,
                0.99558777,
                0.97000609,
                1.01290631,
                1.01530612,
                0.98895899,
                0.9556314,
                0.94039325,
                0.97624327,
                0.95057817,
                0.93868889,
                0.93450256,
                0.89460154,
                0.93498547,
            ],
            [
                0.89160063,
                0.88085508,
                0.87339158,
                0.88955087,
                0.89585871,
                0.92306518,
                0.92878064,
                0.9789675,
                0.92265453,
                0.9741668,
                0.94737654,
                0.95643411,
                0.95793136,
                0.99398734,
                0.99869728,
                0.97818711,
                0.99755541,
                0.97733496,
                0.97162246,
                0.97861289,
                0.98783721,
                0.97800797,
                0.96753247,
                0.98470012,
                0.97781226,
                0.95583119,
                0.96091156,
                0.94530896,
                0.94184462,
                0.92789035,
                0.93375394,
            ],
            [
                0.88634266,
                0.88712337,
                0.89041509,
                0.90485584,
                0.94993857,
                0.92572707,
                0.90724978,
                0.92836414,
                0.96548564,
                0.93780255,
                0.95387345,
                0.94891386,
                0.9467879,
                1.00199356,
                0.97845355,
                0.98441436,
                0.96539526,
                0.95227133,
                0.95610278,
                0.99783984,
                0.97678681,
                0.98369732,
                0.96823956,
                0.95150701,
                0.98718938,
                0.95663957,
                0.96469523,
                0.97492869,
                0.92492625,
                0.90590842,
                0.92359413,
            ],
            [
                0.90735294,
                0.90058852,
                0.88821043,
                0.89650911,
                0.89725191,
                0.91485784,
                0.8934146,
                0.95928953,
                0.92271808,
                0.93770963,
                0.91994897,
                0.95196015,
                0.95407503,
                0.98541072,
                0.96352979,
                0.94927647,
                0.98977378,
                0.98497163,
                0.95981407,
                0.99017979,
                0.95942756,
                0.96743209,
                0.97587687,
                0.95408088,
                0.95013239,
                0.9638608,
                0.93583517,
                0.94199186,
                0.94936901,
                0.92636008,
                0.95273174,
            ],
            [
                0.88998917,
                0.87032902,
                0.87417219,
                0.89663498,
                0.88430384,
                0.91218693,
                0.90995406,
                0.94241078,
                0.96747469,
                0.96770802,
                0.96492271,
                0.96529544,
                0.95137139,
                0.95945323,
                0.93583797,
                0.98578569,
                0.97599623,
                0.98250192,
                0.9328879,
                0.96035177,
                1.00257771,
                0.95317087,
                0.93087288,
                0.97921225,
                0.9436096,
                0.97547346,
                0.92107303,
                0.93802035,
                0.94658665,
                0.91023906,
                0.94014198,
            ],
            [
                0.85384154,
                0.88941718,
                0.87375415,
                0.91403426,
                0.92250809,
                0.90538745,
                0.92183806,
                0.90789474,
                0.93553719,
                0.94110522,
                0.91232548,
                0.95222833,
                0.96512176,
                0.96331138,
                0.94949184,
                0.93161621,
                0.93950285,
                0.95904126,
                0.97491639,
                0.94322344,
                0.96268882,
                0.94320763,
                0.93332302,
                0.95751584,
                0.95738157,
                0.95938462,
                0.92662377,
                0.93898412,
                0.93675379,
                0.91374705,
                0.91876972,
            ],
            [
                0.89200798,
                0.89979123,
                0.91640212,
                0.89840458,
                0.88038984,
                0.90258359,
                0.9000149,
                0.93266497,
                0.94954983,
                0.92774789,
                0.93651744,
                0.94125521,
                0.94361352,
                0.95210582,
                0.98120583,
                0.93424576,
                0.97953391,
                0.97529554,
                0.94564573,
                0.96909691,
                0.97813544,
                0.97317775,
                0.93285264,
                0.95971028,
                0.94657097,
                0.94800777,
                0.93574824,
                0.94755541,
                0.90465231,
                0.96048962,
                0.89841552,
            ],
            [
                0.88558067,
                0.88655594,
                0.89093175,
                0.90982839,
                0.89579249,
                0.91818045,
                0.92906731,
                0.92075876,
                0.92100512,
                0.93808325,
                0.91440905,
                0.93646242,
                0.9369247,
                0.94380996,
                0.940219,
                0.93388308,
                0.95846268,
                0.96441662,
                0.98024964,
                0.99076104,
                0.95145631,
                0.9235701,
                0.96553818,
                0.91014706,
                0.98661687,
                0.90277368,
                0.93455971,
                0.94585473,
                0.91790142,
                0.89320822,
                0.89420655,
            ],
            [
                0.85144656,
                0.8623488,
                0.89077559,
                0.91556647,
                0.89130763,
                0.9085697,
                0.90930859,
                0.91383423,
                0.93874539,
                0.91873156,
                0.93026022,
                0.92,
                0.92847604,
                0.89901803,
                0.94046173,
                0.95250538,
                0.91717654,
                0.96164596,
                0.94526761,
                0.95147236,
                0.95292159,
                0.93614784,
                0.97333737,
                0.95079462,
                0.90166839,
                0.94744101,
                0.94132142,
                0.92119166,
                0.91169062,
                0.89604333,
                0.91362794,
            ],
            [
                0.86072988,
                0.88218924,
                0.86153378,
                0.89724771,
                0.89444867,
                0.87667009,
                0.89329455,
                0.91707012,
                0.91823411,
                0.92989942,
                0.93678789,
                0.93620845,
                0.95530474,
                0.93195444,
                0.96165284,
                0.93779684,
                0.9201942,
                0.9290439,
                0.95210494,
                0.95134809,
                0.94583208,
                0.92627144,
                0.90489408,
                0.9277811,
                0.92194674,
                0.92774789,
                0.88697178,
                0.8928047,
                0.92610762,
                0.89846013,
                0.90640547,
            ],
            [
                0.85658975,
                0.84513343,
                0.9003736,
                0.86358819,
                0.87602802,
                0.89932454,
                0.87299893,
                0.91759734,
                0.92032297,
                0.91670425,
                0.9096564,
                0.90473346,
                0.90584752,
                0.91374747,
                0.90406977,
                0.93353338,
                0.94312068,
                0.91456223,
                0.91444896,
                0.94309927,
                0.95098488,
                0.93338346,
                0.92510532,
                0.92481913,
                0.90388335,
                0.92121865,
                0.8846382,
                0.91007356,
                0.88825473,
                0.89291147,
                0.87490239,
            ],
            [
                0.83119185,
                0.87524813,
                0.84137416,
                0.88438322,
                0.88704773,
                0.87451795,
                0.89974714,
                0.88561804,
                0.90410754,
                0.92017394,
                0.91946706,
                0.91104713,
                0.92396004,
                0.92086222,
                0.94911671,
                0.91249255,
                0.91337768,
                0.9054588,
                0.8938739,
                0.90248201,
                0.92753844,
                0.89966261,
                0.91819401,
                0.92616639,
                0.89285714,
                0.92050209,
                0.89413014,
                0.90612695,
                0.89045764,
                0.94232649,
                0.88021339,
            ],
            [
                0.84547259,
                0.84148034,
                0.85597996,
                0.85349314,
                0.87818561,
                0.89072633,
                0.8832793,
                0.87738678,
                0.9004075,
                0.90451056,
                0.89672544,
                0.90120222,
                0.93209491,
                0.91280287,
                0.90105851,
                0.90631455,
                0.92120131,
                0.9148137,
                0.89834304,
                0.92046693,
                0.91281333,
                0.91406125,
                0.95676312,
                0.90693605,
                0.87987758,
                0.89699906,
                0.90206499,
                0.915827,
                0.87827801,
                0.85588325,
                0.87049057,
            ],
            [
                0.89512307,
                0.87273,
                0.8641532,
                0.87045319,
                0.87300151,
                0.87130834,
                0.87244975,
                0.93020753,
                0.86835785,
                0.8962963,
                0.90013475,
                0.91653739,
                0.8923286,
                0.91141397,
                0.90394674,
                0.91855271,
                0.90549055,
                0.93103448,
                0.92110149,
                0.92533455,
                0.907399,
                0.88996861,
                0.88726359,
                0.90749484,
                0.91685495,
                0.89157169,
                0.8991838,
                0.89862309,
                0.88422619,
                0.87299667,
                0.85047903,
            ],
            [
                0.81370717,
                0.84487099,
                0.87266392,
                0.87390075,
                0.88843236,
                0.86783734,
                0.87648492,
                0.88561993,
                0.87507357,
                0.88150332,
                0.91930091,
                0.91512017,
                0.86812689,
                0.88349216,
                0.89086412,
                0.90180266,
                0.90200517,
                0.89313923,
                0.88769547,
                0.88629185,
                0.90055701,
                0.90236298,
                0.89979388,
                0.87977099,
                0.85814201,
                0.91374747,
                0.87265804,
                0.89889467,
                0.88855771,
                0.87293647,
                0.87220288,
            ],
            [
                0.84145206,
                0.84535928,
                0.8288368,
                0.84728844,
                0.84474546,
                0.86369101,
                0.86945962,
                0.85758276,
                0.88700565,
                0.88125478,
                0.90542959,
                0.89375555,
                0.85639913,
                0.89581175,
                0.8715637,
                0.91605013,
                0.87209125,
                0.91695924,
                0.89088711,
                0.87791225,
                0.89124187,
                0.87351655,
                0.88285802,
                0.9005059,
                0.86329339,
                0.87417812,
                0.8770078,
                0.85532425,
                0.85673893,
                0.86663503,
                0.85959564,
            ],
            [
                0.82835711,
                0.85978081,
                0.86428789,
                0.84445769,
                0.86683875,
                0.86331704,
                0.83871933,
                0.86859356,
                0.8940824,
                0.89625537,
                0.87762031,
                0.87139225,
                0.89831281,
                0.88706208,
                0.89191644,
                0.86018463,
                0.89591248,
                0.89349022,
                0.89823406,
                0.89661938,
                0.87256691,
                0.87265799,
                0.87211018,
                0.87884407,
                0.85943411,
                0.89076663,
                0.86930634,
                0.87603432,
                0.85303752,
                0.85218193,
                0.84379906,
            ],
        ]
    )
