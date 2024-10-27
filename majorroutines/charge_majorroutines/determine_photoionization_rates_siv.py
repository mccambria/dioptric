# -*- coding: utf-8 -*-
"""
Created on mon Apr 8 10:45:09 2020

This file ru ns a sequence that pulses a green pulse either on of off the 
readout spot. It reads out in the SiVs band, to create SiV2- when on the spot
and SiV when off . Then a green pulse of variable power is pulsed on the
readout spot, followed by a yellow readout.

The point of this measurement is to determine how fast the SiV charge states
change under illumination. 

USE WITH 515 AM MOD

@author: agardill
"""
import copy
import time

import labrad
import matplotlib.pyplot as plt
import numpy
from scipy.optimize import curve_fit

import majorroutines.image_sample as image_sample
import majorroutines.targeting as targeting
import utils.tool_belt as tool_belt

siv_pulse_time = 10 * 10**6  # ns
siv_pulse_distance = 0.071  # V

bright_reset_power = 0.63  # 1.2 mW
bright_reset_range = 0.25
bright_reset_steps = 17
bright_reset_time = 10**7

dark_reset_power = 0.603  # 18 uW
dark_reset_range = 0.25
dark_reset_steps = 17
dark_reset_time = 10**7

# %%


def decay_exp(t, a1, d1, a2, d2):
    return a1 * numpy.exp(-t / d1) + a2 * numpy.exp(-t / d2)


def do_plot(on_counts, off_counts, test_pulse_dur_list, test_power_mW):
    test_pulse_dur_list = numpy.array(test_pulse_dur_list) / 10**6
    max_diff = off_counts[0] - on_counts[0]
    norm_counts = (off_counts - on_counts) / max_diff
    init_guess = [1, 0.1, 0.5, 1]
    popt, _ = curve_fit(decay_exp, test_pulse_dur_list, norm_counts, p0=init_guess)
    lin_time = numpy.linspace(test_pulse_dur_list[0], test_pulse_dur_list[-1], 100)
    print(popt)

    fig, axes = plt.subplots(1, 2, figsize=(15, 8))
    ax = axes[0]
    ax.plot(test_pulse_dur_list, norm_counts, "ro", label="Data")
    ax.plot(lin_time, decay_exp(lin_time, *popt), "g-", label="Exp Fit")
    ax.set_xlabel("Test Pulse Illumination Time (ms)")
    ax.set_ylabel("Norm Counts")
    ax.set_title(str(test_power_mW) + "mW test beam")
    ax.set_yscale("log")
    ax.legend()
    text_popt = "\n".join(
        (
            r"$A_0 e^{-t/d_0} + A_1 e^{-t/d_1}$",
            r"$A_0 = $" + "%.3f" % (popt[0]),
            r"$d_0 = $" + "%.3f" % (popt[1]) + " " + r"$ ms$",
            r"$A_1 = $" + "%.3f" % (popt[2]),
            r"$d_1 = $" + "%.3f" % (popt[3]) + " " + r"$ ms$",
        )
    )

    props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    ax.text(
        0.55,
        0.75,
        text_popt,
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=props,
    )

    ax = axes[1]
    ax.plot(test_pulse_dur_list, on_counts, "ko", label="SiV2- init")
    ax.plot(test_pulse_dur_list, off_counts, "bo", label="SiV- init")
    ax.set_xlabel("Test Pulse Illumination Time (ms)")
    ax.set_ylabel("Counts")
    ax.set_title(str(test_power_mW) + "mW test beam")
    ax.set_yscale("log")
    ax.legend()

    return fig, popt


def build_voltage_list(start_coords_drift, signal_coords_drift, num_reps):
    # calculate the x values we want to step thru
    start_x_value = start_coords_drift[0]
    start_y_value = start_coords_drift[1]

    # we want this list to have the pattern [[readout], [readout], [readout], [target],
    #                                                   [readout], [readout], [readout],...]
    # The glavo needs a 0th coord, so we'll pass the readout NV as the "starting" point
    x_points = [start_x_value]
    y_points = [start_y_value]

    # now append the coordinates in the following pattern:
    for i in range(num_reps):
        x_points.append(start_x_value)
        x_points.append(start_x_value)
        x_points.append(signal_coords_drift[0])
        x_points.append(start_x_value)
        x_points.append(start_x_value)
        x_points.append(start_x_value)

        y_points.append(start_y_value)
        y_points.append(start_y_value)
        y_points.append(signal_coords_drift[1])
        y_points.append(start_y_value)
        y_points.append(start_y_value)
        y_points.append(start_y_value)

    return x_points, y_points


# %% Main
# Connect to labrad in this file, as opposed to control panel
def main_AM(nv_sig, apd_indices, num_reps, test_color, test_time, test_power):
    with labrad.connect() as cxn:
        on_counts, off_counts = main_AM_with_cxn(
            cxn, nv_sig, apd_indices, num_reps, test_color, test_time, test_power
        )

    return on_counts, off_counts


def main_AM_with_cxn(
    cxn, nv_sig, apd_indices, num_reps, test_color, test_time, test_power
):
    tool_belt.reset_cfm_wout_uwaves(cxn)

    # Initial Calculation and setup
    readout_time = nv_sig["pulsed_SCC_readout_dur"]
    am_589_power = nv_sig["am_589_power"]
    am_515_power = nv_sig["ao_515_pwr"]

    prep_power_515 = am_515_power
    readout_power_589 = am_589_power
    nd_filter = nv_sig["nd_filter"]

    # set the nd_filter for yellow
    cxn.filter_slider_ell9k.set_filter(nd_filter)

    shared_params = tool_belt.get_shared_parameters_dict(cxn)

    # delay of aoms and laser
    laser_515_delay = shared_params["515_AM_laser_delay"]  ###
    aom_589_delay = shared_params["589_aom_delay"]
    laser_638_delay = shared_params["638_DM_laser_delay"]
    galvo_delay = shared_params["large_angle_galvo_delay"]

    # if using AM for green, add an additional 300 ns to the pulse time.
    # the AM laser has a 300 ns rise time
    #    if test_color == '515a':
    #        test_time = test_time + 300

    # Optimize
    #    opti_coords_list = []
    # Optimize
    #    opti_coords = optimize.main_with_cxn(cxn, nv_sig, apd_indices, '515a', disable=False)
    #    opti_coords_list.append(opti_coords)
    cxn.filter_slider_ell9k_color.set_filter("715 lp")

    # Estimate the lenth of the sequance , load the sequence
    file_name = "isolate_nv_charge_dynamics_moving_target.py"
    seq_args = [
        siv_pulse_time,
        test_time,
        readout_time,
        laser_515_delay,
        aom_589_delay,
        laser_638_delay,
        galvo_delay,
        readout_power_589,
        prep_power_515,
        test_power,
        prep_power_515,
        apd_indices[0],
        "515a",
        test_color,
        589,
    ]
    seq_args_string = tool_belt.encode_seq_args(seq_args)
    ret_vals = cxn.pulse_streamer.stream_load(file_name, seq_args_string)
    seq_dur = ret_vals[0]
    period = seq_dur

    # Set up the voltages to step thru
    # get the drift and add it to the start coordinates
    drift = numpy.array(tool_belt.get_drift())
    start_coords = numpy.array(nv_sig["coords"])
    start_coords_drift = start_coords + drift
    # define the signal coords as start + dx.
    signal_coords_drift = start_coords_drift + [siv_pulse_distance, 0, 0]

    x_voltages, y_voltages = build_voltage_list(
        start_coords_drift, signal_coords_drift, num_reps
    )

    # Collect data
    # start on the readout NV
    tool_belt.set_xyz(cxn, start_coords_drift)

    # Load the galvo
    cxn.galvo.load_arb_points_scan(x_voltages, y_voltages, int(period))

    # Set up the APD
    cxn.apd_tagger.start_tag_stream(apd_indices)

    # Run the sequence double the amount of time, one for the sig and one for the ref
    cxn.pulse_streamer.stream_start(num_reps * 2)

    # We'll be lookign for three samples each repetition, and double that for
    # the ref and sig
    total_num_reps = 3 * 2 * num_reps

    # Read the counts
    new_samples = cxn.apd_tagger.read_counter_simple(total_num_reps)
    # The last of the triplet of readout windows is the counts we are interested in
    on_counts = new_samples[2::6]
    on_counts = [int(el) for el in on_counts]
    off_counts = new_samples[5::6]
    off_counts = [int(el) for el in off_counts]

    cxn.apd_tagger.stop_tag_stream()

    return on_counts, off_counts


# %%
# Connect to labrad in this file, as opposed to control panel
def main_scan(nv_sig, apd_indices, num_reps, test_color, test_time, test_power):
    with labrad.connect() as cxn:
        green_counts, red_counts = main_scan_with_cxn(
            cxn, nv_sig, apd_indices, num_reps, test_color, test_time, test_power
        )

    return green_counts, red_counts


def main_scan_with_cxn(
    cxn, nv_sig, apd_indices, num_reps, test_color, test_time, test_power
):
    tool_belt.reset_cfm_wout_uwaves(cxn)

    # Initial Calculation and setup
    readout_time = nv_sig["pulsed_SCC_readout_dur"]
    am_589_power = nv_sig["am_589_power"]
    am_515_power = nv_sig["ao_515_pwr"]
    #    nd_filter = nv_sig['nd_filter']
    center_coords = nv_sig["coords"]

    apd_index = 0

    shared_params = tool_belt.get_shared_parameters_dict(cxn)

    # delay of aoms and laser
    laser_515_delay = shared_params["515_AM_laser_delay"]

    aom_589_delay = shared_params["589_aom_delay"]
    #    laser_638_delay = shared_params['638_DM_laser_delay']

    on_counts = []  # SiV2-
    off_counts = []  # siv

    cxn.filter_slider_ell9k.set_filter("nd_0")
    targeting.main_with_cxn(cxn, nv_sig, apd_indices, "515a", disable=False)
    run_start_time = time.time()

    for i in range(num_reps):
        # optimize every 2 min or so
        # So first check the time. If the time that has passed since the last
        # optimize is longer that 2 min, optimize again
        current_time = time.time()
        if current_time - run_start_time >= 2 * 60:
            targeting.main_with_cxn(cxn, nv_sig, apd_indices, "515a")
            run_start_time = current_time

        drift = numpy.array(tool_belt.get_drift())
        center_coords_drift = center_coords + drift

        cxn.filter_slider_ell9k_color.set_filter("715 lp")

        # DARK SIV SCAN
        reset_sig = copy.deepcopy(nv_sig)
        reset_sig["coords"] = center_coords_drift
        reset_sig["ao_515_pwr"] = dark_reset_power
        _, _, _ = image_sample.main(
            reset_sig,
            dark_reset_range,
            dark_reset_range,
            dark_reset_steps,
            apd_indices,
            "515a",
            readout=dark_reset_time,
            save_data=False,
            plot_data=False,
        )
        _, _, _ = image_sample.main(
            reset_sig,
            dark_reset_range,
            dark_reset_range,
            dark_reset_steps,
            apd_indices,
            "515a",
            readout=dark_reset_time,
            save_data=False,
            plot_data=False,
        )

        # test pulse at center
        pulse_file_name = "simple_pulse.py"
        seq_args = [laser_515_delay, test_time, am_589_power, test_power, test_color]
        seq_args_string = tool_belt.encode_seq_args(seq_args)
        cxn.pulse_streamer.stream_immediate(pulse_file_name, 1, seq_args_string)

        # readout
        readout_file_name = "simple_readout.py"
        seq_args = [
            aom_589_delay,
            readout_time,
            am_589_power,
            am_515_power,
            apd_index,
            589,
        ]
        seq_args_string = tool_belt.encode_seq_args(seq_args)
        cxn.pulse_streamer.stream_load(readout_file_name, seq_args_string)

        # Load the APD
        cxn.apd_tagger.start_tag_stream(apd_indices)
        # Clear the buffer
        cxn.apd_tagger.clear_buffer()
        # Run the sequence
        cxn.pulse_streamer.stream_immediate(readout_file_name, 1, seq_args_string)

        new_counts = cxn.apd_tagger.read_counter_separate_gates(1)
        sample_counts = new_counts[0]
        on_counts.append(sample_counts[0])

        # RIGHT SIV SCAN
        # Move the ND filter to 0
        reset_sig = copy.deepcopy(nv_sig)
        reset_sig["coords"] = center_coords_drift
        reset_sig["ao_515_pwr"] = bright_reset_power
        _, _, _ = image_sample.main(
            reset_sig,
            bright_reset_range,
            bright_reset_range,
            bright_reset_steps,
            apd_indices,
            "515a",
            readout=bright_reset_time,
            save_data=False,
            plot_data=False,
        )

        # test pulse
        seq_args = [laser_515_delay, test_time, am_589_power, test_power, test_color]
        seq_args_string = tool_belt.encode_seq_args(seq_args)
        cxn.pulse_streamer.stream_immediate(pulse_file_name, 1, seq_args_string)

        # readout
        readout_file_name = "simple_readout.py"
        seq_args = [
            aom_589_delay,
            readout_time,
            am_589_power,
            am_515_power,
            apd_index,
            589,
        ]
        seq_args_string = tool_belt.encode_seq_args(seq_args)
        cxn.pulse_streamer.stream_load(readout_file_name, seq_args_string)

        # Load the APD
        cxn.apd_tagger.start_tag_stream(apd_indices)
        # Clear the buffer
        cxn.apd_tagger.clear_buffer()
        # Run the sequence
        cxn.pulse_streamer.stream_immediate(readout_file_name, 1, seq_args_string)

        new_counts = cxn.apd_tagger.read_counter_separate_gates(1)
        sample_counts = new_counts[0]
        off_counts.append(sample_counts[0])

    return on_counts, off_counts


# %%


def sweep_test_pulse_length(
    nv_sig,
    test_color,
    test_power_mW,
    modulation,
    test_power_V,
    test_pulse_dur_list=None,
):
    apd_indices = [0]
    num_reps = 25
    if not test_pulse_dur_list:
        test_pulse_dur_list = [
            0,
            25,
            50,
            75,
            100,
            150,
            200,
            250,
            300,
            400,
            500,
            750,
            1000,
            1500,
            2000,
        ]
    #        test_pulse_dur_list = [0]
    # measure laser powers:
    #    green_optical_power_pd, green_optical_power_mW, \
    #            red_optical_power_pd, red_optical_power_mW, \
    #            yellow_optical_power_pd, yellow_optical_power_mW = \
    #            tool_belt.measure_g_r_y_power(
    #                                  nv_sig['am_589_power'], nv_sig['nd_filter'])

    # create some lists for data
    on_count_raw = []
    off_count_raw = []

    if modulation == "no_scan":
        main_function = main_AM
    elif modulation == "scan":
        main_function = main_scan
    # Step through the pulse lengths for the test laser
    for test_time in test_pulse_dur_list:
        print("Testing {} us".format(test_time / 10**3))
        on_count, off_count = main_function(
            nv_sig,
            apd_indices,
            num_reps,
            test_color,
            test_time,
            test_power=test_power_V,
        )

        on_count = [int(el) for el in on_count]
        off_count = [int(el) for el in off_count]

        on_count_raw.append(on_count)
        off_count_raw.append(off_count)

    on_counts = numpy.average(on_count_raw, axis=1)
    off_counts = numpy.average(off_count_raw, axis=1)

    fig, popt = do_plot(on_counts, off_counts, test_pulse_dur_list, test_power_mW)

    a0, d0, a1, d1 = popt

    # Save
    test_pulse_dur_list = numpy.array(test_pulse_dur_list)
    timestamp = tool_belt.get_time_stamp()
    raw_data = {
        "timestamp": timestamp,
        "test_color": test_color,
        "test_power_mW": test_power_mW,
        "test_power_mW-units": "mW",
        "test_power_V": test_power_V,
        "test_power_V-units": "mW",
        "test_pulse_dur_list": test_pulse_dur_list.tolist(),
        "test_pulse_dur_list-units": "ns",
        "num_reps": num_reps,
        "nv_sig": nv_sig,
        "nv_sig-units": tool_belt.get_nv_sig_units(),
        "siv_pulse_time": siv_pulse_time,
        "siv_pulse_time-units": "ns",
        "siv_pulse_distance": siv_pulse_distance,
        "siv_pulse_distance-units": "V",
        "bright_reset_range": bright_reset_range,
        "bright_reset_steps": bright_reset_steps,
        "bright_reset_power": bright_reset_power,
        "bright_reset_time": bright_reset_time,
        "dark_reset_range": dark_reset_range,
        "dark_reset_steps": dark_reset_steps,
        "dark_reset_power": dark_reset_power,
        "dark_reset_time": dark_reset_time,
        "a0": a0,
        "d0": d0,
        "d0-units": "ms",
        "a1": a1,
        "d1": d1,
        "d1-units": "ms",
        #            'green_optical_power_pd': green_optical_power_pd,
        #            'green_optical_power_pd-units': 'V',
        #            'green_optical_power_mW': green_optical_power_mW,
        #            'green_optical_power_mW-units': 'mW',
        #            'red_optical_power_pd': red_optical_power_pd,
        #            'red_optical_power_pd-units': 'V',
        #            'red_optical_power_mW': red_optical_power_mW,
        #            'red_optical_power_mW-units': 'mW',
        #            'yellow_optical_power_pd': yellow_optical_power_pd,
        #            'yellow_optical_power_pd-units': 'V',
        #            'yellow_optical_power_mW': yellow_optical_power_mW,
        #            'yellow_optical_power_mW-units': 'mW',
        "on_count_raw": on_count_raw,
        "on_count_raw-units": "counts",
        "off_count_raw": off_count_raw,
        "off_count_raw-units": "counts",
    }

    file_path = tool_belt.get_file_path(__file__, timestamp, nv_sig["name"])
    tool_belt.save_raw_data(raw_data, file_path)
    tool_belt.save_figure(fig, file_path)

    print(" \nRoutine complete!")
    return d0, d1


# %% Run the files

if __name__ == "__main__":
    #    sample_name = 'goepert-mayer'

    expected_count_list = [40, 45, 65, 64, 55, 35, 40, 45]  #
    nv_coords_list = [
        [-0.037, 0.119, 5.14],
        [-0.090, 0.066, 5.04],
        [-0.110, 0.042, 5.13],
        [0.051, -0.115, 5.08],
        [-0.110, 0.042, 5.06],
        [0.063, 0.269, 5.09],
        [0.243, 0.184, 5.12],
        [0.086, 0.220, 5.03],
    ]

    nv_2021_03_30 = {
        "coords": [],
        "name": "",
        "expected_count_rate": None,
        "nd_filter": "nd_0",
        "color_filter": "635-715 bp",
        #            'color_filter': '715 lp',
        "pulsed_readout_dur": 300,
        "pulsed_SCC_readout_dur": 4 * 10**7,
        "am_589_power": 0.6,
        "pulsed_initial_ion_dur": 25 * 10**3,
        "pulsed_shelf_dur": 200,
        "am_589_shelf_power": 0.35,
        "pulsed_ionization_dur": 10**3,
        "cobalt_638_power": 40,
        "ao_515_pwr": 0.645,
        "pulsed_reionization_dur": 100 * 10**3,
        "cobalt_532_power": 10,
        "magnet_angle": 0,
        "resonance_LOW": 2.7,
        "rabi_LOW": 146.2,
        "uwave_power_LOW": 9.0,
        "resonance_HIGH": 2.9774,
        "rabi_HIGH": 95.2,
        "uwave_power_HIGH": 10.0,
    }

    test_pulses = [
        10**3,
        10**4,
        5 * 10**4,
        10**5,
        2.5 * 10**5,
        5 * 10**5,
        7.5 * 10**5,
        10**6,
        2.5 * 10**6,
        5 * 10**6,
        7.5 * 10**6,
        10**7,
        2.5 * 10**7,
        5 * 10**7,
        7.5 * 10**7,
        10**8,
        2.5 * 10**8,
        5 * 10**8,
    ]
    p_mw = [1.2, 1.62, 2.06, 1.82, 2.47, 0.76, 0.32, 0.039, 0.15, 0.018, 0.55]
    p_V = [0.63, 0.64, 0.65, 0.645, 0.66, 0.62, 0.61, 0.605, 0.607, 0.603, 0.615]

    d0_list = []
    d1_list = []
    for i in [5]:  # range(len(nv_coords_list)):
        nv_sig = copy.deepcopy(nv_2021_03_30)
        nv_sig["coords"] = nv_coords_list[i]
        nv_sig["expected_count_rate"] = expected_count_list[i]
        nv_sig["name"] = "goeppert-mayer-nv{}_2021_04_15".format(i)

        test_power_mw = 8.23
        test_power_V = 0.5
        d0, d1 = sweep_test_pulse_length(
            nv_sig,
            638,
            test_power_mw,
            "no_scan",
            test_power_V,
            test_pulse_dur_list=test_pulses,
        )
        d0_list.append(d0)
        d1_list.append(d1)

#    print(d0_list)
#    print(d1_list)
#    fig, ax = plt.subplots()
#    ax.plot(p_mw, 1/numpy.array(d0_list), 'ro', label = 'd0 rate')
#    ax.plot(p_mw, 1/numpy.array(d1_list), 'ko', label = 'd1 rate')
#    ax.set_ylabel('x 10^3 s^-1')
#    ax.set_xlabel('Power (mW)')
#    ax.legend()

#    folder = 'pc_rabi/branch_Spin_to_charge/determine_photoionization_rates_siv/2021_05'
#    file = '2021_05_05-21_10_25-goeppert-mayer-nv5_2021_04_15'
#    file= '2021_05_05-16_22_09-goeppert-mayer-nv5_2021_04_15'
#    data = tool_belt.get_raw_data(folder, file)
#    on_count_raw = data['on_count_raw']
#    off_count_raw = data['off_count_raw']
#    time_list = data['test_pulse_dur_list']
#    test_power_mW = data['test_power_mW']
#    on_counts = numpy.average(on_count_raw, axis = 1)
#    off_counts = numpy.average(off_count_raw, axis = 1)
#    do_plot(on_counts, off_counts, time_list, test_power_mW)
