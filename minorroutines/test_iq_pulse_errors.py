# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 15:17:35 2022

based off this paper: https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.105.077601

The signal, if no errors, should be half the contrast, which we will define as 0

@author: kolkowitz
"""


import copy
import time
from random import shuffle

import labrad
import matplotlib.pyplot as plt
import numpy
from numpy import pi
from scipy.optimize import curve_fit, minimize_scalar

import majorroutines.targeting as targeting
import utils.tool_belt as tool_belt
from utils.tool_belt import States


def measurement(
    cxn,
    nv_sig,
    uwave_pi_pulse,
    num_uwave_pulses,
    iq_phases,
    pulse_1_dur,
    pulse_2_dur,
    pulse_3_dur,
    num_runs=5,
    state=States.HIGH,
    do_plot=False,
    title=None,
    num_reps=int(1e5),
    inter_pulse_time=90,
):
    # print(iq_phases)
    # num_reps = int(5e4)

    tool_belt.reset_cfm(cxn)
    counter_server = tool_belt.get_server_counter(cxn)
    pulsegen_server = tool_belt.get_server_pulse_gen(cxn)
    arbwavegen_server = tool_belt.get_server_arb_wave_gen(cxn)
    seq_file = "test_iq_pulse_errors.py"
    #  Sequence setup

    laser_key = "spin_laser"
    laser_name = nv_sig[laser_key]
    tool_belt.set_filter(cxn, nv_sig, laser_key)
    laser_power = tool_belt.set_laser_power(cxn, nv_sig, laser_key)
    polarization_time = nv_sig["spin_pol_dur"]
    gate_time = nv_sig["spin_readout_dur"]

    ref_0_list = []
    ref_H_list = []
    sig_list = []

    ref_0_ste_list = []
    ref_H_ste_list = []
    sig_ste_list = []

    for n in range(num_runs):
        print(n)
        targeting.main_with_cxn(cxn, nv_sig)
        # Turn on the microwaves for determining microwave delay
        sig_gen_cxn = tool_belt.get_server_sig_gen(cxn, state)
        sig_gen_cxn.set_freq(nv_sig["resonance_{}".format(state.name)])
        sig_gen_cxn.set_amp(nv_sig["uwave_power_{}".format(state.name)])
        sig_gen_cxn.load_iq()
        sig_gen_cxn.uwave_on()
        arbwavegen_server.load_arb_phases(iq_phases)

        counter_server.start_tag_stream()

        seq_args = [
            gate_time,
            uwave_pi_pulse,
            pulse_1_dur,
            pulse_2_dur,
            pulse_3_dur,
            polarization_time,
            inter_pulse_time,
            num_uwave_pulses,
            state.value,
            laser_name,
            laser_power,
        ]
        # print(seq_args)
        # return
        counter_server.clear_buffer()
        seq_args_string = tool_belt.encode_seq_args(seq_args)
        pulsegen_server.stream_immediate(seq_file, num_reps, seq_args_string)

        new_counts = counter_server.read_counter_separate_gates(1)
        sample_counts = new_counts[0]
        if len(sample_counts) != 3 * num_reps:
            print("Error!")
        # first are the counts after polarization into ms = 0
        ref_0_counts = sample_counts[0::3]
        # second are the counts after a pi_x into +/-1
        ref_H_counts = sample_counts[1::3]
        # third are the counts after the uwave sequence
        sig_counts = sample_counts[2::3]

        counter_server.stop_tag_stream()

        tool_belt.reset_cfm(cxn)

        # analysis

        ref_0_sum = sum(ref_0_counts)
        ref_H_sum = sum(ref_H_counts)
        sig_sum = sum(sig_counts)

        ref_0_list.append(int(ref_0_sum))
        ref_H_list.append(int(ref_H_sum))
        sig_list.append(int(sig_sum))

        ref_0_ste_run = numpy.std(ref_0_counts, ddof=1) / numpy.sqrt(num_reps)
        ref_H_ste_run = numpy.std(ref_H_counts, ddof=1) / numpy.sqrt(num_reps)
        sig_ste_run = numpy.std(sig_counts, ddof=1) / numpy.sqrt(num_reps)

        ref_0_ste_list.append(ref_0_ste_run)
        ref_H_ste_list.append(ref_H_ste_run)
        sig_ste_list.append(sig_ste_run)

    ref_0_avg = sum(ref_0_list)  # Make this averaged
    ref_H_avg = sum(ref_H_list)
    sig_avg = sum(sig_list)

    ref_0_ste = sum(ref_0_ste_list)
    ref_H_ste = sum(ref_H_ste_list)
    sig_ste = sum(sig_ste_list)

    pop = (numpy.array(sig_list) - numpy.array(ref_H_list)) / (
        numpy.array(ref_0_list) - numpy.array(ref_H_list)
    )
    if do_plot:
        fig, ax = plt.subplots()
        ax.plot(range(num_runs), pop, "ro")
        ax.set_xlabel(r"Num repitition")
        ax.set_ylabel("Population")
        if title:
            ax.set_title(title)

    timestamp = tool_belt.get_time_stamp()

    raw_data = {
        "timestamp": timestamp,
        "nv_sig": nv_sig,
        "nv_sig-units": tool_belt.get_nv_sig_units(cxn),
        "num_uwave_pulses": num_uwave_pulses,
        "iq_phases": iq_phases,
        "pulse_durations": [pulse_1_dur, pulse_2_dur, pulse_3_dur],
        "state": state.name,
        "num_reps": num_reps,
        "num_runs": num_runs,
        "ref_0_list": ref_0_list,
        "ref_H_list": ref_H_list,
        "sig_list": sig_list,
        "population": pop.tolist(),
        "ref_0_ste_list": ref_0_ste_list,
        "ref_H_ste_list": ref_H_ste_list,
        "sig_ste_list": sig_ste_list,
    }

    nv_name = nv_sig["name"]
    file_path = tool_belt.get_file_path(__file__, timestamp, nv_name)
    if do_plot:
        tool_belt.save_figure(fig, file_path)
    tool_belt.save_raw_data(raw_data, file_path)
    # print(ref_0_avg, ref_H_avg, sig_avg)

    return ref_0_avg, ref_H_avg, sig_avg, ref_0_ste, ref_H_ste, sig_ste


def measure_pulse_error(
    cxn,
    nv_sig,
    uwave_pi_pulse,
    num_uwave_pulses,
    iq_phases,
    pulse_1_dur,
    pulse_2_dur,
    pulse_3_dur,
    state=States.HIGH,
    do_plot=False,
    Title=None,
):
    ret_vals = measurement(
        cxn,
        nv_sig,
        uwave_pi_pulse,
        num_uwave_pulses,
        iq_phases,
        pulse_1_dur,
        pulse_2_dur,
        pulse_3_dur,
        state=States.HIGH,
        do_plot=do_plot,
        title=Title,
    )

    ref_0_avg, ref_H_avg, sig_avg, ref_0_ste, ref_H_ste, sig_ste = ret_vals

    contrast = ref_0_avg - ref_H_avg
    contrast_ste = numpy.sqrt(ref_0_ste**2 + ref_H_ste**2)
    signal_m_H = sig_avg - ref_H_avg
    signal_m_H_ste = numpy.sqrt(sig_ste**2 + ref_H_ste**2)
    half_contrast = 0.5

    signal_perc = signal_m_H / contrast
    # print(signal_perc)
    signal_perc_ste = signal_perc * numpy.sqrt(
        (contrast_ste / contrast) ** 2 + (signal_m_H_ste / signal_m_H) ** 2
    )

    # half_contrast_counts = contrast/2 + ref_H_avg
    # print(half_contrast_counts)
    # half_contrast_ste = numpy.sqrt(ref_H_ste**2 + (contrast_ste/2)**2)

    pulse_error = (signal_perc - half_contrast) * 2
    pulse_error_ste = signal_perc_ste * 2

    return pulse_error, pulse_error_ste


def solve_errors(meas_list):
    A1 = meas_list[0]
    B1 = meas_list[1]

    A2 = meas_list[2]
    B2 = meas_list[3]

    A3 = meas_list[4]
    B3 = meas_list[5]

    S = meas_list[6:11]

    phi_p = -A1 / 2
    chi_p = -B1 / 2

    phi = A2 / 2 - phi_p
    chi = B2 / 2 - chi_p

    v_z = -(A3 - 2 * phi_p) / 2
    e_z = (B3 - 2 * chi_p) / 2

    M = numpy.array(
        [
            [-1, -1, -1, 0, 0],
            [1, -1, 1, 0, 0],
            [1, 1, -1, 2, 0],
            [-1, 1, 1, 2, 0],
            [-1, -1, 1, 0, 2],
            # [1,-1,-1,0,2] #exclude last equation
        ]
    )
    # print(S)
    X = numpy.linalg.inv(M).dot(S)

    e_y_p = 0  # we are setting this value to 0
    e_z_p = X[0]
    v_x_p = X[1]
    v_z_p = X[2]
    e_y = X[3]
    v_x = X[4]

    return [phi_p, chi_p, phi, chi, v_z, e_z, e_y_p, e_z_p, v_x_p, v_z_p, e_y, v_x]


def test_1_pulse(cxn, nv_sig, state=States.HIGH, int_phase=0, plot=False):
    """
    This pulse sequence consists of pi/2 pulses with the same phase:
        1: pi/2_x
        2: pi/2_y
    """

    num_uwave_pulses = 1

    rabi_period = nv_sig["rabi_{}".format(state.name)]

    # Get pulse frequencies
    uwave_pi_pulse = tool_belt.get_pi_pulse_dur(rabi_period)
    uwave_pi_on_2_pulse = tool_belt.get_pi_on_2_pulse_dur(rabi_period)

    pulse_1_dur = uwave_pi_on_2_pulse
    pulse_2_dur = 0
    pulse_3_dur = 0

    pi_x_phase = 0
    pi_2_x_phase = 0
    pi_y_phase = pi / 2 + int_phase
    pi_2_y_phase = pi / 2

    ##### 1
    iq_phases = [0, pi_2_x_phase]
    pulse_error, pulse_error_ste = measure_pulse_error(
        cxn,
        nv_sig,
        uwave_pi_pulse,
        num_uwave_pulses,
        iq_phases,
        pulse_1_dur,
        pulse_2_dur,
        pulse_3_dur,
        state=States.HIGH,
        do_plot=plot,
        Title="pi/2_x",
    )
    pe_1_1 = pulse_error
    pe_1_1_err = pulse_error_ste

    #### 2
    iq_phases = [0, pi_2_y_phase]
    pulse_error, pulse_error_ste = measure_pulse_error(
        cxn,
        nv_sig,
        uwave_pi_pulse,
        num_uwave_pulses,
        iq_phases,
        pulse_1_dur,
        pulse_2_dur,
        pulse_3_dur,
        state=States.HIGH,
        do_plot=plot,
        Title="pi/2_y",
    )
    pe_1_2 = pulse_error
    pe_1_2_err = pulse_error_ste

    print(
        r"pi/2_x rotation angle error, -2 phi' = {:.4f} +/- {:.4f}".format(
            pe_1_1, pe_1_1_err
        )
    )

    print(
        r"pi/2_y rotation angle error, -2 chi' = {:.4f} +/- {:.4f}".format(
            pe_1_2, pe_1_2_err
        )
    )
    return pe_1_1, pe_1_1_err, pe_1_2, pe_1_2_err


def test_2_pulse(cxn, nv_sig, state=States.HIGH, int_phase=0, plot=False):
    """
    1: pi_y - pi/2_x
    2: pi_x - pi/2_y
    3: pi/2_x - pi_y
    4: pi/2_y - pi_x
    5: pi/2_x - pi/2_y
    6: pi/2_y - pi/2_x
    """

    num_uwave_pulses = 2

    rabi_period = nv_sig["rabi_{}".format(state.name)]

    # Get pulse frequencies
    uwave_pi_pulse = tool_belt.get_pi_pulse_dur(rabi_period)
    uwave_pi_on_2_pulse = tool_belt.get_pi_on_2_pulse_dur(rabi_period)

    pi_x_phase = 0
    pi_2_x_phase = 0
    pi_y_phase = pi / 2
    pi_2_y_phase = pi / 2 + int_phase

    ### 1
    pulse_1_dur = uwave_pi_pulse
    pulse_2_dur = uwave_pi_on_2_pulse
    pulse_3_dur = 0

    iq_phases = [0, pi_y_phase, pi_2_x_phase]
    pulse_error, pulse_error_ste = measure_pulse_error(
        cxn,
        nv_sig,
        uwave_pi_pulse,
        num_uwave_pulses,
        iq_phases,
        pulse_1_dur,
        pulse_2_dur,
        pulse_3_dur,
        state,
        do_plot=plot,
        Title="pi_y - pi/2_x",
    )
    pe_2_1 = pulse_error
    pe_2_1_err = pulse_error_ste

    ### 2
    pulse_1_dur = uwave_pi_pulse
    pulse_2_dur = uwave_pi_on_2_pulse
    pulse_3_dur = 0

    iq_phases = [0, pi_x_phase, pi_2_y_phase]
    pulse_error, pulse_error_ste = measure_pulse_error(
        cxn,
        nv_sig,
        uwave_pi_pulse,
        num_uwave_pulses,
        iq_phases,
        pulse_1_dur,
        pulse_2_dur,
        pulse_3_dur,
        state,
        do_plot=plot,
        Title="pi_x - pi/2_y",
    )
    pe_2_2 = pulse_error
    pe_2_2_err = pulse_error_ste
    ### 3
    pulse_1_dur = uwave_pi_on_2_pulse
    pulse_2_dur = uwave_pi_pulse
    pulse_3_dur = 0

    iq_phases = [0, pi_2_x_phase, pi_y_phase]
    pulse_error, pulse_error_ste = measure_pulse_error(
        cxn,
        nv_sig,
        uwave_pi_pulse,
        num_uwave_pulses,
        iq_phases,
        pulse_1_dur,
        pulse_2_dur,
        pulse_3_dur,
        state,
        do_plot=plot,
        Title="pi/2_x - pi_y",
    )
    pe_2_3 = pulse_error
    pe_2_3_err = pulse_error_ste
    ### 4
    pulse_1_dur = uwave_pi_on_2_pulse
    pulse_2_dur = uwave_pi_pulse
    pulse_3_dur = 0

    iq_phases = [0, pi_2_y_phase, pi_x_phase]
    pulse_error, pulse_error_ste = measure_pulse_error(
        cxn,
        nv_sig,
        uwave_pi_pulse,
        num_uwave_pulses,
        iq_phases,
        pulse_1_dur,
        pulse_2_dur,
        pulse_3_dur,
        state,
        do_plot=plot,
        Title="pi/2_y - pi_x",
    )
    pe_2_4 = pulse_error
    pe_2_4_err = pulse_error_ste
    ### 5
    pulse_1_dur = uwave_pi_on_2_pulse
    pulse_2_dur = uwave_pi_on_2_pulse
    pulse_3_dur = 0

    iq_phases = [0, pi_2_x_phase, pi_2_y_phase]
    pulse_error, pulse_error_ste = measure_pulse_error(
        cxn,
        nv_sig,
        uwave_pi_pulse,
        num_uwave_pulses,
        iq_phases,
        pulse_1_dur,
        pulse_2_dur,
        pulse_3_dur,
        state,
        do_plot=plot,
        Title="pi/2_x - pi/2_y",
    )
    pe_2_5 = pulse_error
    pe_2_5_err = pulse_error_ste
    ### 6
    pulse_1_dur = uwave_pi_on_2_pulse
    pulse_2_dur = uwave_pi_on_2_pulse
    pulse_3_dur = 0

    iq_phases = [0, pi_2_y_phase, pi_2_x_phase]
    pulse_error, pulse_error_ste = measure_pulse_error(
        cxn,
        nv_sig,
        uwave_pi_pulse,
        num_uwave_pulses,
        iq_phases,
        pulse_1_dur,
        pulse_2_dur,
        pulse_3_dur,
        state,
        do_plot=plot,
        Title="pi/2_y - pi/2_x",
    )
    pe_2_6 = pulse_error
    pe_2_6_err = pulse_error_ste
    print(r"2 (phi' + phi) = {:.4f} +/- {:.4f}".format(pe_2_1, pe_2_1_err))
    print(r"2 (chi' + chi) = {:.4f} +/- {:.4f}".format(pe_2_2, pe_2_2_err))
    print(r"-2 v_z + 2 phi' = {:.4f} +/- {:.4f}".format(pe_2_3, pe_2_3_err))
    print(r"2 e_z + 2 chi' = {:.4f} +/- {:.4f}".format(pe_2_4, pe_2_4_err))
    print(r"-e_y' - e_z' - v_x' - v_z' = {:.4f} +/- {:.4f}".format(pe_2_5, pe_2_5_err))
    print(r"-e_y' + e_z' - v_x' + v_z' = {:.4f} +/- {:.4f}".format(pe_2_6, pe_2_6_err))

    ret_vals = (
        pe_2_1,
        pe_2_1_err,
        pe_2_2,
        pe_2_2_err,
        pe_2_3,
        pe_2_3_err,
        pe_2_4,
        pe_2_4_err,
        pe_2_5,
        pe_2_5_err,
        pe_2_6,
        pe_2_6_err,
    )
    return ret_vals


def test_3_pulse(cxn, nv_sig, state=States.HIGH, int_phase=0, plot=False):
    """
    pi/2_y - pi_x - pi/2_x
    pi/2_x - pi_x - pi/2_y
    pi/2_y - pi_y - pi/2_x
    pi/2_x - pi_y - pi/2_y
    """
    num_uwave_pulses = 3

    rabi_period = nv_sig["rabi_{}".format(state.name)]

    # Get pulse frequencies
    uwave_pi_pulse = tool_belt.get_pi_pulse_dur(rabi_period)
    uwave_pi_on_2_pulse = tool_belt.get_pi_on_2_pulse_dur(rabi_period)

    ### 1
    pulse_1_dur = uwave_pi_on_2_pulse
    pulse_2_dur = uwave_pi_pulse
    pulse_3_dur = uwave_pi_on_2_pulse

    pi_x_phase = 0
    pi_2_x_phase = 0
    pi_y_phase = pi / 2
    pi_2_y_phase = pi / 2 + int_phase

    iq_phases = [0, pi_2_y_phase, pi_x_phase, pi_2_x_phase]
    pulse_error, pulse_error_ste = measure_pulse_error(
        cxn,
        nv_sig,
        uwave_pi_pulse,
        num_uwave_pulses,
        iq_phases,
        pulse_1_dur,
        pulse_2_dur,
        pulse_3_dur,
        state,
        do_plot=plot,
        Title="pi/2_y - pi_x - pi/2_x",
    )
    pe_3_1 = pulse_error
    pe_3_1_err = pulse_error_ste
    ### 2
    pulse_1_dur = uwave_pi_on_2_pulse
    pulse_2_dur = uwave_pi_pulse
    pulse_3_dur = uwave_pi_on_2_pulse

    iq_phases = [0, pi_2_x_phase, pi_x_phase, pi_2_y_phase]
    pulse_error, pulse_error_ste = measure_pulse_error(
        cxn,
        nv_sig,
        uwave_pi_pulse,
        num_uwave_pulses,
        iq_phases,
        pulse_1_dur,
        pulse_2_dur,
        pulse_3_dur,
        state,
        do_plot=plot,
        Title="pi/2_x - pi_x - pi/2_y",
    )
    pe_3_2 = pulse_error
    pe_3_2_err = pulse_error_ste
    ### 3
    pulse_1_dur = uwave_pi_on_2_pulse
    pulse_2_dur = uwave_pi_pulse
    pulse_3_dur = uwave_pi_on_2_pulse

    iq_phases = [0, pi_2_y_phase, pi_y_phase, pi_2_x_phase]
    pulse_error, pulse_error_ste = measure_pulse_error(
        cxn,
        nv_sig,
        uwave_pi_pulse,
        num_uwave_pulses,
        iq_phases,
        pulse_1_dur,
        pulse_2_dur,
        pulse_3_dur,
        state,
        do_plot=plot,
        Title="pi/2_y - pi_y - pi/2_x",
    )
    pe_3_3 = pulse_error
    pe_3_3_err = pulse_error_ste
    ### 4
    pulse_1_dur = uwave_pi_on_2_pulse
    pulse_2_dur = uwave_pi_pulse
    pulse_3_dur = uwave_pi_on_2_pulse

    iq_phases = [0, pi_2_x_phase, pi_y_phase, pi_2_y_phase]
    pulse_error, pulse_error_ste = measure_pulse_error(
        cxn,
        nv_sig,
        uwave_pi_pulse,
        num_uwave_pulses,
        iq_phases,
        pulse_1_dur,
        pulse_2_dur,
        pulse_3_dur,
        state,
        do_plot=plot,
        Title="pi/2_x - pi_y - pi/2_y",
    )
    pe_3_4 = pulse_error
    pe_3_4_err = pulse_error_ste
    print(
        r"-e_y' + e_z' + v_x' - v_z' + 2e_y  = {:.4f} +/- {:.4f}".format(
            pe_3_1, pe_3_1_err
        )
    )
    print(
        r"-e_y' - e_z' + v_x' + v_z' + 2e_y  = {:.4f} +/- {:.4f}".format(
            pe_3_2, pe_3_2_err
        )
    )
    print(
        r"e_y' - e_z' - v_x' + v_z' + 2v_x  = {:.4f} +/- {:.4f}".format(
            pe_3_3, pe_3_3_err
        )
    )
    print(
        r"e_y' + e_z' - v_x' - v_z' + 2v_x  = {:.4f} +/- {:.4f}".format(
            pe_3_4, pe_3_4_err
        )
    )

    ret_vals = (
        pe_3_1,
        pe_3_1_err,
        pe_3_2,
        pe_3_2_err,
        pe_3_3,
        pe_3_3_err,
        pe_3_4,
        pe_3_4_err,
    )
    return ret_vals


def matrix_test(cxn, nv_sig, state=States.HIGH, int_phase=0, plot=True):
    """
    1: pi_y - pi/2_x
    2: pi_x - pi/2_y
    3: pi/2_x - pi_y
    4: pi/2_y - pi_x
    5: pi/2_x - pi/2_y
    6: pi/2_y - pi/2_x
    """

    num_uwave_pulses = 2

    rabi_period = nv_sig["rabi_{}".format(state.name)]

    # Get pulse frequencies
    uwave_pi_pulse = tool_belt.get_pi_pulse_dur(rabi_period)
    uwave_pi_on_2_pulse = tool_belt.get_pi_on_2_pulse_dur(rabi_period)

    pi_x_phase = 0
    pi_2_x_phase = 0
    pi_y_phase = pi / 2 + int_phase
    pi_2_y_phase = pi / 2

    ### 5
    pulse_1_dur = uwave_pi_on_2_pulse
    pulse_2_dur = uwave_pi_on_2_pulse
    pulse_3_dur = 0

    iq_phases = [0, pi_2_x_phase, pi_2_y_phase]
    pulse_error, pulse_error_ste = measure_pulse_error(
        cxn,
        nv_sig,
        uwave_pi_pulse,
        num_uwave_pulses,
        iq_phases,
        pulse_1_dur,
        pulse_2_dur,
        pulse_3_dur,
        state,
        do_plot=plot,
        Title="pi/2_x - pi/2_y",
    )
    pe_2_5 = pulse_error
    # pe_2_5_err = pulse_error_ste
    ### 6
    pulse_1_dur = uwave_pi_on_2_pulse
    pulse_2_dur = uwave_pi_on_2_pulse
    pulse_3_dur = 0

    iq_phases = [0, pi_2_y_phase, pi_2_x_phase]
    pulse_error, pulse_error_ste = measure_pulse_error(
        cxn,
        nv_sig,
        uwave_pi_pulse,
        num_uwave_pulses,
        iq_phases,
        pulse_1_dur,
        pulse_2_dur,
        pulse_3_dur,
        state,
        do_plot=plot,
        Title="pi/2_y - pi/2_x",
    )
    pe_2_6 = pulse_error
    # pe_2_6_err = pulse_error_ste

    """
        pi/2_y - pi_x - pi/2_x
        pi/2_x - pi_x - pi/2_y
        pi/2_y - pi_y - pi/2_x
        pi/2_x - pi_y - pi/2_y
    """
    num_uwave_pulses = 3

    rabi_period = nv_sig["rabi_{}".format(state.name)]

    # Get pulse frequencies
    uwave_pi_pulse = tool_belt.get_pi_pulse_dur(rabi_period)
    uwave_pi_on_2_pulse = tool_belt.get_pi_on_2_pulse_dur(rabi_period)

    ### 1
    pulse_1_dur = uwave_pi_on_2_pulse
    pulse_2_dur = uwave_pi_pulse
    pulse_3_dur = uwave_pi_on_2_pulse

    iq_phases = [0, pi_2_y_phase, pi_x_phase, pi_2_x_phase]
    pulse_error, pulse_error_ste = measure_pulse_error(
        cxn,
        nv_sig,
        uwave_pi_pulse,
        num_uwave_pulses,
        iq_phases,
        pulse_1_dur,
        pulse_2_dur,
        pulse_3_dur,
        state,
        do_plot=plot,
        Title="pi/2_y - pi_x - pi/2_x",
    )
    pe_3_1 = pulse_error
    # pe_3_1_err = pulse_error_ste
    ### 2
    pulse_1_dur = uwave_pi_on_2_pulse
    pulse_2_dur = uwave_pi_pulse
    pulse_3_dur = uwave_pi_on_2_pulse

    iq_phases = [0, pi_2_x_phase, pi_x_phase, pi_2_y_phase]
    pulse_error, pulse_error_ste = measure_pulse_error(
        cxn,
        nv_sig,
        uwave_pi_pulse,
        num_uwave_pulses,
        iq_phases,
        pulse_1_dur,
        pulse_2_dur,
        pulse_3_dur,
        state,
        do_plot=plot,
        Title="pi/2_x - pi_x - pi/2_y",
    )
    pe_3_2 = pulse_error
    # pe_3_2_err = pulse_error_ste
    ### 3
    pulse_1_dur = uwave_pi_on_2_pulse
    pulse_2_dur = uwave_pi_pulse
    pulse_3_dur = uwave_pi_on_2_pulse

    iq_phases = [0, pi_2_y_phase, pi_y_phase, pi_2_x_phase]
    pulse_error, pulse_error_ste = measure_pulse_error(
        cxn,
        nv_sig,
        uwave_pi_pulse,
        num_uwave_pulses,
        iq_phases,
        pulse_1_dur,
        pulse_2_dur,
        pulse_3_dur,
        state,
        do_plot=plot,
        Title="pi/2_y - pi_y - pi/2_x",
    )
    pe_3_3 = pulse_error
    # pe_3_3_err = pulse_error_ste
    ### 4
    pulse_1_dur = uwave_pi_on_2_pulse
    pulse_2_dur = uwave_pi_pulse
    pulse_3_dur = uwave_pi_on_2_pulse

    iq_phases = [0, pi_2_x_phase, pi_y_phase, pi_2_y_phase]
    pulse_error, pulse_error_ste = measure_pulse_error(
        cxn,
        nv_sig,
        uwave_pi_pulse,
        num_uwave_pulses,
        iq_phases,
        pulse_1_dur,
        pulse_2_dur,
        pulse_3_dur,
        state,
        do_plot=plot,
        Title="pi/2_x - pi_y - pi/2_y",
    )
    pe_3_4 = pulse_error
    # pe_3_4_err = pulse_error_ste

    ret_vals = [pe_2_5, pe_2_6, pe_3_1, pe_3_2, pe_3_3, pe_3_4]

    return ret_vals


def full_test(cxn, nv_sig, apd_indices, state=States.HIGH, int_phase=0, plot=True):
    pe1, pe1e, pe2, pe2e = test_1_pulse(
        cxn, nv_sig, apd_indices, state, int_phase, plot
    )

    ret_vals = test_2_pulse(cxn, nv_sig, apd_indices, state, int_phase, plot)
    pe3, pe3e, pe4, pe4e, pe5, pe5e, pe6, pe6e, pe7, pe7e, pe8, pe8e = ret_vals

    ret_vals = test_3_pulse(cxn, nv_sig, apd_indices, state, int_phase, plot)

    pe9, pe9e, pe10, pe10e, pe11, pe11e, pe12, pe12e = ret_vals

    # print([pe1, pe2, pe3, pe4, pe5, pe6, pe7, pe8, pe9, pe10,
    #        pe11, pe12])

    return [pe1, pe2, pe3, pe4, pe5, pe6, pe7, pe8, pe9, pe10, pe11, pe12]


def sweep_inter_pulse_time(
    cxn,
    nv_sig,
    apd_indices,
    init_phase,
    state=States.HIGH,
):
    num_uwave_pulses = 2

    phi = 180

    rabi_period = nv_sig["rabi_{}".format(state.name)]

    # Get pulse frequencies
    uwave_pi_pulse = tool_belt.get_pi_pulse_dur(rabi_period)
    uwave_pi_on_2_pulse = tool_belt.get_pi_on_2_pulse_dur(rabi_period)

    num_steps = 16
    # num_steps = 2
    num_reps = int(1e5)
    num_runs = 10
    t_range = [0, 150]
    ref_0_list = numpy.zeros([num_steps])
    ref_0_list[:] = numpy.nan
    ref_H_list = numpy.copy(ref_0_list)
    sig_list = numpy.copy(ref_0_list)

    t_list = numpy.linspace(t_range[0], t_range[-1], num_steps)
    # print(phi_list)
    # return
    # Create a list of indices to step through the taus. This will be shuffled
    t_ind_list = list(range(0, num_steps))
    shuffle(t_ind_list)

    for ti in t_ind_list:
        t = t_list[ti]
        print("wait time = {} ns".format(t))
        # iq_phases = [0, 0, phi*pi/180]
        iq_phases = [init_phase, init_phase, phi * pi / 180]

        ### 1
        pulse_1_dur = uwave_pi_pulse
        pulse_2_dur = uwave_pi_pulse
        pulse_3_dur = 0

        ret_vals = measurement(
            cxn,
            nv_sig,
            uwave_pi_pulse,
            num_uwave_pulses,
            iq_phases,
            pulse_1_dur,
            pulse_2_dur,
            pulse_3_dur,
            apd_indices,
            num_runs,
            state,
            num_reps=num_reps,
            do_plot=False,
            inter_pulse_time=t,
        )

        ref_0_avg, ref_H_avg, sig_avg, ref_0_ste, ref_H_ste, sig_ste = ret_vals

        ref_0_list[ti] = ref_0_avg
        ref_H_list[ti] = ref_H_avg
        sig_list[ti] = sig_avg

    population = (sig_list - ref_H_list) / (ref_0_list - ref_H_list)

    fig, axes = plt.subplots(1, 2, figsize=(17, 8.5))
    ax = axes[0]
    ax.plot(t_list, ref_0_list, "r--", label="low reference")
    ax.plot(t_list, ref_H_list, "g--", label="high reference")
    ax.plot(t_list, sig_list, "b-", label="signal")
    ax.set_xlabel(r"Inter pulse wait time for MW pusles (ns)")
    # ax.set_xlabel(r"Relative phase of second pi/2 pulse (degrees)")
    ax.set_ylabel("Counts")
    ax.legend()
    ax = axes[1]
    ax.plot(t_list, population, "b-")
    ax.set_xlabel(r"Inter pulse wait time for MW pusles (ns)")
    # ax.set_xlabel(r"Relative phase of second pi/2 pulse (degrees)")
    ax.set_ylabel("Population")
    ax.set_title("Two pi pulses, {} deg out of phase".format(phi))

    timestamp = tool_belt.get_time_stamp()

    raw_data = {
        "timestamp": timestamp,
        "nv_sig": nv_sig,
        "nv_sig-units": tool_belt.get_nv_sig_units(),
        "phi": phi,
        "phi-units": "degrees",
        "t_range": t_range,
        "t_range-units": "ns",
        "init_phase": init_phase,
        "t_list": t_list.tolist(),
        "t_list-units": "ns",
        "state": state.name,
        "num_steps": num_steps,
        "num_reps": num_reps,
        "t_ind_list": t_ind_list,
        "sig_list": sig_list.astype(int).tolist(),
        "sig_list-units": "counts",
        "ref_0_list": ref_0_list.astype(int).tolist(),
        "ref_0_list-units": "counts",
        "ref_H_list": ref_H_list.astype(int).tolist(),
        "ref_H_list-units": "counts",
        "population": population.tolist(),
    }

    nv_name = nv_sig["name"]
    file_path = tool_belt.get_file_path(__file__, timestamp, nv_name)
    tool_belt.save_figure(fig, file_path)
    tool_belt.save_raw_data(raw_data, file_path)


def custom_phase(
    cxn,
    nv_sig,
    apd_indices,
    init_phase,
    state=States.HIGH,
):
    num_uwave_pulses = 2

    rabi_period = nv_sig["rabi_{}".format(state.name)]

    # Get pulse frequencies
    uwave_pi_pulse = tool_belt.get_pi_pulse_dur(rabi_period)
    uwave_pi_on_2_pulse = tool_belt.get_pi_on_2_pulse_dur(rabi_period)

    num_steps = 31
    inter_pulse = 90
    # num_steps = 2
    num_reps = int(5e5)
    num_runs = 10
    phi_range = [0, 360 * 2]
    ref_0_list = numpy.zeros([num_steps])
    ref_0_list[:] = numpy.nan
    ref_H_list = numpy.copy(ref_0_list)
    sig_list = numpy.copy(ref_0_list)

    phi_list = numpy.linspace(phi_range[0], phi_range[-1], num_steps)
    # print(phi_list)
    # return
    # Create a list of indices to step through the taus. This will be shuffled
    phi_ind_list = list(range(0, num_steps))
    shuffle(phi_ind_list)

    for p in phi_ind_list:
        phi = phi_list[p]
        print("phase = {} deg".format(phi))
        # iq_phases = [0, 0, phi*pi/180]
        iq_phases = [init_phase, init_phase, phi * pi / 180]

        ### 1
        pulse_1_dur = uwave_pi_pulse
        pulse_2_dur = uwave_pi_pulse
        pulse_3_dur = 0

        ret_vals = measurement(
            cxn,
            nv_sig,
            uwave_pi_pulse,
            num_uwave_pulses,
            iq_phases,
            pulse_1_dur,
            pulse_2_dur,
            pulse_3_dur,
            apd_indices,
            num_runs,
            state,
            num_reps=num_reps,
            do_plot=False,
            inter_pulse_time=inter_pulse,
        )

        ref_0_avg, ref_H_avg, sig_avg, ref_0_ste, ref_H_ste, sig_ste = ret_vals

        ref_0_list[p] = ref_0_avg
        ref_H_list[p] = ref_H_avg
        sig_list[p] = sig_avg

    population = (sig_list - ref_H_list) / (ref_0_list - ref_H_list)

    fig, axes = plt.subplots(1, 2, figsize=(17, 8.5))
    ax = axes[0]
    ax.plot(phi_list, ref_0_list, "r--", label="low reference")
    ax.plot(phi_list, ref_H_list, "g--", label="high reference")
    ax.plot(phi_list, sig_list, "b-", label="signal")
    ax.set_xlabel(r"Relative phase of second pi pulse (degrees)")
    # ax.set_xlabel(r"Relative phase of second pi/2 pulse (degrees)")
    ax.set_ylabel("Counts")
    ax.legend()
    ax = axes[1]
    ax.plot(phi_list, population, "b-")
    ax.set_xlabel(r"Relative phase of second pi pulse (degrees)")
    # ax.set_xlabel(r"Relative phase of second pi/2 pulse (degrees)")
    ax.set_ylabel("Population")
    ax.set_title("two consecutive pi pulses")

    timestamp = tool_belt.get_time_stamp()

    raw_data = {
        "timestamp": timestamp,
        "nv_sig": nv_sig,
        "nv_sig-units": tool_belt.get_nv_sig_units(),
        "mw_interpulse_delay": inter_pulse,
        "mw_interpulse_delay-units": "ns",
        "phi_range": phi_range,
        "phi_range-units": "degrees",
        "init_phase": init_phase,
        "phi_list": phi_list.tolist(),
        "phi_list-units": "degrees",
        "state": state.name,
        "num_steps": num_steps,
        "num_reps": num_reps,
        "phi_ind_list": phi_ind_list,
        "sig_list": sig_list.astype(int).tolist(),
        "sig_list-units": "counts",
        "ref_0_list": ref_0_list.astype(int).tolist(),
        "ref_0_list-units": "counts",
        "ref_H_list": ref_H_list.astype(int).tolist(),
        "ref_H_list-units": "counts",
        "population": population.tolist(),
    }

    nv_name = nv_sig["name"]
    file_path = tool_belt.get_file_path(__file__, timestamp, nv_name)
    tool_belt.save_figure(fig, file_path)
    tool_belt.save_raw_data(raw_data, file_path)


def custom_dur(
    cxn,
    nv_sig,
    apd_indices,
    init_phase,
    state=States.HIGH,
):
    num_uwave_pulses = 2

    rabi_period = nv_sig["rabi_{}".format(state.name)]

    # Get pulse frequencies
    uwave_pi_pulse = tool_belt.get_pi_pulse_dur(rabi_period)
    uwave_pi_on_2_pulse = tool_belt.get_pi_on_2_pulse_dur(rabi_period)

    # num_steps = 15
    num_steps = 11
    num_reps = int(1e5)
    num_runs = 10
    dt_range = [-20, 20]
    ref_0_list = numpy.zeros([num_steps])
    ref_0_list[:] = numpy.nan
    ref_H_list = numpy.copy(ref_0_list)
    sig_list = numpy.copy(ref_0_list)

    dt_durs = numpy.linspace(dt_range[0], dt_range[-1], num_steps)
    # print(phi_list)
    # return
    # Create a list of indices to step through the taus. This will be shuffled
    dt_ind_list = list(range(0, num_steps))
    shuffle(dt_ind_list)

    for ind in dt_ind_list:
        dt = dt_durs[ind]
        uwave_pi_on_2_pulse_dt = uwave_pi_on_2_pulse + dt
        print("pi/2 pulse duration = {} ns".format(uwave_pi_on_2_pulse_dt))
        # iq_phases = [0, 0, phi*pi/180]
        iq_phases = [init_phase, init_phase, init_phase]

        ### 1
        pulse_1_dur = uwave_pi_on_2_pulse_dt
        pulse_2_dur = uwave_pi_on_2_pulse_dt
        pulse_3_dur = 0

        ret_vals = measurement(
            cxn,
            nv_sig,
            uwave_pi_pulse,
            num_uwave_pulses,
            iq_phases,
            pulse_1_dur,
            pulse_2_dur,
            pulse_3_dur,
            apd_indices,
            num_runs,
            state,
            num_reps,
        )

        ref_0_avg, ref_H_avg, sig_avg, ref_0_ste, ref_H_ste, sig_ste = ret_vals

        ref_0_list[ind] = ref_0_avg
        ref_H_list[ind] = ref_H_avg
        sig_list[ind] = sig_avg

    population = (sig_list - ref_H_list) / (ref_0_list - ref_H_list)

    fig, axes = plt.subplots(1, 2, figsize=(17, 8.5))
    ax = axes[0]
    ax.plot(dt_durs, ref_0_list, "r--", label="low reference")
    ax.plot(dt_durs, ref_H_list, "g--", label="high reference")
    ax.plot(dt_durs, sig_list, "b-", label="signal")
    ax.set_xlabel(r"Change in duration of pi/2 pulse (ns)")
    # ax.set_xlabel(r"Relative phase of second pi/2 pulse (degrees)")
    ax.set_ylabel("Counts")
    ax.legend()
    ax = axes[1]
    ax.plot(dt_durs, population, "b-")
    ax.set_xlabel(r"Change in duration of pi/2 pulse (ns)")
    # ax.set_xlabel(r"Relative phase of second pi/2 pulse (degrees)")
    ax.set_ylabel("Population")
    ax.set_title("two consecutive pi/2 pulses")

    timestamp = tool_belt.get_time_stamp()

    raw_data = {
        "timestamp": timestamp,
        "nv_sig": nv_sig,
        "nv_sig-units": tool_belt.get_nv_sig_units(),
        "dt_range": dt_range,
        "dt_range-units": "ns",
        "init_phase": init_phase,
        "dt_durs": dt_durs.tolist(),
        "dt_durs-units": "ns",
        "state": state.name,
        "num_steps": num_steps,
        "num_reps": num_reps,
        "dt_ind_list": dt_ind_list,
        "sig_list": sig_list.astype(int).tolist(),
        "sig_list-units": "counts",
        "ref_0_list": ref_0_list.astype(int).tolist(),
        "ref_0_list-units": "counts",
        "ref_H_list": ref_H_list.astype(int).tolist(),
        "ref_H_list-units": "counts",
        "population": population.tolist(),
    }

    nv_name = nv_sig["name"]
    file_path = tool_belt.get_file_path(__file__, timestamp, nv_name)
    tool_belt.save_figure(fig, file_path)
    tool_belt.save_raw_data(raw_data, file_path)


def fit_custom_data(population, phi_list):
    init_params = [0.2, 0.4, 0]

    fit_func = lambda t, amp, offset, phase: tool_belt.sin_1_at_0_phase(
        t, amp, offset, 1, phase
    )

    phi_list_rads = numpy.array(phi_list) * pi / 180
    popt, _ = curve_fit(
        fit_func,
        phi_list_rads,
        population,
        p0=init_params,
        bounds=([0, 0, -numpy.infty], numpy.infty),
    )
    # print(popt)
    phis_lin = numpy.linspace(phi_list_rads[0], phi_list_rads[-1], 100)
    fig, ax = plt.subplots()
    ax.plot(phi_list_rads, population, "ko", label="data")
    ax.plot(phis_lin, fit_func(phis_lin, *popt), "r-", label="fit")
    ax.set_xlabel(r"Relative phase of second pi/2 pulse (radians)")
    ax.set_ylabel("Population")

    text_popt = "\n".join(
        (
            r"$C + A_0 \mathrm{sin}(\nu t + \phi - \pi/2)$",
            r"$C = $" + "%.3f" % (popt[1]),
            r"$A_0 = $" + "%.3f" % (popt[0]),
            # r'$\frac{1}{\nu} = $' + '%.1f'%(1/popt[2]) + ' ns',
            r"$\phi = $" + "%.2f" % (popt[2]) + " " + r"$ rad$",
        )
    )

    props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    ax.text(
        0.55,
        0.45,
        text_popt,
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=props,
    )

    ax.legend()


def replot_imposed_phases(file, folder):
    data = tool_belt.get_raw_data(file, folder)

    phases = data["phases"]
    phi_p_list = data["phi_p_list"]
    chi_p_list = data["chi_p_list"]
    phi_list = data["phi_list"]
    chi_list = data["chi_list"]
    e_z_p_list = data["e_z_p_list"]
    v_x_p_list = data["v_x_p_list"]
    v_z_p_list = data["v_z_p_list"]
    e_y_list = data["e_y_list"]
    v_x_list = data["v_x_list"]
    v_z_list = data["v_z_list"]
    e_z_list = data["e_z_list"]

    plot_errors_vs_changed_param(
        phases,
        "Imposed phase on pi/2_y pulse (deg)",
        phi_p_list,
        chi_p_list,
        phi_list,
        chi_list,
        e_z_p_list,
        v_x_p_list,
        v_z_p_list,
        e_y_list,
        v_x_list,
        v_z_list,
        e_z_list,
        do_expected_phases=True,
    )


def replot_change_freq(file, folder):
    data = tool_belt.get_raw_data(file, folder)

    d_freq = numpy.array(data["d_freq"])
    phi_p_list = data["phi_p_list"]
    chi_p_list = data["chi_p_list"]
    phi_list = data["phi_list"]
    chi_list = data["chi_list"]
    e_z_p_list = data["e_z_p_list"]
    v_x_p_list = data["v_x_p_list"]
    v_z_p_list = data["v_z_p_list"]
    e_y_list = data["e_y_list"]
    v_x_list = data["v_x_list"]
    v_z_list = data["v_z_list"]
    e_z_list = data["e_z_list"]

    plot_errors_vs_changed_param(
        d_freq * 1e3,
        "Detuning (MHz)",
        phi_p_list,
        chi_p_list,
        phi_list,
        chi_list,
        e_z_p_list,
        v_x_p_list,
        v_z_p_list,
        e_y_list,
        v_x_list,
        v_z_list,
        e_z_list,
    )


def lin_line(x, a):
    return x * a


def plot_errors_vs_changed_param(
    x_vals,
    x_axis_label,
    phi_p_list,
    chi_p_list,
    phi_list,
    chi_list,
    e_z_p_list,
    v_x_p_list,
    v_z_p_list,
    e_y_list,
    v_x_list,
    v_z_list,
    e_z_list,
    do_expected_phases=False,
):
    if len(phi_p_list) != 0:
        fig1, ax = plt.subplots()
        ax.plot(x_vals, phi_p_list, "ro", label=r"$\Phi'$")
        ax.plot(x_vals, chi_p_list, "bo", label=r"$\chi'$")
        ax.plot(x_vals, phi_list, "go", label=r"$\Phi$")
        ax.plot(x_vals, chi_list, "mo", label=r"$\chi$")
        ax.set_xlabel(x_axis_label)
        ax.set_ylabel("Error")
        ax.legend()

    if len(e_z_p_list) != 0:
        fig2, ax = plt.subplots()
        ax.plot(x_vals, e_z_p_list, "ro", label=r"$e_z'$")
        ax.plot(x_vals, v_x_p_list, "bo", label=r"$v_x'$")
        ax.plot(x_vals, v_z_p_list, "go", label=r"$v_z'$")
        ax.plot(x_vals, e_y_list, "mo", label=r"$e_y$")
        ax.plot(x_vals, v_x_list, "co", label=r"$v_x$")

        if do_expected_phases:
            x_start = min(x_vals)
            x_end = max(x_vals)
            lin_x = numpy.linspace(x_start, x_end, 100)
            ax.plot(lin_x, lin_line(lin_x, pi / 180), "r-", label="expected")

        ax.set_xlabel(x_axis_label)
        ax.set_ylabel("Error")
        ax.legend()

    if len(v_z_list) != 0:
        fig3, ax = plt.subplots()
        ax.plot(x_vals, v_z_list, "ro", label=r"$v_z$")
        ax.plot(x_vals, e_z_list, "bo", label=r"$e_z$")
        ax.set_xlabel(x_axis_label)
        ax.set_ylabel("Error")
        ax.legend()


def do_full_impose_phase(
    cxn,
    nv_sig,
    apd_indices,
    state=States.HIGH,
):
    phi_list = []
    chi_list = []
    phi_p_list = []
    chi_p_list = []

    v_z_list = []
    e_z_list = []

    e_y_p_list = []
    e_z_p_list = []
    v_x_p_list = []
    v_z_p_list = []
    e_y_list = []
    v_x_list = []

    errs_list = []
    phases = numpy.linspace(-30, 30, 7)
    for p in phases:
        phase_rad = p * pi / 180
        s_list = full_test(
            cxn, nv_sig, apd_indices, state=state, int_phase=phase_rad, plot=False
        )
        errs = solve_errors(s_list)
        errs_list.append(errs)

        phi_p_list.append(errs[0])
        chi_p_list.append(errs[1])
        phi_list.append(errs[2])
        chi_list.append(errs[3])

        v_z_list.append(errs[4])
        e_z_list.append(errs[5])

        e_y_p_list.append(errs[6])

        e_z_p_list.append(errs[7])
        v_x_p_list.append(errs[8])
        v_z_p_list.append(errs[9])
        e_y_list.append(errs[10])
        v_x_list.append(errs[11])

    plot_errors_vs_changed_param(
        phases,
        "Imposed phase on pi/2_y pulse (deg)",
        phi_p_list,
        chi_p_list,
        phi_list,
        chi_list,
        e_z_p_list,
        v_x_p_list,
        v_z_p_list,
        e_y_list,
        v_x_list,
        v_z_list,
        e_z_list,
        do_expected_phases=True,
    )

    timestamp = tool_belt.get_time_stamp()

    raw_data = {
        "timestamp": timestamp,
        "nv_sig": nv_sig,
        "nv_sig-units": tool_belt.get_nv_sig_units(),
        "phases": phases.tolist(),
        "phases-units": "degrees",
        "phi_list": phi_list,
        "chi_list": chi_list,
        "phi_p_list": phi_p_list,
        "chi_p_list": chi_p_list,
        "v_z_list": v_z_list,
        "e_z_list": e_z_list,
        "e_y_p_list": e_y_p_list,
        "e_z_p_list": e_z_p_list,
        "v_x_p_list": v_x_p_list,
        "v_z_p_list": v_z_p_list,
        "e_y_list": e_y_list,
        "v_x_list": v_x_list,
        "errs_list": errs_list,
    }

    nv_name = nv_sig["name"]
    file_path = tool_belt.get_file_path(__file__, timestamp, nv_name)
    tool_belt.save_raw_data(raw_data, file_path)
    # tool_belt.save_figure(fig1, file_path)
    # tool_belt.save_figure(fig2, file_path)
    # tool_belt.save_figure(fig3, file_path)


def do_impose_phase(
    cxn,
    nv_sig,
    state=States.HIGH,
):
    phi_list = []
    chi_list = []
    phi_p_list = []
    chi_p_list = []

    v_z_list = []
    e_z_list = []

    e_y_p_list = []
    e_z_p_list = []
    v_x_p_list = []
    v_z_p_list = []
    e_y_list = []
    v_x_list = []

    errs_list = []

    phases = numpy.linspace(-30, 30, 5)
    # print(phases*pi/180)
    # return
    shuffle(phases)
    for p in phases:
        phase_rad = p * pi / 180
        # print(phase_rad)
        s_list = matrix_test(cxn, nv_sig, state=state, int_phase=phase_rad, plot=False)

        s_list = [0, 0, 0, 0, 0, 0] + s_list
        errs = solve_errors(s_list)
        errs_list.append(errs)

        # phi_p_list.append(errs[0])
        # chi_p_list.append(errs[1])
        # phi_list.append(errs[2])
        # chi_list.append(errs[3])

        # v_z_list.append(errs[4])
        # e_z_list.append(errs[5])

        e_y_p_list.append(errs[6])

        e_z_p_list.append(errs[7])
        v_x_p_list.append(errs[8])
        v_z_p_list.append(errs[9])
        e_y_list.append(errs[10])
        v_x_list.append(errs[11])

    plot_errors_vs_changed_param(
        phases,
        "Imposed phase on pi/2_y pulse (deg)",
        phi_p_list,
        chi_p_list,
        phi_list,
        chi_list,
        e_z_p_list,
        v_x_p_list,
        v_z_p_list,
        e_y_list,
        v_x_list,
        v_z_list,
        e_z_list,
        do_expected_phases=True,
    )

    timestamp = tool_belt.get_time_stamp()

    raw_data = {
        "timestamp": timestamp,
        "nv_sig": nv_sig,
        "nv_sig-units": tool_belt.get_nv_sig_units(cxn),
        "phases": phases.tolist(),
        "phases-units": "degrees",
        "phi_list": phi_list,
        "chi_list": chi_list,
        "phi_p_list": phi_p_list,
        "chi_p_list": chi_p_list,
        "v_z_list": v_z_list,
        "e_z_list": e_z_list,
        "e_y_p_list": e_y_p_list,
        "e_z_p_list": e_z_p_list,
        "v_x_p_list": v_x_p_list,
        "v_z_p_list": v_z_p_list,
        "e_y_list": e_y_list,
        "v_x_list": v_x_list,
        "errs_list": errs_list,
    }

    nv_name = nv_sig["name"]
    file_path = tool_belt.get_file_path(__file__, timestamp, nv_name)
    tool_belt.save_raw_data(raw_data, file_path)
    # tool_belt.save_figure(fig1, file_path)
    # tool_belt.save_figure(fig2, file_path)
    # tool_belt.save_figure(fig3, file_path)


def do_change_freq(
    cxn,
    nv_sig,
    apd_indices,
    state=States.HIGH,
):
    phi_list = []
    chi_list = []
    phi_p_list = []
    chi_p_list = []

    v_z_list = []
    e_z_list = []

    e_y_p_list = []
    e_z_p_list = []
    v_x_p_list = []
    v_z_p_list = []
    e_y_list = []
    v_x_list = []

    errs_list = []

    plot_freqs = []

    d_freq = numpy.linspace(-0.01, 0.01, 7)

    freq = nv_sig["resonance_{}".format(state.name)]

    shuffle(d_freq)
    for df in d_freq:
        adjusted_freq = freq + df
        plot_freqs.append(adjusted_freq)
        nv_sig_copy = copy.deepcopy(nv_sig)
        nv_sig_copy["resonance_{}".format(state.name)] = adjusted_freq
        # s_list = matrix_test(cxn,
        s_list = full_test(
            cxn, nv_sig_copy, apd_indices, state=state, int_phase=0, plot=False
        )

        # s_list = [0,0,0,0,0,0] + s_list
        errs = solve_errors(s_list)
        errs_list.append(errs)

        phi_p_list.append(errs[0])
        chi_p_list.append(errs[1])
        phi_list.append(errs[2])
        chi_list.append(errs[3])

        v_z_list.append(errs[4])
        e_z_list.append(errs[5])

        e_y_p_list.append(errs[6])

        e_z_p_list.append(errs[7])
        v_x_p_list.append(errs[8])
        v_z_p_list.append(errs[9])
        e_y_list.append(errs[10])
        v_x_list.append(errs[11])

    plot_errors_vs_changed_param(
        d_freq * 1e3,
        "Detuning (MHz)",
        phi_p_list,
        chi_p_list,
        phi_list,
        chi_list,
        e_z_p_list,
        v_x_p_list,
        v_z_p_list,
        e_y_list,
        v_x_list,
        v_z_list,
        e_z_list,
    )

    timestamp = tool_belt.get_time_stamp()

    raw_data = {
        "timestamp": timestamp,
        "nv_sig": nv_sig,
        "nv_sig-units": tool_belt.get_nv_sig_units(),
        "d_freq": d_freq.tolist(),
        "d_freq-units": "GHz",
        "phi_list": phi_list,
        "chi_list": chi_list,
        "phi_p_list": phi_p_list,
        "chi_p_list": chi_p_list,
        "v_z_list": v_z_list,
        "e_z_list": e_z_list,
        "e_y_p_list": e_y_p_list,
        "e_z_p_list": e_z_p_list,
        "v_x_p_list": v_x_p_list,
        "v_z_p_list": v_z_p_list,
        "e_y_list": e_y_list,
        "v_x_list": v_x_list,
        "errs_list": errs_list,
    }

    nv_name = nv_sig["name"]
    file_path = tool_belt.get_file_path(__file__, timestamp, nv_name)
    tool_belt.save_raw_data(raw_data, file_path)
    # tool_belt.save_figure(fig1, file_path)
    # tool_belt.save_figure(fig2, file_path)
    # tool_belt.save_figure(fig3, file_path)


# %%
if __name__ == "__main__":
    sample_name = "siena"
    green_power = 8000
    nd_green = "nd_1.1"
    green_laser = "integrated_520"
    yellow_laser = "laserglow_589"
    red_laser = "cobolt_638"

    nv_sig = {
        "coords": [0.030, -0.302, 5.09],
        "name": "{}-nv4_2023_01_16".format(
            sample_name,
        ),
        "disable_opt": False,
        "ramp_voltages": False,
        "expected_count_rate": 42,
        "spin_laser": green_laser,
        "spin_laser_power": green_power,
        "spin_laser_filter": nd_green,
        "spin_readout_dur": 350,
        "spin_pol_dur": 1000.0,
        "imaging_laser": green_laser,
        "imaging_laser_power": green_power,
        "imaging_laser_filter": nd_green,
        "imaging_readout_dur": 1e7,
        "charge_readout_laser": yellow_laser,
        "charge_readout_laser_filter": "nd_0",
        "collection_filter": "715_sp+630_lp",  # NV band only
        "magnet_angle": 53.5,
        "resonance_LOW": 2.81921,
        "rabi_LOW": 67 * 2,
        "uwave_power_LOW": 15,
        "resonance_HIGH": 2.92159,
        "rabi_HIGH": 128 * 2,
        "uwave_power_HIGH": 10,
        "pi_pulse_LOW": 67,
        "pi_on_2_pulse_LOW": 33,  # 37,
        "pi_pulse_HIGH": 128,
        "pi_on_2_pulse_HIGH": 59,
    }

    with labrad.connect() as cxn:
        # do_impose_phase(cxn,
        #               nv_sig,
        #             )

        # do_change_freq(cxn,
        #               nv_sig,
        #               apd_indices)

        # s_list = full_test(cxn,
        #               nv_sig,
        #               apd_indices,
        #               state=States.HIGH,
        #               int_phase = 0)
        # errs = solve_errors(s_list )
        # print(errs)

        # s_list = matrix_test(cxn,
        #               nv_sig,
        #               apd_indices,
        #               state=States.HIGH,
        #               int_phase = 30,
        #               plot = False)
        # print(s_list)

        # init_phase =0
        # custom_phase(cxn,
        #                    nv_sig,
        #                    apd_indices,
        #                    init_phase,
        #                    States.HIGH)

        # sweep_inter_pulse_time(cxn,
        #                 nv_sig,
        #                 apd_indices,
        #                 init_phase,
        #                 States.HIGH,)

        test_3_pulse(
            cxn,
            nv_sig,
            state=States.HIGH,
            int_phase=0,
        )
