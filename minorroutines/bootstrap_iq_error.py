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

# from scipy.optimize import minimize_scalar
from numpy import pi
from scipy.optimize import curve_fit, fsolve

import majorroutines.targeting as targeting
import utils.kplotlib as kpl
import utils.tool_belt as tool_belt
from utils.kplotlib import KplColors
from utils.tool_belt import States


def solve_linear(m, b, y, z):
    x = z[0]

    F = numpy.empty((1))

    F[0] = m * x + b - y
    return F


def lin_line(x, a):
    return x * a


def replot_imposed_phases(file):
    """
    Replotting functionality for measurements where an intentional phase was applied
    """
    data = tool_belt.get_raw_data(file)

    phases = data["phases"]
    # phi_p_list = data['phi_p_list']
    # chi_p_list = data['chi_p_list']
    # phi_list = data['phi_list']
    # chi_list= data['chi_list']
    e_z_p_list = data["e_z_p_list"]
    v_x_p_list = data["v_x_p_list"]
    v_z_p_list = data["v_z_p_list"]
    e_y_list = data["e_y_list"]
    v_x_list = data["v_x_list"]
    # v_z_list = data['v_z_list']
    # e_z_list = data['e_z_list']
    title = data["title"]

    e_z_p_err_list = data["e_z_p_err_list"]
    v_x_p_err_list = data["v_x_p_err_list"]
    v_z_p_err_list = data["v_z_p_err_list"]
    e_y_err_list = data["e_y_err_list"]
    v_x_err_list = data["v_x_err_list"]

    plot_errors_vs_changed_phase(
        phases,
        title,
        e_z_p_list,
        v_x_p_list,
        v_z_p_list,
        e_y_list,
        v_x_list,
        e_z_p_err_list,
        v_x_p_err_list,
        v_z_p_err_list,
        e_y_err_list,
        v_x_err_list,
        do_expected_phases=True,
    )


def plot_errors_vs_changed_duration(x_vals, title, y_vals, y_vals_ste):
    kpl.init_kplotlib()
    # fig2, ax = plt.subplots()
    # ax.errorbar(x_vals,e_z_p_list, yerr = e_z_p_err_list, fmt= 'ko', label = r"$e_z'$" )

    # ax.errorbar(x_vals,phi_p_list, yerr = chi_p_list, fmt= 'bo', label = r"$Phi'$" )
    # ax.errorbar(x_vals,chi_p_list, yerr = chi_p_err_list, fmt= 'ro', label = r"$Chi'$" )

    ### fit linear line to initial slope
    mid_value = y_vals[int(len(y_vals) / 2)]
    fit_func = tool_belt.linear
    init_params = [-1, 0]

    popt, pcov = curve_fit(
        fit_func, x_vals, y_vals, p0=init_params, sigma=y_vals_ste, absolute_sigma=True
    )

    # print(popt)

    fig, ax = plt.subplots()
    ax.set_xlabel("Pulse duration (ns)")
    ax.set_ylabel("Error")
    ax.set_title(title)

    kpl.plot_points(ax, x_vals, y_vals, yerr=y_vals_ste, color=KplColors.BLACK)

    t_start = min(x_vals)
    t_end = max(x_vals)
    smooth_t = numpy.linspace(t_start, t_end, num=100)
    kpl.plot_line(
        ax,
        smooth_t,
        fit_func(smooth_t, *popt),
        color=KplColors.RED,
    )

    # # find intersection of linear line and offset (half of full range)
    solve_linear_func = lambda z: solve_linear(popt[0], popt[1], 0, z)
    zGuess = numpy.array(mid_value)
    solve = fsolve(solve_linear_func, zGuess)
    x_intercept = solve[0]
    print(x_intercept)

    text = "Optimum pulse dur {:.1f} ns".format(x_intercept)
    size = kpl.Size.SMALL
    kpl.anchored_text(ax, text, kpl.Loc.LOWER_LEFT, size=size)

    return fig


def plot_errors_vs_changed_phase(
    x_vals,
    x_axis_label,
    e_z_p_list,
    v_x_p_list,
    v_z_p_list,
    e_y_list,
    v_x_list,
    e_z_p_err_list,
    v_x_p_err_list,
    v_z_p_err_list,
    e_y_err_list,
    v_x_err_list,
    do_expected_phases=False,
):
    """
    Plotting capabilities
    """

    fig2, ax = plt.subplots()
    # ax.errorbar(x_vals,e_z_p_list, yerr = e_z_p_err_list, fmt= 'ko', label = r"$e_z'$" )

    if x_axis_label == "Imposed phase on pi/2_y pulse (deg)":
        multiplier = -1
        ax.errorbar(
            x_vals,
            v_x_p_list,
            yerr=v_x_p_err_list,
            fmt="bo",
            markeredgewidth=1.5,
            markeredgecolor="r",
            label=r"$v_x'$",
        )
    else:
        ax.errorbar(x_vals, v_x_p_list, yerr=v_x_p_err_list, fmt="bo", label=r"$v_x'$")

    # ax.errorbar(x_vals,v_z_p_list, yerr = v_z_p_err_list, fmt= 'go', label = r"$v_z'$" )

    if x_axis_label == "Imposed phase on pi_x pulse (deg)":
        multiplier = 1
        ax.errorbar(
            x_vals,
            e_y_list,
            yerr=e_y_err_list,
            fmt="mo",
            markeredgewidth=1.5,
            markeredgecolor="r",
            label=r"$e_y$",
        )
    else:
        ax.errorbar(x_vals, e_y_list, yerr=e_y_err_list, fmt="mo", label=r"$e_y$")

    if x_axis_label == "Imposed phase on pi_y pulse (deg)":
        multiplier = -1
        ax.errorbar(
            x_vals,
            v_x_list,
            yerr=v_x_err_list,
            fmt="co",
            markeredgewidth=1.5,
            markeredgecolor="r",
            label=r"$v_x$",
        )
    else:
        ax.errorbar(x_vals, v_x_list, yerr=v_x_err_list, fmt="co", label=r"$v_x$")

    if do_expected_phases:
        x_start = min(x_vals)
        x_end = max(x_vals)
        lin_x = numpy.linspace(x_start, x_end, 100)
        ax.plot(lin_x, lin_line(lin_x, multiplier * pi / 180), "r-", label="expected")

    ax.set_xlabel(x_axis_label)
    ax.set_ylabel("Error")
    ax.legend()
    return fig2


def measurement(
    cxn,
    nv_sig,
    num_uwave_pulses,
    iq_phases,
    pulse_1_dur,
    pulse_2_dur,
    pulse_3_dur,
    num_runs,
    num_reps,
    state=States.HIGH,
    do_plot=False,
    title=None,
    do_dq=False,
    inter_pulse_time=30,  # between SQ/DQ MW pulses
    inter_composite_time=80,
):  # between pulses making up one DQ pulse
    """
    The basic building block to perform these measurements. Can apply 1, 2, or 3
    MW pulses, and returns counts from [ms=0, ms=+/-1, counts after mw pulses]

    nv_sig: dictionary
        the dictionary of the nv_sig
    iq_phases: list
        list of phases for the IQ modulation. First value is the phase of the
        pi pulse used to measure counts from +/-1. In radians
    pulse_1_dur: int
        length of time for first MW pulse, either pi/2 or pi pulse
    pulse_2_dur: int
        if applicable, length of time for second MW pulse, either pi/2 or pi pulse
    pulse_3_dur: int
        if applicable, length of time for third MW pulse, either pi/2 or pi pulse
    num_runs: int
        number of runs to sum over. Will optimize before every run
    state: state value
        the state (and thus signal generator) to run MW through (needs IQ mod capabilities)
    do_plot: True/False
        If True, will plot the population calculated from measurement, for each run
    title: string
        if do_plot, provide a title for the plot
    inter_puls_time: int
        duration of time between MW pulses. see lab notebook 11/18/2022


    RETURNS
    ref_0_sum: int
        NV prepared in ms = 0, sum of all counts collected
    ref_H_sum: int
        NV prepared in ms = +/-1, sum of all counts collected
    sig_sum: int
        NV after MW pulses applied, sum of all counts collected
    ref_0_ste: float
        NV prepared in ms = 0, sum of stndard error measured for each run
    ref_H_ste: float
        NV prepared in ms = +/-1, sum of stndard error measured for each run
    sig_ste: float
        NV after MW pulses applied, sum of stndard error measured for each run
    """
    do_sum = True

    tool_belt.reset_cfm(cxn)
    kpl.init_kplotlib()
    counter_server = tool_belt.get_server_counter(cxn)
    pulsegen_server = tool_belt.get_server_pulse_gen(cxn)
    arbwavegen_server = tool_belt.get_server_arb_wave_gen(cxn)

    #  Sequence setup
    laser_key = "spin_laser"
    laser_name = nv_sig[laser_key]
    tool_belt.set_filter(cxn, nv_sig, laser_key)
    laser_power = tool_belt.set_laser_power(cxn, nv_sig, laser_key)
    polarization_time = nv_sig["spin_pol_dur"]
    gate_time = nv_sig["spin_readout_dur"]

    if do_dq:
        if state.value == States.HIGH.value:
            state_activ = States.HIGH
            state_proxy = States.LOW
        elif state.value == States.LOW.value:
            state_activ = States.LOW
            state_proxy = States.HIGH

    ref_0_list = []
    ref_H_list = []
    sig_list = []

    ref_0_ste_list = []
    ref_H_ste_list = []
    sig_ste_list = []
    if do_dq:
        uwave_pi_pulse_low = nv_sig["pi_pulse_X_{}".format(States.LOW.name)]
        uwave_pi_pulse_high = nv_sig["pi_pulse_X_{}".format(States.HIGH.name)]
        # rabi_low =  nv_sig["rabi_{}".format(States.LOW.name)]
        # rabi_high =  nv_sig["rabi_{}".format(States.HIGH.name)]
        # uwave_pi_pulse_low =  tool_belt.get_pi_pulse_dur(rabi_low)
        # uwave_pi_pulse_high = tool_belt.get_pi_pulse_dur(rabi_high)

    else:
        uwave_pi_pulse = nv_sig["pi_pulse_X_{}".format(state.name)]

    for n in range(num_runs):
        print(n)
        targeting.main_with_cxn(cxn, nv_sig)
        # Turn on the microwaves for determining microwave delay
        if do_dq:
            sig_gen_low_cxn = tool_belt.get_server_sig_gen(cxn, States.LOW)
            sig_gen_low_cxn.set_freq(nv_sig["resonance_{}".format(States.LOW.name)])
            sig_gen_low_cxn.set_amp(nv_sig["uwave_power_{}".format(States.LOW.name)])
            sig_gen_low_cxn.uwave_on()
            sig_gen_high_cxn = tool_belt.get_server_sig_gen(cxn, States.HIGH)
            sig_gen_high_cxn.set_freq(nv_sig["resonance_{}".format(States.HIGH.name)])
            sig_gen_high_cxn.set_amp(nv_sig["uwave_power_{}".format(States.HIGH.name)])
            sig_gen_high_cxn.load_iq()
            sig_gen_high_cxn.uwave_on()
        else:
            sig_gen_cxn = tool_belt.get_server_sig_gen(cxn, state)
            sig_gen_cxn.set_freq(nv_sig["resonance_{}".format(state.name)])
            sig_gen_cxn.set_amp(nv_sig["uwave_power_{}".format(state.name)])
            sig_gen_cxn.load_iq()
            sig_gen_cxn.uwave_on()
        arbwavegen_server.load_arb_phases(iq_phases)

        counter_server.start_tag_stream()

        if do_dq:
            seq_file = "test_iq_pulse_errors_dq.py"
            seq_args = [
                gate_time,
                uwave_pi_pulse_low,
                uwave_pi_pulse_high,
                pulse_1_dur,
                pulse_2_dur,
                pulse_3_dur,
                polarization_time,
                inter_pulse_time,
                inter_composite_time,
                num_uwave_pulses,
                state_activ.value,
                state_proxy.value,
                laser_name,
                laser_power,
            ]
        else:
            seq_file = "test_iq_pulse_errors.py"
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
        print(seq_args)
        print(iq_phases)
        # return
        counter_server.clear_buffer()
        seq_args_string = tool_belt.encode_seq_args(seq_args)
        pulsegen_server.stream_immediate(seq_file, num_reps, seq_args_string)

        new_counts = counter_server.read_counter_modulo_gates(3, 1)
        # print(new_counts)
        sample_counts = new_counts[0]
        ref_0 = sample_counts[0]
        ref_H = sample_counts[1]
        sig = sample_counts[2]

        # new_counts = counter_server.read_counter_separate_gates(1)
        # sample_counts = new_counts[0]
        # if len(sample_counts) != 3 * num_reps:
        #     print("Error!")
        # # first are the counts after polarization into ms = 0
        # ref_0_counts = sample_counts[0::3]
        # # second are the counts after a pi_x into +/-1
        # ref_H_counts = sample_counts[1::3]
        # # third are the counts after the uwave sequence
        # sig_counts = sample_counts[2::3]

        counter_server.stop_tag_stream()

        tool_belt.reset_cfm(cxn)

        # analysis

        ref_0_list.append(ref_0)
        ref_H_list.append(ref_H)
        sig_list.append(sig)

    ref_0_avg = numpy.average(ref_0_list)
    ref_H_avg = numpy.average(ref_H_list)
    sig_avg = numpy.average(sig_list)

    print((ref_0_avg / (num_reps * 1000)) / gate_time * 1e9)
    print((ref_H_avg / (num_reps * 1000)) / gate_time * 1e9)
    print((sig_avg / (num_reps * 1000)) / gate_time * 1e9)

    # error analysis for summing each run
    ref_0_ste_avg = numpy.sqrt(ref_0_avg) / numpy.sqrt(num_runs)
    ref_H_ste_avg = numpy.sqrt(ref_H_avg) / numpy.sqrt(num_runs)
    sig_ste_avg = numpy.sqrt(sig_avg) / numpy.sqrt(num_runs)

    sig_counts_avg = sig_avg - ref_H_avg
    sig_counts_ste = numpy.sqrt(sig_ste_avg**2 + ref_H_ste_avg**2)
    con_counts_avg = ref_0_avg - ref_H_avg
    con_counts_ste = numpy.sqrt(ref_0_ste_avg**2 + ref_H_ste_avg**2)

    norm_avg_sig = sig_counts_avg / con_counts_avg

    norm_avg_sig_ste = norm_avg_sig * numpy.sqrt(
        (sig_counts_ste / sig_counts_avg) ** 2 + (con_counts_ste / con_counts_avg) ** 2
    )

    half_contrast = 0.5
    # Subtract 0.5, so that the expectation value is centered at 0, and
    # multiply by 2 so that we report the expectation value from -1 to +1.
    pulse_error = (norm_avg_sig - half_contrast) * 2
    pulse_error_ste = norm_avg_sig_ste * 2

    N = numpy.array(sig_list) - numpy.array(ref_H_list)
    D = numpy.array(ref_0_list) - numpy.array(ref_H_list)
    # N_unc = numpy.sqrt( numpy.array(sig_ste_list)**2 + numpy.array(ref_H_ste_list)**2)
    # D_unc = numpy.sqrt( numpy.array(ref_0_ste_list)**2 + numpy.array(ref_H_ste_list)**2)

    pop = (N / D - half_contrast) * 2
    # pop_unc = pop * numpy.sqrt( (N_unc/N)**2 + (D_unc/D)**2)

    if do_plot:
        fig, ax = plt.subplots()
        kpl.plot_points(ax, range(num_runs), pop, color=KplColors.RED)
        # ax.error_bar(range(num_runs), pop,y_err=pop_unc, "ro")
        ax.set_xlabel(r"Num repitition")
        ax.set_ylabel("Error")
        # kpl.anchored_text(ax, "{} ns delay between HIGH/LOW pulses\n{} ns delay between SQ/DQ pulses\npulse_error: {:.3f}".format(inter_uwave_buffer,inter_pulse_time,pulse_error ), kpl.Loc.UPPER_RIGHT)
        if title:
            ax.set_title(title)

    timestamp = tool_belt.get_time_stamp()

    raw_data = {
        "timestamp": timestamp,
        "nv_sig": nv_sig,
        "do_dq": do_dq,
        "num_uwave_pulses": num_uwave_pulses,
        "iq_phases": iq_phases,
        "pulse_durations": [pulse_1_dur, pulse_2_dur, pulse_3_dur],
        "state": state.name,
        "num_reps": num_reps,
        "num_runs": num_runs,
        "inter_pulse_time": inter_pulse_time,
        "inter_composite_time": inter_composite_time,
        # 'population': pop.tolist(),
        "ref_0_avg": ref_0_avg,
        "ref_H_avg": ref_H_avg,
        "sig_avg": sig_avg,
        "ref_0_ste_avg": ref_0_ste_avg,
        "ref_H_ste_avg": ref_H_ste_avg,
        "sig_ste_avg": sig_ste_avg,
        "ref_0_list": ref_0_list,
        "ref_H_list": ref_H_list,
        "sig_list": sig_list,
        "ref_0_ste_list": ref_0_ste_list,
        "ref_H_ste_list": ref_H_ste_list,
        "sig_ste_list": sig_ste_list,
    }

    nv_name = nv_sig["name"]
    file_path = tool_belt.get_file_path(__file__, timestamp, nv_name)
    # if do_plot:
    #     tool_belt.save_figure(fig, file_path)
    do_save = False
    if do_save:
        tool_belt.save_raw_data(raw_data, file_path)

    return pulse_error, pulse_error_ste


def solve_errors(meas_list):
    """
    Given a list of the measured signals from the bootstrapping method,
    calculate the pulse errors.

    This follows the order of measurements stated in
    https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.105.077601

    Like in the paper, we set the error of the pi2_x error along the y axis as 0

    """
    A1 = meas_list[0]
    B1 = meas_list[1]

    A2 = meas_list[2]
    B2 = meas_list[3]

    A3 = meas_list[4]
    B3 = meas_list[5]

    # S = meas_list[6:11]

    phi_p = -A1 / 2
    chi_p = -B1 / 2

    phi = A2 / 2 - phi_p
    chi = B2 / 2 - chi_p

    v_z = -(A3 - 2 * phi_p) / 2
    e_z = (B3 - 2 * chi_p) / 2

    # S = meas_list[6:12]
    # e_y_p = 0
    # v_x_p = -(S[0]+S[1])/2
    # e_y = (S[2]+S[3]-2*v_x_p)/4
    # v_x = (S[4]+S[5]+2*v_x_p)/4
    # v_z_p = -(S[0] - S[1] + S[2] - S[3])/4
    # e_z_p = -(S[0] - S[1] - S[2] + S[3])/4

    S = meas_list[6:11]
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
    X = numpy.linalg.inv(M).dot(S)

    e_y_p = 0  # we are setting this value to 0
    e_z_p = X[0]
    v_x_p = X[1]
    v_z_p = X[2]
    e_y = X[3]
    v_x = X[4]

    return [phi_p, chi_p, phi, chi, v_z, e_z, e_y_p, e_z_p, v_x_p, v_z_p, e_y, v_x]


def calc_pulse_error_ste(ste_list):
    """
    Given a list of the ste's of the measured signals, calculate the ste of the
    pulse errors

    """
    A1 = ste_list[0]
    B1 = ste_list[1]

    A2 = ste_list[2]
    B2 = ste_list[3]

    A3 = ste_list[4]
    B3 = ste_list[5]

    S = ste_list[6:12]

    phi_p_ste = A1 / 2
    chi_p_ste = B1 / 2

    phi_ste = numpy.sqrt((A2 / 2) ** 2 + (phi_p_ste) ** 2)
    chi_ste = numpy.sqrt((B2 / 2) ** 2 + (chi_p_ste) ** 2)

    v_z_ste = numpy.sqrt((A3 / 2) ** 2 + (phi_p_ste) ** 2)
    e_z_ste = numpy.sqrt((B3 / 2) ** 2 + (chi_p_ste) ** 2)

    e_y_p_ste = 0  # we are setting this value to 0
    v_x_p_ste = numpy.sqrt(S[0] ** 2 + S[1] ** 2) / 2
    e_y_ste = numpy.sqrt(S[2] ** 2 + S[3] ** 2 + (2 * v_x_p_ste) ** 2) / 4
    v_x_ste = numpy.sqrt(S[4] ** 2 + S[5] ** 2 + (2 * v_x_p_ste) ** 2) / 4
    v_z_p_ste = numpy.sqrt(S[0] ** 2 + S[1] ** 2 + S[2] ** 2 + S[3] ** 2) / 4
    e_z_p_ste = numpy.sqrt(S[0] ** 2 + S[1] ** 2 + S[2] ** 2 + S[3] ** 2) / 4

    # I don't know a good way to propegate the uncertainty through a system
    # of linear equations so instead, I'm going to try to give a best guess for the uncertainty

    # first, I take the average value of the uncert from measurements 7 and 8, and then 9-12
    # because the the respective two measurements are identical for extracting uncertainty:
    # ei: the 7th and 8th measurements result in: DS78 = Sqrt(Dey'^2 + Dez'^2 + Dvx'^2 + Dvz'^2)
    # so the ste from those measurements "should" be the sam,e so we'll just average them.
    # S78_ste = numpy.average([S[0], S[1]])
    # S910_ste = numpy.average([S[2], S[3]])
    # S112_ste = numpy.average([S[4], S[5]])

    # e_y_p_ste = 0 # we are setting this value to 0

    # #then, the equations for uncert: DS78 = sqrt(Dey'^2 + Dez'^2 + Dvx'^2 + Dvz'^2)
    # # we set Dey' == 0, so then, and we will weight the other uncert equally,
    # # so i.e.: Dez' = Sqrt(DS7**2 / 3)
    # e_z_p_ste = numpy.sqrt(S78_ste**2 / 3)
    # v_x_p_ste = numpy.sqrt(S78_ste**2 / 3)
    # v_z_p_ste = numpy.sqrt(S78_ste**2 / 3)
    # # Similarly, for the final two variables, they are in the equations like:
    # # DS910 = sqrt(Dey'^2 + Dez'^2 + Dvx'^2 + Dvz'^2 + (2Dvx)^2)
    # # DS910 = sqrt(DS78^2 + (2Dvx)^2)
    # # and to get Dvx,
    # # Dvx = Sqrt((DS910^2 - DS78^2) / 2 )
    # #   note that I take the absolute difference to avoid imaginary values
    # e_y_ste = numpy.sqrt(abs(S910_ste**2 - S78_ste**2) / 2)
    # v_x_ste = numpy.sqrt(abs(S112_ste**2 - S78_ste**2) / 2)

    return [
        phi_p_ste,
        chi_p_ste,
        phi_ste,
        chi_ste,
        v_z_ste,
        e_z_ste,
        e_y_p_ste,
        e_z_p_ste,
        v_x_p_ste,
        v_z_p_ste,
        e_y_ste,
        v_x_ste,
    ]


def do_measurement(
    cxn,
    nv_sig,
    num_runs,
    num_reps,
    experiment_ind,
    pi_y_ph=0,
    pi_x_ph=0,
    pi_2_y_ph=0,
    pi_dt=0,
    pi_2_dt=0,
    state=States.HIGH,
    do_dq=False,
    plot=False,
    inter_pulse_time=20,  # between SQ/DQ MW pulses
    inter_composite_time=80,
):
    """
    shell function to run any of the 12 measurements:
        0: pi/2_x
        1: pi/2_y
        2: pi_x - pi/2_x
        3: pi_y - pi/2_y
        4: pi/2_x - pi_y
        5: pi/2_y - pi_x
        6: pi/2_x - pi/2_y
        7: pi/2_y - pi/2_x
        8: pi/2_y - pi_x - pi/2_x
        9: pi/2_x - pi_x - pi/2_y
        10: pi/2_y - pi_y - pi/2_x
        11: pi/2_x - pi_y - pi/2_y
    """

    # Get pulse frequencies
    uwave_pi_pulse_X = nv_sig["pi_pulse_X_{}".format(state.name)] + pi_dt
    uwave_pi_pulse_Y = nv_sig["pi_pulse_Y_{}".format(state.name)] + pi_dt
    uwave_pi_on_2_pulse_X = nv_sig["pi_on_2_pulse_X_{}".format(state.name)] + pi_2_dt
    uwave_pi_on_2_pulse_Y = nv_sig["pi_on_2_pulse_Y_{}".format(state.name)] + pi_2_dt

    pi_x_phase = 0.0 + pi_x_ph
    pi_2_x_phase = 0.0 + 0
    pi_y_phase = pi / 2 + pi_y_ph
    pi_2_y_phase = pi / 2 + pi_2_y_ph

    if experiment_ind == 0:
        num_uwave_pulses = 1
        pulse_1_dur = uwave_pi_on_2_pulse_X
        pulse_2_dur = 0
        pulse_3_dur = 0
        iq_phases = [0.0, pi_2_x_phase]
        title = "pi/2_x"

    elif experiment_ind == 1:
        num_uwave_pulses = 1
        pulse_1_dur = uwave_pi_on_2_pulse_Y
        pulse_2_dur = 0
        pulse_3_dur = 0
        iq_phases = [0.0, pi_2_y_phase]
        title = "pi/2_y"

    elif experiment_ind == 2:
        num_uwave_pulses = 2
        pulse_1_dur = uwave_pi_pulse_X
        pulse_2_dur = uwave_pi_on_2_pulse_X
        pulse_3_dur = 0
        iq_phases = [0.0, pi_x_phase, pi_2_x_phase]
        title = "pi_x - pi/2_x"

    elif experiment_ind == 3:
        num_uwave_pulses = 2
        pulse_1_dur = uwave_pi_pulse_Y
        pulse_2_dur = uwave_pi_on_2_pulse_Y
        pulse_3_dur = 0
        iq_phases = [0.0, pi_y_phase, pi_2_y_phase]
        title = "pi_y - pi/2_y"

    elif experiment_ind == 4:
        num_uwave_pulses = 2
        pulse_1_dur = uwave_pi_on_2_pulse_X
        pulse_2_dur = uwave_pi_pulse_Y
        pulse_3_dur = 0
        iq_phases = [0.0, pi_2_x_phase, pi_y_phase]
        title = "pi/2_x - pi_y"

    elif experiment_ind == 5:
        num_uwave_pulses = 2
        pulse_1_dur = uwave_pi_on_2_pulse_Y
        pulse_2_dur = uwave_pi_pulse_X
        pulse_3_dur = 0
        iq_phases = [0.0, pi_2_y_phase, pi_x_phase]
        title = "pi/2_y - pi_x"

    elif experiment_ind == 6:
        num_uwave_pulses = 2
        pulse_1_dur = uwave_pi_on_2_pulse_X
        pulse_2_dur = uwave_pi_on_2_pulse_Y
        pulse_3_dur = 0
        iq_phases = [0.0, pi_2_x_phase, pi_2_y_phase]
        title = "pi/2_x - pi/2_y"

    elif experiment_ind == 7:
        num_uwave_pulses = 2
        pulse_1_dur = uwave_pi_on_2_pulse_Y
        pulse_2_dur = uwave_pi_on_2_pulse_X
        pulse_3_dur = 0
        iq_phases = [0.0, pi_2_y_phase, pi_2_x_phase]
        title = "pi/2_y - pi/2_x"

    elif experiment_ind == 8:
        num_uwave_pulses = 3
        pulse_1_dur = uwave_pi_on_2_pulse_Y
        pulse_2_dur = uwave_pi_pulse_X
        pulse_3_dur = uwave_pi_on_2_pulse_X
        iq_phases = [0.0, pi_2_y_phase, pi_x_phase, pi_2_x_phase]
        title = "pi/2_y - pi_x - pi/2_x"

    elif experiment_ind == 9:
        num_uwave_pulses = 3
        pulse_1_dur = uwave_pi_on_2_pulse_X
        pulse_2_dur = uwave_pi_pulse_X
        pulse_3_dur = uwave_pi_on_2_pulse_Y
        iq_phases = [0.0, pi_2_x_phase, pi_x_phase, pi_2_y_phase]
        title = "pi/2_x - pi_x - pi/2_y"

    elif experiment_ind == 10:
        num_uwave_pulses = 3
        pulse_1_dur = uwave_pi_on_2_pulse_Y
        pulse_2_dur = uwave_pi_pulse_Y
        pulse_3_dur = uwave_pi_on_2_pulse_X
        iq_phases = [0.0, pi_2_y_phase, pi_y_phase, pi_2_x_phase]
        title = "pi/2_y - pi_y - pi/2_x"

    elif experiment_ind == 11:
        num_uwave_pulses = 3
        pulse_1_dur = uwave_pi_on_2_pulse_X
        pulse_2_dur = uwave_pi_pulse_Y
        pulse_3_dur = uwave_pi_on_2_pulse_Y
        iq_phases = [0.0, pi_2_x_phase, pi_y_phase, pi_2_y_phase]
        title = "pi/2_x - pi_y - pi/2_y"

    elif experiment_ind == 101:
        num_uwave_pulses = 3
        pulse_1_dur = uwave_pi_on_2_pulse_X
        pulse_2_dur = uwave_pi_pulse_X
        pulse_3_dur = uwave_pi_on_2_pulse_X
        iq_phases = [0.0, pi_2_x_phase, pi_x_phase, pi_2_x_phase]
        title = "pi/2_x - pi_x - pi/2_x"

    # for dq, we need to add in an extra phase at the end for the final pi pulse
    # to read out to be at phase 0.
    if do_dq:
        iq_phases = iq_phases + [0.0]

    print("Running experiment index {}".format(experiment_ind))
    pulse_error, pulse_error_ste = measurement(
        cxn,
        nv_sig,
        num_uwave_pulses,
        iq_phases,
        pulse_1_dur,
        pulse_2_dur,
        pulse_3_dur,
        num_runs,
        num_reps,
        state=state,
        do_plot=plot,
        title=title,
        do_dq=do_dq,
        inter_pulse_time=inter_pulse_time,  # between SQ/DQ MW pulses
        inter_composite_time=inter_composite_time,
    )

    return pulse_error, pulse_error_ste


def measure_pulse_errors(
    cxn,
    nv_sig,
    num_runs,
    num_reps,
    state=States.HIGH,
    pi_y_ph=0,
    pi_x_ph=0,
    pi_2_y_ph=0,
    pi_dt=0,
    pi_2_dt=0,
    do_dq=False,
    plot=False,
    ret_ste=True,
):
    """
    This function runs all 12 measurements, and returns the measured signals
    """
    s_list = []
    ste_list = []
    measurement_ind_list = [el for el in range(12)]
    for ind in measurement_ind_list:
        pulse_error, pulse_error_ste = do_measurement(
            cxn,
            nv_sig,
            num_runs,
            num_reps,
            ind,
            pi_y_ph,
            pi_x_ph,
            pi_2_y_ph,
            pi_dt,
            pi_2_dt,
            state,
            do_dq,
            plot,
        )
        s_list.append(pulse_error)
        ste_list.append(pulse_error_ste)

    print(s_list)
    print(ste_list)
    pulse_errors = solve_errors(s_list)
    pulse_ste = calc_pulse_error_ste(ste_list)

    (
        phi_p,
        chi_p,
        phi,
        chi,
        v_z,
        e_z,
        e_y_p,
        e_z_p,
        v_x_p,
        v_z_p,
        e_y,
        v_x,
    ) = pulse_errors
    (
        phi_p_ste,
        chi_p_ste,
        phi_ste,
        chi_ste,
        v_z_ste,
        e_z_ste,
        e_y_p_ste,
        e_z_p_ste,
        v_x_p_ste,
        v_z_p_ste,
        e_y_ste,
        v_x_ste,
    ) = pulse_ste

    timestamp = tool_belt.get_time_stamp()
    raw_data = {
        "timestamp": timestamp,
        "nv_sig": nv_sig,
        "num_runs": num_runs,
        "num_reps": num_reps,
        "state": state.name,
        "phi": phi,
        "phi_ste": phi_ste,
        "e_y": e_y,
        "e_y_ste": e_y_ste,
        "e_z": e_z,
        "e_z_ste": e_z_ste,
        "chi": chi,
        "chi_ste": chi_ste,
        "v_x": v_x,
        "v_x_ste": v_x_ste,
        "v_z": v_z,
        "v_z_ste": v_z_ste,
        "phi_p": phi_p,
        "phi_p_ste": phi_p_ste,
        "e_y_p": e_y_p,
        "e_y_p_ste": 0,
        "e_z_p": e_z_p,
        "e_z_p_ste": e_z_p_ste,
        "chi_p": chi_p,
        "chi_p_ste": chi_p_ste,
        "v_x_p": v_x_p,
        "v_x_p_ste": v_x_p_ste,
        "v_z_p": v_z_p,
        "v_z_p_ste": v_z_p_ste,
        "original signal list": pulse_errors,
        "original signal ste list": pulse_ste,
    }
    nv_name = nv_sig["name"]
    file_path = tool_belt.get_file_path(__file__, timestamp, nv_name)
    tool_belt.save_raw_data(raw_data, file_path)

    print("")
    print("************RESULTS************")
    print("Pi_X pulse-------")
    print(
        "Rotation angle error,             Phi = {:.5f} +/- {:.5f} rad".format(
            phi, phi_ste
        )
    )
    print(
        "Rotation axis error along y-axis, e_y = {:.5f} +/- {:.5f} rad".format(
            e_y, e_y_ste
        )
    )
    print(
        "Rotation axis error along z-axis, e_z = {:.5f} +/- {:.5f} rad".format(
            e_z, e_z_ste
        )
    )
    print("Pi_Y pulse-------")
    print(
        "Rotation angle error,             Chi = {:.5f} +/- {:.5f} rad".format(
            chi, chi_ste
        )
    )
    print(
        "Rotation axis error along x-axis, v_x = {:.5f} +/- {:.5f} rad".format(
            v_x, v_x_ste
        )
    )
    print(
        "Rotation axis error along z-axis, v_z = {:.5f} +/- {:.5f} rad".format(
            v_z, v_z_ste
        )
    )
    print("")
    print("Pi/2_X pulse-------")
    print(
        "Rotation angle error,             Phi' = {:.5f} +/- {:.5f} rad".format(
            phi_p, phi_p_ste
        )
    )
    print(
        "Rotation axis error along y-axis, e_y' = {:.5f} rad (intentionally set to 0)".format(
            e_y_p
        )
    )
    print(
        "Rotation axis error along z-axis, e_z' = {:.5f} +/- {:.5f} rad".format(
            e_z_p, e_z_p_ste
        )
    )
    print("Pi/2_Y pulse-------")
    print(
        "Rotation angle error,             Chi' = {:.5f} +/- {:.5f} rad".format(
            chi_p, chi_p_ste
        )
    )
    print(
        "Rotation axis error along x-axis, v_x' = {:.5f} +/- {:.5f} rad".format(
            v_x_p, v_x_p_ste
        )
    )
    print(
        "Rotation axis error along z-axis, v_z' = {:.5f} +/- {:.5f} rad".format(
            v_z_p, v_z_p_ste
        )
    )
    print("*******************************")


def do_impose_phase(
    cxn,
    nv_sig,
    num_runs,
    num_reps,
    inter_composite_time,
    imposed_parameter="pi_y",
    state=States.HIGH,
    do_dq=False,
):
    """
    To test that the measurements are working, we can recreate Fig 1 from
    https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.105.077601
    where an intential phase is applied to the pi/2_y pulse, and we
    see that imtentional phase in the calculated pulse errors.

    This just measured the last 6 measurements

    """
    # phi_list = []
    # chi_list = []
    # phi_p_list = []
    # chi_p_list = []

    # v_z_list = []
    # e_z_list = []

    e_y_p_list = []
    e_z_p_list = []
    v_x_p_list = []
    v_z_p_list = []
    e_y_list = []
    v_x_list = []

    e_y_p_err_list = []
    e_z_p_err_list = []
    v_x_p_err_list = []
    v_z_p_err_list = []
    e_y_err_list = []
    v_x_err_list = []

    pi_y_ph = 0
    pi_x_ph = 0
    pi_2_y_ph = 0
    pi_dt = 0
    pi_2_dt = 0

    phases = numpy.linspace(-30, 30, 5)
    shuffle(phases)
    for p in phases:
        phase_rad = p * pi / 180

        if imposed_parameter == "pi_y":
            pi_y_ph = phase_rad
            axis_title = "Imposed phase on pi_y pulse (deg)"
        elif imposed_parameter == "pi_x":
            pi_x_ph = phase_rad
            axis_title = "Imposed phase on pi_x pulse (deg)"
        elif imposed_parameter == "pi_2_y":
            pi_2_y_ph = phase_rad
            axis_title = "Imposed phase on pi/2_y pulse (deg)"

        s_list = [0] * 6
        ste_list = [0] * 6
        # measurement_ind_list = [el for el in range(12)]
        measurement_ind_list = [6, 7, 8, 9, 10, 11]
        for ind in measurement_ind_list:
            pulse_error, pulse_error_ste = do_measurement(
                cxn,
                nv_sig,
                num_runs,
                num_reps,
                ind,
                pi_y_ph,
                pi_x_ph,
                pi_2_y_ph,
                pi_dt,
                pi_2_dt,
                state,
                do_dq,
                plot=False,
                inter_composite_time=inter_composite_time,
            )
            s_list.append(pulse_error)
            ste_list.append(pulse_error_ste)

        errs = solve_errors(s_list)
        errs_ste = calc_pulse_error_ste(ste_list)

        e_y_p_list.append(errs[6])
        e_z_p_list.append(errs[7])
        v_x_p_list.append(errs[8])
        v_z_p_list.append(errs[9])
        e_y_list.append(errs[10])
        v_x_list.append(errs[11])

        e_y_p_err_list.append(errs_ste[6])
        e_z_p_err_list.append(errs_ste[7])
        v_x_p_err_list.append(errs_ste[8])
        v_z_p_err_list.append(errs_ste[9])
        e_y_err_list.append(errs_ste[10])
        v_x_err_list.append(errs_ste[11])

    fig = plot_errors_vs_changed_phase(
        phases,
        axis_title,
        e_z_p_list,
        v_x_p_list,
        v_z_p_list,
        e_y_list,
        v_x_list,
        e_z_p_err_list,
        v_x_p_err_list,
        v_z_p_err_list,
        e_y_err_list,
        v_x_err_list,
        do_expected_phases=True,
    )

    timestamp = tool_belt.get_time_stamp()

    raw_data = {
        "timestamp": timestamp,
        "nv_sig": nv_sig,
        "phases": phases.tolist(),
        "phases-units": "degrees",
        "title": axis_title,
        "inter_composite_time": inter_composite_time,
        "e_y_p_list": e_y_p_list,
        "e_z_p_list": e_z_p_list,
        "v_x_p_list": v_x_p_list,
        "v_z_p_list": v_z_p_list,
        "e_y_list": e_y_list,
        "v_x_list": v_x_list,
        "e_y_p_err_list": e_y_p_err_list,
        "e_z_p_err_list": e_z_p_err_list,
        "v_x_p_err_list": v_x_p_err_list,
        "v_z_p_err_list": v_z_p_err_list,
        "e_y_err_list": e_y_err_list,
        "v_x_err_list": v_x_err_list,
    }

    nv_name = nv_sig["name"]
    file_path = tool_belt.get_file_path(__file__, timestamp, nv_name)
    tool_belt.save_raw_data(raw_data, file_path)
    tool_belt.save_figure(fig, file_path)


def do_impose_pi_2_dur(
    cxn,
    nv_sig,
    num_runs,
    num_reps,
    state=States.HIGH,
    do_dq=False,
):
    """
    To test that the measurements are working, we can recreate Fig 1 from
    https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.105.077601
    where an intential phase is applied to the pi/2_y pulse, and we
    see that imtentional phase in the calculated pulse errors.

    This just measured the last 6 measurements

    """
    phi_p_list = []
    chi_p_list = []

    phi_p_err_list = []
    chi_p_err_list = []

    pi_y_ph = 0
    pi_x_ph = 0
    pi_2_y_ph = 0
    pi_dt = 0
    # pi_2_dt = 0

    dt_array = numpy.linspace(-15, 15, 7)
    shuffle(dt_array)
    for dt in dt_array:
        s_list = []
        ste_list = []
        measurement_ind_list = [0, 1]
        for ind in measurement_ind_list:
            pulse_error, pulse_error_ste = do_measurement(
                cxn,
                nv_sig,
                num_runs,
                num_reps,
                ind,
                pi_y_ph,
                pi_x_ph,
                pi_2_y_ph,
                pi_dt,
                dt,
                state,
                do_dq,
                plot=False,
            )
            s_list.append(pulse_error)
            ste_list.append(pulse_error_ste)
        s_list = s_list + [0] * 10
        ste_list = ste_list + [0] * 10
        errs = solve_errors(s_list)
        errs_ste = calc_pulse_error_ste(ste_list)

        phi_p_list.append(errs[0])
        chi_p_list.append(errs[1])

        phi_p_err_list.append(errs_ste[0])
        chi_p_err_list.append(errs_ste[1])

    fig_phi_p = plot_errors_vs_changed_duration(
        dt_array + nv_sig["pi_on_2_pulse_X_{}".format(state.name)],
        "pi/2_x Phi'",
        phi_p_list,
        phi_p_err_list,
    )

    fig_chi_p = plot_errors_vs_changed_duration(
        dt_array + nv_sig["pi_on_2_pulse_Y_{}".format(state.name)],
        "pi/2_y Chi'",
        chi_p_list,
        chi_p_err_list,
    )

    timestamp = tool_belt.get_time_stamp()

    raw_data = {
        "timestamp": timestamp,
        "nv_sig": nv_sig,
        "dt_array": dt_array.tolist(),
        "dt_array-units": "ns",
        "title": "Imposed change in duration on pi/2 pulse (deg)",
        "phi_p_list": phi_p_list,
        "chi_p_list": chi_p_list,
        "phi_p_err_list": phi_p_err_list,
        "chi_p_err_list": chi_p_err_list,
    }

    nv_name = nv_sig["name"]
    file_path = tool_belt.get_file_path(__file__, timestamp, nv_name)
    tool_belt.save_raw_data(raw_data, file_path)
    tool_belt.save_figure(fig_phi_p, file_path)
    time.sleep(2)
    timestamp = tool_belt.get_time_stamp()
    file_path = tool_belt.get_file_path(__file__, timestamp, nv_name)
    tool_belt.save_figure(fig_chi_p, file_path)


def do_impose_pi_dur(
    cxn,
    nv_sig,
    num_runs,
    num_reps,
    state=States.HIGH,
    do_dq=False,
):
    """
    To test that the measurements are working, we can recreate Fig 1 from
    https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.105.077601
    where an intential phase is applied to the pi/2_y pulse, and we
    see that imtentional phase in the calculated pulse errors.


    """
    phi_p_list = []
    chi_p_list = []
    phi_list = []
    chi_list = []

    phi_err_list = []
    chi_err_list = []

    pi_y_ph = 0
    pi_x_ph = 0
    pi_2_y_ph = 0
    # pi_dt = 0
    pi_2_dt = 0

    dt_array = numpy.linspace(-20, 20, 5)
    shuffle(dt_array)
    for dt in dt_array:
        s_list = []
        ste_list = []
        measurement_ind_list = [0, 1, 2, 3]
        for ind in measurement_ind_list:
            pulse_error, pulse_error_ste = do_measurement(
                cxn,
                nv_sig,
                num_runs,
                num_reps,
                ind,
                pi_y_ph,
                pi_x_ph,
                pi_2_y_ph,
                dt,
                pi_2_dt,
                state,
                do_dq,
                plot=False,
            )
            s_list.append(pulse_error)
            ste_list.append(pulse_error_ste)
        s_list = s_list + [0] * 8
        ste_list = ste_list + [0] * 8
        errs = solve_errors(s_list)
        errs_ste = calc_pulse_error_ste(ste_list)

        phi_p_list.append(errs[0])
        chi_p_list.append(errs[1])
        phi_list.append(errs[2])
        chi_list.append(errs[3])

        phi_err_list.append(errs_ste[2])
        chi_err_list.append(errs_ste[3])

    print(phi_list)
    print(chi_list)
    fig_phi_p = plot_errors_vs_changed_duration(
        dt_array + nv_sig["pi_pulse_X_{}".format(state.name)],
        "pi_x Phi",
        phi_list,
        phi_err_list,
    )

    fig_chi_p = plot_errors_vs_changed_duration(
        dt_array + nv_sig["pi_pulse_Y_{}".format(state.name)],
        "pi_y Chi",
        chi_list,
        chi_err_list,
    )

    timestamp = tool_belt.get_time_stamp()

    raw_data = {
        "timestamp": timestamp,
        "nv_sig": nv_sig,
        "dt_array": dt_array.tolist(),
        "dt_array-units": "ns",
        "title": "Imposed change in duration on pi pulse (deg)",
        "phi_list": phi_list,
        "chi_list": chi_list,
        "phi_err_list": phi_err_list,
        "chi_err_list": chi_err_list,
    }

    nv_name = nv_sig["name"]
    file_path = tool_belt.get_file_path(__file__, timestamp, nv_name)
    tool_belt.save_raw_data(raw_data, file_path)
    tool_belt.save_figure(fig_phi_p, file_path)
    time.sleep(2)
    timestamp = tool_belt.get_time_stamp()
    file_path = tool_belt.get_file_path(__file__, timestamp, nv_name)
    tool_belt.save_figure(fig_chi_p, file_path)


def cosine_fit(x, offset, amp, freq, phase):
    return offset + amp * numpy.cos(x * freq + phase)


def do_two_pi_2_pulse_vary_phase(
    cxn,
    nv_sig,
    num_runs,
    num_reps,
    state=States.HIGH,
    do_dq=False,
):
    kpl.init_kplotlib()

    phases = numpy.linspace(0, 360, 16)  # 16)
    s_list = []
    s_err_list = []
    for p in phases:
        p_rad = p * pi / 180
        pulse_error, pulse_error_ste = do_measurement(
            cxn,
            nv_sig,
            num_runs,
            num_reps,
            6,
            pi_y_ph=0,
            pi_x_ph=0,
            pi_2_y_ph=p_rad,
            pi_dt=0,
            pi_2_dt=0,
            state=States.HIGH,
            do_dq=do_dq,
            plot=False,
        )
        s_list.append(pulse_error)
        s_err_list.append(pulse_error_ste)
    # print(s_list)

    x_smooth = numpy.linspace(phases[0], phases[-1], 1000)

    fit_func = lambda x, offset, amp, phase: cosine_fit(
        x, offset, amp, numpy.pi / 180, phase
    )
    init_params = [0, 1, -1]
    popt, pcov = curve_fit(
        fit_func,
        phases,
        s_list,
        sigma=s_err_list,
        absolute_sigma=True,
        p0=init_params,
    )
    print(popt)

    # Plot setup
    fig, ax = plt.subplots(1, 1)
    ax.set_xlabel("Phase (deg)")
    ax.set_ylabel("Error")
    # ax.set_title(title)

    # Plotting
    kpl.plot_points(
        ax, phases, s_list, yerr=s_err_list, label="data", color=KplColors.BLACK
    )

    kpl.plot_line(
        ax, x_smooth, fit_func(x_smooth, *popt), label="fit", color=KplColors.RED
    )

    ax.legend()

    timestamp = tool_belt.get_time_stamp()

    raw_data = {
        "timestamp": timestamp,
        "nv_sig": nv_sig,
        "phases": phases.tolist(),
        "phases-units": "deg",
        "s_list": s_list,
        "s_err_list": s_err_list,
    }

    nv_name = nv_sig["name"]
    file_path = tool_belt.get_file_path(__file__, timestamp, nv_name)
    tool_belt.save_raw_data(raw_data, file_path)
    tool_belt.save_figure(fig, file_path)


def vary_pi_2_y_pulse(cxn, nv_sig, num_runs, num_reps, state=States.HIGH, do_dq=False):
    # just measured the pi/2_2 pulse with varying phase
    kpl.init_kplotlib()

    phases = numpy.linspace(-30, 30, 5)  # 16)

    # t_list = numpy.linspace(0, 260, 14)

    v_x_p_list = []
    v_x_p_err_list = []
    s6_list = []
    s7_list = []

    for p in phases:
        # for t in t_list:
        # print(p)
        p_rad = p * pi / 180
        # p_rad = 30 * pi/180
        S6, S6_err = do_measurement(
            cxn,
            nv_sig,
            num_runs,
            num_reps,
            6,
            pi_y_ph=0,
            pi_x_ph=0,
            pi_2_y_ph=p_rad,
            pi_dt=0,
            pi_2_dt=0,
            state=States.HIGH,
            # inter_pulse_time = t,
            do_dq=do_dq,
            plot=False,
        )

        S7, S7_err = do_measurement(
            cxn,
            nv_sig,
            num_runs,
            num_reps,
            7,
            pi_y_ph=0,
            pi_x_ph=0,
            pi_2_y_ph=p_rad,
            pi_dt=0,
            pi_2_dt=0,
            state=States.HIGH,
            # inter_pulse_time = t,
            do_dq=do_dq,
            plot=False,
        )
        s6_list.append(S6)
        s7_list.append(S7)
        vxp = -(S7 + S6) / 2
        vxp_err = numpy.sqrt(S6_err**2 + S7_err**2) / 2

        v_x_p_list.append(vxp)
        v_x_p_err_list.append(vxp_err)
    print(s6_list)
    print(s7_list)

    # x_smooth = numpy.linspace(phases[0], phases[-1], 1000)

    # fit_func = lambda x, offset, amp, phase: cosine_fit(x, offset, amp, numpy.pi/180, phase)
    # init_params = [ 0,1,  -1]
    # popt, pcov = curve_fit(
    #     fit_func,
    #       phases,
    #     s_list,
    #     sigma=s_err_list,
    #     absolute_sigma=True,
    #     p0=init_params,
    # )
    # print(popt)

    # # Plot setup
    fig, ax = plt.subplots(1, 1)
    ax.set_xlabel("Phase (deg)")
    # ax.set_xlabel('Timing between pulses (ns)')
    ax.set_ylabel("Error")
    # # ax.set_title(title)

    # Plotting
    kpl.plot_points(
        ax, phases, v_x_p_list, yerr=v_x_p_err_list, label="data", color=KplColors.BLACK
    )
    # kpl.plot_points(ax,  t_list, v_x_p_list, yerr = v_x_p_err_list, label = 'data', color=KplColors.BLACK)

    # kpl.plot_line(ax, x_smooth, fit_func(x_smooth,*popt ), label = 'fit', color=KplColors.RED)

    ax.legend()
    timestamp = tool_belt.get_time_stamp()

    raw_data = {
        "timestamp": timestamp,
        "nv_sig": nv_sig,
        "phases": phases.tolist(),
        "phases-units": "deg",
        "v_x_p_list": v_x_p_list,
        "v_x_p_err_list": v_x_p_err_list,
    }

    nv_name = nv_sig["name"]
    file_path = tool_belt.get_file_path(__file__, timestamp, nv_name)
    tool_belt.save_raw_data(raw_data, file_path)
    tool_belt.save_figure(fig, file_path)


def vary_time_btween_composite_pulses(
    cxn, nv_sig, start_time, end_time, num_steps, int_phase=0
):
    pulse_error_list = []
    pulse_ste_list = []
    taus = numpy.linspace(start_time, end_time, num_steps)
    phase = int_phase * pi / 180
    for t in taus:
        s6, s6_ste = do_measurement(
            cxn,
            nv_sig,
            num_runs,
            num_reps,
            6,
            pi_y_ph=0,
            pi_x_ph=0,
            pi_2_y_ph=phase,
            pi_dt=0,
            pi_2_dt=0,
            state=States.HIGH,
            do_dq=True,
            plot=False,
            inter_composite_time=t,
        )
        s7, s7_ste = do_measurement(
            cxn,
            nv_sig,
            num_runs,
            num_reps,
            7,
            pi_y_ph=0,
            pi_x_ph=0,
            pi_2_y_ph=phase,
            pi_dt=0,
            pi_2_dt=0,
            state=States.HIGH,
            do_dq=True,
            plot=False,
            inter_composite_time=t,
        )
        vxp = -(s7 + s6) / 2
        vxp_err = numpy.sqrt(s6_ste**2 + s7_ste**2) / 2
        pulse_error_list.append(vxp)
        pulse_ste_list.append(vxp_err)
    fig, ax = plt.subplots()
    ax.errorbar(taus, pulse_error_list, yerr=pulse_ste_list, fmt="ro")
    # ax.set_xlabel('Wait time between pulses (ns)')
    ax.set_xlabel(
        "Wait time between DQ pulses (ns) with {} deg phase".format(int_phase)
    )
    ax.set_ylabel("Error")
    ax.set_title("pi/2_y error")
    timestamp = tool_belt.get_time_stamp()

    raw_data = {
        "timestamp": timestamp,
        "nv_sig": nv_sig,
        "phase": phase,
        "taus": taus.tolist(),
        "taus-units": "ns",
        "pulse_error_list": pulse_error_list,
        "pulse_ste_list": pulse_ste_list,
    }

    nv_name = nv_sig["name"]
    file_path = tool_belt.get_file_path(__file__, timestamp, nv_name)
    tool_belt.save_raw_data(raw_data, file_path)
    tool_belt.save_figure(fig, file_path)


def do_vary_composite_timing(
    cxn,
    nv_sig,
    num_runs,
    num_reps,
    time_range,
    num_steps,
    state=States.HIGH,
    pi_y_ph=0,
    pi_x_ph=0,
    pi_2_y_ph=0,
    pi_dt=0,
    pi_2_dt=0,
    do_dq=True,
    plot=False,
    ret_ste=True,
):
    """
    This function runs all 12 measurements, and returns the measured signals
    Does this for various timing between composite pulses
    """
    do_dq = True
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

    phi_err_list = []
    chi_err_list = []
    phi_p_err_list = []
    chi_p_err_list = []

    v_z_err_list = []
    e_z_err_list = []

    e_y_p_err_list = []
    e_z_p_err_list = []
    v_x_p_err_list = []
    v_z_p_err_list = []
    e_y_err_list = []
    v_x_err_list = []

    t_list = numpy.linspace(time_range[0], time_range[-1], num_steps)

    measurement_ind_list = [el for el in range(12)]

    shuffle(t_list)
    for t_comp in t_list:
        s_list = []
        ste_list = []
        for ind in measurement_ind_list:
            pulse_error, pulse_error_ste = do_measurement(
                cxn,
                nv_sig,
                num_runs,
                num_reps,
                ind,
                pi_y_ph,
                pi_x_ph,
                pi_2_y_ph,
                pi_dt,
                pi_2_dt,
                state,
                do_dq,
                plot,
                inter_composite_time=t_comp,
            )
            s_list.append(pulse_error)
            ste_list.append(pulse_error_ste)

        pulse_errors = solve_errors(s_list)
        pulse_ste = calc_pulse_error_ste(ste_list)

        (
            phi_p,
            chi_p,
            phi,
            chi,
            v_z,
            e_z,
            e_y_p,
            e_z_p,
            v_x_p,
            v_z_p,
            e_y,
            v_x,
        ) = pulse_errors
        (
            phi_p_ste,
            chi_p_ste,
            phi_ste,
            chi_ste,
            v_z_ste,
            e_z_ste,
            e_y_p_ste,
            e_z_p_ste,
            v_x_p_ste,
            v_z_p_ste,
            e_y_ste,
            v_x_ste,
        ) = pulse_ste

        phi_p_list.append(phi_p)
        chi_p_list.append(chi_p)
        phi_list.append(phi)
        chi_list.append(chi)
        v_z_list.append(v_z)
        e_z_list.append(e_z)
        e_y_p_list.append(e_y_p)
        e_z_p_list.append(e_z_p)
        v_x_p_list.append(v_x_p)
        v_z_p_list.append(v_z_p)
        e_y_list.append(e_y)
        v_x_list.append(v_x)

        phi_p_err_list.append(phi_p_ste)
        chi_p_err_list.append(chi_p_ste)
        phi_err_list.append(phi_ste)
        chi_err_list.append(chi_ste)
        v_z_err_list.append(v_z_ste)
        e_z_err_list.append(e_z_ste)
        e_y_p_err_list.append(e_y_p_ste)
        e_z_p_err_list.append(e_z_p_ste)
        v_x_p_err_list.append(v_x_p_ste)
        v_z_p_err_list.append(v_z_p_ste)
        e_y_err_list.append(e_y_ste)
        v_x_err_list.append(v_x_ste)

    fig_pix, ax = plt.subplots()
    ax.errorbar(
        t_list, phi_list, yerr=phi_err_list, fmt="ro", label="Rotation angle error"
    )
    ax.errorbar(
        t_list, e_y_list, yerr=e_y_err_list, fmt="bo", label="Error along Y axis"
    )
    ax.errorbar(
        t_list, e_z_list, yerr=e_z_err_list, fmt="go", label="Error along Z axis"
    )
    ax.set_xlabel("Time between composite pulses (ns)")
    ax.set_ylabel("Error")
    ax.set_title("Pi_x pulse")
    ax.legend()

    fig_piy, ax = plt.subplots()
    ax.errorbar(
        t_list, chi_list, yerr=chi_err_list, fmt="ro", label="Rotation angle error"
    )
    ax.errorbar(
        t_list, v_x_list, yerr=v_x_err_list, fmt="bo", label="Error along X axis"
    )
    ax.errorbar(
        t_list, v_z_list, yerr=v_z_err_list, fmt="go", label="Error along Z axis"
    )
    ax.set_xlabel("Time between composite pulses (ns)")
    ax.set_ylabel("Error")
    ax.set_title("Pi_y pulse")
    ax.legend()

    fig_pi2x, ax = plt.subplots()
    ax.errorbar(
        t_list, phi_p_list, yerr=phi_p_err_list, fmt="ro", label="Rotation angle error"
    )
    # ax.errorbar(t_list,e_y_p_list, yerr = e_y_p_err_list, fmt= 'bo', label = 'Error along Y axis'  )
    ax.errorbar(
        t_list, e_z_p_list, yerr=e_z_p_err_list, fmt="go", label="Error along Z axis"
    )
    ax.set_xlabel("Time between composite pulses (ns)")
    ax.set_ylabel("Error")
    ax.set_title("Pi/2_x pulse")
    ax.legend()

    fig_pi2y, ax = plt.subplots()
    ax.errorbar(
        t_list, chi_p_list, yerr=chi_p_err_list, fmt="ro", label="Rotation angle error"
    )
    ax.errorbar(
        t_list, v_x_p_list, yerr=v_x_p_err_list, fmt="bo", label="Error along X axis"
    )
    ax.errorbar(
        t_list, v_z_p_list, yerr=v_z_p_err_list, fmt="go", label="Error along Z axis"
    )
    ax.set_xlabel("Time between composite pulses (ns)")
    ax.set_ylabel("Error")
    ax.set_title("Pi/2_y pulse")
    ax.legend()

    timestamp = tool_belt.get_time_stamp()
    raw_data = {
        "timestamp": timestamp,
        "nv_sig": nv_sig,
        "num_runs": num_runs,
        "num_reps": num_reps,
        "time_range": time_range,
        "num_steps": num_steps,
        "state": state.name,
        "t_list": t_list.tolist(),
        "phi_list": phi_list,
        "phi_err_list": phi_err_list,
        "e_y_list": e_y_list,
        "e_y_err_list": e_y_err_list,
        "e_z_list": e_z_list,
        "e_z_err_list": e_z_err_list,
        "chi_list": chi_list,
        "chi_err_list": chi_err_list,
        "v_x_list": v_x_list,
        "v_x_err_list": v_x_err_list,
        "v_z_list": v_z_list,
        "v_z_err_list": v_z_err_list,
        "phi_p_list": phi_p_list,
        "phi_p_err_list": phi_p_err_list,
        "e_y_p_list": e_y_p_list,
        "e_y_p_err_list": e_y_p_err_list,
        "e_z_p_list": e_z_p_list,
        "e_z_p_err_list": e_z_p_err_list,
        "chi_p_list": chi_p_list,
        "chi_p_err_list": chi_p_err_list,
        "v_x_p_list": v_x_p_list,
        "v_x_p_err_list": v_x_p_err_list,
        "v_z_p_list": v_z_p_list,
        "v_z_p_err_list": v_z_p_err_list,
    }
    nv_name = nv_sig["name"]
    file_path = tool_belt.get_file_path(__file__, timestamp, nv_name)
    tool_belt.save_raw_data(raw_data, file_path)
    tool_belt.save_figure(fig_pix, file_path)
    time.sleep(2)
    timestamp = tool_belt.get_time_stamp()
    file_path = tool_belt.get_file_path(__file__, timestamp, nv_name)
    tool_belt.save_figure(fig_piy, file_path)
    time.sleep(2)
    timestamp = tool_belt.get_time_stamp()
    file_path = tool_belt.get_file_path(__file__, timestamp, nv_name)
    tool_belt.save_figure(fig_pi2x, file_path)
    time.sleep(2)
    timestamp = tool_belt.get_time_stamp()
    file_path = tool_belt.get_file_path(__file__, timestamp, nv_name)
    tool_belt.save_figure(fig_pi2y, file_path)


# %%
if __name__ == "__main__":
    sample_name = "siena"
    green_power = 8000
    nd_green = "nd_1.1"
    green_laser = "integrated_520"
    yellow_laser = "laserglow_589"
    red_laser = "cobolt_638"

    nv_sig = {
        "coords": [0.04, -0.123, 7.6],
        "name": "{}-nv0_2023_03_20".format(
            sample_name,
        ),
        "disable_opt": False,
        "ramp_voltages": False,
        "expected_count_rate": 13,
        "spin_laser": green_laser,
        "spin_laser_power": green_power,
        "spin_laser_filter": nd_green,
        "spin_readout_dur": 480,
        "spin_pol_dur": 1000.0,
        "imaging_laser": green_laser,
        "imaging_laser_power": green_power,
        "imaging_laser_filter": nd_green,
        "imaging_readout_dur": 1e7,
        "charge_readout_laser": yellow_laser,
        "charge_readout_laser_filter": "nd_0",
        "collection_filter": "715_sp+630_lp",  # NV band only
        "magnet_angle": 163,
        "resonance_LOW": 2.82309,
        # "rabi_LOW":110*2,
        "uwave_power_LOW": -2,
        "resonance_HIGH": 2.91872,
        # "rabi_HIGH":110*2,
        "uwave_power_HIGH": -2,
        # DQ
        "pi_pulse_X_LOW": 39.33,
        "pi_on_2_pulse_X_LOW": 20.94,
        "pi_pulse_Y_LOW": 39.33,
        "pi_on_2_pulse_Y_LOW": 20.94,
        "pi_pulse_X_HIGH": 38,
        "pi_on_2_pulse_X_HIGH": 23.23,
        "pi_pulse_Y_HIGH": 38,
        "pi_on_2_pulse_Y_HIGH": 23.23,
    }

    import majorroutines.image_sample as image_sample

    with labrad.connect() as cxn:
        num_runs = 10
        num_reps = int(5e4)

        # for t in numpy.linspace(40,120, 9):
        #   inter_composite_time=t
        #   do_impose_phase(cxn, nv_sig, num_runs, num_reps,inter_composite_time, imposed_parameter = 'pi_2_y',  do_dq = True)
        # do_impose_phase(cxn, nv_sig, num_runs, num_reps,inter_composite_time, imposed_parameter = 'pi_y', do_dq = True)
        # do_impose_phase(cxn, nv_sig, num_runs, num_reps,inter_composite_time, imposed_parameter = 'pi_x', do_dq = True)

        scan_range = 0.1
        num_steps = 35
        # image_sample.main(nv_sig, scan_range, scan_range, num_steps)
        # vary_time_btween_composite_pulses(cxn, nv_sig, 0, 300, 41, int_phase = 0)
        #  image_sample.main(nv_sig, scan_range, scan_range, num_steps)
        #  vary_time_btween_composite_pulses(cxn, nv_sig, 0, 300, 41, int_phase = 30)
        #  image_sample.main(nv_sig, scan_range, scan_range, num_steps)
        #  vary_time_btween_composite_pulses(cxn, nv_sig, 0, 300, 41, int_phase = -30)
        #  image_sample.main(nv_sig, scan_range, scan_range, num_steps)
        # vary_time_btween_composite_pulses(cxn, nv_sig, 400, 500, 101, int_phase = 0)
        # image_sample.main(nv_sig, scan_range, scan_range, num_steps)
        # vary_time_btween_composite_pulses(cxn, nv_sig, 400, 500, 101, int_phase = -30)
        # image_sample.main(nv_sig, scan_range, scan_range, num_steps)
        # vary_time_btween_composite_pulses(cxn, nv_sig, 400, 500, 101, int_phase = 30)
        # image_sample.main(nv_sig, scan_range, scan_range, num_steps)
        # vary_time_btween_composite_pulses(cxn, nv_sig, 800, 1000, 41, int_phase = 0)
        # image_sample.main(nv_sig, scan_range, scan_range, num_steps)
        # vary_time_btween_composite_pulses(cxn, nv_sig, 800, 1000, 41, int_phase = 30)
        # image_sample.main(nv_sig, scan_range, scan_range, num_steps)
        # vary_time_btween_composite_pulses(cxn, nv_sig, 800, 1000, 41, int_phase = -30)
        # image_sample.main(nv_sig, scan_range, scan_range, num_steps)

        if False:
            s6, s6_ste = do_measurement(
                cxn,
                nv_sig,
                num_runs,
                num_reps,
                0,
                pi_y_ph=0,
                pi_x_ph=0,
                pi_2_y_ph=0,
                pi_dt=0,
                pi_2_dt=0,
                state=States.HIGH,
                do_dq=False,
                plot=True,
            )
            print(s6)
            print(s6_ste)

        if False:
            do_vary_composite_timing(cxn, nv_sig, num_runs, num_reps, [0, 500], 51)

        ### measure the phase errors, will print them out
        # measure_pulse_errors(cxn, nv_sig, num_runs,num_reps, pi_y_ph = 0, pi_x_ph =0,pi_2_y_ph = 0,state=States.HIGH, do_dq = False)

        ### Run a test by intentionally adding phase to pi_y pulses and
        ### see that in the extracted pulse errors
        do_impose_phase(
            cxn, nv_sig, num_runs, num_reps, 0, imposed_parameter="pi_2_y", do_dq=False
        )
        # do_impose_phase(cxn, nv_sig, num_runs, num_reps,0, imposed_parameter = 'pi_y', do_dq = False)
        # do_impose_phase(cxn, nv_sig, num_runs, num_reps, 0,imposed_parameter = 'pi_x', do_dq = False)

        # measure_pulse_errors(cxn, nv_sig, num_runs,num_reps, pi_y_ph = 0, pi_x_ph =0,pi_2_y_ph = 0,state=States.HIGH, do_dq = False)

        # 114, 228, 454, 285, 0

        # do_impose_phase(cxn, nv_sig, num_runs, num_reps, 194,  imposed_parameter = 'pi_2_y',  do_dq = True)
        # do_impose_phase(cxn, nv_sig, num_runs, num_reps,0,  imposed_parameter = 'pi_2_y',  do_dq = True)
        #  do_impose_phase(cxn, nv_sig, num_runs, num_reps, imposed_parameter = 'pi_y', do_dq = True)
        #   do_impose_phase(cxn, nv_sig, num_runs, num_reps, imposed_parameter = 'pi_x', do_dq = True)

        # do_impose_pi_2_dur(cxn, nv_sig,num_runs, num_reps, state=States.HIGH, do_dq = False,)

        # do_impose_pi_dur(cxn,  nv_sig, num_runs,num_reps, state=States.HIGH,do_dq = False,)

        # file ='2023_02_13-15_23_19-siena-nv4_2023_01_16'
        # replot_imposed_phases(file)

        # do_two_pi_2_pulse_vary_phase(cxn,  nv_sig, num_runs,num_reps)

        # vary_pi_2_y_pulse(cxn,  nv_sig, num_runs,num_reps, do_dq = False)

        if False:
            phases = numpy.linspace(0, 360, 19)
            s_list = []
            s_err_list = []
            for p in phases:
                p_rad = p * pi / 180
                pulse_error, pulse_error_ste = do_measurement(
                    cxn,
                    nv_sig,
                    num_runs,
                    num_reps,
                    101,
                    pi_y_ph=0,
                    pi_x_ph=p_rad,
                    pi_2_y_ph=0,
                    pi_dt=0,
                    pi_2_dt=0,
                    do_dq=True,
                )
                s_list.append(pulse_error)
                s_err_list.append(pulse_error_ste)

            fig, ax = plt.subplots()
            ax.errorbar(phases, s_list, yerr=s_err_list, fmt="ko")
            ax.set_xlabel("Change in phase on pi pulse (deg)")
            ax.set_ylabel("Error")
            ax.set_title("pi/2_x - pi_p - pi/2_x")

            timestamp = tool_belt.get_time_stamp()

            raw_data = {
                "timestamp": timestamp,
                "nv_sig": nv_sig,
                "phases": phases.tolist(),
                "phases-units": "deg",
                "s_list": s_list,
                "s_err_list": s_err_list,
            }

            nv_name = nv_sig["name"]
            file_path = tool_belt.get_file_path(__file__, timestamp, nv_name)
            tool_belt.save_raw_data(raw_data, file_path)
            tool_belt.save_figure(fig, file_path)
