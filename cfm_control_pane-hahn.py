# -*- coding: utf-8 -*-
"""This file contains functions to control the CFM. Just change the function call
in the main section at the bottom of this file and run the file. Shared or
frequently changed parameters are in the __main__ body and relatively static
parameters are in the function definitions.

Created on November 25th, 2018

@author: mccambria
"""


# region Imports and constants


import labrad
import numpy as np
import time
import copy
import utils.tool_belt as tool_belt
import majorroutines.image_sample as image_sample
import majorroutines.optimize as optimize
import majorroutines.stationary_count as stationary_count
import majorroutines.resonance as resonance
import majorroutines.pulsed_resonance as pulsed_resonance
import majorroutines.optimize_magnet_angle as optimize_magnet_angle
import majorroutines.rabi as rabi
import majorroutines.g2_measurement as g2_measurement
import majorroutines.t1_dq_main as t1_dq_main
import majorroutines.ramsey as ramsey
import majorroutines.spin_echo as spin_echo
import chargeroutines.determine_charge_readout_params as determine_charge_readout_params
import minorroutines.determine_standard_readout_params as determine_standard_readout_params
import chargeroutines.scc_pulsed_resonance as scc_pulsed_resonance
from utils.tool_belt import States
import time


# endregion
# region Routines


def do_image_sample(
    nv_sig,
    nv_minus_initialization=False,
    cbarmin=None,
    cbarmax=None,
):

    # scan_range = 0.2
    # num_steps = 60

    scan_range = 0.5
    num_steps = 90

    # For now we only support square scans so pass scan_range twice
    image_sample.main(
        nv_sig,
        scan_range,
        scan_range,
        num_steps,
        nv_minus_initialization=nv_minus_initialization,
        cmin=cbarmin,
        cmax=cbarmax,
    )


def do_image_sample_zoom(nv_sig):

    scan_range = 0.05
    num_steps = 30

    image_sample.main(
        nv_sig,
        scan_range,
        scan_range,
        num_steps,
    )


def do_optimize(nv_sig):

    optimize.main(
        nv_sig,
        set_to_opti_coords=False,
        save_data=True,
        plot_data=True,
    )


def do_stationary_count(
    nv_sig,
    disable_opt=None,
    nv_minus_initialization=False,
    nv_zero_initialization=False,
):

    run_time = 3 * 60 * 10**9  # ns

    stationary_count.main(
        nv_sig,
        run_time,
        disable_opt=disable_opt,
        nv_minus_initialization=nv_minus_initialization,
        nv_zero_initialization=nv_zero_initialization,
    )


def do_g2_measurement(nv_sig, apd_a_index, apd_b_index):

    run_time = 60 * 10  # s
    diff_window = 200  # ns

    g2_measurement.main(nv_sig, run_time, diff_window, apd_a_index, apd_b_index)


def do_resonance(nv_sig, freq_center=2.87, freq_range=0.2):

    num_steps = 51
    num_runs = 20
    uwave_power = -5.0

    resonance.main(
        nv_sig,
        freq_center,
        freq_range,
        num_steps,
        num_runs,
        uwave_power,
        state=States.HIGH,
    )


def do_resonance_state(nv_sig, state):

    freq_center = nv_sig["resonance_{}".format(state.name)]
    uwave_power = -5.0

    # freq_range = 0.200
    # num_steps = 51
    # num_runs = 2

    # Zoom
    freq_range = 0.05
    num_steps = 51
    num_runs = 10

    resonance.main(
        nv_sig,
        freq_center,
        freq_range,
        num_steps,
        num_runs,
        uwave_power,
    )


def do_determine_standard_readout_params(nv_sig):

    num_reps = 1e5
    max_readouts = [1e6]
    filters = ["nd_0"]
    state = States.LOW

    determine_standard_readout_params.main(
        nv_sig,
        num_reps,
        max_readouts,
        filters=filters,
        state=state,
    )


def do_pulsed_resonance(nv_sig, freq_center=2.87, freq_range=0.2):

    num_steps = 51

    num_reps = 2e4
    num_runs = 16

    # num_reps = 1e3
    # num_runs = 8

    uwave_power = 16.5
    uwave_pulse_dur = 400

    pulsed_resonance.main(
        nv_sig,
        freq_center,
        freq_range,
        num_steps,
        num_reps,
        num_runs,
        uwave_power,
        uwave_pulse_dur,
    )


def do_pulsed_resonance_state(nv_sig, state):

    freq_range = 0.020
    num_steps = 51
    num_reps = 2e4
    num_runs = 16

    # Zoom
    # freq_range = 0.035
    # # freq_range = 0.120
    # num_steps = 51
    # num_reps = 8000
    # num_runs = 3

    composite = False

    res, _ = pulsed_resonance.state(
        nv_sig,
        state,
        freq_range,
        num_steps,
        num_reps,
        num_runs,
        composite,
    )
    nv_sig["resonance_{}".format(state.name)] = res
    return res


def do_scc_pulsed_resonance(nv_sig, state):

    opti_nv_sig = nv_sig
    freq_center = nv_sig["resonance_{}".format(state)]
    uwave_power = nv_sig["uwave_power_{}".format(state)]
    uwave_pulse_dur = tool_belt.get_pi_pulse_dur(nv_sig["rabi_{}".format(state)])
    freq_range = 0.020
    num_steps = 25
    num_reps = int(1e3)
    num_runs = 5

    scc_pulsed_resonance.main(
        nv_sig,
        opti_nv_sig,
        freq_center,
        freq_range,
        num_steps,
        num_reps,
        num_runs,
        uwave_power,
        uwave_pulse_dur,
    )


def do_determine_charge_readout_params(nv_sig):

    readout_durs = [10e6]
    readout_durs = [int(el) for el in readout_durs]
    max_readout_dur = max(readout_durs)

    readout_powers = [1.0]
    readout_powers = [round(val, 3) for val in readout_powers]

    num_reps = 1000

    determine_charge_readout_params.main(
        nv_sig,
        num_reps,
        readout_powers,
        max_readout_dur,
        plot_readout_durs=readout_durs,
    )


def do_optimize_magnet_angle(nv_sig):

    angle_range = [0, 150]
    num_angle_steps = 6
    freq_center = 2.87
    freq_range = 0.200
    num_freq_steps = 51
    num_freq_runs = 15

    # Pulsed
    uwave_power = 16.5
    uwave_pulse_dur = 85
    num_freq_reps = 5000

    # CW
    # uwave_power = -5.0
    # uwave_pulse_dur = None
    # num_freq_reps = None

    optimize_magnet_angle.main(
        nv_sig,
        angle_range,
        num_angle_steps,
        freq_center,
        freq_range,
        num_freq_steps,
        num_freq_reps,
        num_freq_runs,
        uwave_power,
        uwave_pulse_dur,
    )


def do_rabi(nv_sig, state, uwave_time_range=[0, 200]):

    num_steps = 51
    num_reps = 2e4
    num_runs = 16

    period = rabi.main(
        nv_sig,
        uwave_time_range,
        state,
        num_steps,
        num_reps,
        num_runs,
    )
    nv_sig["rabi_{}".format(state.name)] = period


def do_t1_dq(nv_sig):

    # T1 experiment parameters, formatted:
    # [[init state, read state], relaxation_time_range, num_steps, num_reps]
    num_runs = 500
    num_reps = 1000
    num_steps = 12
    min_tau = 10e3
    max_tau_omega = int(18e6)
    max_tau_gamma = int(8.5e6)
    # fmt: off
    t1_exp_array = np.array(
        [[[States.ZERO, States.HIGH], [min_tau, max_tau_omega], num_steps, num_reps, num_runs],
        [[States.ZERO, States.ZERO], [min_tau, max_tau_omega], num_steps, num_reps, num_runs],
        [[States.ZERO, States.HIGH], [min_tau, max_tau_omega // 3], num_steps, num_reps, num_runs],
        [[States.ZERO, States.ZERO], [min_tau, max_tau_omega // 3], num_steps, num_reps, num_runs],
        [[States.LOW, States.HIGH], [min_tau, max_tau_gamma], num_steps, num_reps, num_runs],
        [[States.LOW, States.LOW], [min_tau, max_tau_gamma], num_steps, num_reps, num_runs],
        [[States.LOW, States.HIGH], [min_tau, max_tau_gamma // 3], num_steps, num_reps, num_runs],
        [[States.LOW, States.LOW], [min_tau, max_tau_gamma // 3], num_steps, num_reps, num_runs]],
        dtype=object,
    )
    # fmt: on

    t1_dq_main.main(nv_sig, t1_exp_array, num_runs)


def do_ramsey(nv_sig):

    detuning = 2.5  # MHz
    precession_time_range = [0, 4 * 10**3]
    num_steps = 151
    num_reps = 3 * 10**5
    num_runs = 1

    ramsey.main(
        nv_sig,
        detuning,
        precession_time_range,
        num_steps,
        num_reps,
        num_runs,
    )


def do_spin_echo(nv_sig):

    # T2* in nanodiamond NVs is just a couple us at 300 K
    # In bulk it"s more like 100 us at 300 K
    max_time = 120  # us
    num_steps = max_time  # 1 point per us
    precession_time_range = [1e3, max_time * 10**3]
    num_reps = 4e3
    num_runs = 20

    state = States.LOW

    angle = spin_echo.main(
        nv_sig,
        precession_time_range,
        num_steps,
        num_reps,
        num_runs,
        state,
    )
    return angle


# endregion


if __name__ == "__main__":

    ### Shared parameters

    green_laser = "laserglow_532"
    yellow_laser = "laserglow_589"
    red_laser = "cobolt_638"

    # fmt: off

    sample_name = "wu"
    nv_sig = {
        "coords": [0.240, -0.426, 1], "name": "{}-nv8_2022_11_14".format(sample_name),
        "disable_opt": False, "disable_z_opt": True, "expected_count_rate": 13,

        "imaging_laser": green_laser, "imaging_laser_filter": "nd_0", "imaging_readout_dur": 1e7,
        "spin_laser": green_laser, "spin_laser_filter": "nd_0", "spin_pol_dur": 2e3, "spin_readout_dur": 440,

        "nv-_reionization_laser": green_laser, "nv-_reionization_dur": 1e6, "nv-_reionization_laser_filter": "nd_1.0",
        "nv-_prep_laser": green_laser, "nv-_prep_laser_dur": 1e6, "nv-_prep_laser_filter": "nd_0",
        "nv0_ionization_laser": red_laser, "nv0_ionization_dur": 75, "nv0_prep_laser": red_laser, "nv0_prep_laser_dur": 75,
        "spin_shelf_laser": yellow_laser, "spin_shelf_dur": 0, "spin_shelf_laser_power": 1.0,
        "initialize_laser": green_laser, "initialize_dur": 1e4,
        "charge_readout_laser": yellow_laser, "charge_readout_dur": 100e6, "charge_readout_laser_power": 1.0,

        "collection_filter": None, "magnet_angle": None,
        "resonance_LOW": 2.878, "rabi_LOW": 400, "uwave_power_LOW": 16.5,
        "resonance_HIGH": 2.882, "rabi_HIGH": 400, "uwave_power_HIGH": 16.5,
        }

    # fmt: on

    ### Routines to execute

    try:

        tool_belt.init_safe_stop()

        # tool_belt.set_drift([0.0, 0.0, 0.0])  # Totally reset
        # drift = tool_belt.get_drift()
        # tool_belt.set_drift([0.0, 0.0, drift[2]])  # Keep z
        # tool_belt.set_drift([drift[0], drift[1], 0.0])  # Keep xy

        # for z in np.arange(-24, 20, 4):
        # for z in np.arange(0, -100, -5):
        # # while True:
        #     if tool_belt.safe_stop():
        #         break
        #     nv_sig["coords"][2] = int(z)
        # do_image_sample(nv_sig)
        # nv_sig["imaging_readout_dur"] = 5e7
        do_image_sample(nv_sig)
        # do_image_sample_zoom(nv_sig)
        # do_image_sample(nv_sig, nv_minus_initialization=True)
        # do_image_sample_zoom(nv_sig, nv_minus_initialization=True)

        # do_optimize(nv_sig)
        # nv_sig["imaging_readout_dur"] = 5e7
        # do_stationary_count(nv_sig, disable_opt=True)
        # do_stationary_count(nv_sig, disable_opt=True, nv_minus_initialization=True)
        # do_stationary_count(nv_sig, disable_opt=True, nv_zero_initialization=True)

        # do_resonance(nv_sig, 2.87, 0.200)
        # do_resonance_state(nv_sig , States.LOW)
        # do_resonance_state(nv_sig, States.HIGH)
        # do_pulsed_resonance(nv_sig, 2.87, 0.200)
        # do_pulsed_resonance_state(nv_sig, States.LOW)
        # do_pulsed_resonance_state(nv_sig, States.HIGH)
        # do_rabi(nv_sig, States.LOW, uwave_time_range=[0, 400])
        # do_rabi(nv_sig, States.HIGH, uwave_time_range=[0, 400])
        # do_spin_echo(nv_sig)
        # do_g2_measurement(nv_sig, 0, 1)
        # do_determine_standard_readout_params(nv_sig)

        # SCC characterization
        # do_determine_charge_readout_params(nv_sig,nbins=200,nreps=100)
        # do_scc_pulsed_resonance(nv_sig)

    ### Error handling and wrap-up

    except Exception as exc:
        recipient = "cambria@wisc.edu"
        tool_belt.send_exception_email(email_to=recipient)
        raise exc
    finally:
        tool_belt.reset_cfm()
        tool_belt.reset_safe_stop()
