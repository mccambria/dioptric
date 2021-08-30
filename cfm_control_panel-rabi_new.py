# -*- coding: utf-8 -*-
"""
This file contains functions to control the CFM. Just change the function call
in the main section at the bottom of this file and run the file. Shared or
frequently changed parameters are in the __main__ body and relatively static
parameters are in the function definitions.

Created on Sun Nov 25 14:00:28 2018

@author: mccambria
"""


# %% Imports


import labrad
import numpy
import time
import copy
import utils.tool_belt as tool_belt
import majorroutines.image_sample as image_sample
import majorroutines.image_sample_xz as image_sample_xz
import chargeroutines.image_sample_charge_state_compare as image_sample_charge_state_compare
import majorroutines.optimize as optimize
import majorroutines.stationary_count as stationary_count
import majorroutines.resonance as resonance
import majorroutines.pulsed_resonance as pulsed_resonance
import majorroutines.optimize_magnet_angle as optimize_magnet_angle
import majorroutines.rabi as rabi
import majorroutines.discrete_rabi as discrete_rabi
import majorroutines.g2_measurement as g2_measurement
import majorroutines.t1_double_quantum as t1_double_quantum
import majorroutines.t1_dq_knill as t1_dq_knill
import majorroutines.t1_interleave as t1_interleave
import majorroutines.t1_interleave_knill as t1_interleave_knill
import majorroutines.ramsey as ramsey
import majorroutines.spin_echo as spin_echo
import majorroutines.lifetime as lifetime
import majorroutines.lifetime_v2 as lifetime_v2
import chargeroutines.SPaCE as SPaCE
import chargeroutines.g2_measurement as g2_SCC_branch

# import majorroutines.set_drift_from_reference_image as set_drift_from_reference_image
import debug.test_major_routines as test_major_routines
from utils.tool_belt import States
import time


# %% Major Routines


def do_image_sample(nv_sig, apd_indices):

    # scan_range = 0.5
    # num_steps = 90
    # num_steps = 120
    #
    # scan_range = 0.15
    # num_steps = 60
    #
    # scan_range = 0.75
    # num_steps = 150

    #    scan_range = 5.0
    # scan_range = 3.0
    # scan_range = 1.5
    # scan_range = 1.0
    # scan_range = 0.5
    # scan_range = 0.3
    # scan_range = 0.2
    # scan_range = 0.15
    scan_range = 0.1
    # scan_range = 0.05
    # scan_range = 0.025
    #
    # num_steps = 300
    # num_steps = 200
    # num_steps = 150
    # num_steps = 135
    # num_steps = 120
    # num_steps = 90
    num_steps = 60
    # num_steps = 40
    # num_steps = 20

    # For now we only support square scans so pass scan_range twice
    image_sample.main(nv_sig, scan_range, scan_range, num_steps, apd_indices)


def do_image_sample_xz(nv_sig, apd_indices):

    scan_range_x = 0.1

    scan_range_z = 0.7

    num_steps = 120

    image_sample_xz.main(
        nv_sig,
        scan_range_x,
        scan_range_z,
        num_steps,
        apd_indices,
        um_scaled=False,
    )


def do_image_charge_states(nv_sig, apd_indices):

    scan_range = 0.1

    num_steps = 90

    image_sample_charge_state_compare.main(
        nv_sig, scan_range, scan_range, num_steps, apd_indices
    )


def do_optimize(nv_sig, apd_indices):

    optimize.main(
        nv_sig,
        apd_indices,
        set_to_opti_coords=False,
        save_data=True,
        plot_data=True,
    )


def do_optimize_list(nv_sig_list, apd_indices):

    optimize.optimize_list(nv_sig_list, apd_indices)


def do_opti_z(nv_sig_list, apd_indices):

    optimize.opti_z(
        nv_sig_list,
        apd_indices,
        set_to_opti_coords=False,
        save_data=True,
        plot_data=True,
    )


def do_stationary_count(nv_sig, apd_indices):

    run_time = 1 * 60 * 10 ** 9  # ns

    stationary_count.main(nv_sig, run_time, apd_indices)


def do_g2_measurement(nv_sig, apd_a_index, apd_b_index):

    run_time = 2*60  # s
    diff_window = 150  # ns

    # g2_measurement.main(
    g2_SCC_branch.main(
        nv_sig, run_time, diff_window, apd_a_index, apd_b_index
    )


def do_resonance(nv_sig, apd_indices, freq_center=2.87, freq_range=0.2):

    num_steps = 75
    num_runs = 5
    uwave_power = -5.0

    resonance.main(
        nv_sig,
        apd_indices,
        freq_center,
        freq_range,
        num_steps,
        num_runs,
        uwave_power,
        state=States.HIGH,
    )


def do_resonance_state(nv_sig, apd_indices, state):

    freq_center = nv_sig["resonance_{}".format(state.name)]
    uwave_power = -5.0

    freq_range = 0.200
    num_steps = 51
    num_runs = 2

    # Zoom
    # freq_range = 0.060
    # num_steps = 51
    # num_runs = 10

    resonance.main(
        nv_sig,
        apd_indices,
        freq_center,
        freq_range,
        num_steps,
        num_runs,
        uwave_power,
    )


def do_pulsed_resonance(nv_sig, apd_indices, freq_center=2.87, freq_range=0.2):

    num_steps = 51
    num_reps = 500
    num_runs = 2
    uwave_power = 14.5
    uwave_pulse_dur = 100

    pulsed_resonance.main(
        nv_sig,
        apd_indices,
        freq_center,
        freq_range,
        num_steps,
        num_reps,
        num_runs,
        uwave_power,
        uwave_pulse_dur,
    )


def do_pulsed_resonance_state(nv_sig, apd_indices, state):

    # freq_range = 0.150
    # num_steps = 51
    # num_reps = 10**4
    # num_runs = 8

    # Zoom
    freq_range = 0.1
    # freq_range = 0.120
    num_steps = 51
    num_reps = 500
    num_runs = 5

    composite = False

    res, _ = pulsed_resonance.state(
        nv_sig,
        apd_indices,
        state,
        freq_range,
        num_steps,
        num_reps,
        num_runs,
        composite,
    )
    nv_sig["resonance_{}".format(state.name)] = res


def do_optimize_magnet_angle(nv_sig, apd_indices):

    angle_range = [132, 147]
    #    angle_range = [315, 330]
    num_angle_steps = 6
    #    freq_center = 2.7921
    #    freq_range = 0.060
    #    angle_range = [0, 150]
    #    num_angle_steps = 6
    freq_center = 2.87
    freq_range = 0.220
    num_freq_steps = 51
    num_freq_runs = 10

    # Pulsed
    #    uwave_power = 14.5
    #    uwave_pulse_dur = 100
    num_freq_reps = int(1e4)

    # CW
    uwave_power = -5.0
    uwave_pulse_dur = None
    num_freq_reps = None

    optimize_magnet_angle.main(
        nv_sig,
        apd_indices,
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


def do_rabi(nv_sig, apd_indices, state, uwave_time_range=[0, 200]):

    num_steps = 51
    num_reps = 1000
    num_runs = 10

    period = rabi.main(
        nv_sig,
        apd_indices,
        uwave_time_range,
        state,
        num_steps,
        num_reps,
        num_runs,
    )
    nv_sig["rabi_{}".format(state.name)] = period


def do_discrete_rabi(nv_sig, apd_indices, state, max_num_pi_pulses=4):

    # num_reps = 2 * 10**4
    num_reps = 1000
    num_runs = 20

    iq_delay = None

    discrete_rabi.main(
        nv_sig,
        apd_indices,
        state,
        max_num_pi_pulses,
        num_reps,
        num_runs,
        iq_delay,
    )


#    for iq_delay in numpy.linspace(690, 710, 6):
#    for iq_delay in numpy.linspace(585, 595, 6):
##
#        discrete_rabi.main(nv_sig, apd_indices,
#                           state, max_num_pi_pulses, num_reps, num_runs, iq_delay)


def do_t1_battery(nv_sig, apd_indices):

    # T1 experiment parameters, formatted:
    # [[init state, read state], relaxation_time_range, num_steps, num_reps, num_runs]

    num_runs = 250
    num_reps = 500
    num_steps = 12
    min_tau = 20e3
    max_tau_omega = 29e6
    max_tau_gamma = 18e6
    # max_tau_omega = 1e6
    # max_tau_gamma = max_tau_omega
    t1_exp_array = numpy.array(
        [
            [
                [States.ZERO, States.HIGH],
                [min_tau, max_tau_omega],
                num_steps,
                num_reps,
                num_runs,
            ],
            [
                [States.ZERO, States.ZERO],
                [min_tau, max_tau_omega],
                num_steps,
                num_reps,
                num_runs,
            ],
            [
                [States.ZERO, States.HIGH],
                [min_tau, max_tau_omega // 3],
                num_steps,
                num_reps,
                num_runs,
            ],
            [
                [States.ZERO, States.ZERO],
                [min_tau, max_tau_omega // 3],
                num_steps,
                num_reps,
                num_runs,
            ],
            [
                [States.HIGH, States.LOW],
                [min_tau, max_tau_gamma],
                num_steps,
                num_reps,
                num_runs,
            ],
            [
                [States.HIGH, States.HIGH],
                [min_tau, max_tau_gamma],
                num_steps,
                num_reps,
                num_runs,
            ],
            [
                [States.HIGH, States.LOW],
                [min_tau, max_tau_gamma // 3],
                num_steps,
                num_reps,
                num_runs,
            ],
            [
                [States.HIGH, States.HIGH],
                [min_tau, max_tau_gamma // 3],
                num_steps,
                num_reps,
                num_runs,
            ],
            [
                [States.LOW, States.HIGH],
                [min_tau, max_tau_gamma],
                num_steps,
                num_reps,
                num_runs,
            ],
            [
                [States.LOW, States.LOW],
                [min_tau, max_tau_gamma],
                num_steps,
                num_reps,
                num_runs,
            ],
            [
                [States.LOW, States.HIGH],
                [min_tau, max_tau_gamma // 3],
                num_steps,
                num_reps,
                num_runs,
            ],
            [
                [States.LOW, States.LOW],
                [min_tau, max_tau_gamma // 3],
                num_steps,
                num_reps,
                num_runs,
            ],
        ],
        dtype=object,
    )

    # Loop through the experiments
    for exp_ind in range(len(t1_exp_array)):

        init_read_states = t1_exp_array[exp_ind, 0]
        relaxation_time_range = t1_exp_array[exp_ind, 1]
        num_steps = t1_exp_array[exp_ind, 2]
        num_reps = t1_exp_array[exp_ind, 3]
        num_runs = t1_exp_array[exp_ind, 4]

        t1_double_quantum.main(
            nv_sig,
            apd_indices,
            relaxation_time_range,
            num_steps,
            num_reps,
            num_runs,
            init_read_states,
        )


def do_t1_dq_knill_battery(nv_sig, apd_indices):

    # T1 experiment parameters, formatted:
    # [[init state, read state], relaxation_time_range, num_steps, num_reps, num_runs]
    num_runs = 150
    num_reps = 1e3
    num_steps = 2
    min_tau = 20e3
    # max_tau_omega = 29e6
    # max_tau_gamma = 18e6
    max_tau_omega = 1e6
    max_tau_gamma = max_tau_omega
    t1_exp_array = numpy.array(
        [
            [
                [States.ZERO, States.HIGH],
                [min_tau, max_tau_omega],
                num_steps,
                num_reps,
                num_runs,
            ],
            [
                [States.ZERO, States.ZERO],
                [min_tau, max_tau_omega],
                num_steps,
                num_reps,
                num_runs,
            ],
            [
                [States.ZERO, States.HIGH],
                [min_tau, max_tau_omega // 3],
                num_steps,
                num_reps,
                num_runs,
            ],
            [
                [States.ZERO, States.ZERO],
                [min_tau, max_tau_omega // 3],
                num_steps,
                num_reps,
                num_runs,
            ],
            # [[States.HIGH, States.LOW], [min_tau, max_tau_gamma], num_steps, num_reps, num_runs],
            # [[States.HIGH, States.HIGH], [min_tau, max_tau_gamma], num_steps, num_reps, num_runs],
            # [[States.HIGH, States.LOW], [min_tau, max_tau_gamma//3], num_steps, num_reps, num_runs],
            # [[States.HIGH, States.HIGH], [min_tau, max_tau_gamma//3], num_steps, num_reps, num_runs],
            # [[States.LOW, States.HIGH], [min_tau, max_tau_gamma], num_steps, num_reps, num_runs],
            # [[States.LOW, States.LOW], [min_tau, max_tau_gamma], num_steps, num_reps, num_runs],
            # [[States.LOW, States.HIGH], [min_tau, max_tau_gamma//3], num_steps, num_reps, num_runs],
            # [[States.LOW, States.LOW], [min_tau, max_tau_gamma//3], num_steps, num_reps, num_runs],
        ],
        dtype=object,
    )

    # Loop through the experiments
    for exp_ind in range(len(t1_exp_array)):

        init_read_states = t1_exp_array[exp_ind, 0]
        relaxation_time_range = t1_exp_array[exp_ind, 1]
        num_steps = t1_exp_array[exp_ind, 2]
        num_reps = t1_exp_array[exp_ind, 3]
        num_runs = t1_exp_array[exp_ind, 4]

        t1_dq_knill.main(
            nv_sig,
            apd_indices,
            relaxation_time_range,
            num_steps,
            num_reps,
            num_runs,
            init_read_states,
        )


def do_t1_interleave_knill(nv_sig, apd_indices):
    # T1 experiment parameters, formatted:
    # [[init state, read state], relaxation_time_range, num_steps, num_reps]
    num_runs = 80
    num_reps = 1000
    num_steps = 12
    min_tau = 20e3
    max_tau_omega = int(8e6)
    max_tau_gamma = int(6e6)
    t1_exp_array = numpy.array(
        [
            [
                [States.ZERO, States.HIGH],
                [min_tau, max_tau_omega],
                num_steps,
                num_reps,
            ],
            [
                [States.ZERO, States.ZERO],
                [min_tau, max_tau_omega],
                num_steps,
                num_reps,
            ],
            [
                [States.ZERO, States.HIGH],
                [min_tau, max_tau_omega // 3],
                num_steps,
                num_reps,
            ],
            [
                [States.ZERO, States.ZERO],
                [min_tau, max_tau_omega // 3],
                num_steps,
                num_reps,
            ],
            #            [[States.HIGH, States.LOW], [min_tau, max_tau_gamma], num_steps, num_reps],
            #            [[States.HIGH, States.HIGH], [min_tau, max_tau_gamma], num_steps, num_reps],
            #            [[States.HIGH, States.LOW], [min_tau, max_tau_gamma//3], num_steps, num_reps],
            #            [[States.HIGH, States.HIGH], [min_tau, max_tau_gamma//3], num_steps, num_reps],
            [
                [States.LOW, States.HIGH],
                [min_tau, max_tau_gamma],
                num_steps,
                num_reps,
            ],
            [
                [States.LOW, States.LOW],
                [min_tau, max_tau_gamma],
                num_steps,
                num_reps,
            ],
            [
                [States.LOW, States.HIGH],
                [min_tau, max_tau_gamma // 3],
                num_steps,
                num_reps,
            ],
            [
                [States.LOW, States.LOW],
                [min_tau, max_tau_gamma // 3],
                num_steps,
                num_reps,
            ],
        ],
        dtype=object,
    )

    t1_interleave_knill.main(nv_sig, apd_indices, t1_exp_array, num_runs)


def do_lifetime(nv_sig, apd_indices, filter, voltage, reference=False):

    #    num_reps = 100 #MM
    num_reps = 500  # SM
    num_bins = 101
    num_runs = 10
    readout_time_range = [0, 1.0 * 10 ** 6]  # ns
    polarization_time = 60 * 10 ** 3  # ns

    lifetime_v2.main(
        nv_sig,
        apd_indices,
        readout_time_range,
        num_reps,
        num_runs,
        num_bins,
        filter,
        voltage,
        polarization_time,
        reference,
    )


def do_ramsey(nv_sig, apd_indices):

    detuning = 2.5  # MHz
    precession_time_range = [0, 4 * 10 ** 3]
    num_steps = 151
    num_reps = 3 * 10 ** 5
    num_runs = 1

    ramsey.main(
        nv_sig,
        apd_indices,
        detuning,
        precession_time_range,
        num_steps,
        num_reps,
        num_runs,
    )


def do_spin_echo_battery(nv_sig, apd_indices):

    for magnet_angle in numpy.linspace(125, 135, 6):
        nv_sig["magnet_angle"] = magnet_angle
        do_pulsed_resonance_state(nv_sig, apd_indices, States.LOW)
        do_pulsed_resonance_state(nv_sig, apd_indices, States.HIGH)
        do_rabi(nv_sig, apd_indices, States.LOW, uwave_time_range=[0, 400])
        do_rabi(nv_sig, apd_indices, States.HIGH, uwave_time_range=[0, 400])
        angle = do_spin_echo(nv_sig, apd_indices)
        if angle < 6:
            break


def do_spin_echo(nv_sig, apd_indices):

    # T2* in nanodiamond NVs is just a couple us at 300 K
    # In bulk it's more like 100 us at 300 K
    max_time = 120  # us
    num_steps = int(max_time + 1)  # 1 point per us
    #    num_steps = int(max_time/2) + 1  # 2 point per us
    #    max_time = 1  # us
    #    num_steps = 51
    precession_time_range = [0, max_time * 10 ** 3]
    #    num_reps = 8000
    #    num_runs = 5
    num_reps = 1000
    num_runs = 40

    #    num_steps = 151
    #    precession_time_range = [0, 10*10**3]
    #    num_reps = int(10.0 * 10**4)
    #    num_runs = 6

    state = States.LOW

    angle = spin_echo.main(
        nv_sig,
        apd_indices,
        precession_time_range,
        num_steps,
        num_reps,
        num_runs,
        state,
    )
    return angle


def do_SPaCE(nv_sig, opti_nv_sig, img_range, num_steps,num_runs,dz,  measurement_type ):
    # dr = 0.025 / numpy.sqrt(2)
    # img_range = [[-dr,-dr],[dr, dr]] #[[x1, y1], [x2, y2]]
    # num_steps = 101
    # num_runs = 50
    # measurement_type = "1D"

    # img_range = 0.075
    # num_steps = 71
    # num_runs = 1
    # measurement_type = "2D"

    # dz = 0
    SPaCE.main(nv_sig, opti_nv_sig,img_range, num_steps, num_runs, measurement_type, dz)


def do_sample_nvs(nv_sig_list, apd_indices):

    # g2 parameters
    run_time = 60 * 5
    diff_window = 150

    # PESR parameters
    num_steps = 101
    num_reps = 10 ** 5
    num_runs = 3
    uwave_power = 9.0
    uwave_pulse_dur = 100

    g2 = g2_measurement.main_with_cxn
    pesr = pulsed_resonance.main_with_cxn

    with labrad.connect() as cxn:
        for nv_sig in nv_sig_list:
            g2_zero = g2(
                cxn,
                nv_sig,
                run_time,
                diff_window,
                apd_indices[0],
                apd_indices[1],
            )
            if g2_zero < 0.5:
                pesr(
                    cxn,
                    nv_sig,
                    apd_indices,
                    2.87,
                    0.1,
                    num_steps,
                    num_reps,
                    num_runs,
                    uwave_power,
                    uwave_pulse_dur,
                )


def do_test_major_routines(nv_sig, apd_indices):
    """Run this whenver you make a significant code change. It'll make sure
    you didn't break anything in the major routines.
    """

    test_major_routines.main(nv_sig, apd_indices)


# %% Run the file


if __name__ == "__main__":

    # In debug mode, don't bother sending email notifications about exceptions
    debug_mode = True

    # %% Shared parameters

    apd_indices = [0]
    # apd_indices = [1]
    # apd_indices = [0,1]

    nd_yellow = "nd_0.5"
    sample_name = "johnson"
    green_laser = "cobolt_515"
    yellow_laser = "laserglow_589"
    red_laser = "cobolt_638"

    nv_sig = {
        "coords": [0,0, 4.7],
        "name": "{}-search".format(sample_name),
        "disable_opt": False,
        "expected_count_rate": None,
        "imaging_laser": green_laser,
        "imaging_laser_power": 6,
        "imaging_readout_dur": 1e7,
        "collection_filter": "630_lp",
        "magnet_angle": None,
        "resonance_LOW": 2.8012,
        "rabi_LOW": 141.5,
        "uwave_power_LOW": 15.5,  # 15.5 max
        "resonance_HIGH": 2.9445,
        "rabi_HIGH": 191.9,
        "uwave_power_HIGH": 14.5,
    }  # 14.5 max

    
    nv_sig = {
        "coords": [-0.008, -0.097, 4.7],
        "name": "{}-nv1_2021_08_30".format(sample_name,),
        "disable_opt": False,
        "expected_count_rate": 18,
        "spin_laser": green_laser,
        "spin_laser_power": 6,
        "spin_pol_dur": 1e5,
        "spin_readout_laser_power": 6,
        "spin_readout_dur": 350,
        
        "imaging_laser":green_laser,
        "imaging_laser_power": 6,
        "imaging_readout_dur": 1e7,
        
        "initialize_laser": green_laser,
        "initialize_laser_power": 6,
        "initialize_dur": 1e3,
        "CPG_laser": red_laser,
        'CPG_laser_power': 80,
        "CPG_laser_dur": 150e3,
        "charge_readout_laser": green_laser,
        # "charge_readout_laser_filter": nd_yellow,
        "charge_readout_laser_power": 6,
        "charge_readout_dur": 1e7,
        # "charge_readout_laser": yellow_laser,
        # "charge_readout_laser_filter": nd_yellow,
        # "charge_readout_laser_power": 0.1,
        # "charge_readout_dur": 50e6,
        
        "dir_1D": "x",
        "collection_filter": "630_lp",
        "magnet_angle": None,
        "resonance_LOW": 2.8012,
        "rabi_LOW": 141.5,
        "uwave_power_LOW": 15.5,  # 15.5 max
        "resonance_HIGH": 2.9445,
        "rabi_HIGH": 191.9,
        "uwave_power_HIGH": 14.5,
    }  # 14.5 max
    
    # nv_sig = {
    #     "coords": [-0.042, -0.100, 5.0],
    #     "name": "{}-nv2_2021_08_27".format(sample_name,),
    #     "disable_opt": False,
    #     "expected_count_rate": 32,
        
    #     "spin_laser": green_laser,
    #     "spin_laser_power": 6,
    #     "spin_pol_dur": 1e5,
    #     "spin_readout_laser_power": 6,
    #     "spin_readout_dur": 350,
        
    #     "imaging_laser":green_laser,
    #     "imaging_laser_power": 6,
    #     "imaging_readout_dur": 1e7,
        
    #     "initialize_laser": green_laser,
    #     "initialize_laser_power": 6,
    #     "initialize_dur": 1e3,
    #     "CPG_laser": red_laser,
    #     'CPG_laser_power': 80,
    #     "CPG_laser_dur": 150e3,
    #     "charge_readout_laser": yellow_laser,
    #     "charge_readout_laser_filter": nd_yellow,
    #     "charge_readout_laser_power": 0.1,
    #     "charge_readout_dur": 50e6,
        
    #     "dir_1D": "x",
    #     "collection_filter": "630_lp",
    #     "magnet_angle": None,
    #     "resonance_LOW": 2.8012,
    #     "rabi_LOW": 141.5,
    #     "uwave_power_LOW": 15.5,  # 15.5 max
    #     "resonance_HIGH": 2.9445,
    #     "rabi_HIGH": 191.9,
    #     "uwave_power_HIGH": 14.5,
    # }  # 14.5 max

    # %% Functions to run

    try:

        # tool_belt.init_safe_stop()
        # while True:
        #     if tool_belt.safe_stop():
        #         break
        #     do_image_sample(nv_sig, apd_indices)
        #     do_pulsed_resonance_state(nv_sig, apd_indices, States.LOW)
        #     do_pulsed_resonance_state(nv_sig, apd_indices, States.HIGH)

        # for dz in numpy.linspace(3.5, 6, 7)/16:
        # for dz in numpy.linspace(-0.1, 0.1, 5):
            # # do_optimize(nv_sig, apd_indices)
            # nv_sig_copy = copy.deepcopy(nv_sig)
            # coords = nv_sig['coords']
            # nv_sig_copy['coords'] = [coords[0],coords[1],coords[2]+dz]
            # do_image_sample(nv_sig_copy, apd_indices)
        # do_optimize(nv_sig, apd_indices)
        # do_image_sample(nv_sig, apd_indices)
        # do_image_sample_xz(nv_sig, apd_indices)
        # do_image_charge_states(nv_sig, apd_indices)
        # tool_belt.set_drift([0.0, 0.0, 0.0])  # Totally reset
        # drift = tool_belt.get_drift()
        # tool_belt.set_drift([0.0, 0.0, drift[2]])  # Keep z
        # tool_belt.set_drift([drift[0], drift[1], 0.0])  # Keep xy
        # do_stationary_count(nv_sig, apd_indices)
        # do_resonance(nv_sig, apd_indices, 2.87, 0.25)
        # do_resonance_state(nv_sig, apd_indices, States.HIGH)
        # do_pulsed_resonance(nv_sig, apd_indices, 2.87, 0.220)
        # do_resonance_state(nv_sig, apd_indices, States.LOW)
        # do_resonance_state(nv_sig, apd_indices, States.HIGH)
        # do_pulsed_resonance_state(nv_sig, apd_indices, States.LOW)
        # do_pulsed_resonance_state(nv_sig, apd_indices, States.HIGH)
        #         do_optimize_magnet_angle(nv_sig, apd_indices)
        # do_rabi(nv_sig, apd_indices, States.LOW, uwave_time_range=[0, 400])
        # do_rabi(nv_sig, apd_indices, States.HIGH, uwave_time_range=[0, 400])
        # do_discrete_rabi(nv_sig, apd_indices, States.LOW, 4)
        # do_discrete_rabi(nv_sig, apd_indices, States.HIGH, 4)
        # do_spin_echo(nv_sig, apd_indices)
        
        
        # t_list = [50e3 ]#, 400e3, 600e3, 800e3, 1000e3]
        t_list = [50e3]
        for t in t_list:
            nv_sig_copy = copy.deepcopy(nv_sig)
            nv_sig_copy['CPG_laser_dur'] = t
            dz = 2/16
            
            img_range = 0.055
            num_steps = 76
            num_runs = 1
            measurement_type = '2D'
        
            do_SPaCE(nv_sig_copy, nv_sig_copy, img_range,  num_steps,num_runs,dz, measurement_type)
        t_list = [500e3, 1e6, 1.5e6, 2e6]
        for t in t_list:
            dz = 0
            nv_sig_copy = copy.deepcopy(nv_sig)
            nv_sig_copy['CPG_laser_dur'] = t
            # dr = -0.4/35
            img_range = 0.1# [[-dr ,0],[dr, 0]] #[[x1, y1], [x2, y2]]
            num_steps = 76
            num_runs = 1
            measurement_type = '2D'
        
            # do_SPaCE(nv_sig_copy, nv_sig_copy, img_range,  num_steps,num_runs,dz,  measurement_type)
            
        
            
        # do_spin_echo_battery(nv_sig, apd_indices)
        # do_g2_measurement(nv_sig, 0, 1)  # 0, (394.6-206.0)/31 = 6.084 ns, 164.3 MHz; 1, (396.8-203.6)/33 = 5.855 ns, 170.8 MHz
        # do_t1_battery(nv_sig, apd_indices)
        # do_t1_interleave_knill(nv_sig, apd_indices)

        # Operations that don't need an NV
        # tool_belt.set_drift([0.0, 0.0, 0.0])  # Totally reset
        # tool_belt.set_drift([0.0, 0.0, tool_belt.get_drift()[2]])  # Keep z
        # tool_belt.set_xyz(labrad.connect(), [0.0, 0.0 , 5.0])
        # set_xyz([0.454, 0.832, -88])

    except Exception as exc:
        # Intercept the exception so we can email it out and re-raise it
        if not debug_mode:
            tool_belt.send_exception_email()
        raise exc

    finally:
        # Reset our hardware - this should be done in each routine, but
        # let's double check here
        tool_belt.reset_cfm()
        # Kill safe stop
        if tool_belt.check_safe_stop_alive():
            print("\n\nRoutine complete. Press enter to exit.")
            tool_belt.poll_safe_stop()
