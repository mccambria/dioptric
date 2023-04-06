# -*- coding: utf-8 -*-
"""
Control panel specifically for ZFS vs temp measurements

Created on November 15th, 2022

@author: mccambria
"""


### Imports


import labrad
import numpy as np
import time
import copy
import utils.tool_belt as tool_belt
import utils.positioning as positioning
import majorroutines.image_sample as image_sample
import majorroutines.optimize as optimize
import majorroutines.stationary_count as stationary_count
import majorroutines.resonance as resonance
import majorroutines.pulsed_resonance as pulsed_resonance
import majorroutines.four_point_esr as four_point_esr
import majorroutines.rabi as rabi
import majorroutines.determine_standard_readout_params as determine_standard_readout_params
from utils.tool_belt import States, NormStyle
from figures.zfs_vs_t.zfs_vs_t_main import cambria_fixed
from random import shuffle
import sys


### Major Routines


def do_image_sample(nv_sig, nv_minus_init=False):

    # scan_range = 0.2
    # num_steps = 60

    # scan_range = 0.4
    # num_steps = 90
    # num_steps = 120

    # scan_range = 0.5
    # num_steps = 90
    # num_steps = 150

    # scan_range = 0.3
    # num_steps = 80

    # scan_range = 1.0
    # num_steps = 90

    # scan_range = 1.5
    # num_steps = 180

    scan_range = 2.0
    num_steps = 90

    # scan_range = 5.0
    # num_steps = 90*6
    # num_steps = 180

    # For now we only support square scans so pass scan_range twice
    image_sample.main(
        nv_sig,
        scan_range,
        scan_range,
        num_steps,
        nv_minus_init=nv_minus_init,
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
    nv_minus_init=False,
    nv_zero_init=False,
):

    run_time = 3 * 60 * 10**9  # ns

    stationary_count.main(
        nv_sig,
        run_time,
        disable_opt=disable_opt,
        nv_minus_init=nv_minus_init,
        nv_zero_init=nv_zero_init,
    )


def do_stationary_count_bg_subt(nv_sig, bg_coords):

    run_time = 3 * 60 * 10**9  # ns

    stationary_count.main(
        nv_sig,
        run_time,
        disable_opt=True,
        background_subtraction=True,
        background_coords=bg_coords,
    )


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


def do_four_point_esr(nv_sig, state):

    detuning = 0.004
    d_omega = 0.002
    num_reps = 1e5
    num_runs = 4

    ret_vals = four_point_esr.main(
        nv_sig,
        num_reps,
        num_runs,
        state,
        detuning,
        d_omega,
        ret_file_name=True,
    )

    # print(resonance, res_err)
    return ret_vals


def do_determine_standard_readout_params(nv_sig):

    num_reps = 1e6
    max_readouts = [1e3]
    # num_reps = 4e3
    # max_readouts = [6e6]
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

    # num_reps = 2e4
    # num_runs = 16

    num_reps = 1e2
    num_runs = 16

    uwave_power = 10
    uwave_pulse_dur = 100

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


def do_pulsed_resonance_batch(nv_list, temp, freq_range=None):

    num_steps = 51

    # Microdiamond
    num_reps = 1e2
    num_runs = 32
    if freq_range is None:
        freq_range = 0.060

    # Single
    # num_reps = 5e4
    # num_runs = 32
    # if freq_range is None:
    #     freq_range = 0.020
        # freq_range = 0.040

    # num_reps = 50
    # num_runs = 8

    uwave_power = 10
    uwave_pulse_dur = 100

    freq_center = cambria_fixed(temp)
    # freq_center = 2.8773

    for nv_sig in nv_list:
        if tool_belt.safe_stop():
            break
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


def do_rabi(nv_sig, state, uwave_time_range=[0, 300]):

    num_steps = 51

    # num_reps = 0.5e4  # 2e4
    # num_runs = 16

    num_reps = 1e2
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


def do_rabi_batch(nv_list):

    num_steps = 51

    num_reps = 1e2
    num_runs = 16

    uwave_time_range = [0, 300]
    state = States.LOW

    for nv_sig in nv_list:
        if tool_belt.safe_stop():
            break
        rabi.main(
            nv_sig,
            uwave_time_range,
            state,
            num_steps,
            num_reps,
            num_runs,
        )


### Run the file


if __name__ == "__main__":

    ### Shared parameters

    green_laser = "laserglow_532"
    yellow_laser = "laserglow_589"
    red_laser = "cobolt_638"

    # fmt: off

    sample_name = "15micro"
    z_coord = 5.5
    ref_coords = [0.142, -0.162, z_coord]
    ref_coords = np.array(ref_coords)

    nvref = {
        'coords': ref_coords,
        'name': '{}-nvref_zfs_vs_t'.format(sample_name),
        'disable_opt': True, "disable_z_opt": True, 'expected_count_rate': 2000,
        'imaging_laser': green_laser, 'imaging_laser_filter': "nd_0.3", 'imaging_readout_dur': 1e7,
        # 'imaging_laser': green_laser, 'imaging_laser_filter': "nd_0.5", 'imaging_readout_dur': 1e7,

        # Microdiamond
        # "spin_laser": green_laser, "spin_laser_filter": "nd_1.0",
        # "spin_laser": green_laser, "spin_laser_filter": "nd_0.5",
        "spin_laser": green_laser, "spin_laser_filter": "nd_0.3",
        # "spin_laser": green_laser, "spin_laser_filter": "nd_0",
        "spin_pol_dur": 3e6, "spin_readout_dur": 5e5,
        # "spin_laser": green_laser, "spin_laser_filter": "nd_0",
        # "spin_pol_dur": 3e6, "spin_readout_dur": 5e5,

        # Single
        # "spin_laser": green_laser, "spin_laser_filter": "nd_0",
        # "spin_pol_dur": 2e3, "spin_readout_dur": 440,

        "norm_style": NormStyle.POINT_TO_POINT, 'collection_filter': "nd_0.2", 'magnet_angle': None,
        # "norm_style": NormStyle.POINT_TO_POINT, 'collection_filter': "nd_0.1", 'magnet_angle': None,
        # "norm_style": NormStyle.POINT_TO_POINT, 'collection_filter': "nd_0", 'magnet_angle': None,
        'resonance_LOW': 2.86, 'rabi_LOW': 200, 'uwave_power_LOW': 10.0,
        }

    nv6 = copy.deepcopy(nvref)
    nv6["coords"] = ref_coords + np.array([0.103, -0.119, 0])
    nv6["name"] =  f"{sample_name}-nv6_zfs_vs_t"

    nv7 = copy.deepcopy(nvref)
    nv7["coords"] = ref_coords + np.array([0.061, -0.109, 0])
    nv7["name"] =  f"{sample_name}-nv7_zfs_vs_t"

    nv8 = copy.deepcopy(nvref)
    nv8["coords"] = ref_coords + np.array([-0.143,  0.033, 0])
    nv8["name"] =  f"{sample_name}-nv8_zfs_vs_t"

    nv9 = copy.deepcopy(nvref)
    nv9["coords"] = ref_coords + np.array([-0.035,  0.2, 0])
    nv9["name"] =  f"{sample_name}-nv9_zfs_vs_t"

    nv11 = copy.deepcopy(nvref)
    nv11["coords"] = ref_coords
    nv11["name"] =  f"{sample_name}-nv11_zfs_vs_t"

    # fmt: on

    # nv_sig = nv8
    nv_sig = nvref
    bg_coords = np.array(nv_sig["coords"]) + np.array([0.04, -0.06, 0])
    nv_list = [nv6, nv7, nv8, nv9, nv11]
    # nv_list = [nv9]

    ### Functions to run

    try:

        # pass

        tool_belt.init_safe_stop()

        # Increasing x moves the image down, increasing y moves the image left
        # with labrad.connect() as cxn:
        #     cxn.pos_xyz_ATTO_piezos.write_xy(5, 3)

        # with labrad.connect() as cxn:
        #     positioning.set_drift(cxn, [0.0, 0.0, 0])  # Totally reset
        #     drift = positioning.get_drift(cxn)
        #     positioning.set_drift(cxn, [0.0, 0.0, drift[2]])  # Keep z
        #     positioning.set_drift([drift[0], drift[1], 0.0])  # Keep xy

        # for z in np.arange(-24, 20, 4):
        # for z in np.arange(2.0, 8.0, 0.3):
        # # z = 0
        # # while True:
        #     if tool_belt.safe_stop():
        #         break
        # #     with labrad.connect() as cxn:
        # #         cxn.pos_xyz_ATTO_piezos.write_z(z)
        # #     print(z)
        # #     z += 5
        # #     time.sleep(5)
        #     nv_sig["coords"][2] = z
        #     do_image_sample(nv_sig)

        # nv_sig = nvref
        # nv_sig['imaging_readout_dur'] = 4e7
        do_image_sample(nv_sig)
        # do_image_sample_zoom(nv_sig)

        # for nv in nv_list:
        #     if tool_belt.safe_stop():
        #         break
        #     # print(nv["coords"])
        #     do_image_sample_zoom(nv)

        # do_optimize(nv_sig)

        # nv_sig['imaging_readout_dur'] = 1e8
        # do_stationary_count(nv_sig, disable_opt=True)
        # nv_sig['imaging_readout_dur'] = 1e8
        # do_stationary_count_bg_subt(nv_sig, bg_coords)

        # do_determine_standard_readout_params(nv_sig)

        # do_pulsed_resonance(nv_sig, 2.86, 0.060)
        # do_rabi(nv_sig, States.LOW, uwave_time_range=[0, 500])
        # do_four_point_esr(nv_sig, States.LOW)

        # temp = 300
        # shuffle(nv_list)
        # do_pulsed_resonance_batch(nv_list, temp, freq_range=0.060)
        # shuffle(nv_list)
        # do_pulsed_resonance_batch(nv_list, temp, freq_range=0.040)
        # do_rabi_batch(nv_list)

    # except Exception as exc:
    #     recipient = "cambria@wisc.edu"
    #     tool_belt.send_exception_email(email_to=recipient)
    #     raise exc

    finally:

        # msg = "Experiment complete!"
        # recipient = "cambria@wisc.edu"
        # tool_belt.send_email(msg, email_to=recipient)

        # Make sure everything is reset
        tool_belt.reset_cfm()
        tool_belt.reset_safe_stop()
