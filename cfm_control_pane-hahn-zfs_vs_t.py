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
import majorroutines.image_sample as image_sample
import majorroutines.image_sample_temperature as image_sample_temperature
import majorroutines.map_rabi_contrast_NIR as map_rabi_contrast_NIR
import majorroutines.ensemble_image_sample_NIR_differential as ensemble_image_sample_NIR_differential
import majorroutines.ensemble_image_sample_NIR_differential_faster as ensemble_image_sample_NIR_differential_faster
import majorroutines.optimize as optimize
import majorroutines.stationary_count as stationary_count
import majorroutines.resonance as resonance
import majorroutines.pulsed_resonance as pulsed_resonance
import majorroutines.four_point_esr as four_point_esr
import majorroutines.optimize_magnet_angle as optimize_magnet_angle
import majorroutines.rabi as rabi
import majorroutines.discrete_rabi as discrete_rabi
import majorroutines.g2_measurement as g2_measurement
import majorroutines.t1_dq_main as t1_dq_main
import majorroutines.ramsey as ramsey
import majorroutines.spin_echo as spin_echo
import majorroutines.lifetime as lifetime
import majorroutines.lifetime_v2 as lifetime_v2
import chargeroutines.determine_charge_readout_params as determine_charge_readout_params
import chargeroutines.determine_charge_readout_params_moving_target as determine_charge_readout_params_moving_target
import chargeroutines.determine_charge_readout_params_1Dscan_target as determine_charge_readout_params_1Dscan_target
import minorroutines.determine_standard_readout_params as determine_standard_readout_params
import chargeroutines.scc_pulsed_resonance as scc_pulsed_resonance
import debug.test_major_routines as test_major_routines
from utils.tool_belt import States
import time
import services.calibrated_temp_monitor as calibrated_temp_monitor
from analysis.temp_from_resonances import cambria_fixed
from random import shuffle


### Major Routines


def do_image_sample(
    nv_sig,
    apd_indices,
    nv_minus_initialization=False,
    cbarmin=None,
    cbarmax=None,
):

    # scan_range = 0.2
    # num_steps = 60

    scan_range = 0.5
    num_steps = 90

    # scan_range = 0.3
    # num_steps = 80

    # scan_range = 1.0
    # num_steps = 240

    # scan_range = 3.0
    # num_steps = 300

    # For now we only support square scans so pass scan_range twice
    image_sample.main(
        nv_sig,
        scan_range,
        scan_range,
        num_steps,
        apd_indices,
        nv_minus_initialization=nv_minus_initialization,
        cmin=cbarmin,
        cmax=cbarmax,
    )


def do_image_sample_zoom(nv_sig, apd_indices):

    scan_range = 0.05
    num_steps = 30

    image_sample.main(
        nv_sig,
        scan_range,
        scan_range,
        num_steps,
        apd_indices,
    )


def do_optimize(nv_sig, apd_indices):

    optimize.main(
        nv_sig,
        apd_indices,
        set_to_opti_coords=False,
        save_data=True,
        plot_data=True,
    )


def do_stationary_count(
    nv_sig,
    apd_indices,
    disable_opt=None,
    nv_minus_initialization=False,
    nv_zero_initialization=False,
):

    run_time = 3 * 60 * 10**9  # ns

    stationary_count.main(
        nv_sig,
        run_time,
        apd_indices,
        disable_opt=disable_opt,
        nv_minus_initialization=nv_minus_initialization,
        nv_zero_initialization=nv_zero_initialization,
    )

def do_stationary_count_bg_subt(
    nv_sig,
    apd_indices,
    bg_coords
):

    run_time = 3 * 60 * 10**9  # ns

    stationary_count.main(
        nv_sig,
        run_time,
        apd_indices,
        disable_opt=True,
        background_subtraction=True,
        background_coords=bg_coords,
    )


def do_resonance(nv_sig, apd_indices, freq_center=2.87, freq_range=0.2):

    num_steps = 51
    num_runs = 20
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


def do_four_point_esr(nv_sig, apd_indices, state):

    detuning = 0.004
    d_omega = 0.002
    num_reps = 1e5
    num_runs = 4

    ret_vals = four_point_esr.main(
        nv_sig,
        apd_indices,
        num_reps,
        num_runs,
        state,
        detuning,
        d_omega,
        ret_file_name=True,
    )

    # print(resonance, res_err)
    return ret_vals


def do_determine_standard_readout_params(nv_sig, apd_indices):

    num_reps = 1e5
    max_readouts = [1e6]
    filters = ["nd_0"]
    state = States.LOW

    determine_standard_readout_params.main(
        nv_sig,
        apd_indices,
        num_reps,
        max_readouts,
        filters=filters,
        state=state,
    )


def do_pulsed_resonance(nv_sig, apd_indices, freq_center=2.87, freq_range=0.2):

    num_steps = 51

    num_reps = 2e4
    num_runs = 16

    # num_reps = 1e3
    # num_runs = 8

    uwave_power = 16.5
    uwave_pulse_dur = 400

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


def do_pulsed_resonance_batch(nv_list, apd_indices, temp):

    num_steps = 51
    num_reps = 2e4
    num_runs = 32

    uwave_power = 4
    uwave_pulse_dur = 100

    freq_center = cambria_fixed(temp)
    freq_center = 2.8773
    freq_range = 0.020

    for nv_sig in nv_list:
        if tool_belt.safe_stop():
            break
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


def do_rabi(nv_sig, apd_indices, state, uwave_time_range=[0, 200]):

    num_steps = 51

    num_reps = 2e4
    num_runs = 16

    # num_reps = 1e3
    # num_runs = 8

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


def do_rabi_batch(nv_list, apd_indices):

    num_steps = 51
    num_reps = 2e4
    num_runs = 8
    uwave_time_range=[0, 300]
    state = States.LOW

    for nv_sig in nv_list:
        if tool_belt.safe_stop():
            break
        rabi.main(
            nv_sig,
            apd_indices,
            uwave_time_range,
            state,
            num_steps,
            num_reps,
            num_runs,
        )


def wait_for_stable_temp():

    calibrated_temp_monitor.main()


### Run the file


if __name__ == "__main__":

    ### Shared parameters

    apd_indices = [0]
    # apd_indices = [1]
    # apd_indices = [0, 1]

    green_laser = "laserglow_532"
    yellow_laser = "laserglow_589"
    red_laser = "cobolt_638"

    # fmt: off

    sample_name = "wu"
    
    ref_coords = [0.268, -0.421, 8]
    ref_coords = np.array(ref_coords)
    freq = 2.8773
    rabi_per = 200
    uwave_power = 4
    
    
    nvref = {
        'coords': ref_coords, 
        'name': '{}-nvref_zfs_vs_t'.format(sample_name),
        'disable_opt': False, "disable_z_opt": True, 'expected_count_rate': 10,
        'imaging_laser': green_laser, 'imaging_laser_filter': "nd_0", 'imaging_readout_dur': 1e7,
        "spin_laser": green_laser, "spin_laser_filter": "nd_0", "spin_pol_dur": 2e3, "spin_readout_dur": 440,
        'collection_filter': None, 'magnet_angle': None,
        'resonance_LOW': freq, 'rabi_LOW': rabi_per, 'uwave_power_LOW': uwave_power,
        }
    nv1 = {
        'coords': ref_coords + np.array([0.174, 0.108, 0]), 
        'name': '{}-nv1_zfs_vs_t'.format(sample_name),
        'disable_opt': False, "disable_z_opt": True, 'expected_count_rate': 6.5,
        'imaging_laser': green_laser, 'imaging_laser_filter': "nd_0", 'imaging_readout_dur': 1e7,
        "spin_laser": green_laser, "spin_laser_filter": "nd_0", "spin_pol_dur": 2e3, "spin_readout_dur": 440,
        'collection_filter': None, 'magnet_angle': None,
        'resonance_LOW': freq, 'rabi_LOW': rabi_per, 'uwave_power_LOW': uwave_power,
        }
    nv2 = {
        'coords': ref_coords + np.array([0.157, -0.021, 0]), 
        'name': '{}-nv2_zfs_vs_t'.format(sample_name),
        'disable_opt': False, "disable_z_opt": True, 'expected_count_rate': 6.5,
        'imaging_laser': green_laser, 'imaging_laser_filter': "nd_0", 'imaging_readout_dur': 1e7,
        "spin_laser": green_laser, "spin_laser_filter": "nd_0", "spin_pol_dur": 2e3, "spin_readout_dur": 440,
        'collection_filter': None, 'magnet_angle': None,
        'resonance_LOW': freq, 'rabi_LOW': rabi_per, 'uwave_power_LOW': uwave_power,
        }
    nv3 = {
        'coords': ref_coords + np.array([0.052, 0.147, 0]),
        'name': '{}-nv3_zfs_vs_t'.format(sample_name),
        'disable_opt': False, "disable_z_opt": True, 'expected_count_rate': 6.5,
        'imaging_laser': green_laser, 'imaging_laser_filter': "nd_0", 'imaging_readout_dur': 1e7,
        "spin_laser": green_laser, "spin_laser_filter": "nd_0", "spin_pol_dur": 2e3, "spin_readout_dur": 440,
        'collection_filter': None, 'magnet_angle': None,
        'resonance_LOW': freq, 'rabi_LOW': rabi_per, 'uwave_power_LOW': uwave_power,
        }
    nv4 = {
        'coords': ref_coords + np.array([-0.237, 0.026, 0]), 
        'name': '{}-nv4_zfs_vs_t'.format(sample_name),
        'disable_opt': False, "disable_z_opt": True, 'expected_count_rate': 6.5,
        'imaging_laser': green_laser, 'imaging_laser_filter': "nd_0", 'imaging_readout_dur': 1e7,
        "spin_laser": green_laser, "spin_laser_filter": "nd_0", "spin_pol_dur": 2e3, "spin_readout_dur": 440,
        'collection_filter': None, 'magnet_angle': None,
        'resonance_LOW': freq, 'rabi_LOW': rabi_per, 'uwave_power_LOW': uwave_power,
        }
    nv5 = {
        'coords': ref_coords + np.array([0.074, -0.050, 0]), 
        # "coords": [0.343, -0.467, 6],
        'name': '{}-nv5_zfs_vs_t'.format(sample_name),
        'disable_opt': False, "disable_z_opt": True, 'expected_count_rate': 7,
        'imaging_laser': green_laser, 'imaging_laser_filter': "nd_0", 'imaging_readout_dur': 1e7,
        "spin_laser": green_laser, "spin_laser_filter": "nd_0", "spin_pol_dur": 2e3, "spin_readout_dur": 440,
        'collection_filter': None, 'magnet_angle': None,
        'resonance_LOW': freq, 'rabi_LOW': rabi_per, 'uwave_power_LOW': uwave_power,
        }

    # sample_name = "15micro"
    # nv_sig = {
    #     # 'coords': [0.0, 0.0, 0], 'name': '{}-search'.format(sample_name),
    #     # 'coords': [0.205, -0.111, 0], 'name': '{}-search'.format(sample_name),
    #     'coords': [-0.168, 0.200, -3], 'name': '{}-nv2_2022_11_02'.format(sample_name),
    #     'disable_opt': True, "disable_z_opt": True, 'expected_count_rate': 120,

    #     # 'imaging_laser': green_laser, 'imaging_laser_filter': "nd_0", 'imaging_readout_dur': 5e7,
    #     'imaging_laser': green_laser, 'imaging_laser_filter': "nd_0", 'imaging_readout_dur': 1e7,
    #     # "imaging_laser": green_laser, "imaging_laser_filter": "nd_0.5", "imaging_readout_dur": 5e7,
    #     # "imaging_laser": green_laser, "imaging_laser_filter": "nd_0.5", "imaging_readout_dur": 1e7,
    #     "spin_laser": green_laser, "spin_laser_filter": "nd_0", "spin_pol_dur": 1e6, "spin_readout_dur": 200e3,

    #     "nv-_reionization_laser": green_laser, "nv-_reionization_dur": 1e6, "nv-_reionization_laser_filter": "nd_1.0",
    #     # 'nv-_reionization_laser': green_laser, 'nv-_reionization_dur': 1E5, 'nv-_reionization_laser_filter': 'nd_0.5',
    #     "nv-_prep_laser": green_laser, "nv-_prep_laser_dur": 1e6, "nv-_prep_laser_filter": "nd_0",
    #     # 'nv-_prep_laser': green_laser, 'nv-_prep_laser_dur': 1E4, 'nv-_prep_laser_filter': 'nd_0.5',
    #     "nv0_ionization_laser": red_laser, "nv0_ionization_dur": 75, "nv0_prep_laser": red_laser, "nv0_prep_laser_dur": 75,
    #     "spin_shelf_laser": yellow_laser, "spin_shelf_dur": 0, "spin_shelf_laser_power": 1.0,
    #     # 'spin_shelf_laser': green_laser, 'spin_shelf_dur': 50,
    #     "initialize_laser": green_laser, "initialize_dur": 1e4,
    #     "charge_readout_laser": yellow_laser, "charge_readout_dur": 100e6, "charge_readout_laser_power": 1.0,
    #     # "charge_readout_laser": yellow_laser, "charge_readout_dur": 10e6, "charge_readout_laser_power": 1.0,

    #     'collection_filter': None, 'magnet_angle': None,
    #     'resonance_LOW': 2.878, 'rabi_LOW': 300, 'uwave_power_LOW': 16.5,
    #     'resonance_HIGH': 2.882, 'rabi_HIGH': 400, 'uwave_power_HIGH': 16.5,
    #     }

    # fmt: on

    nv_sig = nv5
    # nv_sig = nvref
    bg_coords = np.array(nv_sig["coords"]) + np.array([0.05, -0.05, 0])
    nv_list = [nv1, nv2, nv3, nv4, nv5]
    # nv_list = [nv2, nv3, nv4, nv5]
    shuffle(nv_list)
    nv_list.append(nv_list[0])

    ### Functions to run

    try:

        # pass

        tool_belt.init_safe_stop()

        # Increasing x moves the image down, increasing y moves the image left
        # with labrad.connect() as cxn:
        #     cxn.cryo_piezos.write_xy(0, -20)

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
        # do_image_sample(nv_sig, apd_indices)

        # nv_sig['imaging_readout_dur'] = 5e7
        # nv3["coords"] = ref_coords
        # do_image_sample(nv_sig, apd_indices)
        # do_image_sample_zoom(nv_sig, apd_indices)
        # do_optimize(nv_sig, apd_indices)
        # nv_sig['imaging_readout_dur'] = 1e8
        # do_stationary_count_bg_subt(nv_sig, apd_indices, bg_coords)
        # do_stationary_count(nv_sig, apd_indices, disable_opt=True)
        # do_determine_standard_readout_params(nv_sig, apd_indices)

        # do_pulsed_resonance(nv_sig, apd_indices, 2.878, 0.020)
        do_rabi(nv_sig, apd_indices, States.LOW, uwave_time_range=[0, 300])
        # do_four_point_esr(nv_sig, apd_indices, States.LOW)

        # temp = 45
        # do_pulsed_resonance_batch(nv_list, apd_indices, temp)
        # do_rabi_batch(nv_list, apd_indices)

    except Exception as exc:
        recipient = "cambria@wisc.edu"
        tool_belt.send_exception_email(email_to=recipient)
        raise exc

    finally:
        
        msg = "Experiment complete!"
        recipient = "cambria@wisc.edu"
        tool_belt.send_email(msg, email_to=recipient)

        # Make sure everything is reset
        tool_belt.reset_cfm()
        tool_belt.reset_safe_stop()
