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


### Major Routines


def do_image_sample(nv_sig, nv_minus_init=False):

    # scan_range = 0.2
    # num_steps = 60

    scan_range = 0.4
    num_steps = 90
    # num_steps = 120

    # scan_range = 0.5
    # num_steps = 90
    # num_steps = 150

    # scan_range = 0.3
    # num_steps = 80

    # scan_range = 1.0
    # num_steps = 180
    
    # scan_range = 2.0
    # num_steps = 90*4

    # scan_range = 3.0
    # num_steps = 90*6
    # num_steps = 90

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

def do_stationary_count_bg_subt(
    nv_sig,
    bg_coords
):

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

    num_reps = 4e3
    max_readouts = [6e6]
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
    num_runs = 32

    uwave_power = 4
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


def do_pulsed_resonance_batch(nv_list, temp):

    num_steps = 51
    
    # num_reps = 2e4
    # num_runs = 32
    
    # num_reps = 1e2
    num_reps = 50
    num_runs = 32
    # num_runs = 8

    uwave_power = 4
    uwave_pulse_dur = 100

    freq_center = cambria_fixed(temp)
    # freq_center = 2.8773
    freq_range = 0.060

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

    # num_reps = 2e4
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
    
    uwave_time_range=[0, 300]
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

    # sample_name = "wu"
    # z_coord = 7
    # ref_coords = [0.437, -0.295, z_coord]
    # ref_coords = np.array(ref_coords)
    
    # nvref = {
    #     'coords': ref_coords, 
    #     'name': '{}-nvref_zfs_vs_t'.format(sample_name),
    #     'disable_opt': False, "disable_z_opt": True, 'expected_count_rate': 10,
    #     'imaging_laser': green_laser, 'imaging_laser_filter': "nd_0", 'imaging_readout_dur': 1e7,
    #     "spin_laser": green_laser, "spin_laser_filter": "nd_0", "spin_pol_dur": 2e3, "spin_readout_dur": 440,
    #     'collection_filter': None, 'magnet_angle': None,
    #     'resonance_LOW': freq, 'rabi_LOW': rabi_per, 'uwave_power_LOW': uwave_power,
    #     }
    # nv1 = {
    #     'coords': ref_coords + np.array([0.174, 0.108, 0]),  
    #     # "coords": [0.467, -0.285, z_coord],
    #     'name': '{}-nv1_zfs_vs_t'.format(sample_name),
    #     'disable_opt': False, "disable_z_opt": True, 'expected_count_rate': 7.0,
    #     'imaging_laser': green_laser, 'imaging_laser_filter': "nd_0", 'imaging_readout_dur': 1e7,
    #     "spin_laser": green_laser, "spin_laser_filter": "nd_0", "spin_pol_dur": 2e3, "spin_readout_dur": 440,
    #     'collection_filter': None, 'magnet_angle': None,
    #     'resonance_LOW': freq, 'rabi_LOW': rabi_per, 'uwave_power_LOW': uwave_power,
    #     }
    # nv2 = {
    #     'coords': ref_coords + np.array([0.157, -0.021, 0]),
    #     # "coords": [0.429, -0.423 , z_coord],
    #     'name': '{}-nv2_zfs_vs_t'.format(sample_name),
    #     'disable_opt': False, "disable_z_opt": True, 'expected_count_rate': 9.0,
    #     'imaging_laser': green_laser, 'imaging_laser_filter': "nd_0", 'imaging_readout_dur': 1e7,
    #     "spin_laser": green_laser, "spin_laser_filter": "nd_0", "spin_pol_dur": 2e3, "spin_readout_dur": 440,
    #     'collection_filter': None, 'magnet_angle': None,
    #     'resonance_LOW': freq, 'rabi_LOW': rabi_per, 'uwave_power_LOW': uwave_power,
    #     }
    # nv3 = {
    #     'coords': ref_coords + np.array([0.052, 0.147, 0]),
    #     # "coords": [0.360, -0.247, z_coord],
    #     'name': '{}-nv3_zfs_vs_t'.format(sample_name),
    #     'disable_opt': False, "disable_z_opt": True, 'expected_count_rate': 8.0,
    #     'imaging_laser': green_laser, 'imaging_laser_filter': "nd_0", 'imaging_readout_dur': 1e7,
    #     "spin_laser": green_laser, "spin_laser_filter": "nd_0", "spin_pol_dur": 2e3, "spin_readout_dur": 440,
    #     'collection_filter': None, 'magnet_angle': None,
    #     'resonance_LOW': freq, 'rabi_LOW': rabi_per, 'uwave_power_LOW': uwave_power,
    #     }
    # nv4 = {
    #     'coords': ref_coords + np.array([-0.237, 0.026, 0]), 
    #     # "coords": [0.051, -0.372, z_coord],
    #     'name': '{}-nv4_zfs_vs_t'.format(sample_name),
    #     'disable_opt': False, "disable_z_opt": True, 'expected_count_rate': 8.0,
    #     'imaging_laser': green_laser, 'imaging_laser_filter': "nd_0", 'imaging_readout_dur': 1e7,
    #     "spin_laser": green_laser, "spin_laser_filter": "nd_0", "spin_pol_dur": 2e3, "spin_readout_dur": 440,
    #     'collection_filter': None, 'magnet_angle': None,
    #     'resonance_LOW': freq, 'rabi_LOW': rabi_per, 'uwave_power_LOW': uwave_power,
    #     }
    # nv5 = {
    #     'coords': ref_coords + np.array([0.074, -0.050, 0]), 
    #     # "coords": [0.511, -0.341, z_coord],
    #     'name': '{}-nv5_zfs_vs_t'.format(sample_name),
    #     'disable_opt': False, "disable_z_opt": True, 'expected_count_rate': 9.0,
    #     'imaging_laser': green_laser, 'imaging_laser_filter': "nd_0", 'imaging_readout_dur': 1e7,
    #     "spin_laser": green_laser, "spin_laser_filter": "nd_0", "spin_pol_dur": 2e3, "spin_readout_dur": 440,
    #     'collection_filter': None, 'magnet_angle': None,
    #     'resonance_LOW': freq, 'rabi_LOW': rabi_per, 'uwave_power_LOW': uwave_power,
    #     }
    
    sample_name = "15micro"
    z_coord = 0
    ref_coords = [0.639, -0.84, z_coord]
    ref_coords = np.array(ref_coords)
    
    nvref = {
        'coords': ref_coords, 
        'name': '{}-nvref_zfs_vs_t'.format(sample_name),
        'disable_opt': True, "disable_z_opt": True, 'expected_count_rate': 800,
        'imaging_laser': green_laser, 'imaging_laser_filter': "nd_0", 'imaging_readout_dur': 1e7,
        "spin_laser": green_laser, "spin_laser_filter": "nd_0", 
        "spin_pol_dur": 3e6, "spin_readout_dur": 5e5,
        "norm_style": NormStyle.POINT_TO_POINT, 'collection_filter': None, 'magnet_angle': None,
        'resonance_LOW': 2.87, 'rabi_LOW': 200, 'uwave_power_LOW': 4.0,
        }
    
    nv1 = copy.deepcopy(nvref)
    nv1["coords"] = ref_coords + np.array([0.144, 0.089, 0])
    nv1["name"] =  f"{sample_name}-nv1_zfs_vs_t"
    nv1["expected_count_rate"] = 1000
    
    nv2 = copy.deepcopy(nvref)
    nv2["coords"] = ref_coords + np.array([0.032, -0.060, 0])
    nv2["name"] =  f"{sample_name}-nv2_zfs_vs_t"
    nv2["expected_count_rate"] = 1000
    
    nv3 = copy.deepcopy(nvref)
    nv3["coords"] = ref_coords + np.array([-0.125, -0.037, 0])
    nv3["name"] =  f"{sample_name}-nv3_zfs_vs_t"
    nv3["expected_count_rate"] = 1000
    
    nv4 = copy.deepcopy(nvref)
    nv4["coords"] = np.array([0.759, -0.501, z_coord])
    nv4["name"] =  f"{sample_name}-nv4_zfs_vs_t"
    nv4["expected_count_rate"] = 2300
    
    nv5 = copy.deepcopy(nvref)
    nv5["coords"] = np.array([0.849, -0.669, z_coord])
    nv5["name"] =  f"{sample_name}-nv5_zfs_vs_t"
    nv5["expected_count_rate"] = 300

    # fmt: on

    # nv_sig = nv1
    nv_sig = nvref
    # bg_coords = np.array(nv_sig["coords"]) + np.array([0.05, -0.05, 0])
    nv_list = [nv1, nv2, nv3]
    # for nv in nv_list:
    #     print(nv["coords"])
    # nv_list = [nv2, nv3, nv4, nv5]
    shuffle(nv_list)
    nv_list.append(nv_list[0])

    ### Functions to run

    try:

        # pass

        tool_belt.init_safe_stop()

        # Increasing x moves the image down, increasing y moves the image left
        # with labrad.connect() as cxn:
        #     cxn.cryo_piezos.write_xy(5, 3)

        # tool_belt.set_drift([0.0, 0.0, 0])  # Totally reset
        # drift = tool_belt.get_drift()
        # tool_belt.set_drift([0.0, 0.0, drift[2]])  # Keep z
        # tool_belt.set_drift([drift[0], drift[1], 0.0])  # Keep xy

        # for z in np.arange(-24, 20, 4):
        # for z in np.arange(10, -10, -5):
        # # while True:
        #     if tool_belt.safe_stop():
        #         break
        #     nv_sig["coords"][2] = int(z)
        #     do_image_sample(nv_sig)
        
        # num_steps = 5
        # step_size = 5
        # locs = [[0,0],[1,0],[1,1],[0,1],[-1,1],[-1,0],[-1,-1],[0,-1],[1,-1],[2,-1],[2,0],[2,1],[2,2],[1,2],[0,2],[-1,2],[-2,2],[-2,1],[-2,0],[-2,-1],[-2,-2],[-1,-2],[0,-2],[1,-2],[2,-2],[2,-1]]
        # for ind in range(num_steps**2):
        #     loc = locs[ind]
        #     loc = [val * step_size for val in loc]
        #     with labrad.connect() as cxn:
        #         cxn.cryo_piezos.write_xy(loc[0],loc[1])
        #     do_image_sample(nv_sig)

        # nv_sig = nvref 
        # nv_sig['imaging_readout_dur'] = 4e7
        # do_image_sample(nv_sig)
        # do_image_sample_zoom(nv_sig)
        # do_optimize(nv_sig)
        # nv_sig['imaging_readout_dur'] = 1e8
        # do_stationary_count_bg_subt(nv_sig, bg_coords)
        # do_stationary_count(nv_sig, disable_opt=True)
        # do_determine_standard_readout_params(nv_sig)

        # do_pulsed_resonance(nv_sig, 2.87, 0.060)
        # do_rabi(nv_sig, States.LOW, uwave_time_range=[0, 300])
        # do_four_point_esr(nv_sig, States.LOW)

        temp = 15
        do_pulsed_resonance_batch(nv_list, temp)
        # do_rabi_batch(nv_list)

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
