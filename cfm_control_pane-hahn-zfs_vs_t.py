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
    
    scan_range = 1.0
    num_steps = 90

    # scan_range = 1.5
    # num_steps = 180
    
    # scan_range = 2.0
    # num_steps = 90*2

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
    
    # Microdiamond
    # num_reps = 1e2
    # num_runs = 32
    # freq_range = 0.060
    
    # Single
    num_reps = 5e4
    num_runs = 32
    freq_range = 0.020
    
    # num_reps = 50
    # num_runs = 8

    uwave_power = 10
    uwave_pulse_dur = 200

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

    num_reps = 0.5e4  # 2e4
    num_runs = 16

    # num_reps = 1e2
    # num_runs = 16

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
    
    # sample_name = "15micro"
    # z_coord = 0
    # ref_coords = [1.186, -0.614, z_coord]
    # ref_coords = np.array(ref_coords)
    
    # nvref = {
    #     'coords': ref_coords, 
    #     'name': '{}-nvref_zfs_vs_t'.format(sample_name),
    #     'disable_opt': True, "disable_z_opt": True, 'expected_count_rate': 800,
    #     'imaging_laser': green_laser, 'imaging_laser_filter': "nd_0", 'imaging_readout_dur': 1e7,
        
    #     # Microdiamond
    #     # "spin_laser": green_laser, "spin_laser_filter": "nd_0.3", 
    #     # "spin_pol_dur": 3e6, "spin_readout_dur": 5e5,
    #     # "spin_laser": green_laser, "spin_laser_filter": "nd_0", 
    #     # "spin_pol_dur": 3e6, "spin_readout_dur": 5e5,
        
    #     # Single
    #     "spin_laser": green_laser, "spin_laser_filter": "nd_0",
    #     "spin_pol_dur": 2e3, "spin_readout_dur": 440,
        
    #     "norm_style": NormStyle.POINT_TO_POINT, 'collection_filter': "nd_0.4", 'magnet_angle': None,
    #     # "norm_style": NormStyle.POINT_TO_POINT, 'collection_filter': "nd_0", 'magnet_angle': None,
    #     'resonance_LOW': 2.87, 'rabi_LOW': 200, 'uwave_power_LOW': 4.0,
    #     }
    
    # nv1 = copy.deepcopy(nvref)
    # nv1["coords"] = ref_coords + np.array([0.144, 0.089, 0])
    # nv1["name"] =  f"{sample_name}-nv1_zfs_vs_t"
    # nv1["expected_count_rate"] = 1000
    
    # nv2 = copy.deepcopy(nvref)
    # nv2["coords"] = ref_coords + np.array([0.032, -0.060, 0])
    # nv2["name"] =  f"{sample_name}-nv2_zfs_vs_t"
    # nv2["expected_count_rate"] = 1000
    
    # nv3 = copy.deepcopy(nvref)
    # nv3["coords"] = ref_coords + np.array([-0.125, -0.037, 0])
    # nv3["name"] =  f"{sample_name}-nv3_zfs_vs_t"
    # nv3["expected_count_rate"] = 1000
    
    # nv4 = copy.deepcopy(nvref)
    # nv4["coords"] = np.array([0.759, -0.501, z_coord])
    # nv4["name"] =  f"{sample_name}-nv4_zfs_vs_t"
    # nv4["expected_count_rate"] = 2300
    
    # nv5 = copy.deepcopy(nvref)
    # nv5["coords"] = np.array([0.849, -0.669, z_coord])
    # nv5["name"] =  f"{sample_name}-nv5_zfs_vs_t"
    # nv5["expected_count_rate"] = 300

    sample_name = "wu"
    z_coord = 6.2
    ref_coords = [-0.194, -0.154, z_coord]
    ref_coords = np.array(ref_coords)
    
    nvref = {
        'coords': ref_coords, 
        'name': '{}-nvref_zfs_vs_t'.format(sample_name),
        'disable_opt': False, "disable_z_opt": True, 'expected_count_rate': 26,
        # 'imaging_laser': green_laser, 'imaging_laser_filter': "nd_0", 'imaging_readout_dur': 0.5e7,
        'imaging_laser': green_laser, 'imaging_laser_filter': "nd_0", 'imaging_readout_dur': 1e7,
        "spin_laser": green_laser, "spin_laser_filter": "nd_0", 
        "spin_pol_dur": 1e3, "spin_readout_dur": 350,
        "norm_style": NormStyle.SINGLE_VALUED, 'collection_filter': None, 'magnet_angle': None,
        'resonance_LOW': 2.87, 'rabi_LOW': 400, 'uwave_power_LOW': 10.0,
        }
        
    nv6 = copy.deepcopy(nvref)
    nv6["coords"] = ref_coords + np.array([0.259, -0.223, 0])
    nv6["name"] =  f"{sample_name}-nv6_zfs_vs_t"
    nv6["expected_count_rate"] = 24
    
    nv7 = copy.deepcopy(nvref)
    nv7["coords"] = ref_coords + np.array([0.189, -0.244, 0])
    nv7["name"] =  f"{sample_name}-nv7_zfs_vs_t"
    nv7["expected_count_rate"] = 26
    
    nv8 = copy.deepcopy(nvref)
    nv8["coords"] = ref_coords + np.array([-0.002, -0.254, 0])
    nv8["name"] =  f"{sample_name}-nv8_zfs_vs_t"
    nv8["expected_count_rate"] = 22
    
    # nv9 = copy.deepcopy(nvref)
    # nv9["coords"] = ref_coords 
    # nv9["name"] =  f"{sample_name}-nv9_zfs_vs_t"
    # nv9["expected_count_rate"] = 20

    nv10 = copy.deepcopy(nvref)
    nv10["coords"] = ref_coords + np.array([-0.058,  0.318, 0])
    nv10["name"] =  f"{sample_name}-nv10_zfs_vs_t"
    nv10["expected_count_rate"] = 22
    
    nv11 = copy.deepcopy(nvref)
    nv11["coords"] = ref_coords + np.array([-0.283,  0.451, 0])
    nv11["name"] =  f"{sample_name}-nv11_zfs_vs_t"
    nv11["expected_count_rate"] = 19
    
    # Region 2

    region_name = "region2"
    z_coord = 5.4
    ref_coords = [-0.146, 0.258, z_coord]
    ref_coords = np.array(ref_coords)
    
    nvref = {
        'coords': ref_coords, 
        'name': f'{sample_name}-nvref_{region_name}',
        'disable_opt': False, "disable_z_opt": True, 'expected_count_rate': 26,
        # 'imaging_laser': green_laser, 'imaging_laser_filter': "nd_0", 'imaging_readout_dur': 0.5e7,
        'imaging_laser': green_laser, 'imaging_laser_filter': "nd_0", 'imaging_readout_dur': 1e7,
        "spin_laser": green_laser, "spin_laser_filter": "nd_0", 
        "spin_pol_dur": 1e3, "spin_readout_dur": 350,
        "norm_style": NormStyle.SINGLE_VALUED, 'collection_filter': None, 'magnet_angle': None,
        'resonance_LOW': 2.87, 'rabi_LOW': 400, 'uwave_power_LOW': 10.0,
        }
        
    nv1 = copy.deepcopy(nvref)
    nv1["coords"] = ref_coords + np.array([0.438, -0.218, 0])
    nv1["name"] =  f"{sample_name}-nv1_{region_name}"
    nv1["expected_count_rate"] = 26

    nv2 = copy.deepcopy(nvref)
    nv2["coords"] = ref_coords + np.array([0.407, 0.048, 0])
    nv2["name"] =  f"{sample_name}-nv2_{region_name}"
    nv2["expected_count_rate"] = 26
    
    nv3 = copy.deepcopy(nvref)
    nv3["coords"] = ref_coords + np.array([0.215, 0.307, 0])
    nv3["name"] =  f"{sample_name}-nv3_{region_name}"
    nv3["expected_count_rate"] = 23

    nv4 = copy.deepcopy(nvref)
    nv4["coords"] = ref_coords 
    nv4["name"] =  f"{sample_name}-nv4_{region_name}"
    nv4["expected_count_rate"] = 19
    
    nv5 = copy.deepcopy(nvref)
    nv5["coords"] = ref_coords + np.array([0.021, 0.477, 0])
    nv5["name"] =  f"{sample_name}-nv5_{region_name}"
    nv5["expected_count_rate"] = 18
    
    # Region 3

    sample_name = "wu"
    region_name = "region3"
    z_coord = 8.0
    ref_coords = [-0.306, 0.235, z_coord]
    ref_coords = np.array(ref_coords)
    
    nv1 = copy.deepcopy(nvref)
    nv1["coords"] = ref_coords + np.array([-0.003, -0.068, 0])
    nv1["name"] =  f"{sample_name}-nv1_{region_name}"
    nv1["expected_count_rate"] = 20

    nv2 = copy.deepcopy(nvref)
    nv2["coords"] = ref_coords
    nv2["name"] =  f"{sample_name}-nv2_{region_name}"
    nv2["expected_count_rate"] = 19
    
    nv3 = copy.deepcopy(nvref)
    nv3["coords"] = ref_coords + np.array([0.012, 0.278, 0])
    nv3["name"] =  f"{sample_name}-nv3_{region_name}"
    nv3["expected_count_rate"] = 19

    nv4 = copy.deepcopy(nvref)
    nv4["coords"] = ref_coords  + np.array([0.092, 0.319, 0])
    nv4["name"] =  f"{sample_name}-nv4_{region_name}"
    nv4["expected_count_rate"] = 19
    
    nv5 = copy.deepcopy(nvref)
    nv5["coords"] = ref_coords + np.array([0.223, 0.353, 0])
    nv5["name"] =  f"{sample_name}-nv5_{region_name}"
    nv5["expected_count_rate"] = 22
    
    # Region 4

    sample_name = "wu"
    region_name = "region4"
    z_coord = 6.9
    ref_coords = [0.004, -0.025, z_coord]
    ref_coords = np.array(ref_coords)
    
    nvref["coords"] = ref_coords
    nvref["name"] =  f"{sample_name}-nvref_{region_name}"
    nvref["expected_count_rate"] = 27
        
    nv1 = copy.deepcopy(nvref)
    nv1["coords"] = ref_coords + np.array([-0.027, -0.09 , 0])
    nv1["name"] =  f"{sample_name}-nv1_{region_name}"
    nv1["expected_count_rate"] = 27

    nv2 = copy.deepcopy(nvref)
    nv2["coords"] = ref_coords
    nv2["name"] =  f"{sample_name}-nv2_{region_name}"
    nv2["expected_count_rate"] = 27
    
    nv3 = copy.deepcopy(nvref)
    nv3["coords"] = ref_coords + np.array([0.156, -0.172, 0])
    nv3["name"] =  f"{sample_name}-nv3_{region_name}"
    nv3["expected_count_rate"] = 25

    nv4 = copy.deepcopy(nvref)
    nv4["coords"] = ref_coords  + np.array([0.087, 0.252, 0])
    nv4["name"] =  f"{sample_name}-nv4_{region_name}"
    nv4["expected_count_rate"] = 22
    
    nv5 = copy.deepcopy(nvref)
    nv5["coords"] = ref_coords + np.array([-0.186,  0.244, 0])
    nv5["name"] =  f"{sample_name}-nv5_{region_name}"
    nv5["expected_count_rate"] = 25
    
    # Region 5

    sample_name = "wu"
    region_name = "region5"
    z_coord = 5.9
    ref_coords = [-0.025, -0.04, z_coord]
    ref_coords = np.array(ref_coords)
    
    nvref["coords"] = ref_coords
    nvref["name"] =  f"{sample_name}-nvref_{region_name}"
    nvref["expected_count_rate"] = 24
        
    nv1 = copy.deepcopy(nvref)
    nv1["coords"] = ref_coords + np.array([0.239, -0.085, 0])
    nv1["name"] =  f"{sample_name}-nv1_{region_name}"
    nv1["expected_count_rate"] = 24

    nv2 = copy.deepcopy(nvref)
    nv2["coords"] = ref_coords + np.array([0.254, -0.012, 0])
    nv2["name"] =  f"{sample_name}-nv2_{region_name}"
    nv2["expected_count_rate"] = 26
    
    nv3 = copy.deepcopy(nvref)
    nv3["coords"] = ref_coords
    nv3["name"] =  f"{sample_name}-nv3_{region_name}"
    nv3["expected_count_rate"] = 21

    nv4 = copy.deepcopy(nvref)
    nv4["coords"] = ref_coords  + np.array([-0.095,  0.029, 0])
    nv4["name"] =  f"{sample_name}-nv4_{region_name}"
    nv4["expected_count_rate"] = 22
    
    nv5 = copy.deepcopy(nvref)
    nv5["coords"] = ref_coords + np.array([-0.151,  0.06, 0])
    nv5["name"] =  f"{sample_name}-nv5_{region_name}"
    nv5["expected_count_rate"] = 20

    # fmt: on

    # nv_sig = nv11
    nv_sig = nvref
    bg_coords = np.array(nv_sig["coords"]) + np.array([0.04, -0.06, 0])
    # nv_list = [nv6, nv7, nv8, nv10, nv11]
    nv_list = [nv1, nv2, nv3, nv4, nv5]
    # nv_list = [nv10, nv11]
    # shuffle(nv_list)
    # nv_list.append(nv_list[0])

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
        # for z in np.arange(7.0, 3.0, -0.4):
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
        # do_image_sample(nv_sig)
        # do_image_sample_zoom(nv_sig)
        
        for nv in nv_list:
            if tool_belt.safe_stop():
                break
            print(nv["coords"])
            # do_image_sample_zoom(nv)
        
        # do_optimize(nv_sig)
        
        # nv_sig['imaging_readout_dur'] = 1e8
        # do_stationary_count(nv_sig, disable_opt=True)
        # nv_sig['imaging_readout_dur'] = 1e8
        # do_stationary_count_bg_subt(nv_sig, bg_coords)
        
        # do_determine_standard_readout_params(nv_sig)

        # do_pulsed_resonance(nv_sig, 2.87, 0.060)
        # do_rabi(nv10, States.LOW, uwave_time_range=[0, 500])
        # do_four_point_esr(nv_sig, States.LOW)

        # shuffle(nv_list)
        # temp = 295
        # do_pulsed_resonance_batch(nv_list, temp)
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
