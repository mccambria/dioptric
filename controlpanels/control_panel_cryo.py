# -*- coding: utf-8 -*-
"""This file contains functions to control the CFM. Just change the function call
in the main section at the bottom of this file and run the file. Shared or
frequently changed parameters are in the __main__ body and relatively static
parameters are in the function definitions.

Created on Oct 7th, 2025

@author: chemistatcode
@author: Saroj B Chand
@author: ericvin
@author: mccambria
"""


# region Imports and constants

import copy
import time
import labrad
import numpy as np
from utils import positioning as pos
from utils import kplotlib as kpl
# import majorroutines.confocal.determine_standard_readout_params as determine_standard_readout_params
# import majorroutines.confocal.g2_measurement as g2_measurement
import majorroutines.confocal.confocal_image_sample as image_sample
# import majorroutines.confocal.image_sample as image_sample

# import majorroutines.confocal.optimize_magnet_angle as optimize_magnet_angle
# import majorroutines.confocal.pulsed_resonance as pulsed_resonance
# import majorroutines.confocal.confocal_rabi as rabi

# import majorroutines.confocal.ramsey as ramsey
# import majorroutines.confocal.resonance as resonance
# import majorroutines.confocal.spin_echo as spin_echo
import majorroutines.confocal.confocal_stationary_count as stationary_count

# import majorroutines.confocal.t1_dq_main as t1_dq_main
import majorroutines.targeting as targeting
import utils.tool_belt as tool_belt
from utils.constants import Axes, CoordsKey, NVSig, VirtualLaserKey
from majorroutines.confocal.confocal_1D_scan import confocal_scan_1d

# from utils.tool_belt import States

# endregion
# region Routines


def do_image_sample(
    nv_sig,
):
    # scan_range = 0.2
    # num_steps = 60

    scan_range = 0.5 #voltage #cryo image conversion: 37um/V; step size: x,y,z=40V (was 30)
    num_steps = 90

    # For now we only support square scans so pass scan_range twice
    image_sample.confocal_scan(
        nv_sig,
        scan_range,
        scan_range,
        num_steps,
    )

# def do_image_sample_Hahn( # From Hahn control panel, should not work with current version of image_sample
#     nv_sig,
#     nv_minus_initialization=False,
#     cbarmin=None,
#     cbarmax=None,
# ):
#     # scan_range = 0.2
#     # num_steps = 60

#     scan_range = 0.5
#     num_steps = 90

#     # For now we only support square scans so pass scan_range twice
#     image_sample.main(
#         nv_sig,
#         scan_range,
#         scan_range,
#         num_steps,
#         nv_minus_initialization=nv_minus_initialization,
#         cmin=cbarmin,
#         cmax=cbarmax,
#     )

def do_1D_scan(nv_sig):
    scan_range = 0.4  # voltage range for X axis
    num_steps = 60   # number of points along X
    
    # 1D scan function
    counts, x_positions = confocal_scan_1d(
        nv_sig,
        scan_range,
        num_steps,
    )
    
    return counts, x_positions

def do_image_sample_zoom(nv_sig):
    scan_range = 0.1 #TOO ZOOM FOR CURRENT SET-UP
    num_steps = 30

    image_sample.confocal_scan(
        nv_sig,
        scan_range,
        scan_range,
        num_steps,
    )


def do_optimize(nv_sig):
    targeting.main(
        nv_sig,
        set_to_opti_coords=False,
        save_data=True,
        plot_data=True,
    )


def do_stationary_count(
    nv_sig,
    disable_opt=None,
):
    run_time = 3 * 60 * 10**9  # ns

    stationary_count.main(
        nv_sig,
        run_time,
        disable_opt=disable_opt,
        # nv_minus_initialization=nv_minus_initialization,
        # nv_zero_initialization=nv_zero_initialization,
    )


# def do_g2_measurement(nv_sig, apd_a_index, apd_b_index):
#     run_time = 60 * 10  # s
#     diff_window = 200  # ns

#     g2_measurement.main(nv_sig, run_time, diff_window, apd_a_index, apd_b_index)


# def do_resonance(nv_sig, freq_center=2.87, freq_range=0.2):
#     num_steps = 51
#     num_runs = 20
#     uwave_power = -5.0

#     resonance.main(
#         nv_sig,
#         freq_center,
#         freq_range,
#         num_steps,
#         num_runs,
#         uwave_power,
#         state=States.HIGH,
#     )


# def do_resonance_state(nv_sig, state):
#     freq_center = nv_sig["resonance_{}".format(state.name)]
#     uwave_power = -5.0

#     # freq_range = 0.200
#     # num_steps = 51
#     # num_runs = 2

#     # Zoom
#     freq_range = 0.05
#     num_steps = 51
#     num_runs = 10

#     resonance.main(
#         nv_sig,
#         freq_center,
#         freq_range,
#         num_steps,
#         num_runs,
#         uwave_power,
#     )


# def do_determine_standard_readout_params(nv_sig):
#     num_reps = 1e5
#     max_readouts = [1e6]
#     filters = ["nd_0"]
#     state = States.LOW

#     determine_standard_readout_params.main(
#         nv_sig,
#         num_reps,
#         max_readouts,
#         filters=filters,
#         state=state,
#     )


# def do_pulsed_resonance(nv_sig, freq_center=2.87, freq_range=0.2):
#     num_steps = 51

#     num_reps = 2e4
#     num_runs = 16

#     # num_reps = 1e3
#     # num_runs = 8

#     uwave_power = 16.5
#     uwave_pulse_dur = 400

#     pulsed_resonance.main(
#         nv_sig,
#         freq_center,
#         freq_range,
#         num_steps,
#         num_reps,
#         num_runs,
#         uwave_power,
#         uwave_pulse_dur,
#     )


# def do_pulsed_resonance_state(nv_sig, state):
#     freq_range = 0.020
#     num_steps = 51
#     num_reps = 2e4
#     num_runs = 16

#     # Zoom
#     # freq_range = 0.035
#     # # freq_range = 0.120
#     # num_steps = 51
#     # num_reps = 8000
#     # num_runs = 3

#     composite = False

#     res, _ = pulsed_resonance.state(
#         nv_sig,
#         state,
#         freq_range,
#         num_steps,
#         num_reps,
#         num_runs,
#         composite,
#     )
#     nv_sig["resonance_{}".format(state.name)] = res
#     return res


# def do_scc_pulsed_resonance(nv_sig, state):
#     opti_nv_sig = nv_sig
#     freq_center = nv_sig["resonance_{}".format(state)]
#     uwave_power = nv_sig["uwave_power_{}".format(state)]
#     uwave_pulse_dur = tool_belt.get_pi_pulse_dur(nv_sig["rabi_{}".format(state)])
#     freq_range = 0.020
#     num_steps = 25
#     num_reps = int(1e3)
#     num_runs = 5

#     scc_pulsed_resonance.main(
#         nv_sig,
#         opti_nv_sig,
#         freq_center,
#         freq_range,
#         num_steps,
#         num_reps,
#         num_runs,
#         uwave_power,
#         uwave_pulse_dur,
#     )


# def do_determine_charge_readout_params(nv_sig):
#     readout_durs = [10e6]
#     readout_durs = [int(el) for el in readout_durs]
#     max_readout_dur = max(readout_durs)

#     readout_powers = [1.0]
#     readout_powers = [round(val, 3) for val in readout_powers]

#     num_reps = 1000

#     determine_charge_readout_params.main(
#         nv_sig,
#         num_reps,
#         readout_powers,
#         max_readout_dur,
#         plot_readout_durs=readout_durs,
#     )


# def do_optimize_magnet_angle(nv_sig):
#     angle_range = [0, 150]
#     num_angle_steps = 6
#     freq_center = 2.87
#     freq_range = 0.200
#     num_freq_steps = 51
#     num_freq_runs = 15

#     # Pulsed
#     uwave_power = 16.5
#     uwave_pulse_dur = 85
#     num_freq_reps = 5000

#     # CW
#     # uwave_power = -5.0
#     # uwave_pulse_dur = None
#     # num_freq_reps = None

#     optimize_magnet_angle.main(
#         nv_sig,
#         angle_range,
#         num_angle_steps,
#         freq_center,
#         freq_range,
#         num_freq_steps,
#         num_freq_reps,
#         num_freq_runs,
#         uwave_power,
#         uwave_pulse_dur,
#     )


# def do_rabi(nv_sig):
#     num_steps = 51
#     num_reps = 2e4
#     num_runs = 16
#     min_tau = 8
#     max_tau = 400
#     uwave_ind_list = [0, 1]

#     rabi.main(
#         nv_sig,
#         num_steps,
#         num_reps,
#         num_runs,
#         min_tau,
#         max_tau,
#         uwave_ind_list,
#     )
#     # nv_sig["rabi_{}".format(state.name)] = period


# def do_t1_dq(nv_sig):
#     # T1 experiment parameters, formatted:
#     # [[init state, read state], relaxation_time_range, num_steps, num_reps]
#     num_runs = 500
#     num_reps = 1000
#     num_steps = 12
#     min_tau = 10e3
#     max_tau_omega = int(18e6)
#     max_tau_gamma = int(8.5e6)
#     # fmt: off
#     t1_exp_array = np.array(
#         [[[States.ZERO, States.HIGH], [min_tau, max_tau_omega], num_steps, num_reps, num_runs],
#         [[States.ZERO, States.ZERO], [min_tau, max_tau_omega], num_steps, num_reps, num_runs],
#         [[States.ZERO, States.HIGH], [min_tau, max_tau_omega // 3], num_steps, num_reps, num_runs],
#         [[States.ZERO, States.ZERO], [min_tau, max_tau_omega // 3], num_steps, num_reps, num_runs],
#         [[States.LOW, States.HIGH], [min_tau, max_tau_gamma], num_steps, num_reps, num_runs],
#         [[States.LOW, States.LOW], [min_tau, max_tau_gamma], num_steps, num_reps, num_runs],
#         [[States.LOW, States.HIGH], [min_tau, max_tau_gamma // 3], num_steps, num_reps, num_runs],
#         [[States.LOW, States.LOW], [min_tau, max_tau_gamma // 3], num_steps, num_reps, num_runs]],
#         dtype=object,
#     )
#     # fmt: on

#     t1_dq_main.main(nv_sig, t1_exp_array, num_runs)


# def do_ramsey(nv_sig):
#     detuning = 2.5  # MHz
#     precession_time_range = [0, 4 * 10**3]
#     num_steps = 151
#     num_reps = 3 * 10**5
#     num_runs = 1

#     ramsey.main(
#         nv_sig,
#         detuning,
#         precession_time_range,
#         num_steps,
#         num_reps,
#         num_runs,
#     )


# def do_spin_echo(nv_sig):
#     # T2* in nanodiamond NVs is just a couple us at 300 K
#     # In bulk it"s more like 100 us at 300 K
#     max_time = 120  # us
#     num_steps = max_time  # 1 point per us
#     precession_time_range = [1e3, max_time * 10**3]
#     num_reps = 4e3
#     num_runs = 20

#     state = States.LOW

#     angle = spin_echo.main(
#         nv_sig,
#         precession_time_range,
#         num_steps,
#         num_reps,
#         num_runs,
#         state,
#     )
#     return angle


# endregion

def get_sample_name() -> str:
    sample = "Wu" #lovelace
    return sample

# region main

if __name__ == "__main__":
    ### Shared parameters

    green_laser = "laser_COBO_520"
    # yellow_laser = "laserglow_589"
    # red_laser = "cobolt_638"

    # fmt: off
     #lovelace"
    # nv_sig = {
    #     "coords": [0.240, -0.426, 1], "name": "{}-nv8_2022_11_14".format(sample_name),
    #     "disable_opt": False, "disable_z_opt": True, "expected_count_rate": 13,

    #     "imaging_laser": green_laser, "imaging_laser_filter": "nd_0", "imaging_readout_dur": 1e7,
    #     "spin_laser": green_laser, "spin_laser_filter": "nd_0", "spin_pol_dur": 2e3, "spin_readout_dur": 440,

    #     "nv-_reionization_laser": green_laser, "nv-_reionization_dur": 1e6, "nv-_reionization_laser_filter": "nd_1.0",
    #     "nv-_prep_laser": green_laser, "nv-_prep_laser_dur": 1e6, "nv-_prep_laser_filter": "nd_0",
    #     "nv0_ionization_laser": red_laser, "nv0_ionization_dur": 75, "nv0_prep_laser": red_laser, "nv0_prep_laser_dur": 75,
    #     "spin_shelf_laser": yellow_laser, "spin_shelf_dur": 0, "spin_shelf_laser_power": 1.0,
    #     "initialize_laser": green_laser, "initialize_dur": 1e4,
    #     "charge_readout_laser": yellow_laser, "charge_readout_dur": 100e6, "charge_readout_laser_power": 1.0,

    #     "collection_filter": None, "magnet_angle": None,
    #     "resonance_LOW": 2.878, "rabi_LOW": 400, "uwave_power_LOW": 16.5,
    #     "resonance_HIGH": 2.882, "rabi_HIGH": 400, "uwave_power_HIGH": 16.5,
    #     }
    # fmt: on

    # coords: SAMPLE (piezo) xyz 
    # current step rate: 30.0V XYZ
    # region Postion and Time Control
    sample_xy = [0.0,0.0] # piezo XY voltage input (1.0=1V) (not coordinates, relative)
    coord_z = 0  # piezo z voltage (0 is the set midpoint, absolute) (negative is closer to smaple, move unit steps in sample; 37 is good surface focus with bs for Lovelace; 20 is good for dye)
    pixel_xy = [0.0,0.0]   # galvo ref

    nv_sig = NVSig(
        name=f"({get_sample_name()})",
        coords={
            CoordsKey.SAMPLE: sample_xy,
            CoordsKey.Z: coord_z,
            CoordsKey.PIXEL: pixel_xy, #galvo 
        },
        disable_opt=False,
        disable_z_opt=True,
        expected_counts=13,
        pulse_durations={
            VirtualLaserKey.IMAGING: int(10e6), # readout is in ns (5e6 = 5ms)
            VirtualLaserKey.CHARGE_POL: int(1e4),
            VirtualLaserKey.SPIN_POL: 2000,
            VirtualLaserKey.SINGLET_DRIVE: 300,  # placeholder
        },
    )
    # endregion
    ### Routines to execute

    try:
        tool_belt.init_safe_stop()
        # tool_belt.set_drift([0.0, 0.0, 0.0])  # Totally reset
        # drift = tool_belt.get_drift()
        # tool_belt.set_drift([0.0, 0.0, drift[2]])  # Keep z
        # tool_belt.set_drift([drift[0], drift[1], 0.0])  # Keep xy
        
        pos.set_xyz_on_nv(nv_sig) # Hahn omits this line


        # region 1D scan

        # do_1D_scan(nv_sig)
        # z_range = np.linspace(750, 700, 51)
        # for z in z_range:
        #     nv_sig.coords[CoordsKey.Z] = z
        #     pos.set_xyz_on_nv(nv_sig)
        #     do_1D_scan(nv_sig)

        # endregion 1D scan

        # region Image sample

        # do_image_sample(nv_sig)
        # do_image_sample_zoom(nv_sig, nv_minus_initialization=True)
        # Z AXIS PIEZO SCAN
        #z_range = np.linspace(0, 10, 22)
        #for z in z_range:
            #nv_sig.coords[CoordsKey.Z] = z
            #pos.set_xyz_on_nv(nv_sig)
            #do_image_sample(nv_sig)

        # do_image_sample_zoom(nv_sig)
        # do_image_sample(nv_sig, nv_minus_initialization=True)
        # do_image_sample_zoom(nv_sig, nv_minus_initialization=True)
        #end region Image sample

        # do_optimize(nv_sig)
        # nv_sig["imaging_readout_dur"] = 5e7-
        
        #Hahn control panel image sample
        # for z in np.arange(0, -100, -5):
        # # while True:
        #     if tool_belt.safe_stop():
        #         break
        #     nv_sig["coords"][2] = int(z)
        # do_image_sample(nv_sig)
        # nv_sig["imaging_readout_dur"] = 5e7
        # do_image_sample_Hahn(nv_sig)
        # do_image_sample_Hahn(nv_sig, nv_minus_initialization=True)

        # # region Stationary count
        do_stationary_count(nv_sig, disable_opt=True) #Note there is a slow response time w/ the APD
        # do_stationary_count(nv_sig, disable_opt=True, nv_minus_initialization=True)
        # do_stationary_count(nv_sig, disable_opt=True, nv_zero_initialization=True)
        # endregion Stationary count

        # do_resonance(nv_sig, 2.87, 0.200)
        # do_resonance_state(nv_sig , States.LOW)
        # do_resonance_state(nv_sig, States.HIGH)
        # do_pulsed_resonance(nv_sig, 2.87, 0.200)
        # do_pulsed_resonance_state(nv_sig, States.LOW)
        # do_pulsed_resonance_state(nv_sig, States.HIGH)
        # do_rabi(nv_sig)
        # do_rabi(nv_sig, States.HIGH, uwave_time_range=[0, 400])
        # do_spin_echo(nv_sig)
        # do_g2_measurement(nv_sig, 0, 1)
        # do_determine_standard_readout_params(nv_sig)

        # SCC characterization
        # do_determine_charge_readout_params(nv_sig,nbins=200,nreps=100)
        # do_scc_pulsed_resonance(nv_sig)

    ### Error handling and wrap-up

    except Exception as exc:
        recipient = "cmreiter@berkeley.edu"
        tool_belt.send_exception_email(email_to=recipient)
        raise exc
    finally:
        tool_belt.reset_cfm()
        tool_belt.reset_safe_stop()
        kpl.show(block=True)

# region end main