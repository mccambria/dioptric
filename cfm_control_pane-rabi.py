# -*- coding: utf-8 -*-
"""
Control panel for the PC Rabi

Created on June 16th, 2023

@author: mccambria
"""


### Imports


import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import copy
from utils import tool_belt as tb
from utils import kplotlib as kpl
from utils import positioning as pos
from utils import widefield, common
from majorroutines.widefield import (
    charge_state_histograms,
    image_sample,
    optimize,
    relaxation_interleave,
    resonance,
    rabi,
    optimize_scc,
    scc_snr_check,
    spin_echo,
    xy8,
    calibrate_iq_delay,
)
from utils.constants import LaserKey, NVSpinState

green_laser = "laser_INTE_520"
red_laser = "laser_COBO_638"
yellow_laser = "laser_OPTO_589"
green_laser_dict = {"name": green_laser, "duration": 10e6}
red_laser_dict = {"name": red_laser, "duration": 10e6}
yellow_laser_dict = {"name": yellow_laser, "duration": 35e6}

### Major Routines


def do_widefield_image_sample(nv_sig, num_reps=1):
    nv_sig[LaserKey.IMAGING] = yellow_laser_dict
    image_sample.widefield(nv_sig, num_reps)


def do_scanning_image_sample(nv_sig):
    scan_range = 9
    num_steps = 60
    nv_sig[LaserKey.IMAGING] = green_laser_dict
    image_sample.scanning(nv_sig, scan_range, scan_range, num_steps)


def do_scanning_image_sample_zoom(nv_sig):
    scan_range = 0.2
    num_steps = 30
    nv_sig[LaserKey.IMAGING] = green_laser_dict
    image_sample.scanning(nv_sig, scan_range, scan_range, num_steps)


def do_image_nv_list(nv_list):
    return image_sample.nv_list(nv_list)


def do_image_single_nv(nv_sig):
    nv_sig[LaserKey.IMAGING] = green_laser_dict
    return image_sample.single_nv(nv_sig)


def do_charge_state_histograms(nv_list, num_reps):
    ion_duration = 1000
    return charge_state_histograms.main(nv_list, num_reps, ion_duration=ion_duration)


def do_optimize_green(nv_sig, do_plot=True):
    coords_suffix = tb.get_laser_name(LaserKey.IMAGING)
    ret_vals = optimize.main(
        nv_sig, coords_suffix=coords_suffix, no_crash=True, do_plot=do_plot
    )
    opti_coords = ret_vals[0]
    return opti_coords


def do_optimize_red(nv_sig, do_plot=True):
    laser_key = LaserKey.IONIZATION
    coords_suffix = red_laser
    ret_vals = optimize.main(
        nv_sig,
        laser_key=laser_key,
        coords_suffix=coords_suffix,
        no_crash=True,
        do_plot=do_plot,
    )
    opti_coords = ret_vals[0]
    return opti_coords


def do_optimize_z(nv_sig, do_plot=False):
    optimize.main(nv_sig, no_crash=True, do_plot=do_plot)


def do_optimize_pixel(nv_sig):
    opti_coords = optimize.optimize_pixel(nv_sig, do_plot=True)
    return opti_coords


def do_optimize_widefield_calibration():
    with common.labrad_connect() as cxn:
        optimize.optimize_widefield_calibration(cxn)


def do_optimize_scc(nv_list):
    min_tau = 16
    max_tau = 400
    num_steps = 13
    num_reps = 150
    num_runs = 6
    optimize_scc.main(nv_list, num_steps, num_reps, num_runs, min_tau, max_tau)


def do_scc_snr_check(nv_list):
    num_reps = 100
    scc_snr_check.main(nv_list, num_reps)


def do_calibrate_iq_delay(nv_list):
    min_tau = -200
    max_tau = +200
    num_steps = 26
    num_reps = 150
    num_runs = 6
    calibrate_iq_delay.main(nv_list, num_steps, num_reps, num_runs, min_tau, max_tau)


def do_resonance(nv_list):
    freq_center = 2.87
    freq_range = 0.180
    num_steps = 40
    # num_reps = 80
    # num_runs = 6
    num_reps = 15
    num_runs = 35
    resonance.main(nv_list, num_steps, num_reps, num_runs, freq_center, freq_range)


def do_resonance_zoom(nv_list):
    freq_center = 2.85
    freq_range = 0.05
    num_steps = 20
    num_reps = 150
    num_runs = 6
    resonance.main(nv_list, num_steps, num_reps, num_runs, freq_center, freq_range)


def do_rabi(nv_list):
    min_tau = 16
    max_tau = 160
    num_steps = 19
    num_reps = 15
    num_runs = 35
    rabi.main(nv_list, num_steps, num_reps, num_runs, min_tau, max_tau)


def do_spin_echo(nv_list):
    min_tau = 1e3
    max_tau = 75e3 + min_tau
    num_steps = 26
    # num_reps = 150
    # num_runs = 12
    num_reps = 15
    num_runs = 100
    spin_echo.main(nv_list, num_steps, num_reps, num_runs, min_tau, max_tau)


def do_xy8(nv_list):
    min_tau = 1e3
    max_tau = 1e6 + min_tau
    num_steps = 21
    num_reps = 150
    num_runs = 12
    # num_reps = 20
    # num_runs = 2
    xy8.main(nv_list, num_steps, num_reps, num_runs, min_tau, max_tau)


def do_sq_relaxation(nv_list):
    min_tau = 1e3
    max_tau = 30e6 + min_tau
    num_steps = 21
    num_reps = 150
    num_runs = 12
    relaxation_interleave.sq_relaxation(
        nv_list, num_steps, num_reps, num_runs, min_tau, max_tau
    )


def do_dq_relaxation(nv_list):
    min_tau = 1e3
    max_tau = 18e6 + min_tau
    num_steps = 19
    num_reps = 150
    num_runs = 12
    relaxation_interleave.dq_relaxation(
        nv_list, num_steps, num_reps, num_runs, min_tau, max_tau
    )


def do_opx_constant_ac():
    cxn = common.labrad_connect()
    opx = cxn.QM_opx

    # Microwave test
    # if True:
    #     sig_gen = cxn.sig_gen_STAN_sg394
    #     amp = 9
    #     chan = 10
    # else:
    #     sig_gen = cxn.sig_gen_STAN_sg394_2
    #     amp = 11
    #     chan = 3
    # sig_gen.set_amp(amp)  # 12
    # sig_gen.set_freq(2.87)
    # sig_gen.uwave_on()
    # opx.constant_ac([chan])

    # Camera frame rate test
    seq_args = [500]
    seq_args_string = tb.encode_seq_args(seq_args)
    opx.stream_load("camera_test.py", seq_args_string)
    opx.stream_start()

    # Yellow
    # opx.constant_ac(
    #     [],  # Digital channels
    #     [7],  # Analog channels
    #     [0.25],  # Analog voltages
    #     [0],  # Analog frequencies
    # )
    # Green
    # opx.constant_ac(
    #     [4],  # Digital channels
    #     [6, 4],  # Analog channels
    #     [0.19, 0.19],  # Analog voltages
    #     [110, 110],  # Analog frequencies
    # )
    # Red
    # opx.constant_ac(
    #     [1],  # Digital channels
    #     [2, 3],  # Analog channels
    #     [0.31, 0.31],  # Analog voltages
    #     [75, 75],  # Analog frequencies
    # )
    # Red + green
    # opx.constant_ac(
    #     [1, 4],  # Digital channels
    #     [2, 3, 6, 4],  # Analog channels
    #     [0.32, 0.32, 0.19, 0.19],  # Analog voltages
    #     # [73.8, 76.2, 110.011, 110.845],  # Analog frequencies
    #     [72.6, 77.1, 108.3, 112.002],  # Analog frequencies
    #     # [75, 75, 110, 110],  # Analog frequencies
    # )
    # Red + green
    # opx.constant_ac(
    #     [1, 4],  # Digital channels
    #     [2, 3, 6, 4],  # Analog channels
    #     [0.32, 0.32, 0.19, 0.19],  # Analog voltages
    #     #
    #     # [73.8, 76.2, 110.011, 110.845],  # Analog frequencies
    #     # [72.6, 77.1, 108.3, 112.002],  # Analog frequencies
    #     #
    #     [73.8, 74.6, 110.011, 110.845],  # Analog frequencies
    #     # [72.6, 75.5, 108.3, 112.002],  # Analog frequencies
    #     #
    #     # [75, 75, 110, 110],  # Analog frequencies
    # )
    input("Press enter to stop...")
    # sig_gen.uwave_off()


def compile_speed_test(nv_list):
    cxn = common.labrad_connect()
    pulse_gen = cxn.QM_opx

    seq_file = "resonance.py"
    num_reps = 20
    uwave_index = 1

    seq_args = widefield.get_base_scc_seq_args(nv_list)
    seq_args.append(uwave_index)
    seq_args_string = tb.encode_seq_args(seq_args)

    start = time.time()
    pulse_gen.stream_load(seq_file, seq_args_string, num_reps)
    stop = time.time()
    print(stop - start)

    start = time.time()
    pulse_gen.stream_load(seq_file, seq_args_string, num_reps)
    stop = time.time()
    print(stop - start)


### Run the file


if __name__ == "__main__":
    ### Shared parameters

    green_coords_key = f"coords-{green_laser}"
    red_coords_key = f"coords-{red_laser}"
    pixel_coords_key = "pixel_coords"

    sample_name = "johnson"
    z_coord = 5.17
    magnet_angle = 90

    nv_sig_shell = {
        "coords": [None, None, z_coord],
        "disable_opt": False,
        "disable_z_opt": True,
        "expected_count_rate": None,
        "collection": {"filter": None},
        "magnet_angle": None,
    }

    # region Coords

    nv0 = copy.deepcopy(nv_sig_shell)
    nv0["name"] = f"{sample_name}-nv0_2023_12_13"
    nv0[pixel_coords_key] = [96.609, 84.329]
    nv0[green_coords_key] = [111.291, 108.945]
    nv0[red_coords_key] = [75.461, 74.456]

    nv1 = copy.deepcopy(nv_sig_shell)
    nv1["name"] = f"{sample_name}-nv1_2023_12_13"
    nv1[pixel_coords_key] = [78.018, 96.897]
    nv1[green_coords_key] = [110.517, 109.426]
    nv1[red_coords_key] = [74.738, 74.645]

    nv2 = copy.deepcopy(nv_sig_shell)
    nv2["name"] = f"{sample_name}-nv2_2023_12_13"
    nv2[pixel_coords_key] = [85.419, 58.812]
    nv2[green_coords_key] = [111.01, 108.261]
    nv2[red_coords_key] = [74.977, 73.595]

    nv3 = copy.deepcopy(nv_sig_shell)
    nv3["name"] = f"{sample_name}-nv3_2023_12_13"
    nv3[pixel_coords_key] = [102.672, 52.145]
    nv3[green_coords_key] = [111.402, 107.889]
    nv3[red_coords_key] = [75.755, 73.769]

    nv4 = copy.deepcopy(nv_sig_shell)
    nv4["name"] = f"{sample_name}-nv4_2023_12_13"
    nv4[pixel_coords_key] = [117.814, 64.7]
    nv4[green_coords_key] = [112.058, 108.434]
    nv4[red_coords_key] = [75.877, 73.699]

    nv5 = copy.deepcopy(nv_sig_shell)
    nv5["name"] = f"{sample_name}-nv5_2023_12_13"
    nv5[pixel_coords_key] = [127.2, 66.0]
    nv5[green_coords_key] = [112.415, 108.597]
    nv5[red_coords_key] = [76.083, 73.916]

    nv6 = copy.deepcopy(nv_sig_shell)
    nv6["name"] = f"{sample_name}-nv6_2023_12_13"
    nv6[pixel_coords_key] = [129.038, 76.12]
    nv6[green_coords_key] = [112.329, 108.789]
    nv6[red_coords_key] = [76.259, 74.131]

    nv7 = copy.deepcopy(nv_sig_shell)
    nv7["name"] = f"{sample_name}-nv7_2023_12_13"
    nv7[pixel_coords_key] = [118.074, 97.754]
    nv7[green_coords_key] = [111.945, 109.483]
    nv7[red_coords_key] = [75.818, 74.723]

    nv8 = copy.deepcopy(nv_sig_shell)
    nv8["name"] = f"{sample_name}-nv8_2023_12_13"
    nv8[pixel_coords_key] = [104.124, 163.099]
    nv8[green_coords_key] = [111.623, 111.5]
    nv8[red_coords_key] = [75.403, 76.555]

    nv9 = copy.deepcopy(nv_sig_shell)
    nv9["name"] = f"{sample_name}-nv9_2023_12_13"
    nv9[pixel_coords_key] = [52.88, 142.52]
    nv9[green_coords_key] = [109.632, 110.92]
    nv9[red_coords_key] = [74.003, 76.007]

    # endregion

    # nv_sig = nv8
    # nv_list = [nv_sig]
    nv_list = [nv0, nv1, nv2, nv3, nv4, nv5, nv6, nv7, nv8, nv9]
    # nv_list = [nv6, nv7, nv8, nv9]
    nv_sig = nv_list[0]

    ### Functions to run

    email_recipient = "mccambria@berkeley.edu"
    do_email = False
    try:
        # pass

        kpl.init_kplotlib()
        tb.init_safe_stop()

        # Make sure the OPX config is up to date
        # cxn = common.labrad_connect()
        # opx = cxn.QM_opx
        # opx.update_config()

        # time.sleep(3)
        # mag_rot_server = tb.get_server_magnet_rotation()
        # mag_rot_server.set_angle(magnet_angle)
        # print(mag_rot_server.get_angle())

        # widefield.reset_all_drift()
        # widefield.set_pixel_drift([+14, -5])
        # widefield.set_all_scanning_drift_from_pixel_drift()

        # pos.set_xyz_on_nv(nv_sig)

        # for z in np.linspace(5.2, 4.9, 11):
        #     nv_sig["coords"][2] = z
        #     do_widefield_image_sample(nv_sig, 100)
        # do_widefield_image_sample(nv_sig, 100)
        # do_optimize_pixel(nv_sig)

        # do_scc_snr_check(nv_list)

        # do_resonance(nv_list)
        # do_resonance_zoom(nv_list)
        # do_rabi(nv_list)
        # do_sq_relaxation(nv_list)
        # do_dq_relaxation(nv_list)
        do_spin_echo(nv_list)
        # do_xy8(nv_list)

        ### Infrequent stuff down here

        # Full optimize
        # opti_coords_list = []
        # for nv in nv_list:
        #     widefield.reset_all_drift()
        #     #
        #     # opti_coords = do_optimize_pixel(nv)
        #     #
        #     # widefield.set_nv_scanning_coords_from_pixel_coords(nv, green_laser)
        #     # opti_coords = do_optimize_green(nv)
        #     #
        #     # widefield.set_nv_scanning_coords_from_pixel_coords(nv, red_laser)
        #     opti_coords = do_optimize_red(nv)
        #     #
        #     opti_coords_list.append(opti_coords)
        #     widefield.reset_all_drift()
        # for opti_coords in opti_coords_list:
        #     r_opti_coords = [round(el, 3) for el in opti_coords]
        #     print(r_opti_coords)

        # do_charge_state_histograms(nv_list, 1000)
        # do_optimize_z(nv_sig)
        # do_opx_constant_ac()
        # do_calibrate_iq_delay(nv_list)
        # do_image_nv_list(nv_list)
        # do_optimize_scc(nv_list)
        # compile_speed_test(nv_list)

    except Exception as exc:
        if do_email:
            recipient = email_recipient
            tb.send_exception_email(email_to=recipient)
        else:
            print(exc)
        raise exc

    finally:
        if do_email:
            msg = "Experiment complete!"
            recipient = email_recipient
            tb.send_email(msg, email_to=recipient)

        print()
        print("Routine complete")

        # Maybe necessary to make sure we don't interrupt a sequence prematurely
        # tb.poll_safe_stop()

        # Make sure everything is reset
        tb.reset_cfm()
        cxn = common.labrad_connect()
        cxn.disconnect()
        plt.show(block=True)
        tb.reset_safe_stop()
