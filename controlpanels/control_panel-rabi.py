# -*- coding: utf-8 -*-
"""
Control panel for the PC Rabi

Created on June 16th, 2023

@author: mccambria
"""


### Imports

import copy
import os
import sys
import time

import keyboard
import matplotlib.pyplot as plt
import numpy as np

from majorroutines.widefield import (
    calibrate_iq_delay,
    charge_state_histograms,
    correlation_test,
    image_sample,
    optimize_scc,
    rabi,
    ramsey,
    relaxation_interleave,
    resonance,
    scc_snr_check,
    spin_echo,
    targeting,
    xy8,
)
from utils import common, widefield
from utils import kplotlib as kpl
from utils import positioning as pos
from utils import tool_belt as tb
from utils.constants import NVSig, NVSpinState, VirtualLaserKey

green_laser = "laser_INTE_520"
red_laser = "laser_COBO_638"
yellow_laser = "laser_OPTO_589"
green_laser_dict = {"name": green_laser, "duration": 10e6}
red_laser_dict = {"name": red_laser, "duration": 10e6}
yellow_laser_dict = {"name": yellow_laser, "duration": 35e6}

### Major Routines


def do_widefield_image_sample(nv_sig, num_reps=1):
    nv_sig[VirtualLaserKey.IMAGING] = yellow_laser_dict
    image_sample.widefield_image(nv_sig, num_reps)


def do_scanning_image_sample(nv_sig):
    scan_range = 4
    num_steps = 60
    nv_sig[VirtualLaserKey.IMAGING] = green_laser_dict
    image_sample.scanning(nv_sig, scan_range, scan_range, num_steps)


def do_scanning_image_sample_zoom(nv_sig):
    scan_range = 0.2
    num_steps = 30
    nv_sig[VirtualLaserKey.IMAGING] = green_laser_dict
    image_sample.scanning(nv_sig, scan_range, scan_range, num_steps)


def do_image_nv_list(nv_list):
    return image_sample.nv_list(nv_list)


def do_image_single_nv(nv_sig):
    nv_sig[VirtualLaserKey.IMAGING] = green_laser_dict
    return image_sample.single_nv(nv_sig)


def do_charge_state_histograms(nv_list, num_reps):
    ion_duration = 1000
    return charge_state_histograms.main(nv_list, num_reps, ion_duration=ion_duration)


def do_optimize_green(nv_sig, do_plot=True):
    coords_suffix = tb.get_physical_laser_name(VirtualLaserKey.IMAGING)
    ret_vals = targeting.main(
        nv_sig, coords_suffix=coords_suffix, no_crash=True, do_plot=do_plot
    )
    opti_coords = ret_vals[0]
    return opti_coords


def do_optimize_red(nv_sig, do_plot=True):
    laser_key = VirtualLaserKey.IONIZATION
    coords_suffix = red_laser
    ret_vals = targeting.main(
        nv_sig,
        laser_key=laser_key,
        coords_suffix=coords_suffix,
        no_crash=True,
        do_plot=do_plot,
    )
    opti_coords = ret_vals[0]
    return opti_coords


def do_optimize_z(nv_sig, do_plot=False):
    targeting.main(nv_sig, no_crash=True, do_plot=do_plot, axes_to_optimize=[2])


def do_optimize_pixel(nv_sig):
    opti_coords = targeting.optimize_pixel(nv_sig, do_plot=True)
    return opti_coords


def do_optimize_loop(nv_list, coords_suffix, scanning_from_pixel=False):
    repr_nv_sig = widefield.get_repr_nv_sig(nv_list)

    opti_coords_list = []
    for nv in nv_list:
        # Pixel coords
        if coords_suffix is None:
            imaging_laser = tb.get_physical_laser_name(VirtualLaserKey.IMAGING)
            if scanning_from_pixel:
                widefield.set_nv_scanning_coords_from_pixel_coords(nv, imaging_laser)
            opti_coords = do_optimize_pixel(nv)
            # widefield.reset_all_drift()

        # Scanning coords
        else:
            if scanning_from_pixel:
                widefield.set_nv_scanning_coords_from_pixel_coords(nv, coords_suffix)

            if coords_suffix == green_laser:
                opti_coords = do_optimize_green(nv)
            elif coords_suffix == red_laser:
                opti_coords = do_optimize_red(nv)

            # Adjust for the drift that may have occurred since beginning the loop
            do_optimize_pixel(repr_nv_sig)
            drift = pos.get_drift(coords_suffix)
            drift = [-1 * el for el in drift]
            opti_coords = pos.adjust_coords_for_drift(opti_coords, drift=drift)

        opti_coords_list.append(opti_coords)

    # Report back
    for opti_coords in opti_coords_list:
        r_opti_coords = [round(el, 3) for el in opti_coords]
        print(f"{r_opti_coords},")


def do_optimize_widefield_calibration():
    with common.labrad_connect() as cxn:
        targeting.optimize_widefield_calibration(cxn)


def do_optimize_scc(nv_list):
    min_tau = 16
    max_tau = 400
    num_steps = 13
    num_reps = 15
    num_runs = 50
    optimize_scc.main(nv_list, num_steps, num_reps, num_runs, min_tau, max_tau)


def do_scc_snr_check(nv_list):
    num_reps = 1000
    scc_snr_check.main(nv_list, num_reps)


def do_calibrate_iq_delay(nv_list):
    min_tau = -100
    max_tau = +100
    num_steps = 21
    num_reps = 10
    num_runs = 25
    calibrate_iq_delay.main(
        nv_list, num_steps, num_reps, num_runs, min_tau, max_tau, i_or_q=False
    )


def do_resonance(nv_list):
    freq_center = 2.87
    freq_range = 0.100
    num_steps = 40
    num_reps = 10
    # num_runs = 30
    num_runs = 100
    resonance.main(nv_list, num_steps, num_reps, num_runs, freq_center, freq_range)


def do_resonance_zoom(nv_list):
    freq_center = 2.848
    freq_range = 0.05
    num_steps = 20
    num_reps = 15
    num_runs = 30
    resonance.main(nv_list, num_steps, num_reps, num_runs, freq_center, freq_range)


def do_rabi(nv_list):
    min_tau = 16
    max_tau = 240 + min_tau
    # num_steps = 21
    # num_reps = 15
    # num_runs = 30
    num_steps = 31
    num_reps = 20
    num_runs = 50
    uwave_ind = 0
    rabi.main(
        nv_list,
        num_steps,
        num_reps,
        num_runs,
        min_tau,
        max_tau,
        uwave_ind_list=uwave_ind,
    )


def do_spin_echo(nv_list):
    min_tau = 100
    max_tau = 200e3 + min_tau

    # Zooms
    # min_tau = 100
    # min_tau = 83.7e3
    # min_tau = 167.4e3
    # max_tau = 2e3 + min_tau

    # min_tau = 100
    # max_tau = 15e3 + min_tau

    num_steps = 51

    # num_reps = 150
    # num_runs = 12
    num_reps = 10
    num_runs = 400
    # num_runs = 2

    spin_echo.main(nv_list, num_steps, num_reps, num_runs, min_tau, max_tau)


def do_spin_echo_short(nv_list):
    # min_tau = 100
    # min_tau = 83.7e3
    min_tau = 167.4e3
    max_tau = 2e3 + min_tau
    num_steps = 51
    num_reps = 10
    num_runs = 400
    spin_echo.main(nv_list, num_steps, num_reps, num_runs, min_tau, max_tau)


def do_spin_echo_medium(nv_list):
    min_tau = 100
    max_tau = 15e3 + min_tau
    num_steps = 51
    num_reps = 10
    num_runs = 400
    spin_echo.main(nv_list, num_steps, num_reps, num_runs, min_tau, max_tau)


def do_spin_echo_long(nv_list):
    min_tau = 100
    max_tau = 200e3 + min_tau
    num_steps = 51
    num_reps = 10
    num_runs = 400
    spin_echo.main(nv_list, num_steps, num_reps, num_runs, min_tau, max_tau)


def do_ramsey(nv_list):
    min_tau = 0
    # max_tau = 2000 + min_tau
    max_tau = 3200 + min_tau
    detuning = 3
    # num_steps = 21
    # num_reps = 15
    # num_runs = 30
    num_steps = 101
    num_reps = 10
    num_runs = 400
    uwave_ind = 0
    ramsey.main(nv_list, num_steps, num_reps, num_runs, min_tau, max_tau, detuning)


def do_xy8(nv_list):
    min_tau = 1e3
    max_tau = 1e6 + min_tau
    num_steps = 21
    num_reps = 150
    num_runs = 12
    # num_reps = 20
    # num_runs = 2
    xy8.main(nv_list, num_steps, num_reps, num_runs, min_tau, max_tau)


def do_correlation_test(nv_list):
    min_tau = 16
    max_tau = 72
    num_steps = 15

    num_reps = 10
    num_runs = 400

    # MCC
    # min_tau = 16
    # max_tau = 240 + min_tau
    # num_steps = 31
    # num_reps = 20
    # num_runs = 30

    # anticorrelation_inds = None
    anticorrelation_inds = [0, 5]

    correlation_test.main(
        nv_list, num_steps, num_reps, num_runs, min_tau, max_tau, anticorrelation_inds
    )


def do_sq_relaxation(nv_list):
    min_tau = 1e3
    max_tau = 20e6 + min_tau
    num_steps = 21
    num_reps = 10
    num_runs = 400
    relaxation_interleave.sq_relaxation(
        nv_list, num_steps, num_reps, num_runs, min_tau, max_tau
    )


def do_dq_relaxation(nv_list):
    min_tau = 1e3
    max_tau = 15e6 + min_tau
    num_steps = 21
    num_reps = 10
    num_runs = 400
    relaxation_interleave.dq_relaxation(
        nv_list, num_steps, num_reps, num_runs, min_tau, max_tau
    )


def do_opx_square_wave():
    cxn = common.labrad_connect()
    opx = cxn.QM_opx

    # Yellow
    # opx.square_wave(
    #     [],  # Digital channels
    #     [7],  # Analog channels
    #     [1.0],  # Analog voltages
    #     1000,  # Period (ns)
    # )
    # Camera trigger
    opx.square_wave(
        [5],  # Digital channels
        [],  # Analog channels
        [],  # Analog voltages
        1000,  # Period (ns)
    )
    input("Press enter to stop...")
    # sig_gen.uwave_off()


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
    # seq_args = [500]
    # seq_args_string = tb.encode_seq_args(seq_args)
    # opx.stream_load("camera_test.py", seq_args_string)
    # opx.stream_start()

    # Yellow
    # opx.constant_ac(
    #     [],  # Digital channels
    #     [7],  # Analog channels
    #     [1.1],  # Analog voltages
    #     [0],  # Analog frequencies
    # )
    # Green
    opx.constant_ac(
        [4],  # Digital channels
        [3, 4],  # Analog channels
        [0.19, 0.19],  # Anal og voltages
        [110, 110],  # Analog frequencies
    )
    # Red
    # freqs = [65, 75, 85]
    # # freqs = [73, 75, 77]
    # while not keyboard.is_pressed("q"):
    #     for freq in freqs:
    #         opx.constant_ac(
    #             [1],  # Digital channels
    #             [2, 6],  # Analog channels
    #             [0.17, 0.17],  # Analog voltages
    #             [
    #                 75,
    #                 freq,
    #             ],  # Analog frequencies                                                                                                                                                                              uencies
    #         )
    #         time.sleep(0.5)
    #     opx.halt()
    # opx.constant_ac(
    #     [1],  # Digital channels
    #     [2, 6],  # Analog channels
    #     [0.17, 0.17],  # Analog voltages
    #     [
    #         75,
    #         75,
    #     ],  # Analog frequencies                                                                                                                                                                              uencies
    # )
    # green + Red
    # opx.constant_ac(
    #     [4, 1],  # Digital channels
    #     [3, 4, 2, 6],  # Analog channels
    #     [0.19, 0.19, 0.17, 0.17],  # Analog voltages
    #     [110, 110, 75, 75],  # Analog frequencies
    # )
    # Red + green +  Yellow
    # opx.constant_ac(
    #     [4, 1],  # Digital channels
    #     [3, 4, 2, 6, 7],  # Analog channels
    #     [0.19, 0.19, 0.17, 0.17, 1.0],  # Analog voltages
    #     [110, 110, 75, 75, 0],  # Analog frequencies
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
    z_coord = 5.0
    global_coords = [None, None, z_coord]
    magnet_angle = 90
    date_str = "2024_02_23"

    # region Coords

    pixel_coords_list = [
        [110, 75],
        # [74.828, 109.09],
        [110, 50],
        [85.41, 60.905],
        [72.062, 51.179],
        [72.573, 16.985],
        [52.824, 95.547],
    ]
    green_coords_list = [
        [110, 110],
        # [110.7, 110.817],
        [112.041, 109.005],
        [110.867, 109.187],
        [110.671, 108.92],
        [110.867, 107.816],
        [109.872, 110.341],
    ]
    red_coords_list = [
        [75, 75],
        # [75.007, 76.233],
        [76.102, 74.64],
        [75.252, 74.807],
        [74.921, 74.602],
        [74.903, 73.591],
        [74.391, 75.83],
    ]

    # endregion

    # region NV sigs

    coords = {
        "global": global_coords,
        "pixel": pixel_coords_list.pop(0),
        green_laser: green_coords_list.pop(0),
        red_laser: red_coords_list.pop(0),
    }
    nv0 = NVSig(
        name=f"{sample_name}-nv0_{date_str}",
        coords=coords,
        representative=True,
        expected_count_rate=5000,
    )

    nv1 = copy.deepcopy(nv_sig_shell) | {
        "name": f"{sample_name}-nv1_{date_str}",
        pixel_coords_key: pixel_coords_list.pop(0),
        green_coords_key: green_coords_list.pop(0),
        red_coords_key: red_coords_list.pop(0),
    }

    nv2 = copy.deepcopy(nv_sig_shell) | {
        "name": f"{sample_name}-nv2_{date_str}",
        pixel_coords_key: pixel_coords_list.pop(0),
        green_coords_key: green_coords_list.pop(0),
        red_coords_key: red_coords_list.pop(0),
    }

    nv3 = copy.deepcopy(nv_sig_shell) | {
        "name": f"{sample_name}-nv3_{date_str}",
        pixel_coords_key: pixel_coords_list.pop(0),
        green_coords_key: green_coords_list.pop(0),
        red_coords_key: red_coords_list.pop(0),
    }

    nv4 = copy.deepcopy(nv_sig_shell) | {
        "name": f"{sample_name}-nv4_{date_str}",
        pixel_coords_key: pixel_coords_list.pop(0),
        green_coords_key: green_coords_list.pop(0),
        red_coords_key: red_coords_list.pop(0),
    }

    nv5 = copy.deepcopy(nv_sig_shell) | {
        "name": f"{sample_name}-nv5_{date_str}",
        pixel_coords_key: pixel_coords_list.pop(0),
        green_coords_key: green_coords_list.pop(0),
        red_coords_key: red_coords_list.pop(0),
    }

    # nv6 = copy.deepcopy(nv_sig_shell) | {
    #     "name": f"{sample_name}-nv6_{date_str}",
    #     pixel_coords_key: pixel_coords_list.pop(0),
    #     green_coords_key: green_coords_list.pop(0),
    #     red_coords_key: red_coords_list.pop(0),
    # }

    # nv7 = copy.deepcopy(nv_sig_shell) | {
    #     "name": f"{sample_name}-nv7_{date_str}",
    #     pixel_coords_key: pixel_coords_list.pop(0),
    #     green_coords_key: green_coords_list.pop(0),
    #     red_coords_key: red_coords_list.pop(0),
    # }

    # nv8 = copy.deepcopy(nv_sig_shell) | {
    #     "name": f"{sample_name}-nv8_{date_str}",
    #     pixel_coords_key: pixel_coords_list.pop(0),
    #     green_coords_key: green_coords_list.pop(0),
    #     red_coords_key: red_coords_list.pop(0),
    # }

    # nv9 = copy.deepcopy(nv_sig_shell) | {
    #     "name": f"{sample_name}-nv9_{date_str}",
    #     pixel_coords_key: pixel_coords_list.pop(0),
    #     green_coords_key: green_coords_list.pop(0),
    #     red_coords_key: red_coords_list.pop(0),
    # }

    # endregion

    # nv_sig = nv8
    # nv_list = [nv_sig]
    # nv_list = [nv0, nv1, nv2, nv3, nv4, nv5, nv6]
    # nv_list = [nv0, nv1, nv2, nv3, nv4, nv5, nv6, nv7]
    nv_list = [nv0, nv1, nv2, nv3, nv4, nv5]
    # nv_list = [nv0, nv2]
    nv_sig = widefield.get_repr_nv_sig(nv_list)

    # for nv in nv_list:
    #     coords = widefield.get_nv_pixel_coords(nv)
    #     r_coords = [round(el, 3) for el in coords]
    #     print(f"{r_coords},")
    # for nv in nv_list:
    #     widefield.set_nv_scanning_coords_from_pixel_coords(nv, green_laser)
    #     coords = nv[green_coords_key]
    #     r_coords = [round(el, 3) for el in coords]
    #     print(f"{r_coords},")
    # for nv in nv_list:
    #     widefield.set_nv_scanning_coords_from_pixel_coords(nv, red_laser)
    #     coords = nv[red_coords_key]
    #     r_coords = [round(el, 3) for el in coords]
    #     print(f"{r_coords},")
    # sys.exit()

    # sig_gen = tb.get_server_sig_gen(ind=1)
    # sig_gen.load_iq()
    # input("STOP")

    ### Functions to run

    email_recipient = "mccambria@berkeley.edu"
    do_email = False
    try:
        # pass

        kpl.init_kplotlib()
        # tb.init_safe_stop()

        # safe stop test
        # ind = 0
        # while not tb.safe_stop() and ind < 10:
        #     time.sleep(1)
        #     ind += 1
        #     print(ind)

        # Make sure the OPX config is up to date
        # cxn = common.labrad_connect()
        # opx = cxn.QM_opx
        # opx.update_config()

        # time.sleep(3)
        # mag_rot_server = tb.get_server_magnet_rotation()
        # mag_rot_server.set_angle(magnet_angle)
        # print(mag_rot_server.get_angle())

        # widefield.reset_all_drift()
        # widefield.set_pixel_drift([-22, +20])
        # widefield.set_all_scanning_drift_from_pixel_drift()

        # pos.set_xyz_on_nv(nv_sig)

        # for z in np.linspace(3.0, 7.0, 21):
        #     # for z in np.linspace(7.0, 3.0, 21):
        #     nv_sig["coords"][2] = z
        #     # do_scanning_image_sample(nv_sig)
        #     # for ind in range(100):
        #     do_widefield_image_sample(nv_sig, 1)
        #     time.sleep(5)
        # do_widefield_image_sample(nv_sig, 100)
        # do_scanning_image_sample(nv_sig)
        # do_optimize_pixel(nv_sig)

        # do_resonance(nv_list)
        # do_resonance_zoom(nv_list)
        # do_rabi(nv_list)
        do_correlation_test(nv_list)
        # do_spin_echo(nv_list)
        # do_spin_echo_long(nv_list)
        # do_spin_echo_medium(nv_list)
        # do_spin_echo_short(nv_list)
        # do_ramsey(nv_list)
        # do_sq_relaxation(nv_list)
        # do_dq_relaxation(nv_list)
        # do_xy8(nv_list)

        ## Infrequent stuff down here

        # # widefield.reset_all_drift()
        # coords_suffix = None  # Pixel coords
        # coords_suffix = green_laser
        # coords_suffix = red_laser
        # do_optimize_loop(nv_list, coords_suffix, scanning_from_pixel=True)

        # do_opx_constant_ac()
        # do_opx_square_wave()

        # do_charge_state_histograms(nv_list, 1000)
        # do_optimize_z(nv_sig)
        # do_calibrate_iq_delay(nv_list)
        # do_image_nv_list(nv_list)
        # do_optimize_scc(nv_list)
        # compile_speed_test(nv_list)
        # do_optimize_red(nv_sig)
        # do_scc_snr_check(nv_list)

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
