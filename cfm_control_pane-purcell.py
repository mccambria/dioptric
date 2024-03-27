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
    optimize,
    optimize_scc,
    rabi,
    ramsey,
    relaxation_interleave,
    resonance,
    scc_snr_check,
    spin_echo,
    xy8,
)
from utils import common, widefield
from utils import kplotlib as kpl
from utils import positioning as pos
from utils import tool_belt as tb
from utils.constants import CoordsKey, LaserKey, NVSig, NVSpinState

green_laser = "laser_INTE_520"
red_laser = "laser_COBO_638"
yellow_laser = "laser_OPTO_589"

### Major Routines


def do_widefield_image_sample(nv_sig, num_reps=1):
    image_sample.widefield_image(nv_sig, num_reps)


def do_scanning_image_sample(nv_sig):
    scan_range = 6
    num_steps = 5
    image_sample.scanning(nv_sig, scan_range, scan_range, num_steps)


def do_scanning_image_sample_zoom(nv_sig):
    scan_range = 0.5
    num_steps = 5
    image_sample.scanning(nv_sig, scan_range, scan_range, num_steps)


def do_image_nv_list(nv_list):
    num_reps = 100
    return image_sample.nv_list(nv_list, num_reps)


def do_image_single_nv(nv_sig):
    num_reps = 100
    return image_sample.single_nv(nv_sig, num_reps)


def do_charge_state_histograms(nv_list, num_reps):
    return charge_state_histograms.main(nv_list, num_reps)


def do_optimize_green(nv_sig, do_plot=True):
    coords_suffix = tb.get_laser_name(LaserKey.IMAGING)
    ret_vals = optimize.main(
        nv_sig,
        coords_suffix=coords_suffix,
        no_crash=True,
        do_plot=do_plot,
        axes_to_optimize=[0, 1],
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
        axes_to_optimize=[0, 1],
    )
    opti_coords = ret_vals[0]
    return opti_coords


def do_optimize_z(nv_sig, do_plot=True):
    optimize.main(nv_sig, no_crash=True, do_plot=do_plot, axes_to_optimize=[2])


def do_optimize_pixel(nv_sig):
    opti_coords = optimize.optimize_pixel(nv_sig, do_plot=True)
    return opti_coords


def do_optimize_loop(nv_list, coords_suffix, scanning_from_pixel=False):
    repr_nv_sig = widefield.get_repr_nv_sig(nv_list)

    opti_coords_list = []
    for nv in nv_list:
        # Pixel coords
        if coords_suffix is None:
            imaging_laser = tb.get_laser_name(LaserKey.IMAGING)
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
        optimize.optimize_widefield_calibration(cxn)


def do_optimize_scc(nv_list):
    min_tau = 16
    max_tau = 400
    num_steps = 13
    num_reps = 15
    num_runs = 50
    optimize_scc.main(nv_list, num_steps, num_reps, num_runs, min_tau, max_tau)


def do_scc_snr_check(nv_list):
    num_reps = 200
    num_runs = 20
    scc_snr_check.main(nv_list, num_reps, num_runs)


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
    freq_range = 0.180
    num_steps = 40
    num_reps = 10
    num_runs = 240
    resonance.main(nv_list, num_steps, num_reps, num_runs, freq_center, freq_range)


def do_resonance_zoom(nv_list):
    freq_center = 2.87
    freq_range = 0.06
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
    num_reps = 10
    num_runs = 20
    uwave_ind = 0
    rabi.main(
        nv_list, num_steps, num_reps, num_runs, min_tau, max_tau, uwave_ind=uwave_ind
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
    min_tau = 100
    # min_tau = 83.7e3
    # min_tau = 167.4e3
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
    anticorrelation_inds = [2, 3]

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
        [4],  # Digital channels
        [],  # Analog channels
        [],  # Analog voltages
        100000,  # Period (ns)
    )
    input("Press enter to stop...")
    # sig_gen.uwave_off()


def do_opx_constant_ac():
    cxn = common.labrad_connect()
    opx = cxn.QM_opx

    # Microwave test
    # if True:
    #     sig_gen = cxn.sig_gen_STAN_sg394
    #     amp = 10
    #     chan = 10
    # else:
    #     sig_gen = cxn.sig_gen_STAN_sg394_2
    #     amp = 10
    #     chan = 9
    # sig_gen.set_amp(amp)  # 12
    # sig_gen.set_freq(0.1)
    # sig_gen.uwave_on()
    # opx.constant_ac([chan])

    # Camera frame rate test
    # seq_args = [500]
    # seq_args_string = tb.encode_seq_args(seq_args)
    # opx.stream_load("camera_test.py", seq_args_string)
    # opx.stream_start()

    # Yellow
    opx.constant_ac(
        [],  # Digital channels
        [7],  # Analog channels
        [0.5],  # Analog voltages
        [0],  # Analog frequencies
    )
    # Green
    # opx.constant_ac(
    #     [4],  # Digital channels
    #     [3, 4],  # Analog channels
    #     [0.19, 0.19],  # Analog voltages
    #     [111, 111],  # Analog frequencies
    # )
    # opx.constant_ac([4])  # Just laser
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
    # opx.constant_ac([1])  # Just laser
    # # Green + red
    # opx.constant_ac(
    #     [4, 1],  # Digital channels
    #     [3, 4, 2, 6],  # Analog channels
    #     [0.19, 0.19, 0.17, 0.17],  # Analog voltages
    #     # [108.249, 108.582, 72.85, 73.55],  # Analog frequencies
    #     [113.229, 112.796, 76.6, 76.6],
    # )
    # red
    # opx.constant_ac(
    #     [1],  # Digital channels
    #     [2, 6],  # Analog channels
    #     [0.17, 0.17],  # Analog voltages
    #     [76.7, 76.6],  # Analog frequencies
    # )
    # Green + yellow
    # opx.constant_ac(
    #     [4],  # Digital channels
    #     [3, 4, 7],  # Analog channels
    #     [0.19, 0.19, 1.0],  # Analog voltages
    #     [110.5, 110, 0],  # Analog frequencies
    # )
    # Red + green + Yellow
    # opx.constant_ac(
    #     [4, 1],  # Digital channels
    #     [3, 4, 2, 6, 7],  # Analog channels
    #     [0.19, 0.19, 0.17, 0.17, 1.0],  # Analog voltages
    #     [110, 110, 75, 75, 0],  # Analog frequencies
    # )
    input("Press enter to stop...")
    # sig_gen.uwave_off()


def compile_speed_test(nv_list):
    cxn = common.labrad_connect()
    pulse_gen = cxn.QM_opx

    seq_file = "resonance_ref.py"
    num_reps = 20
    uwave_index = 0

    seq_args = widefield.get_base_scc_seq_args(nv_list)
    seq_args.append(uwave_index)
    seq_args.append([2.1, 2.3, 2.5, 2.7, 2.9])
    seq_args_string = tb.encode_seq_args(seq_args)

    start = time.time()
    pulse_gen.stream_load(seq_file, seq_args_string, num_reps)
    stop = time.time()
    print(stop - start)

    seq_args[-2] = 1
    seq_args_string = tb.encode_seq_args(seq_args)

    start = time.time()
    pulse_gen.stream_load(seq_file, seq_args_string, num_reps)
    stop = time.time()
    print(stop - start)


### Run the file


if __name__ == "__main__":
    # region Shared parameters

    green_coords_key = f"coords-{green_laser}"
    red_coords_key = f"coords-{red_laser}"
    pixel_coords_key = "pixel_coords"

    sample_name = "johnson"
    z_coord = 4.252
    magnet_angle = 90
    date_str = "2024_03_12"
    global_coords = [None, None, z_coord]

    # endregion
    # region Coords (from March 12th)

    # pixel_coords_list = [
    #     [126.905, 114.634],
    #     [163.243, 117.933],
    #     [83.205, 115.313],
    #     [72.362, 125.984],
    #     [94.422, 164.308],
    #     [101.672, 142.676],
    #     [99.67, 126.488],
    #     [115.954, 128.468],
    #     [124.404, 142.99],
    #     [120.505, 169.064],
    #     [138.882, 160.072],
    #     [151.59, 144.97],
    #     [160.774, 88.405],
    #     [147.589, 73.976],
    # ]
    # green_coords_list = [
    #     [108.333, 110.935],
    #     [109.127, 110.828],
    #     [107.44, 110.812],
    #     [107.167, 110.596],
    #     [107.624, 109.545],
    #     [107.708, 110.114],
    #     [107.792, 110.606],
    #     [107.964, 110.409],
    #     [108.561, 110.278],
    #     [108.182, 109.55],
    #     [108.657, 109.894],
    #     [108.856, 110.129],
    #     [109.084, 111.507],
    #     [108.889, 111.896],
    # ]
    # red_coords_list = [
    #     [73.226, 75.736],
    #     [73.881, 75.68],
    #     [72.427, 75.715],
    #     [72.185, 75.526],
    #     [72.549, 74.866],
    #     [72.742, 75.307],
    #     [72.721, 75.547],
    #     [73.004, 75.583],
    #     [73.194, 75.288],
    #     [73.169, 74.814],
    #     [73.484, 75.007],
    #     [73.688, 75.275],
    #     [73.762, 76.351],
    #     [73.552, 76.608],
    # ]
    # endregion
    # region Coords (smiley)

    pixel_coords_list = [
        [134.395, 145.486],
        [152.557, 136.398],
        [165.503, 121.443],
        [177.514, 94.414],
        [174.805, 65.13],
        [161.244, 50.454],
    ]
    green_coords_list = [
        [108.492, 110.038],
        [108.889, 110.291],
        [109.126, 110.648],
        [109.356, 111.328],
        [109.253, 111.974],
        [108.944, 112.355],
    ]
    red_coords_list = [
        [73.278, 75.07],
        [73.591, 75.244],
        [73.841, 75.566],
        [74.083, 75.969],
        [73.922, 76.603],
        [73.748, 76.776],
    ]

    # endregion
    # region NV list construction

    # nv_list[i] will have the ith coordinates from the above lists
    num_nvs = len(pixel_coords_list)
    nv_list = []
    for ind in range(num_nvs):
        coords = {
            CoordsKey.GLOBAL: global_coords,
            CoordsKey.PIXEL: pixel_coords_list.pop(0),
            green_laser: green_coords_list.pop(0),
            red_laser: red_coords_list.pop(0),
        }
        nv_sig = NVSig(name=f"{sample_name}-nv{ind}_{date_str}", coords=coords)
        nv_list.append(nv_sig)

    # Additional properties for the representative NV
    nv_list[0].representative = True
    # nv_list[0].expected_counts = None
    # nv_list[0].expected_counts = 3900
    # nv_list[0].expected_counts = 5000
    nv_list[0].expected_counts = 6000
    # nv_list[0].expected_counts = 7400
    nv_sig = widefield.get_repr_nv_sig(nv_list)

    # endregion
    # region Coordinate printing

    # for nv in nv_list:
    #     pixel_drift = widefield.get_pixel_drift()
    #     pixel_drift = [-el for el in pixel_drift]
    #     coords = widefield.get_nv_pixel_coords(nv, drift=pixel_drift)
    #     r_coords = [round(el, 3) for el in coords]
    #     print(f"{r_coords},")
    # for nv in nv_list:
    #     widefield.set_nv_scanning_coords_from_pixel_coords(
    #         nv, green_laser, drift_adjust=False
    #     )
    #     coords = nv[green_coords_key]
    #     r_coords = [round(el, 3) for el in coords]
    #     print(f"{r_coords},")
    # for nv in nv_list:
    #     widefield.set_nv_scanning_coords_from_pixel_coords(
    #         nv, red_laser, drift_adjust=False
    #     )
    #     coords = nv[red_coords_key]
    #     r_coords = [round(el, 3) for el in coords]
    #     print(f"{r_coords},")
    # sys.exit()

    # endregion
    ### Functions to run

    email_recipient = "mccambria@berkeley.edu"
    do_email = False
    try:
        # pass

        kpl.init_kplotlib()
        # tb.init_safe_stop()

        # widefield.reset_all_drift()
        # pos.reset_drift()  # Reset z drift
        # widefield.set_pixel_drift([-1, -12])
        # widefield.set_all_scanning_drift_from_pixel_drift()

        # do_optimize_z(nv_sig)

        # pos.set_xyz_on_nv(nv_sig)

        # for z in np.linspace(4.0, 4.3, 11):
        #     nv_sig.coords[CoordsKey.GLOBAL][2] = z
        #     do_widefield_image_sample(nv_sig, 20)

        # do_scanning_image_sample(nv_sig)
        # do_scanning_image_sample_zoom(nv_sig)
        do_widefield_image_sample(nv_sig, 20)
        # do_widefield_image_sample(nv_sig, 100)

        # do_image_nv_list(nv_list)
        # do_image_single_nv(nv_sig)

        # for nv_sig in nv_list:
        #     widefield.reset_all_drift()
        #     # do_optimize_pixel(nv_sig)
        #     do_optimize_green(nv_sig)
        # do_optimize_red(nv_sig)
        # do_image_single_nv(nv_sig)

        # do_optimize_pixel(nv_sig)
        # # # # do_optimize_green(nv_sig)
        # # # # do_optimize_red(nv_sig)
        # do_optimize_z(nv_sig)

        # widefield.reset_all_drift()
        # # coords_suffix = None  # Pixel coords
        # # coords_suffix = green_laser
        # coords_suffix = red_laser
        # do_optimize_loop(nv_list, coords_suffix, scanning_from_pixel=True)

        # num_nvs = len(nv_list)
        # for ind in range(num_nvs):
        #     if ind == 0:
        #         continue
        #     nv = nv_list[ind]
        #     green_coords = nv[green_coords_key]
        #     nv[green_coords_key][0] += 0.500

        # do_charge_state_histograms(nv_list, 100)
        # do_charge_state_histograms(nv_list, 1000)

        # do_resonance(nv_list)
        # do_resonance_zoom(nv_list)
        # do_rabi(nv_list)
        # do_correlation_test(nv_list)
        # do_spin_echo(nv_list)
        # do_spin_echo_long(nv_list)
        # do_spin_echo_medium(nv_list)
        # do_spin_echo_short(nv_list)
        # do_ramsey(nv_list)
        # do_sq_relaxation(nv_list)
        # do_dq_relaxation(nv_list)
        # do_xy8(nv_list)

        # do_opx_constant_ac()
        # do_opx_square_wave()

        # do_scc_snr_check(nv_list)
        # do_optimize_scc(nv_list)

    # region Cleanup

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

    # endregion
