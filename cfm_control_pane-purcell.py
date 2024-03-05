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
from utils.constants import LaserKey, NVSpinState

green_laser = "laser_INTE_520"
red_laser = "laser_COBO_638"
yellow_laser = "laser_OPTO_589"
green_laser_dict = {"name": green_laser, "duration": 10e6}
red_laser_dict = {"name": red_laser, "duration": 10e6}
yellow_laser_dict = {"name": yellow_laser, "duration": 35e6}

### Major Routines


def do_widefield_image_sample(nv_sig, num_reps=1):
    image_sample.widefield_image(nv_sig, num_reps)


def do_scanning_image_sample(nv_sig):
    scan_range = 6
    num_steps = 10
    image_sample.scanning(nv_sig, scan_range, scan_range, num_steps)


def do_scanning_image_sample_zoom(nv_sig):
    scan_range = 0.5
    num_steps = 5
    image_sample.scanning(nv_sig, scan_range, scan_range, num_steps)


def do_image_nv_list(nv_list):
    num_reps = 10
    return image_sample.nv_list(nv_list, num_reps)


def do_image_single_nv(nv_sig):
    return image_sample.single_nv(nv_sig)


def do_charge_state_histograms(nv_list, num_reps):
    ion_duration = 10e3
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
    freq_center = 2.87
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
        [],  # Digital channels
        [7],  # Analog channels
        [0.38],  # Analog voltages
        10000,  # Period (ns)
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
    #     [1.0],  # Analog voltages
    #     [0],  # Analog frequencies
    # )
    # Green
    # opx.constant_ac(
    #     [4],  # Digital channels
    #     [3, 4],  # Analog channels
    #     [0.19, 0.19],  # Anal og voltages
    #     [108.632, 108.583],  # Analog frequencies
    # )
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
    # # Green + red
    opx.constant_ac(
        [4, 1],  # Digital channels
        [3, 4, 2, 6],  # Analog channels
        [0.19, 0.19, 0.17, 0.17],  # Analog voltages
        # [108.249, 108.582, 72.85, 73.55],  # Analog frequencies
        [113.229, 112.796, 76.6, 76.6],
    )
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
    z_coord = 4.45
    magnet_angle = 90
    date_str = "2024_03_04"

    nv_sig_shell = {
        "coords": [None, None, z_coord],
        "disable_opt": False,
        "disable_z_opt": True,
        "expected_count_rate": None,
        "collection": {"filter": None},
        "magnet_angle": None,
    }

    # region Coords

    pixel_coords_list = [
        [52.386, 128.953],
        [117.078, 66.748],
        [64.859, 95.757],
        [33.511, 91.747],
        [64.298, 13.227],
        [124.5, 126.189],
        [133.793, 158.078],
        [153.968, 116.817],
    ]
    green_coords_list = [
        [109.291, 111.509],
        [110.78, 110.173],
        [109.579, 110.724],
        [108.808, 110.636],
        [109.614, 108.876],
        [110.824, 111.447],
        [110.992, 112.176],
        [111.49, 111.224],
    ]
    red_coords_list = [
        [73.756, 76.25],
        [74.9, 75.07],
        [74.145, 75.439],
        [73.678, 75.354],
        [74.136, 73.692],
        [75.032, 76.083],
        [75.17, 76.759],
        [75.471, 75.885],
    ]

    # endregion
    # region NV sigs

    try:
        nv0 = copy.deepcopy(nv_sig_shell) | {
            "name": f"{sample_name}-nv0_{date_str}",
            pixel_coords_key: pixel_coords_list.pop(0),
            green_coords_key: green_coords_list.pop(0),
            red_coords_key: red_coords_list.pop(0),
            "repr": True,
        }

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

        nv6 = copy.deepcopy(nv_sig_shell) | {
            "name": f"{sample_name}-nv6_{date_str}",
            pixel_coords_key: pixel_coords_list.pop(0),
            green_coords_key: green_coords_list.pop(0),
            red_coords_key: red_coords_list.pop(0),
        }

        nv7 = copy.deepcopy(nv_sig_shell) | {
            "name": f"{sample_name}-nv7_{date_str}",
            pixel_coords_key: pixel_coords_list.pop(0),
            green_coords_key: green_coords_list.pop(0),
            red_coords_key: red_coords_list.pop(0),
        }

        nv8 = copy.deepcopy(nv_sig_shell) | {
            "name": f"{sample_name}-nv8_{date_str}",
            pixel_coords_key: pixel_coords_list.pop(0),
            green_coords_key: green_coords_list.pop(0),
            red_coords_key: red_coords_list.pop(0),
        }

        nv9 = copy.deepcopy(nv_sig_shell) | {
            "name": f"{sample_name}-nv9_{date_str}",
            pixel_coords_key: pixel_coords_list.pop(0),
            green_coords_key: green_coords_list.pop(0),
            red_coords_key: red_coords_list.pop(0),
        }

        nv10 = copy.deepcopy(nv_sig_shell) | {
            "name": f"{sample_name}-nv10_{date_str}",
            pixel_coords_key: pixel_coords_list.pop(0),
            green_coords_key: green_coords_list.pop(0),
            red_coords_key: red_coords_list.pop(0),
        }

        nv11 = copy.deepcopy(nv_sig_shell) | {
            "name": f"{sample_name}-nv11_{date_str}",
            pixel_coords_key: pixel_coords_list.pop(0),
            green_coords_key: green_coords_list.pop(0),
            red_coords_key: red_coords_list.pop(0),
        }

        nv12 = copy.deepcopy(nv_sig_shell) | {
            "name": f"{sample_name}-nv12_{date_str}",
            pixel_coords_key: pixel_coords_list.pop(0),
            green_coords_key: green_coords_list.pop(0),
            red_coords_key: red_coords_list.pop(0),
        }

        nv13 = copy.deepcopy(nv_sig_shell) | {
            "name": f"{sample_name}-nv13_{date_str}",
            pixel_coords_key: pixel_coords_list.pop(0),
            green_coords_key: green_coords_list.pop(0),
            red_coords_key: red_coords_list.pop(0),
        }

        nv14 = copy.deepcopy(nv_sig_shell) | {
            "name": f"{sample_name}-nv14_{date_str}",
            pixel_coords_key: pixel_coords_list.pop(0),
            green_coords_key: green_coords_list.pop(0),
            red_coords_key: red_coords_list.pop(0),
        }

        nv15 = copy.deepcopy(nv_sig_shell) | {
            "name": f"{sample_name}-nv15_{date_str}",
            pixel_coords_key: pixel_coords_list.pop(0),
            green_coords_key: green_coords_list.pop(0),
            red_coords_key: red_coords_list.pop(0),
        }

    # Probably tried to define more NVs than there are coords - just ignore
    except Exception:
        pass

    # endregion

    # nv_sig = nv8
    # nv_list = [nv_sig]
    # nv_list = [nv0, nv1, nv2, nv3, nv4, nv5, nv6]
    # nv_list = [nv0, nv1]
    nv_list = [nv1, nv0]
    # nv_list = [nv0, nv2]

    # nv_list = [nv0, nv1, nv2, nv3, nv4, nv5, nv6, nv7]
    # nv_list = [nv0, nv1, nv2, nv3, nv4, nv5, nv6, nv7, nv8, nv9]
    # nv_list = [nv0, nv1, nv2, nv3, nv4, nv5, nv6, nv7, nv8, nv9, nv10, nv11]
    # nv_list = [nv0, nv1, nv2, nv3, nv4, nv5]
    # nv_list = [nv0, nv1]
    nv_sig = widefield.get_repr_nv_sig(nv_list)
    # nv_sig = nv0
    # nv_sig = nv1

    # Coordinate printing
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

    ### Functions to run

    email_recipient = "mccambria@berkeley.edu"
    do_email = False
    try:
        # pass

        kpl.init_kplotlib()
        # tb.init_safe_stop()

        # widefield.reset_all_drift()
        # widefield.set_all_scanning_drift_from_pixel_drift()

        # pos.set_xyz_on_nv(nv_sig)

        # for z in np.linspace(4.48, 4.56, 5):
        #     nv_sig["coords"][2] = z
        #     # do_scanning_image_sample(nv_sig)
        #     do_widefield_image_sample(nv_sig, 100)
        # for ind in range(100):
        #     do_widefield_image_sample(nv_sig, 100)
        #     time.sleep(5)

        # do_scanning_image_sample(nv_sig)
        # do_scanning_image_sample_zoom(nv_sig)
        do_widefield_image_sample(nv_sig, 100)

        # do_image_nv_list(nv_list)
        # do_image_single_nv(nv_sig)

        # for nv_sig in nv_list:
        # do_optimize_pixel(nv_sig)
        # do_optimize_green(nv_sig)
        # do_optimize_red(nv_sig)
        # do_image_single_nv(nv_sig)

        # do_optimize_pixel(nv_sig)
        # do_optimize_green(nv_sig)
        # do_optimize_red(nv_sig)

        # widefield.reset_all_drift()
        # coords_suffix = None  # Pixel coords
        # coords_suffix = green_laser
        # coords_suffix = red_laser
        # do_optimize_loop(nv_list, coords_suffix, scanning_from_pixel=False)

        # do_charge_state_histograms(nv_list, 100)
        # do_charge_state_histograms(nv_list, 1000)

        # do_resonance_zoom(nv_list)

        # do_opx_constant_ac()
        # do_opx_square_wave()

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
