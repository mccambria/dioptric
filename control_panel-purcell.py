# -*- coding: utf-8 -*-
"""
Control panel for the PC Rabi

Created on June 16th, 2023

@author: mccambria
"""


### Imports

import datetime
import os
import random
import sys
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np

from majorroutines.widefield import (
    ac_stark,
    calibrate_iq_delay,
    charge_monitor,
    charge_state_conditional_init,
    charge_state_histograms,
    correlation_test,
    crosstalk_check,
    image_sample,
    optimize_scc,
    power_rabi,
    rabi,
    ramsey,
    relaxation_interleave,
    resonance,
    scc_snr_check,
    simple_correlation_test,
    spin_echo,
    spin_pol_check,
    targeting,
    xy8,
)

# from slmsuite import optimize_slm_calibration
from utils import common, widefield
from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import positioning as pos
from utils import tool_belt as tb
from utils.constants import CoordsKey, NVSig, VirtualLaserKey

green_laser = "laser_INTE_520"
red_laser = "laser_COBO_638"
yellow_laser = "laser_OPTO_589"
green_laser_aod = f"{green_laser}_aod"
red_laser_aod = f"{red_laser}_aod"

### Major Routines


def do_widefield_image_sample(nv_sig, num_reps=1):
    return image_sample.widefield_image(nv_sig, num_reps)


def do_scanning_image_sample(nv_sig):
    scan_range = 12
    num_steps = 12
    image_sample.scanning(nv_sig, scan_range, scan_range, num_steps)


def do_scanning_image_sample_zoom(nv_sig):
    scan_range = 0.01
    num_steps = 5
    image_sample.scanning(nv_sig, scan_range, scan_range, num_steps)


def do_image_nv_list(nv_list):
    num_reps = 200
    # num_reps = 2
    return image_sample.nv_list(nv_list, num_reps)


def do_image_single_nv(nv_sig):
    num_reps = 100
    return image_sample.single_nv(nv_sig, num_reps)


def do_charge_state_histograms(nv_list):
    num_reps = 150
    # num_reps = 100
    # num_runs = 50
    num_runs = 15
    # num_runs = 10
    # num_runs = 2
    # for ion_include_inds in [None, [0, 1, 2, 3, 4, 5]]:
    #     charge_state_histograms.main(
    #         nv_list, num_reps, num_runs, ion_include_inds=ion_include_inds
    #     )
    return charge_state_histograms.main(
        nv_list, num_reps, num_runs, plot_histograms=True
    )


def do_charge_state_conditional_init(nv_list):
    num_reps = 20
    num_runs = 100
    # num_runs = 400
    return charge_state_conditional_init.main(nv_list, num_reps, num_runs)


def do_optimize_green(nv_sig, do_plot=True):
    coords_key = tb.get_laser_name(VirtualLaserKey.IMAGING)
    ret_vals = targeting.main(
        nv_sig,
        coords_key=coords_key,
        no_crash=True,
        do_plot=do_plot,
        axes_to_optimize=[0, 1],
    )
    opti_coords = ret_vals[0]
    return opti_coords


def do_optimize_red(nv_sig, do_plot=True, axes_to_optimize=[0, 1]):
    coords_key = red_laser
    ret_vals = targeting.main(
        nv_sig,
        coords_key=coords_key,
        no_crash=True,
        do_plot=do_plot,
        axes_to_optimize=[0, 1],
    )
    opti_coords = ret_vals[0]
    return opti_coords


def do_optimize_z(nv_sig, do_plot=True):
    targeting.main(nv_sig, no_crash=True, do_plot=do_plot, axes_to_optimize=[2])


def do_optimize_xyz(nv_sig, do_plot=True):
    targeting.optimize_xyz_using_piezo(
        nv_sig, do_plot=do_plot, axes_to_optimize=[0, 1, 2]
    )


def do_optimize_pixel(nv_sig):
    opti_coords = targeting.optimize_pixel(nv_sig, do_plot=True)
    return opti_coords


def do_optimize_loop(nv_list, coords_key, scanning_from_pixel=False):
    repr_nv_sig = widefield.get_repr_nv_sig(nv_list)

    # Pixel optimization in parallel with widefield yellow
    if coords_key is None:
        num_reps = 200
        img_array = do_widefield_image_sample(nv_sig, num_reps=num_reps)

    opti_coords_list = []
    for nv in nv_list:
        # Pixel coords
        if coords_key is None:
            # imaging_laser = tb.get_laser_name(LaserKey.IMAGING)
            # if scanning_from_pixel:
            #     widefield.set_nv_scanning_coords_from_pixel_coords(nv, imaging_laser)
            opti_coords = do_optimize_pixel(nv)
            # opti_coords = optimize.optimize_pixel_with_img_array(img_array, nv_sig=nv)
            # widefield.reset_all_drift()

        # Scanning coords
        else:
            if scanning_from_pixel:
                widefield.set_nv_scanning_coords_from_pixel_coords(nv, coords_key)

            if coords_key == green_laser:
                opti_coords = do_optimize_green(nv)
            elif coords_key == red_laser:
                opti_coords = do_optimize_red(nv)
            # Adjust for the drift that may have occurred since beginning the loop
            # optimize.optimize_pixel_and_z(repr_nv_sig, do_plot=False)
            targeting.optimize_xyz_using_piezo(repr_nv_sig)
            drift = pos.get_drift(coords_key)
            drift = [-1 * el for el in drift]
            opti_coords = pos.adjust_coords_for_drift(opti_coords, drift=drift)
            widefield.reset_scanning_optics_drift()  # reset drift before optimizing next NV
        opti_coords_list.append(opti_coords)

    # Report back
    for opti_coords in opti_coords_list:
        r_opti_coords = [round(el, 3) for el in opti_coords]
        print(f"{r_opti_coords},")


def optimize_slm_Phase_calibration(repr_nv_sig, target_coords):
    repr_nv_sig = widefield.get_repr_nv_sig(nv_list)
    target_coords = np.array([[110.186, 129.281], [128.233, 88.007], [86.294, 103.0]])
    # optimize_slm_calibration.main(repr_nv_sig, target_coords)


def do_calibrate_green_red_delay():
    cxn = common.labrad_connect()
    pulse_gen = cxn.QM_opx

    seq_file = "calibrate_green_red_delay.py"

    seq_args = [2000]
    seq_args_string = tb.encode_seq_args(seq_args)
    num_reps = -1

    pulse_gen.stream_immediate(seq_file, seq_args_string, num_reps)

    input("Press enter to stop...")
    pulse_gen.halt()


def do_optimize_scc_duration(nv_list):
    min_tau = 16
    max_tau = 224
    num_steps = 14
    num_reps = 15

    # num_runs = 20 * 25
    num_runs = 30
    num_runs = 50
    num_runs = 2

    optimize_scc.optimize_scc_duration(
        nv_list, num_steps, num_reps, num_runs, min_tau, max_tau
    )


def do_optimize_scc_amp(nv_list):
    min_tau = 0.7
    max_tau = 1.3
    num_steps = 16
    num_reps = 20

    num_runs = 60
    # num_runs = 2

    optimize_scc.optimize_scc_amp(
        nv_list, num_steps, num_reps, num_runs, min_tau, max_tau
    )


def do_scc_snr_check(nv_list):
    # num_reps = 100
    # num_runs = 100
    num_reps = 300
    num_runs = 30
    # num_runs = 160 * 4
    # num_runs = 2
    scc_snr_check.main(nv_list, num_reps, num_runs, uwave_ind_list=[0, 1])


def do_simple_correlation_test(nv_list):
    num_reps = 300
    # num_runs = 2000
    num_runs = 1000
    # num_runs = 2
    simple_correlation_test.main(nv_list, num_reps, num_runs)

    # for ind in range(4):
    #     for flipped in [True, False]:
    #         for nv_ind in range(3):
    #             nv = nv_list[nv_ind]
    #             if ind == nv_ind:
    #                 nv.spin_flip = flipped
    #             else:
    #                 nv.spin_flip = not flipped
    #         simple_correlation_test.main(nv_list, num_reps, num_runs)


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
    freq_range = 0.240
    num_steps = 60
    # Single ref
    # num_reps = 8
    num_runs = 300
    # num_runs = 50

    # Both refs
    num_reps = 2
    # num_runs = 300

    # num_runs = 2

    resonance.main(nv_list, num_steps, num_reps, num_runs, freq_center, freq_range)


def do_resonance_zoom(nv_list):
    # for freq_center in (2.85761751, 2.812251747511455):
    for freq_center in (2.87 + (2.87 - 2.85856), 2.87 + (2.87 - 2.81245)):
        freq_range = 0.030
        num_steps = 20
        num_reps = 15
        num_runs = 60
        resonance.main(nv_list, num_steps, num_reps, num_runs, freq_center, freq_range)


def do_rabi(nv_list):
    min_tau = 16
    max_tau = 240 + min_tau
    # max_tau = 360 + min_tau
    # max_tau = 480 + min_tau
    num_steps = 31
    num_reps = 10
    num_runs = 200
    # num_runs = 100
    # num_runs = 50
    # num_runs = 20

    # uwave_ind_list = [1]
    uwave_ind_list = [0, 1]
    rabi.main(nv_list, num_steps, num_reps, num_runs, min_tau, max_tau, uwave_ind_list)
    # uwave_ind_list = [0]
    # rabi.main(nv_list, num_steps, num_reps, num_runs, min_tau, max_tau, uwave_ind_list)
    # uwave_ind_list = [1]
    # rabi.main(nv_list, num_steps, num_reps, num_runs, min_tau, max_tau, uwave_ind_list)


def do_ac_stark(nv_list):
    min_tau = 0
    # max_tau = 240 + min_tau
    # max_tau = 360 + min_tau
    max_tau = 480 + min_tau
    num_steps = 31
    num_reps = 10
    # num_runs = 100
    num_runs = 50
    # num_runs = 2

    # uwave_ind_list = [1]
    uwave_ind_list = [0, 1]

    ac_stark.main(
        nv_list, num_steps, num_reps, num_runs, min_tau, max_tau, uwave_ind_list
    )


def do_power_rabi(nv_list):
    # power_center = -3.6
    power_range = 6
    num_steps = 16
    num_reps = 20
    num_runs = 150
    # num_runs = 50
    # num_runs = 2

    # uwave_ind_list = [0]
    uwave_ind_list = [0, 1]

    power_rabi.main(
        nv_list,
        num_steps,
        num_reps,
        num_runs,
        power_range,
        uwave_ind_list,
    )


def do_spin_echo(nv_list):
    min_tau = 200
    max_tau = 84e3 + min_tau
    num_steps = 29
    num_reps = 3
    num_runs = 1000
    # num_runs = 2
    spin_echo.main(nv_list, num_steps, num_reps, num_runs, min_tau, max_tau)


def do_ramsey(nv_list):
    min_tau = 100
    max_tau = 3200 + min_tau
    detuning = 3
    num_steps = 101
    num_reps = 3
    num_runs = 1600
    # num_runs = 2
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
    num_reps = 15
    num_runs = 200
    # num_runs = 2
    relaxation_interleave.sq_relaxation(
        nv_list, num_steps, num_reps, num_runs, min_tau, max_tau
    )


def do_dq_relaxation(nv_list):
    min_tau = 1e3
    max_tau = 15e6 + min_tau
    num_steps = 21
    num_reps = 15
    num_runs = 200
    relaxation_interleave.dq_relaxation(
        nv_list, num_steps, num_reps, num_runs, min_tau, max_tau
    )


def do_opx_square_wave():
    cxn = common.labrad_connect()
    opx = cxn.QM_opx

    # Yellow
    opx.square_wave(
        [],  # Digital channels
        [7],  # Analog channels
        [0.4],  # Analog voltages
        10000,  # Period (ns)
    )
    # Camera trigger
    # opx.square_wave(
    #     [4],  # Digital channels
    #     [],  # Analog channels
    #     [],  # Analog voltages
    #     100000,  # Period (ns)
    # )
    input("Press enter to stop...")
    # sig_gen.uwave_off()


def do_crosstalk_check(nv_sig):
    num_steps = 21
    num_reps = 10
    num_runs = 150
    # aod_freq_range = 3.0
    laser_name = red_laser
    # laser_name = green_laser
    # axis_ind = 0  # 0: x, 1: y, 2: z
    uwave_ind = [0, 1]

    if laser_name is red_laser:
        aod_freq_range = 2.0
    elif laser_name is green_laser:
        aod_freq_range = 3.0
    for axis_ind in [0, 1]:
        crosstalk_check.main(
            nv_sig,
            num_steps,
            num_reps,
            num_runs,
            aod_freq_range,
            laser_name,
            axis_ind,  # 0: x, 1: y, 2: z
            uwave_ind,
        )


def do_spin_pol_check(nv_sig):
    num_steps = 16
    num_reps = 10
    num_runs = 40
    aod_min_voltage = 0.01
    aod_max_voltage = 0.05
    uwave_ind = 0

    spin_pol_check.main(
        nv_sig,
        num_steps,
        num_reps,
        num_runs,
        aod_min_voltage,
        aod_max_voltage,
        uwave_ind,
    )


def do_detect_cosmic_rays(nv_list):
    num_reps = 60
    num_runs = 6 * 60
    # num_runs = 2
    dark_time = 1e9

    charge_monitor.detect_cosmic_rays(nv_list, num_reps, num_runs, dark_time)


def do_check_readout_fidelity(nv_list):
    num_reps = 200
    num_runs = 20

    charge_monitor.check_readout_fidelity(nv_list, num_reps, num_runs)


def do_charge_quantum_jump(nv_list):
    num_reps = 2000

    charge_monitor.charge_quantum_jump(nv_list, num_reps)


def do_opx_constant_ac():
    cxn = common.labrad_connect()
    opx = cxn.QM_opx

    # num_reps = 1000
    # start = time.time()
    # for ind in range(num_reps):
    #     opx.test("_cache_charge_pol_incomplete", False)
    # stop = time.time()
    # print((stop - start) / num_reps)

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
        [0.45],  # Analog voltages
        [0],  # Analog frequencies
    )

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
    #     # [2, 6],  # Analog channels
    #     # [0.19, 0.19],  # Analog voltages
    #     # [
    #     #     75,
    #     #     75,
    #     # ],  # Analog frequencies                                                                                                                                                                       uencies
    # )
    # opx.constant_ac([1])  # Just laser
    # Green
    # opx.constant_ac(
    #     [4],  # Digital channels
    #     [3, 4],  # Analog channels
    #     [0.19, 0.19],  # Analog voltages
    #     [105.253, 104.867],  # Analog frequencies
    # )
    # Green + red
    # opx.constant_ac(
    #     [4, 1],  # Digital channels
    #     [3, 4, 2, 6],  # Analog channels
    #     [0.19, 0.19, 0.17, 0.17],  # Analog voltages;
    #     # [109.409, 111.033, 73.0, 77.3],  # Analog frequencies
    #     # [108.907, 112.362, 74.95, 78.65],  # Analog frequencies
    #     [107.477, 106.641, 78.1, 71.976],
    # )
    #     green_coords_list = [
    #     [109.366, 111.43],
    #     [113.25, 106.469],
    #     [107.477, 106.641],
    #     # [105.253, 104.867],
    # ]
    # red_coords_list = [
    #     [74.927, 76.473],
    #     [78.25, 71.476],
    #     [72.466, 71.641],
    #     # [70.772, 71.758],
    # ]
    # red
    # opx.constant_ac(
    #     [1],  # Digital channels
    #     [2, 6],  # Analog channels
    #     [0.17, 0.17],  # Analog voltages
    #     [73.166, 72.941],  # Analog frequencies
    # )
    # Green + yellow
    # opx.constant_ac(
    #     [4],  # Digital channels
    #     [3, 4, 7],  # Analog channels
    #     [0.19, 0.19, 0.35],  # Analog voltages
    #     [110, 110, 0],  # Analog frequencies
    # )
    # Red + green + Yellow
    # opx.constant_ac(
    #     [4, 1],  # Digital channels
    #     [3, 4, 2, 6, 7],  # Analog channels
    #     [0.19, 0.19, 0.17, 0.17, 0.3],  # Analog voltages
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


def piezo_voltage_to_pixel_calibration():
    cal_voltage_coords = np.array(
        [[0.0, 0.0], [-0.25, -0.25], [0.25, -0.25]], dtype="float32"
    )  # Voltage system coordinates
    # cal_pixel_coords = np.array(
    #     [[81.109, 110.177], [64.986, 94.177], [96.577, 95.047]], dtype="float32"
    # )  # Corresponding pixel coordinates
    cal_pixel_coords = np.array(
        [
            [91.778, 122.027],
            [109.388, 139.694],
            [75.396, 138.755],
        ],
        dtype="float32",
    )  # Corresponding pixel coordinates
    # Compute the affine transformation matrix
    M = cv2.getAffineTransform(cal_voltage_coords, cal_pixel_coords)
    # Convert the 2x3 matrix to a 3x3 matrix
    M = np.vstack([M, [0, 0, 1]])
    M_inv = np.linalg.inv(M)

    # Format and print the affine matrix as a list of lists
    affine_voltage2pixel = M.tolist()
    inverse_affine_voltage2pixel = M_inv.tolist()
    print("affine_voltage2pixel = [")
    for row in affine_voltage2pixel:
        print("    [{:.8f}, {:.8f}, {:.8f}],".format(row[0], row[1], row[2]))
    print("]")

    print("\nInverse affine matrix (M_inv) as a list of lists:")
    print("[")
    for row in inverse_affine_voltage2pixel:
        print(f"    [{row[0]:.8f}, {row[1]:.8f}, {row[2]:.8f}],")
    print("]")
    return M_inv


def pixel_to_voltage(initial_pixel_coords, final_pixel_coords):
    # Convert initial and final pixel coordinates to homogeneous coordinates (x, y, 1)
    initial_pixel_coords_h = np.array(
        [initial_pixel_coords[0], initial_pixel_coords[1], 1.0]
    )
    final_pixel_coords_h = np.array([final_pixel_coords[0], final_pixel_coords[1], 1.0])

    # Calculate pixel drift
    pixel_drift = final_pixel_coords_h - initial_pixel_coords_h

    # Get the inverse affine transformation matrix
    M_inv = piezo_voltage_to_pixel_calibration()

    # Calculate the corresponding voltage drift using the inverse affine matrix
    voltage_drift_h = np.dot(M_inv, pixel_drift)  # No transpose needed

    # Update only the x and y components of the global coordinates
    final_voltage = np.array(
        global_coords
    )  # Start with all original global coordinates
    final_voltage[:2] += voltage_drift_h[:2]  # Update x and y components with drift

    print(f"Pixel drift: {pixel_drift[:2]}")
    print(f"Voltage drift: {voltage_drift_h[:2]}")
    print(f"Final voltage coordinates: {final_voltage.tolist()}")

    return final_voltage.tolist()


def do_optimize_SLM_calibation(nv_list, coords_key):
    repr_nv_sig = widefield.get_repr_nv_sig(nv_list)
    # Pixel optimization in parallel with widefield yellow
    if coords_key is None:
        num_reps = 50
        img_array = do_widefield_image_sample(nv_sig, num_reps=num_reps)

    opti_coords_list = []
    for nv in nv_list:
        # Pixel coords
        if coords_key is None:
            # imaging_laser = tb.get_laser_name(LaserKey.IMAGING)
            opti_coords = do_optimize_pixel(nv)
            # opti_coords = optimize.optimize_pixel_with_img_array(img_array, nv_sig=nv)
            # widefield.reset_all_drift()
            targeting.optimize_xyz_using_piezo(repr_nv_sig)
            widefield.reset_scanning_optics_drift()  # reset drift before optimizing next NV
        opti_coords_list.append(opti_coords)

    # Report back
    for opti_coords in opti_coords_list:
        r_opti_coords = [round(el, 3) for el in opti_coords]
        print(f"{r_opti_coords},")


# Load the saved NV coordinates and radii from the .npz file
def load_nv_coords(
    file_path="slmsuite/nv_blob_detection/nv_blob_filtered_multiple_nv302.npz",
    x_min=0,
    x_max=250,
    y_min=0,
    y_max=250,
):
    data = np.load(file_path, allow_pickle=True)
    nv_coordinates = data["nv_coordinates"]

    # Create a mask based on the min/max thresholds for x and y
    mask = (
        (nv_coordinates[:, 0] >= x_min)
        & (nv_coordinates[:, 0] <= x_max)
        & (nv_coordinates[:, 1] >= y_min)
        & (nv_coordinates[:, 1] <= y_max)
    )
    nv_coordinates_clean = nv_coordinates[mask]
    return nv_coordinates_clean


def load_thresholds(file_path="slmsuite/nv_blob_detection/threshold_list_nvs_162.npz"):
    with np.load(file_path) as data_file:
        thresholds = data_file["arr_0"]
    return thresholds


### Run the file

if __name__ == "__main__":
    # region Shared parameters

    green_coords_key = f"coords-{green_laser}"
    red_coords_key = f"coords-{red_laser}"
    pixel_coords_key = "pixel_coords"

    sample_name = "johnson"
    # z_coord = 3.85
    # magnet_angle = 90
    date_str = "2024_03_12"
    # global_coords = [None, None, z_coord]
    global_coords = [0.7, 0.2, 1.243]

    # Load NV pixel coordinates
    pixel_coords_list = load_nv_coords(
        # file_path="slmsuite/nv_blob_detection/nv_blob_filtered_162nvs_ref.npz"
        file_path="slmsuite/nv_blob_detection/nv_blob_filtered_116nvs_updated.npz",
    ).tolist()

    print(f"Number of NVs: {len(pixel_coords_list)}")
    print(f"Reference NV:{pixel_coords_list[0]}")

    # Define transformations using `transform_coords`
    green_coords_list = [
        [
            round(coord, 3)
            for coord in pos.transform_coords(
                nv_pixel_coords, CoordsKey.PIXEL, green_laser_aod
            )
        ]
        for nv_pixel_coords in pixel_coords_list
    ]

    red_coords_list = [
        [
            round(coord, 3)
            for coord in pos.transform_coords(
                nv_pixel_coords, CoordsKey.PIXEL, red_laser_aod
            )
        ]
        for nv_pixel_coords in pixel_coords_list
    ]

    # Optional: Print first coordinate set for verification
    print(f"First Green Laser Coordinates: {green_coords_list[0]}")
    print(f"First Red Laser Coordinates: {red_coords_list[0]}")
    # print(red_coords_list[0])
    # print(pixel_coords_list[8])

    # pixel_coords_list = [
    #     [125.000, 160.887],
    #     [75.302, 95.265],
    #     [199.053, 94.250],
    # ]
    # green_coords_list = [
    #     [109.504, 113.073],
    #     [115.361, 106.287],
    #     [102.128, 105.29],
    # ]
    # red_coords_list = [
    #     [74.367, 78.506],
    #     [78.919, 72.896],
    #     [68.183, 72.361],
    # ]

    num_nvs = len(pixel_coords_list)
    threshold_list = [15.5] * num_nvs
    # threshold_list = load_thresholds(
    #     file_path="slmsuite/nv_blob_detection/threshold_list_nvs_162.npz"
    # ).tolist()
    scc_duration_list = [140] * num_nvs
    scc_duration_list = [4 * round(el / 4) for el in scc_duration_list]
    scc_amp_list = [1] * num_nvs

    # nv_list[i] will have the ith coordinates from the above lists
    nv_list: list[NVSig] = []
    for ind in range(num_nvs):
        coords = {
            CoordsKey.SAMPLE: global_coords,
            CoordsKey.PIXEL: pixel_coords_list.pop(0),
            green_laser_aod: green_coords_list.pop(0),
            red_laser_aod: red_coords_list.pop(0),
        }
        nv_sig = NVSig(
            name=f"{sample_name}-nv{ind}_{date_str}",
            coords=coords,
            scc_duration=scc_duration_list[ind],
            scc_amp=scc_amp_list[ind],
            threshold=threshold_list[ind],
        )
        nv_list.append(nv_sig)
    # Additional properties for the representative NV
    nv_list[0].representative = True
    # nv_list[1].representative = True
    nv_sig = widefield.get_repr_nv_sig(nv_list)
    nv_sig.expected_counts = None
    # nv_sig.expected_counts = 2249.0
    # nv_sig.expected_counts = 3359.0
    nv_sig.expected_counts = 1050.0
    # num_nvs = len(nv_list)
    # print(f"Final NV List: {nv_list}")
    # Ensure data is defined before accessing it
    # data = None

    # try:
    #     pulse_gen.stream_load(seq_file, seq_args_string, num_reps)
    #     counts = np.array(data["counts"])[0] if data else None
    # except Exception as e:
    # print(f"Error occurred: {e}")
    # nv_inds = [0, 1]
    # nv_list = [nv_list[ind] for ind in range(num_nvs) if ind in nv_inds]
    # num_nvs = len(nv_list)
    # for nv in nv_list[::2]:
    # for nv in nv_list[num_nvs // 2 :]:
    #     nv.spin_flip = True
    # print([nv.spin_flip for nv in nv_list])

    # for nv in nv_list:
    #     nv.init_spin_flipped = True
    # nv_list[1].init_spin_flipped = True
    # nv_list[3].init_spin_flipped = True
    # seq_args = widefield.get_base_scc_seq_args(nv_list[:3], [0, 1])
    # print(seq_args)

    # nv_list = nv_list[::-1]  # flipping the order of NVs

    # endregion

    # region Coordinate printing

    # for nv in nv_list:
    #     pixel_drift = widefield.get_pixel_drift()
    #     # pixel_drift = [-el for el in pixel_drift]
    #     coords = widefield.get_nv_pixel_coords(nv, drift=pixel_drift)
    #     r_coords = [round(el, 3) for el in coords]
    #     print(f"{r_coords},")
    # for nv in nv_list:
    #     coords = widefield.set_nv_scanning_coords_from_pixel_coords(
    #         nv, green_laser, drift_adjust=False
    #     )
    #     r_coords = [round(el, 3) for el in coords]
    #     print(f"{r_coords},")
    # for nv in nv_list:
    #     coords = widefield.set_nv_scanning_coords_from_pixel_coords(
    #         nv, red_laser, drift_adjust=False
    #     )
    #     r_coords = [round(el, 3) for el in coords]
    #     print(f"{r_coords},")
    # sys.exit()

    # nv_list = [nv_list[
    # nv_list = [nv_list[2]]
    # nv_list = nv_list[: len(nv_list)]

    # endregion

    # region Functions to run

    email_recipient = "mccambria@berkeley.edu"
    do_email = False
    try:
        # pass
        kpl.init_kplotlib()
        # tb.init_safe_stop()

        # widefield.reset_all_drift()
        # widefield.reset_scanning_optics_drift()
        # pos.reset_drift()  # Reset z drift
        # widefield.set_pixel_drift(
        #     np.array([93.093, 120.507])  # New coords
        #     - np.array([96.549, 119.583])  # Original coords
        # )
        # widefield.reset_pixel_drift()
        # widefield.set_all_scanning_drift_from_pixel_drift()

        # do_optimize_z(nv_sig)
        # do_optimize_xyz(nv_sig)
        # pos.set_xyz_on_nv(nv_sig)
        # piezo_voltage_to_pixel_calibration()

        # Generate points for forward diagonal motion
        # x_values = np.linspace(0.3, -0.3, 6)
        # y_values = np.linspace(0.3, -0.3, 6)
        # Define the list of points in the triangle
        # Example coordinates for the triangle
        # p1 = (0.0, 0.0)
        # p2 = (-0.25, -0.25)
        # p3 = (0.50, 0)
        # p4 = (-0.25, 0.25)
        # points = [p1, p2, p3, p4]

        # for point in points:
        #     x, y = point
        #     nv_sig.coords[CoordsKey.SAMPLE][0] += x
        #     nv_sig.coords[CoordsKey.SAMPLE][1] += y
        #     print(nv_sig.coords[CoordsKey.SAMPLE])
        #     do_scanning_image_sample(nv_sig)

        # Move diagonally forward
        # for x, y in zip(x_values, y_values):
        # nv_sig.coords[CoordsKey.SAMPLE][0] = x
        # nv_sig.coords[CoordsKey.SAMPLE][1] = y
        # do_scanning_image_sample(nv_sig)

        # for z in np.linspace(-0.6, 0.0, 11):
        #     nv_sig.coords[CoordsKey.SAMPLE][2] = z
        #     do_scanning_image_sample(nv_sig)
        # do_widefield_image_sample(nv_sig, 50)

        # do_scanning_image_sample(nv_sig)
        # do_scanning_image_sample_zoom(nv_sig)
        do_widefield_image_sample(nv_sig, 50)
        # do_widefield_image_sample(nv_sig, 100)

        # do_image_nv_list(nv_list)
        # do_image_single_nv(nv_sig)

        # for nv_sig in nv_list:
        # widefield.reset_all_drift()
        # do_optimize_pixel(nv_sig)
        # do_image_single_nv(nv_sig)

        # optimize.optimize_pixel_and_z(nv_sig, do_plot=True)
        # do_image_nv_list(nv_list)
        # for ind in range(20):
        # do_optimize_pixel(nv_sig)
        # do_optimize_z(nv_sig)
        # do_optimize_green(nv_sig)
        # do_optimize_red(nv_sig)

        # widefield.reset_all_drift()
        # coords_key = None  # Pixel coords
        # coords_key = green_laser
        # coords_key = red_laser
        # do_optimize_loop(nv_list, coords_key, scanning_from_pixel=False)
        # optimize_slm_Phase_calibration(nv_sig, target_coords=target_coords)

        # nv_list = nv_list[::-1]
        # do_charge_state_histograms(nv_list)
        # do_charge_state_conditional_init(nv_list)
        # do_check_readout_fidelity(nv_list)

        # do_resonance_zoom(nv_list)
        # do_rabi(nv_l ist)
        # do_resonance(nv_list)
        # do_spin_echo(nv_list)s

        # do_power_rabi(nv_list)
        # do_correlation_test(nv_list)
        # do_ramsey(nv_list)
        # do_sq_relaxation(nv_list)
        # do_dq_relaxation(nv_list)
        # do_xy8(nv_list)
        # do_detect_cosmic_rays(nv_list)
        # do_check_readout_fidelity(nv_list)
        # do_charge_quantum_jump(nv_list)
        # do_ac_stark(nv_list)

        # do_opx_constant_ac()
        # do_opx_square_wave()

        # nv_list = nv_list[::-1]
        # do_scc_snr_check(nv_list)
        # do_optimize_scc_duration(nv_list)
        # do_optimize_scc_amp(nv_list)
        # do_crosstalk_check(nv_sig)
        # do_spin_pol_check(nv_sig)
        # do_calibrate_green_red_delay()
        # do_simple_correlation_test(nv_list)

        # do_simple_correlation_test(nv_list)

        # for nv in nv_list:
        #     nv.spin_flip = False
        # Get the indices of well-separated NVs
        # selected_indices = widefield.select_well_separated_nvs(nv_list, 15)
        # for index in selected_indices:
        #     nv = nv_list[index]
        #     nv.spin_flip = True
        # do_simple_correlation_test(nv_list)

        # for nv in nv_list:
        #     nv.spin_flip = False
        # for nv in nv_list[: num_nvs // 2]:
        #     nv.spin_flip = True
        # do_simple_correlation_test(nv_list)
        # for nv in nv_list:
        #     nv.spin_flip = False
        # for nv in nv_list[num_nvs // 2 :]:
        #     nv.spin_flip = True
        # # do_simple_correlation_test(nv_list)
        # nv_list = nv_list[::-1]
        # spin_flips = [nv.spin_flip for nv in nv_list]
        # print(spin_flips)
        # do_simple_correlation_test(nv_list)

        # Performance testing
        # data = dm.get_raw_data(file_id=1513523816819, load_npz=True)
        # img_array = np.array(data["ref_img_array"])
        # num_nvs = len(nv_list)
        # counts = [
        #     widefield.integrate_counts(
        #         img_array, widefield.get_nv_pixel_coords(nv_list[ind])
        #     )
        #     for ind in range(num_nvs)
        # ]
        # res_thresh = [counts[ind] > nv_list[ind].threshold for ind in range(num_nvs)]
        # res_mle = widefield.charge_state_mle(nv_list, img_array)
        # num_reps = 1000
        # start = time.time()
        # for ind in range(num_reps):
        #     widefield.charge_state_mle(nv_list, img_array)
        # stop = time.time()
        # print(stop - start)
        # print(res_thresh)
        # print(res_mle)s
        # print([res_mle[ind] == res_thresh[ind] for ind in range(num_nvs)])

        # region Cleanup
        # do_optimize_green(nv_sig)
        # do_optimize_red(nv_sig)
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
        tb.reset_safe_stop()
        plt.show(block=True)

    # endregion
