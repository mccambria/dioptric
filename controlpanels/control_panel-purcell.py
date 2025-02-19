# -*- coding: utf-8 -*-
"""
Control panel for the PC Rabi

Created on June 16th, 2023

@author: mccambria
@author: saroj chand
"""

### Imports
import os
import random
import sys
import time
from random import shuffle

import cv2
import matplotlib.pyplot as plt
import numpy as np

from majorroutines import targeting
from majorroutines.widefield import (
    ac_stark,
    calibrate_iq_delay,
    charge_monitor,
    charge_state_conditional_init,
    charge_state_histograms,
    charge_state_histograms_images,
    correlation_test,
    crosstalk_check,
    image_sample,
    optimize_amp_duration_charge_state_histograms,
    optimize_charge_state_histograms_mcc,
    optimize_scc,
    optimize_scc_amp_duration,
    power_rabi,
    power_rabi_scc_snr,
    rabi,
    ramsey,
    relaxation_interleave,
    resonance,
    scc_snr_check,
    simple_correlation_test,
    spin_echo,
    spin_pol_check,
    xy8,
)

# from slmsuite import optimize_slm_calibration
from utils import common, widefield
from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import positioning as pos
from utils import tool_belt as tb
from utils.constants import Axes, CoordsKey, NVSig, VirtualLaserKey

green_laser = "laser_INTE_520"
red_laser = "laser_COBO_638"
yellow_laser = "laser_OPTO_589"
green_laser_aod = f"{green_laser}_aod"
red_laser_aod = f"{red_laser}_aod"

### Major Routines


def do_widefield_image_sample(nv_sig, num_reps=1):
    return image_sample.widefield_image(nv_sig, num_reps)


def do_scanning_image_sample(nv_sig):
    scan_range = 15
    num_steps = 15
    image_sample.scanning(nv_sig, scan_range, scan_range, num_steps)


def do_scanning_image_full_roi(nv_sig):
    total_range = 24
    scan_range = 8
    num_steps = 8
    image_sample.scanning_full_roi(nv_sig, total_range, scan_range, num_steps)


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
    # 50 ms
    num_reps = 100
    num_runs = 20

    # 100 ms
    # num_reps = 100
    # num_runs = 20

    # 200 ms
    # num_reps = 50
    # num_runs = 20

    # Test
    # num_runs = 2

    return charge_state_histograms.main(
        nv_list, num_reps, num_runs, do_plot_histograms=False
    )


def do_optimize_pol_duration(nv_list):
    num_steps = 22
    num_reps = 10
    num_runs = 400
    # num_reps = 5
    # num_runs = 2
    min_duration = 32
    max_duration = 956
    return optimize_charge_state_histograms_mcc.optimize_pol_duration(
        nv_list, num_steps, num_reps, num_runs, min_duration, max_duration
    )


def do_optimize_pol_amp(nv_list):
    num_steps = 24
    # num_reps = 150
    # num_runs = 5
    num_reps = 8
    num_runs = 300
    min_amp = 0.6
    max_amp = 1.4
    return optimize_charge_state_histograms_mcc.optimize_pol_amp(
        nv_list, num_steps, num_reps, num_runs, min_amp, max_amp
    )


def do_optimize_readout_duration(nv_list):
    num_steps = 16
    # num_reps = 150
    # num_runs = 5
    num_reps = 10
    num_runs = 225
    min_duration = 12e6
    max_duration = 108e6
    return optimize_charge_state_histograms_mcc.optimize_readout_duration(
        nv_list, num_steps, num_reps, num_runs, min_duration, max_duration
    )


def do_optimize_readout_amp(nv_list):
    num_steps = 21
    # num_reps = 150
    # num_runs = 5
    num_reps = 10
    num_runs = 400
    min_amp = 0.8
    max_amp = 1.2
    return optimize_charge_state_histograms_mcc.optimize_readout_amp(
        nv_list, num_steps, num_reps, num_runs, min_amp, max_amp
    )


def optimize_readout_amp_and_duration(nv_list):
    num_amp_steps = 16
    num_dur_steps = 5
    num_reps = 3
    num_runs = 1000
    min_amp = 0.9
    max_amp = 1.2
    min_duration = 12e6
    max_duration = 60e6
    return (
        optimize_amp_duration_charge_state_histograms.optimize_readout_amp_and_duration(
            nv_list,
            num_amp_steps,
            num_dur_steps,
            num_reps,
            num_runs,
            min_amp,
            max_amp,
            min_duration,
            max_duration,
        )
    )


def do_charge_state_histograms_images(nv_list, vary_pol_laser=False):
    aom_voltage_center = 1.0
    aom_voltage_range = 0.1
    num_steps = 6
    # num_reps = 15
    # num_reps = 100
    # num_runs = 50
    # num_runs = 100
    num_reps = 20
    num_runs = 60
    return charge_state_histograms_images.main(
        nv_list,
        num_steps,
        num_reps,
        num_runs,
        aom_voltage_center,
        aom_voltage_range,
        vary_pol_laser,
        aom_voltage_center,
        aom_voltage_range,
    )


def do_charge_state_conditional_init(nv_list):
    num_reps = 20
    num_runs = 10
    # num_runs = 400
    return charge_state_conditional_init.main(nv_list, num_reps, num_runs)


def do_optimize_green(nv_sig):
    ret_vals = targeting.optimize(nv_sig, coords_key=green_laser_aod)
    opti_coords = ret_vals[0]
    return opti_coords


def do_optimize_red(nv_sig, ref_nv_sig):
    opti_coords = []
    # axes_list = [Axes.X, Axes.Y]
    axes_list = [Axes.Y, Axes.X]
    # shuffle(axes_list)
    for ind in range(1):
        axes = axes_list[ind]
        ret_vals = targeting.optimize(nv_sig, coords_key=red_laser_aod, axes=axes)
        opti_coords.append(ret_vals[0])
        # Compensate for drift after first optimization along X axis
        # if ind == 0:
        #     do_compensate_for_drift(ref_nv_sig)
    return opti_coords


def do_optimize_z(nv_sig):
    ret_vals = targeting.optimize(nv_sig, coords_key=CoordsKey.Z)
    opti_coords = ret_vals[0]
    return opti_coords


def do_compensate_for_drift(nv_sig):
    return targeting.compensate_for_drift(nv_sig)


def do_optimize_xyz(nv_sig, do_plot=True):
    targeting.optimize_xyz_using_piezo(
        nv_sig, do_plot=do_plot, axes_to_optimize=[0, 1, 2]
    )


def do_optimize_sample(nv_sig):
    opti_coords = targeting.optimize_sample(nv_sig)
    if not opti_coords:
        print("Optimization failed: No coordinates found.")
    return opti_coords


# def do_optimize_sample(nv_sig):
#     opti_coords = targeting.optimize_sample(nv_sig)
# return opti_coords


def do_optimize_pixel(nv_sig):
    ret_vals = targeting.optimize(nv_sig, coords_key=CoordsKey.PIXEL)
    opti_coords = ret_vals[0]
    return opti_coords


def do_optimize_loop(nv_list, coords_key):
    repr_nv_sig = widefield.get_repr_nv_sig(nv_list)

    opti_coords_list = []
    for nv in nv_list:
        if coords_key == green_laser:
            opti_coords = do_optimize_green(nv)
        elif coords_key == red_laser:
            opti_coords = do_optimize_red(nv, repr_nv_sig)
        # Adjust for the drift that may have occurred since beginning the loop
        do_compensate_for_drift(repr_nv_sig)
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


def optimize_scc_amp_and_duration(nv_list):
    # num_amp_steps = 15
    # num_dur_steps = 17
    # num_reps = 1
    # num_runs = 1500
    # num_runs = 400  # Short test version
    # num_runs = 5
    # min_amp = 0.75
    # max_amp = 1.25
    min_duration = 16
    max_duration = 272
    num_dur_steps = 17

    # Single amp
    num_reps = 11
    num_runs = 200
    min_amp = 1.0
    max_amp = 1.0
    num_amp_steps = 1
    # min_duration = 32
    # max_duration = 496
    # num_dur_steps = 30

    return optimize_scc_amp_duration.optimize_scc_amp_and_duration(
        nv_list,
        num_amp_steps,
        num_dur_steps,
        num_reps,
        num_runs,
        min_amp,
        max_amp,
        min_duration,
        max_duration,
    )


def do_optimize_scc_duration(nv_list):
    min_tau = 48
    max_tau = 304
    num_steps = 17
    num_reps = 15

    # num_runs = 20 * 25
    num_runs = 200
    # num_runs = 50
    # num_runs = 2

    optimize_scc.optimize_scc_duration(
        nv_list, num_steps, num_reps, num_runs, min_tau, max_tau
    )


def do_optimize_scc_amp(nv_list):
    min_tau = 0.6
    max_tau = 1.2
    num_steps = 16
    num_reps = 15
    num_runs = 200
    # num_runs = 2
    optimize_scc.optimize_scc_amp(
        nv_list, num_steps, num_reps, num_runs, min_tau, max_tau
    )


def do_scc_snr_check(nv_list):
    num_reps = 240
    num_runs = 60
    # num_runs = 200
    # num_runs = 160 * 4
    # num_runs = 3
    scc_snr_check.main(nv_list, num_reps, num_runs, uwave_ind_list=[0, 1])


def do_power_rabi_scc_snr(nv_list):
    num_reps = 10
    num_runs = 200
    power_range = 6
    num_steps = 16
    # num_runs = 200
    # num_runs = 160 * 4
    # num_runs = 3
    power_rabi_scc_snr.main(
        nv_list, num_steps, num_reps, num_runs, power_range, uwave_ind_list=[0, 1]
    )


def do_simple_correlation_test(nv_list):
    # Run this for a quick test experiment to debug.
    # num_reps = 200
    # num_runs = 5
    # simple_correlation_test.main(nv_list, num_reps, num_runs)

    # # Uncomment this to set up spin flips
    # # fmt: off    # snr_list = [0.208, 0.202, 0.186, 0.198, 0.246, 0.211, 0.062, 0.178, 0.161, 0.192, 0.246, 0.139, 0.084, 0.105, 0.089, 0.198, 0.242, 0.068, 0.134, 0.214, 0.185, 0.149, 0.172, 0.122, 0.128, 0.205, 0.202, 0.174, 0.192, 0.172, 0.145, 0.169, 0.135, 0.184, 0.204, 0.174, 0.13, 0.174, 0.06, 0.178, 0.237, 0.167, 0.198, 0.147, 0.176, 0.154, 0.118, 0.157, 0.113, 0.202, 0.084, 0.117, 0.117, 0.182, 0.157, 0.121, 0.181, 0.124, 0.135, 0.121, 0.15, 0.099, 0.107, 0.198, 0.09, 0.153, 0.159, 0.153, 0.177, 0.182, 0.139, 0.202, 0.141, 0.173, 0.114, 0.057, 0.193, 0.172, 0.191, 0.165, 0.076, 0.116, 0.072, 0.105, 0.152, 0.139, 0.186, 0.049, 0.197, 0.072, 0.072, 0.158, 0.175, 0.142, 0.132, 0.173, 0.063, 0.172, 0.141, 0.147, 0.138, 0.151, 0.169, 0.147, 0.148, 0.117, 0.149, 0.07, 0.135, 0.152, 0.163, 0.189, 0.116, 0.124, 0.129, 0.158, 0.079]
    # # fmt: on
    # snr_sorted_nv_inds = np.argsort(snr_list)[::-1]
    # parity = 1
    # for ind in snr_sorted_nv_inds:
    #     nv_list[ind].spin_flip = parity == -1
    #     parity *= -1

    selected_indices = widefield.select_half_left_side_nvs(nv_list)
    for index in selected_indices:
        nv = nv_list[index]
        nv.spin_flip = True
    print(f"Assigned spin_flip to {len(selected_indices)}")
    # print(f"Assigned spin_flip to {selected_indices}")

    # Run this for the main experiment, ~15 hours
    # num_steps = 200
    num_reps = 200
    num_runs = 400
    for _ in range(5):
        simple_correlation_test.main(nv_list, num_reps, num_runs)


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
    # freq_range = 0.300
    num_steps = 60
    # num_steps = 80
    # Single ref
    # num_reps = 8
    num_runs = 600
    # num_runs = 750
    # num_runs = 350
    # num_runs = 50
    # num_runs = 10
    # num_runs = 2

    # Both refs
    num_reps = 2
    # num_runs = 300

    # num_runs = 2
    resonance.main(nv_list, num_steps, num_reps, num_runs, freq_center, freq_range)

    # for _ in range(2):
    #     resonance.main(nv_list, num_steps, num_reps, num_runs, freq_center, freq_range)


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
    num_runs = 400
    # num_runs = 100
    # num_runs = 20
    # num_runs = 5

    # uwave_ind_list = [1]
    uwave_ind_list = [0, 1]
    rabi.main(nv_list, num_steps, num_reps, num_runs, min_tau, max_tau, uwave_ind_list)
    # for _ in range(2):
    #     rabi.main(
    #         nv_list, num_steps, num_reps, num_runs, min_tau, max_tau, uwave_ind_list
    #     )
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
    num_reps = 24
    # num_reps = 20
    num_runs = 300
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
    revival_period = int(51.5e3 / 2)
    min_tau = 200
    taus = []
    revival_width = 5e3
    decay = np.linspace(min_tau, min_tau + revival_width, 6)
    taus.extend(decay.tolist())
    gap = np.linspace(min_tau + revival_width, revival_period - revival_width, 7)
    taus.extend(gap[1:-1].tolist())
    first_revival = np.linspace(
        revival_period - revival_width, revival_period + revival_width, 61
    )
    taus.extend(first_revival.tolist())
    gap = np.linspace(
        revival_period + revival_width, 2 * revival_period - revival_width, 7
    )
    taus.extend(gap[1:-1].tolist())
    second_revival = np.linspace(
        2 * revival_period - revival_width, 2 * revival_period + revival_width, 11
    )
    taus.extend(second_revival.tolist())
    taus = [round(el / 4) * 4 for el in taus]

    # Remove duplicates and sort
    taus = sorted(set(taus))

    # Experiment settings
    num_steps = len(taus)

    # Automatic taus setup, linear spacing
    # min_tau = 200
    # max_tau = 84e3 + min_tau
    # num_steps = 29

    num_reps = 4
    num_runs = 200
    # num_runs = 3
    # spin_echo.main(nv_list, num_steps, num_reps, num_runs, min_tau, max_tau)
    # spin_echo.main(nv_list, num_steps, num_reps, num_runs, taus=taus)
    for ind in range(5):
        spin_echo.main(nv_list, num_steps, num_reps, num_runs, taus=taus)


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
    num_reps = 40
    num_runs = 200
    # num_runs = 2
    # dark_time = 1e9 # 1s
    # dark_time = 10e6  # 10ms
    dark_time_1 = 1e6  # 1 ms in nanoseconds
    dark_time_2 = 1e9  # 1 s in nanoseconds
    # charge_monitor.detect_cosmic_rays(nv_list, num_reps, num_runs, dark_time)
    for _ in range(6):
        charge_monitor.detect_cosmic_rays(
            nv_list, num_reps, num_runs, dark_time_1, dark_time_2
        )
    # dark_times = [100e6, 500e6, 5e6, 506, 250e6]
    # for dark_time in dark_times:
    #     charge_monitor.detect_cosmic_rays(nv_list, num_reps, num_runs, dark_time)


def do_check_readout_fidelity(nv_list):
    num_reps = 200
    num_runs = 20

    charge_monitor.check_readout_fidelity(nv_list, num_reps, num_runs)


def do_charge_quantum_jump(nv_list):
    num_reps = 3000
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
    # opx.constant_ac(
    #     [],  # Digital channels
    #     [7],  # Analog channels
    #     [0.45],  # Analog voltages
    #     [0],  # Analog frequencies
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
    #     [105.0, 105.0],  # Analog frequencies
    # )
    # Green + red
    # opx.constant_ac(
    #     [4, 1],  # Digital channels
    #     [3, 4, 2, 6],  # Analog channels
    #     [0.19, 0.19, 0.17, 0.17],  # Analog voltages;
    #     # [109.409, 111.033, 73.0, 77.3],  # Analog frequencies
    #     # [108.907, 112.362, 74.95, 78.65],  # Analog frequencies
    #     [105.181, 105.867, 68.123, 75.932],
    # )
    #   green_coords_list = [
    #     [107.336, 107.16],
    #     [106.36, 103.736],
    #     [111.622, 109.491],
    #     [102.181, 111.867],
    # ]
    # red_coords_list = [
    #     [72.917, 73.798],
    #   71.352, 69.193,
    # [75.818, 73.939],
    # [67.923, 76.832],
    # ]
    # red
    # opx.constant_ac(
    #     [1],  # Digital channels
    #     [2, 6],  # Analog channels
    #     [0.17, 0.17],  # Analog voltages
    #     [73.166, 72.941],  # Analog frequencies
    # )
    # Green + yellow
    opx.constant_ac(
        [4],  # Digital channels
        [3, 4, 7],  # Analog channels
        [0.19, 0.19, 0.45],  # Analog voltages
        [107, 107, 0],  # Analog frequencies
    )
    # Red + green + Yellow
    # opx.constant_ac(
    #     [4, 1],  # Digital channels1
    #     [3, 4, 2, 6, 7],  # Analog channels
    #     [0.19, 0.19, 0.17, 0.17, 0.40],  # Analog voltages
    #     [107, 107, 72, 72, 0],  # Analog frequencies
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
        [[3.0, 0.0], [2.0, 1.0], [1.5, -0.5]], dtype="float32"
    )  # Voltage system coordinates
    # cal_pixel_coords = np.array(
    #     [[81.109, 110.177], [64.986, 94.177], [96.577, 95.047]], dtype="float32"
    # )
    cal_pixel_coords = np.array(
        [
            [50.133, 115.925],
            [91.972, 64.584],
            [130.875, 153.92],
        ],
        dtype="float32",
    )
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
    sample_name = "cannon"
    # magnet_angle = 90
    date_str = "2025_01_29"
    sample_coords = [0.0, 0.4]
    z_coord = 1.5
    # Load NV pixel coordinates1
    pixel_coords_list = load_nv_coords(
        # file_path="slmsuite/nv_blob_detection/nv_blob_filtered_160nvs_reordered.npz",
        # file_path="slmsuite/nv_blob_detection/nv_blob_shallow_52nvs_reordered.npz",
        # file_path="slmsuite/nv_blob_detection/nv_blob_shallow_161nvs_reordered.npz",
        file_path="slmsuite/nv_blob_detection/nv_blob_shallow_148nvs_reordered.npz",
    ).tolist()
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

    # Print first coordinate set for verification
    # print(f"Number of NVs: {red_coords_list}")
    print(f"Number of NVs: {len(pixel_coords_list)}")
    print(f"Reference NV:{pixel_coords_list[0]}")
    print(f"Green Laser Coordinates: {green_coords_list[0]}")
    print(f"Red Laser Coordinates: {red_coords_list[0]}")
    # sys.exit()
    # pixel_coords_list = [
    #     [117.516, 129.595],
    #     [241.12, 24.247],
    #     [15.121, 53.43],
    #     [85.847, 227.993],
    # ]
    # green_coords_list = [
    #     [107.266, 107.516],
    #     [95.376, 94.823],
    #     [119.098, 100.353],
    #     [109.422, 118.221],
    # ]
    # red_coords_list = [
    #     [72.017, 73.008],
    #     [62.148, 62.848],
    #     [81.574, 67.148],
    #     [74.061, 81.78],
    # ]
    num_nvs = len(pixel_coords_list)
    threshold_list = [25.5] * num_nvs
    # threshold_list = load_thresholds
    #     file_path="slmsuite/nv_blob_detection/threshold_list_nvs_162.npz"
    # ).tolist()

    # polrizaton data
    charge_pol_amps_data = dm.get_raw_data(file_id=1726176332479)
    charge_pol_amps = charge_pol_amps_data["optimal step values"]
    # print(charge_pol_amps)
    # scc_data = dm.get_raw_data(file_id=1724869770494)
    scc_data = dm.get_raw_data(file_id=1725870710271)
    scc_optimal_durations = scc_data["optimal_durations"]
    # Cross pattern
    # scc_duration_list = list(scc_optimal_durations.values())
    # scc_amp_list = list(scc_optimal_amplitudes.values())

    # fmt: off
    # amps = 1.0, red calibration
    # scc_duration_list = [168, 160, 164, 124, 188, 132, 116, 124, 160, 160, 164, 120, 140, 144, 124, 136, 136, 88, 152, 140, 140, 116, 104, 120, 112, 164, 136, 112, 96, 112, 140, 144, 196, 192, 120, 140, 228, 140, 32, 140, 148, 108, 164, 152, 132, 140, 176, 132, 136, 120, 112, 108, 144, 116, 132, 36, 192, 84, 148, 112, 132, 152, 176, 176, 176, 112, 120, 140, 168, 140, 92, 132, 92, 124, 68, 32, 92, 148, 164, 104, 32, 148, 188, 32, 112, 148, 168, 64, 140, 140, 96, 124, 176, 108, 108, 216, 216, 116, 112, 132, 148, 132, 132, 140, 160, 132, 148, 192, 160, 116, 140, 120, 152, 140, 144, 124, 160]
    # scc_duration_list = [168, 184, 220, 136, 140, 104, 104, 144, 240, 188, 160, 148, 116, 164, 124, 140, 132, 104, 304, 184, 144, 148, 116, 68, 132, 120, 112, 124, 116, 148, 212, 144, 132, 172, 116, 160, 304, 144, 60, 180, 100, 112, 172, 192, 144, 184, 292, 200, 96, 116, 156, 144, 144, 80, 160, 160, 168, 76, 176, 136, 172, 192, 264, 140, 104, 112, 140, 176, 208, 148, 116, 140, 80, 152, 140, 116, 96, 120, 112, 96, 48, 188, 48, 84, 96, 228, 172, 172, 124, 96, 128, 120, 196, 104, 88, 140, 80, 116, 112, 160, 120, 140, 112, 148, 108, 140, 152, 292, 124, 116, 140, 140, 160, 212, 140, 140, 196]
    # scc_duration_list = [112, 100, 92, 84, 144, 100, 100, 80, 108, 116, 92, 96, 108, 100, 88, 112, 108, 76, 76, 100, 132, 84, 92, 68, 76, 116, 124, 80, 100, 84, 76, 108, 128, 192, 92, 84, 92, 84, 108, 96, 132, 104, 116, 92, 100, 84, 92, 72, 84, 100, 116, 72, 124, 96, 84, 72, 164, 100, 56, 76, 64, 116, 92, 144, 172, 96, 60, 84, 100, 116, 80, 112, 88, 80, 64, 116, 100, 120, 112, 112, 128, 96, 108, 100, 108, 84, 144, 84, 128, 92, 108, 116, 148, 120, 88, 168, 64, 124, 104, 116, 100, 124, 112, 124, 120, 100, 172, 116, 124, 84, 92, 116, 80, 96, 88, 80, 92]
    # scc_duration_list = [112, 100, 112, 76, 160, 108, 100, 92, 96, 100, 84, 92, 120, 108, 72, 100, 108, 72, 72, 124, 116, 84, 80, 80, 84, 156, 140, 92, 116, 72, 80, 124, 124, 128, 112, 84, 84, 92, 104, 104, 164, 92, 100, 92, 124, 72, 96, 100, 128, 104, 104, 68, 124, 92, 124, 100, 132, 100, 84, 132, 80, 104, 80, 172, 172, 116, 92, 92, 112, 124, 80, 136, 96, 104, 60, 88, 128, 144, 116, 116, 180, 96, 84, 108, 84, 100, 124, 272, 152, 76, 100, 108, 128, 116, 92, 152, 124, 140, 108, 120, 132, 156, 108, 160, 124, 96, 180, 100, 144, 92, 124, 116, 92, 112, 124, 108, 108]
    # scc_duration_list = [136, 116, 116, 84, 180, 104, 108, 96, 84, 108, 128, 72, 144, 116, 84, 100, 116, 64, 84, 124, 116, 88, 92, 84, 80, 180, 132, 92, 120, 108, 92, 124, 108, 164, 132, 144, 100, 100, 144, 128, 216, 96, 124, 100, 84, 60, 92, 104, 108, 104, 96, 128, 116, 124, 88, 100, 168, 88, 72, 100, 76, 172, 44, 136, 272, 116, 100, 172, 128, 160, 80, 112, 104, 128, 104, 132, 80, 136, 112, 100, 128, 144, 136, 116, 96, 100, 200, 140, 128, 72, 108, 152, 212, 100, 88, 160, 124, 124, 124, 176, 272, 168, 184, 272, 164, 228, 208, 172, 272, 272, 264, 228, 216, 136, 176, 272, 164]
    # scc_duration_list = [136, 112, 124, 88, 164, 104, 216, 84, 92, 116, 136, 88, 92, 120, 108, 100, 124, 52, 92, 124, 124, 100, 104, 80, 68, 156, 160, 108, 124, 104, 100, 116, 136, 168, 116, 168, 116, 116, 84, 156, 156, 84, 116, 80, 92, 64, 84, 108, 124, 120, 108, 172, 124, 136, 84, 128, 136, 108, 76, 100, 80, 108, 68, 156, 272, 112, 84, 180, 156, 184, 84, 108, 72, 128, 120, 120, 80, 140, 132, 88, 116, 120, 144, 92, 88, 112, 164, 128, 128, 64, 112, 196, 164, 92, 104, 168, 108, 132, 128, 196, 184, 164, 148, 272, 116, 216, 212, 236, 272, 204, 248, 272, 116, 176, 128, 232, 272]
    # scc_duration_list = [128, 124, 136, 84, 144, 112, 124, 100, 108, 116, 140, 84, 120, 112, 112, 100, 116, 68, 100, 124, 136, 128, 100, 88, 80, 160, 144, 112, 112, 108, 108, 136, 124, 168, 124, 172, 136, 116, 84, 200, 144, 108, 124, 92, 100, 64, 96, 116, 92, 112, 100, 188, 188, 124, 92, 136, 140, 108, 80, 92, 84, 92, 76, 164, 272, 144, 92, 272, 160, 172, 92, 108, 80, 140, 140, 108, 88, 160, 120, 108, 140, 140, 148, 100, 100, 108, 164, 272, 116, 64, 164, 136, 152, 100, 104, 180, 96, 140, 164, 144, 272, 172, 136, 272, 136, 244, 272, 272, 272, 172, 272, 228, 120, 196, 144, 272, 180]
    # scc_duration_list = [116, 108, 108, 72, 152, 104, 236, 96, 76, 108, 116, 84, 100, 108, 84, 92, 116, 68, 80, 104, 124, 108, 92, 76, 64, 152, 124, 88, 108, 92, 80, 112, 120, 164, 108, 116, 84, 116, 80, 124, 164, 92, 116, 80, 96, 64, 84, 116, 88, 100, 84, 128, 128, 108, 84, 144, 136, 92, 64, 104, 80, 104, 80, 124, 272, 100, 76, 108, 128, 128, 76, 120, 56, 104, 108, 96, 92, 136, 124, 100, 100, 108, 100, 84, 88, 92, 200, 116, 120, 72, 116, 116, 180, 112, 96, 136, 92, 108, 96, 196, 216, 136, 124, 260, 112, 164, 272, 140, 272, 128, 272, 272, 132, 192, 172, 188, 272]
    # scc_duration_list = [128, 108, 100, 80, 160, 100, 92, 88, 84, 116, 112, 92, 104, 100, 96, 104, 100, 80, 84, 100, 128, 92, 84, 72, 64, 164, 136, 92, 124, 92, 96, 124, 116, 148, 112, 112, 92, 116, 80, 116, 172, 80, 124, 72, 84, 64, 116, 100, 72, 100, 92, 128, 100, 96, 84, 124, 136, 100, 92, 100, 84, 16, 92, 124, 272, 96, 84, 124, 156, 128, 72, 124, 64, 116, 120, 136, 92, 160, 108, 80, 84, 108, 92, 92, 100, 136, 160, 124, 112, 56, 128, 128, 204, 108, 104, 152, 84, 108, 100, 144, 208, 144, 132, 272, 132, 272, 252, 124, 272, 128, 208, 208, 92, 144, 136, 160, 272]
    # scc_duration_list = [108, 100, 92, 72, 176, 108, 108, 80, 64, 120, 116, 88, 120, 108, 92, 96, 108, 60, 72, 88, 124, 80, 84, 72, 56, 140, 120, 92, 108, 76, 80, 104, 124, 136, 100, 108, 84, 116, 64, 112, 164, 80, 108, 80, 72, 48, 80, 112, 100, 108, 84, 112, 92, 108, 100, 132, 160, 76, 88, 116, 80, 92, 92, 124, 272, 92, 120, 116, 144, 116, 64, 136, 72, 112, 100, 88, 80, 112, 108, 84, 92, 144, 120, 92, 72, 104, 188, 100, 116, 60, 108, 104, 196, 84, 108, 120, 100, 112, 92, 172, 188, 124, 128, 272, 112, 272, 272, 160, 272, 144, 240, 272, 132, 172, 272, 272, 204]
    # scc_duration_list = [128, 104, 92, 84, 152, 124, 128, 80, 100, 116, 108, 88, 120, 100, 92, 100, 112, 60, 76, 92, 164, 68, 84, 84, 64, 136, 136, 76, 92, 72, 76, 116, 144, 180, 96, 92, 96, 124, 80, 100, 164, 80, 108, 80, 92, 80, 84, 96, 80, 100, 92, 64, 116, 100, 84, 76, 188, 92, 72, 72, 72, 72, 72, 184, 140, 80, 68, 116, 160, 112, 72, 132, 84, 108, 48, 108, 96, 124, 112, 84, 96, 84, 84, 84, 84, 80, 124, 272, 124, 72, 100, 100, 160, 96, 72, 204, 72, 128, 84, 120, 116, 108, 128, 136, 108, 104, 148, 128, 144, 96, 100, 108, 72, 100, 80, 88, 80]
    # scc_duration_list = [136, 108, 96, 92, 208, 124, 100, 92, 88, 112, 108, 92, 108, 92, 100, 100, 116, 116, 64, 100, 136, 68, 92, 72, 60, 124, 116, 72, 92, 64, 72, 120, 124, 232, 92, 96, 96, 116, 84, 96, 144, 80, 116, 84, 100, 80, 84, 72, 80, 108, 84, 72, 136, 108, 100, 100, 188, 92, 64, 84, 60, 100, 76, 184, 152, 92, 68, 108, 160, 108, 72, 132, 80, 112, 60, 76, 104, 116, 108, 96, 96, 92, 84, 92, 84, 64, 124, 100, 124, 80, 108, 96, 136, 80, 80, 188, 188, 128, 84, 116, 124, 100, 100, 124, 112, 84, 196, 108, 124, 100, 100, 104, 76, 104, 84, 84, 84]
    # 103nvs
    # include_inds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 13, 14, 15, 16, 17, 18, 21, 22, 23, 24, 25, 26, 28, 29, 31, 33, 36, 37, 38, 39, 40, 42, 44, 46, 47, 48, 49, 51, 52, 53, 56, 57, 58, 60, 62, 64, 65, 66, 68, 69, 70, 71, 72, 74, 75, 77, 78, 79, 80, 83, 88, 90, 91, 92, 94, 95, 96, 97, 100, 101, 103, 105, 106, 107, 108, 109, 110, 112, 114, 116, 118, 120, 121, 122, 123, 124, 125, 126, 128, 131, 134, 136, 138, 140, 141, 145, 146, 147, 152, 153, 154, 156, 157, 158]
    # 117nvs
    # include_inds = [0,1,2,3,5,6,7,8,13,14,15,16,17,18,20,21,22,23,24,25,26,28,29,31,32,33,34,36,37,39,42,44,45,46,47,48,49,51,52,53,55,56,57,58,60,61,62,64,65,66,68,69,70,71,72,73,74,75,77,79,83,84,85,88,89,90,91,92,94,95,96,97,99,100,101,102,103,105,106,107,108,109,110,111,113,114,116,117,118,120,122,123,124,125,128,131,132,134,136,137,138,140,141,142,145,146,147,148,149,152,153,154,155,156,157,158,159,]

    # Shallow NVs
    include_inds = [0, 1, 5, 6, 10, 17, 19, 20, 24, 26, 29, 35, 36, 43, 44, 46, 48, 49, 52, 53, 55, 57, 58, 62, 64, 65, 66, 68, 70, 72, 73, 74, 75, 78, 80, 82, 83, 90, 91, 93, 94, 95, 98, 99, 102, 103, 110, 111, 112, 113, 116, 122, 124, 126, 129, 130, 131, 136, 138, 142, 146]
    # scc_duration_list = [104, 56, 88, 92, 80, 72, 80, 80, 48, 88, 64, 272, 20, 124, 96, 84, 216, 16, 124, 92, 60, 96, 272, 108, 16, 64, 100, 56, 100, 84, 16, 100, 92, 64, 128, 128, 60, 100, 44, 272, 16, 120, 20, 88, 88, 116, 112, 48, 16, 272, 92, 272, 80, 84, 92, 96, 112, 112, 24, 92, 80, 28, 84, 92, 72, 92, 156, 56, 272, 272, 272, 124, 84, 272, 80, 68, 200, 72, 272, 272, 272, 72, 248, 116, 180, 120, 92, 16, 16, 72, 68, 84, 272, 40, 272, 80, 140, 144, 248, 72, 92, 56, 100, 92, 108, 108, 88, 88, 88, 228, 100, 128, 100, 72, 128, 16, 16, 124, 140, 116, 272, 80, 272, 96, 144, 80, 80, 68, 76, 192, 272, 16, 48, 96, 272, 180, 120, 36, 272, 152, 16, 76, 136, 124, 164, 200, 124, 140]
    # scc_duration_list = [116, 92, 72, 116, 92, 80, 92, 92, 72, 108, 88, 56, 72, 128, 92, 108, 56, 104, 108, 92, 56, 84, 84, 120, 128, 80, 96, 64, 92, 72, 108, 80, 80, 56, 112, 40, 40, 84, 72, 56, 128, 128, 108, 80, 112, 80, 108, 108, 64, 100, 96, 84, 92, 140, 140, 120, 60, 88, 88, 80, 60, 60, 124, 108, 16, 96, 96, 92, 136, 272, 144, 172, 72, 84, 84, 60, 116, 144, 56, 80, 112, 244, 80, 120, 80, 272, 64, 52, 116, 124, 16, 116, 56, 80, 120, 88, 272, 272, 116, 64, 16, 272, 80, 116, 112, 64, 272, 92, 272, 116, 48, 116, 40, 68, 104, 272, 272, 144, 272, 108, 216, 116, 84, 104, 108, 52, 272, 96, 136, 92, 108, 272, 212, 96, 204, 204, 84, 272, 88, 212, 56, 96, 120, 60, 100, 76, 100, 144]
    scc_duration_list =[104, 84, 84, 84, 100, 80, 88, 104, 72, 72, 100, 80, 68, 96, 92, 100, 108, 84, 112, 76, 64, 88, 92, 100, 72, 76, 72, 80, 84, 80, 84, 64, 72, 80, 84, 56, 52, 80, 80, 80, 72, 80, 92, 72, 76, 80, 72, 56, 80, 72, 56, 68, 60, 72, 92, 80, 76, 72, 80, 64, 72, 72, 80, 116, 68, 88, 56, 100, 72, 52, 68, 76, 64, 72, 68, 80, 80, 92, 56, 84, 64, 56, 76, 80, 84, 108, 72, 164, 80, 64, 88, 88, 96, 76, 128, 80, 100, 116, 88, 64, 80, 108, 108, 84, 100, 84, 128, 72, 80, 72, 72, 116, 52, 100, 80, 80, 68, 132, 92, 100, 80, 72, 96, 64, 84, 80, 68, 72, 88, 72, 92, 88, 72, 76, 80, 64, 92, 72, 80, 72, 84, 92, 80, 100, 96, 80, 80, 112]
    # median_value = np.median(scc_duration_list)
    # print(median_value)
    # Replace values less than 50 with the median
    # scc_duration_list = [median_value if val < 50 or  val > 200 else val for val in scc_duration_list]
    # print(scc_duration_list)
    # sys.exit()
    # pol_duration_list =  [188, 108, 120, 96, 148, 120, 140, 128, 120, 128, 120, 140, 148, 128, 120, 128, 108, 96, 108, 120, 120, 120, 140, 128, 120, 128, 128, 108, 120, 128, 140, 404, 148, 96, 160, 128, 128, 128, 140, 120, 128, 188, 120, 120, 168, 140, 140, 128, 128, 120, 140, 140, 108, 68, 120, 96, 140, 120, 140, 128, 140, 160, 108, 120, 120, 140, 120, 140, 140, 424, 168, 140, 140, 120, 120, 140, 180, 128, 56, 120, 128, 240, 128, 120, 160, 160, 96, 128, 120, 128, 128, 168, 168, 140, 120, 128, 108, 140, 140, 120, 140, 120, 128, 148, 168, 168, 140, 128, 220, 120, 108, 148, 120, 140, 140, 128, 120, 168, 160, 108, 148, 120, 128, 120, 212, 180, 128, 120, 120, 120, 140, 168, 120, 128, 140, 128, 128, 140, 140, 108, 140, 140, 140, 180, 120, 220, 128, 140]
    pol_duration_list = [144, 192, 172, 192, 172, 124, 144, 192, 192, 248, 192, 220, 180, 152, 180, 200, 236, 164, 192, 180, 192, 200, 172, 200, 192, 192, 172, 124, 192, 172, 152, 320, 124, 152, 192, 264, 136, 164, 192, 220, 208, 192, 108, 144, 180, 192, 180, 236, 152, 200, 152, 192, 172, 144, 192, 192, 200, 164, 220, 164, 180, 200, 192, 180, 220, 172, 180, 192, 172, 192, 220, 220, 180, 180, 172, 192, 292, 208, 192, 180, 208, 236, 192, 200, 264, 220, 164, 192, 152, 228, 180, 236, 208, 220, 180, 164, 248, 256, 192, 180, 208, 172, 192, 208, 236, 180, 256, 164, 304, 192, 192, 236, 172, 228, 208, 192, 172, 208, 264, 200, 248, 180, 192, 192, 236, 248, 172, 192, 152, 172, 180, 236, 192, 192, 256, 200, 180, 236, 172, 192, 264, 180, 180, 320, 144, 228, 164, 192]
    # Initialize a list with None values

    # arranged_scc_duration_list = [None] * num_nvs
    # for i, idx in enumerate(include_inds):
    #     arranged_scc_duration_list[idx] = scc_duration_list[i]
    # scc_duration_list = arranged_scc_duration_list
    # Initialize a list with None values
    # arranged_pol_duration_list = [None] * num_nvs
    # for i, idx in enumerate(include_inds):
    #     arranged_pol_duration_list[idx] = pol_duration_list[i]
    # pol_duration_list = arranged_pol_duration_list

    # threshold_list = [17.5, 16.5, 12.5, 24.5, 21.5, 22.5, 19.5, 18.5, 17.5, 18.5, 27.5, 20.5, 23.5, 17.5, 18.5, 17.5, 23.5, 19.5, 10.5, 16.5, 17.5, 15.5, 21.5, 17.5, 18.5, 19.5, 23.5, 17.5, 23.5, 18.5, 15.5, 16.5, 23.5, 16.5, 19.5, 18.5, 15.5, 20.5, 14.5, 17.5, 23.5, 26.5, 17.5, 17.5, 16.5, 12.5, 13.5, 15.5, 16.5, 18.5, 20.5, 12.5, 18.5, 23.5, 16.5, 17.5, 22.5, 13.5, 14.5, 22.5, 14.5, 15.5, 13.5, 21.5, 18.5, 18.5, 14.5, 17.5, 17.5, 18.5, 15.5, 17.5, 13.5, 15.5, 14.5, 21.5, 17.5, 17.5, 18.5, 16.5, 16.5, 13.5, 17.5, 17.5, 14.5, 14.5, 18.5, 29.5, 19.5, 16.5, 21.5, 16.5, 17.5, 14.5, 19.5, 18.5, 15.5, 15.5, 20.5, 16.5, 14.5, 16.5, 14.5, 17.5, 16.5, 21.5, 13.5, 14.5, 15.5, 12.5, 17.5, 16.5, 12.5, 12.5, 12.5, 12.5, 12.5]
    # arranged_threshold_list = [None] * num_nvs
    # for i, idx in enumerate(include_inds):
    #     arranged_threshold_list[idx] = threshold_list[i]
    # threshold_list = arranged_threshold_list

    # final_drop_inds = [23, 73, 89, 99, 117, 120, 132, 137, 155, 157, 159]
    # include_inds = [ind for ind in include_inds if ind not in final_drop_inds]
    # fmt: on
    # orientation_data = dm.get_raw_data(file_id=1723161184641)
    # orientation_a_inds = orientation_data["orientation_indices"]["0.041"]["nv_indices"]
    # orientation_b_inds = orientation_data["orientation_indices"]["0.147"]["nv_indices"]
    # orientation_ab_inds = orientation_a_inds + orientation_b_inds
    # snr_list = np.array(snr_list)
    # # Gets 103 best NVs of two target orientations
    # include_inds = [
    #     ind
    #     for ind in range(num_nvs)
    #     if snr_list[ind] > 0.07 and ind in orientation_ab_inds
    # ]
    # print(np.array(snr_list)[include_inds].tolist())
    # sys.exit()

    # test = []
    # for ind in range(num_nvs):
    #     if ind in include_inds:
    #         test.append(scc_duration_list.pop(0))
    #     else:
    #         test.append(None)
    # print(test)
    # sys.exit()

    # Analysis
    # kpl.init_kplotlib()
    # fig, ax = plt.subplots()
    # kpl.plot_points(
    #     ax, range(103), np.array(snr_list2) - np.array(snr_list)[include_inds]
    # )
    # # kpl.plot_points(ax, prep_fidelity_list, snr_list_from_resonance)
    # # ax.set_xlabel("Charge preparation fidelity")
    # # kpl.plot_points(ax, readout_fidelity_list, snr_list_from_resonance)
    # # ax.set_xlabel("Readout fidelity")
    # # kpl.plot_points(ax, red_chi_sq_list, snr_list_from_resonance)
    # # ax.set_xlabel("Reduced chi sq of charge state histogram")
    # ax.set_xlabel("NV order index")
    # ax.set_ylabel("SNR difference")
    # kpl.show(block=True)
    # sys.exit()
    scc_duration_list = [
        4 * round(el / 4) if el is not None else None for el in scc_duration_list
    ]
    # scc_amp_list = [1.0] * num_nvs
    # scc_duration_list = [144] * num_nvs
    # nv_list[i] will have the ith coordinates from the above lists
    nv_list: list[NVSig] = []
    for ind in range(num_nvs):
        # if ind not in include_inds:
        #     continue
        coords = {
            CoordsKey.SAMPLE: sample_coords,
            CoordsKey.Z: z_coord,
            CoordsKey.PIXEL: pixel_coords_list[ind],
            green_laser_aod: green_coords_list[ind],
            red_laser_aod: red_coords_list[ind],
        }
        nv_sig = NVSig(
            name=f"{sample_name}-nv{ind}_{date_str}",
            coords=coords,
            # scc_duration=scc_duration_list[ind],
            # scc_amp=scc_amp_list[ind],
            threshold=threshold_list[ind],
            pulse_durations={
                VirtualLaserKey.SCC: scc_duration_list[ind],
                VirtualLaserKey.CHARGE_POL: pol_duration_list[ind],
            },
            # pulse_amps={
            #     VirtualLaserKey.SCC: scc_amp_list[ind],
            #     VirtualLaserKey.CHARGE_POL: charge_pol_amps[ind],
            # },
        )
        nv_list.append(nv_sig)
    # print(nv_sig)
    # Additional properties for the representative NV
    nv_list[0].representative = True
    # nv_list[1].representative = True
    repr_nv_sig = widefield.get_repr_nv_sig(nv_list)
    nv_sig = widefield.get_repr_nv_sig(nv_list)
    # print(f"Created NV: {nv_sig.name}, Coords: {nv_sig.coords}")
    # nv_sig.expected_counts = 4500
    # nv_sig.expected_counts = 1320
    nv_sig.expected_counts = 2020

    # num_nvs = len(nv_list)
    # print(f"Final NV List: {nv_list}")
    # Ensure data is defined before accessing it
    # data = None

    # nv_sig.coords[green_laser_aod] = [111.56373974967805, 111.0353971667772]

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
    # nv_list = nv_list[:2]
    print(f"length of NVs list:{len(nv_list)}")
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

        do_compensate_for_drift(nv_sig)
        # do_widefield_image_sample(nv_sig, 50)
        # do_charge_state_histograms(nv_list)
        # do_charge_state_conditional_init(nv_list)

        # for point in points:
        #     x, y = point
        #     nv_sig.coords[CoordsKey.SAMPLE][0] += x
        #     nv_sig.coords[CoordsKey.SAMPLE][1] += y
        #     print(nv_sig.coords[CoordsKey.SAMPLE])

        # Move diagonally forward
        # for x, y in zip(x_values, y_values):
        # nv_sig.coords[CoordsKey.SAMPLE][0] = x
        # nv_sig.coords[CoordsKey.SAMPLE][1] = y
        # do_scanning_image_sample(nv_sig)

        # for z in np.linspace(0.0, 2.0, 15):
        #     nv_sig.coords[CoordsKey.Z] = z
        #     do_scanning_image_sample(nv_sig)

        # nv_sig.coords[CoordsKey.z] = 0.4
        # do_scanning_image_sample(nv_sig)

        # for y in np.linspace(0, 16, 5):
        #     for y in np.linspace(0, 16, 5):
        # nv_sig.coords[green_laser_aod : green_coords_list[ind]] + x
        #         do_scanning_image_sample(nv_sig)

        # do_scanning_image_sample(nv_sig)
        # do_scanning_image_full_roi(nv_sig)
        # do_scanning_image_sample_zoom(nv_sig)
        # do_widefield_image_sample(nv_sig, 50)
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
        # do_optimize_green(nv_sig)
        # do_optimize_red(nv_sig, repr_nv_sig)
        # do_optimize_z(nv_sig)
        ## do_optimize_sample(nv_sig)

        # widefield.reset_all_drift()
        # coords_key = None
        # coords_key = green_laser
        # coords_key = red_laser
        # do_optimize_loop(nv_list, coords_key)

        # do_optimize_pol_amp(nv_list)
        # do_optimize_pol_duration(nv_list)
        # do_optimize_readout_amp(nv_list)
        # do_optimize_readout_duration(nv_list)
        # optimize_readout_amp_and_duration(nv_list)
        # do_charge_state_histograms_images(nv_list, vary_pol_laser=True)
        # do_check_readout_fidelity(nv_list)

        # do_resonance_zoom(nv_list)
        # do_rabi(nv_list)
        do_resonance(nv_list)
        # do_spin_echo(nv_list)

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
        # do_power_rabi_scc_snr(nv_list)
        # do_optimize_scc_duration(nv_list)
        # do_optimize_scc_amp(nv_list)
        # optimize_scc_amp_and_duration(nv_list)
        # do_crosstalk_check(nv_sig)
        # do_spin_pol_check(nv_sig)
        # do_calibrate_green_red_delay()
        # do_simple_correlation_test(nv_list)

        # do_simple_correlation_test(nv_list)

        # for nv in nv_list:
        #     nv.spin_flip = False
        # Get the indices of well-separated NVs
        # selected_indices = widefield.select_well_separated_nvs(nv_list, 58)
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
        # do_optimize_red(nv_sig, repr_nv_sig)
    except Exception as exc:
        if do_email:
            recipient = email_recipient
            tb.send_exception_email(email_to=recipient)
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
