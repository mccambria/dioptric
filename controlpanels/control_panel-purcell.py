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
    bootstrapped_pulse_error_tomography,
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
    optimize_spin_pol,
    power_rabi,
    rabi,
    ramsey,
    relaxation_interleave,
    resonance,
    scc_snr_check,
    simple_correlation_test,
    spin_echo,
    spin_echo_phase_scan_test,
    spin_pol_check,
    xy,
)

# from slmsuite import optimize_slm_calibration
from utils import common, widefield
from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import positioning as pos
from utils import tool_belt as tb
from utils.constants import Axes, CoordsKey, NVSig, VirtualLaserKey
from utils.positioning import get_scan_1d as calculate_freqs

green_laser = "laser_INTE_520"
red_laser = "laser_COBO_638"
yellow_laser = "laser_OPTO_589"
green_laser_aod = f"{green_laser}_aod"
red_laser_aod = f"{red_laser}_aod"

### Major Routines


def do_widefield_image_sample(nv_sig, num_reps=1):
    return image_sample.widefield_image(nv_sig, num_reps)


def do_scanning_image_sample(nv_sig):
    scan_range = 10
    num_steps = 10
    image_sample.scanning(nv_sig, scan_range, scan_range, num_steps)


def do_scanning_image_full_roi(nv_sig):
    total_range = 30
    scan_range = 10
    num_steps = 10
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
    num_runs = 200
    # num_reps = 5
    # num_runs = 2
    min_duration = 20
    max_duration = 608
    return optimize_charge_state_histograms_mcc.optimize_pol_duration(
        nv_list, num_steps, num_reps, num_runs, min_duration, max_duration
    )


def do_optimize_pol_amp(nv_list):
    num_steps = 18
    # num_reps = 150
    # num_runs = 5
    num_reps = 8
    num_runs = 200
    min_amp = 0.7
    max_amp = 1.3
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
    # num_steps = 18
    # num_reps = 150
    # num_runs = 5
    num_reps = 12
    num_runs = 300
    # num_runs = 200
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
    # # Single amp
    min_duration = 16
    max_duration = 272
    num_dur_steps = 17
    min_amp = 1.0
    max_amp = 1.0
    num_amp_steps = 1

    # # Single dur
    # min_amp = 0.6
    # max_amp = 1.4
    # num_amp_steps = 15
    # min_duration = 84
    # max_duration = 84
    # num_dur_steps = 1
    # reps and runs
    num_reps = 11
    num_runs = 200
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


def do_optimize_spin_pol_amp(nv_list):
    min_tau = 0.9
    max_tau = 1.2
    num_steps = 16
    num_reps = 15
    num_runs = 200
    # num_runs = 2
    uwave_ind_list = [1]  # iq modulated
    optimize_spin_pol.optimize_spin_pol_amp(
        nv_list,
        num_steps,
        num_reps,
        num_runs,
        min_tau,
        max_tau,
        uwave_ind_list,
    )


def do_scc_snr_check(nv_list):
    num_reps = 200
    num_runs = 60
    # num_runs = 200
    # num_runs = 160 * 4
    # num_runs = 3
    scc_snr_check.main(nv_list, num_reps, num_runs, uwave_ind_list=[1])


def do_bootstrapped_pulse_error_tomography(nv_list):
    num_reps = 11
    num_runs = 200
    # num_runs = 10
    # num_runs = 1100
    # bootstrapped_pulse_error_tomography.main(
    #     nv_list, num_reps, num_runs, uwave_ind_list=[1]
    # )
    for _ in range(2):
        bootstrapped_pulse_error_tomography.main(
            nv_list, num_reps, num_runs, uwave_ind_list=[1]
        )


def do_power_rabi(nv_list):
    num_reps = 10
    num_runs = 200
    power_range = 1.5
    num_steps = 10
    uwave_ind_list = [1]
    powers = np.linspace(0, power_range, num_steps)
    # num_runs = 200
    # num_runs = 3
    power_rabi.main(
        nv_list,
        num_steps,
        num_reps,
        num_runs,
        powers,
        uwave_ind_list,
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
    # num_runs = 2
    for _ in range(5):
        simple_correlation_test.main(nv_list, num_reps, num_runs)


def do_calibrate_iq_delay(nv_list):
    min_tau = 20
    max_tau = 292
    num_steps = 18
    num_reps = 10
    num_runs = 100
    uwave_ind_list = [1]
    taus = np.linspace(min_tau, max_tau, num_steps)
    calibrate_iq_delay.main(
        nv_list, num_steps, num_reps, num_runs, taus, uwave_ind_list
    )


def do_resonance(nv_list):
    # freq_center = 2.87
    # freq_range = 0.240
    # freq_range = 0.36
    # num_steps = 60
    # num_steps = 72
    # Single ref
    # num_reps = 8
    # num_runs = 1100
    # num_runs = 200
    # Both refs
    num_reps = 3
    num_runs = 400
    freqs = []
    centers = [2.730700, 3.022277]
    range_each = 0.1
    lower_freqs = calculate_freqs(centers[0], range_each, 20)
    freqs.extend(lower_freqs)
    upper_freqs = calculate_freqs(centers[1], range_each, 20)
    freqs.extend(upper_freqs)
    ##
    # Remove duplicates and sort
    freqs = sorted(set(freqs))
    num_steps = len(freqs)
    # sys.exit()
    resonance.main(nv_list, num_steps, num_reps, num_runs, freqs=freqs)
    # for _ in range(2):
    #     resonance.main(nv_list, num_steps, num_reps, num_runs, freqs=freqs)


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
    uwave_ind_list = [1]  # only one
    # uwave_ind_list = [0, 1]
    rabi.main(nv_list, num_steps, num_reps, num_runs, min_tau, max_tau, uwave_ind_list)
    # for _ in range(2):
    #     rabi.main(
    #         nv_list, num_steps, num_reps, num_runs, min_tau, max_tau, uwave_ind_list
    #     )
    # uwave_ind_list = [0]
    # rabi.main(nv_list, num_steps, num_reps, num_runs, min_tau, max_tau, uwave_ind_list)
    # uwave_ind_list = [1]
    # rabi.main(nv_list, num_steps, num_reps, num_runs, min_tau, max_tau, uwave_ind_list)


def do_spin_echo_phase_scan_test(nv_list):
    num_steps = 21
    num_reps = 11
    num_runs = 150
    # num_runs = 2
    # phi_list = np.linspace(0, 360, num_steps)
    # fmt: off
    phi_list = [0, 18, 36, 54, 72, 90, 108, 126, 144, 162, 180, 198, 216, 234, 252, 270, 288, 306, 324, 342, 360]
    # fmt: on
    uwave_ind_list = [1]  # only one has iq modulation
    spin_echo_phase_scan_test.main(
        nv_list, num_steps, num_reps, num_runs, phi_list, uwave_ind_list
    )
    # for _ in range(2):
    #     spin_echo_phase_scan_test.main(
    #         nv_list, num_steps, num_reps, num_runs, min_phi, max_phi, uwave_ind_list
    #     )


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


# def do_spin_echo(nv_list):
#     # revival_period = int(51.5e3 / 2)
#     revival_period = int(30e3 / 2)
#     min_tau = 200
#     taus = []
#     # revival_width = 5e3
#     revival_width = 4e3
#     decay = np.linspace(min_tau, min_tau + revival_width, 6)
#     taus.extend(decay.tolist())
#     gap = np.linspace(min_tau + revival_width, revival_period - revival_width, 7)
#     taus.extend(gap[1:-1].tolist())
#     first_revival = np.linspace(
#         revival_period - revival_width, revival_period + revival_width, 61
#     )
#     taus.extend(first_revival.tolist())
#     gap = np.linspace(
#         revival_period + revival_width, 2 * revival_period - revival_width, 7
#     )
#     taus.extend(gap[1:-1].tolist())
#     second_revival = np.linspace(
#         2 * revival_period - revival_width, 2 * revival_period + revival_width, 11
#     )
#     taus.extend(second_revival.tolist())
#     taus = [round(el / 4) * 4 for el in taus]

#     # Remove duplicates and sort
#     taus = sorted(set(taus))

#     # Experiment settings
#     num_steps = len(taus)

#     # Automatic taus setup, linear spacing
#     # min_tau = 200
#     # max_tau = 84e3 + min_tau
#     # num_steps = 29

#     num_reps = 3
#     num_runs = 200
#     # num_runs = 2
#     # spin_echo.main(nv_list, num_steps, num_reps, num_runs, min_tau, max_tau)
#     # spin_echo.main(nv_list, num_steps, num_reps, num_runs, taus=taus)
#     for ind in range(6):
#         spin_echo.main(nv_list, num_steps, num_reps, num_runs, taus=taus)


def do_spin_echo(nv_lis):
    min_tau = 200  # ns
    # max_tau = 20e3  # fallback
    # revival_period = int(15e3)
    revival_period = int(13e3)
    taus = []
    revival_width = 6e3
    decay = np.linspace(min_tau, min_tau + revival_width, 6)
    taus.extend(decay.tolist())
    gap = np.linspace(min_tau + revival_width, revival_period - revival_width, 6)
    taus.extend(gap[1:-1].tolist())
    first_revival = np.linspace(
        revival_period - revival_width, revival_period + revival_width, 61
    )
    taus.extend(first_revival.tolist())
    # Round to clock-cycle-compatible units
    taus = [round(el / 4) * 4 for el in taus]
    # Remove duplicates and sort
    taus = sorted(set(taus))
    num_steps = len(taus)
    num_reps = 3
    num_runs = 600

    print(
        f"[Spin Echo] Running with {num_steps} τ values, revival_period={revival_period}"
    )

    for _ in range(1):
        spin_echo.main(nv_list, num_steps, num_reps, num_runs, taus=taus)


def do_spin_echo_1(nv_lis):
    min_tau = 200  # ns
    # max_tau = 20e3  # fallback
    revival_period = int(15e3)
    # revival_period = int(13e3)
    taus = []
    revival_width = 5e3
    decay = np.linspace(min_tau, min_tau + revival_width, 6)
    taus.extend(decay.tolist())
    gap = np.linspace(min_tau + revival_width, revival_period - revival_width, 6)
    taus.extend(gap[1:-1].tolist())
    first_revival = np.linspace(
        revival_period - revival_width, revival_period + revival_width, 61
    )
    taus.extend(first_revival.tolist())
    # Round to clock-cycle-compatible units
    taus = [round(el / 4) * 4 for el in taus]
    # Remove duplicates and sort
    taus = sorted(set(taus))
    num_steps = len(taus)
    num_reps = 3
    num_runs = 600

    print(
        f"[Spin Echo] Running with {num_steps} τ values, revival_period={revival_period}"
    )

    for _ in range(1):
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


def do_xy(nv_list, xy_seq="xy8"):
    min_tau = 200
    max_tau = 1e6 + min_tau
    num_steps = 24
    num_reps = 10
    uwave_ind_list = [1]  # iq modulated
    num_runs = 400
    # taus calculation
    taus = widefield.generate_log_spaced_taus(min_tau, max_tau, num_steps, base=4)
    # print(taus)
    # sys.exit()
    # num_runs = 2
    # xy8.main(nv_list, num_steps, num_reps, num_runs, taus , uwave_ind_list)
    for _ in range(3):
        xy.main(
            nv_list,
            num_steps,
            num_reps,
            num_runs,
            taus,
            uwave_ind_list,
            xy_seq,
        )


def do_xy_uniform_revival_scan(nv_list, xy_seq="xy8-1"):
    T_min = 2e3  # ns, total evolution time (1 μs)
    T_max = 42e3  # ns, total evolution time (20 μs)
    N = 8  # XY8 has 8 π pulses
    factor = 2 * N  # total time T = 2Nτ = 16τ

    num_steps = 100
    taus = np.linspace(T_min, T_max, num_steps)
    # Convert total evolution time to τ
    # taus = [T / factor for T in total_times]

    # Round τ to 4 ns resolution
    taus = [round(tau / 4) * 4 for tau in taus]
    taus = sorted(set(taus))  # remove duplicates

    num_reps = 2
    num_runs = 600
    num_steps = len(taus)
    uwave_ind_list = [1]  # IQ-modulated channel index

    print(
        f"[XY8 Uniform] Scanning {num_steps} τ values from {taus[0]} to {taus[-1]} ns"
    )

    for ind in range(4):
        xy.main(
            nv_list,
            num_steps,
            num_reps,
            num_runs,
            uwave_ind_list=uwave_ind_list,
            taus=taus,
            xy_seq=xy_seq,
        )


def do_xy_revival_scan(nv_list, xy_seq="xy8-1"):
    min_total_time = 100  # ns
    revival_time = 14.1e3  # ns
    revival_width = 4e3  # ns
    high_res_points = 24
    gap_points = 6
    decay_points = 6
    num_revivals = 4
    taus = []
    # Initial coherence decay region
    decay = np.linspace(min_total_time, min_total_time + revival_width, decay_points)
    taus.extend(decay.tolist())

    for i in range(1, num_revivals + 1):
        center = i * revival_time

        # Gap before revival
        if i == 1:
            gap_start = min_total_time + revival_width
        else:
            gap_start = (i - 1) * revival_time + revival_width
        gap_end = center - revival_width

        if gap_end > gap_start:
            gap = np.linspace(gap_start, gap_end, gap_points)
            taus.extend(gap[1:-1].tolist())  # exclude endpoints to avoid duplication

        # High-resolution scan across revival
        revival_scan = np.linspace(
            center - revival_width, center + revival_width, high_res_points
        )
        taus.extend(revival_scan.tolist())

    # Round to 4 ns granularity
    taus = sorted(set(round(tau / 4) * 4 for tau in taus))

    num_steps = len(taus)
    num_reps = 1
    num_runs = 2000
    uwave_ind_list = [1]

    print(
        f"[{xy_seq}] Running with {num_steps} τ values, targeting {num_revivals} revivals starting at {revival_time} ns"
    )

    for _ in range(4):
        xy.main(
            nv_list,
            num_steps,
            num_reps,
            num_runs,
            uwave_ind_list=uwave_ind_list,
            taus=taus,
            xy_seq=xy_seq,
        )


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
    num_runs = 200
    # num_runs = 2
    # relaxation_interleave.sq_relaxation(
    #     nv_list, num_steps, num_reps, num_runs, min_tau, max_tau
    # )
    for _ in range(4):
        relaxation_interleave.sq_relaxation(
            nv_list, num_steps, num_reps, num_runs, min_tau, max_tau
        )


def do_dq_relaxation(nv_list):
    min_tau = 1e3
    max_tau = 15e6 + min_tau
    num_steps = 21
    num_reps = 10
    num_runs = 200

    # relaxation_interleave.dq_relaxation(
    #     nv_list, num_steps, num_reps, num_runs, min_tau, max_tau
    # )
    for _ in range(4):
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
    num_reps = 4
    num_runs = 600
    # num_runs = 2
    # dark_time = 1e9 # 1s
    # dark_time = 10e6  # 10ms
    dark_time_1 = 1e6  # 1 ms in nanoseconds
    dark_time_2 = 8e9  # 8 s in nanoseconds
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
    opx.constant_ac(
        [],  # Digital channels
        [7],  # Analog channels
        [0.20],  # Analog voltages
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
        [0.19, 0.19, 0.20],  # Analog voltages
        [107, 107, 0],  # Analog frequencies
    )
    # # Red + green + Yellow
    # opx.constant_ac(
    #     [4, 1],  # Digital channels1
    #     [3, 4, 2, 6, 7],  # Analog channels
    #     [0.19, 0.19, 0.17, 0.17, 0.25],  # Analog voltages
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
        [[0.4, 0.2], [-0.1999, 0.5464], [-0.2, -0.1464]], dtype="float32"
    )
    cal_pixel_coords = np.array(
        [[135.141, 117.788], [97.234, 144.799], [92.568, 98.422]], dtype="float32"
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


def estimate_z(x, y, z0=0.15, slope=-0.0265):
    """Estimate Z from (x, y) using diagonal slope."""
    return z0 + slope * (x + y) / np.sqrt(2)


def generate_equilateral_triangle_around_center(center=(0, 0), r=2.0):
    angles = [0, 120, 240]  # degrees
    points = []
    for angle_deg in angles:
        theta = np.radians(angle_deg)
        x = center[0] + r * np.cos(theta)
        y = center[1] + r * np.sin(theta)
        points.append((x, y))
    return points


def scan_equilateral_triangle(nv_sig, center_coord=(0, 0), radius=0.2):
    triangle_coords = generate_equilateral_triangle_around_center(
        center_coord, r=radius
    )
    triangle_coords.append(center_coord)  # Return to center
    print(triangle_coords)
    for sample_coord in triangle_coords:
        # z = estimate_z(*sample_coord)
        nv_sig.coords[CoordsKey.SAMPLE] = sample_coord
        # nv_sig.coords[CoordsKey.Z] = z
        # print(f"Scanning SAMPLE: {sample_coord}, estimated Z: {z:.3f}")
        do_scanning_image_sample(nv_sig)


### Run the file
if __name__ == "__main__":
    # region Shared parameters
    green_coords_key = f"coords-{green_laser}"
    red_coords_key = f"coords-{red_laser}"
    pixel_coords_key = "pixel_coords"
    sample_name = "rubin"
    # magnet_angle = 90
    date_str = "2025_02_26"
    sample_coords = [0.0, 0.0]
    z_coord = 0.6
    # Load NV pixel coordinates1
    pixel_coords_list = load_nv_coords(
        # file_path="slmsuite/nv_blob_detection/nv_blob_rubin_shallow_154nvs_reordered.npz",
        file_path="slmsuite/nv_blob_detection/nv_blob_rubin_shallow_75nvs_reordered.npz",
    ).tolist()
    # pixel_coords_list = [
    #     [122.027, 118.236],
    #     [113.173, 128.034],
    #     [27.44, 23.014],
    #     [108.384, 227.38],
    #     [227.438, 19.199],
    # ]
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

    # # Print first coordinate set for verification
    # print(f"Number of NVs: {green_coords_list}")
    # print(f"Number of NVs: {red_coords_list}")
    # sys.exit()
    print(f"Number of NVs: {len(pixel_coords_list)}")
    print(f"Reference NV:{pixel_coords_list[0]}")
    print(f"Green Laser Coordinates: {green_coords_list[0]}")
    print(f"Red Laser Coordinates: {red_coords_list[0]}")
    pixel_coords_list = [
        # [113.173, 128.034],
        [126.55, 128.472],
        [27.44, 23.014],
        [108.384, 227.38],
        [227.438, 19.199],
    ]
    green_coords_list = [
        [108.628, 107.119],
        [118.127, 97.472],
        [107.036, 118.416],
        [96.822, 94.821],
    ]
    red_coords_list = [
        [72.466, 73.251],
        [80.703, 64.786],
        [72.119, 81.942],
        [63.276, 62.851],
    ]

    num_nvs = len(pixel_coords_list)
    threshold_list = [None] * num_nvs
    # fmt: off
    #81NVs
    # pol_duration_list = [132, 140, 140, 132, 116, 156, 104, 164, 156, 156, 108, 152, 168, 116, 220, 92, 168, 116, 120, 140, 104, 180, 144, 152, 232, 132, 156, 228, 200, 96, 188, 168, 300, 128, 200, 176, 108, 220, 164, 128, 288, 436, 376, 108, 132, 252, 176, 128, 312, 140, 180, 116, 220, 328, 128, 324, 132, 164, 292, 176, 364, 276, 92, 104, 352, 388, 180, 328, 412, 152, 156, 164, 116, 168, 580, 372, 168, 152, 176, 164, 244]
    # scc_duration_list = [64, 80, 80, 80, 64, 88, 64, 100, 84, 84, 76, 92, 92, 80, 116, 76, 104, 72, 60, 72, 84, 68, 84, 80, 120, 80, 72, 100, 88, 72, 116, 84, 116, 88, 92, 84, 48, 128, 104, 72, 136, 128, 52, 84, 84, 136, 88, 88, 124, 56, 112, 104, 72, 108, 64, 120, 80, 148, 84, 76, 108, 80, 80, 64, 148, 120, 100, 148, 136, 72, 92, 96, 52, 88, 156, 84, 128, 72, 124, 72, 188]
    #75NVs
    # drop_indices = [42, 49, 53, 62, 75, 79] #drop these from 81 Nvs
    # pol_duration_list = [
    #     val for ind, val in enumerate(pol_duration_list) if ind not in drop_indices
    # ]
    # scc_duration_list = [
    #     val for ind, val in enumerate(scc_duration_list) if ind not in drop_indices
    # ]
    #75NVs optimized
    # pol_duration_list = [164, 144, 168, 108, 132, 176, 132, 152, 176, 168, 140, 200, 204, 120, 268, 116, 200, 128, 152, 144, 116, 192, 156, 156, 256, 140, 156, 240, 232, 116, 200, 176, 340, 116, 108, 216, 104, 200, 144, 140, 304, 416, 140, 156, 292, 188, 164, 352, 180, 156, 232, 144, 328, 132, 228, 288, 164, 384, 292, 140, 400, 388, 192, 348, 412, 144, 200, 180, 120, 188, 436, 180, 164, 232, 252]
    # pol_duration_list = [116, 152, 140, 104, 96, 152, 104, 128, 192, 164, 120, 152, 164, 108, 212, 96, 144, 104, 132, 164, 84, 176, 144, 132, 240, 120, 140, 168, 280, 116, 156, 176, 360, 128, 104, 240, 96, 228, 128, 116, 360, 460, 108, 108, 352, 152, 132, 384, 200, 164, 128, 104, 264, 116, 240, 200, 116, 300, 228, 84, 384, 336, 176, 268, 348, 116, 164, 132, 92, 152, 408, 156, 96, 244, 128]
    # pol_duration_list = [96, 152, 156, 92, 108, 192, 120, 144, 216, 212, 120, 152, 220, 128, 244, 104, 212, 120, 132, 132, 104, 204, 192, 144, 300, 128, 152, 220, 312, 128, 188, 192, 396, 132, 80, 264, 108, 232, 156, 120, 424, 476, 104, 128, 440, 132, 156, 416, 252, 180, 144, 128, 292, 116, 256, 216, 128, 348, 232, 96, 428, 352, 176, 324, 376, 152, 216, 180, 104, 176, 460, 180, 120, 268, 188]
    #new set 75NVs optimized
    # pol_duration_list = [132, 132, 144, 176, 104, 164, 116, 128, 168, 204, 116, 156, 180, 104, 220, 96, 132, 104, 120, 120, 104, 192, 132, 128, 228, 120, 132, 200, 276, 96, 204, 192, 376, 120, 84, 244, 104, 232, 116, 108, 340, 436, 96, 116, 340, 116, 104, 416, 168, 120, 108, 104, 300, 104, 192, 188, 116, 336, 220, 92, 372, 328, 156, 300, 384, 120, 144, 140, 120, 132, 472, 132, 96, 192, 168]
    pol_duration_list =[144, 128, 132, 312, 108, 128, 96, 152, 156, 144, 96, 120, 132, 108, 168, 108, 132, 116, 108, 96, 72, 140, 140, 104, 192, 108, 120, 144, 212, 84, 128, 108, 268, 104, 168, 376, 92, 140, 132, 116, 268, 352, 116, 128, 276, 116, 140, 304, 152, 132, 104, 96, 168, 84, 176, 144, 96, 232, 156, 72, 288, 216, 128, 192, 228, 96, 104, 128, 92, 180, 340, 132, 96, 176, 108]
    # scc_duration_list = [88, 80, 100, 100, 76, 88, 68, 88, 88, 92, 72, 68, 88, 80, 116, 64, 112, 48, 64, 60, 96, 92, 92, 72, 108, 84, 68, 100, 108, 76, 108, 108, 124, 84, 92, 72, 56, 140, 96, 76, 104, 136, 88, 64, 108, 80, 124, 120, 144, 88, 72, 68, 124, 80, 116, 84, 80, 132, 80, 36, 88, 108, 92, 152, 140, 68, 136, 80, 64, 84, 152, 140, 76, 92, 196]
    # scc_duration_list = [96, 100, 92, 108, 76, 88, 100, 84, 124, 92, 96, 92, 88, 72, 124, 72, 92, 56, 72, 72, 56, 96, 80, 80, 108, 92, 80, 128, 96, 60, 112, 144, 116, 80, 96, 72, 64, 140, 100, 72, 104, 124, 80, 56, 120, 80, 112, 128, 108, 128, 68, 48, 112, 64, 156, 84, 68, 128, 96, 44, 136, 136, 100, 132, 84, 84, 152, 96, 52, 92, 164, 136, 84, 108, 164]
    # scc_duration_list = [40, 88, 84, 84, 76, 92, 76, 84, 112, 100, 84, 72, 160, 80, 112, 72, 92, 64, 72, 64, 72, 84, 104, 72, 100, 80, 80, 96, 96, 64, 100, 104, 108, 84, 132, 64, 60, 152, 84, 60, 120, 128, 116, 56, 120, 80, 92, 124, 104, 84, 56, 48, 116, 60, 120, 104, 76, 148, 108, 56, 160, 136, 92, 156, 108, 68, 80, 100, 48, 80, 156, 112, 80, 84, 164] 
    # scc_duration_list = [92, 88, 96, 92, 72, 88, 56, 84, 92, 108, 80, 80, 88, 72, 108, 64, 92, 72, 72, 64, 76, 128, 84, 76, 100, 76, 64, 104, 104, 64, 112, 116, 124, 80, 72, 100, 56, 136, 84, 64, 104, 124, 92, 64, 120, 80, 120, 116, 100, 100, 76, 56, 120, 64, 132, 84, 72, 144, 100, 44, 92, 136, 92, 176, 128, 76, 116, 88, 68, 96, 144, 120, 68, 92, 160] 
    scc_duration_list = [48, 84, 84, 68, 76, 92, 72, 92, 116, 84, 72, 76, 64, 60, 88, 60, 84, 56, 68, 56, 56, 80, 80, 72, 88, 72, 72, 92, 96, 72, 96, 84, 100, 72, 72, 64, 68, 124, 80, 56, 100, 116, 72, 48, 96, 60, 80, 120, 92, 80, 60, 60, 108, 56, 124, 160, 64, 116, 108, 64, 128, 108, 92, 136, 120, 72, 80, 76, 64, 72, 124, 112, 72, 92, 80]
    # selected_indices_68MHz = [0, 7, 8, 9, 11, 14, 18, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 35, 38, 44, 45, 46, 47, 48, 49, 53, 55, 57, 58, 60, 62, 64, 66, 67, 68, 69, 70, 71, 72, 73]
    # selected_indices_185MHz  =[0, 1, 2, 3, 4, 5, 6, 10, 12, 13, 15, 16, 17, 19, 20, 21, 23, 29, 34, 36, 39, 40, 41, 42, 43, 50, 51, 52, 54, 56, 59, 61, 63, 65, 74]
    # fmt: on

    # arranged_scc_amp_list = [None] * num_nvs
    # arranged_scc_duration_list = [None] * num_nvs
    # arranged_pol_duration_list = [None] * len(pol_duration_list)
    # for i, idx in enumerate(include_indices):
    #     arranged_scc_duration_list[idx] = scc_duration_list[i]
    #     arranged_pol_duration_list[idx] = pol_duration_list[i]
    #     # arranged_scc_amp_list[idx] = scc_amp_list[i]
    # # # Assign back to original lists
    # scc_duration_list = arranged_scc_duration_list
    # pol_duration_list = arranged_pol_duration_list
    # scc_amp_list = arranged_scc_amp_list

    scc_duration_list = [
        4 * round(el / 4) if el is not None else None for el in scc_duration_list
    ]
    pol_duration_list = [
        4 * round(el / 4) if el is not None else None for el in pol_duration_list
    ]
    # print(f"Length of pol_duration_list: {len(pol_duration_list)}")
    # print(f"First 10 SCC durations: {scc_duration_list[:10]}")
    # print(f"First 10 POL durations: {pol_duration_list[:10]}")
    # sys.exit()

    # scc_amp_list = [1.0] * num_nvs
    # scc_duration_list = [112] * num_nvs
    # pol_duration_list = [200] * num_nvs
    # nv_list[i] will have the ith coordinates from the above lists
    nv_list: list[NVSig] = []
    for ind in range(num_nvs):
        # if ind not in selected_indices_68MHz:
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
            threshold=threshold_list[ind],
            pulse_durations={
                VirtualLaserKey.SCC: scc_duration_list[ind],
                VirtualLaserKey.CHARGE_POL: pol_duration_list[ind],
            },
            pulse_amps={
                # VirtualLaserKey.SCC: scc_amp_list[ind],
                # VirtualLaserKey.CHARGE_POL: charge_pol_amps[ind],
            },
        )
        nv_list.append(nv_sig)
    # print(nv_sig)
    # Additional properties for the representative NV
    nv_list[0].representative = True
    # nv_list[1].representative = True
    repr_nv_sig = widefield.get_repr_nv_sig(nv_list)
    nv_sig = widefield.get_repr_nv_sig(nv_list)
    # print(f"Created NV: {nv_sig.name}, Coords: {nv_sig.coords}")
    # nv_sig.expected_counts = 900
    # nv_sig.expected_counts = 1160
    # nv_sig.expected_counts = 1200

    # nv_list = nv_list[::-1]  # flipping the order of NVs
    # nv_list = nv_list[:2]
    print(f"length of NVs list:{len(nv_list)}")
    # sys.exit()
    # endregion

    # region Functions to run
    email_recipient = "mccambria@berkeley.edu"
    do_email = False
    try:
        # pass
        kpl.init_kplotlib()
        # tb.init_safe_stop()
        # widefield.reset_all_drift()
        # do_optimize_z(nv_sig)
        # do_optimize_xyz(nv_sig)
        # pos.set_xyz_on_nv(nv_sig)
        # piezo_voltage_to_pixel_calibration()

        # do_compensate_for_drift(nv_sig)

        # do_widefield_image_sample(nv_sig, 50)
        # do_widefield_image_sample(nv_sig, 200)

        # do_scanning_image_sample(nv_sig)
        # do_scanning_image_full_roi(nv_sig)
        # do_scanning_image_sample_zoom(nv_sig)
        # scan_equilateral_triangle(nv_sig, center_coord=sample_coords, radius=0.4)
        # do_image_nv_list(nv_list)
        # do_image_single_nv(nv_sig)
        # z_range = np.linspace(0.0, 1.0, 6)
        # for z in z_range:
        #     nv_sig.coords[CoordsKey.Z] = z
        #     do_scanning_image_sample(nv_sig)
        # x_range = np.linspace(-2.0, 6.0, 6)
        # y_range = np.linspace(-2.0, 6.0, 6)
        # # --- Step 1: Start at (0, 0) ---
        # sample_coord = [0.0, 0.0]
        # z = estimate_z(*sample_coord)
        # nv_sig.coords[CoordsKey.SAMPLE] = sample_coord
        # nv_sig.coords[CoordsKey.Z] = z
        # print(f"[START] Scanning SAMPLE: {sample_coord}, estimated Z: {z:.3f}")
        # do_scanning_image_sample(nv_sig)

        # # --- Step 2: Loop over all other (x, y) positions ---
        # for x in x_range:
        #     for y in y_range:
        #         if np.isclose(x, 0.0) and np.isclose(y, 0.0):
        #             continue  # already scanned at (0, 0)
        #         sample_coord = [x, y]
        #         z = estimate_z(x, y)
        #         nv_sig.coords[CoordsKey.SAMPLE] = sample_coord
        #         nv_sig.coords[CoordsKey.Z] = z
        #         print(f"Scanning SAMPLE: {sample_coord}, estimated Z: {z:.3f}")
        #         do_scanning_image_sample(nv_sig)

        # do_opx_constant_ac()
        # do_opx_square_wave()

        # do_optimize_pixel(nv_sig)
        # do_optimize_green(nv_sig)
        # do_optimize_red(nv_sig, repr_nv_sig)
        # do_optimize_z(nv_sig)
        # do_optimize_sample(nv_sig)
        # optimize.optimize_pixel_and_z(nv_sig, do_plot=True)
        # coords_key = None
        # coords_key = green_laser
        # coords_key = red_laser
        # do_optimize_loop(nv_list, coords_key)

        # do_charge_state_histograms(nv_list)
        # do_charge_state_conditional_init(nv_list)
        # do_charge_state_histograms_images(nv_list, vary_pol_laser=True)

        # do_optimize_pol_amp(nv_list)
        # do_optimize_pol_duration(nv_list)
        # do_optimize_readout_amp(nv_list)
        # do_optimize_readout_duration(nv_list)
        # optimize_readout_amp_and_duration(nv_list)
        # do_optimize_spin_pol_amp(nv_list)
        # do_check_readout_fidelity(nv_list)

        # do_scc_snr_check(nv_list)
        # do_optimize_scc_duration(nv_list)
        # do_optimize_scc_amp(nv_list)
        # optimize_scc_amp_and_duration(nv_list)
        # do_crosstalk_check(nv_sig)
        # do_spin_pol_check(nv_sig)
        # do_calibrate_green_red_delay()

        # do_spin_echo_phase_scan_test(nv_list)  # for iq mod test
        # do_bootstrapped_pulse_error_tomography(nv_list)
        # do_calibrate_iq_delay(nv_list)

        # do_rabi(nv_list)
        # do_power_rabi(nv_list)
        # do_resonance(nv_list)
        # do_resonance_zoom(nv_list)
        # do_spin_echo(nv_list)
        # do_spin_echo_1(nv_list)
        # do_ramsey(nv_list)

        # do_simple_correlation_test(nv_list)

        # do_sq_relaxation(nv_list)
        # do_dq_relaxation(nv_list)
        # do_detect_cosmic_rays(nv_list)
        # do_check_readout_fidelity(nv_list)
        # do_charge_quantum_jump(nv_list)
        # do_ac_stark(nv_list)

        # AVAILABLE_XY = ["hahn-n", "xy2-n", "xy4-n", "xy8-n", "xy16-n"]
        # do_xy(nv_list, xy_seq="xy16-1")
        # do_xy_uniform_revival_scan(nv_list, xy_seq="xy4-1")
        # do_xy_revival_scan(nv_list, xy_seq="xy4-1")

        # for nv in nv_list:
        #     nv.spin_flip = False
        # for nv in nv_list[: num_nvs // 2]:
        #     nv.spin_flip = True
        # do_simple_correlation_test(nv_list)
        # do_correlation_test(nv_list)

        # region Cleanup
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
