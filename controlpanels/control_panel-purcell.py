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
    num_reps = 300
    # num_reps = 100
    # num_runs = 50
    # num_runs = 15
    num_runs = 10
    # num_runs = 2
    return charge_state_histograms.main(
        nv_list, num_reps, num_runs, do_plot_histograms=False
    )


def do_optimize_pol_duration(nv_list):
    num_steps = 4
    # num_reps = 150
    # num_runs = 5
    num_reps = 5
    num_runs = 2
    min_duration = 500
    max_duration = 2000
    return optimize_charge_state_histograms_mcc.optimize_pol_duration(
        nv_list, num_steps, num_reps, num_runs, min_duration, max_duration
    )


def do_optimize_pol_amp(nv_list):
    num_steps = 24
    # num_reps = 150
    # num_runs = 5
    num_reps = 10
    num_runs = 225
    min_amp = 0.6
    max_amp = 1.2
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
    num_runs = 225
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
    num_runs = 100
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
    for ind in range(1):
        axes = axes_list[ind]
        ret_vals = targeting.optimize(nv_sig, coords_key=red_laser_aod, axes=axes)
        opti_coords.append(ret_vals[0])
        # Compensate for drift after first optimization along X axis
        if ind == 0:
            do_compensate_for_drift(ref_nv_sig)
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
    num_amp_steps = 15
    num_dur_steps = 17
    num_reps = 1
    num_runs = 1500
    # num_runs = 400  # Short test version
    # num_runs = 5
    min_amp = 0.75
    max_amp = 1.25
    min_duration = 48
    max_duration = 304

    # Single amp
    num_reps = 16
    num_runs = 100
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
    min_tau = 0.8
    max_tau = 1.2
    num_steps = 16
    num_reps = 20
    num_runs = 100
    # num_runs = 2
    optimize_scc.optimize_scc_amp(
        nv_list, num_steps, num_reps, num_runs, min_tau, max_tau
    )


def do_scc_snr_check(nv_list):
    num_reps = 250
    num_runs = 40
    # num_runs = 200
    # num_runs = 160 * 4
    # num_runs = 3
    scc_snr_check.main(nv_list, num_reps, num_runs, uwave_ind_list=[0, 1])


def do_simple_correlation_test(nv_list):
    num_reps = 200
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
    # num_runs = 500
    # num_runs = 750
    num_runs = 350
    # num_runs = 50
    # num_runs = 10
    # num_runs = 5

    # Both refs
    num_reps = 4
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
    num_runs = 300
    # num_runs = 100
    # num_runs = 20
    # num_runs = 5

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
    num_reps = 24
    # num_reps = 20
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
    # Manual taus setup
    revival_period = int(51.5e3 / 2)  # ns
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
    num_steps = len(taus)

    # Automatic taus setup, linear spacing
    # min_tau = 200
    # max_tau = 84e3 + min_tau
    # num_steps = 29

    num_reps = 4
    # num_runs = 200
    num_runs = 3

    # spin_echo.main(nv_list, num_steps, num_reps, num_runs, min_tau, max_tau)
    spin_echo.main(nv_list, num_steps, num_reps, num_runs, taus=taus)
    # for ind in range(5):
    #     spin_echo.main(nv_list, num_steps, num_reps, num_runs, taus=taus)


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
    num_runs = 10 * 60
    # num_runs = 2
    dark_time = 1e9

    charge_monitor.detect_cosmic_rays(nv_list, num_reps, num_runs, dark_time)


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
        [[0.0, 0.0], [-0.25, -0.25], [0.25, -0.25]], dtype="float32"
    )  # Voltage system coordinates
    # cal_pixel_coords = np.array(
    #     [[81.109, 110.177], [64.986, 94.177], [96.577, 95.047]], dtype="float32"
    # )
    cal_pixel_coords = np.array(
        [
            [91.778, 122.027],
            [109.388, 139.694],
            [75.396, 138.755],
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
    final_voltage = np.array()  # Start with all original global coordinates
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
    # magnet_angle = 90
    date_str = "2024_03_12"
    sample_coords = [2.0, 0.0]
    z_coord = 2.45
    # Load NV pixel coordinates
    pixel_coords_list = load_nv_coords(
        file_path="slmsuite/nv_blob_detection/nv_blob_filtered_160nvs_reordered.npz",
    ).tolist()

    # pixel_coords_list = [
    #     [106.923, 120.549],
    #     [52.761, 64.24],
    #     [95.923, 201.438],
    #     [207.435, 74.049],
    # ]

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
    print(f"Number of NVs: {len(pixel_coords_list)}")
    print(f"Reference NV:{pixel_coords_list[0]}")
    print(f"Green Laser Coordinates: {green_coords_list[0]}")
    print(f"Red Laser Coordinates: {red_coords_list[0]}")

    # pixel_coords_list = [
    #     [106.923, 120.549],
    #     [52.761, 64.24],
    #     [95.923, 201.438],
    #     [207.435, 74.049],
    # ]
    # green_coords_list = [
    #     [109.154, 107.063],
    #     [115.699, 101.502],
    #     [109.238, 115.859],
    #     [98.66, 100.902],
    # ]
    # red_coords_list = [
    #     [73.493, 72.342],
    #     [78.617, 67.752],
    #     [73.831, 79.567],
    #     [65.156, 67.45],
    # ]
    num_nvs = len(pixel_coords_list)
    threshold_list = [45.5] * num_nvs
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
    scc_optimal_amplitudes = scc_data["optimal_amplitudes"]
    # Cross pattern
    # scc_duration_list = list(scc_optimal_durations.values())
    # scc_amp_list = list(scc_optimal_amplitudes.values())

    # fmt: off
    # after calibration with amp 1.0
    snr_list = [0.222, 0.218, 0.204, 0.211, 0.256, 0.169, 0.04, 0.196, 0.176, 0.179, 0.191, 0.156, 0.083, 0.111, 0.071, 0.24, 0.231, 0.094, 0.15, 0.214, 0.14, 0.172, 0.165, 0.071, 0.206, 0.203, 0.132, 0.203, 0.096, 0.195, 0.211, 0.169, 0.057, 0.166, 0.225, 0.145, 0.121, 0.19, 0.126, 0.191, 0.108, 0.178, 0.165, 0.112, 0.19, 0.164, 0.143, 0.178, 0.08, 0.158, 0.093, 0.149, 0.105, 0.191, 0.201, 0.055, 0.174, 0.078, 0.152, 0.157, 0.124, 0.118, 0.133, 0.181, 0, 0.179, 0.168, 0.173, 0.188, 0.175, 0.165, 0.198, 0.067, 0.172, 0.123, 0.076, 0.168, 0.138, 0.199, 0.161, 0.069, 0.091, 0.059, 0.066, 0.136, 0.145, 0.116, 0.041, 0.218, 0, 0.143, 0.151, 0.141, 0.18, 0.14, 0.144, 0, 0.137, 0.163, 0.092, 0.132, 0.148, 0.138, 0.116, 0.133, 0.149, 0.147, 0.05, 0.076, 0.119, 0.178, 0.177, 0.081, 0.111, 0.087, 0.182, 0.085]
    # scc_duration_list = [168, 160, 164, 124, 188, 132, 116, 124, 160, 160, 164, 120, 140, 144, 124, 136, 136, 88, 152, 140, 140, 116, 104, 120, 112, 164, 136, 112, 96, 112, 140, 144, 196, 192, 120, 140, 228, 140, 32, 140, 148, 108, 164, 152, 132, 140, 176, 132, 136, 120, 112, 108, 144, 116, 132, 36, 192, 84, 148, 112, 132, 152, 176, 176, 176, 112, 120, 140, 168, 140, 92, 132, 92, 124, 68, 32, 92, 148, 164, 104, 32, 148, 188, 32, 112, 148, 168, 64, 140, 140, 96, 124, 176, 108, 108, 216, 216, 116, 112, 132, 148, 132, 132, 140, 160, 132, 148, 192, 160, 116, 140, 120, 152, 140, 144, 124, 160]
    scc_duration_list = [168, 184, 220, 136, 140, 104, 104, 144, 240, 188, 160, 148, 116, 164, 124, 140, 132, 104, 304, 184, 144, 148, 116, 68, 132, 120, 112, 124, 116, 148, 212, 144, 132, 172, 116, 160, 304, 144, 60, 180, 100, 112, 172, 192, 144, 184, 292, 200, 96, 116, 156, 144, 144, 80, 160, 160, 168, 76, 176, 136, 172, 192, 264, 140, 104, 112, 140, 176, 208, 148, 116, 140, 80, 152, 140, 116, 96, 120, 112, 96, 48, 188, 48, 84, 96, 228, 172, 172, 124, 96, 128, 120, 196, 104, 88, 140, 80, 116, 112, 160, 120, 140, 112, 148, 108, 140, 152, 292, 124, 116, 140, 140, 160, 212, 140, 140, 196]
    # SCC sweep: Full 2D 
    snr_list = [0.207, 0.206, 0.211, 0.183, 0.08, 0.224, 0.095, 0.078, 0.136, 0.038, 0.034, 0.026, 0.039, 0.165, 0.13, 0.18, 0.153, 0.074, 0.08, 0.028, 0.053, 0.142, 0.188, 0.077, 0.121, 0.137, 0.085, 0.067, 0.157, 0.135, 0.036, 0.075, 0.135, 0.168, 0.045, 0.067, 0.158, 0.12, 0.074, 0.167, 0.073, 0.046, 0.149, 0.054, 0.135, 0.064, 0.119, 0.193, 0.104, 0.091, 0.04, 0.127, 0.125, 0.105, 0.054, 0.069, 0.139, 0.151, 0.119, 0.068, 0.134, 0.054, 0.11, 0.096, 0.105, 0.133, 0.149, 0.057, 0.102, 0.083, 0.097, 0.175, 0.096, 0.058, 0.161, 0.158, 0.048, 0.1, 0.093, 0.132, 0.131, 0.055, 0.028, 0.083, 0.05, 0.061, 0.06, 0.082, 0.114, 0.065, 0.144, 0.142, 0.116, 0.095, 0.143, 0.121, 0.116, 0.102, 0.032, 0.061, 0.113, 0.087, 0.061, 0.119, 0.027, 0.119, 0.131, 0.144, 0.122, 0.087, 0.087, 0.067, 0.089, 0.068, 0.089, 0.043, 0.131, 0.05, 0.075, 0.039, 0.09, 0.085, 0.099, 0.123, 0.133, 0.097, 0.083, 0.04, 0.097, 0.032, 0.043, 0.148, 0.092, 0.037, 0.118, 0.051, 0.078, 0.053, 0.081, 0.056, 0.112, 0.119, 0.05, 0.044, 0.131, 0.137, 0.133, 0.074, 0.049, 0.06, 0.043, 0.063, 0.106, 0.165, 0.16, 0.05, 0.132, 0.088, 0.081, 0.062]
    # scc_duration_list = [304, 304, 304, 156, 304, 148, 244, 100, 304, 60, 304, 76, 88, 304, 112, 304, 144, 304, 304, 48, 76, 140, 144, 88, 304, 304, 304, 112, 304, 172, 304, 96, 72, 168, 128, 48, 304, 112, 124, 304, 48, 304, 304, 48, 304, 304, 168, 144, 304, 304, 60, 304, 108, 304, 48, 304, 164, 160, 304, 268, 240, 196, 304, 112, 304, 48, 264, 304, 152, 304, 184, 148, 304, 52, 160, 112, 104, 304, 88, 116, 56, 304, 68, 304, 304, 112, 52, 304, 304, 96, 304, 120, 304, 140, 304, 304, 156, 48, 304, 64, 304, 304, 132, 124, 304, 148, 304, 148, 80, 136, 124, 148, 108, 132, 132, 68, 124, 132, 304, 92, 80, 64, 304, 152, 136, 304, 48, 96, 304, 48, 64, 304, 64, 304, 216, 304, 304, 144, 176, 140, 304, 136, 104, 304, 56, 136, 76, 112, 304, 120, 164, 304, 88, 104, 128, 152, 132, 112, 100, 304]
    # SCC sweep: amps just 1.0 and 1.4 
    snr_list2 = [0.212, 0.202, 0.19, 0.176, 0.097, 0.223, 0.166, 0.056, 0.135, None, None, None, None, 0.115, 0.187, 0.211, 0.183, 0.061, 0.085, None, None, 0.194, 0.21, 0.121, 0.102, 0.214, 0.079, None, 0.176, 0.151, None, 0.069, None, 0.158, None, None, 0.15, 0.14, 0.103, 0.172, 0.088, None, 0.114, None, 0.129, None, 0.146, 0.193, 0.124, 0.081, None, 0.158, 0.099, 0.116, None, None, 0.166, 0.169, 0.113, None, 0.155, None, 0.127, None, 0.106, 0.11, 0.155, None, 0.146, 0.076, 0.106, 0.19, 0.148, None, 0.16, 0.171, None, 0.123, 0.082, 0.169, 0.03, None, None, 0.094, None, None, None, None, 0.143, None, 0.131, 0.165, 0.131, None, 0.147, 0.134, 0.097, 0.18, None, None, 0.158, 0.093, None, 0.152, None, 0.165, 0.125, 0.133, 0.123, 0.103, 0.111, None, 0.079, None, 0.122, None, 0.168, None, 0.097, None, 0.096, 0.088, 0.157, 0.158, 0.159, 0.1, 0.065, None, 0.122, None, None, 0.167, None, None, 0.135, None, 0.11, None, 0.147, None, 0.085, 0.156, None, None, None, 0.157, 0.146, 0.156, None, None, None, None, 0.149, 0.17, 0.158, None, 0.14, 0.111, 0.148, None]
    # scc_duration_list = [264, 304, 304, 168, 304, 144, 148, 120, 176, None, None, None, None, 304, 304, 304, 136, 128, 304, None, None, 160, 132, 116, 268, 304, 208, None, 304, 176, None, 60, None, 212, None, None, 304, 84, 144, 304, 212, None, 228, None, 304, None, 228, 140, 304, 304, None, 304, 264, 304, None, None, 160, 172, 188, None, 192, None, 264, None, 204, 104, 140, None, 156, 304, 204, 136, 144, None, 136, 112, None, 256, 48, 116, 96, None, None, 148, None, None, None, None, 304, None, 248, 124, 304, None, 304, 304, 124, 140, None, None, 304, 304, None, 128, None, 140, 184, 160, 84, 204, 104, None, 172, None, 140, None, 132, None, 156, None, 48, 80, 304, 176, 176, 216, 304, None, 188, None, None, 304, None, None, 188, None, 304, None, 156, None, 304, 192, None, None, None, 212, 104, 152, None, None, None, None, 104, 104, 48, None, 124, 140, 52, None]
    # scc_amp_list = [1.4, 1.4, 1.0, 1.0, 1.4, 1.4, 1.4, 1.0, 1.4, None, None, None, None, 1.4, 1.4, 1.0, 1.0, 1.0, 1.0, None, None, 1.0, 1.4, 1.0, 1.0, 1.0, 1.0, None, 1.0, 1.0, None, 1.4, None, 1.4, None, None, 1.0, 1.0, 1.4, 1.0, 1.0, None, 1.4, None, 1.4, None, 1.4, 1.4, 1.0, 1.4, None, 1.0, 1.0, 1.0, None, None, 1.4, 1.0, 1.0, None, 1.0, None, 1.0, None, 1.4, 1.4, 1.4, None, 1.0, 1.4, 1.4, 1.0, 1.4, None, 1.4, 1.4, None, 1.4, 1.0, 1.4, 1.4, None, None, 1.4, None, None, None, None, 1.0, None, 1.4, 1.0, 1.0, None, 1.0, 1.4, 1.0, 1.0, None, None, 1.0, 1.4, None, 1.0, None, 1.0, 1.0, 1.0, 1.0, 1.0, 1.4, None, 1.0, None, 1.0, None, 1.4, None, 1.0, None, 1.0, 1.0, 1.4, 1.4, 1.0, 1.0, 1.0, None, 1.0, None, None, 1.0, None, None, 1.4, None, 1.0, None, 1.0, None, 1.4, 1.0, None, None, None, 1.4, 1.4, 1.0, None, None, None, None, 1.4, 1.0, 1.4, None, 1.4, 1.4, 1.4, None]

    prep_fidelity_list = [0.7029534938859874, 0.7241410071217993, 0.6228508086337025, 0.7409512168448724, 0.2935525126375358, 0.7542954864259674, 0.6041468031580808, 0.7712536632144968, 0.6566444950570944, 0.5655046174544283, 0.7340537549614712, 0.7062209950348056, 0.7190972870607999, 0.7335670196053963, 0.7422286104725382, 0.6766523051246035, 0.7227075194847027, 0.6768336012011913, 0.6661881497042252, 0.6896271787272867, 0.7430113654928503, 0.6236753722657262, 0.6793880847096185, 0.7143064927716576, 0.5695113872566384, 0.6737206758625858, 0.7143440553626683, 0.6877551438556412, 0.6446313627533844, 0.7017886707698276, 0.7031627895912459, 0.7072380474470246, 0.7506760823722072, 0.6444891429138274, 0.650890009594846, 0.6920485569716457, 0.7201698606941512, 0.60670200328326, 0.5894873131606171, 0.641401223668745, 0.7210772221486743, 0.6801702516775086, 0.44655381229823554, 0.7171385991995467, 0.6816429688470695, 0.6879404151121263, 0.6093162442793276, 0.6889597559787989, 0.6010885962041363, 0.6663597048077694, 0.5949416591965662, 0.6469562575173182, 0.728714463019381, 0.6874330664381139, 0.6928703252004071, 0.6950033400827879, 0.7218483144066311, 0.5956990024958378, 0.6863201165087107, 0.5932694632962714, 0.6890160481874854, 0.655515971134492, 0.657037130099258, 0.705252892357855, 0.7786780059069358, 0.665970007556629, 0.7465119231630474, 0.7486837024203105, 0.6920693198769481, 0.3611252510703641, 0.5205679943506748, 0.7284248198109047, 0.5139769731374807, 0.7209537961010746, 0.7585020699005565, 0.6903572009416821, 0.585838918841846, 0.7148604497510886, 0.7136577725140401, 0.6071435566130093, 0.7434328646453574, 0.7690038398517386, 0.7423763909630026, 0.25886959054584235, 0.7358050084259684, 0.5769181481543877, 0.6872723110191601, 0.7207610512996692, 0.69495694556857, 0.7541498546975705, 0.6298395432903121, 0.607104154856901, 0.73455925393991, 0.22957855429281948, 0.6771868893333238, 0.6827941666557424, 0.43130337490324167, 0.07411438959345862, 0.6945028933793915, 0.7786373755635871, 0.7423783813049775, 0.6101635694745573, 0.1938032326743535]
    readout_fidelity_list = [0.9688044990715144, 0.9514063844751961, 0.9688769436100197, 0.9641949311198268, 0.847177199814062, 0.9568704705291954, 0.9984847612758602, 0.816421052925206, 0.9942722669544317, 0.8749819727587355, 0.9611006172213226, 0.9706488698768883, 0.9611340599220835, 0.9698410106027282, 0.9270197485345437, 0.9924678184948563, 0.9824018801926846, 0.966143913483525, 0.9038703786243549, 0.9776521771395241, 0.9058480393326721, 0.9538862674332026, 0.9925247653347922, 0.9180989231937564, 0.993135427904695, 0.9782837881405991, 0.97963337424356, 0.9877985596845422, 0.9651632191724555, 0.97790723196456, 0.932103631205577, 0.9677499235996958, 0.8818978638230962, 0.9951617692302577, 0.9777209134644756, 0.730400903448591, 0.9779379662009355, 0.7645243632405326, 0.9969513297499841, 0.9656364295802362, 0.9346837035279206, 0.8883006412943756, 0.978395833701813, 0.93363222164878, 0.977019809811571, 0.9350395532990888, 0.9919850266135013, 0.9399383450555051, 0.9710126181245683, 0.9949912668165968, 0.9976265183154931, 0.9895407941292331, 0.9498183805150814, 0.9509706251280421, 0.93633165258562, 0.9804246104637805, 0.9419647984336766, 0.9978702457232009, 0.9513766046266074, 0.8366529817300092, 0.9806257579832072, 0.9890304900025553, 0.9789288673949057, 0.9639464521970528, 0.8688290473425304, 0.9719548052221133, 0.9835408515387369, 0.9433631659402127, 0.9805432669877248, 0.9820506418057253, 0.9269741164702274, 0.963936138904116, 0.9962097405417185, 0.9329093452066429, 0.8420128942878584, 0.9754412167401575, 0.9925888905187685, 0.880848995435944, 0.8098006259636209, 0.9941278199973324, 0.8832234893977475, 0.9513482271295688, 0.9530362279384703, 0.893782540827057, 0.9270609069842044, 0.9826246326537309, 0.9547282333497022, 0.971832431847879, 0.9073704822741532, 0.9261486633957353, 0.9923277318491459, 0.854731849619671, 0.9679633818519171, 0.9719528494790816, 0.9557730322712704, 0.9929119860226663, 0.9065404775849846, 0.9161803872609, 0.9710632882189207, 0.9591959710212463, 0.8783189800377531, 0.9488484120419992, 0.9844166921338738]
    red_chi_sq_list = [0.9226006164038223, 0.8394512163138113, 0.8300329526030091, 0.6375682501419008, 0.960934115404814, 0.8090124236151802, 3.3306373402473057, 0.7587403777657264, 2.397815867677327, 0.9062345785309248, 0.8021368134190159, 1.6029512985465775, 1.0228431038796366, 0.8256472331118277, 0.8209555789586398, 1.5457918897795564, 0.8827894924312786, 1.5135139767660335, 1.1982348390556083, 0.8259628988123171, 0.5067704950667581, 0.8805513194364127, 1.6161028634979913, 0.7833660528369646, 2.4824708274109657, 1.107892205541358, 0.9006801564541498, 1.2373216414211956, 1.1704669927709381, 1.06827932443789, 0.8598936532210887, 0.9016409210602291, 1.0163250484758692, 2.168643178419172, 1.2570717655864228, 0.7440243030025026, 1.199269960989459, 1.0953148286105359, 3.296423466026412, 2.1288212915888605, 0.789798254278298, 1.2868774964195322, 1.2595319830156495, 0.7295174370138822, 1.1709147318532485, 0.9176941347248658, 2.951513023782647, 0.951524289344862, 1.2604085415687658, 1.786621513799328, 2.4989388156934615, 1.7774102441042658, 1.017170615007283, 0.6256559360825594, 0.8638469209805578, 1.1324017889804208, 1.0642178775074678, 2.809615259433896, 0.980520554842089, 0.8882908068331088, 1.1436904620066857, 1.3911551418931376, 0.823355044545783, 1.223580106596424, 0.8471156888411187, 1.1195255875176355, 1.4691199121890823, 0.87173348415454, 1.2909927218468444, 6.547381365346114, 0.9834632613794224, 1.281506112827628, 2.448771055795894, 1.088296455872872, 1.0909344907241891, 1.1417809223591147, 2.8824806947215236, 0.7430441079143196, 1.2176032456903747, 2.7194555008788797, 0.834964000446476, 0.8181463623412424, 0.9399323641049441, 0.9888638730162348, 1.1114273726862431, 1.5505946581686465, 1.3597973388294242, 1.0464481087173139, 0.9362675088363762, 0.998201276026406, 2.413643452782471, 1.2737560320169652, 1.0556569064909072, 1.1697594972046272, 0.9671141969252813, 1.9683613180462272, 0.6487565567434719, 0.9691162889661228, 0.9817015865304953, 0.9761956782366511, 1.0394601195824655, 1.046149969508797, 1.0243543826767634]

    snr_list_from_resonance = [0.17774355921275042, 0.16100987032353067, 0.13677097057469514, 0.16380018666230875, 0.03853638482790593, 0.20731333097267335, 0.09876470150052456, 0.08183294964354494, 0.1246151530208468, 0.11247648051912171, 0.04390099507479191, 0.18881403061567176, 0.14892306957480592, 0.02944738982542843, 0.07683175381037124, 0.14345966384938943, 0.1661994258748476, 0.04066586762964176, 0.11046988846974745, 0.1277820326527986, 0.052194372150444035, 0.12064056178798445, 0.10784036782169813, 0.05299961504117946, 0.15802136777663622, 0.0946642795540692, 0.0803292814913278, 0.043697809578309314, 0.12252217747515479, 0.02197506182889821, 0.13151043764690992, 0.10819169328789556, 0.10797613249858289, 0.17326590842230383, 0.08958720670689067, 0.07501382327953252, 0.12066224034717361, 0.004544350526148948, 0.09142732176753793, 0.1501807391040204, 0.1537113298483631, 0.09873457729469304, 0.14041446504807054, 0.10381389237582375, 0.10750268883128927, 0.10337188838515941, 0.15628179599916553, 0.10053955682414573, 0.05662602458189284, 0.08087967683176395, 0.18059194000094056, 0.09845799988557247, 0.14613874936484483, 0.17578489203824285, 0.07755942703079502, 0.06773006370951189, 0.10782745305114706, 0.04037477533838154, 0.07907591255892463, 0.10750137916072282, 0.1155631929059236, 0.1374991218680856, 0.07942261593565046, 0.11667643137601869, 0.12088799698610042, 0.09991552365477659, 0.049523497000793205, 0.08893154110655495, 0.025770354150570655, 0.1303030713381284, 0.14622475401501836, 0.091745278907141, 0.13653903161645234, 0.1317966796114374, 0.09396037160687128, 0.10460942504657664, 0.093551883872755, 0.12371620551532755, 0.1287631746956545, 0.1158724414958125, 0.12035660327735165, 0.051213055860286426, 0.10961292069055319, 0.10788185557705382, 0.1399686628505069, 0.09513285018795897, 0.060922371189182215, 0.07982161769697853, 0.11634893403329778, 0.12344857413245512, 0.07830085688261275, 0.11520756624428106, 0.08196797839250042, 0.09901186475410613, 0.14341327319343003, 0.12291609098515775, 0.11770692278218561, 0.046847800097108684, 0.15538195327769821, 0.1391859161627152, 0.13519160537915956, 0.07846015030187811, 0.10975042344606116]
    #103nvs 
    # include_inds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 13, 14, 15, 16, 17, 18, 21, 22, 23, 24, 25, 26, 28, 29, 31, 33, 36, 37, 38, 39, 40, 42, 44, 46, 47, 48, 49, 51, 52, 53, 56, 57, 58, 60, 62, 64, 65, 66, 68, 69, 70, 71, 72, 74, 75, 77, 78, 79, 80, 83, 88, 90, 91, 92, 94, 95, 96, 97, 100, 101, 103, 105, 106, 107, 108, 109, 110, 112, 114, 116, 118, 120, 121, 122, 123, 124, 125, 126, 128, 131, 134, 136, 138, 140, 141, 145, 146, 147, 152, 153, 154, 156, 157, 158]
    #117nvs 
    include_inds =[0, 1, 2, 3, 5, 6, 7, 8, 13, 14, 15, 16, 17, 18, 20, 21, 22, 23, 24, 25, 26, 28, 29, 31, 32, 33, 34, 36, 37, 39, 42, 44, 45, 46, 47, 48, 49, 51, 52, 53, 55, 56, 57, 58, 60, 61, 62, 64, 65, 66, 68, 69, 70, 71, 72, 73, 74, 75, 77, 79, 83, 84, 85, 88, 89, 90, 91, 92, 94, 95, 96, 97, 99, 100, 101, 102, 103, 105, 106, 107, 108, 109, 110, 111, 113, 114, 116, 117, 118, 120, 122, 123, 124, 125, 128, 131, 132, 134, 136, 137, 138, 140, 141, 142, 145, 146, 147, 148, 149, 152, 153, 154, 155, 156, 157, 158, 159]
    # Initialize a list with None values
    arranged_scc_duration_list = [None] * num_nvs
    for i, idx in enumerate(include_inds):
        arranged_scc_duration_list[idx] = scc_duration_list[i]
    scc_duration_list = arranged_scc_duration_list

    final_drop_inds = [23, 73, 89, 99, 117, 120, 132, 137, 155, 157, 159]
    include_inds = [ind for ind in include_inds if ind not in final_drop_inds]
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
        if ind not in include_inds:
            continue
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
            # threshold=threshold_list[ind],
            pulse_durations={VirtualLaserKey.SCC: scc_duration_list[ind]},
            pulse_amps={
                # VirtualLaserKey.SCC: scc_amp_list[ind],
                #     VirtualLaserKey.CHARGE_POL: charge_pol_amps[ind],
            },
        )
        nv_list.append(nv_sig)

    # Additional properties for the representative NV
    nv_list[0].representative = True
    # nv_list[1].representative = True
    repr_nv_sig = widefield.get_repr_nv_sig(nv_list)
    nv_sig = widefield.get_repr_nv_sig(nv_list)
    # print(f"Created NV: {nv_sig.name}, Coords: {nv_sig.coords}")
    # nv_sig.expected_counts = 1650
    # nv_sig.expected_counts = 3359.0
    # nv_sig.expected_counts = 1181.0
    nv_sig.expected_counts = 1500
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
    # nv_list = nv_list[:3]
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

        # do_compensate_for_drift(nv_sig)

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

        # for z in np.linspace(1.0, 2.0, 11):
        #     nv_sig.coords[CoordsKey.Z] = z
        # do_scanning_image_sample(nv_sig)

        # nv_sig.coords[CoordsKey.z] = 0.4
        # do_scanning_image_sample(nv_sig)

        # do_scanning_image_sample(nv_sig)
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
        # coords_key = None  # Pixel coords
        # coords_key = green_laser
        # coords_key = red_laser
        # do_optimize_loop(nv_list, coords_key)

        # nv_list = nv_list[::-1]
        # do_charge_state_histograms(nv_list)
        # do_optimize_pol_amp(nv_list)
        # do_optimize_readout_amp(nv_list)
        # do_optimize_readout_duration(nv_list)
        # optimize_readout_amp_and_duration(nv_list)
        # do_optimize_pol_duration(nv_list)
        # do_charge_state_histograms_images(nv_list, vary_pol_laser=True)
        # do_charge_state_conditional_init(nv_list)
        # do_check_readout_fidelity(nv_list)

        # do_resonance_zoom(nv_list)
        # do_rabi(nv_list)
        # do_resonance(nv_list)
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
        do_scc_snr_check(nv_list)
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
