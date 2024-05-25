# -*- coding: utf-8 -*-
"""
Control panel for the PC Rabi

Created on June 16th, 2023

@author: mccambria
"""


### Imports

import sys
import time

import matplotlib.pyplot as plt
import numpy as np

from majorroutines.widefield import (
    calibrate_iq_delay,
    charge_monitor,
    charge_state_histograms,
    correlation_test,
    crosstalk_check,
    image_sample,
    optimize,
    optimize_scc,
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
from utils import common, widefield
from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import positioning as pos
from utils import tool_belt as tb
from utils.constants import CoordsKey, LaserKey, NVSig

green_laser = "laser_INTE_520"
red_laser = "laser_COBO_638"
yellow_laser = "laser_OPTO_589"

### Major Routines


def do_widefield_image_sample(nv_sig, num_reps=1):
    return image_sample.widefield_image(nv_sig, num_reps)


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


def do_charge_state_histograms(nv_list, verify_charge_states=False):
    num_reps = 200
    num_runs = 40
    # num_runs = 2
    return charge_state_histograms.main(
        nv_list, num_reps, num_runs, verify_charge_states=verify_charge_states
    )


def do_optimize_green(nv_sig, do_plot=True):
    coords_key = tb.get_laser_name(LaserKey.IMAGING)
    ret_vals = optimize.main(
        nv_sig,
        coords_key=coords_key,
        no_crash=True,
        do_plot=do_plot,
        axes_to_optimize=[0, 1],
    )
    opti_coords = ret_vals[0]
    return opti_coords


def do_optimize_red(nv_sig, do_plot=True):
    coords_key = red_laser
    ret_vals = optimize.main(
        nv_sig,
        coords_key=coords_key,
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
            # opti_coords = do_optimize_pixel(nv)
            opti_coords = optimize.optimize_pixel_with_img_array(img_array, nv_sig=nv)
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
            optimize.optimize_pixel_and_z(repr_nv_sig, do_plot=False)
            drift = pos.get_drift(coords_key)
            drift = [-1 * el for el in drift]
            opti_coords = pos.adjust_coords_for_drift(opti_coords, drift=drift)

        opti_coords_list.append(opti_coords)

    # Report back
    for opti_coords in opti_coords_list:
        r_opti_coords = [round(el, 3) for el in opti_coords]
        print(f"{r_opti_coords},")


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


def do_optimize_scc(nv_list):
    min_tau = 16
    max_tau = 224
    # min_tau = 100
    # max_tau = 308
    num_steps = 14
    num_reps = 5

    # min_tau = 16
    # max_tau = 104
    # num_steps = 12
    # num_reps = 8

    # num_runs = 2
    num_runs = 20 * 25

    optimize_scc.main(nv_list, num_steps, num_reps, num_runs, min_tau, max_tau)


def do_scc_snr_check(nv_list):
    num_reps = 100
    num_runs = 30
    scc_snr_check.main(nv_list, num_reps, num_runs)


def do_simple_correlation_test(nv_list):
    num_reps = 100
    num_runs = 2000
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
    freq_range = 0.180
    num_steps = 40
    num_reps = 10
    num_runs = 100
    num_runs = 50

    num_reps = 3
    num_runs = 150
    # num_runs = 2

    resonance.main(nv_list, num_steps, num_reps, num_runs, freq_center, freq_range)


def do_resonance_zoom(nv_list):
    # freq_center = 2.8572
    for freq_center in (2.858, 2.812):
        freq_range = 0.060
        num_steps = 20
        num_reps = 15
        num_runs = 120
        resonance.main(nv_list, num_steps, num_reps, num_runs, freq_center, freq_range)


def do_rabi(nv_list):
    min_tau = 16
    max_tau = 240 + min_tau
    # max_tau = 796
    num_steps = 31
    # num_steps = 40
    num_reps = 10
    num_runs = 100
    num_runs = 50
    # num_runs = 2
    # uwave_ind_list = [1]
    uwave_ind_list = [0, 1]

    # min_tau = 64
    # num_steps = 1
    # num_reps = 50

    # nv_list[1].spin_flip = True
    rabi.main(nv_list, num_steps, num_reps, num_runs, min_tau, max_tau, uwave_ind_list)
    # for ind in range(4):
    #     for flipped in [True, False]:
    #         for nv_ind in range(3):
    #             nv = nv_list[nv_ind]
    #             if ind == nv_ind:
    #                 nv.spin_flip = flipped
    #             else:
    #                 nv.spin_flip = not flipped
    #         rabi.main(nv_list, num_steps, num_reps, num_runs, min_tau, max_tau)


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
    dark_time = 1e9

    charge_monitor.detect_cosmic_rays(nv_list, num_reps, num_runs, dark_time)


def do_check_readout_fidelity(nv_list):
    num_reps = 200
    num_runs = 40

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
    # opx.constant_ac(
    #     [],  # Digital channels
    #     [7],  # Analog channels
    #     [0.34],  # Analog voltages
    #     [0],  # Analog frequencies
    # )
    # Green
    # opx.constant_ac(
    #     [4],  # Digital channels
    #     [3, 4],  # Analog channels
    #     [0.11, 0.11],  # Analog voltages
    #     [110, 110],  # Analog frequencies
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
    opx.constant_ac(
        [1],  # Digital channels
        [2, 6],  # Analog channels
        [0.19, 0.19],  # Analog voltages
        [
            75,
            75,
        ],  # Analog frequencies                                                                                                                                                                       uencies
        # [73.76, 75.257],
    )
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
    z_coord = 4.57
    magnet_angle = 90
    date_str = "2024_03_12"
    global_coords = [None, None, z_coord]

    # endregion
    # region Coords (from March 12th)

    pixel_coords_list = [
        [131.447, 154.371],
        [149.612, 145.434],
        [162.289, 130.925],
        [135.679, 128.482],
        [126.597, 113.929],
        [110.558, 113.266],
        [112.347, 127.94],
        [83.832, 112.609],
        [144.666, 188.907],
        [159.542, 186.902],
        [174.584, 103.798],
        [171.363, 74.501],
        [158.976, 59.177],
        [105.906, 149.857],
        [60.542, 102.178],
    ]
    num_nvs = len(pixel_coords_list)
    green_coords_list = [
        [108.485, 109.627],
        [108.925, 109.878],
        [109.177, 110.226],
        [108.606, 110.28],
        [108.363, 110.59],
        [107.995, 110.61],
        [108.077, 110.238],
        [107.374, 110.612],
        [108.848, 108.844],
        [109.176, 108.916],
        [109.383, 110.881],
        [109.319, 111.517],
        [109.02, 111.876],
        [107.923, 109.72],
        [106.92, 110.791],
    ]
    red_coords_list = [
        [73.166, 74.936],
        [73.442, 75.083],
        [73.701, 75.35],
        [73.191, 75.334],
        [72.997, 75.654],
        [72.835, 75.638],
        [72.754, 75.386],
        [72.278, 75.661],
        [73.401, 74.276],
        [73.64, 74.29],
        [73.978, 75.856],
        [73.854, 76.464],
        [73.673, 76.742],
        [72.754, 74.993],
        [71.845, 75.925],
    ]
    threshold_list = [
        27.5,
        29.5,
        27.5,
        25.5,
        26.5,
        25.5,
        27.5,
        29.5,
        23.5,
        22.5,
        23.5,
        21.5,
        19.5,
        23.5,
        20.5,
    ]
    nvn_dist_params_list = [
        (0.0743790997693472, 0.49243735948571105, 4.273609343036658),
        (0.10309718046655189, 0.46973570049751356, 4.197247761091586),
        (0.09180231145372664, 0.47455038483121154, 4.106815710367739),
        (0.10317066887596402, 0.38318795496960856, 4.154859076199519),
        (0.11666315685494058, 0.39051320471698014, 3.9673747080502557),
        (0.10486219919226811, 0.3510649292843532, 4.205933398393933),
        (0.0929983642221821, 0.41660935339954097, 4.329751354770707),
        (0.1405594825258909, 0.3166155422804541, 3.95928440378325),
        (0.07885560222640701, 0.3654986245171775, 4.119590214945802),
        (0.0813716277445373, 0.3477543594526663, 4.197370425075404),
        (0.08829324682066712, 0.3303466874395145, 4.105873465693761),
        (0.07840901367552365, 0.2864792643147898, 4.317161603008103),
        (0.06851973774302232, 0.22769362621930464, 4.669128290191102),
        (0.10114463323084008, 0.31435370475412344, 4.060191867734814),
        (0.07685628782646109, 0.2497550236464285, 4.350185904031819),
    ]
    scc_duration_list = [
        124,
        104,
        116,
        208,
        92,
        132,
        76,
        152,
        120,
        100,
        200,
        136,
        96,
        204,
        116,
    ]
    # endregion
    # region Coords (smiley)

    # pixel_coords_list = [
    #     [142.851, 193.093],
    #     [161.181, 184.12],
    #     [173.95, 169.448],
    #     [186.133, 142.627],
    #     [183.492, 113.143],
    #     [170.196, 98.414],
    # ]
    # green_coords_list = [
    #     [108.712, 108.84],
    #     [109.137, 109.074],
    #     [109.363, 109.421],
    #     [109.587, 110.096],
    #     [109.532, 110.746],
    #     [109.22, 111.088],
    # ]
    # red_coords_list = [
    #     [73.62, 74.141],
    #     [73.813, 74.268],
    #     [74.074, 74.529],
    #     [74.37, 75.109],
    #     [74.26, 75.662],
    #     [74.009, 75.891],
    # ]
    # threshold_list = [25.5, 26.5, 25.5, 21.5, 21.5, 20.5]
    # nvn_dist_params_list = [
    #     (0.08864254977843133, 0.31782688165126877, 3.8583663986211434),
    #     (0.07940066813697125, 0.3064650349721238, 4.313006125005059),
    #     (0.08060280012144272, 0.28900824292747535, 4.1945931058872095),
    #     (0.0970023907871867, 0.16389041639281693, 3.7315296874137602),
    #     (0.05820311354749401, 0.22557983981381505, 4.7204760434244895),
    #     (0.079625551762134, 0.16060140009866478, 4.222986241289107),
    # ]

    # endregion
    # region Coords (publication set)
    pixel_coords_list = [
        [131.144, 129.272],
        [161.477, 105.335],
        [135.139, 104.013],
        [110.023, 87.942],
        [144.169, 163.787],
        [173.93, 78.505],
        [171.074, 49.877],
        [170.501, 132.597],
        [137.025, 74.662],
        [58.628, 139.616],
    ]
    num_nvs = len(pixel_coords_list)
    green_coords_list = [
        [108.507, 110.208],
        [109.089, 110.756],
        [108.53, 110.804],
        [107.93, 111.171],
        [108.778, 109.393],
        [109.305, 111.414],
        [109.237, 112.058],
        [109.294, 110.155],
        [108.479, 111.479],
        [106.85, 109.898],
    ]
    red_coords_list = [
        [73.293, 75.462],
        [73.83, 75.913],
        [73.345, 75.937],
        [72.85, 76.202],
        [73.536, 74.8],
        [74.025, 76.402],
        [73.968, 76.974],
        [74.035, 75.432],
        [73.333, 76.454],
        [71.956, 75.199],
    ]
    threshold_list = [27.5, 27.5, 25.5, 23.5, 27.5, 22.5, 17.5, 24.5, 22.5, 20.5]
    nvn_dist_params_list = [
        (0.06949197853215423, 0.5631391420690471, 4.216881064360549),
        (0.11193774130398557, 0.47231178723798944, 4.014239435395525),
        (0.10381090503410967, 0.3769014823750456, 4.2066099249824465),
        (0.0885175358700188, 0.3582170426257401, 4.128432914770158),
        (0.0805859848201578, 0.527147580304454, 4.126784289287142),
        (0.08303824820569747, 0.3533486405172414, 4.153486955348017),
        (0.0798138155994101, 0.25658246969626625, 4.152802177012195),
        (0.06476534188853134, 0.493100136132556, 4.172789007253776),
        (0.08984212570556235, 0.34267315254366004, 4.171807751951278),
        (0.0827199390683145, 0.3142148295165403, 4.010679074571614),
    ]
    scc_duration_list = [145, 160, 163, 187, 168, 188, 143, 174, 193, 156]
    scc_duration_list = [4 * round(el / 4) for el in scc_duration_list]
    # scc_duration_list = [None] * num_nvs
    # endregion
    # region NV list construction

    # nv_list[i] will have the ith coordinates from the above lists
    nv_list: list[NVSig] = []
    for ind in range(num_nvs):
        coords = {
            CoordsKey.GLOBAL: global_coords,
            CoordsKey.PIXEL: pixel_coords_list.pop(0),
            green_laser: green_coords_list.pop(0),
            red_laser: red_coords_list.pop(0),
        }
        nv_sig = NVSig(
            name=f"{sample_name}-nv{ind}_{date_str}",
            coords=coords,
            threshold=threshold_list[ind],
            nvn_dist_params=nvn_dist_params_list[ind],
            scc_duration=scc_duration_list[ind],
        )
        nv_list.append(nv_sig)

    # Additional properties for the representative NV
    nv_list[0].representative = True
    nv_sig = widefield.get_repr_nv_sig(nv_list)
    nv_sig.expected_counts = 1150
    num_nvs = len(nv_list)

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
    #     pixel_drift = [-el for el in pixel_drift]
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
        # widefield.set_pixel_drift([+14, +15])  # [131.144, 129.272]
        # widefield.set_all_scanning_drift_from_pixel_drift()

        # do_optimize_z(nv_sig)

        # pos.set_xyz_on_nv(nv_sig)

        # for z in np.linspace(4.5, 4.0, 11):
        #     nv_sig.coords[CoordsKey.GLOBAL][2] = z
        #     do_widefield_image_sample(nv_sig, 20)

        # do_scanning_image_sample(nv_sig)
        # do_scanning_image_sample_zoom(nv_sig)
        # do_widefield_image_sample(nv_sig, 20)
        # do_widefield_image_sample(nv_sig, 100)

        # do_image_nv_list(nv_list)
        # do_image_single_nv(nv_sig)

        # for nv_sig in nv_list:
        #     widefield.reset_all_drift()
        #     # do_optimize_pixel(nv_sig)
        #     do_optimize_green(nv_sig)
        # do_optimize_red(nv_sig)
        # do_image_single_nv(nv_sig)

        optimize.optimize_pixel_and_z(nv_sig, do_plot=True)
        # for ind in range(20):
        #     do_optimize_pixel(nv_sig)
        # do_optimize_pixel(nv_sig)
        # do_optimize_z(nv_sig)
        # do_optimize_green(nv_sig)
        # do_optimize_red(nv_sig)

        # widefield.reset_all_drift()
        # coords_key = None  # Pixel coords
        # coords_key = green_laser
        # coords_key = red_laser
        # do_optimize_loop(nv_list, coords_key, scanning_from_pixel=False)

        # nv_list = nv_list[::-1]
        # do_charge_state_histograms(nv_list)
        # do_check_readout_fidelity(nv_list)

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
        # do_detect_cosmic_rays(nv_list)
        # do_check_readout_fidelity(nv_list)
        # do_charge_quantum_jump(nv_list)

        # do_opx_constant_ac()
        # do_opx_square_wave()

        # nv_list = nv_list[::-1]
        # do_scc_snr_check(nv_list)
        # do_optimize_scc(nv_list)
        # do_crosstalk_check(nv_sig)
        # do_spin_pol_check(nv_sig)
        # do_calibrate_green_red_delay()
        # do_simple_correlation_test(nv_list)

        # do_simple_correlation_test(nv_list)

        # for nv in nv_list:
        #     nv.spin_flip = False
        # for nv in nv_list[::2]:
        #     nv.spin_flip = True
        # do_simple_correlation_test(nv_list)

        # for nv in nv_list:
        #     nv.spin_flip = False
        # for nv in nv_list[num_nvs // 2 :]:
        #     nv.spin_flip = True
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
        # print(res_mle)
        # print([res_mle[ind] == res_thresh[ind] for ind in range(num_nvs)])

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
        tb.reset_safe_stop()
        plt.show(block=True)

    # endregion
