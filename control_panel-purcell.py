# -*- coding: utf-8 -*-
"""
Control panel for the PC Rabi

Created on June 16th, 2023

@author: mccambria
"""


### Imports

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
from utils import kplotlib as kpl
from utils import positioning as pos
from utils import tool_belt as tb
from utils.constants import CoordsKey, LaserKey, NVSig

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


def do_charge_state_histograms(nv_list, charge_prep_verification=False):
    num_reps = 50
    num_runs = 10
    # num_runs = 2
    return charge_state_histograms.main(
        nv_list, num_reps, num_runs, charge_prep_verification=charge_prep_verification
    )


def do_calibrate_nvn_dist_params(nv_list):
    data = do_charge_state_histograms(nv_list, charge_prep_verification=True)
    ref_img_array = data["ref_img_array"]

    nvn_dist_params_list = []
    for nv in nv_list:
        popt = optimize.optimize_pixel_with_img_array(
            ref_img_array, nv, return_popt=True
        )
        # bg, amp, sigma
        nvn_dist_params_list.append((popt[-1], popt[0], popt[-2]))
    print(nvn_dist_params_list)


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

    opti_coords_list = []
    for nv in nv_list:
        # Pixel coords
        if coords_key is None:
            imaging_laser = tb.get_laser_name(LaserKey.IMAGING)
            if scanning_from_pixel:
                widefield.set_nv_scanning_coords_from_pixel_coords(nv, imaging_laser)
            opti_coords = do_optimize_pixel(nv)
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
    max_tau = 208
    num_steps = 13
    num_reps = 10
    num_runs = 50
    optimize_scc.main(nv_list, num_steps, num_reps, num_runs, min_tau, max_tau)


def do_scc_snr_check(nv_list):
    num_reps = 20
    num_runs = 10
    scc_snr_check.main(nv_list, num_reps, num_runs)


def do_simple_correlation_test(nv_list):
    num_reps = 60
    num_runs = 40
    simple_correlation_test.main(nv_list, num_reps, num_runs)

    for ind in range(4):
        for flipped in [True, False]:
            for nv_ind in range(3):
                nv = nv_list[nv_ind]
                if ind == nv_ind:
                    nv.spin_flip = flipped
                else:
                    nv.spin_flip = not flipped
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
    freq_range = 0.180
    num_steps = 40
    num_reps = 10
    # num_runs = 120
    num_runs = 20
    resonance.main(nv_list, num_steps, num_reps, num_runs, freq_center, freq_range)


def do_resonance_zoom(nv_list):
    freq_center = 2.8572
    freq_range = 0.060
    num_steps = 20
    num_reps = 10
    num_runs = 20
    resonance.main(nv_list, num_steps, num_reps, num_runs, freq_center, freq_range)


def do_rabi(nv_list):
    min_tau = 16
    max_tau = 240 + min_tau
    num_steps = 31
    num_reps = 4
    num_runs = 20
    # num_runs = 2

    # min_tau = 64
    # num_steps = 1
    # num_reps = 50

    # nv_list[1].spin_flip = True
    rabi.main(nv_list, num_steps, num_reps, num_runs, min_tau, max_tau)
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
    num_runs = 160
    aod_freq_range = 3.0
    # laser_name = red_laser
    laser_name = green_laser
    axis_ind = 0  # 0: x, 1: y, 2: z
    uwave_ind = 0

    for laser_name in [red_laser, green_laser]:
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
    num_reps = 100
    num_runs = 100
    dark_time = 0

    charge_monitor.detect_cosmic_rays(nv_list, num_reps, num_runs, dark_time)


def do_check_readout_fidelity(nv_list):
    num_reps = 200
    num_runs = 10

    charge_monitor.check_readout_fidelity(nv_list, num_reps, num_runs)


def do_charge_quantum_jump(nv_list):
    num_reps = 2000

    charge_monitor.charge_quantum_jump(nv_list, num_reps)


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
    # opx.constant_ac(
    #     [],  # Digital channels
    #     [7],  # Analog channels
    #     [0.4],  # Analog voltages
    #     [0],  # Analog frequencies
    # )
    # Green
    opx.constant_ac(
        [4],  # Digital channels
        # [3, 4],  # Analog channels
        # [0.03, 0.03],  # Analog voltages
        # [110, 110],  # Analog frequencies
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
    #     # [0.17, 0.17],  # Analog voltages
    #     # [
    #     #     75,
    #     #     75,
    #     # ],  # Analog frequencies                                                                                                                                                                              uencies
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
    z_coord = 4.42
    magnet_angle = 90
    date_str = "2024_03_12"
    global_coords = [None, None, z_coord]

    # endregion
    # region Coords (from March 12th)

    pixel_coords_list = [
        [130.424, 152.027],
        [149.37, 142.24],
        [161.89, 127.546],
        [134.965, 126.404],
        [126.476, 111.281],
        [110.102, 110.01],
        [111.687, 124.857],
        [82.449, 109.831],
        [144.021, 186.017],
        [159.297, 184.106],
        [173.838, 100.29],
        [171.224, 71.041],
        [157.345, 56.683],
        [104.585, 147.209],
        [60.285, 99.869],
    ]
    num_nvs = len(pixel_coords_list)
    green_coords_list = [
        [108.521, 109.715],
        [108.921, 109.958],
        [109.169, 110.303],
        [108.587, 110.341],
        [108.369, 110.67],
        [108.004, 110.702],
        [108.103, 110.336],
        [107.394, 110.71],
        [108.863, 108.943],
        [109.167, 108.999],
        [109.409, 110.958],
        [109.348, 111.623],
        [109.039, 111.972],
        [107.953, 109.844],
        [106.937, 110.9],
    ]
    red_coords_list = [
        [73.14, 74.937],
        [73.457, 75.128],
        [73.685, 75.413],
        [73.248, 75.478],
        [73.037, 75.735],
        [72.835, 75.712],
        [72.796, 75.461],
        [72.33, 75.684],
        [73.406, 74.299],
        [73.669, 74.341],
        [74.036, 75.932],
        [73.868, 76.536],
        [73.672, 76.756],
        [72.707, 75.044],
        [71.834, 75.944],
    ]
    threshold_list = [
        27.5,
        29.5,
        26.5,
        26.5,
        25.5,
        25.5,
        27.5,
        30.5,
        25.5,
        24.5,
        21.5,
        20.5,
        18.5,
        26.5,
        19.5,
    ]
    nvn_dist_params_list = [
        (0.08315799961152527, 0.3864189769082783, 4.048553589061115),
        (0.11005828903421873, 0.3814565061766918, 4.100601509449526),
        (0.12180045422854645, 0.3517852928374418, 3.916893498011493),
        (0.10669062511770351, 0.3085388288047047, 4.144928339948053),
        (0.16111318055498183, 0.2998192413934377, 3.458275332418897),
        (0.14416026317186623, 0.2940628997005982, 3.471266801912637),
        (0.11468325059939738, 0.3395653477068342, 3.9594836469313615),
        (0.09657099298862892, 0.29912341831268285, 4.356696175225211),
        (0.09422259066462835, 0.31203234269710906, 4.089553359350482),
        (0.09569534729173282, 0.3013564204854867, 4.020874846264347),
        (0.09743199369003977, 0.261914435264353, 4.003147918604017),
        (0.07226343042817052, 0.2483207060558696, 4.583601391686538),
        (0.0669240107004653, 0.19765853908103118, 4.779803178027258),
        (0.10392426590628977, 0.2710379213589781, 4.029635174209021),
        (0.0742268651079635, 0.21851859748730523, 4.316543516909557),
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
    # region NV list construction

    # nv_list[i] will have the ith coordinates from the above lists
    nv_list = []
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
        )
        nv_list.append(nv_sig)

    # nv_list = nv_list[::-1]  # flipping the order of NVs
    # Additional properties for the representative NV
    nv_list[0].representative = True
    nv_list[0].expected_counts = 1200
    nv_sig = widefield.get_repr_nv_sig(nv_list)

    # nv_inds = [0]
    # nv_inds.extend(list(range(8, 15)))
    # nv_list = [nv_list[ind] for ind in nv_inds]
    # for nv in nv_list:
    #     nv.threshold = 27.5

    # for nv in nv_list:
    #     nv.init_spin_flipped = True
    # nv_list[1].init_spin_flipped = True
    # nv_list[3].init_spin_flipped = True
    # seq_args = widefield.get_base_scc_seq_args(nv_list, 0)
    # print(seq_args)

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
        # widefield.set_pixel_drift([0, -40])
        # widefield.set_all_scanning_drift_from_pixel_drift()

        # do_optimize_z(nv_sig)

        # pos.set_xyz_on_nv(nv_sig)

        # for z in np.linspace(4.5, 4.0, 11):
        #     nv_sig.coords[CoordsKey.GLOBAL][2] = z
        #     do_widefield_image_sample(nv_sig, 20)

        # do_scanning_image_sample(nv_sig)
        # do_scanning_image_sample_zoom(nv_sig)
        # do_widefield_image_sample(nv_sig, 50)
        # do_widefield_image_sample(nv_sig, 100)

        # do_image_nv_list(nv_list)
        # do_image_single_nv(nv_sig)

        # for nv_sig in nv_list:
        #     widefield.reset_all_drift()
        #     # do_optimize_pixel(nv_sig)
        #     do_optimize_green(nv_sig)
        # do_optimize_red(nv_sig)
        # do_image_single_nv(nv_sig)

        # for ind in range(10):
        #     do_optimize_pixel(nv_sig)
        #     time.sleep(5)

        # optimize.optimize_pixel_and_z(nv_sig, do_plot=True)
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
        # do_optimize_loop(nv_list, coords_key, scanning_from_pixel=True)

        # num_nvs = len(nv_list)
        # for ind in range(num_nvs):
        #     if ind == 0:
        #         continue
        #     nv = nv_list[ind]
        #     green_coords = nv[green_coords_key]
        #     nv[green_coords_key][0] += 0.500

        # do_charge_state_histograms(nv_list)
        do_check_readout_fidelity(nv_list)
        # do_calibrate_nvn_dist_params(nv_list)

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

        # do_scc_snr_check(nv_list)
        # do_optimize_scc(nv_list)
        # do_crosstalk_check(nv_sig)
        # do_spin_pol_check(nv_sig)
        # do_calibrate_green_red_delay()
        # do_simple_correlation_test(nv_list)

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
