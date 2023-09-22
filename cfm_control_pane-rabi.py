# -*- coding: utf-8 -*-
"""
Control panel for the PC Rabi

Created on June 16th, 2023

@author: mccambria
"""


### Imports


import numpy as np
from utils import tool_belt as tb
from utils import positioning as pos
from utils import widefield
from utils import common
from utils.constants import LaserKey, NVSpinState
from majorroutines import image_sample
from majorroutines.widefield import image_nv_list
from majorroutines.widefield import resonance
from majorroutines import optimize
import matplotlib.pyplot as plt
import copy


### Major Routines


def do_image_sample(nv_sig):
    # scan_range = 1.0
    # num_steps = 180

    # scan_range = 0.5
    scan_range = 0.4
    # scan_range = 0.2
    # num_steps = int(180 * 0.5 / 0.2)
    num_steps = 180

    # scan_range = 0.05
    # num_steps = 60

    # scan_range = 0.0
    # num_steps = 20

    image_sample.main(nv_sig, scan_range, scan_range, num_steps)


def do_image_sample_zoom(nv_sig):
    scan_range = 0.02
    num_steps = 60
    # scan_range = 0.005
    # num_steps = 30
    image_sample.main(nv_sig, scan_range, scan_range, num_steps)


def do_image_nv_list(nv_list):
    return image_nv_list.main(nv_list)


def do_image_single_nv(nv_sig):
    return image_nv_list.image_single_nv(nv_sig)


def do_optimize(nv_sig):
    optimize.main(nv_sig)


def do_optimize_plot(nv_sig):
    opti_coords, _ = optimize.main(
        nv_sig,
        set_to_opti_coords=False,
        save_data=True,
        plot_data=True,
        set_drift=False,
    )
    r_opti_coords = [
        round(opti_coords[0], 3),
        round(opti_coords[1], 3),
        round(opti_coords[2], 2),
    ]
    nv_sig["coords"] = r_opti_coords


def do_optimize_pixel(nv_sig):
    # Take an image and update the pixel coords from that image
    img_array = do_image_single_nv(nv_sig)
    pixel_coords = nv_sig["pixel_coords"]
    pixel_coords = optimize.optimize_pixel(
        img_array, pixel_coords, set_pixel_drift=False, set_scanning_drift=False
    )
    pixel_coords = [round(el, 2) for el in pixel_coords]
    print(pixel_coords)
    nv_sig["pixel_coords"] = pixel_coords


def do_optimize_pixel_set_drift(nv_sig):
    # Take an image and update the pixel coords from that image
    img_array = do_image_single_nv(nv_sig)
    pixel_coords = nv_sig["pixel_coords"]
    pixel_coords = optimize.optimize_pixel(
        img_array, pixel_coords, set_pixel_drift=True, set_scanning_drift=True
    )
    pixel_coords = [round(el, 2) for el in pixel_coords]
    print(pixel_coords)
    nv_sig["pixel_coords"] = pixel_coords


def do_optimize_widefield_calibration():
    with common.labrad_connect() as cxn:
        optimize.optimize_widefield_calibration(cxn)


def do_resonance(nv_list):
    freq_center = 2.87
    freq_range = 0.030
    num_steps = 20
    num_reps = 400
    num_runs = 64
    uwave_power = -16.0
    laser_filter = "nd_0.7"
    resonance.main(
        nv_list,
        freq_center,
        freq_range,
        num_steps,
        num_reps,
        num_runs,
        uwave_power,
        laser_filter=laser_filter,
    )


# def do_stationary_count(nv_sig, disable_opt=True):
#     nv_sig["imaging_readout_dur"] *= 10
#     run_time = 3 * 60 * 10**9  # ns
#     stationary_count.main(nv_sig, run_time, disable_opt=disable_opt)


# def do_pulsed_resonance(nv_sig, freq_center=2.87, freq_range=0.2):
#     num_steps = 51

#     # num_reps = 2e4
#     # num_runs = 16

#     num_reps = 1e2
#     num_runs = 32

#     uwave_power = 4
#     uwave_pulse_dur = 100

#     pulsed_resonance.main(
#         nv_sig,
#         freq_center,
#         freq_range,
#         num_steps,
#         num_reps,
#         num_runs,
#         uwave_power,
#         uwave_pulse_dur,
#     )


# def do_rabi(nv_sig, state, uwave_time_range=[0, 300]):
#     num_steps = 51

#     # num_reps = 2e4
#     # num_runs = 16

#     num_reps = 1e2
#     num_runs = 16

#     period = rabi.main(nv_sig, uwave_time_range, state, num_steps, num_reps, num_runs)
#     nv_sig["rabi_{}".format(state.name)] = period


### Run the file


if __name__ == "__main__":
    ### Shared parameters

    green_laser = "laser_INTE_520"
    yellow_laser = "laser_LGLO_589"
    red_laser = "laser_COBO_638"

    sample_name = "johnson"
    z_coord = 5.78
    # ref_coords = [0.0, 0.0, z_coord]
    ref_coords = [0.0, 0.0, z_coord]
    ref_coords = np.array(ref_coords)

    # ref_pixel_coords = [316.7, 238.8]
    # # ref_pixel_coords = [306.79, 310.572]
    # # ref_pixel_coords = [123.251, 198.218]
    # ref_scanning_coords = widefield.pixel_to_scanning(ref_pixel_coords)
    # ref_coords = np.array([*ref_scanning_coords, z_coord])
    # # print(ref_coords)

    nv_ref = {
        "coords": ref_coords,
        "name": f"{sample_name}-nvref",
        "disable_opt": False,
        "disable_z_opt": True,
        "expected_count_rate": None,
        #
        LaserKey.IMAGING: {
            "name": green_laser,
            "readout_dur": 1e7,
            "num_reps": 100,
            "filter": "nd_0",
        },
        #
        LaserKey.SPIN: {"name": green_laser, "pol_dur": 2e3, "readout_dur": 440},
        #
        "collection": {"filter": "514_notch+630_lp"},
        "magnet_angle": None,
        #
        NVSpinState.LOW: {"freq": 2.885, "rabi": 150, "uwave_power": 10.0},
    }

    # Experiment NVs

    nv0 = copy.deepcopy(nv_ref)
    nv0["name"] = f"{sample_name}-nv0_2023_09_11"
    nv0["pixel_coords"] = [183.66, 201.62]
    nv0["coords"] = [-0.039, 0.157, z_coord]

    nv1 = copy.deepcopy(nv_ref)
    nv1["name"] = f"{sample_name}-nv1_2023_09_11"
    nv1["pixel_coords"] = [177.28, 233.34]
    nv1["coords"] = [-0.047, 0.108, z_coord]

    nv2 = copy.deepcopy(nv_ref)
    nv2["name"] = f"{sample_name}-nv2_2023_09_11"
    nv2["pixel_coords"] = [237.42, 314.84]
    nv2["coords"] = [0.05, -0.013, z_coord]

    nv3 = copy.deepcopy(nv_ref)
    nv3["name"] = f"{sample_name}-nv3_2023_09_11"
    nv3["pixel_coords"] = [239.56, 262.84]
    nv3["coords"] = [0.049, 0.066, z_coord]

    nv4 = copy.deepcopy(nv_ref)
    nv4["name"] = f"{sample_name}-nv4_2023_09_12"
    nv4["pixel_coords"] = [315.58, 203.56]
    nv4["coords"] = [0.162, 0.161, z_coord]

    # Calibration NVs

    nv5, nv6 = widefield.get_widefield_calibration_nvs()

    nv5 = copy.deepcopy(nv_ref)
    nv5["name"] = f"{sample_name}-cal_check1"
    nv5["pixel_coords"] = [139.5840657600651, 257.70994378810946]
    nv5["coords"] = [-0.10233730341013306, 0.0699854266603339, z_coord]
    nv5["disable_z_opt"] = False

    nv6 = copy.deepcopy(nv_ref)
    nv6["name"] = f"{sample_name}-cal_check2"
    nv6["pixel_coords"] = [324.4796398557366, 218.27466265286117]
    nv6["coords"] = [0.1770304266201686, 0.1391538157833249, z_coord]

    nv_list = [nv0, nv1, nv2, nv3, nv4]
    # nv_list = [nv0, nv1, nv3, nv4]
    # nv_list = [nv1, nv2]
    # nv_list = [nv5, nv6]
    # nv_list = [nv4]

    nv_sig = nv2
    # nv_sig = nv_ref

    ### Functions to run

    email_recipient = "cambria@wisc.edu"
    do_email = False
    try:
        # pass

        # tb.init_safe_stop()

        # pos.reset_xy_drift()
        # widefield.reset_pixel_drift()

        with common.labrad_connect() as cxn:
            camera = tb.get_server_camera(cxn)
            readout_mode = camera.get_readout_mode()
            print(readout_mode)

        # Optimize pixels coords batch
        # raw_data = tb.get_raw_data("2023_09_11-13_52_01-johnson-nvref")
        # img_array = np.array(raw_data["img_array"])
        # for nv in nv_list:
        #     # pixel_coords = optimize.optimize_pixel(
        #     #     img_array,
        #     #     nv["pixel_coords"],
        #     #     set_pixel_drift=False,
        #     #     set_scanning_drift=False,
        #     # )
        #     # pixel_coords = [round(el, 2) for el in pixel_coords]
        #     # print(pixel_coords)
        #     pixel_coords = nv["pixel_coords"]
        #     scanning_coords = widefield.pixel_to_scanning_coords(pixel_coords)
        #     scanning_coords = [round(el, 3) for el in scanning_coords]
        #     print(scanning_coords)

        # for nv in nv_list:
        #     do_optimize_pixel(nv)
        #     do_optimize_plot(nv)
        # for nv in nv_list:
        #     print(nv["pixel_coords"])
        #     print(nv["coords"])

        # do_image_sample(nv_ref)
        # do_image_sample_zoom(nv_sig)
        # do_image_nv_list(nv_list)
        # do_image_single_nv(nv_sig)
        # for nv in nv_list:
        #     do_image_single_nv(nv)
        # do_stationary_count(nv_sig)
        # do_resonance(nv_list)
        # do_optimize(nv_sig)
        do_optimize_pixel(nv_sig)
        # do_optimize_pixel_set_drift(nv_sig)
        # do_optimize_plot(nv_sig)
        # do_optimize_widefield_calibration()
        # for nv in nv_list:
        #     do_optimize(nv)
        # do_pulsed_resonance(nv_sig, 2.87, 0.060)
        # do_rabi(nv_sig, States.LOW, uwave_time_range=[0, 300])

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
        plt.show(block=True)
        tb.reset_safe_stop()
