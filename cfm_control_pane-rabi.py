# -*- coding: utf-8 -*-
"""
Control panel for the PC Rabi

Created on June 16th, 2023

@author: mccambria
"""


### Imports


import numpy as np
from utils import tool_belt as tb
from utils import kplotlib as kpl
from utils import positioning as pos
from utils import widefield
from utils import common
from utils.constants import LaserKey, NVSpinState, LaserPosStyle
from majorroutines.widefield import image_sample
from majorroutines.widefield import resonance
from majorroutines import optimize
import matplotlib.pyplot as plt
import copy


### Major Routines


def do_widefield_image_sample(nv_sig):
    image_sample.widefield(nv_sig)


def do_scanning_image_sample(nv_sig):
    scan_range = 9
    num_steps = 60
    image_sample.scanning(nv_sig, scan_range, scan_range, num_steps)


def do_scanning_image_sample_zoom(nv_sig):
    scan_range = 0.2
    num_steps = 30
    image_sample.scanning(nv_sig, scan_range, scan_range, num_steps)


def do_image_nv_list(nv_list):
    return image_sample.nv_list(nv_list)


def do_image_single_nv(nv_sig):
    return image_sample.single_nv(nv_list)


def do_optimize(nv_sig, set_drift=False, plot_data=False):
    opti_coords, _ = optimize.main(
        nv_sig,
        set_to_opti_coords=False,
        save_data=plot_data,
        plot_data=plot_data,
        set_scanning_drift=set_drift,
        set_pixel_drift=set_drift,
    )
    r_opti_coords = [
        round(opti_coords[0], 3),
        round(opti_coords[1], 3),
        round(opti_coords[2], 2),
    ]
    nv_sig["coords"] = r_opti_coords


def do_optimize_pixel(nv_sig, set_pixel_drift=False, set_scanning_drift=False):
    pixel_coords = optimize.optimize_pixel(
        nv_sig,
        set_pixel_drift=set_pixel_drift,
        set_scanning_drift=set_scanning_drift,
        plot_data=True,
    )
    pixel_coords = [round(el, 2) for el in pixel_coords]
    print(pixel_coords)
    nv_sig["pixel_coords"] = pixel_coords


def do_optimize_widefield_calibration():
    with common.labrad_connect() as cxn:
        optimize.optimize_widefield_calibration(cxn)


def do_resonance(nv_list):
    freq_center = 2.87
    freq_range = 0.040
    num_steps = 30
    num_reps = 200
    num_runs = 17
    uwave_power = -23.0
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


def do_camera_test():
    with common.labrad_connect() as cxn:
        pulse_gen = tb.get_server_pulse_gen(cxn)
        camera = tb.get_server_camera(cxn)

        seq_file_name = "camera_test.py"
        pulse_gen.stream_load(seq_file_name, "")
        camera.arm()
        pulse_gen.stream_start(1)
        img_array = camera.read()
        camera.disarm()

    fig, ax = plt.subplots()
    kpl.imshow(ax, img_array)


def do_opx_constant_ac():
    with common.labrad_connect() as cxn:
        opx = cxn.QM_opx
        # opx.constant_ac([])
        opx.constant_ac(
            [4],  # Digital channels
            [4, 6],  # Analog channels
            [0.34, 0.34],  # Analog voltages
            [110e6, 110e6],  # Analog frequencies
        )
        input("Press enter to stop...")
        # opx.constant_ac()


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
    yellow_laser = "laser_OPTO_589"
    red_laser = "laser_COBO_638"

    # Imaging laser dicts
    yellow_laser_dict = {"name": yellow_laser, "readout_dur": 5e9, "num_reps": 1}
    green_laser_dict = {"name": green_laser, "readout_dur": 10e6, "num_reps": 1000}
    green_laser_dict = {"name": green_laser, "readout_dur": 10e6, "num_reps": 1000}

    sample_name = "johnson"
    z_coord = 4.0
    # ref_coords = [111.0, 113.0, z_coord]
    ref_coords = [110.0, 110.0, z_coord]
    ref_coords = np.array(ref_coords)

    nv_ref = {
        "coords": ref_coords,
        "name": f"{sample_name}-nvref",
        "disable_opt": False,
        "disable_z_opt": True,
        "expected_count_rate": None,
        #
        # LaserKey.IMAGING: yellow_laser_dict,
        LaserKey.IMAGING: green_laser_dict,
        #
        LaserKey.SPIN: {"name": green_laser, "pol_dur": 2e3, "readout_dur": 440},
        #
        "collection": {"filter": "514_notch+630_lp"},
        "magnet_angle": None,
        #
        NVSpinState.LOW: {"freq": 2.885, "rabi": 150, "uwave_power": 10.0},
    }

    # region Experiment NVs

    nv0 = copy.deepcopy(nv_ref)
    nv0["name"] = f"{sample_name}-nv0_2023_09_11"
    nv0["pixel_coords"] = [181.3, 227.9]
    nv0["coords"] = [-0.041, 0.115]

    nv1 = copy.deepcopy(nv_ref)
    nv1["name"] = f"{sample_name}-nv1_2023_09_11"
    nv1["pixel_coords"] = [187.28, 196.58]
    nv1["coords"] = [-0.034, 0.164]

    nv2 = copy.deepcopy(nv_ref)
    nv2["name"] = f"{sample_name}-nv2_2023_09_11"
    nv2["pixel_coords"] = [204.75, 202.81]
    nv2["coords"] = [-0.009, 0.154]

    nv3 = copy.deepcopy(nv_ref)
    nv3["name"] = f"{sample_name}-nv3_2023_09_11"
    nv3["pixel_coords"] = [296.65, 199.84]
    nv3["coords"] = [0.134, 0.165]

    nv4 = copy.deepcopy(nv_ref)
    nv4["name"] = f"{sample_name}-nv4_2023_09_12"
    nv4["pixel_coords"] = [248.11, 302.35]
    nv4["coords"] = [0.064, 0.001]

    # endregion
    # Calibration NVs

    nv5, nv6 = widefield.get_widefield_calibration_nvs()

    nv_list = [nv0, nv1, nv2, nv3, nv4]
    # nv_list = [nv0, nv1, nv2, nv3, nv4, nv5, nv6]
    for nv in nv_list:
        if len(nv["coords"]) < 3:
            nv["coords"].append(z_coord)
        else:
            nv["coords"][2] = z_coord

    # nv_sig = nv5
    nv_sig = nv_ref

    ### Functions to run

    email_recipient = "cambria@wisc.edu"
    do_email = False
    try:
        # pass

        kpl.init_kplotlib()
        # tb.init_safe_stop()

        # pos.reset_xy_drift()
        # pos.reset_drift()
        # widefield.reset_pixel_drift()
        # z_drift = pos.get_drift()[2]
        # pos.set_drift([+0.03, +0.03, z_drift])

        # Convert pixel coords to scanning coords
        # for nv in nv_list:
        #     scanning_coords = widefield.pixel_to_scanning_coords(nv["pixel_coords"])
        #     print([round(el, 3) for el in scanning_coords])
        # pixel_coords = [191.027, 284.665]
        # # pixel_coords = [142.25, 254.26]
        # scanning_coords = widefield.pixel_to_scanning_coords(pixel_coords)
        # print([round(el, 3) for el in scanning_coords])

        # for nv in nv_list:
        #     do_optimize_pixel(nv)
        #     do_optimize_plot(nv)
        # for nv in nv_list:
        #     print(nv["pixel_coords"])
        #     print(nv["coords"][0:2])

        # with common.labrad_connect() as cxn:
        #     pos.set_xyz(cxn, [0.0, 0.0, 5.0])
        # do_opx_constant_ac()

        # for z in np.linspace(3.0, 7.0, 21):
        #     nv_ref["coords"][2] = z
        #     do_image_sample(nv_ref)

        # nv_ref[LaserKey.IMAGING] = yellow_laser_dict
        # do_widefield_image_sample(nv_ref)
        # do_scanning_image_sample(nv_ref)
        # do_scanning_image_sample_zoom(nv_ref)
        # do_image_nv_list(nv_list)
        do_image_single_nv(nv_sig)
        # for nv in nv_list:
        #     do_image_single_nv(nv)
        # do_stationary_count(nv_sig)
        # do_resonance(nv_list)
        # do_optimize(nv_sig, set_drift=True, plot_data=True)
        # do_optimize_pixel(nv_sig)
        # do_optimize_pixel(nv_sig, set_pixel_drift=True, set_scanning_drift=True)
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
