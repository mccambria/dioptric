# -*- coding: utf-8 -*-
"""
Control panel for the PC Rabi

Created on June 16th, 2023

@author: mccambria
"""


### Imports


import sys
import numpy as np
import matplotlib.pyplot as plt
import copy
from utils import tool_belt as tb
from utils import kplotlib as kpl
from utils import positioning as pos
from utils import widefield, common
from majorroutines.widefield import (
    charge_state_histograms,
    image_sample,
    optimize,
    resonance,
)
from utils.constants import LaserKey, NVSpinState
import time
from servers.inputs.nuvu_camera.nc_camera import NuvuException

green_laser = "laser_INTE_520"
red_laser = "laser_COBO_638"
yellow_laser = "laser_OPTO_589"
green_laser_dict = {"name": green_laser, "duration": 10e6}
red_laser_dict = {"name": red_laser, "duration": 5e6}
yellow_laser_dict = {"name": yellow_laser, "duration": 50e6}

### Major Routines


def do_widefield_image_sample(nv_sig, num_reps=1):
    nv_sig[LaserKey.IMAGING] = yellow_laser_dict
    image_sample.widefield(nv_sig, num_reps)


def do_scanning_image_sample(nv_sig):
    scan_range = 9
    num_steps = 60
    nv_sig[LaserKey.IMAGING] = green_laser_dict
    image_sample.scanning(nv_sig, scan_range, scan_range, num_steps)


def do_scanning_image_sample_zoom(nv_sig):
    scan_range = 0.2
    num_steps = 30
    nv_sig[LaserKey.IMAGING] = green_laser_dict
    image_sample.scanning(nv_sig, scan_range, scan_range, num_steps)


def do_image_nv_list(nv_list):
    return image_sample.nv_list(nv_list)


def do_image_single_nv(nv_sig):
    nv_sig[LaserKey.IMAGING] = green_laser_dict
    return image_sample.single_nv(nv_sig)


def do_image_single_nv_ionization(nv_sig, num_reps):
    return charge_state_histograms.single_nv_ionization(nv_sig, num_reps)


def do_image_single_nv_polarization(nv_sig, num_reps):
    # return image_sample.single_nv_polarization(nv_sig, num_reps)
    return charge_state_histograms.single_nv_polarization(nv_sig, num_reps)


def do_charge_state_histograms(nv_sig, num_reps):
    return charge_state_histograms.main(nv_sig, num_reps)


def do_optimize_green(nv_sig, set_drift=False, plot_data=True):
    prev_imaging_dict = nv_sig[LaserKey.IMAGING]
    nv_sig[LaserKey.IMAGING] = green_laser_dict
    coords_suffix = green_laser
    opti_coords, _ = optimize.main(
        nv_sig,
        save_data=plot_data,
        plot_data=plot_data,
        set_drift=set_drift,
        coords_suffix=coords_suffix,
    )
    # pos.set_nv_coords(nv_sig, opti_coords, coords_suffix)
    nv_sig[LaserKey.IMAGING] = prev_imaging_dict


def do_optimize_red(nv_sig, set_drift=False, plot_data=True):
    coords_suffix = red_laser
    opti_coords, _ = optimize.main(
        nv_sig,
        save_data=plot_data,
        plot_data=plot_data,
        set_drift=set_drift,
        laser_key=LaserKey.IONIZATION,
        coords_suffix=coords_suffix,
        no_crash=True,
    )
    # pos.set_nv_coords(nv_sig, opti_coords, coords_suffix)


def do_optimize_z(nv_sig, coords_suffix=None, set_drift=False, plot_data=False):
    opti_coords, _ = optimize.main(
        nv_sig,
        save_data=plot_data,
        plot_data=plot_data,
        set_drift=set_drift,
        coords_suffix=coords_suffix,
        set_pixel_drift=set_drift,
        no_crash=True,
    )
    nv_sig["coords"] = opti_coords


def do_optimize_pixel(nv_sig, set_drift=False):
    prev_imaging_dict = nv_sig[LaserKey.IMAGING]
    nv_sig[LaserKey.IMAGING] = green_laser_dict

    pixel_coords = optimize.optimize_pixel(
        nv_sig, set_scanning_drift=set_drift, set_pixel_drift=set_drift, plot_data=True
    )
    pixel_coords = [round(el, 2) for el in pixel_coords]

    # nv_sig["pixel_coords"] = pixel_coords
    nv_sig[LaserKey.IMAGING] = prev_imaging_dict


def do_optimize_widefield_calibration():
    with common.labrad_connect() as cxn:
        optimize.optimize_widefield_calibration(cxn)


def do_resonance(nv_list):
    freq_center = 2.87
    freq_range = 0.040
    num_steps = 20
    num_reps = 50
    num_runs = 4
    resonance.main(
        nv_list,
        freq_center,
        freq_range,
        num_steps,
        num_reps,
        num_runs,
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
        # opx.constant_ac([3])
        # Yellow
        # opx.constant_ac(
        #     [],  # Digital channels
        #     [7],  # Analog channels
        #     [0.225],  # Analog voltages
        #     [0],  # Analog frequencies
        # )
        # Green
        # opx.constant_ac(
        #     [4],  # Digital channels
        #     [6, 4],  # Analog channels
        #     [0.19, 0.19],  # Analog voltages
        #     [110, 110],  # Analog frequencies
        # )
        # Red
        opx.constant_ac(
            [1],  # Digital channels
            [2, 3],  # Analog channels
            [0.41, 0.41],  # Analog voltages
            [75, 75],  # Analog frequencies
        )
        # Red + green
        # opx.constant_ac(
        #     [1, 4],  # Digital channels
        #     [2, 3, 6, 4],  # Analog channels
        #     [0.32, 0.32, 0.19, 0.19],  # Analog voltages
        #     # [73.8, 76.2, 110.011, 110.845],  # Analog frequencies
        #     [72.6, 77.1, 108.3, 112.002],  # Analog frequencies
        #     # [75, 75, 110, 110],  # Analog frequencies
        # )
        # Red + green
        # opx.constant_ac(
        #     [1, 4],  # Digital channels
        #     [2, 3, 6, 4],  # Analog channels
        #     [0.32, 0.32, 0.19, 0.19],  # Analog voltages
        #     #
        #     # [73.8, 76.2, 110.011, 110.845],  # Analog frequencies
        #     # [72.6, 77.1, 108.3, 112.002],  # Analog frequencies
        #     #
        #     [73.8, 74.6, 110.011, 110.845],  # Analog frequencies
        #     # [72.6, 75.5, 108.3, 112.002],  # Analog frequencies
        #     #
        #     # [75, 75, 110, 110],  # Analog frequencies
        # )
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


# def make_histogram(nv_sig, num_reps=1):
#     charge_state_histogram(nv_sig, num_reps)


### Run the file


if __name__ == "__main__":
    ### Shared parameters

    green_coords_key = f"coords-{green_laser}"
    red_coords_key = f"coords-{red_laser}"
    pixel_coords_key = "pixel_coords"

    sample_name = "johnson"
    z_coord = 4.05
    # ref_coords = [110.900, 108.8, z_coord]
    ref_coords = [110.0, 110.0]
    ref_coords = np.array(ref_coords)

    nv_ref = {
        "coords": [None, None, z_coord],
        green_coords_key: ref_coords,
        red_coords_key: ref_coords,
        "name": f"{sample_name}-nvref",
        "disable_opt": False,
        "disable_z_opt": True,
        "expected_count_rate": None,
        #
        LaserKey.IMAGING: green_laser_dict,
        LaserKey.SPIN_READOUT: {"name": green_laser, "duration": 440},
        LaserKey.POLARIZATION: {"name": green_laser, "duration": 10e3},
        # LaserKey.IONIZATION: {"name": red_laser, "duration": 1e3},
        LaserKey.IONIZATION: {"name": red_laser, "duration": 150},
        LaserKey.CHARGE_READOUT: yellow_laser_dict,
        #
        "collection": {"filter": None},
        "magnet_angle": None,
        #
        NVSpinState.LOW: {"frequency": 2.87, "rabi_period": 100, "uwave_power": 12.0},
    }

    # region Experiment NVs

    nv0 = copy.deepcopy(nv_ref)
    nv0["name"] = f"{sample_name}-nv0_2023_11_09"
    nv0[pixel_coords_key] = [345.354, 260.217]
    nv0[green_coords_key] = [112.274, 109.94]
    nv0[red_coords_key] = [76.113, 75.136]
    # print(widefield.set_nv_scanning_coords_from_pixel_coords(nv0, green_laser))
    # print(widefield.set_nv_scanning_coords_from_pixel_coords(nv0, red_laser))
    # sys.exit()

    nv1 = copy.deepcopy(nv_ref)
    nv1["name"] = f"{sample_name}-nv1_2023_11_02"
    nv1[pixel_coords_key] = [217.197, 331.628]
    nv1[green_coords_key] = [108.3, 112.002]
    nv1[red_coords_key] = [75, 75]

    # endregion
    # Calibration NVs

    nv5, nv6 = widefield.get_widefield_calibration_nvs()

    nv_list = [nv0]
    # nv_list = [nv0, nv1, nv2, nv3, nv4]
    # nv_list = [nv0, nv1, nv2, nv3, nv4, nv5, nv6]

    nv_sig = nv0
    # nv_sig = nv1
    # nv_sig = nv_ref
    ### Clean up and save the data

    ### Functions to run

    email_recipient = "mccambria@berkeley.edu"
    do_email = False
    try:
        # pass

        # kpl.init_kplotlib()
        tb.init_safe_stop()

        # widefield.reset_all_drift()
        # widefield.reset_pixel_drift()
        # pos.reset_drift(green_laser)
        # pos.reset_drift(red_laser)

        # with common.labrad_connect() as cxn:
        #     pos.set_xyz_on_nv(cxn, nv_sig)

        # Convert pixel coords to scanning coords
        # pixel_coords = widefield.get_nv_pixel_coords(nv_sig)
        # for laser in [green_laser, red_laser]:
        #     scanning_coords = widefield.pixel_to_scanning_coords(pixel_coords, laser)
        #     print([round(el, 3) for el in scanning_coords])

        # do_opx_constant_ac()

        # # for z in np.linspace(3, 7, 21):
        # for z in np.linspace(4.0, 5.0, 11):
        #     nv_sig["coords"][2] = z
        #     do_widefield_image_sample(nv_sig, 10)
        # do_widefield_image_sample(nv_sig, 100)

        # do_scanning_image_sample(nv_sig)
        # do_scanning_image_sample_zoom(nv_sig)
        # do_image_nv_list(nv_list)
        # do_image_single_nv(nv_sig)

        # do_charge_state_histograms(nv_list, 1000)

        # do_optimize_pixel(nv_sig, set_drift=True)
        # do_optimize_red(nv_sig)
        # do_optimize_pixel(nv_sig, set_drift=False)
        # do_optimize_green(nv_sig)
        # do_optimize_z(nv_sig)
        # do_optimize_widefield_calibration()
        # for nv in nv_list:
        #     do_optimize(nv)

        do_resonance(nv_list)

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
        plt.show(block=True)
        tb.reset_safe_stop()
