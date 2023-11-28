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

green_laser = "laser_INTE_520"
red_laser = "laser_COBO_638"
yellow_laser = "laser_OPTO_589"
green_laser_dict = {"name": green_laser, "duration": 10e6}
red_laser_dict = {"name": red_laser, "duration": 10e6}
yellow_laser_dict = {"name": yellow_laser, "duration": 35e6}

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


def do_charge_state_histograms(nv_list, num_reps):
    for nv in nv_list:
        nv[LaserKey.IONIZATION]["duration"] = 1e3
    return charge_state_histograms.main(nv_list, num_reps)


def do_optimize_green(nv_sig, set_drift=False, do_plot=True):
    prev_imaging_dict = nv_sig[LaserKey.IMAGING]
    nv_sig[LaserKey.IMAGING] = green_laser_dict
    coords_suffix = green_laser
    optimize.main(nv_sig, coords_suffix=coords_suffix, no_crash=True, do_plot=do_plot)
    nv_sig[LaserKey.IMAGING] = prev_imaging_dict
    if not set_drift:
        pos.reset_drift(coords_suffix)


def do_optimize_red(nv_sig, set_drift=False, do_plot=True):
    laser_key = LaserKey.IONIZATION
    nv_sig[laser_key]["duration"] = 1e3
    coords_suffix = red_laser
    optimize.main(
        nv_sig,
        laser_key=laser_key,
        coords_suffix=coords_suffix,
        no_crash=True,
        do_plot=do_plot,
    )
    if not set_drift:
        pos.reset_drift(coords_suffix)


def do_optimize_z(nv_sig, do_plot=False):
    optimize.main(nv_sig, no_crash=True, do_plot=do_plot)


def do_optimize_pixel(nv_sig):
    prev_imaging_dict = nv_sig[LaserKey.IMAGING]
    nv_sig[LaserKey.IMAGING] = green_laser_dict
    optimize.optimize_pixel(nv_sig, do_plot=True)
    nv_sig[LaserKey.IMAGING] = prev_imaging_dict


def do_optimize_widefield_calibration():
    with common.labrad_connect() as cxn:
        optimize.optimize_widefield_calibration(cxn)


def do_resonance(nv_list):
    freq_center = 2.87
    freq_range = 0.050
    num_steps = 20
    num_reps = 50
    num_runs = 8
    resonance.main(nv_list, freq_center, freq_range, num_steps, num_reps, num_runs)


def do_opx_constant_ac():
    with common.labrad_connect() as cxn:
        opx = cxn.QM_opx
        # opx.constant_ac([3])
        # Yellow
        # opx.constant_ac(
        #     [],  # Digital channels
        #     [7],  # Analog channels
        #     [0.25],  # Analog voltages
        #     [0],  # Analog frequencies
        # )
        # Green
        opx.constant_ac(
            [4],  # Digital channels
            [6, 4],  # Analog channels
            [0.19, 0.19],  # Analog voltages
            [110, 110],  # Analog frequencies
        )
        # Red
        # opx.constant_ac(
        #     [1],  # Digital channels
        #     [2, 3],  # Analog channels
        #     [0.41, 0.41],  # Analog voltages
        #     [75, 75],  # Analog frequencies
        # )
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


### Run the file


if __name__ == "__main__":
    ### Shared parameters

    green_coords_key = f"coords-{green_laser}"
    red_coords_key = f"coords-{red_laser}"
    pixel_coords_key = "pixel_coords"

    sample_name = "johnson"
    z_coord = 3.47
    magnet_angle = 30

    nv_ref = {
        "coords": [None, None, z_coord],
        green_coords_key: np.array([110, 110]),
        red_coords_key: np.array([75, 75]),
        "name": f"{sample_name}-nvref",
        "disable_opt": False,
        "disable_z_opt": True,
        "expected_count_rate": None,
        #
        LaserKey.IMAGING: green_laser_dict,
        LaserKey.SPIN_READOUT: {"name": green_laser, "duration": 440},
        LaserKey.POLARIZATION: {"name": green_laser, "duration": 10e3},
        LaserKey.IONIZATION: {"name": red_laser, "duration": 200},
        LaserKey.CHARGE_READOUT: yellow_laser_dict,
        #
        "collection": {"filter": None},
        "magnet_angle": magnet_angle,
        #
        NVSpinState.LOW: {"frequency": 2.87, "rabi_period": 80, "uwave_power": 12.0},
    }

    nv0 = copy.deepcopy(nv_ref)
    nv0["name"] = f"{sample_name}-nv0_2023_11_25"
    nv0[pixel_coords_key] = [331.591, 281.997]
    nv0[green_coords_key] = [111.754, 110.772]
    nv0[red_coords_key] = [75.671, 75.774]

    nv1 = copy.deepcopy(nv_ref)
    nv1["name"] = f"{sample_name}-nv1_2023_11_25"
    nv1[pixel_coords_key] = [347.948, 248.368]
    nv1[green_coords_key] = [112.347, 109.556]
    nv1[red_coords_key] = [76.059, 74.738]

    nv2 = copy.deepcopy(nv_ref)
    nv2["name"] = f"{sample_name}-nv2_2023_11_25"
    nv2[pixel_coords_key] = [371.143, 242.199]
    nv2[green_coords_key] = [113.14139162658678, 109.19081375337272]
    nv2[red_coords_key] = [76.73, 74.587]

    nv3 = copy.deepcopy(nv_ref)
    nv3["name"] = f"{sample_name}-nv3_2023_11_25"
    nv3[pixel_coords_key] = [369.707, 305.252]
    nv3[green_coords_key] = [113.18, 109.278]
    nv3[red_coords_key] = [76.784, 76.326]

    nv4 = copy.deepcopy(nv_ref)
    nv4["name"] = f"{sample_name}-nv4_2023_11_25"
    nv4[pixel_coords_key] = [345.208, 312.324]
    nv4[green_coords_key] = [112.398, 111.685]
    nv4[red_coords_key] = [76.036, 76.61]

    nv5 = copy.deepcopy(nv_ref)
    nv5["name"] = f"{sample_name}-nv5_2023_11_25"
    nv5[pixel_coords_key] = [316.119, 299.436]
    nv5[green_coords_key] = [111.185, 111.135]
    nv5[red_coords_key] = [75.231, 76.262]

    nv6 = copy.deepcopy(nv_ref)
    nv6["name"] = f"{sample_name}-nv6_2023_11_25"
    nv6[pixel_coords_key] = [308.186, 227.034]
    nv6[green_coords_key] = [110.974, 108.824]
    nv6[red_coords_key] = [75.018, 74.221]

    nv7 = copy.deepcopy(nv_ref)
    nv7["name"] = f"{sample_name}-nv7_2023_11_25"
    nv7[pixel_coords_key] = [334.802, 218.992]
    nv7[green_coords_key] = [111.95, 108.658]
    nv7[red_coords_key] = [75.776, 74.076]

    nv8 = copy.deepcopy(nv_ref)
    nv8["name"] = f"{sample_name}-nv8_2023_11_25"
    nv8[pixel_coords_key] = [323.756, 304.543]
    nv8[green_coords_key] = [111.388, 111.469]
    nv8[red_coords_key] = [75.346, 76.223]

    nv9 = copy.deepcopy(nv_ref)
    nv9["name"] = f"{sample_name}-nv9_2023_11_25"
    nv9[pixel_coords_key] = [299.588, 253.558]
    nv9[green_coords_key] = [110.672, 109.681]
    nv9[red_coords_key] = [74.77, 74.88]

    nv_sig = nv0
    # nv_sig = nv5
    # nv_sig = nv_ref
    # nv_list = [nv_sig]
    # nv_list = [nv8, nv9]
    nv_list = [nv0, nv1, nv2, nv3, nv4, nv6, nv7, nv8, nv9]
    # nv_list = [nv0, nv1, nv4, nv6, nv7, nv9]
    # nv_list = [nv3, nv4, nv5, nv6, nv7]

    # for nv in nv_list:
    #     # widefield.set_nv_scanning_coords_from_pixel_coords(nv, green_laser)
    #     widefield.set_nv_scanning_coords_from_pixel_coords(nv, red_laser)
    # sys.exit()

    ### Clean up and save the data

    ### Functions to run

    email_recipient = "mccambria@berkeley.edu"
    do_email = False
    try:
        # pass

        with common.labrad_connect() as cxn:
            mag_rot_server = tb.get_server_magnet_rotation(cxn)
            mag_rot_server.set_angle(magnet_angle)
            # print(mag_rot_server.get_angle())

        # kpl.init_kplotlib()
        tb.init_safe_stop()

        # widefield.reset_all_drift()
        # widefield.reset_pixel_drift()
        # pos.reset_drift(green_laser)
        # pos.reset_drift(red_laser)
        # widefield.set_pixel_drift([-1.8, -4])
        # widefield.set_all_scanning_drift_from_pixel_drift()

        # with common.labrad_connect() as cxn:
        #     pos.set_xyz_on_nv(cxn, nv_sig)

        # Get updated coords before drift reset
        # for nv in nv_list:
        #     print(widefield.get_nv_pixel_coords(nv))
        #     print(pos.get_nv_coords(nv, green_laser))
        #     print(pos.get_nv_coords(nv, red_laser))
        # print()

        # Convert pixel coords to scanning coords
        # for nv in nv_list:
        #     pixel_coords = widefield.get_nv_pixel_coords(nv)
        #     print(widefield.pixel_to_scanning_coords(pixel_coords, green_laser))
        #     print(widefield.pixel_to_scanning_coords(pixel_coords, red_laser))
        #     # print()

        # do_opx_constant_ac()

        # # for z in np.linspace(3, 7, 21):
        # for z in np.linspace(3.25, 3.5, 6):
        #     nv_sig["coords"][2] = z
        #     do_widefield_image_sample(nv_sig, 100)
        do_widefield_image_sample(nv_sig, 100)

        # do_scanning_image_sample(nv_sig)
        # do_scanning_image_sample_zoom(nv_sig)
        # do_image_nv_list(nv_list)
        # do_image_single_nv(nv_sig)

        # do_optimize_pixel(nv_sig)
        # do_charge_state_histograms(nv_list, 1000)

        # for nv in nv_list:
        # #     do_optimize_pixel(nv)
        #     # do_optimize_green(nv)
        #     do_optimize_red(nv)
        #     widefield.reset_all_drift()
        # do_optimize_z(nv_sig)
        # do_optimize_widefield_calibration()
        # for nv in nv_list:
        #     do_optimize(nv)

        # do_resonance(nv_list)

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

"""

Normalized separation:
2.348 / sqrt(shot)
10.501 / sqrt(s)
Normalized separation:
2.196 / sqrt(shot)
9.82 / sqrt(s)
Normalized separation:
2.431 / sqrt(shot)
10.872 / sqrt(s)
Normalized separation:
1.792 / sqrt(shot)
8.015 / sqrt(s)
Normalized separation:
2.651 / sqrt(shot)
11.858 / sqrt(s)
Normalized separation:
2.335 / sqrt(shot)
10.441 / sqrt(s)
Normalized separation:
2.545 / sqrt(shot)
11.382 / sqrt(s)
Normalized separation:
2.8 / sqrt(shot)
12.523 / sqrt(s)
Normalized separation:
2.498 / sqrt(shot)
11.171 / sqrt(s)

"""
