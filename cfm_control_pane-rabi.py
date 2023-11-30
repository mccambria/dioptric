# -*- coding: utf-8 -*-
"""
Control panel for the PC Rabi

Created on June 16th, 2023

@author: mccambria
"""


### Imports


import sys
import time
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
    rabi,
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


def do_optimize_green(nv_sig, do_plot=True):
    prev_imaging_dict = nv_sig[LaserKey.IMAGING]
    nv_sig[LaserKey.IMAGING] = green_laser_dict
    coords_suffix = green_laser
    ret_vals = optimize.main(
        nv_sig, coords_suffix=coords_suffix, no_crash=True, do_plot=do_plot
    )
    opti_coords = ret_vals[0]
    nv_sig[LaserKey.IMAGING] = prev_imaging_dict
    return opti_coords


def do_optimize_red(nv_sig, do_plot=True):
    laser_key = LaserKey.IONIZATION
    nv_sig[laser_key]["duration"] = 1e3
    coords_suffix = red_laser
    ret_vals = optimize.main(
        nv_sig,
        laser_key=laser_key,
        coords_suffix=coords_suffix,
        no_crash=True,
        do_plot=do_plot,
    )
    opti_coords = ret_vals[0]
    return opti_coords


def do_optimize_z(nv_sig, do_plot=False):
    optimize.main(nv_sig, no_crash=True, do_plot=do_plot)


def do_optimize_pixel(nv_sig):
    prev_imaging_dict = nv_sig[LaserKey.IMAGING]
    nv_sig[LaserKey.IMAGING] = green_laser_dict
    opti_coords = optimize.optimize_pixel(nv_sig, do_plot=True)
    nv_sig[LaserKey.IMAGING] = prev_imaging_dict
    return opti_coords


def do_optimize_widefield_calibration():
    with common.labrad_connect() as cxn:
        optimize.optimize_widefield_calibration(cxn)


def do_resonance(nv_list):
    freq_center = 2.87
    freq_range = 0.200
    num_steps = 60
    num_reps = 15
    num_runs = 32
    resonance.main(nv_list, freq_center, freq_range, num_steps, num_reps, num_runs)


def do_resonance_zoom(nv_list):
    freq_center = 2.80
    freq_range = 0.050
    num_steps = 20
    num_reps = 50
    num_runs = 16
    resonance.main(nv_list, freq_center, freq_range, num_steps, num_reps, num_runs)


def do_rabi(nv_list):
    uwave_freq = 2.793
    max_tau = 120
    num_steps = 16
    num_reps = 50
    num_runs = 16
    rabi.main(nv_list, uwave_freq, max_tau, num_steps, num_reps, num_runs)


def do_opx_constant_ac():
    with common.labrad_connect() as cxn:
        opx = cxn.QM_opx

        # Microwave test
        # sig_gen = cxn.sig_gen_STAN_sg394_2
        # sig_gen.set_freq(2.87)
        # sig_gen.set_amp(15.5) # 12
        # sig_gen.uwave_on()
        # opx.constant_ac([3])

        # Yellow
        opx.constant_ac(
            [],  # Digital channels
            [7],  # Analog channels
            [0.25],  # Analog voltages
            [0],  # Analog frequencies
        )
        # Green
        # opx.constant_ac(
        #     [4],  # Digital channels
        #     [6, 4],  # Analog channels
        #     [0.19, 0.19],  # Analog voltages
        #     [110, 110],  # Analog frequencies
        # )
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
        # sig_gen.uwave_off()


### Run the file


if __name__ == "__main__":
    ### Shared parameters

    green_coords_key = f"coords-{green_laser}"
    red_coords_key = f"coords-{red_laser}"
    pixel_coords_key = "pixel_coords"

    sample_name = "johnson"
    z_coord = 3.36
    magnet_angle = 0

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
        "magnet_angle": None,
        #
        NVSpinState.LOW: {"frequency": 2.87, "rabi_period": 48, "uwave_power": 12.0},
        NVSpinState.HIGH: {"frequency": 2.87, "rabi_period": 48, "uwave_power": 15.5},
    }

    nv0 = copy.deepcopy(nv_ref)
    nv0["name"] = f"{sample_name}-nv0_2023_11_29"
    nv0[pixel_coords_key] = [352.467, 253.072]
    nv0[green_coords_key] = [112.566, 109.654]
    nv0[red_coords_key] = [76.383, 74.961]

    nv1 = copy.deepcopy(nv_ref)
    nv1["name"] = f"{sample_name}-nv1_2023_11_29"
    nv1[pixel_coords_key] = [350.05, 263.677]
    nv1[green_coords_key] = [112.388, 110.097]
    nv1[red_coords_key] = [76.196, 75.127]

    nv2 = copy.deepcopy(nv_ref)
    nv2["name"] = f"{sample_name}-nv2_2023_11_29"
    nv2[pixel_coords_key] = [366.264, 267.229]
    nv2[green_coords_key] = [113.059, 110.195]
    nv2[red_coords_key] = [76.626, 75.258]

    nv3 = copy.deepcopy(nv_ref)
    nv3["name"] = f"{sample_name}-nv3_2023_11_29"
    nv3[pixel_coords_key] = [369.989, 279.857]
    nv3[green_coords_key] = [113.103, 110.668]
    nv3[red_coords_key] = [76.704, 75.585]

    nv4 = copy.deepcopy(nv_ref)
    nv4["name"] = f"{sample_name}-nv4_2023_11_29"
    nv4[pixel_coords_key] = [366.006, 242.578]
    nv4[green_coords_key] = [113.135, 109.506]
    nv4[red_coords_key] = [76.714, 74.622]

    nv5 = copy.deepcopy(nv_ref)
    nv5["name"] = f"{sample_name}-nv5_2023_11_29"
    nv5[pixel_coords_key] = [366.689, 241.889]
    nv5[green_coords_key] = [112.977, 109.386]
    nv5[red_coords_key] = [76.678, 74.595]

    nv6 = copy.deepcopy(nv_ref)
    nv6["name"] = f"{sample_name}-nv6_2023_11_29"
    nv6[pixel_coords_key] = [372.844, 229.129]
    nv6[green_coords_key] = [113.201, 108.928]
    nv6[red_coords_key] = [77.028, 74.372]

    nv7 = copy.deepcopy(nv_ref)
    nv7["name"] = f"{sample_name}-nv7_2023_11_29"
    nv7[pixel_coords_key] = [363.392, 224.612]
    nv7[green_coords_key] = [113.151, 108.921]
    nv7[red_coords_key] = [76.704, 74.184]

    nv8 = copy.deepcopy(nv_ref)
    nv8["name"] = f"{sample_name}-nv8_2023_11_29"
    nv8[pixel_coords_key] = [346.448, 224.743]
    nv8[green_coords_key] = [112.436, 108.883]
    nv8[red_coords_key] = [76.182, 74.214]

    # nv_sig = nv0
    # nv_sig = nv1
    # nv_sig = nv5
    # nv_sig = nv_ref
    # nv_list = [nv_sig]
    # nv_list = [nv8, nv9]
    nv_list = [nv8, nv0, nv1, nv2, nv3, nv4, nv5, nv6, nv7]
    # nv_list = [nv2, nv3, nv4, nv5]
    nv_sig = nv_list[0]

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
            #     mag_rot_server.set_angle(magnet_angle)
            print(mag_rot_server.get_angle())

        # kpl.init_kplotlib()
        tb.init_safe_stop()

        # widefield.reset_all_drift()
        # widefield.reset_pixel_drift()
        # pos.reset_drift(green_laser)
        # pos.reset_drift(red_laser)
        # widefield.set_pixel_drift([+11, +6])
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
        # for z in np.linspace(3.25, 3.75, 11):
        #     nv_sig["coords"][2] = z
        #     do_widefield_image_sample(nv_sig, 100)
        # for ind in range(20):
        #     time.sleep(5)
        #     do_widefield_image_sample(nv_sig, 100)
        # do_widefield_image_sample(nv_sig, 100)

        # do_scanning_image_sample(nv_sig)
        # do_scanning_image_sample_zoom(nv_sig)
        # do_image_nv_list(nv_list)
        # do_image_single_nv(nv_sig)

        # do_optimize_pixel(nv_sig)
        # do_charge_state_histograms(nv_list, 1000)

        # opti_coords_list = []
        # for nv in nv_list:
        #     #
        #     opti_coords = do_optimize_pixel(nv)
        #     #
        #     # widefield.set_nv_scanning_coords_from_pixel_coords(nv, green_laser)
        #     # opti_coords = do_optimize_green(nv)
        #     #
        #     # widefield.set_nv_scanning_coords_from_pixel_coords(nv, red_laser)
        #     # opti_coords = do_optimize_red(nv)
        #     #
        #     opti_coords_list.append(opti_coords)
        #     widefield.reset_all_drift()
        #     #
        # for opti_coords in opti_coords_list:
        #     r_opti_coords = [round(el, 3) for el in opti_coords]
        #     print(r_opti_coords)

        # do_optimize_z(nv_sig)
        # do_optimize_widefield_calibration()
        # for nv in nv_list:
        #     do_optimize(nv)

        # do_resonance(nv_list)
        # do_resonance_zoom(nv_list)
        do_rabi(nv_list)

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



"""
