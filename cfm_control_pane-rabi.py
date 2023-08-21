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
from majorroutines import image_sample
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


def do_image_nv_list(nv_list):
    image_sample.main(nv_list)


def do_image_sample_zoom(nv_sig):
    scan_range = 0.02
    num_steps = 60
    image_sample.main(nv_sig, scan_range, scan_range, num_steps)


def do_optimize(nv_sig):
    optimize.main(
        nv_sig,
        set_to_opti_coords=False,
        save_data=True,
        plot_data=True,
        set_drift=False,
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
    z_coord = 5.85
    # ref_coords = [0.0, 0.0, z_coord]
    ref_coords = [0.0, 0.0, z_coord]
    ref_coords = np.array(ref_coords)

    # ref_pixel_coords = [316.7, 238.8]
    # # ref_pixel_coords = [306.79, 310.572]
    # # ref_pixel_coords = [123.251, 198.218]
    # ref_scanning_coords = widefield.pixel_to_scanning(ref_pixel_coords)
    # ref_coords = np.array([*ref_scanning_coords, z_coord])
    # # print(ref_coords)

    # fmt: off
    nv_sig = {
        "coords": ref_coords, "name": f"{sample_name}-nvref",
        "disable_opt": False, "disable_z_opt": True, "expected_count_rate": None,
        #
        "imaging_laser": green_laser, "imaging_readout_dur": 1e7, # "imaging_laser_filter": "nd_0",
        #
        "spin_laser": green_laser, "spin_pol_dur": 2e3, "spin_readout_dur": 440, # "spin_laser_filter": "nd_0",
        #
        "collection_filter": "514_notch+630_lp", "magnet_angle": None,
        #
        "resonance_LOW": 2.885, "rabi_LOW": 150, "uwave_power_LOW": 10.0,
    }
    # fmt: on

    nv0 = copy.deepcopy(nv_sig)
    nv0["name"] = f"{sample_name}-nv0_2023_08_21"
    nv0["pixel_coords"] = [189.87, 267.62]
    nv0["coords"] = [-0.032, 0.171, z_coord]

    nv1 = copy.deepcopy(nv_sig)
    nv1["name"] = f"{sample_name}-nv1_2023_08_21"
    nv1["pixel_coords"] = [241.78, 194.4]
    nv1["coords"] = [0.05, 0.097, z_coord]

    nv2 = copy.deepcopy(nv_sig)
    nv2["name"] = f"{sample_name}-nv2_2023_08_21"
    nv2["pixel_coords"] = [296.26, 198.03]
    nv2["coords"] = [0.136, 0.019, z_coord]

    nv3 = copy.deepcopy(nv_sig)
    nv3["name"] = f"{sample_name}-nv3_2023_08_21"
    nv3["pixel_coords"] = [217.28, 275.39]
    nv3["coords"] = [0.011, 0.132, z_coord]

    nv_list = [nv0, nv1, nv2, nv3]

    ### Functions to run

    email_recipient = "cambria@wisc.edu"
    do_email = False
    try:
        # pass

        tb.init_safe_stop()

        # coords = nv_sig["coords"]
        # # for x in [-0.15, 0, 0.15]:
        # #     for y in [-0.15, 0, 0.15]:
        # for x in [coords[0] - 0.001, coords[0], coords[0] + 0.001]:
        #     for y in [coords[1] - 0.001, coords[1], coords[1] + 0.001]:
        #         nv_sig["coords"] = [x, y, z_coord]
        #         do_image_sample(nv_sig)

        # do_image_sample(nv_sig)
        do_image_nv_list(nv_list)
        # do_image_sample_zoom(nv_sig)
        # do_stationary_count(nv_sig)
        # do_optimize(nv_sig)
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
