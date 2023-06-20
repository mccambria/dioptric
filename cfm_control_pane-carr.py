# -*- coding: utf-8 -*-
"""
Control panel for the PC Carr

Created on June 16th, 2023

@author: mccambria
"""


### Imports


import numpy as np
import utils.tool_belt as tool_belt
import majorroutines.image_sample as image_sample
import majorroutines.optimize as optimize
import majorroutines.stationary_count as stationary_count
import majorroutines.pulsed_resonance as pulsed_resonance
import majorroutines.rabi as rabi
from majorroutines.widefield import qm_OPX_tests


### Major Routines


def do_image_sample(nv_sig):
    scan_range = 0.2
    num_steps = 60

    # scan_range = 1.0
    # num_steps = 180

    image_sample.main(
        nv_sig,
        scan_range,
        scan_range,
        num_steps,
    )


def do_image_sample_zoom(nv_sig):
    scan_range = 0.05
    num_steps = 30
    image_sample.main(nv_sig, scan_range, scan_range, num_steps)


def do_optimize(nv_sig):
    optimize.main(nv_sig, set_to_opti_coords=False, save_data=True, plot_data=True)


def do_stationary_count(nv_sig, disable_opt=None):
    run_time = 3 * 60 * 10**9  # ns
    stationary_count.main(nv_sig, run_time, disable_opt=disable_opt)


def do_pulsed_resonance(nv_sig, freq_center=2.87, freq_range=0.2):
    num_steps = 51

    # num_reps = 2e4
    # num_runs = 16

    num_reps = 1e2
    num_runs = 32

    uwave_power = 4
    uwave_pulse_dur = 100

    pulsed_resonance.main(
        nv_sig,
        freq_center,
        freq_range,
        num_steps,
        num_reps,
        num_runs,
        uwave_power,
        uwave_pulse_dur,
    )


def do_rabi(nv_sig, state, uwave_time_range=[0, 300]):
    num_steps = 51

    # num_reps = 2e4
    # num_runs = 16

    num_reps = 1e2
    num_runs = 16

    period = rabi.main(nv_sig, uwave_time_range, state, num_steps, num_reps, num_runs)
    nv_sig["rabi_{}".format(state.name)] = period


def do_qm_OPX_tests(nv_sig):
    qm_OPX_tests.main(nv_sig)


### Run the file


if __name__ == "__main__":
    ### Shared parameters

    green_laser = "laserglow_532"
    yellow_laser = "laserglow_589"
    red_laser = "cobolt_638"

    sample_name = "wu"
    ref_coords = [0.437, -0.295, 0]
    ref_coords = np.array(ref_coords)

    nv_sig = {
        "coords": ref_coords,
        "name": "{}-nvref_zfs_vs_t".format(sample_name),
        "disable_opt": False,
        "disable_z_opt": True,
        "expected_count_rate": 10,
        #
        "imaging_laser": green_laser,
        "imaging_laser_filter": "nd_0",
        "imaging_readout_dur": 1e7,
        #
        "spin_laser": green_laser,
        "spin_laser_filter": "nd_0",
        "spin_pol_dur": 2e3,
        "spin_readout_dur": 440,
        #
        "collection_filter": None,
        "magnet_angle": None,
        #
        "resonance_LOW": 2.885,
        "rabi_LOW": 150,
        "uwave_power_LOW": 10.0,
    }

    ### Functions to run

    email_recipient = "cambria@wisc.edu"
    try:
        # pass

        tool_belt.init_safe_stop()

        do_image_sample(nv_sig)
        # do_image_sample_zoom(nv_sig)
        # do_optimize(nv_sig)
        # do_pulsed_resonance(nv_sig, 2.87, 0.060)
        # do_rabi(nv_sig, States.LOW, uwave_time_range=[0, 300])

    except Exception as exc:
        recipient = email_recipient
        tool_belt.send_exception_email(email_to=recipient)
        raise exc

    finally:
        msg = "Experiment complete!"
        recipient = email_recipient
        tool_belt.send_email(msg, email_to=recipient)

        # Make sure everything is reset
        tool_belt.reset_cfm()
        tool_belt.reset_safe_stop()
