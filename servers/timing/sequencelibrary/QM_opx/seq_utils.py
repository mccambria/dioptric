# -*- coding: utf-8 -*-
"""
QM OPX sequence utils

Created June 25th, 2023

@author: mccambria
"""


from qm import qua
from utils import common
from utils.constants import ModMode, CollectionMode


def handle_reps(
    one_rep_macro,
    num_reps,
    wait_for_trigger=None,
    post_trigger_macro=None,
):
    """Handle repetitions of a given sequence - you just have to pass
    a function defining the behavior for a single loop. Optionally
    waits for trigger pulse between loops.

    Parameters
    ----------
    one_rep_macro : QUA macro
        QUA "macro" to be repeated
    num_reps : int
        Number of times to repeat, -1 for infinite loop
    wait_for_trigger : bool, optional
        Whether or not to pause execution between loops until a trigger
        pulse is received by the OPX, defaults to True for camera, False otherwise
    post_trigger_macro : QUA macro
        Macro to be played after receiving a trigger
    """

    if wait_for_trigger is None:
        config = common.get_config_dict()
        collection_mode = config["collection_mode"]
        wait_for_trigger = collection_mode == CollectionMode.CAMERA

    dummy_element = "do1"  # wait_for_trigger requires us to pass some element

    if num_reps == -1:
        with qua.infinite_loop_():
            one_rep_macro()
            if wait_for_trigger:
                qua.wait_for_trigger(dummy_element)
                qua.align()
                post_trigger_macro()
    elif num_reps == 1:
        one_rep_macro()
    else:
        handle_reps_ind = qua.declare(int, value=0)
        with qua.while_(handle_reps_ind < num_reps):
            one_rep_macro()
            qua.assign(handle_reps_ind, handle_reps_ind + 1)
            if wait_for_trigger:
                qua.wait_for_trigger(dummy_element)
                qua.align()
                post_trigger_macro()


def get_camera_pad_duration(rep_duration_cc):
    """The camera can't return images arbitrarily fast and it has no way
    to limit itself - you have to limit it manually or else it will disconnect.
    Use this QUA macro to pad your sequence so it respects the camera's
    maximum frame rate

    rep_duration_cc : int
        Duration of a sequence rep in clock cycles - doesn't need to be exact
    """
    config = common.get_config_dict()
    max_frame_rate = config["Camera"]["max_frame_rate"]
    min_sequence_time = 1 / max_frame_rate
    min_sequence_time_cc = round(min_sequence_time * 1e9 / 4)
    pad_duration_cc = min_sequence_time_cc - rep_duration_cc
    return pad_duration_cc


def convert_ns_to_clock_cycles(duration_ns):
    return round(duration_ns / 4)


def get_laser_mod_element(laser_name):
    config = common.get_config_dict()
    mod_mode = config["Optics"][laser_name]["mod_mode"]
    if mod_mode == ModMode.ANALOG:
        laser_mod_element = f"ao_{laser_name}_am"
    elif mod_mode == ModMode.DIGITAL:
        laser_mod_element = f"do_{laser_name}_dm"
    return laser_mod_element
