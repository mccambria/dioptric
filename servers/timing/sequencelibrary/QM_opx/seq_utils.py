# -*- coding: utf-8 -*-
"""
QM OPX sequence utils. Should only be used by sequence files

Created June 25th, 2023

@author: mccambria
"""


from qm import qua
from utils import common
from utils.constants import ModMode, CollectionMode


# region QUA macros


def handle_reps(
    one_rep_macro,
    num_reps,
    wait_for_trigger=None,
):
    """Handle repetitions of a given sequence - you just have to pass
    a function defining the behavior for a single loop. Optionally
    waits for trigger pulse between loops.

    Parameters
    ----------
    one_rep_macro : QUA macro
        QUA macro to be repeated
    num_reps : int
        Number of times to repeat, -1 for infinite loop
    wait_for_trigger : bool, optional
        Whether or not to pause execution between loops until a trigger
        pulse is received by the OPX, defaults to True for camera, False otherwise
    """

    if wait_for_trigger is None:
        config = common.get_config_dict()
        collection_mode = config["collection_mode"]
        wait_for_trigger = collection_mode == CollectionMode.CAMERA

    dummy_element = "do1"  # wait_for_trigger requires us to pass some element
    if num_reps == -1:
        with qua.infinite_loop_():
            one_rep_macro()
            qua.align()
            if wait_for_trigger:
                qua.wait_for_trigger(dummy_element)
                qua.align()
    elif num_reps == 1:
        one_rep_macro()
    else:
        handle_reps_ind = qua.declare(int, value=0)
        with qua.while_(handle_reps_ind < num_reps):
            one_rep_macro()
            qua.align()
            qua.assign(handle_reps_ind, handle_reps_ind + 1)
            if wait_for_trigger:
                qua.wait_for_trigger(dummy_element)
                qua.align()


def polarize(laser_name, duration_ns, coords_list):
    _pulse_list(laser_name, duration_ns, coords_list)


def ionize(laser_name, duration_ns, coords_list):
    _pulse_list(laser_name, duration_ns, coords_list)


def _pulse_list(laser_name, duration_ns, coords_list):
    # Setup
    laser_el = get_laser_mod_element(laser_name)
    x_el = f"ao_{laser_name}_x"
    y_el = f"ao_{laser_name}_y"

    x_freq = qua.declare(int)
    y_freq = qua.declare(int)

    duration = convert_ns_to_cc(duration_ns)
    buffer = get_widefield_operation_buffer()

    access_time = get_aod_access_time()

    qua.align()

    for coords_pair in coords_list:
        # Update AOD frequencies
        qua.assign(x_freq, coords_pair[0] * 10**6)
        qua.assign(y_freq, coords_pair[1] * 10**6)
        qua.update_frequency(x_el, x_freq)
        qua.update_frequency(y_el, y_freq)
        qua.play("aod_cw", x_el)
        qua.play("aod_cw", y_el)

        # Pulse the laser
        qua.wait(buffer + access_time, laser_el)
        qua.play("on", laser_el, duration=duration)
        qua.wait(buffer, laser_el)

        qua.align()


# endregion


def convert_ns_to_cc(duration_ns):
    """Convert a duration from nanoseconds to clock cycles"""
    return round(duration_ns / 4)


def get_default_pulse_duration():
    """Get the default OPX pulse duration in units of clock cycles"""
    return get_common_duration("default_pulse_duration")


def get_aod_access_time():
    return get_common_duration("aod_access_time")


def get_widefield_operation_buffer():
    return get_common_duration("widefield_operation_buffer")


def get_common_duration(key):
    config = common.get_config_dict()
    common_duration_ns = config["CommonDurations"][key]
    common_duration_cc = convert_ns_to_cc(common_duration_ns)
    return common_duration_cc


def get_laser_mod_element(laser_name):
    config = common.get_config_dict()
    mod_mode = config["Optics"][laser_name]["mod_mode"]
    if mod_mode == ModMode.ANALOG:
        laser_mod_element = f"ao_{laser_name}_am"
    elif mod_mode == ModMode.DIGITAL:
        laser_mod_element = f"do_{laser_name}_dm"
    return laser_mod_element
