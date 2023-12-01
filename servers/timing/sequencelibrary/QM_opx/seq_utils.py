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
    wait_for_trigger=True,
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
        pulse is received by the OPX, defaults to True
    """

    if num_reps == -1:
        with qua.infinite_loop_():
            one_rep_macro()
            qua.align()
            if wait_for_trigger:
                macro_wait_for_trigger()
    elif num_reps == 1:
        one_rep_macro()
    else:
        handle_reps_ind = qua.declare(int, value=0)
        with qua.while_(handle_reps_ind < num_reps):
            one_rep_macro()
            qua.align()
            qua.assign(handle_reps_ind, handle_reps_ind + 1)
            if wait_for_trigger:
                macro_wait_for_trigger()


def macro_polarize(pol_laser_name, pol_duration_ns, pol_coords_list, dummy_pulse=False):
    """Apply a polarization pulse to each coordinate pair in the passed coords_list.
    Pulses are applied in series

    Parameters
    ----------
    pol_laser_name : str
        Name of polarization laser
    pol_duration_ns : numeric
        Duration of the pulse in ns
    pol_coords_list : list(coordinate pairs)
        List of coordinate pairs to target
    """
    _macro_pulse_list(pol_laser_name, pol_duration_ns, pol_coords_list, dummy_pulse)


def macro_ionize(ion_laser_name, ion_duration_ns, ion_coords_list, dummy_pulse=False):
    """Apply an ionitization pulse to each coordinate pair in the passed coords_list.
    Pulses are applied in series

    Parameters
    ----------
    ion_laser_name : str
        Name of ionitization laser
    ion_duration_ns : numeric
        Duration of the pulse in ns
    ion_coords_list : list(coordinate pairs)
        List of coordinate pairs to target
    """
    _macro_pulse_list(ion_laser_name, ion_duration_ns, ion_coords_list, dummy_pulse)


def macro_charge_state_readout(readout_laser_name, readout_duration_ns):
    readout_laser_el = get_laser_mod_element(readout_laser_name)
    camera_el = f"do_camera_trigger"

    readout_duration = convert_ns_to_cc(readout_duration_ns)

    qua.align()
    qua.play("charge_readout", readout_laser_el, duration=readout_duration)
    qua.play("on", camera_el)
    qua.align()
    qua.play("off", camera_el)
    qua.align()


def macro_wait_for_trigger():
    """Pauses the entire sequence and waits for a trigger pulse from the camera.
    The wait does not start until all running pulses finish"""
    dummy_element = "do1"  # wait_for_trigger requires us to pass some element
    qua.align()
    qua.wait_for_trigger(dummy_element)
    qua.align()


# def declare_scc_qua_vars():
#     pol_x_freq = qua.declare(int)
#     pol_y_freq = qua.declare(int)
#     ion_x_freq = qua.declare(int)
#     ion_y_freq = qua.declare(int)

#     scc_vars = [pol_x_freq, pol_y_freq, ion_x_freq, ion_y_freq]
#     return scc_vars


def turn_on_aods(laser_names):
    for laser_name in laser_names:
        x_el = f"ao_{laser_name}_x"
        y_el = f"ao_{laser_name}_y"
        qua.play("aod_cw", x_el)
        qua.play("aod_cw", y_el)


def _macro_pulse_list(laser_name, duration_ns, coords_list, dummy_pulse=False):
    """Apply a laser pulse to each coordinate pair in the passed coords_list.
    Pulses are applied in series

    Parameters
    ----------
    laser_name : str
        Name of laser to pulse
    duration_ns : numeric
        Duration of the pulse in ns
    coords_list : list(coordinate pairs)
        List of coordinate pairs to target
    """
    # Setup
    laser_el = get_laser_mod_element(laser_name)
    x_el = f"ao_{laser_name}_x"
    y_el = f"ao_{laser_name}_y"

    duration = convert_ns_to_cc(duration_ns)
    buffer = get_widefield_operation_buffer()

    access_time = get_aod_access_time()

    qua.align()

    for coords_pair in coords_list:
        # Update AOD frequencies
        # The continue pulse doesn't actually change anything - without a new
        # pulse the compiler will overwrite the frequency of whatever is playing
        # retroactively
        qua.play("continue", x_el)
        qua.play("continue", y_el)
        qua.update_frequency(x_el, round(coords_pair[0] * 10**6))
        qua.update_frequency(y_el, round(coords_pair[1] * 10**6))

        # Pulse the laser
        qua.wait(access_time + buffer, laser_el)
        pulse = "off" if dummy_pulse else "on"
        qua.play(pulse, laser_el, duration=duration)
        qua.wait(buffer, laser_el)

        qua.align()


# endregion


def convert_ns_to_cc(duration_ns, raise_error=False):
    """Convert a duration from nanoseconds to clock cycles"""
    if raise_error:
        if duration_ns % 4 != 0:
            raise RuntimeError("OPX pulse durations (in ns) must be divisible by 4")
    # Raise this error regardless of raise_error because this will lead to unexpected behavior
    if duration_ns < 16:
        raise RuntimeError("Minimum OPX pulse duration is 16 ns")
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
