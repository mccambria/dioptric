# -*- coding: utf-8 -*-
"""
QM OPX sequence utils. Should only be used by sequence files

Created June 25th, 2023

@author: mccambria
"""


import time
from qm import qua
from utils import common
from utils.constants import LaserKey, ModMode, CollectionMode
from utils import tool_belt as tb


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


def macro_polarize(pol_coords_list, pol_duration_ns=None):
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
    pol_laser_name = tb.get_laser_name(LaserKey.POLARIZATION)
    _macro_pulse_list(pol_laser_name, pol_coords_list, "polarize", pol_duration_ns)


def macro_ionize(ion_coords_list, ion_duration_ns=None):
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
    ion_laser_name = tb.get_laser_name(LaserKey.IONIZATION)
    _macro_pulse_list(ion_laser_name, ion_coords_list, "ionize", ion_duration_ns)


def macro_charge_state_readout(readout_duration_ns=None):
    readout_laser_name = tb.get_laser_name(LaserKey.CHARGE_READOUT)
    readout_laser_el = get_laser_mod_element(readout_laser_name, sticky=True)
    camera_el = f"do_camera_trigger"

    default_duration = get_default_pulse_duration()
    if readout_duration_ns is None:
        readout_laser_dict = tb.get_laser_dict(LaserKey.CHARGE_READOUT)
        readout_duration_ns = readout_laser_dict["duration"]
    readout_duration = convert_ns_to_cc(readout_duration_ns)

    qua.align()
    qua.play("charge_readout", readout_laser_el)
    qua.play("on", camera_el)
    qua.align()
    qua.wait(readout_duration - default_duration)
    qua.align()
    qua.ramp_to_zero(readout_laser_el)
    qua.ramp_to_zero(camera_el)
    qua.align()


def macro_wait_for_trigger():
    """Pauses the entire sequence and waits for a trigger pulse from the camera.
    The wait does not start until all running pulses finish"""
    dummy_element = "do1"  # wait_for_trigger requires us to pass some element
    qua.align()
    qua.wait_for_trigger(dummy_element)
    qua.align()


def turn_on_aods(laser_names=None):
    # By default search the config for the lasers with AOD
    if laser_names is None:
        config = common.get_config_dict()
        config_optics = config["Optics"]
        optics_keys = config_optics.keys()
        laser_names = []
        for key in optics_keys:
            if "aod" in config_optics[key] and config_optics[key]["aod"]:
                laser_names.append(key)
    for laser_name in laser_names:
        x_el = f"ao_{laser_name}_x"
        y_el = f"ao_{laser_name}_y"
        qua.play("aod_cw", x_el)
        qua.play("aod_cw", y_el)


def _macro_pulse_list(laser_name, coords_list, pulse_name="on", duration_ns=None):
    """Apply a laser pulse to each coordinate pair in the passed coords_list.
    Pulses are applied in series

    Parameters
    ----------
    laser_name : str
        Name of laser to pulse
    coords_list : list(coordinate pairs)
        List of coordinate pairs to target
    pulse_name : str
        Name of the pulse to play - "on" by default
    duration_ns : numeric
        Duration of the pulse in ns - if None, uses the default duration of the passed pulse
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
        # The continue pulse doesn't actually change anything - without a new pulse the
        # compiler will overwrite the frequency of whatever is playing retroactively
        qua.play("continue", x_el)
        qua.play("continue", y_el)
        qua.update_frequency(x_el, round(coords_pair[0] * 10**6))
        qua.update_frequency(y_el, round(coords_pair[1] * 10**6))

        # Pulse the laser
        qua.wait(access_time + buffer, laser_el)
        if duration is None:
            qua.play(pulse_name, laser_el)
        elif duration > 0:
            qua.play(pulse_name, laser_el)
        qua.wait(buffer, laser_el)

        qua.align()


# endregion


def convert_ns_to_cc(duration_ns, raise_error=False):
    """Convert a duration from nanoseconds to clock cycles"""
    if duration_ns is None:
        return None
    if raise_error:
        if duration_ns % 4 != 0:
            raise RuntimeError("OPX pulse durations (in ns) must be divisible by 4")
    # Raise this error regardless of raise_error because this will lead to unexpected behavior
    if duration_ns < 16:
        raise RuntimeError("Minimum OPX pulse duration is 16 ns")
    return round(duration_ns / 4)


def get_default_pulse_duration():
    """Get the default OPX pulse duration in units of clock cycles"""
    return get_common_duration_cc("default_pulse_duration")


def get_aod_access_time():
    return get_common_duration_cc("aod_access_time")


def get_widefield_operation_buffer():
    return get_common_duration_cc("widefield_operation_buffer")


def get_common_duration_cc(key):
    common_duration_ns = tb.get_common_duration(key)
    common_duration_cc = convert_ns_to_cc(common_duration_ns)
    return common_duration_cc


def get_laser_mod_element(laser_name, sticky=False):
    config = common.get_config_dict()
    mod_mode = config["Optics"][laser_name]["mod_mode"]
    if sticky:
        if mod_mode == ModMode.ANALOG:
            laser_mod_element = f"ao_{laser_name}_am_sticky"
        elif mod_mode == ModMode.DIGITAL:
            laser_mod_element = f"do_{laser_name}_dm_sticky"
    else:
        if mod_mode == ModMode.ANALOG:
            laser_mod_element = f"ao_{laser_name}_am"
        elif mod_mode == ModMode.DIGITAL:
            laser_mod_element = f"do_{laser_name}_dm"
    return laser_mod_element


def get_sig_gen_element(uwave_ind=0):
    config = common.get_config_dict()
    sig_gen_name = config["Microwaves"][f"sig_gen_{uwave_ind}"]["name"]
    return f"do_{sig_gen_name}_dm"


def get_iq_mod_elements(uwave_ind=0):
    config = common.get_config_dict()
    sig_gen_name = config["Microwaves"][f"sig_gen_{uwave_ind}"]["name"]
    i_el = f"ao_{sig_gen_name}_i"
    q_el = f"ao_{sig_gen_name}_q"
    return i_el, q_el


def get_rabi_period(uwave_ind=0):
    config = common.get_config_dict()
    rabi_period_ns = config["Microwaves"][f"sig_gen_{uwave_ind}"]["rabi_period"]
    return convert_ns_to_cc(rabi_period_ns)

    i_el, q_el = seq_utils.get_iq_mod_elements()
    rabi_period = seq_utils.get_rabi_period()


if __name__ == "__main__":
    turn_on_aods()
