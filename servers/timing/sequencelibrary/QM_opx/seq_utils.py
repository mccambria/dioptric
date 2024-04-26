# -*- coding: utf-8 -*-
"""
QM OPX sequence utils. Should only be used by sequence files

Created June 25th, 2023

@author: mccambria
"""

import logging
import time
from functools import cache

from qm import qua

from utils import common
from utils import tool_belt as tb
from utils.constants import LaserKey, ModMode

# Cached QUA variables to save on declaring variables more than we have to
_cache_x_freq = None
_cache_y_freq = None
_cache_x_freq_2 = None
_cache_y_freq_2 = None
_cache_macro_run_aods = None
_cache_charge_pol_target_list = None
_cache_charge_pol_incomplete = None


# region QUA macros


def init(num_nvs=None):
    """
    This should be the first command we call in any sequence
    """
    # Declare cached QUA variables (helps reduce compile times)
    global _cache_x_freq
    global _cache_y_freq
    global _cache_x_freq_2
    global _cache_y_freq_2
    _cache_x_freq = qua.declare(int)
    _cache_y_freq = qua.declare(int)
    _cache_x_freq_2 = qua.declare(int)
    _cache_y_freq_2 = qua.declare(int)

    global _cache_target
    _cache_target = qua.declare(bool)

    global _cache_macro_run_aods
    _cache_macro_run_aods = {}
    macro_run_aods()

    global _cache_charge_pol_incomplete
    _cache_charge_pol_incomplete = qua.declare_input_stream(
        bool, name="_cache_charge_pol_incomplete"
    )

    if num_nvs is not None:
        global _cache_target_list
        _cache_target_list = qua.declare_input_stream(
            bool, name="_cache_target_list", size=num_nvs
        )


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
            if wait_for_trigger:
                macro_wait_for_trigger()
    elif num_reps == 1:
        one_rep_macro()
    else:
        handle_reps_ind = qua.declare(int)
        with qua.for_(
            handle_reps_ind, 0, handle_reps_ind < num_reps, handle_reps_ind + 1
        ):
            one_rep_macro()
            if wait_for_trigger:
                macro_wait_for_trigger()


# def macro_polarize(pol_coords_list, pol_duration=None):
#     """Apply a polarization pulse to each coordinate pair in the passed coords_list.
#     Pulses are applied in series

#     Parameters
#     ----------
#     pol_laser_name : str
#         Name of polarization laser
#     pol_duration : numeric
#         Duration of the pulse in ns
#     pol_coords_list : list(coordinate pairs)
#         List of coordinate pairs to target
#     """

#     pol_laser_name = tb.get_laser_name(LaserKey.CHARGE_POL)
#     pulse_name = "charge_pol"
#     macro_run_aods(laser_names=[pol_laser_name], aod_suffices=[pulse_name])
#     _macro_pulse_list(pol_laser_name, pol_coords_list, pulse_name, pol_duration)

#     # Spin polarization with widefield yellow
#     readout_laser_name = tb.get_laser_name(LaserKey.WIDEFIELD_SPIN_POL)
#     readout_laser_el = get_laser_mod_element(readout_laser_name)
#     buffer = get_widefield_operation_buffer()
#     qua.align()
#     qua.play("spin_pol", readout_laser_el)
#     qua.wait(buffer, readout_laser_el)


def macro_polarize(
    pol_coords_list,
    pol_duration=None,
    spin_pol=True,
    targeted_polarization=True,
    verify_charge_states=False,
):
    """Apply a polarization pulse to each coordinate pair in the passed coords_list.
    Pulses are applied in series

    Parameters
    ----------
    pol_laser_name : str
        Name of polarization laser
    pol_duration : numeric
        Duration of the pulse in ns
    pol_coords_list : list(coordinate pairs)
        List of coordinate pairs to target
    """

    global _cache_charge_pol_incomplete
    global _cache_target_list

    pol_laser_name = tb.get_laser_name(LaserKey.CHARGE_POL)
    pulse_name = "charge_pol"
    macro_run_aods(laser_names=[pol_laser_name], aod_suffices=[pulse_name])

    if verify_charge_states:
        qua.advance_input_stream(_cache_charge_pol_incomplete)
        with qua.while_(_cache_charge_pol_incomplete):
            qua.advance_input_stream(_cache_target_list)
            _macro_pulse_list(
                pol_laser_name,
                pol_coords_list,
                pulse_name,
                pol_duration,
                target_list=_cache_target_list,
            )
            macro_charge_state_readout()
            macro_wait_for_trigger()
            qua.advance_input_stream(_cache_charge_pol_incomplete)
    elif targeted_polarization:
        qua.advance_input_stream(_cache_target_list)
        _macro_pulse_list(
            pol_laser_name,
            pol_coords_list,
            pulse_name,
            pol_duration,
            target_list=_cache_target_list,
        )
    else:
        _macro_pulse_list(pol_laser_name, pol_coords_list, pulse_name, pol_duration)

    # Spin polarization with widefield yellow
    if spin_pol:
        spin_pol_laser_name = tb.get_laser_name(LaserKey.WIDEFIELD_SPIN_POL)
        spin_pol_laser_el = get_laser_mod_element(spin_pol_laser_name)
        buffer = get_widefield_operation_buffer()
        qua.align()
        qua.play("spin_pol", spin_pol_laser_el)
        qua.wait(buffer, spin_pol_laser_el)


# def macro_polarize(pol_coords_list, pol_duration=None):
#     """Apply a polarization pulse to each coordinate pair in the passed coords_list.
#     Pulses are applied in series

#     Parameters
#     ----------
#     pol_laser_name : str
#         Name of polarization laser
#     pol_duration : numeric
#         Duration of the pulse in ns
#     pol_coords_list : list(coordinate pairs)
#         List of coordinate pairs to target
#     """

#     pol_laser_name = tb.get_laser_name(LaserKey.CHARGE_POL)
#     pol_pulse_name = "charge_pol"
#     readout_laser_name = tb.get_laser_name(LaserKey.WIDEFIELD_SPIN_POL)
#     readout_laser_el = get_laser_mod_element(readout_laser_name)
#     spin_pulse_name = "spin_pol"
#     buffer = get_widefield_operation_buffer()
#     uwave_ind = 0

#     macro_run_aods(laser_names=[pol_laser_name], aod_suffices=[pol_pulse_name])
#     _macro_pulse_list(pol_laser_name, pol_coords_list, pol_pulse_name, pol_duration)

#     # Spin polarization with widefield yellow
#     qua.align()
#     qua.play(spin_pulse_name, readout_laser_el)
#     qua.wait(buffer, readout_laser_el)

#     pol_reps_ind = qua.declare(int)
#     with qua.for_(pol_reps_ind, 0, pol_reps_ind < 20, pol_reps_ind + 1):
#         # MCC
#         sig_gen_el = get_sig_gen_element(uwave_ind)
#         qua.align()
#         qua.play("pi_pulse", sig_gen_el)
#         qua.wait(buffer, sig_gen_el)

#         macro_run_aods(laser_names=[pol_laser_name], aod_suffices=[pol_pulse_name])
#         _macro_pulse_list(pol_laser_name, pol_coords_list, pol_pulse_name, 15)

#         # Spin polarization with widefield yellow
#         qua.align()
#         qua.play(spin_pulse_name, readout_laser_el)
#         qua.wait(buffer, readout_laser_el)


def macro_ionize(ion_coords_list, ion_duration=None):
    """Apply an ionitization pulse to each coordinate pair in the passed coords_list.
    Pulses are applied in series

    Parameters
    ----------
    ion_laser_name : str
        Name of ionitization laser
    ion_duration : numeric
        Duration of the pulse in ns
    ion_coords_list : list(coordinate pairs)
        List of coordinate pairs to target
    """
    ion_laser_name = tb.get_laser_name(LaserKey.ION)
    macro_run_aods([ion_laser_name], aod_suffices=["ion"])
    ion_pulse_name = "ion"
    _macro_pulse_list(ion_laser_name, ion_coords_list, ion_pulse_name, ion_duration)


def macro_scc(
    ion_coords_list,
    spin_flip_ind_list=None,
    uwave_ind=None,
    ion_duration=None,
    shelving_coords_list=None,
):
    """Apply an ionitization pulse to each coordinate pair in the passed coords_list.
    Pulses are applied in series

    Parameters
    ----------
    ion_coords_list : list(coordinate pairs)
        List of coordinate pairs to target
    ion_duration : numeric
        Duration of the pulse in clock cycles (4 ns)
    """

    config = common.get_config_dict()
    do_shelving_pulse = config["Optics"]["scc_shelving_pulse"]

    if do_shelving_pulse:
        if spin_flip_ind_list is not None:
            raise NotImplementedError(
                "Shelving SCC with spin_flips not yet implemented."
            )
        _macro_scc_shelving(ion_coords_list, ion_duration, shelving_coords_list)
    else:
        _macro_scc_no_shelving(
            ion_coords_list, spin_flip_ind_list, uwave_ind, ion_duration
        )


def _macro_scc_shelving(ion_coords_list, ion_duration, shelving_coords_list):
    shelving_laser_name = tb.get_laser_name(LaserKey.SHELVING)
    ion_laser_name = tb.get_laser_name(LaserKey.SCC)
    laser_name_list = [shelving_laser_name, ion_laser_name]
    shelving_pulse_name = "shelving"
    ion_pulse_name = "scc"
    pulse_name_list = [shelving_pulse_name, ion_pulse_name]
    shelving_laser_dict = tb.get_optics_dict(LaserKey.SHELVING)
    shelving_pulse_duration = shelving_laser_dict["duration"]
    shelving_scc_gap_ns = 16
    shelving_scc_gap = convert_ns_to_cc(shelving_scc_gap_ns)
    delays = [0, shelving_pulse_duration + shelving_scc_gap]
    duration_list = [None, ion_duration]

    macro_run_aods(laser_name_list, aod_suffices=pulse_name_list)

    # Unpack the coords and convert to Hz
    x_shelving_coords_list = [int(el[0] * 10**6) for el in shelving_coords_list]
    y_shelving_coords_list = [int(el[1] * 10**6) for el in shelving_coords_list]
    x_ion_coords_list = [int(el[0] * 10**6) for el in ion_coords_list]
    y_ion_coords_list = [int(el[1] * 10**6) for el in ion_coords_list]

    # These are declared in macro_run_aods
    global _cache_x_freq
    global _cache_y_freq
    global _cache_x_freq_2
    global _cache_y_freq_2
    freq_vars = (_cache_x_freq, _cache_y_freq, _cache_x_freq_2, _cache_y_freq_2)
    freq_lists = (
        x_shelving_coords_list,
        y_shelving_coords_list,
        x_ion_coords_list,
        y_ion_coords_list,
    )

    qua.align()
    with qua.for_each_(freq_vars, freq_lists):
        macro_multi_pulse(
            laser_name_list,
            ((_cache_x_freq, _cache_y_freq), (_cache_x_freq_2, _cache_y_freq_2)),
            pulse_name_list,
            duration_list=duration_list,
            delays=delays,
            convert_to_Hz=False,
        )


def _macro_scc_no_shelving(
    ion_coords_list, spin_flip_ind_list=None, uwave_ind=None, ion_duration=None
):
    # Basic setup

    ion_laser_name = tb.get_laser_name(LaserKey.SCC)
    ion_pulse_name = "scc"
    macro_run_aods([ion_laser_name], aod_suffices=[ion_pulse_name])

    if spin_flip_ind_list is None:
        spin_flip_ind_list = []

    num_nvs = len(ion_coords_list)
    first_ion_coords_list = [
        ion_coords_list[ind] for ind in range(num_nvs) if ind not in spin_flip_ind_list
    ]

    # Actual commands

    _macro_pulse_list(
        ion_laser_name, first_ion_coords_list, ion_pulse_name, ion_duration
    )

    # Just exit here if all NVs are SCC'ed in the first batch
    if len(spin_flip_ind_list) == 0:
        return

    sig_gen_el = get_sig_gen_element(uwave_ind)
    buffer = get_widefield_operation_buffer()

    second_ion_coords_list = [
        ion_coords_list[ind] for ind in range(num_nvs) if ind in spin_flip_ind_list
    ]

    qua.align()
    qua.play("pi_pulse", sig_gen_el)
    qua.wait(buffer, sig_gen_el)

    _macro_pulse_list(
        ion_laser_name, second_ion_coords_list, ion_pulse_name, ion_duration
    )


def macro_charge_state_readout(readout_duration_ns=None):
    readout_laser_name = tb.get_laser_name(LaserKey.WIDEFIELD_CHARGE_READOUT)
    readout_laser_el = get_laser_mod_element(readout_laser_name, sticky=True)

    camera_el = "do_camera_trigger"

    default_duration = get_default_pulse_duration()
    if readout_duration_ns is not None:
        readout_duration = convert_ns_to_cc(readout_duration_ns)
    else:
        readout_duration = get_default_charge_readout_duration()
    wait_duration = readout_duration - default_duration

    qua.align()
    qua.play("charge_readout", readout_laser_el)
    qua.play("on", camera_el)
    qua.wait(wait_duration, readout_laser_el)
    qua.wait(wait_duration, camera_el)
    qua.ramp_to_zero(readout_laser_el)
    qua.ramp_to_zero(camera_el)


def macro_wait_for_trigger():
    """Pauses the entire sequence and waits for a trigger pulse from the camera.
    The wait does not start until all running pulses finish"""
    dummy_element = (
        "do_camera_trigger"  # wait_for_trigger requires us to pass some element
    )
    qua.align()
    qua.wait_for_trigger(dummy_element)


def macro_pause():
    buffer = get_widefield_operation_buffer()
    # Make sure everything is off before pausing for the next step
    qua.align()
    qua.wait(buffer)
    qua.pause()


def _get_default_aod_suffix(laser_name):
    config = common.get_config_dict()
    return config["Optics"][laser_name]["default_aod_suffix"]


def macro_run_aods(laser_names=None, aod_suffices=None, amps=None):
    """Turn on the AODs. They'll run indefinitely. Use pulse_suffix to run a pulse
    with a different power, etc"""
    # By default search the config for the lasers with AOD
    if laser_names is None:
        config = common.get_config_dict()
        config_optics = config["Optics"]
        optics_keys = config_optics.keys()
        laser_names = []
        for key in optics_keys:
            val = config_optics[key]
            if isinstance(val, dict) and "aod" in val and val["aod"]:
                laser_names.append(key)

    num_lasers = len(laser_names)

    # Adjust the pulse names for the passed suffices - defaults in the config
    base_pulse_name = "aod_cw"
    pulse_names = []
    if aod_suffices is None:
        aod_suffices = [_get_default_aod_suffix(el) for el in laser_names]
    pulse_names = []
    for ind in range(num_lasers):
        suffix = aod_suffices[ind]
        if not suffix:
            laser_name = laser_names[ind]
            suffix = _get_default_aod_suffix(laser_name)
        pulse_name = f"{base_pulse_name}-{suffix}"
        pulse_names.append(pulse_name)

    if amps is None:
        amps = [None for el in laser_names]

    ### Check if the requested pulse is already running using the cache

    global _cache_macro_run_aods
    skip_inds = []
    for ind in range(num_lasers):
        laser_name = laser_names[ind]
        aod_suffix = aod_suffices[ind]
        amp = amps[ind]
        # Don't cache commands with amps since those may be QUA variables and so
        # could behave unexpectedly
        if amp is not None:
            continue
        if laser_name in _cache_macro_run_aods:
            cache_dict = _cache_macro_run_aods[laser_name]
            if cache_dict["aod_suffix"] == aod_suffix:
                skip_inds.append(ind)
                continue
        # If we get here then the pulse is not already running, so cache it
        _cache_macro_run_aods[laser_name] = {}
        cache_dict = _cache_macro_run_aods[laser_name]
        cache_dict["aod_suffix"] = aod_suffix

    ### Actual commands here

    qua.align()

    for ind in range(num_lasers):
        if ind in skip_inds:
            continue
        laser_name = laser_names[ind]
        x_el = f"ao_{laser_name}_x"
        y_el = f"ao_{laser_name}_y"
        pulse_name = pulse_names[ind]

        qua.ramp_to_zero(x_el)
        qua.ramp_to_zero(y_el)

        amp = amps[ind]
        if amp is not None:
            qua.play(pulse_name * qua.amp(amp), x_el)
            qua.play(pulse_name * qua.amp(amp), y_el)
        else:
            qua.play(pulse_name, x_el)
            qua.play(pulse_name, y_el)


def _macro_pulse_list(
    laser_name, coords_list, pulse_name="on", duration=None, target_list=None
):
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
    duration : numeric
        Duration of the pulse in clock cycles (4 ns) - if None, uses the default
        duration of the passed pulse
    """

    if len(coords_list) == 0:
        return

    # Unpack the coords and convert to Hz
    x_coords_list = [int(el[0] * 10**6) for el in coords_list]
    y_coords_list = [int(el[1] * 10**6) for el in coords_list]

    # These are declared in init
    global _cache_x_freq, _cache_y_freq, _cache_target

    qua.align()
    if target_list is None:
        with qua.for_each_(
            (_cache_x_freq, _cache_y_freq), (x_coords_list, y_coords_list)
        ):
            macro_pulse(
                laser_name,
                (_cache_x_freq, _cache_y_freq),
                pulse_name=pulse_name,
                duration=duration,
                convert_to_Hz=False,
            )
    else:
        # qua.advance_input_stream(input_stream)
        with qua.for_each_(
            (_cache_x_freq, _cache_y_freq, _cache_target),
            (x_coords_list, y_coords_list, target_list),
        ):
            with qua.if_(_cache_target):
                macro_pulse(
                    laser_name,
                    (_cache_x_freq, _cache_y_freq),
                    pulse_name=pulse_name,
                    duration=duration,
                    convert_to_Hz=False,
                )


def macro_pulse(laser_name, coords, pulse_name="on", duration=None, convert_to_Hz=True):
    qua.align()
    _macro_single_pulse(
        laser_name, coords, pulse_name, duration, convert_to_Hz=convert_to_Hz
    )


def macro_multi_pulse(
    laser_name_list,
    coords_list,
    pulse_name_list,
    duration_list=None,
    delays=None,
    convert_to_Hz=True,
):
    num_pulses = len(laser_name_list)
    if delays is None:
        delays = [0 for ind in range(num_pulses)]
    if duration_list is None:
        duration_list = [None for ind in range(num_pulses)]

    qua.align()
    for ind in range(num_pulses):
        # for ind in [1]:
        laser_name = laser_name_list[ind]
        coords = coords_list[ind]
        pulse_name = pulse_name_list[ind]
        duration = duration_list[ind]
        delay = delays[ind]
        _macro_single_pulse(
            laser_name, coords, pulse_name, duration, delay, convert_to_Hz
        )


def _macro_single_pulse(
    laser_name, coords, pulse_name="on", duration=None, delay=0, convert_to_Hz=True
):
    # Setup
    laser_el = get_laser_mod_element(laser_name)
    x_el = f"ao_{laser_name}_x"
    y_el = f"ao_{laser_name}_y"

    buffer = get_widefield_operation_buffer()
    access_time = get_aod_access_time()

    if convert_to_Hz:
        coords = [int(el * 10**6) for el in coords]

    qua.play("continue", x_el)
    qua.play("continue", y_el)
    qua.update_frequency(x_el, coords[0])
    qua.update_frequency(y_el, coords[1])

    # Pulse the laser
    qua.wait(access_time + buffer + delay, laser_el)
    if duration is None:
        qua.play(pulse_name, laser_el)
    elif isinstance(duration, int) and duration == 0:
        pass
    else:
        qua.play(pulse_name, laser_el, duration=duration)
    qua.wait(buffer, laser_el)


# endregion


def convert_ns_to_cc(duration_ns, allow_rounding=False, allow_zero=False):
    """Convert a duration from nanoseconds to clock cycles"""
    if duration_ns is None:
        return None
    if not allow_rounding and duration_ns % 4 != 0:
        raise RuntimeError("OPX pulse durations (in ns) must be divisible by 4")
    if not allow_zero and duration_ns == 0:
        raise RuntimeError("OPX pulse duration 0 not supported here")
    if 0 < duration_ns < 16:
        raise RuntimeError("Minimum OPX pulse duration is 16 ns")
    return round(duration_ns / 4)


@cache
def get_default_charge_readout_duration():
    readout_laser_dict = tb.get_optics_dict(LaserKey.WIDEFIELD_CHARGE_READOUT)
    readout_duration_ns = readout_laser_dict["duration"]
    return convert_ns_to_cc(readout_duration_ns)


@cache
def get_default_pulse_duration():
    """Get the default OPX pulse duration in units of clock cycles"""
    return get_common_duration_cc("default_pulse_duration")


@cache
def get_aod_access_time():
    return get_common_duration_cc("aod_access_time")


@cache
def get_widefield_operation_buffer():
    return get_common_duration_cc("widefield_operation_buffer")


@cache
def get_common_duration_cc(key):
    common_duration_ns = tb.get_common_duration(key)
    common_duration_cc = convert_ns_to_cc(common_duration_ns)
    return common_duration_cc


@cache
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


@cache
def get_sig_gen_element(uwave_ind=0):
    config = common.get_config_dict()
    sig_gen_name = config["Microwaves"][f"sig_gen_{uwave_ind}"]["name"]
    sig_gen_element = f"do_{sig_gen_name}_dm"
    return sig_gen_element


@cache
def get_iq_mod_elements(uwave_ind=0):
    config = common.get_config_dict()
    sig_gen_name = config["Microwaves"][f"sig_gen_{uwave_ind}"]["name"]
    i_el = f"ao_{sig_gen_name}_i"
    q_el = f"ao_{sig_gen_name}_q"
    return i_el, q_el


@cache
def get_rabi_period(uwave_ind=0):
    config = common.get_config_dict()
    rabi_period_ns = config["Microwaves"][f"sig_gen_{uwave_ind}"]["rabi_period"]
    rabi_period = convert_ns_to_cc(rabi_period_ns)
    return rabi_period


if __name__ == "__main__":
    start = time.time()
    for ind in range(1000):
        get_rabi_period()
    stop = time.time()
    print(stop - start)
