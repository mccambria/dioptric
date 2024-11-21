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
from utils import positioning as pos
from utils import tool_belt as tb
from utils.constants import ModMode, VirtualLaserKey

# region Public QUA macros - sequence management


def init(num_nvs=None):
    """This should be the first command we call in any sequence"""

    # Declare cached QUA variables (helps reduce compile times)
    global _cache_x_freq
    _cache_x_freq = qua.declare(int)
    global _cache_y_freq
    _cache_y_freq = qua.declare(int)
    global _cache_x_freq_2
    _cache_x_freq_2 = qua.declare(int)
    global _cache_y_freq_2
    _cache_y_freq_2 = qua.declare(int)

    global _cache_duration
    _cache_duration = qua.declare(int)

    global _cache_amp
    _cache_amp = qua.declare(qua.fixed)

    global _cache_target
    _cache_target = qua.declare(bool)

    global _cache_macro_run_aods
    _cache_macro_run_aods = {}
    macro_run_aods()

    global _cache_pol_reps_ind
    _cache_pol_reps_ind = qua.declare(int)

    global _cache_charge_pol_incomplete
    _cache_charge_pol_incomplete = qua.declare_input_stream(
        bool, name="_cache_charge_pol_incomplete"
    )

    if num_nvs is not None:
        global _cache_target_list
        _cache_target_list = qua.declare_input_stream(
            bool, name="_cache_target_list", size=num_nvs
        )


def handle_reps(one_rep_macro: qua.macro, num_reps: int, wait_for_trigger: bool = True):
    """Handle repetitions of a given sequence - you just have to pass
    a function defining the behavior for a single loop. Optionally
    waits for trigger pulse between loops.


    Parameters
    ----------
    one_rep_macro : qua.macro
        QUA macro to be repeated
    num_reps : int
        Number of times to repeat, -1 for infinite loop
    wait_for_trigger : bool, optional
        Whether or not to pause execution between loops until a trigger
        pulse is received by the OPX, by default True
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
            one_rep_macro(handle_reps_ind)
            if wait_for_trigger:
                macro_wait_for_trigger()


def macro_wait_for_trigger():
    """Pauses the entire sequence and waits for a trigger pulse from the camera.
    The wait does not start until all running pulses finish
    """
    # Wait_for_trigger requires us to pass some element so use an arbitrary dummy
    dummy_element = "do_camera_trigger"
    qua.align()
    qua.wait_for_trigger(dummy_element)


def macro_pause():
    """Pause the sequence and pass control back to the top-level routine for processing,
    switching signal generator frequencies, etc.
    """
    buffer = get_widefield_operation_buffer()
    # Make sure everything is off before pausing for the next step
    qua.align()
    qua.wait(buffer)
    qua.pause()


# endregion
# region Public QUA macros - laser and microwave pulses


def macro_polarize(
    coords_list,
    duration_list=None,
    pol_amp=None,
    targeted_polarization=False,
    verify_charge_states=False,
    spin_pol=True,
):
    """Apply a polarization pulse to each coordinate pair in the passed coords_list.
    Pulses are applied in series. Optionally apply a secondary spin polarization pulse
    with yellow after charge polarization with green

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

    pol_laser_name = tb.get_physical_laser_name(VirtualLaserKey.CHARGE_POL)
    pulse_name = "charge_pol"
    macro_run_aods(
        laser_names=[pol_laser_name],
        aod_suffices=[pulse_name],
        amps=[pol_amp],  # Pass the amplitude here
    )

    def macro_sub():
        target_list = _cache_target_list if targeted_polarization else None
        _macro_pulse_series(
            pol_laser_name,
            pulse_name,
            coords_list,
            duration_list=duration_list,
            target_list=target_list,
        )

    if verify_charge_states:
        qua.advance_input_stream(_cache_charge_pol_incomplete)
        with qua.while_(_cache_charge_pol_incomplete):
            qua.advance_input_stream(_cache_target_list)
            macro_sub()
            macro_charge_state_readout()
            macro_wait_for_trigger()
            qua.advance_input_stream(_cache_charge_pol_incomplete)
    elif targeted_polarization:
        qua.advance_input_stream(_cache_target_list)
        macro_sub()
    else:
        macro_sub()

    # Spin polarization with widefield yellow
    if spin_pol:
        spin_pol_laser_name = tb.get_physical_laser_name(
            VirtualLaserKey.WIDEFIELD_SPIN_POL
        )
        spin_pol_laser_el = get_laser_mod_element(spin_pol_laser_name)
        buffer = get_widefield_operation_buffer()
        qua.align()
        qua.play("spin_pol", spin_pol_laser_el)
        qua.wait(buffer, spin_pol_laser_el)


def macro_ionize(ion_coords_list):
    """Apply an ionitization pulse to each coordinate pair in the passed coords_list.
    Pulses are applied in series

    Parameters
    ----------
    ion_coords_list : list(coordinate pairs)
        List of coordinate pairs to target
    """
    ion_laser_name = tb.get_physical_laser_name(VirtualLaserKey.ION)
    ion_pulse_name = "ion"
    macro_run_aods([ion_laser_name], aod_suffices=[ion_pulse_name])
    _macro_pulse_series(ion_laser_name, ion_pulse_name, ion_coords_list)


def macro_scc(
    scc_coords_list,
    scc_duration_list=None,
    scc_amp_list=None,
    spin_flip_ind_list=None,
    uwave_ind_list=None,
    shelving_coords_list=None,
    scc_duration_override=None,
    scc_amp_override=None,
    exp_spin_flip=True,
    ref_spin_flip=False,
):
    """Apply an ionitization pulse to each coordinate pair in the passed coords_list.
    Pulses are applied in series

    Parameters
    ----------
    scc_coords_list : list(coordinate pairs)
        List of coordinate pairs to target
    ion_duration : numeric
        Duration of the pulse in clock cycles (4 ns)
    """

    config = common.get_config_dict()
    do_shelving_pulse = config["Optics"]["PulseSettings"]["scc_shelving_pulse"]

    if do_shelving_pulse:
        # if spin_flip_ind_list is not None:
        #     msg = "Shelving SCC with spin_flips not yet implemented."
        #     raise NotImplementedError(msg)
        _macro_scc_shelving(
            scc_coords_list,
            scc_duration_list,
            spin_flip_ind_list,
            uwave_ind_list,
            shelving_coords_list,
            exp_spin_flip=exp_spin_flip,
        )
    else:
        _macro_scc_no_shelving(
            scc_coords_list,
            scc_duration_list,
            scc_duration_override,
            scc_amp_list,
            scc_amp_override,
            spin_flip_ind_list,
            uwave_ind_list,
            exp_spin_flip=exp_spin_flip,
            ref_spin_flip=ref_spin_flip,
        )


def macro_charge_state_readout(readout_duration_ns=None, readout_amp=None):
    readout_laser_name = tb.get_physical_laser_name(
        VirtualLaserKey.WIDEFIELD_CHARGE_READOUT
    )
    readout_laser_el = get_laser_mod_element(readout_laser_name, sticky=True)

    camera_el = "do_camera_trigger"

    default_duration = get_default_pulse_duration()
    if readout_duration_ns is not None:
        readout_duration = convert_ns_to_cc(readout_duration_ns)
    else:
        readout_duration = get_default_charge_readout_duration()
    wait_duration = readout_duration - default_duration

    qua.align()
    if readout_amp is not None:
        qua.play("charge_readout" * qua.amp(readout_amp), readout_laser_el)
    else:
        qua.play("charge_readout", readout_laser_el)
    qua.play("on", camera_el)
    qua.wait(wait_duration, readout_laser_el)
    qua.wait(wait_duration, camera_el)
    qua.ramp_to_zero(readout_laser_el)
    qua.ramp_to_zero(camera_el)


def macro_pi_pulse(uwave_ind_list, duration_cc=None):
    if uwave_ind_list is None:
        return
    uwave_buffer = get_uwave_buffer()
    for uwave_ind in uwave_ind_list:
        sig_gen_el = get_sig_gen_element(uwave_ind)
        qua.align()
        if duration_cc is None:
            qua.play("pi_pulse", sig_gen_el)
        else:
            with qua.if_(duration_cc > 0):
                qua.play("pi_pulse", sig_gen_el, duration=duration_cc)
        qua.wait(uwave_buffer, sig_gen_el)


def macro_pi_on_2_pulse(uwave_ind_list):
    if uwave_ind_list is None:
        return
    uwave_buffer = get_uwave_buffer()
    for uwave_ind in uwave_ind_list:
        sig_gen_el = get_sig_gen_element(uwave_ind)
        qua.align()
        qua.play("pi_on_2_pulse", sig_gen_el)
        qua.wait(uwave_buffer, sig_gen_el)


def macro_run_aods(
    laser_names: list[str] = None,
    aod_suffices: list[str] = None,
    amps: list[float] = None,
):
    """Turn on the AODs. They'll run indefinitely. Use pulse_suffix to run a pulse
    with a different power, etc

    Parameters
    ----------
    laser_names : list[str], optional
        Lasers whose AODs we'll turn on, by default all lasers with AODs
    aod_suffices : list[str], optional
        Pulse format is "aod_cw-<suffix>", by default no suffix
    amps : list[float], optional
        Multiplicative amplitudes for the pulses, by default None
    """
    # By default search the config for the lasers with AOD
    if laser_names is None:
        config = common.get_config_dict("purcell")
        positioners_dict = config["Positioning"]["Positioners"]
        positioners_keys = positioners_dict.keys()
        laser_names = []
        for key in positioners_keys:
            positioner = positioners_dict[key]
            if "aod" in positioner and positioner["aod"]:
                virtual_laser_key = positioner["opti_virtual_laser_key"]
                physical_laser_name = tb.get_physical_laser_name(virtual_laser_key)
                laser_names.append(physical_laser_name)

    num_lasers = len(laser_names)

    # Adjust the pulse names for the passed suffices
    base_pulse_name = "aod_cw"
    pulse_names = []
    if aod_suffices is None:
        aod_suffices = [None] * num_lasers
    pulse_names = []
    for ind in range(num_lasers):
        suffix = aod_suffices[ind]
        if suffix is None:
            pulse_name = base_pulse_name
        else:
            pulse_name = f"{base_pulse_name}-{suffix}"
        pulse_names.append(pulse_name)

    if amps is None:
        amps = [None] * num_lasers

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


def macro_pulse(
    laser_name, coords, pulse_name="on", duration=None, amp=None, convert_to_Hz=True
):
    qua.align()
    _macro_single_pulse(
        laser_name, coords, pulse_name, duration, amp, convert_to_Hz=convert_to_Hz
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
    # for ind in range(num_pulses):
    for ind in [1]:
        laser_name = laser_name_list[ind]
        coords = coords_list[ind]
        pulse_name = pulse_name_list[ind]
        duration = duration_list[ind]
        delay = delays[ind]
        _macro_single_pulse(
            laser_name, coords, pulse_name, duration, delay, convert_to_Hz
        )


# endregion
# region Private QUA macros


def _macro_pulse_series(
    laser_name: str,
    pulse_name: str,
    coords_list: list[list[float]],
    duration_list: list[int] = None,
    duration_override: None | int = None,
    amp_list: list[float] = None,
    amp_override: None | float = None,
    target_list: None | list[bool] = None,
):
    """Apply a laser pulse to each coordinate pair in the passed coords_list.
    Pulses are applied in series from one location to the next.

    Parameters
    ----------
    laser_name : str
        Name of laser to pulse
    pulse_name : str
        Name of the pulse to play
    coords_list : list[list[float]]
        List of coordinate pairs to target
    duration_list : list[int], optional
        List of pulse durations, by default whatever value is in config
    duration_override : None | int, optional
        Pulse duration for all pulses - overrides duration_list.
        Useful for parameters sweeps. By default do not override
    amp_list : list[float], optional
        List of pulse amplitudes, by default whatever value is in config
    amp_override : None | float, optional
        Pulse amplitude for all pulses - overrides amp_list.
        Useful for parameters sweeps. By default do not override
    target_list : None | list[bool], optional
        List of whether to target an NV or not. Used to skip certain NVs.
        By default target all NVs
    """
    if len(coords_list) == 0:
        return

    # Unpack the coords and convert to Hz
    x_coords_list = [int(el[0] * 10**6) for el in coords_list]
    y_coords_list = [int(el[1] * 10**6) for el in coords_list]

    # Convert durations to clock cycles
    if duration_list is not None:
        duration_list = [convert_ns_to_cc(el) for el in duration_list]

    # These are declared in init
    global _cache_x_freq, _cache_y_freq, _cache_duration, _cache_amp, _cache_target

    def macro_sub():
        if duration_override is not None:
            duration = duration_override
        elif duration_list is not None:
            duration = _cache_duration
        else:
            duration = None
        if amp_override is not None:
            amp = amp_override
        elif amp_list is not None:
            amp = _cache_amp
        else:
            amp = None
        with qua.if_(_cache_target):
            macro_pulse(
                laser_name,
                (_cache_x_freq, _cache_y_freq),
                pulse_name=pulse_name,
                duration=duration,
                amp=amp,
                convert_to_Hz=False,
            )

    qua.align()

    # Adjust the for each based on what lists are populated

    list_1 = [_cache_x_freq, _cache_y_freq]
    list_2 = [x_coords_list, y_coords_list]
    if duration_list is not None:
        list_1.append(_cache_duration)
        list_2.append(duration_list)
    if amp_list is not None:
        list_1.append(_cache_amp)
        amp_list = [float(el) for el in amp_list]
        list_2.append(amp_list)
    if target_list is not None:
        list_1.append(_cache_target)
        list_2.append(target_list)
    else:  # Just set _cache_target to true if we want to hit every NV
        qua.assign(_cache_target, True)

    with qua.for_each_(tuple(list_1), tuple(list_2)):
        macro_sub()


def _macro_single_pulse(
    laser_name: str,
    coords: list[float],
    pulse_name: str,
    duration: None | int = None,
    amp: None | float = None,
    convert_to_Hz: bool = True,
):
    """_summary_

    Parameters
    ----------
    laser_name : _type_
        _description_
    coords : _type_
        _description_
    pulse_name : str, optional
        _description_, by default "on"
    duration : _type_, optional
        _description_, by default None
    amp : _type_, optional
        _description_, by default None
    delay : int, optional
        _description_, by default 0
    convert_to_Hz : bool, optional
        _description_, by default True
    """
    # Setup
    laser_el = get_laser_mod_element(laser_name)
    x_el = f"ao_{laser_name}_x"
    y_el = f"ao_{laser_name}_y"

    global _cache_pol_reps_ind

    buffer = get_widefield_operation_buffer()
    access_time = get_aod_access_time()

    if convert_to_Hz:
        coords = [int(el * 10**6) for el in coords]

    if amp is None:
        qua.play("continue", x_el)
        qua.play("continue", y_el)
    else:
        macro_run_aods(laser_names=[laser_name], aod_suffices=[pulse_name], amps=[amp])
    qua.update_frequency(x_el, coords[0])
    qua.update_frequency(y_el, coords[1])

    if True:
        # if laser_name != "laser_INTE_520":
        # Pulse the laser
        qua.wait(access_time + buffer + delay, laser_el)
        if duration is None:
            qua.play(pulse_name, laser_el)
        elif isinstance(duration, int) and duration == 0:
            pass
        else:
            qua.play(pulse_name, laser_el, duration=duration)
        qua.wait(buffer, laser_el)
    else:  # Green pulsed initialization with microwaves test MCC
        qua.wait(access_time, laser_el)
        with qua.for_(
            _cache_pol_reps_ind, 0, _cache_pol_reps_ind < 10, _cache_pol_reps_ind + 1
        ):
            macro_pi_pulse([0, 1])
            qua.align()
            qua.play(pulse_name, laser_el, duration=50)
            qua.wait(buffer, laser_el)


def _macro_scc_shelving(
    scc_coords_list,
    scc_duration_list,
    spin_flip_ind_list,
    uwave_ind_list,
    shelving_coords_list,
    exp_spin_flip=True,
):
    shelving_laser_name = tb.get_physical_laser_name(VirtualLaserKey.SHELVING)
    ion_laser_name = tb.get_physical_laser_name(VirtualLaserKey.SCC)
    laser_name_list = [shelving_laser_name, ion_laser_name]
    shelving_pulse_name = "shelving"
    ion_pulse_name = "scc"
    pulse_name_list = [shelving_pulse_name, ion_pulse_name]
    shelving_laser_dict = tb.get_virtual_laser_dict(VirtualLaserKey.SHELVING)
    shelving_pulse_duration = shelving_laser_dict["duration"]
    shelving_scc_gap_ns = 0
    scc_delay = convert_ns_to_cc(shelving_pulse_duration + shelving_scc_gap_ns)
    delays = [0, scc_delay]
    # duration_list = [None, ion_duration]

    macro_run_aods(laser_name_list, aod_suffices=pulse_name_list)

    # Unpack the coords and convert to Hz
    x_shelving_coords_list = [int(el[0] * 10**6) for el in shelving_coords_list]
    y_shelving_coords_list = [int(el[1] * 10**6) for el in shelving_coords_list]
    x_scc_coords_list = [int(el[0] * 10**6) for el in scc_coords_list]
    y_scc_coords_list = [int(el[1] * 10**6) for el in scc_coords_list]

    # These are declared in macro_run_aods
    global _cache_x_freq
    global _cache_y_freq
    global _cache_x_freq_2
    global _cache_y_freq_2
    freq_vars = (_cache_x_freq, _cache_y_freq, _cache_x_freq_2, _cache_y_freq_2)
    freq_lists = (
        x_shelving_coords_list,
        y_shelving_coords_list,
        x_scc_coords_list,
        y_scc_coords_list,
    )

    qua.align()
    with qua.for_each_(freq_vars, freq_lists):
        macro_multi_pulse(
            laser_name_list,
            ((_cache_x_freq, _cache_y_freq), (_cache_x_freq_2, _cache_y_freq_2)),
            pulse_name_list,
            # duration_list=duration_list,
            delays=delays,
            convert_to_Hz=False,
        )


def _macro_scc_no_shelving(
    coords_list,
    duration_list=None,
    duration_override=None,
    amp_list=None,
    amp_override=None,
    exp_spin_flip_ind_list=None,
    uwave_ind_list=None,
    exp_spin_flip=True,
    ref_spin_flip=False,
):
    # Basic setup

    ion_laser_name = tb.get_physical_laser_name(VirtualLaserKey.SCC)
    ion_pulse_name = "scc"
    macro_run_aods([ion_laser_name], aod_suffices=[ion_pulse_name])

    if exp_spin_flip_ind_list is None:
        exp_spin_flip_ind_list = []

    num_nvs = len(coords_list)
    first_coords_list = [
        coords_list[ind] for ind in range(num_nvs) if ind not in exp_spin_flip_ind_list
    ]
    first_duration_list = [
        duration_list[ind]
        for ind in range(num_nvs)
        if ind not in exp_spin_flip_ind_list
    ]
    first_amp_list = [
        amp_list[ind] for ind in range(num_nvs) if ind not in exp_spin_flip_ind_list
    ]

    # Actual commands

    if ref_spin_flip:
        macro_pi_pulse(uwave_ind_list)

    # MCC antiphase by orientation
    # if exp_spin_flip:
    #     macro_pi_pulse(uwave_ind_list[:1])

    _macro_pulse_series(
        ion_laser_name,
        ion_pulse_name,
        first_coords_list,
        duration_list=first_duration_list,
        duration_override=duration_override,
        amp_list=first_amp_list,
        amp_override=amp_override,
    )

    # Just exit here if all NVs are SCC'ed in the first batch
    if len(exp_spin_flip_ind_list) == 0:
        return

    second_coords_list = [
        coords_list[ind] for ind in range(num_nvs) if ind in exp_spin_flip_ind_list
    ]
    second_duration_list = [
        duration_list[ind] for ind in range(num_nvs) if ind in exp_spin_flip_ind_list
    ]
    second_amp_list = [
        amp_list[ind] for ind in range(num_nvs) if ind in exp_spin_flip_ind_list
    ]

    if exp_spin_flip:
        macro_pi_pulse(uwave_ind_list)

    _macro_pulse_series(
        ion_laser_name,
        ion_pulse_name,
        second_coords_list,
        duration_list=second_duration_list,
        duration_override=duration_override,
        amp_list=second_amp_list,
        amp_override=amp_override,
    )


# endregion
# region Getters and utils


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


def get_macro_pi_pulse_duration(uwave_ind_list):
    duration = 0
    uwave_buffer = get_uwave_buffer()
    for uwave_ind in uwave_ind_list:
        duration += get_rabi_period(uwave_ind) // 2
        duration += uwave_buffer
    return duration


def get_macro_pi_on_2_pulse_duration(uwave_ind_list):
    duration = 0
    uwave_buffer = get_uwave_buffer()
    for uwave_ind in uwave_ind_list:
        duration += get_rabi_period(uwave_ind) // 4
        duration += uwave_buffer
    return duration


@cache
def get_default_charge_readout_duration():
    readout_laser_dict = tb.get_virtual_laser_dict(
        VirtualLaserKey.WIDEFIELD_CHARGE_READOUT
    )
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
def get_uwave_buffer():
    return get_common_duration_cc("uwave_buffer")


@cache
def get_common_duration_cc(key):
    common_duration_ns = tb.get_common_duration(key)
    common_duration_cc = convert_ns_to_cc(common_duration_ns)
    return common_duration_cc


@cache
def get_laser_mod_element(laser_name, sticky=False):
    physical_laser_dict = tb.get_physical_laser_dict(laser_name)
    mod_mode = physical_laser_dict["mod_mode"]
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
    virtual_sig_gen_dict = tb.get_virtual_sig_gen_dict(uwave_ind)
    sig_gen_name = virtual_sig_gen_dict["physical_name"]
    sig_gen_element = f"do_{sig_gen_name}_dm"
    return sig_gen_element


@cache
def get_iq_mod_elements(uwave_ind=0):
    virtual_sig_gen_dict = tb.get_virtual_sig_gen_dict(uwave_ind)
    sig_gen_name = virtual_sig_gen_dict["physical_name"]
    i_el = f"ao_{sig_gen_name}_i"
    q_el = f"ao_{sig_gen_name}_q"
    return i_el, q_el


@cache
def get_rabi_period(uwave_ind=0):
    virtual_sig_gen_dict = tb.get_virtual_sig_gen_dict(uwave_ind)
    rabi_period_ns = virtual_sig_gen_dict["rabi_period"]
    rabi_period = convert_ns_to_cc(rabi_period_ns)
    return rabi_period


# endregion

if __name__ == "__main__":
    config = common.get_config_dict("purcell")
    positioners_dict = config["Positioning"]["Positioners"]
    positioners_keys = positioners_dict.keys()
    laser_names = []
    for key in positioners_keys:
        positioner = positioners_dict[key]
        if "aod" in positioner and positioner["aod"]:
            virtual_laser_key = positioner["opti_virtual_laser_key"]
            physical_laser_name = tb.get_physical_laser_name(virtual_laser_key)
            laser_names.append(physical_laser_name)
    print(laser_names)
