# -*- coding: utf-8 -*-
"""This file contains functions, classes, and other objects that are useful
in a variety of contexts. Since they are expected to be used in many
files, I put them all in one place so that they don't have to be redefined
in each file.

Created on November 23rd, 2018

@author: mccambria
"""

# region Imports and constants

import matplotlib.pyplot as plt
import os
import csv
import datetime
import numpy as np
from numpy import exp
import json
import time
import labrad
from git import Repo
from pathlib import Path, PurePath
from enum import Enum, IntEnum, auto
import socket
import smtplib
from email.mime.text import MIMEText
import traceback
import keyring
import math
import utils.common as common
import utils.search_index as search_index
import signal
import copy


class States(Enum):
    LOW = auto()
    ZERO = auto()
    HIGH = auto()


# Normalization style for comparing experimental data to reference data
class NormStyle(Enum):
    SINGLE_VALUED = auto()  # Use a single-valued reference
    POINT_TO_POINT = auto()  # Normalize each signal point by its own reference


class ModTypes(Enum):
    DIGITAL = auto()
    ANALOG = auto()


class Digital(IntEnum):
    LOW = 0
    HIGH = 1


Boltzmann = 8.617e-2  # meV / K

# endregion
# region Laser utils


def get_mod_type(laser_name):
    with labrad.connect() as cxn:
        mod_type = common.get_registry_entry(
            cxn, "mod_type", ["", "Config", "Optics", laser_name]
        )
    mod_type = eval(mod_type)
    return mod_type.name


def laser_off(cxn, laser_name):
    laser_switch_sub(cxn, False, laser_name)


def laser_on(cxn, laser_name, laser_power=None):
    laser_switch_sub(cxn, True, laser_name, laser_power)


def get_opx_laser_pulse_info(config, laser_name, laser_power):

    mod_type = config["Optics"][laser_name]["mod_type"]
    laser_delay = config["Optics"][laser_name]["delay"]

    laser_pulse_name = "laser_ON_{}".format(eval(mod_type).name)

    if eval(mod_type).name == "ANALOG":
        laser_pulse_amplitude = laser_power

    elif eval(mod_type).name == "DIGITAL":
        laser_pulse_amplitude = 1

    return laser_pulse_name, laser_delay, laser_pulse_amplitude


def laser_switch_sub(cxn, turn_on, laser_name, laser_power=None):

    mod_type = common.get_registry_entry(
        cxn, "mod_type", ["", "Config", "Optics", laser_name]
    )
    mod_type = eval(mod_type)

    if mod_type is ModTypes.DIGITAL:
        if turn_on:
            laser_chan = common.get_registry_entry(
                cxn,
                "do_{}_dm".format(laser_name),
                ["", "Config", "Wiring", "PulseStreamer"],
            )
            cxn.pulse_streamer.constant([laser_chan])
    elif mod_type is ModTypes.ANALOG:
        if turn_on:
            laser_chan = common.get_registry_entry(
                cxn,
                "do_{}_dm".format(laser_name),
                ["", "Config", "Wiring", "PulseStreamer"],
            )
            if laser_chan == 0:
                cxn.pulse_streamer.constant([], 0.0, laser_power)
            elif laser_chan == 1:
                cxn.pulse_streamer.constant([], laser_power, 0.0)

    # If we're turning things off, turn everything off. If we wanted to really
    # do this nicely we'd find a way to only turn off the specific channel,
    # but it's not worth the effort.
    if not turn_on:
        pulse_gen_server = get_server_pulse_gen(cxn)
        pulse_gen_server.constant([])


def set_laser_power(
    cxn, nv_sig=None, laser_key=None, laser_name=None, laser_power=None
):
    """Set a laser power, or return it for analog modulation.
    Specify either a laser_key/nv_sig or a laser_name/laser_power.
    """

    if (nv_sig is not None) and (laser_key is not None):
        laser_name = nv_sig[laser_key]
        power_key = "{}_power".format(laser_key)
        # If the power isn't specified, then we assume it's set some other way
        if power_key in nv_sig:
            laser_power = nv_sig[power_key]
    elif (laser_name is not None) and (laser_power is not None):
        pass  # All good
    else:
        raise Exception(
            "Specify either a laser_key/nv_sig or a laser_name/laser_power."
        )

    # If the power is controlled by analog modulation, we'll need to pass it
    # to the pulse streamer
    mod_type = common.get_registry_entry(
        cxn, "mod_type", ["", "Config", "Optics", laser_name]
    )
    mod_type = eval(mod_type)
    if mod_type == ModTypes.ANALOG:
        return laser_power
    else:
        laser_server = get_filter_server(cxn, laser_name)
        if (laser_power is not None) and (laser_server is not None):
            laser_server.set_laser_power(laser_power)
        return None


def set_filter(cxn, nv_sig=None, optics_key=None, optics_name=None, filter_name=None):
    """optics_key should be either 'collection' or a laser key.
    Specify either an optics_key/nv_sig or an optics_name/filter_name.
    """

    if (nv_sig is not None) and (optics_key is not None):
        if optics_key in nv_sig:
            optics_name = nv_sig[optics_key]
        else:
            optics_name = optics_key
        filter_key = "{}_filter".format(optics_key)
        # Just exit if there's no filter specified in the nv_sig
        if filter_key not in nv_sig:
            return
        filter_name = nv_sig[filter_key]
    elif (optics_name is not None) and (filter_name is not None):
        pass  # All good
    else:
        raise Exception(
            "Specify either an optics_key/nv_sig or an" " optics_name/filter_name."
        )

    filter_server = get_filter_server(cxn, optics_name)
    if filter_server is None:
        return
    pos = common.get_registry_entry(
        cxn,
        filter_name,
        ["", "Config", "Optics", optics_name, "FilterMapping"],
    )
    filter_server.set_filter(pos)


def get_filter_server(cxn, optics_name):
    """Try to get a filter server. If there isn't one listed on the registry,
    just return None.
    """
    try:
        server_name = common.get_registry_entry(
            cxn, "filter_server", ["", "Config", "Optics", optics_name]
        )
        return getattr(cxn, server_name)
    except Exception:
        return None


def get_laser_server(cxn, laser_name):
    """Try to get a laser server. If there isn't one listed on the registry,
    just return None.
    """
    try:
        server_name = common.get_registry_entry(
            cxn, "laser_server", ["", "Config", "Optics", laser_name]
        )
        return getattr(cxn, server_name)
    except Exception:
        return None


def process_laser_seq(pulse_streamer, seq, config, laser_name, laser_power, train):
    """Some lasers may require special processing of their Pulse Streamer
    sequence. For example, the Cobolt lasers expect 3.5 V for digital
    modulation, but the Pulse Streamer only supplies 2.6 V.
    """

    pulser_wiring = config["Wiring"]["PulseGen"]
    # print(config)
    mod_type = config["Optics"][laser_name]["mod_type"]
    mod_type = eval(mod_type)

    processed_train = []

    if mod_type is ModTypes.DIGITAL:
        processed_train = train.copy()
        pulser_laser_mod = pulser_wiring["do_{}_dm".format(laser_name)]
        seq.setDigital(pulser_laser_mod, processed_train)

    # Analog, convert LOW / HIGH to 0.0 / analog voltage
    # currently can't handle multiple powers of the AM within the same sequence
    # Possibly, we could pass laser_power as a list, and then build the sequences
    # for each power (element) in the list.
    elif mod_type is ModTypes.ANALOG:
        high_count = 0
        for el in train:
            dur = el[0]
            val = el[1]
            if type(laser_power) == list:
                if val == 0:
                    power_dict = {Digital.LOW: 0.0}
                else:
                    power_dict = {Digital.HIGH: laser_power[high_count]}
                    if val == Digital.HIGH:
                        high_count += 1
            # If a list wasn't passed, just use the single value for laser_power
            elif type(laser_power) != list:
                power_dict = {Digital.LOW: 0.0, Digital.HIGH: laser_power}
            processed_train.append((dur, power_dict[val]))

        pulser_laser_mod = pulser_wiring["ao_{}_am".format(laser_name)]
        # print(processed_train)
        seq.setAnalog(pulser_laser_mod, processed_train)


# endregion
# region Pulse generator utils


def set_delays_to_zero(config):
    """Pass this a config dictionary and it'll set all the delays to zero.
    Useful for testing sequences without having to worry about delays.
    """
    for key in config:
        # Check if any entries are delays and set them to 0
        if key.endswith("delay"):
            config[key] = 0
            return
        # Check if we're at a sublevel - if so, recursively set its delay to 0
        val = config[key]
        if type(val) is dict:
            set_delays_to_zero(val)


def set_delays_to_sixteen(config):
    """Pass this a config dictionary and it'll set all the delays to 16ns,
    which is the minimum wait() time for the OPX. Useful for testing
    sequences without having to worry about delays.
    """
    for key in config:
        # Check if any entries are delays and set them to 0
        if key.endswith("delay"):
            config[key] = 16
            return
        # Check if we're at a sublevel - if so, recursively set its delay to 0
        val = config[key]
        if type(val) is dict:
            set_delays_to_sixteen(val)


def seq_train_length_check(train):
    """Print out the length of a the sequence train for a specific channel.
    Useful for debugging sequences
    """
    total = 0
    for el in train:
        total += el[0]
    print(total)


def encode_seq_args(seq_args):
    # Recast np ints to Python ints so json knows what to do
    for ind in range(len(seq_args)):
        el = seq_args[ind]
        if type(el) is np.int32:
            seq_args[ind] = int(el)
    return json.dumps(seq_args)


def decode_seq_args(seq_args_string):
    if seq_args_string == "":
        return []
    else:
        return json.loads(seq_args_string)


def get_pulse_streamer_wiring(cxn):
    config = get_config_dict(cxn)
    pulse_streamer_wiring = config["Wiring"]["PulseStreamer"]
    return pulse_streamer_wiring


def get_tagger_wiring(cxn):
    cxn.registry.cd(["", "Config", "Wiring", "Tagger"])
    sub_folders, keys = cxn.registry.dir()
    if keys == []:
        return {}
    p = cxn.registry.packet()
    for key in keys:
        p.get(key, key=key)  # Return as a dictionary
    wiring = p.send()
    tagger_wiring = {}
    for key in keys:
        tagger_wiring[key] = wiring[key]
    return tagger_wiring


# endregion
# region Math functions


def get_pi_pulse_dur(rabi_period):
    return round(rabi_period / 2)


def get_pi_on_2_pulse_dur(rabi_period):
    return round(rabi_period / 4)


def iq_comps(phase, amp):
    """Given the phase and amplitude of the IQ vector, calculate the I (real) and
    Q (imaginary) components
    """
    if type(phase) is list:
        ret_vals = []
        for val in phase:
            ret_vals.append(np.round(amp * np.exp((0 + 1j) * val), 5))
        return (np.real(ret_vals).tolist(), np.imag(ret_vals).tolist())
    else:
        ret_val = np.round(amp * np.exp((0 + 1j) * phase), 5)
        return (np.real(ret_val), np.imag(ret_val))


def lorentzian(x, x0, A, L, offset):
    """Calculates the value of a lorentzian for the given input and parameters

    Params:
        x: float
            Input value
        params: tuple
            The parameters that define the lorentzian
            0: x0, mean postiion in x
            1: A, amplitude of curve
            2: L, related to width of curve
            3: offset, constant y value offset
    """

    x_center = x - x0
    return offset + A * 0.5 * L / (x_center**2 + (0.5 * L) ** 2)


def exp_decay(x, amp, decay, offset):
    return offset + amp * np.exp(-x / decay)


def exp_stretch_decay(x, amp, decay, offset, B):
    return offset + amp * np.exp(-((x / decay) ** B))


def gaussian(x, *params):
    """Calculates the value of a gaussian for the given input and parameters

    Params:
        x: float
            Input value
        params: tuple
            The parameters that define the Gaussian
            0: coefficient that defines the peak height
            1: mean, defines the center of the Gaussian
            2: standard deviation, defines the width of the Gaussian
            3: constant y value to account for background
    """

    coeff, mean, stdev, offset = params
    var = stdev**2  # variance
    centDist = x - mean  # distance from the center
    return offset + coeff**2 * np.exp(-(centDist**2) / (2 * var))


def sinexp(t, offset, amp, freq, decay):
    two_pi = 2 * np.pi
    half_pi = np.pi / 2
    return offset + (amp * np.sin((two_pi * freq * t) + half_pi)) * exp(
        -(decay**2) * t
    )


def cosexp(t, offset, amp, freq, decay):
    two_pi = 2 * np.pi
    return offset + (np.exp(-t / abs(decay)) * abs(amp) * np.cos((two_pi * freq * t)))


def cosexp_1_at_0(t, offset, freq, decay):
    two_pi = 2 * np.pi
    amp = 1 - offset
    return offset + (np.exp(-t / abs(decay)) * abs(amp) * np.cos((two_pi * freq * t)))


def sin_1_at_0_phase(t, amp, offset, freq, phase):
    two_pi = 2 * np.pi
    # amp = 1 - offset
    return offset + (abs(amp) * np.sin((freq * t - np.pi / 2 + phase)))


def cosine_sum(t, offset, decay, amp_1, freq_1, amp_2, freq_2, amp_3, freq_3):
    two_pi = 2 * np.pi

    return offset + np.exp(-t / abs(decay)) * (
        amp_1 * np.cos(two_pi * freq_1 * t)
        + amp_2 * np.cos(two_pi * freq_2 * t)
        + amp_3 * np.cos(two_pi * freq_3 * t)
    )


def cosine_one(t, offset, decay, amp_1, freq_1):
    two_pi = 2 * np.pi

    return offset + np.exp(-t / abs(decay)) * (amp_1 * np.cos(two_pi * freq_1 * t))


def t2_func(t, amplitude, offset, t2):
    n = 3
    return amplitude * np.exp(-((t / t2) ** n)) + offset


def calc_snr(sig_count, ref_count):
    """Take a list of signal and reference counts, and take their average,
    then calculate a snr.
    inputs:
        sig_count = list
        ref_counts = list
    outputs:
        snr = list
    """

    sig_count_avg = np.average(sig_count)
    ref_count_avg = np.average(ref_count)
    dif = sig_count_avg - ref_count_avg
    sum_ = sig_count_avg + ref_count_avg
    noise = np.sqrt(sig_count_avg)
    snr = dif / noise

    return snr


def get_scan_vals(center, scan_range, num_steps, dtype=float):
    """
    Returns a linspace for a scan centered about specified point
    """

    half_scan_range = scan_range / 2
    low = center - half_scan_range
    high = center + half_scan_range
    scan_vals = np.linspace(low, high, num_steps, dtype=dtype)
    # Deduplicate - may be necessary for ints and low scan ranges
    scan_vals = np.unique(scan_vals)
    return scan_vals


def bose(energy, temp):
    """Calculate Bose Einstein occupation number for a mode with given energy
    (meV) at given temperature (K)"""
    # For very low temps we can get divide by zero and overflow warnings.
    # Fortunately, numpy is smart enough to know what we mean when this
    # happens, so let's let numpy figure it out and suppress the warnings.
    old_settings = np.seterr(divide="ignore", over="ignore")
    # print(energy / (Boltzmann * temp))
    val = 1 / (np.exp(energy / (Boltzmann * temp)) - 1)
    # Return error handling to default state for other functions
    np.seterr(**old_settings)
    return val


# endregion
# region LabRAD registry utils
# Core registry functions in Common


def get_config_dict(cxn=None):
    """Get the whole config from the registry as a dictionary"""
    if cxn is None:
        with labrad.connect() as cxn:
            return get_config_dict_sub(cxn)
    else:
        return get_config_dict_sub(cxn)


def get_config_dict_sub(cxn):
    config_dict = {}
    populate_config_dict(cxn, ["", "Config"], config_dict)
    return config_dict


def populate_config_dict(cxn, reg_path, dict_to_populate):
    """Populate the config dictionary recursively"""

    # Sub-folders
    cxn.registry.cd(reg_path)
    sub_folders, keys = cxn.registry.dir()
    for el in sub_folders:
        sub_dict = {}
        sub_path = reg_path + [el]
        populate_config_dict(cxn, sub_path, sub_dict)
        dict_to_populate[el] = sub_dict

    # Keys
    if len(keys) == 1:
        cxn.registry.cd(reg_path)
        p = cxn.registry.packet()
        key = keys[0]
        p.get(key)
        val = p.send()["get"]
        if type(val) == np.ndarray:
            val = val.tolist()
        dict_to_populate[key] = val

    elif len(keys) > 1:
        cxn.registry.cd(reg_path)
        p = cxn.registry.packet()
        for key in keys:
            p.get(key)
        vals = p.send()["get"]

        for ind in range(len(keys)):
            key = keys[ind]
            val = vals[ind]
            if type(val) == np.ndarray:
                val = val.tolist()
            dict_to_populate[key] = val


def get_apd_indices(cxn):
    "Get a list of the APD indices in use from the registry"
    return common.get_registry_entry(cxn, "apd_indices", [""])


# endregion
# region Server getters
"""Each getter looks up the requested server from the registry and 
returns a usable reference to the requested server (i.e. cxn.<server>)
"""


def get_server_pulse_gen(cxn):
    """Get the pulse gen server for this setup, e.g. opx or swabian"""
    return common.get_server(cxn, "pulse_gen")


def get_server_awg(cxn):
    """Get the arbitrary waveform generator server for this setup, e.g. opx or keysight"""
    return common.get_server(cxn, "awg")


def get_server_counter(cxn):
    """Get the photon counter server for this setup, e.g. opx or swabian"""
    return common.get_server(cxn, "counter")


def get_server_tagger(cxn):
    """Get the photon time tagger server for this setup, e.g. opx or swabian"""
    return common.get_server(cxn, "tagger")


def get_server_temp_controller(cxn):
    return common.get_server(cxn, "temp_controller")


def get_server_temp_monitor(cxn):
    return common.get_server(cxn, "temp_monitor")


def get_server_sig_gen(cxn, state):
    """Get the signal generator that controls transitions to the specified NV state"""
    return common.get_server(cxn, f"sig_gen_{state.name}")


# endregion
# region File and data handling utils


def get_raw_data(
    file_name,
    path_from_nvdata=None,
    nvdata_dir=None,
):
    """Returns a dictionary containing the json object from the specified
    raw data file. If path_from_nvdata is not specified, we assume we're
    looking for an autogenerated experiment data file. In this case we'll
    use glob (a pattern matching module for pathnames) to efficiently find
    the file based on the known structure of the directories rooted from
    nvdata_dir (ie nvdata_dir / pc_folder / routine / year_month / file.txt)
    """
    file_path = get_raw_data_path(file_name, path_from_nvdata, nvdata_dir)
    with file_path.open() as f:
        res = json.load(f)
        return res


def get_raw_data_path(
    file_name,
    path_from_nvdata=None,
    nvdata_dir=None,
):
    """Same as get_raw_data, but just returns the path to the file"""
    if nvdata_dir is None:
        nvdata_dir = common.get_nvdata_dir()
    if path_from_nvdata is None:
        path_from_nvdata = search_index.get_data_path(file_name)
    data_dir = nvdata_dir / path_from_nvdata
    file_name_ext = "{}.txt".format(file_name)
    file_path = data_dir / file_name_ext
    return file_path


def get_branch_name():
    """Return the name of the active branch of kolkowitz-nv-experiment-v1.0"""
    home_to_repo = PurePath("Documents/GitHub/kolkowitz-nv-experiment-v1.0")
    repo_path = PurePath(Path.home()) / home_to_repo
    repo = Repo(repo_path)
    return repo.active_branch.name


def get_time_stamp():
    """Get a formatted timestamp for file names and metadata.

    Returns:
        string: <year>_<month>_<day>-<hour>_<minute>_<second>
    """

    timestamp = str(datetime.datetime.now())
    timestamp = timestamp.split(".")[0]  # Keep up to seconds
    timestamp = timestamp.replace(":", "_")  # Replace colon with dash
    timestamp = timestamp.replace("-", "_")  # Replace dash with underscore
    timestamp = timestamp.replace(" ", "-")  # Replace space with dash
    return timestamp


def get_folder_dir(source_name, subfolder):

    source_name = os.path.basename(source_name)
    source_name = os.path.splitext(source_name)[0]

    branch_name = get_branch_name()
    pc_name = socket.gethostname()

    nvdata_dir = common.get_nvdata_dir()
    joined_path = (
        nvdata_dir
        / "pc_{}".format(pc_name)
        / "branch_{}".format(branch_name)
        / source_name
    )

    if subfolder is not None:
        joined_path = os.path.join(joined_path, subfolder)

    folderDir = os.path.abspath(joined_path)

    # Make the required directory if it doesn't exist already
    if not os.path.isdir(folderDir):
        os.makedirs(folderDir)

    return folderDir


def get_files_in_folder(folderDir, filetype=None):
    """
    folderDir: str
        full file path, use previous function get_folder_dir
    filetype: str
        must be a 3-letter file extension, do NOT include the period. ex: 'txt'
    """
    file_list_temp = os.listdir(folderDir)
    if filetype:
        file_list = []
        for file in file_list_temp:
            if file[-3:] == filetype:
                file_list.append(file)
    else:
        file_list = file_list_temp

    return file_list


def get_file_path(source_name, time_stamp="", name="", subfolder=None):
    """Get the file path to save to. This will be in a subdirectory of nvdata.

    Params:
        source_name: string
            Source file name - alternatively, __file__ of the caller which will
            be parsed to get the name of the subdirectory we will write to
        time_stamp: string
            Formatted timestamp to include in the file name
        name: string
            The full file name consists of <timestamp>_<name>.<ext>
            Ext is supplied by the save functions
        subfolder: string
            Subfolder to save to under file name
    """

    date_folder_name = None  # Init to None
    # Set up the file name
    if (time_stamp != "") and (name != ""):
        fileName = "{}-{}".format(time_stamp, name)
        # locate the subfolder that matches the month and year when the data is taken
        date_folder_name = "_".join(time_stamp.split("_")[0:2])
    elif (time_stamp == "") and (name != ""):
        fileName = name
    elif (time_stamp != "") and (name == ""):
        fileName = "{}-{}".format(time_stamp, "untitled")
        date_folder_name = "_".join(time_stamp.split("_")[0:2])
    else:
        fileName = "{}-{}".format(get_time_stamp(), "untitled")

    # Create the subfolder combined name, if needed
    subfolder_name = None
    if (subfolder != None) and (date_folder_name != None):
        subfolder_name = str(date_folder_name + "/" + subfolder)
    elif (subfolder == None) and (date_folder_name != None):
        subfolder_name = date_folder_name

    folderDir = get_folder_dir(source_name, subfolder_name)
    fileDir = os.path.abspath(os.path.join(folderDir, fileName))

    file_path_Path = Path(fileDir)

    return file_path_Path


def utc_from_file_name(file_name):
    f_split = file_name.split("-")
    date = f_split[0]
    date_split = date.split("_")
    date_ints = [int(el) for el in date_split]
    time = f_split[1]
    time_split = time.split("_")
    time_ints = [int(el) for el in time_split]
    dt = datetime.datetime(*date_ints, *time_ints)
    utc_time = dt.replace(tzinfo=datetime.timezone.utc)
    utc_timestamp = utc_time.timestamp()
    return utc_timestamp


def save_figure(fig, file_path):
    """Save a matplotlib figure as a svg.

    Params:
        fig: matplotlib.figure.Figure
            The figure to save
        file_path: string
            The file path to save to including the file name, excluding the
            extension
    """

    fig.savefig(str(file_path.with_suffix(".svg")), dpi=300)


def save_raw_data(rawData, filePath):
    """Save raw data in the form of a dictionary to a text file. New lines
    will be printed between entries in the dictionary.

    Params:
        rawData: dict
            The raw data as a dictionary - will be saved via JSON
        filePath: string
            The file path to save to including the file name, excluding the
            extension
    """

    # Just to be safe, work with a copy of the raw data rather than the
    # raw data itself
    rawData = copy.deepcopy(rawData)

    file_path_ext = filePath.with_suffix(".txt")

    # Add in a few things that should always be saved here. In particular,
    # sharedparameters so we have as snapshot of the configuration and
    # nv_sig_units. If these have already been defined in the routine,
    # then they'll just be overwritten.
    try:
        rawData["config"] = get_config_dict()  # Include a snapshot of the config
    except Exception as e:
        print(e)

    # Casting for JSON compatibility
    nv_sig = rawData["nv_sig"]
    for key in nv_sig:
        if type(nv_sig[key]) == np.ndarray:
            nv_sig[key] = nv_sig[key].tolist()
        elif isinstance(nv_sig[key], Enum):
            nv_sig[key] = nv_sig[key].name

    with open(file_path_ext, "w") as file:
        json.dump(rawData, file, indent=2)

    if file_path_ext.match(search_index.search_index_glob):
        search_index.add_to_search_index(file_path_ext)


def save_combine_data(file_list, folder_list, py_file_name):
    """This routine takes any number of files and attempts to combine the data,
    then saves them in one array.

    Only use this for data that was collected under the same conditions. Works
    best for measurements that save data like Rabi, PESR, etc.

    It will throw an error if the num_steps of the data files do not match!

    py_file_name is the string of the name of the file, ex: 'rabi.py'
    """

    # do an initial check if the num_steps all match.If they do, we can add
    # the data together.
    num_steps_list = []
    for f in range(len(file_list)):
        file = file_list[f]
        folder = folder_list[f]
        data1 = get_raw_data(file, folder)
        num_steps = data1["num_steps"]
        num_steps_list.append(num_steps)
    # check that all num_steps of the files match
    num_steps_result = all(element == num_steps_list[0] for element in num_steps_list)

    if num_steps_result:
        # create initial empty arrays to add data to
        sig_counts = np.zeros([1, num_steps])
        ref_counts = np.zeros([1, num_steps])
        num_runs = 0

        for f in range(len(file_list)):
            file = file_list[f]
            folder = folder_list[f]
            data1 = get_raw_data(file, folder)

            sig_counts1 = np.array(data1["sig_counts"])
            ref_counts1 = np.array(data1["ref_counts"])
            nv_sig = data1["nv_sig"]
            num_runs1 = data1["num_runs"]

            sig_counts = np.concatenate((sig_counts, sig_counts1), axis=0)
            ref_counts = np.concatenate((ref_counts, ref_counts1), axis=0)
            num_runs += num_runs1

        # delete the first row of data that was a placeholder.
        sig_counts = sig_counts[1:]
        ref_counts = ref_counts[1:]

        # Calc the norm avg sig
        avg_sig_counts = np.average(sig_counts, axis=0)
        avg_ref_counts = np.average(ref_counts, axis=0)
        norm_avg_sig = avg_sig_counts / np.average(avg_ref_counts)

        timestamp = get_time_stamp()

        # take the dictionary from the last file, add the entry for
        # file_list and folder_list, and update:
        # sig counts
        # ref counts
        # norm_avg_sig
        # num_runs

        raw_data = data1
        raw_data["file_list"] = file_list
        raw_data["folder_list"] = folder_list

        raw_data["num_runs"] = num_runs
        raw_data["sig_counts"] = sig_counts.tolist()
        raw_data["ref_counts"] = ref_counts.tolist()
        raw_data["norm_avg_sig"] = norm_avg_sig.tolist()

        nv_name = nv_sig["name"]
        file_path = get_file_path(py_file_name, timestamp, nv_name)
        save_raw_data(raw_data, file_path)


# endregion
# region Email utils


def send_exception_email(
    email_from=common.shared_email,
    email_to=common.shared_email,
):
    # format_exc extracts the stack and error message from
    # the exception currently being handled.
    now = time.localtime()
    date = time.strftime("%A, %B %d, %Y", now)
    timex = time.strftime("%I:%M:%S %p", now)
    exc_info = traceback.format_exc()
    content = f"An unhandled exception occurred on {date} at {timex}.\n{exc_info}"
    send_email(content, email_from=email_from, email_to=email_to)


def send_email(
    content,
    email_from=common.shared_email,
    email_to=common.shared_email,
):

    pc_name = socket.gethostname()
    msg = MIMEText(content)
    msg["Subject"] = f"Alert from {pc_name}"
    msg["From"] = email_from
    msg["To"] = email_to

    pw = keyring.get_password("system", email_from)

    server = smtplib.SMTP("smtp.gmail.com", 587)  # port 465 or 587
    server.ehlo()
    server.starttls()
    server.ehlo()
    server.login(email_from, pw)
    server.sendmail(email_from, email_to, msg.as_string())
    server.close()


# endregion
# region Miscellaneous (probably consider deprecated)


def get_dd_model_coeff_dict():
    # fmt: off
    dd_model_coeff_dict = {
        "1": [6, -8, 2],
        "2": [10, -8, -8, 8, -2],
        "4": [18, -8, -24, 8, 16, -8, -8, 8, -2],
        "8": [34, -8, -56, 8, 48, -8, -40, 8, 32, -8, -24, 8, 16, -8, -8, 8, -2],
    }
    # fmt: on

    return dd_model_coeff_dict


def x_y_image_grid(x_center, y_center, x_range, y_range, num_steps):

    if x_range != y_range:
        raise ValueError("x_range must equal y_range for now")

    x_num_steps = num_steps
    y_num_steps = num_steps

    # Force the scan to have square pixels by only applying num_steps
    # to the shorter axis
    half_x_range = x_range / 2
    half_y_range = y_range / 2

    x_low = x_center - half_x_range
    x_high = x_center + half_x_range
    y_low = y_center - half_y_range
    y_high = y_center + half_y_range

    # Apply scale and offset to get the voltages we'll apply to the galvo
    # Note that the polar/azimuthal angles, not the actual x/y positions
    # are linear in these voltages. For a small range, however, we don't
    # really care.
    x_voltages_1d = np.linspace(x_low, x_high, num_steps)
    y_voltages_1d = np.linspace(y_low, y_high, num_steps)

    ######### Works for any x_range, y_range #########

    # Winding cartesian product
    # The x values are repeated and the y values are mirrored and tiled
    # The comments below shows what happens for [1, 2, 3], [4, 5, 6]

    # [1, 2, 3] => [1, 2, 3, 3, 2, 1]
    x_inter = np.concatenate((x_voltages_1d, np.flipud(x_voltages_1d)))
    # [1, 2, 3, 3, 2, 1] => [1, 2, 3, 3, 2, 1, 1, 2, 3]
    if y_num_steps % 2 == 0:  # Even x size
        x_voltages = np.tile(x_inter, int(y_num_steps / 2))
    else:  # Odd x size
        x_voltages = np.tile(x_inter, int(np.floor(y_num_steps / 2)))
        x_voltages = np.concatenate((x_voltages, x_voltages_1d))

    # [4, 5, 6] => [4, 4, 4, 5, 5, 5, 6, 6, 6]
    y_voltages = np.repeat(y_voltages_1d, x_num_steps)

    voltages = np.vstack((x_voltages, y_voltages))

    return x_voltages, y_voltages


def write_csv(
    csv_data,
    file,
    folder_path,
    nvdata_dir="E:/Shared drives/Kolkowitz Lab Group/nvdata",
):
    with open(
        "{}/{}/{}.csv".format(nvdata_dir, folder_path, file), "w", newline=""
    ) as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=",", quoting=csv.QUOTE_NONE)
        csv_writer.writerows(csv_data)


def save_image_data_csv(
    img_array,
    x_voltages,
    y_voltages,
    file_path="pc_rabi/branch_master/image_sample",
    csv_file_name=None,
    timestamp=None,
    nvdata_dir="E:/Shared drives/Kolkowitz Lab Group/nvdata",
):

    csv_data = []
    for ind in range(len(img_array)):
        csv_data.append(img_array[ind])

    x_voltages.insert(0, "x_voltages")
    y_voltages.insert(0, "y_voltages")
    csv_data.append(x_voltages)
    csv_data.append(y_voltages)

    if not csv_file_name:
        if not timestamp:
            timestamp = get_time_stamp()
        csv_file_name = "{}".format(timestamp)

    with open(
        "{}/{}.csv".format(nvdata_dir + "/" + file_path, csv_file_name),
        "w",
        newline="",
    ) as csv_file:

        csv_writer = csv.writer(csv_file, delimiter=",", quoting=csv.QUOTE_NONE)
        csv_writer.writerows(csv_data)
    return


def opt_power_via_photodiode(color_ind, AO_power_settings=None, nd_filter=None):
    cxn = labrad.connect()
    optical_power_list = []
    if color_ind == 532:
        cxn.pulse_streamer.constant([3], 0.0, 0.0)  # Turn on the green laser
        time.sleep(0.3)
        for i in range(10):
            optical_power_list.append(cxn.photodiode.read_optical_power())
            time.sleep(0.01)
    elif color_ind == 589:
        cxn.filter_slider_ell9k.set_filter(
            nd_filter
        )  # Change the nd filter for the yellow laser
        cxn.pulse_streamer.constant(
            [], 0.0, AO_power_settings
        )  # Turn on the yellow laser
        time.sleep(0.3)
        for i in range(10):
            optical_power_list.append(cxn.photodiode.read_optical_power())
            time.sleep(0.01)
    elif color_ind == 638:
        cxn.pulse_streamer.constant([7], 0.0, 0.0)  # Turn on the red laser
        time.sleep(0.3)
        for i in range(10):
            optical_power_list.append(cxn.photodiode.read_optical_power())
            time.sleep(0.01)

    optical_power = np.average(optical_power_list)
    time.sleep(0.1)
    cxn.pulse_streamer.constant([], 0.0, 0.0)
    return optical_power


def calc_optical_power_mW(color_ind, optical_power_V):
    # Values found from experiments. See Notebook entry 3/19/2020 and 3/20/2020
    if color_ind == 532:
        return 11.84 * optical_power_V + 0.0493
    elif color_ind == 589:
        return 13.41 * optical_power_V + 0.06
    if color_ind == 638:
        return 4.14 * optical_power_V + 0.0492


def measure_g_r_y_power(aom_ao_589_pwr, nd_filter):
    green_optical_power_pd = opt_power_via_photodiode(532)

    red_optical_power_pd = opt_power_via_photodiode(638)

    yellow_optical_power_pd = opt_power_via_photodiode(
        589, AO_power_settings=aom_ao_589_pwr, nd_filter=nd_filter
    )

    # Convert V to mW optical power
    green_optical_power_mW = calc_optical_power_mW(532, green_optical_power_pd)

    red_optical_power_mW = calc_optical_power_mW(638, red_optical_power_pd)

    yellow_optical_power_mW = calc_optical_power_mW(589, yellow_optical_power_pd)

    return (
        green_optical_power_pd,
        green_optical_power_mW,
        red_optical_power_pd,
        red_optical_power_mW,
        yellow_optical_power_pd,
        yellow_optical_power_mW,
    )


# endregion
# region Rounding
"""Make sure to check the end results produced by this code! Rounding turns out
to be a nontrivial problem...
"""


def round_sig_figs(val, num_sig_figs):
    if val == 0:
        return 0
    func = lambda val, num_sig_figs: round(
        val, -int(math.floor(math.log10(abs(val))) - num_sig_figs + 1)
    )
    if type(val) is list:
        return [func(el, num_sig_figs) for el in val]
    elif type(val) is np.ndarray:
        val_list = val.tolist()
        rounded_val_list = [func(el, num_sig_figs) for el in val_list]
        return np.array(rounded_val_list)
    else:
        return func(val, num_sig_figs)


def presentation_round(val, err):
    if val == 0:
        return [0, None, None]
    err_mag = math.floor(math.log10(err))
    sci_err = err / (10**err_mag)
    first_err_digit = int(str(sci_err)[0])
    if first_err_digit == 1:
        err_sig_figs = 2
    else:
        err_sig_figs = 1
    power_of_10 = math.floor(math.log10(abs(val)))
    mag = 10**power_of_10
    rounded_err = round_sig_figs(err, err_sig_figs) / mag
    rounded_val = round(val / mag, (power_of_10 - err_mag) + err_sig_figs - 1)
    return [rounded_val, rounded_err, power_of_10]


def presentation_round_latex(val, err):
    if val == 0:
        return "0"
    # if val <= 0 or err > val:
    #     return ""
    rounded_val, rounded_err, power_of_10 = presentation_round(val, err)
    err_mag = math.floor(math.log10(rounded_err))
    val_mag = math.floor(math.log10(abs(rounded_val)))

    # Turn 0.0000016 into 0.16
    # The round is to deal with floating point leftovers eg 9 = 9.00000002
    shifted_rounded_err = round(rounded_err / 10 ** (err_mag + 1), 5)
    # - 1 to remove the "0." part
    err_last_decimal_mag = len(str(shifted_rounded_err)) - 2
    pad_val_to = -err_mag + err_last_decimal_mag

    if err_mag > val_mag:
        return 1 / 0
    elif err_mag == val_mag:
        print_err = rounded_err
    else:
        print_err = int(str(shifted_rounded_err).replace(".", ""))

    str_val = str(rounded_val)
    decimal_pos = str_val.find(".")
    num_padding_zeros = pad_val_to - len(str_val[decimal_pos:])
    padded_val = str(rounded_val) + "0" * num_padding_zeros
    # return "{}({})e{}".format(padded_val, print_err, power_of_10)
    return r"\num{{{}({})e{}}}".format(padded_val, print_err, power_of_10)


# endregion
# region Safe Stop
"""Use this to safely stop experiments without risking data loss or weird state.
Works by reassigning CTRL + C to set a global variable rather than raise a
KeyboardInterrupt exception. That way we can check on the global variable
whenever we like and stop the experiment appropriately. It's up to you (the
routine author) to place this in your routine appropriately.
"""


def init_safe_stop():
    """Call this at the beginning of a loop or other section which you may
    want to interrupt
    """
    global SAFESTOPFLAG
    # Tell the user safe stop has started if it was stopped or just not started
    try:
        if SAFESTOPFLAG:
            print("\nPress CTRL + C to stop...\n")
    except Exception as exc:
        print("\nPress CTRL + C to stop...\n")
    SAFESTOPFLAG = False
    signal.signal(signal.SIGINT, safe_stop_handler)
    return


def safe_stop_handler(sig, frame):
    """This should never need to be called directly"""
    global SAFESTOPFLAG
    SAFESTOPFLAG = True


def safe_stop():
    """Call this to check whether the user asked us to stop"""
    global SAFESTOPFLAG
    time.sleep(0.1)  # Pause execution to allow safe_stop_handler to run
    return SAFESTOPFLAG


def reset_safe_stop():
    """Reset the Safe Stop flag, but don't remove the handler in case we
    want to reuse it.
    """
    global SAFESTOPFLAG
    SAFESTOPFLAG = False


def poll_safe_stop():
    """Blocking version of safe stop"""
    init_safe_stop()
    while not safe_stop():
        time.sleep(0.1)


# endregion
# region Reset hardware


def reset_cfm(cxn=None):
    """Reset our cfm so that it's ready to go for a new experiment. Avoids
    unnecessarily resetting components that may suffer hysteresis (ie the
    components that control xyz since these need to be reset in any
    routine where they matter anyway).
    """
    if cxn is None:
        with labrad.connect() as cxn:
            reset_cfm_with_cxn(cxn)
    else:
        reset_cfm_with_cxn(cxn)


def reset_cfm_with_cxn(cxn):
    cxn_server_names = cxn.servers
    for name in cxn_server_names:
        server = cxn[name]
        # Check for servers that ask not to be reset automatically
        if hasattr(server, "reset_cfm_opt_out"):
            continue
        if hasattr(server, "reset"):
            server.reset()


# endregion
