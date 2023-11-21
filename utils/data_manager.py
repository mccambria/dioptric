# -*- coding: utf-8 -*-
"""
Tools for managing our experimental database

Created November 15th, 2023

@author: mccambria
"""

# region Imports and constants

from io import BytesIO
from datetime import datetime
from utils import common
from utils import _cloud
from pathlib import Path
import json
from git import Repo
from enum import Enum
import numpy as np
import socket
import labrad
import copy
from utils.constants import *  # Star import is bad practice, but useful here for json deescape

data_manager_folder = common.get_data_manager_folder()


# endregion
# region Save functions


def get_time_stamp():
    """Get a formatted timestamp for file names and metadata.

    Returns:
        string: <year>_<month>_<day>-<hour>_<minute>_<second>
    """

    timestamp = str(datetime.now())
    timestamp = timestamp.split(".")[0]  # Keep up to seconds
    timestamp = timestamp.replace(":", "_")  # Replace colon with dash
    timestamp = timestamp.replace("-", "_")  # Replace dash with underscore
    timestamp = timestamp.replace(" ", "-")  # Replace space with dash
    return timestamp


def get_file_path(source_file, time_stamp, name, subfolder=None):
    """Get the file path to save to. This will be in a subdirectory of nvdata

    Parameters
    ----------
    source_file : string
        Source __file__ of the caller which will be parsed to get the
        name of the subdirectory we will write to
    time_stamp : string
        Formatted timestamp to include in the file name
    name : string
        The full file name consists of <timestamp>_<name>.<ext>
        Ext is supplied by the save functions
    subfolder : string, optional
        Subfolder to save to under file name, by default None

    Returns
    -------
    Path
        Path to save to
    """

    pc_name = socket.gethostname()
    branch_name = _get_branch_name()
    source_name = Path(source_file).stem
    date_folder = "_".join(time_stamp.split("_")[0:2])  # yyyy_mm

    folder_path = Path(f"pc_{pc_name}/branch_{branch_name}/{source_name}/{date_folder}")

    if subfolder is not None:
        folder_path = folder_path / subfolder

    file_name = f"{time_stamp}-{name}"

    return folder_path / file_name


def save_figure(fig, file_path):
    """Save a matplotlib figure as a svg.

    Params:
        fig: matplotlib.figure.Figure
            The figure to save
        file_path: string
            The file path to save to including the file name, excluding the
            extension
    """

    # Save locally
    file_path_svg = file_path.with_suffix(".svg")
    file_name = file_path_svg.name
    temp_file_path = data_manager_folder / file_name
    fig.savefig(str(temp_file_path), dpi=300)

    # Upload to cloud
    folder_path = file_path_svg.parent
    _cloud.upload(folder_path, temp_file_path)


def save_raw_data(raw_data, file_path, keys_to_compress=None):
    """Save raw data in the form of a dictionary to a text file. New lines
    will be printed between entries in the dictionary.

    Params:
        raw_data: dict
            The raw data as a dictionary - will be saved via JSON
        file_path: string
            The file path to save to including the file name, excluding the
            extension
        keys_to_compress: list(string)
            Keys to values in raw_data that we want to extract and save to
            a separate compressed file. Currently supports numpy arrays
    """

    file_path_txt = file_path.with_suffix(".txt")
    file_name = file_path_txt.name
    temp_file_path_txt = data_manager_folder / file_name

    # Work with a copy of the raw data to avoid mutation
    raw_data = copy.deepcopy(raw_data)

    # Compress numpy arrays to linked file
    temp_file_path_npz = None
    if keys_to_compress is not None:
        temp_file_path_npz = temp_file_path_txt.with_suffix(".npz")
        kwargs = {}
        for key in keys_to_compress:
            kwargs[key] = raw_data[key]
            # Replace the value in the data with .npz to indicate that the array has been compressed
            raw_data[key] = ".npz"
        with open(temp_file_path_npz, "wb") as f:
            np.savez_compressed(f, **kwargs)

    # Always include the config
    config = common.get_config_dict()
    config = copy.deepcopy(config)
    raw_data["config"] = config

    _json_escape(raw_data)

    with open(temp_file_path_txt, "w") as f:
        json.dump(raw_data, f, indent=2)

    # Upload to cloud
    folder_path = file_path_txt.parent
    _cloud.upload(folder_path, temp_file_path_txt)
    if temp_file_path_npz is not None:
        _cloud.upload(folder_path, temp_file_path_npz)


# endregion
# region Load functions


def get_raw_data(file_name=None, file_id=None):
    """Returns a dictionary containing the json object from the specified
    raw data file
    """

    file_content = _cloud.download(file_name, "txt", file_id)
    data = json.loads(file_content)

    # Find and decompress the linked numpy arrays
    # npz_file = None
    # for key in data:
    #     val = data[key]
    #     if isinstance(val, str) and val.endswith(".npz"):
    #         if npz_file is None:
    #             npz_file_content = _cloud.download(file_name, "npz")
    #             npz_file = np.load(BytesIO(npz_file_content))
    #         data[key] = npz_file[key]

    _json_deescape(data)

    return data


# endregion
# region Misc public functions


def get_time_stamp_from_file_name(file_name):
    """Get the formatted timestamp from a file name

    Returns:
        string: <year>_<month>_<day>-<hour>_<minute>_<second>
    """

    file_name_split = file_name.split("-")
    time_stamp_parts = file_name_split[0:2]
    timestamp = "-".join(time_stamp_parts)
    return timestamp


def utc_from_file_name(file_name, time_zone="CST"):
    # First 19 characters are human-readable timestamp
    date_time_str = file_name[0:19]
    # Assume timezone is CST
    date_time_str += f"-{time_zone}"
    date_time = datetime.strptime(date_time_str, r"%Y_%m_%d-%H_%M_%S-%Z")
    timestamp = date_time.timestamp()
    return timestamp


def get_nv_sig_units_no_cxn():
    with labrad.connect() as cxn:
        nv_sig_units = get_nv_sig_units(cxn)
    return nv_sig_units


def get_nv_sig_units():
    try:
        config = common.get_config_dict()
        nv_sig_units = config["nv_sig_units"]
    except Exception:
        nv_sig_units = ""
    return nv_sig_units


# endregion
# region Private functions


def _get_branch_name():
    """Return the name of the active branch of dioptric"""
    repo_path = common.get_repo_path()
    repo = Repo(repo_path)
    return repo.active_branch.name


def _json_deescape(raw_data):
    """Recursively deescape a raw data object from JSON.
    Currently just escapes enums that are saved as strings
    """

    # See what kind of loop we need to do through the object
    if isinstance(raw_data, dict):
        # Just get the original keys
        keys = list(raw_data.keys())
    elif isinstance(raw_data, list):
        keys = range(len(raw_data))

    for key in keys:
        val = raw_data[key]

        # Deescape the key itself if necessary
        try:
            if isinstance(key, str):
                str_key = key
                eval_key = eval(key)
                if isinstance(eval_key, Enum):
                    raw_data[eval_key] = val
                    del raw_data[str_key]
                    key = eval_key
        except:
            pass

        # Descape the value
        try:
            if isinstance(val, str):
                eval_val = eval(val)
                if isinstance(eval_val, Enum):
                    raw_data[key] = eval_val
        except:
            pass

        # Recursion for dictionaries and lists
        if isinstance(val, dict) or isinstance(val, list):
            _json_deescape(val)


def _json_escape(raw_data):
    """Recursively escape a raw data object for JSON"""

    # See what kind of loop we need to do through the object
    if isinstance(raw_data, dict):
        # Just get the original keys
        keys = list(raw_data.keys())
    elif isinstance(raw_data, list):
        keys = range(len(raw_data))

    for key in keys:
        val = raw_data[key]

        # Escape the key itself if necessary
        if isinstance(key, Enum):
            raw_data[str(key)] = val
            del raw_data[key]

        # Escape the value
        if type(val) == np.ndarray:
            raw_data[key] = val.tolist()
        elif isinstance(val, Enum):
            raw_data[key] = str(val)
        elif isinstance(val, Path):
            raw_data[key] = str(val)
        elif isinstance(val, type):
            raw_data[key] = str(val)
        # Recursion for dictionaries and lists
        elif isinstance(val, dict) or isinstance(val, list):
            _json_escape(val)


# endregion


if __name__ == "__main__":
    time_stamp = get_time_stamp()
    file_path = get_file_path(__file__, time_stamp, "MCCTEST")
    data = {"matt": "Cambria!"}
    save_raw_data(data, file_path)
