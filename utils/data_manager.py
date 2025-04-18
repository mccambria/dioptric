# -*- coding: utf-8 -*-
"""
Tools for managing our experimental database

Created November 15th, 2023

@author: mccambria
@author:
"""

# region Imports and constants

import copy
import io
import os
import socket
import time
import traceback
from dataclasses import fields
from datetime import datetime
from io import BytesIO
from pathlib import Path

import labrad
import numpy as np
import orjson  # orjson is faster and more lightweight than ujson, but can't write straight to file
import ujson  # usjson is faster than standard json library
from git import Repo
from PIL import Image

# from utils import _cloud as cloud
# from utils import _cloudnew2 as cloud
from utils import common, widefield
from utils.constants import NVSig

USE_NEW_CLOUD = False  # by defualt box cloud
if USE_NEW_CLOUD:
    from utils import _cloudnew2 as cloud
else:
    from utils import _cloud as cloud

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
    """Save a matplotlib figure as a png.

    Params:
        fig: matplotlib.figure.Figure
            The figure to save
        file_path: string
            The file path to save to including the file name, excluding the
            extension
    """

    # Write to bytes then upload that to the cloud
    ext = "png"
    file_path_ext = file_path.with_suffix(f".{ext}")
    content = BytesIO()
    # fig.savefig(content, format=ext)
    fig.savefig(content, format=ext, dpi=300, bbox_inches="tight")
    cloud.upload(file_path_ext, content)


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

    if keys_to_compress is None:
        keys_to_compress = widefield.get_default_keys_to_compress(raw_data)

    # start = time.time()
    file_path_txt = file_path.with_suffix(".txt")

    # Work with a copy of the raw data to avoid mutation
    # raw_data = copy.deepcopy(raw_data)

    # Compress numpy arrays to linked file
    try:
        if len(keys_to_compress) > 0:
            # Build the object to compress
            kwargs = {}
            for key in keys_to_compress:
                kwargs[key] = raw_data[key]
            # Upload to cloud
            content = BytesIO()
            np.savez_compressed(content, **kwargs)
            file_path_npz = file_path.with_suffix(".npz")
            npz_file_id = cloud.upload(file_path_npz, content)
            # Replace the value in the raw data with a string that tells us where
            # to find the compressed file
            for key in keys_to_compress:
                raw_data[key] = f"{npz_file_id}.npz"
    except Exception:
        print(traceback.format_exc())

    # Always include the config dict
    config = common.get_config_dict()
    config_copy = copy.deepcopy(config)
    _json_escape(config_copy)
    raw_data["config"] = config_copy

    # And the OPX config dict if there is one
    opx_config = common.get_opx_config_dict()
    if opx_config is not None:
        opx_config_copy = copy.deepcopy(opx_config)
        _json_escape(opx_config_copy)
        raw_data["opx_config"] = opx_config_copy

    # Upload raw data to the cloud
    try:
        option = (
            orjson.OPT_INDENT_2 | orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_NON_STR_KEYS
        )
        content = orjson.dumps(raw_data, option=option)
        file_id = cloud.upload(file_path_txt, BytesIO(content))
    except Exception:
        print(traceback.format_exc())
        # Save to local file instead
        with open(data_manager_folder / file_path_txt.name, "wb") as f:
            f.write(content)
        file_id = None

    return file_id

    # stop = time.time()
    # print(stop - start)


# region Load functions


def get_file_name(file_id):
    try:
        with open(data_manager_folder / "cache_manifest.txt") as f:
            cache_manifest = ujson.load(f)
        file_name = cache_manifest[file_id]
    except Exception:
        _, _, file_name = cloud.download(file_id=file_id)
    return file_name


def get_raw_data(file_name=None, file_id=None, use_cache=True, load_npz=False):
    """Returns a dictionary containing the json object from the specified
    raw data file

    Parameters
    ----------
    file_name : str, optional
        Name of the raw data file to load, w/o extension. If file_name is passed,
        file_id is None, and the file is not in the cache, then we'll identify
        the proper file by searching the cloud for it. By default None
    file_id : str, optional
        Cloud ID of the file to load. Loaded directly from the cloud or cache, no
        search necessary. By default None
    use_cache : bool, optional
        Whether or not to use the cache. If True, we'll try to pull the file from
        the cache - if it's not there already we'll add it to the cache. Otherwise
        we'll get the file straight from the cloud and skip caching it. By default
        True
    load_npz: bool, optional
        Whether or not to retrieve any linked compressed numpy files (.npz files).
        Retrieving these can be slow if the file is very large so it's better to
        skip it if you don't need it.

    Returns
    -------
    dict
        Dictionary containing the json object from the specified raw data file
    """

    # Recurse if multiple file_ids passed
    if isinstance(file_id, (list, tuple)):
        file_ids = file_id
        data = get_raw_data(file_id=file_ids[0])
        for file_id in file_ids[1:]:
            new_data = get_raw_data(file_id=file_id)
            data["num_runs"] += new_data["num_runs"]
            data["counts"] = np.append(data["counts"], new_data["counts"], axis=2)
        return data

    if file_id is not None:
        file_id = str(file_id)

    ### Check the cache first

    # Try to open an existing cache manifest
    retrieved_from_cache = False
    if use_cache:
        try:
            with open(data_manager_folder / "cache_manifest.txt") as f:
                cache_manifest = ujson.load(f)
        except Exception:
            cache_manifest = None

        # Try to open the cached file
        try:
            cache_entry = cache_manifest[file_name]
            # Only load from cache if we either don't need the npz or we already have it
            if not load_npz or cache_entry["load_npz"]:
                with open(data_manager_folder / f"{file_name}.txt", "rb") as f:
                    file_content = f.read()
                data = orjson.loads(file_content)
                retrieved_from_cache = True
        except Exception:
            pass

    ### If not in cache, download from the cloud

    if not retrieved_from_cache:
        # Download the base file
        # file_content = cloud.download(file_name, "txt", file_id)
        file_content_tuple = cloud.download(file_name, "txt", file_id)
        # support both old (tuple) and new (bytes-only) formats
        if isinstance(file_content_tuple, tuple):
            file_content, file_id, file_name = file_content_tuple
        else:
            file_content = file_content_tuple

        data = orjson.loads(file_content)

        # Find and decompress the linked numpy arrays
        if load_npz:
            for key in data:
                val = data[key]
                if not isinstance(val, str):
                    continue
                val_split = val.split(".")
                if val_split[-1] != "npz":
                    continue
                first_part = val_split[0]
                npz_file_id = first_part if first_part else None
                # npz_file_content, _, _ = cloud.download(file_name, "npz", npz_file_id)
                npz_file_file_content_tuple = cloud.download(
                    file_name, "npz", npz_file_id
                )
                if isinstance(npz_file_file_content_tuple, tuple):
                    npz_file_content, file_id, file_name = npz_file_file_content_tuple
                else:
                    npz_file_content = npz_file_file_content_tuple

                npz_data = np.load(BytesIO(npz_file_content))
                data |= npz_data
                break
    ### Add to cache and return the data

    # Update the cache manifest
    if use_cache:
        cache_manifest_updated = False
        if not retrieved_from_cache:
            if cache_manifest is None:
                cache_manifest = {}
            cached_file_names = list(cache_manifest.keys())
            # Add the new file to the manifest
            cache_manifest[file_name] = {"file_id": file_id, "load_npz": load_npz}
            cached_file_names.append(file_name)
            while len(cached_file_names) > 10:
                file_name_to_remove = cached_file_names.pop(0)
                del cache_manifest[file_name_to_remove]
                file_to_remove = data_manager_folder / f"{file_name_to_remove}.txt"
                if file_to_remove.exists():
                    os.remove(file_to_remove)
                # os.remove(data_manager_folder / f"{file_name_to_remove}.txt")
            cache_manifest_updated = True
        if cache_manifest_updated:
            with open(data_manager_folder / "cache_manifest.txt", "w+") as f:
                ujson.dump(cache_manifest, f, indent=2)

        # Write the actual data file to the cache
        if not retrieved_from_cache:
            file_content = orjson.dumps(data, option=orjson.OPT_SERIALIZE_NUMPY)
            with open(data_manager_folder / f"{file_name}.txt", "wb") as f:
                f.write(file_content)

    # Retrieve valid fields for NVSig
    valid_fields = {field.name for field in fields(NVSig)}

    if "nv_list" in data:
        nv_list = data["nv_list"]
        # Filter out unexpected fields
        nv_list = [
            NVSig(**{key: nv[key] for key in nv if key in valid_fields})
            for nv in nv_list
        ]
        data["nv_list"] = nv_list

    return data

    # if "nv_list" in data:
    #     nv_list = data["nv_list"]
    #     nv_list = [NVSig(**nv) for nv in nv_list]
    #     data["nv_list"] = nv_list

    # return data


# from io import BytesIO

# import numpy as np
# import orjson
# import ujson


# def get_raw_data(file_name=None, file_id=None, use_cache=True, load_npz=False):
#     """Returns a dictionary containing the JSON object from the specified raw data file,
#     correctly loading NumPy `.npz` files and converting arrays to lists.

#     Parameters
#     ----------
#     file_name : str, optional
#         Name of the raw data file to load, w/o extension.
#     file_id : str, optional
#         Cloud ID of the file to load.
#     use_cache : bool, optional
#         Whether or not to use the cache.
#     load_npz: bool, optional
#         Whether or not to retrieve any linked compressed numpy files.

#     Returns
#     -------
#     dict
#         Dictionary containing the JSON object with NumPy arrays properly converted.
#     """

#     if file_id is not None:
#         file_id = str(file_id)

#     # Attempt to load from cache
#     retrieved_from_cache = False
#     if use_cache:
#         try:
#             with open(data_manager_folder / "cache_manifest.txt") as f:
#                 cache_manifest = ujson.load(f)
#         except Exception:
#             cache_manifest = None

#         try:
#             if file_id is None:
#                 for key in cache_manifest.keys():
#                     if cache_manifest[key]["file_name"] == file_name:
#                         file_id = key
#             cache_entry = cache_manifest[file_id]
#             if not load_npz or cache_entry["load_npz"]:
#                 if file_name is None:
#                     file_name = cache_entry["file_name"]
#                 with open(data_manager_folder / f"{file_name}.txt", "rb") as f:
#                     file_content = f.read()
#                 data = orjson.loads(file_content)
#                 retrieved_from_cache = True
#         except Exception:
#             pass

#     # If not in cache, download from the cloud
#     if not retrieved_from_cache:
#         file_content, file_id, file_name = cloud.download(file_name, "txt", file_id)
#         data = orjson.loads(file_content)

#     # Load and process .npz files if required
#     if load_npz:
#         npz_keys_to_load = []
#         for key, val in data.items():
#             if isinstance(val, str) and val.endswith(".npz"):
#                 npz_keys_to_load.append((key, val))

#         for key, npz_file_name in npz_keys_to_load:
#             print(f"Loading NPZ file: {npz_file_name} for key: {key}")

#             npz_file_id = npz_file_name.split(".")[0] if npz_file_name else None
#             try:
#                 npz_file_content, _, _ = cloud.download(file_name, "npz", npz_file_id)
#                 npz_data = np.load(BytesIO(npz_file_content), allow_pickle=True)

#                 # Convert NumPy arrays to lists for JSON serialization
#                 data[key] = {
#                     k: v.tolist() if isinstance(v, np.ndarray) else v
#                     for k, v in npz_data.items()
#                 }
#                 print(f"Successfully loaded NPZ data for key: {key}")
#             except Exception as e:
#                 print(f"Error loading NPZ file {npz_file_name}: {e}")

#     # Convert NumPy types to Python-native types for JSON serialization
#     def convert_numpy(obj):
#         if isinstance(obj, np.ndarray):
#             return obj.tolist()
#         elif isinstance(obj, (np.float64, np.float32, np.float16)):
#             return float(obj)
#         elif isinstance(obj, (np.int64, np.int32, np.int16)):
#             return int(obj)
#         elif isinstance(obj, dict):
#             return {key: convert_numpy(value) for key, value in obj.items()}
#         elif isinstance(obj, list):
#             return [convert_numpy(item) for item in obj]
#         return obj

#     data = convert_numpy(data)

#     return data


def get_img(file_name=None, ext=None, file_id=None):
    # file_content, file_id, file_name = cloud.download(file_name, ext, file_id)
    file_content_tuple = cloud.download(file_name, ext, file_id)
    if isinstance(file_content_tuple, tuple):
        file_content, file_id, file_name = file_content_tuple
    else:
        file_content = file_content_tuple

    img = Image.open(io.BytesIO(file_content))
    img = np.asarray(img)
    return np.asarray(img)


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


def _json_escape(raw_data):
    """Recursively escape a raw data object for JSON. Slow, so use sparingly"""

    # See what kind of loop we need to do through the object
    if isinstance(raw_data, dict):
        # Just get the original keys
        keys = list(raw_data.keys())
    elif isinstance(raw_data, list):
        keys = range(len(raw_data))

    for key in keys:
        val = raw_data[key]

        # Escape the value
        if isinstance(val, Path):
            raw_data[key] = str(val)
        elif isinstance(val, type):
            raw_data[key] = str(val)

        # Recursion for dictionaries and lists
        elif isinstance(val, dict) or isinstance(val, list):
            _json_escape(val)


# endregion


if __name__ == "__main__":
    file_name = "2025_03_14-11_57_49-rubin-nv0_2025_02_26"
    data = get_raw_data(file_name, use_cache=True, load_npz=False)
    # timestamp = get_time_stamp()
    # repr_nv_name = "testing"
    # file_path = get_file_path(__file__, timestamp, repr_nv_name)
    # save_raw_data(data, file_path)
    debu = 0
