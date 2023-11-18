# -*- coding: utf-8 -*-
"""
Tools for managing our experimental database, which lives on Box

Created November 15th, 2023

@author: mccambria
"""

# region Imports and constants

import datetime
from io import BytesIO
import utils.common as common
import os
from pathlib import PurePath, Path
import sqlite3
import time
import json
from boxsdk import Client, OAuth2, JWTAuth
from git import Repo
from enum import Enum
import numpy as np
import socket
import labrad
import copy


# auth = OAuth2(
#     client_id="dkp31zlkfbc21qj974iuncf5f2p6yypr",
#     client_secret="Wi8mjAXUF25RapQ2OPs454AIqFcvBttJ",
#     access_token="AaUIaa6409TxqnDsGmuQMlYOpgkLkzFR",
# )
auth = JWTAuth.from_settings_file(Path.home() / "Downloads/81479_7yezvcxy_config.json")

search_index_file_name = "search_index.db"
nvdata_path = common.get_nvdata_path()
nvdata_path_str = str(nvdata_path)
date_glob = "[0-9][0-9][0-9][0-9]_[0-9][0-9]"
search_index_glob = f"{nvdata_path_str}/pc_*/branch_*/*/{date_glob}/*.txt"

# endregion
# region Private functions


def process_full_path(full_path):
    """Return just what we want for writing to the database. Expects a Path
    containing the entire path to the file, including nvdata, the file
    name, the extension...
    """

    # Make sure we have a PurePath to manipulate
    full_path = PurePath(full_path)

    # Get the path to the file separated from the file name itself and nvdata
    path_to_file = full_path.parent
    path_to_file_parts = path_to_file.parts
    nvdata_ind = path_to_file_parts.index("nvdata")
    index_path_parts = path_to_file_parts[nvdata_ind + 1 :]
    index_path = PurePath(index_path_parts[0]).joinpath(*index_path_parts[1:])
    # Save the path string in the posix format
    index_path = str(index_path.as_posix())

    # Get the file name, no extension
    index_file_name = full_path.stem

    return (index_file_name, index_path)


def add_to_search_index(data_full_path):
    db_vals = process_full_path(data_full_path)
    search_index = sqlite3.connect(nvdata_path / search_index_file_name)
    cursor = search_index.cursor()
    cursor.execute("INSERT INTO search_index VALUES (?, ?)", db_vals)
    search_index.commit()
    search_index.close()
    # Sleep for 1 second so that every file name from the same PC should be unique
    time.sleep(1)
    return db_vals[1]


def get_data_path_from_nvdata(data_file_name):
    try:
        search_index = sqlite3.connect(nvdata_path / search_index_file_name)
        cursor = search_index.cursor()
        cursor.execute(
            "SELECT * FROM search_index WHERE file_name = '{}'".format(data_file_name)
        )
        res = cursor.fetchone()
        return res[1]
    except Exception as exc:
        print(f"Failed to find file {data_file_name} in search index.")
        print("Attempting on-the-fly indexing.")
        index_path = index_on_the_fly(data_file_name)
        if index_path is None:
            msg = f"File {data_file_name} does not appear to exist in data" " folders."
            raise RuntimeError(msg)
        return index_path


def index_on_the_fly(data_file_name):
    """If a file fails to be indexed for whatever reason and we subsequently
    unsuccesfully attempt to look it up, we'll just index it on the fly
    """

    data_file_name_w_ext = f"{data_file_name}.txt"
    data_full_path = None
    yyyy_mm = data_file_name[0:7]

    for root, _, files in os.walk(nvdata_path):
        path_root = PurePath(root)
        # Before looping through all the files make sure the folder fits
        # the glob
        test_path_root = path_root / "test.txt"
        if not test_path_root.match(search_index_glob):
            continue
        # Make sure the folder matches when the file was created
        if not root.endswith(yyyy_mm):
            continue
        if data_file_name_w_ext in files:
            # for f in files:
            data_full_path = f"{root}/{data_file_name_w_ext}"
            break

    if data_full_path is None:
        print(f"Failed to index file {data_file_name} on the fly.")
        return None
    else:
        index_path = add_to_search_index(data_full_path)
        return index_path


# endregion
# region Public functions
"""See also get_raw_data and get_raw_data_path in tool_belt"""


def gen_search_index():
    """Create the search index from scratch. This will take several minutes.
    Once complete, delete the old index file and remove the "new_" prefix
    from the fresh index.
    """

    # Determine how many files
    # start = time.time()
    # counter = 0
    # for root, _, files in os.walk(nvdata_path):
    #     path_root = PurePath(root)
    #     # Before looping through all the files make sure the folder fits the glob
    #     test_path_root = path_root / "test.txt"
    #     print(path_root)
    #     if not test_path_root.match(search_index_glob):
    #         continue
    #     for f in files:
    #         print(f)
    #         if f.split(".")[-1] == "txt":
    #             counter += 1
    #             if counter >= 2000:
    #                 break
    # end = time.time()
    # print(end - start)
    # print(counter)
    # return

    # Create the table
    temp_name = "new_" + search_index_file_name
    search_index = sqlite3.connect(nvdata_path / temp_name)
    cursor = search_index.cursor()
    cursor.execute(
        """CREATE TABLE search_index (file_name text, path_from_nvdata text)"""
    )

    for root, _, files in os.walk(nvdata_path):
        path_root = PurePath(root)
        # Before looping through all the files make sure the folder fits the glob
        test_path_root = path_root / "test.txt"
        if not test_path_root.match(search_index_glob):
            continue
        for f in files:
            if f.split(".")[-1] == "txt":
                db_vals = process_full_path(f"{root}/{f}")
                cursor.execute("INSERT INTO search_index VALUES (?, ?)", db_vals)

    search_index.commit()
    search_index.close()


# endregion
# region File and data handling utils


def get_raw_data(file_name, path_from_nvdata=None, nvdata_dir=None):
    """Returns a dictionary containing the json object from the specified
    raw data file. If path_from_nvdata is not specified, we assume we're
    looking for an autogenerated experiment data file. In this case we'll
    use glob (a pattern matching module for pathnames) to efficiently find
    the file based on the known structure of the directories rooted from
    nvdata_dir (ie nvdata_dir / pc_folder / routine / year_month / file.txt)
    """
    # file_path = get_raw_data_path(file_name, path_from_nvdata, nvdata_dir)
    # with file_path.open() as f:
    #     data = json.load(f)

    ###

    client = Client(auth)

    # config = JWTAuth.from_settings_file(Path.home() / "lab/dioptric_box_config.json")
    # client = Client(config)

    search_results = client.search().query(
        f'"{file_name}"',
        type="file",
        limit=1,
        content_types=["name"],
        file_extensions=["txt"],
    )
    search_results = list(search_results)
    if len(search_results) == 0:
        raise RuntimeError("No file found with the passed file_name.")
    elif len(search_results) > 1:
        raise Warning(
            "Multiple files found with the same file_name. Using first file..."
        )
    match = search_results[0]
    file_content = client.file(match.id).content()
    data = json.loads(file_content)

    ###

    # Find and decompress the linked numpy arrays
    npz_file = None
    for key in data:
        val = data[key]
        if isinstance(val, str) and val.endswith(".npz"):
            if npz_file is None:
                # The npz_file path is saved without the part up to nvdata so that
                # it isn't tied to a specific PC
                # generic_path = val
                # if nvdata_dir is None:
                #     nvdata_dir = common.get_nvdata_path()
                # full_path = nvdata_dir / generic_path
                # npz_file = np.load(full_path)

                search_results = client.search().query(
                    f'"{file_name}"',
                    type="file",
                    limit=1,
                    content_types=["name"],
                    file_extensions=["npz"],
                )
                search_results = list(search_results)
                match = search_results[0]
                file_content = client.file(match.id).content()
                npz_file = np.load(BytesIO(file_content))

            data[key] = npz_file[key]

    return data


def get_raw_data_path(
    file_name,
    path_from_nvdata=None,
    nvdata_dir=None,
):
    """Same as get_raw_data, but just returns the path to the file"""
    if nvdata_dir is None:
        nvdata_dir = common.get_nvdata_path()
    if path_from_nvdata is None:
        path_from_nvdata = get_data_path_from_nvdata(file_name)
    data_dir = nvdata_dir / path_from_nvdata
    file_name_ext = "{}.txt".format(file_name)
    file_path = data_dir / file_name_ext
    return file_path


def get_branch_name():
    """Return the name of the active branch of dioptric (fka kolkowitz-nv-experiment-v1.0)"""
    repo_path = common.get_repo_path()
    repo = Repo(repo_path)
    return repo.active_branch.name


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


def get_time_stamp_from_file_name(file_name):
    """Get the formatted timestamp from a file name

    Returns:
        string: <year>_<month>_<day>-<hour>_<minute>_<second>
    """

    file_name_split = file_name.split("-")
    time_stamp_parts = file_name_split[0:2]
    timestamp = "-".join(time_stamp_parts)
    return timestamp


def get_files_in_folder(folderDir, filetype=None):
    """
    folderDir: str
        full file path, use previous function get_folder_dir
    filetype: str
        must be a 3-letter file extension, do NOT include the period. ex: 'txt'
    """
    # print(folderDir)
    file_list_temp = os.listdir(folderDir)
    if filetype:
        file_list = []
        for file in file_list_temp:
            if file[-3:] == filetype:
                file_list.append(file)
    else:
        file_list = file_list_temp

    return file_list


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

    nvdata_path = common.get_nvdata_path()
    pc_name = socket.gethostname()
    branch_name = get_branch_name()
    source_name = Path(source_file).stem
    date_folder = "_".join(time_stamp.split("_")[0:2])  # yyyy_mm

    folder_dir = (
        nvdata_path
        / f"pc_{pc_name}"
        / f"branch_{branch_name}"
        / source_name
        / date_folder
    )

    if subfolder is not None:
        folder_dir = folder_dir / subfolder

    # Make the required directories if it doesn't exist already
    folder_dir.mkdir(parents=True, exist_ok=True)

    file_name = f"{time_stamp}-{name}"

    return folder_dir / file_name


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

    file_path_ext = file_path.with_suffix(".txt")

    # Work with a copy of the raw data to avoid mutation
    raw_data = copy.deepcopy(raw_data)

    # Compress numpy arrays to linked file
    if keys_to_compress is not None:
        file_path_npz = file_path.with_suffix(".npz")
        # Get a generic version of the path so that the part up to nvdata can be
        # filled in upon retrieval
        _, file_path_npz_generic = process_full_path(file_path_npz)
        file_path_npz_generic += f"/{file_path_npz.name}"
        kwargs = {}
        for key in keys_to_compress:
            kwargs[key] = raw_data[key]
            # Replace the value in the data with the npz file path
            raw_data[key] = file_path_npz_generic
        with open(file_path_npz, "wb") as f:
            np.savez_compressed(f, **kwargs)

    # Always include the config
    config = common.get_config_dict()
    config = copy.deepcopy(config)
    raw_data["config"] = config

    _json_escape(raw_data)

    with open(file_path_ext, "w") as file:
        json.dump(raw_data, file, indent=2)

    if file_path_ext.match(search_index_glob):
        add_to_search_index(file_path_ext)


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


def test():
    client = Client(auth)

    # config = JWTAuth.from_settings_file(Path.home() / "lab/dioptric_box_config.json")
    # client = Client(config)

    file_name = "2023_11_10-12_16_36-johnson-nv0_2023_11_09"

    search_results = client.search().query(
        f'"{file_name}"',
        type="file",
        limit=1,
        content_types=["name"],
        file_extensions=["txt"],
    )
    search_results = list(search_results)
    match = search_results[0]
    print(match.id)
    file_content = client.file(match.id).content()
    data = json.loads(file_content)
    print(data["nv_sig"])

    search_results = client.search().query(
        f'"{file_name}"',
        type="file",
        limit=1,
        content_types=["name"],
        file_extensions=["npz"],
    )
    search_results = list(search_results)
    match = search_results[0]
    print(match.id)
    file_content = client.file(match.id).content()
    img_arrays = np.load(file_content)
    print(len(img_arrays))


# endregion


if __name__ == "__main__":
    client = Client(auth)
    print(client.user().get())

    # users = client.users(user_type="all")
    # for user in users:
    #     print(f"{user.name} (User ID: {user.id})")

    file_content = client.file("1363392618720").content()
    data = json.loads(file_content)
    print(data["nv_sig"])

    # test()
    # gen_search_index()
    # index_on_the_fly("2022_07_06-16_38_20-hopper-search")

    # root = nvdata_path / "pc_rabi/branch_master/image_sample/2023_11"
    # file_list = [
    #     '2023_11_01-10_07_18-johnson-nv0_2023_10_30.txt',
    # ]
    # paths = [root / el for el in file_list]
    # for el in paths:
    #     add_to_search_index(el)
