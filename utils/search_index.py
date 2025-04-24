# -*- coding: utf-8 -*-
"""
Here are functions for our search index, which allows us to quickly look up
data files without specifying the file path. All you need is the file name!
You probably just need one of the end-user facing functions in data_manager.
If you're here to call a function, check out the public functions at the bottom

Created September 10th, 2021

@author: mccambria
"""

# region Imports and constants

import os
import sqlite3
import sys
import time
from pathlib import PurePath

import utils.common as common

search_index_file_name = "search_index.db"
nvdata_dir = common.get_nvdata_dir()
nvdata_dir_str = str(nvdata_dir)
date_glob = "[0-9][0-9][0-9][0-9]_[0-9][0-9]"
search_index_glob = f"{nvdata_dir_str}/pc_*/branch_*/*/{date_glob}/*.txt"

# endregion
# region Private functions


def process_file_path(file_path):
    """Extract the information we want to index from the file path for quick
    lookup later

    Parameters
    ----------
    file_path : Path
        Complete path to the file

    Returns
    -------
    tuple(str, str)
        File stem and file parent relative to nvdata as a string
    """
    file_stem = file_path.stem
    relative_parent = str(file_path.parent.relative_to(nvdata_dir))
    return (file_stem, relative_parent)


def add_to_search_index(file_path):
    db_vals = process_file_path(file_path)
    search_index = sqlite3.connect(nvdata_dir / search_index_file_name)
    cursor = search_index.cursor()
    cursor.execute("INSERT INTO search_index VALUES (?, ?)", db_vals)
    search_index.commit()
    search_index.close()
    # Sleep for 1 second so that every file name from the same PC should be unique
    time.sleep(1)
    return db_vals[1]


def index_on_the_fly(file_stem):
    """If a file fails to be indexed for whatever reason and we subsequently
    unsuccesfully attempt to look it up, we'll just index it on the fly
    """

    file_name = f"{file_stem}.txt"
    file_path = None
    yyyy_mm = file_stem[0:7]

    for root, _, files in os.walk(nvdata_dir):
        path_root = PurePath(root)
        # Before looping through all the files make sure the folder fits
        # the glob
        test_path_root = path_root / "test.txt"
        if not test_path_root.match(search_index_glob):
            continue
        # Make sure the folder matches when the file was created
        if not root.endswith(yyyy_mm):
            continue
        if file_name in files:
            # for f in files:
            file_path = f"{root}/{file_name}"
            break

    if file_path is None:
        print(f"Failed to index file {file_stem} on the fly.")
        return None
    else:
        index_path = add_to_search_index(file_path)
        return index_path


def get_file_parent(file_name):
    """Return the file parent for a file name in nvdata. Allows for easy retrieval of
    the file without needing to manually input the file path."""
    try:
        search_index = sqlite3.connect(nvdata_dir / search_index_file_name)
        cursor = search_index.cursor()
        cursor.execute(
            "SELECT * FROM search_index WHERE file_name = '{}'".format(file_name)
        )
        res = cursor.fetchone()
        parent_from_nvdata = res[1]
    except Exception as exc:
        print(f"Failed to find file {file_name} in search index.")
        print("Attempting on-the-fly indexing.")
        index_path = index_on_the_fly(file_name)
        if index_path is None:
            msg = f"File {file_name} does not appear to exist in data folders."
            raise RuntimeError(msg)
        parent_from_nvdata = index_path
    return nvdata_dir / parent_from_nvdata


# endregion
# region Public functions
"""See also get_raw_data and get_raw_data_path in tool_belt"""


def gen_search_index():
    """Create the search index from scratch. This will take several minutes.
    Once complete, delete the old index file and remove the "new_" prefix
    from the fresh index.
    """

    # Create the table
    temp_name = "new_" + search_index_file_name
    search_index = sqlite3.connect(nvdata_dir / temp_name)
    cursor = search_index.cursor()
    cursor.execute(
        """CREATE TABLE search_index (file_name text, path_from_nvdata text)"""
    )

    for root, _, files in os.walk(nvdata_dir):
        path_root = PurePath(root)
        # Before looping through all the files make sure the folder fits
        # the glob
        test_path_root = path_root / "test.txt"
        if not test_path_root.match(search_index_glob):
            continue
        for f in files:
            if f.split(".")[-1] == "txt":
                db_vals = process_file_path(f"{root}/{f}")
                cursor.execute("INSERT INTO search_index VALUES (?, ?)", db_vals)

    search_index.commit()
    search_index.close()


# endregion


if __name__ == "__main__":
    gen_search_index()
    sys.exit()
    # index_on_the_fly("2022_07_06-16_38_20-hopper-search")

    root = nvdata_dir / "pc_hahn/branch_master/pulsed_resonance/2023_02"
    file_list = [
        "2023_02_24-10_33_55-15micro-nv7_zfs_vs_t.txt",
    ]
    paths = [root / el for el in file_list]
    for el in paths:
        add_to_search_index(el)
