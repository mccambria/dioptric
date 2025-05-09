# -*- coding: utf-8 -*-
"""
Tools for talking to the NAS, which hosts all our data. The
NAS is assumed to be present as a network drive on your machine.
This file should only be accessed by data_manager

Created March 2025

@author: egediman
"""

import os
import re
import shutil
import time
from pathlib import Path

import numpy as np
import orjson

from utils import common, search_index

nvdata_dir = common.get_nvdata_dir()

# region Required public functions


def download(file_stem=None, ext=".txt", file_parent=None):
    """Download file from the NAS

    Parameters
    ----------
    file_stem : str
        Name of the file, without file extension
    ext : str
        File extension
    file_parent : Path
        Parent Path for the file. If specified, we will not query the search
        index to find the parent

    Returns
    -------
    bytes
        bytes from the file
    """
    if file_parent is None:
        file_parent = search_index.get_file_parent(file_stem)
    file_path = file_parent / f"{file_stem}{ext}"
    with file_path.open("rb") as f:
        return f.read()


def upload(file_path, content, do_add_to_search_index=False):
    """Upload file to the NAS

    Parameters
    ----------
    file_path : Path
        Complete file path to upload to
    content : bytes | memoryview
        bytes to write to the file
    do_add_to_search_index : bool, optional
        Whether to add the file stem and relative parent to the search index for
        quick lookup later, by default False
    """
    if not file_path.parent.is_dir():
        file_path.parent.mkdir(parents=True)
    with file_path.open("wb+") as f:
        f.write(content)
    if do_add_to_search_index:
        search_index.add_to_search_index(file_path)


# endregion
# region Delete functions, for cleaning up old data


def _delete_folders(reg_exp, folder_path):
    def condition_fn(folder_path):
        return re.fullmatch(reg_exp, folder_path)

    return _batch_delete(condition_fn, folder_path)


def _delete_empty_folders(folder_path):
    def condition_fn(folder_path):
        return os.listdir(folder_path) == 0

    return _batch_delete(condition_fn, folder_path)


def _batch_delete(condition_fn, folder_path):
    # folder path from root
    folder_path = common.get_nvdata_dir / folder_path

    # Delete this folder if it satisfies the passed condition
    if condition_fn(folder_path):
        shutil.rmtree(folder_path)
        print(f"{folder_path}")
        return True

    # Otherwise recurse through the folder's contents and check for other folders to delete
    items = os.listdir(folder_path)
    for item in items:
        if os.path.isdir(item):
            res = _batch_delete(condition_fn, item)
            # Uncomment the next two lines to just delete the first item we find and quit
            # if res:
            #     return res


# endregion

if __name__ == "__main__":
    file_stem = "2025_04_17-22_09_45-rubin-nv0_2025_02_26"
    print(search_index.get_file_parent(file_stem))
