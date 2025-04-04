# -*- coding: utf-8 -*-
"""
Tools for talking to our cloud provider, Box, which hosts all our data.
This file should only be accessed by the data_manager util

Created November 18th, 2023

@author: mccambria
"""

import re
import time
from pathlib import Path

import os
import shutil
from utils import common
import utils.search_index as search_index


def download(file_name,
    path_from_nvdata=None,
    nvdata_dir=None,):
    """Download file from the cloud

    Parameters
    ----------
    file_name : str
        Name of the file, without file extension
    path_from_nvdata = path to file, searched for if not provided
    nvdata_dir = directory for nvdata folder, grabbed from common otherwise

    Returns
    -------
    Binary string
        Contents of the file
    """
    file_path = get_raw_data_path(file_name, path_from_nvdata, nvdata_dir)
    with file_path.open() as f:
        res = f.read()
        return res



def upload(file_path, content):
    """Upload file to the cloud

    Parameters
    ----------
    file_path : Path
        File path to upload to. Form should be folder1/folder2/... where folder1
        is under directly the root data folder. Should include extension
    content : BytesIO
        Byte stream to write to the file
    """
    path = common.get_nvdata_dir / file_path
    with path.open('w') as f:
        f.write(content.getbuffer())


def get_folder_id(folder_path, no_create=False):
    """Gets the Box ID of the specified folder. Optionally creates the folder if
    it does not exist yet

    Parameters
    ----------
    folder_path : Path
        Folder path to ID. Form should be folder1/folder2/... where folder1
        is directly under the root data folder

    Returns
    -------
    str
        ID of the folder
    """

    # See if the ID is stored in the cache
    global folder_path_cache
    if folder_path in folder_path_cache:
        return folder_path_cache[folder_path]

    # If it's not in the cache, look it up from the cloud
    folder_path_parts = list(folder_path.parts)
    folder_id = _get_folder_id_recursion(folder_path_parts, no_create=no_create)
    folder_path_cache[folder_path] = folder_id
    return folder_id


def _get_folder_id_recursion(
    folder_path_parts, start_id=root_folder_id, no_create=False
):
    """
    Starting from the root data folder, find each subsequent folder in folder_path_parts,
    finally returning the ID of the last folder. Optionally create the folders that don't
    exist yet
    """
    target_folder_name = folder_path_parts.pop(0)

    # Find the target folder if it already exists
    target_folder_id = None
    start_folder = box_client.folder(start_id)
    items = start_folder.get_items()
    for item in items:
        if item.type == "folder" and item.name == target_folder_name:
            target_folder_id = item.id

    # Otherwise create it
    if target_folder_id is None:
        if no_create:
            return None
        else:
            target_folder = start_folder.create_subfolder(target_folder_name)
            target_folder_id = target_folder.id

    # Return or recurse
    if len(folder_path_parts) == 0:
        return target_folder_id
    else:
        return _get_folder_id_recursion(
            folder_path_parts, start_id=target_folder_id, no_create=no_create
        )


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
    #folder path from root
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


def get_raw_data_path(
    file_name,
    path_from_nvdata=None,
    nvdata_dir=None,
):
    """Same as get_raw_data, but just returns the path to the file"""
    if nvdata_dir is None:
        nvdata_dir = common.get_nvdata_dir()
    if path_from_nvdata is None:
        path_from_nvdata = search_index.get_data_path_from_nvdata(file_name)
    data_dir = nvdata_dir / path_from_nvdata
    file_name_ext = "{}.txt".format(file_name)
    file_path = data_dir / file_name_ext
    return file_path



def _get_folder_path(folder_info):
    folder_path = [parent.name for parent in folder_info.path_collection["entries"][1:]]
    folder_path.append(folder_info.name)
    folder_path = "/".join(folder_path)
    return folder_path


# endregion

if __name__ == "__main__":
    # print(box_client.folder("235259643840").get().item_status)

    # reg_exp = r"nvdata\/pc_[a-zA-Z]*\/branch_[a-zA-Z]*\/.+\/2018_[0-9]{2}"
    # _delete_folders(reg_exp)

    _delete_empty_folders()
