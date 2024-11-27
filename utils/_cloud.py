# -*- coding: utf-8 -*-
"""
Tools for talking to our cloud provider, Box, which hosts all our data.
This file should only be accessed by the data_manager util

Created November 18th, 2023

@author: mccambria
"""

import time
from pathlib import Path

from boxsdk import Client, JWTAuth

from utils import common

# ID for the root data folder in Box, nvdata
root_folder_id = "235146666549"

data_manager_folder = common.get_data_manager_folder()
folder_path_cache = {}

try:
    box_auth_file_name = "dioptric_box_authorization.json"
    box_auth = JWTAuth.from_settings_file(data_manager_folder / box_auth_file_name)
    box_client = Client(box_auth)
except Exception as exc:
    print(
        "\n"
        f"Make sure you have the Box authorization file for dioptric in your "
        f"checkout of the GitHub repo. It should live here: {data_manager_folder}. "
        f"Create the folder if it doesn't exist yet. The file, {box_auth_file_name}, "
        f"can be found in the nvdata folder of the Kolkowitz group Box account."
        "\n"
    )
    raise exc


def download(file_name=None, ext=None, file_id=None):
    """Download file from the cloud

    Parameters
    ----------
    file_name : str
        Name of the file, without file extension
    ext : str
        File extension
    file_id : id
        Box file ID, can be read off from the URL:
        https://berkeley.app.box.com/file/<file_id>

    Returns
    -------
    Binary string
        Contents of the file
    """
    if file_id is None:
        search_results = box_client.search().query(
            f'"{file_name}"',
            type="file",
            limit=1,
            content_types=["name"],
            file_extensions=[ext],
        )
        try:
            match = next(search_results)
            file_id = match.id
        except Exception:
            raise RuntimeError("No file found with the passed file_name.")
    box_file = box_client.file(file_id)
    file_content = box_file.content()
    file_info = box_file.get()
    file_name = file_info.name.split(".")[0]
    return file_content, file_id, file_name


def upload(file_path_w_ext, content):
    """Upload file to the cloud

    Parameters
    ----------
    file_path : Path
        File path to upload to. Form should be folder1/folder2/... where folder1
        is under directly the root data folder. Should include extension
    content : BytesIO
        Byte stream to write to the file
    """
    folder_path = file_path_w_ext.parent
    folder_id = get_folder_id(folder_path)
    file_name = file_path_w_ext.name
    new_file = box_client.folder(folder_id).upload_stream(content, file_name)
    return new_file.id


def get_folder_id(folder_path):
    """Gets the Box ID of the specified folder. Creates the folder if it does
    not exist yet

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
    folder_id = _get_folder_id_recursion(folder_path_parts)
    folder_path_cache[folder_path] = folder_id
    return folder_id


def _get_folder_id_recursion(folder_path_parts, start_id=root_folder_id):
    """
    Starting from the root data folder, find each subsequent folder in folder_path_parts,
    finally returning the ID of the last folder. Create the folders that don't exist yet
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
        target_folder = start_folder.create_subfolder(target_folder_name)
        target_folder_id = target_folder.id

    # Return or recurse
    if len(folder_path_parts) == 0:
        return target_folder_id
    else:
        return _get_folder_id_recursion(folder_path_parts, start_id=target_folder_id)


if __name__ == "__main__":
    for ind in range(3):
        start = time.time()
        folder_id = get_folder_id(Path("pc_rabi/branch_master/rabi/2023_12"))
        stop = time.time()
        print(stop - start)
