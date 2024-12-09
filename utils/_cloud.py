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


# SBC , New Version
def get_folder_id(folder_path, no_create=False):
    """Gets the Box ID of the specified folder. Optionally creates the folder if
    it does not exist yet.

    Parameters
    ----------
    folder_path : Path
        Folder path to ID. Form should be folder1/folder2/... where folder1
        is directly under the root data folder.

    Returns
    -------
    str
        ID of the folder
    """

    # Cache lookup
    global folder_path_cache
    if folder_path in folder_path_cache:
        # print(f"DEBUG: Cache hit for folder path: {folder_path}")
        return folder_path_cache[folder_path]

    # Resolve folder path
    folder_path_parts = list(folder_path.parts)
    folder_id = _get_folder_id_recursion(folder_path_parts, no_create=no_create)
    if not folder_id:
        raise ValueError(f"Failed to resolve folder path: {folder_path}")

    # Cache the result
    folder_path_cache[folder_path] = folder_id
    return folder_id


def _get_folder_id_recursion(
    folder_path_parts, start_id=root_folder_id, no_create=False
):
    """
    Starting from the root data folder, find each subsequent folder in folder_path_parts,
    finally returning the ID of the last folder. Optionally create the folders that don't
    exist yet.
    """
    if not folder_path_parts:
        raise ValueError("Invalid folder path: empty parts.")

    target_folder_name = folder_path_parts.pop(0)
    # print(f"DEBUG: Resolving folder '{target_folder_name}' under parent ID: {start_id}")

    # Find the target folder if it already exists
    target_folder_id = None
    start_folder = box_client.folder(start_id)
    items = start_folder.get_items()
    for item in items:
        if item.type == "folder" and item.name == target_folder_name:
            target_folder_id = item.id
            # print(
            #     f"DEBUG: Found folder '{target_folder_name}' with ID: {target_folder_id}"
            # )
            break

    # Create folder if not found
    if target_folder_id is None:
        if no_create:
            raise ValueError(
                f"Folder '{target_folder_name}' does not exist and creation is disabled (no_create=True)."
            )
        else:
            # print(
            #     f"DEBUG: Creating folder '{target_folder_name}' under parent ID: {start_id}"
            # )
            target_folder = start_folder.create_subfolder(target_folder_name)
            target_folder_id = target_folder.id
            # print(
            #     f"DEBUG: Created folder '{target_folder_name}' with ID: {target_folder_id}"
            # )

    # Recurse or return
    if folder_path_parts:
        return _get_folder_id_recursion(
            folder_path_parts, start_id=target_folder_id, no_create=no_create
        )
    else:
        return target_folder_id


# OLD MCC
# def get_folder_id(folder_path, no_create=False):
#     """Gets the Box ID of the specified folder. Optionally creates the folder if
#     it does not exist yet

#     Parameters
#     ----------
#     folder_path : Path
#         Folder path to ID. Form should be folder1/folder2/... where folder1
#         is directly under the root data folder

#     Returns
#     -------
#     str
#         ID of the folder
#     """

#     # See if the ID is stored in the cache
#     global folder_path_cache
#     if folder_path in folder_path_cache:
#         return folder_path_cache[folder_path]

#     # If it's not in the cache, look it up from the cloud
#     folder_path_parts = list(folder_path.parts)
#     folder_id = _get_folder_id_recursion(folder_path_parts, no_create)
#     folder_path_cache[folder_path] = folder_id
#     return folder_id


# def _get_folder_id_recursion(
#     folder_path_parts, start_id=root_folder_id, no_create=False
# ):
#     """
#     Starting from the root data folder, find each subsequent folder in folder_path_parts,
#     finally returning the ID of the last folder. Optionally create the folders that don't
#     exist yet
#     """
#     target_folder_name = folder_path_parts.pop(0)

#     # Find the target folder if it already exists
#     target_folder_id = None
#     start_folder = box_client.folder(start_id)
#     items = start_folder.get_items()
#     for item in items:
#         if item.type == "folder" and item.name == target_folder_name:
#             target_folder_id = item.id

#     # Otherwise create it
#     if target_folder_id is None:
#         if no_create:
#             return None
#         else:
#             target_folder = start_folder.create_subfolder(target_folder_name)
#             target_folder_id = target_folder.id

#     # Return or recurse
#     if len(folder_path_parts) == 0:
#         return target_folder_id
#     else:
#         return _get_folder_id_recursion(
#             folder_path_parts, start_id=target_folder_id, no_create=no_create
#         )


# region Delete functions, for cleaning up old data


def _delete_folders(reg_exp, start_id=root_folder_id):
    def condition_fn(folder_info):
        folder_path = _get_folder_path(folder_info)
        return re.fullmatch(reg_exp, folder_path)

    return _batch_delete(condition_fn, start_id)


def _delete_empty_folders(start_id=root_folder_id):
    def condition_fn(folder_info):
        return folder_info.item_collection["total_count"] == 0

    return _batch_delete(condition_fn, start_id)


def _batch_delete(condition_fn, folder_id):
    folder = box_client.folder(folder_id)
    folder_info = folder.get()

    # Delete this folder if it satisfies the passed condition
    if condition_fn(folder_info):
        folder.delete(recursive=True)
        folder_path = _get_folder_path(folder_info)
        print(f"{folder_id}: {folder_path}")
        return True

    # Otherwise recurse through the folder's contents and check for other folders to delete
    items = folder.get_items()
    for item in items:
        if item.type == "folder":
            res = _batch_delete(condition_fn, item.id)
            # Uncomment the next two lines to just delete the first item we find and quit
            # if res:
            #     return res


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
