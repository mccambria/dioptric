# -*- coding: utf-8 -*-
"""
Tools for talking to our cloud provider, Box, which hosts all our data.
This file should only be accessed by the data_manager util

Created November 18th, 2023

@author: mccambria
"""

import time
from utils import common
from boxsdk import Client, JWTAuth

nvdata_folder_id = "235146666549"  # ID for the nvdata folder in Box
data_manager_folder = common.get_data_manager_folder()

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


def download(file_name, ext, file_id=None):
    """Download file from the cloud

    Parameters
    ----------
    file_name : str
        Name of the file, without file extension
    ext : str
        File extension

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
        except Exception as exc:
            raise RuntimeError("No file found with the passed file_name.")
    box_file = box_client.file(file_id)
    file_content = box_file.content()
    file_info = box_file.get()
    file_name = file_info.name.split(".")[0]
    return file_content, file_id, file_name


# def upload(folder_path, temp_file_path):
#     """Upload file to the cloud

#     Parameters
#     ----------
#     folder_path : Path
#         Folder path to upload to. Form should be folder1/folder2/... where folder1
#         is under directly nvdata
#     temp_file_path : Path
#         Full file path to write the file to before it can be uploaded to the cloud.
#         Get this by calling dm.get_file_path()
#     """
#     folder_id = get_folder_id(folder_path)
#     new_file = box_client.folder(folder_id).upload(str(temp_file_path))
#     return new_file.id


def upload(file_path_w_ext, content):
    """Upload file to the cloud

    Parameters
    ----------
    file_path : Path
        File path to upload to. Form should be folder1/folder2/... where folder1
        is under directly nvdata. Should include extension
    content : BytesIO
        Byte stream to write to the file
    """
    folder_path = file_path_w_ext.parent
    start = time.time()
    folder_id = get_folder_id(folder_path)
    stop = time.time()
    print(stop - start)
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
        is under directly nvdata

    Returns
    -------
    str
        ID of the folder
    """
    folder_path_parts = list(folder_path.parts)
    return _get_folder_id_recursion(folder_path_parts)


def _get_folder_id_recursion(folder_path_parts, start_id=nvdata_folder_id):
    """
    Starting from nvdata, find each subsequent folder in folder_path_parts, finally
    returning the ID of the last folder. Create the folders that don't exist yet
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
    pass
