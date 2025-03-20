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
from ftplib import FTP

from utils import common
from utils import indexer
from io import BytesIO

hostname = "192.168.1.197"
ftp = FTP(host=hostname,user=kolkowitzadmin,password="r8Y.>CL$y=P}X7^")

folder_path_cache = {}
try:
    ftp.login()
except:
    print(
        "\n"
        "FTP login failed"
        "\n"
    )


def download(file_name=None, ext=None):
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
    
    search_results = indexer.get_data_path(file_name)
    if not search_results:
        raise RuntimeError("No file found. Try running gen_search_index")
    search_command = "RETR " + search_results
    file_content = BytesIO()
    ftp.retrbinary(search_cmd, file_content.write)
    return file_content, file_name


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
    file_name = file_path_w_ext.name
    ftp.storbinary(f'STOR {folder_path}/{file_name}', content)



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

def quit():
    #close connection, should do this
    ftp.quit()
# endregion

if __name__ == "__main__":
    # print(box_client.folder("235259643840").get().item_status)

    # reg_exp = r"nvdata\/pc_[a-zA-Z]*\/branch_[a-zA-Z]*\/.+\/2018_[0-9]{2}"
    # _delete_folders(reg_exp)

    _delete_empty_folders()
