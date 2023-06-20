# -*- coding: utf-8 -*-
"""
Functions, etc to be referenced only by other utils. If you're running into
a circular reference in utils, put the function or whatever here. 

Created September 10th, 2021

@author: mccambria
"""

import platform
from pathlib import Path
import socket
import json

### Lab-specific stuff here

shared_email = "kolkowitznvlab@gmail.com"
home = Path.home()
windows_nvdata_path = Path("E:/Shared drives/Kolkowitz Lab Group/nvdata")
linux_nvdata_path = home / "E/nvdata"
windows_repo_path = home / "Documents/GitHub/dioptric"
linux_repo_path = home / "Documents/GitHub/dioptric"

###


def get_nvdata_path():
    """Returns an OS-dependent Path to the nvdata directory (configured above)"""
    os_name = platform.system()
    if os_name == "Windows":
        nvdata_dir = windows_nvdata_path
    elif os_name == "Linux":
        nvdata_dir = linux_nvdata_path

    return nvdata_dir


def get_repo_path():
    """Returns an OS-dependent Path to the repo directory (configured above)"""
    os_name = platform.system()
    if os_name == "Windows":
        nvdata_dir = windows_repo_path
    elif os_name == "Linux":
        nvdata_dir = linux_repo_path

    return nvdata_dir


def get_config_dict():
    repo_path = get_repo_path()
    pc_name = socket.gethostname()
    json_path = repo_path / f"{pc_name}.json"
    with json_path.open() as f:
        res = json.load(f)
        return res
