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
from importlib import import_module
import sys


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


def get_config_module():
    pc_name = socket.gethostname()
    module_name = f"config.{pc_name}"
    module = import_module(module_name)  
    return module


def get_config_dict():
    module = get_config_module() 
    return module.config

def get_server(cxn, server_name):
    config = get_config_dict()
    dev_name = config["Servers"]["server_name"]
    return eval(f"cxn.{dev_name}")
