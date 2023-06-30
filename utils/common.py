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


def get_config_module(pc_name=None):
    if pc_name is None:
        pc_name = socket.gethostname()
    module_name = f"config.{pc_name}"
    module = import_module(module_name)
    return module


def get_config_dict(pc_name=None):
    module = get_config_module(pc_name)
    return module.config


def get_default_email():
    config = get_config_dict()
    return config["default_email"]


def _get_os_config_val(key):
    os_name = platform.system()  # Windows or Linux
    os_name_lower = os_name.lower()
    config = get_config_dict()
    val = config[f"{os_name_lower}_{key}"]
    return val


def get_nvdata_path():
    """Returns an OS-dependent Path to the nvdata directory"""
    return _get_os_config_val("nvdata_path")


def get_repo_path():
    """Returns an OS-dependent Path to the repo directory"""
    return _get_os_config_val("repo_path")


def get_server(cxn, server_name):
    config = get_config_dict()
    dev_name = config["Servers"]["server_name"]
    return eval(f"cxn.{dev_name}")
