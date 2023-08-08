# -*- coding: utf-8 -*-
"""
Config file for laptop knuth

Created June 20th, 2023

@author: mccambria
"""

from pathlib import Path

home = Path.home()

config = {
    ###
    "default_email": "kolkowitznvlab@gmail.com",
    "windows_nvdata_path": Path("E:/Shared drives/Kolkowitz Lab Group/nvdata"),
    "linux_nvdata_path": home / "E/nvdata",
    "windows_repo_path": home / "Documents/GitHub/dioptric",
    "linux_repo_path": home / "Documents/GitHub/dioptric",
}
