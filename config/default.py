# -*- coding: utf-8 -*-
"""
Default config file for laptops, home PCs, etc 

Created August 8th, 2023

@author: mccambria
"""

from utils.constants import ModMode, ControlMode, CountFormat
from utils.constants import CollectionMode, LaserKey, LaserPosMode
from pathlib import Path
import numpy as np

home = Path.home()

config = {
    ###
    "default_email": "kolkowitznvlab@gmail.com",
    "windows_nvdata_path": Path("E:/Shared drives/Kolkowitz Lab Group/nvdata"),
    "linux_nvdata_path": home / "E/nvdata",
    "windows_repo_path": home / "Documents/GitHub/dioptric",
    "linux_repo_path": home / "Documents/GitHub/dioptric",
    "camera_spot_radius": 6,
    "count_format": CountFormat.RAW,
}
