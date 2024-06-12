# -*- coding: utf-8 -*-
"""
Default config file for laptops, home PCs, etc 

Created June 10th, 2024

@author: sbchand
"""

from utils.constants import ModMode, ControlMode, CountFormat
from utils.constants import CollectionMode, LaserKey, LaserPosMode
from pathlib import Path
import numpy as np

home = Path.home()

config = {
    ###
    "shared_email": "kolkowitznvlab@gmail.com",
    "windows_repo_path": home / "Documents/dioptric",
    "linux_repo_path": home / "Documents/dioptric",
    "nv_sig_units": "{'coords': 'V', 'expected_count_rate': 'kcps', 'durations': 'ns', 'magnet_angle': 'deg', 'resonance': 'GHz', 'rabi': 'ns', 'uwave_power': 'dBm'}",
}
