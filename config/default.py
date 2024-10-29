# -*- coding: utf-8 -*-
"""
Default config file for laptops, home PCs, etc 

Created August 8th, 2023

@author: mccambria
"""

from pathlib import Path

home = Path.home()

config = {
    ###
    "shared_email": "kolkowitznvlab@gmail.com",
    "windows_repo_path": home / "Documents/GitHub/dioptric",
    "linux_repo_path": home / "Documents/GitHub/dioptric",
    "nv_sig_units": "{'coords': 'V', 'expected_count_rate': 'kcps', 'durations': 'ns', 'magnet_angle': 'deg', 'resonance': 'GHz', 'rabi': 'ns', 'uwave_power': 'dBm'}",
}
