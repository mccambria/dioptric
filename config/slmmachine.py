# -*- coding: utf-8 -*-
"""
Config file for Thorslm server

Created June 21st, 2024

@author: sbchand
"""

from pathlib import Path

# Set home directory path
home = Path.home()

# Configuration dictionary
config = {
    "shared_email": "kolkowitznvlab@gmail.com",
    "windows_repo_path": home / "Documents/dioptric",
    "linux_repo_path": home / "Documents/dioptric",
    "Servers": {
        "thorslm": "ThorslmServer",
    },
}

# Print configuration for verification
if __name__ == "__main__":
    for key, value in config.items():
        print(f"{key}: {value}")
