# -*- coding: utf-8 -*-
"""
Created on June 30th, 2025

@author: Eric Gediman
@author: Saroj B Chand

"""

import datetime
import os
import time

from utils import common
from utils import tool_belt as tb

TEMP_CHANNELS = {
    "4A": b"4A?\n",
    "4B": b"4B?\n",
    "4C": b"4C?\n",
    "4D": b"4D?\n",
}

base_folder = "G:\\NV_Widefield_RT_Setup_Enclosure_Temp_Logs"

cxn = common.labrad_connect()
opx = cxn.temp_monitor_SRS_ptc10

LOG_INTERVAL = 15 * 60  # seconds between samples
LOG_INTERVAL = 5  # seconds between samples


def get_common_duration(key):
    config = common.get_config_dict()
    common_duration = config["CommonDurations"][key]
    return common_duration


while True:
    interval = LOG_INTERVAL
    month_str = datetime.datetime.now().strftime("%m%Y")
    folder_path = os.path.join(base_folder, month_str)
    os.makedirs(folder_path, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Reading Temperature at {timestamp} ...")
    for channel, cmd in TEMP_CHANNELS.items():
        try:
            temp = opx.get_temp(cmd)
            filename = f"temp_{channel}.csv"
            filepath = os.path.join(folder_path, filename)

            with open(filepath, "a") as file:
                file.write(f"{timestamp},{temp:.3f}\n")
                print(f"[{timestamp}] {channel}: {temp:.3f} °C")

        except Exception as e:
            print(f"Error reading {channel}: {e}")

    time.sleep(interval)
