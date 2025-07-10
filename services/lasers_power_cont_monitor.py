# -*- coding: utf-8 -*-
"""
Created on July 3rd, 2025

@author: Saroj B Chand, Eric Gediman

"""

import datetime
import os
import sys
import time

import nidaqmx
from nidaqmx.constants import TerminalConfiguration

with nidaqmx.Task() as task:
    task.ai_channels.add_ai_voltage_chan(
        "Dev2/ai0",  # your channel
        terminal_config=TerminalConfiguration.RSE,  # use RSE for BNC-2110
        min_val=-0.2,  #  Set low range
        max_val=0.2,
    )
    voltage = task.read()
    print(f"Measured voltage: {voltage:.6f} V")

sys.exit()

# === USER SETTINGS ===
LOG_INTERVAL = 15 * 60  # seconds between samples
DAQ_DEVICE = "Dev2"  # NI DAQ device name in NI MAX

# Responsivity and gain settings for 589 nm
# RESPONSIVITY_589NM = 0.43  # A/W
# GAIN_V_PER_A = 1000  # V/A from Thorlabs DET10A2
# CALIBRATION_FACTOR = 1000 / (GAIN_V_PER_A * RESPONSIVITY_589NM)  # mW/V

# Analog input channels to monitor
LASER_CHANNELS = {
    "589nm_fiber_out": "ai0",
    "589nm_laser_head_out": "ai1",
    "reference": "ai3",
    # "638nm_back_reflection": "ai2",
}

CALIBRATION_FACTORS = {
    "589nm_fiber_out": 60.0,  # Empirical
    "589nm_laser_head_out": 550.0,  # Empirical
    "reference": 550.0,  # Empirical
    # "405nm_probe": 71.8,
}

# Base log folder
BASE_FOLDER = "G:\\NV_Widefield_RT_Setup_Lasers_Power_Logs"


# === LOGGER LOOP ===
def read_voltage(dev, channel):
    full_channel = f"{dev}/{channel}"
    with nidaqmx.Task() as task:
        task.ai_channels.add_ai_voltage_chan(
            full_channel,
            terminal_config=TerminalConfiguration.RSE,
            min_val=-0.2,
            max_val=0.2,
        )
        voltage = task.read()
    return voltage


def main():

    filepath = "SET HERE"
    channel = "SET HERE"
    label = "SET HERE"
    with open(filepath, "a") as f:
        while True:        
            timestamp = time.time_NS()
            voltage = read_voltage(DAQ_DEVICE, channel)

            try:   
                f.write(f"{timestamp},{voltage:.6f},\n")

            except Exception as e:
                print(f"  Error reading {label} ({channel}): {e}")

if __name__ == "__main__":
    main()
