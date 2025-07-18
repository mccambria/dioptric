# -*- coding: utf-8 -*-
"""
Created on July 3rd, 2025

@author: Saroj B Chand

"""

import datetime
import os
import sys
import time

import nidaqmx
from nidaqmx.constants import TerminalConfiguration

with nidaqmx.Task() as task:
    task.ai_channels.add_ai_voltage_chan(
        "Dev2/ai1",  # your channel
        terminal_config=TerminalConfiguration.RSE,  # use RSE for BNC-2110
        min_val=-0.2,  #  Set low range
        max_val=0.2,
    )
    voltage = task.read()
    print(f"Measured voltage: {voltage:.6f} V")

# sys.exit()

# === USER SETTINGS ===
# LOG_INTERVAL = 15 * 60  # seconds between samples
LOG_INTERVAL = 15  # seconds between samples
DAQ_DEVICE = "Dev2"  # NI DAQ device name in NI MAX

# Responsivity and gain settings for 589 nm
# RESPONSIVITY_589NM = 0.43  # A/W
# GAIN_V_PER_A = 1000  # V/A from Thorlabs DET10A2
# CALIBRATION_FACTOR = 1000 / (GAIN_V_PER_A * RESPONSIVITY_589NM)  # mW/V

# Analog input channels to monitor
LASER_CHANNELS = {
    "589nm_fiber_out": "ai0",
    "589nm_laser_head_out": "ai1",
    # "reference": "ai3",
    # "638nm_back_reflection": "ai2",
}

CALIBRATION_FACTORS = {
    "589nm_fiber_out": 60.0,  # Empirical
    "589nm_laser_head_out": 65.0,  # Empirical
    # "reference": 550.0,  # Empirical
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
            min_val=-1.0,
            max_val=1.0,
        )
        voltage = task.read()
    return voltage


def main():
    while True:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] Logging laser power...")

        # Organize by month folder
        month_str = datetime.datetime.now().strftime("%m%Y")
        folder_path = os.path.join(BASE_FOLDER, month_str)
        os.makedirs(folder_path, exist_ok=True)

        for label, channel in LASER_CHANNELS.items():
            try:
                voltage = read_voltage(DAQ_DEVICE, channel)
                power_mW = CALIBRATION_FACTORS[label] * voltage

                filename = f"laser_{label}.csv"
                filepath = os.path.join(folder_path, filename)

                with open(filepath, "a") as f:
                    f.write(f"{timestamp},{voltage:.6f},{power_mW:.3f}\n")

                print(f"  {label}: {voltage:.6f} V → {power_mW:.3f} mW")

            except Exception as e:
                print(f"  Error reading {label} ({channel}): {e}")

        time.sleep(LOG_INTERVAL)


if __name__ == "__main__":
    main()
