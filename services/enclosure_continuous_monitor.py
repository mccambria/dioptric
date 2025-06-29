import datetime
import os
import time

from utils import common
from utils import tool_belt as tb

TEMP_CHANNELS = {
    "4A": b"4A?\n",
    "4B": b"4B?\n",
    "4C": b"4C?\n",
}

base_folder = "G:\\Enclosure_Temp"

cxn = common.labrad_connect()
opx = cxn.temp_monitor_SRS_ptc10


def get_common_duration(key):
    config = common.get_config_dict()
    common_duration = config["CommonDurations"][key]
    return common_duration


while True:
    interval = get_common_duration("temp_reading_interval")
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
