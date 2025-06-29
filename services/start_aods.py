import logging
import os
import time

from utils import common

FLAG_PATH = r"C:\Users\matth\GitHub\dioptric\experiment_running.flag"
CHECK_INTERVAL = 10  # seconds


def run_aods():
    cxn = common.labrad_connect()
    opx = cxn.QM_opx

    analog_chans = [3, 4, 2, 6]  # Green x/y, Red x/y
    analog_volts = [0.11, 0.11, 0.15, 0.15]
    analog_freqs = [107.0, 107.0, 72.0, 72.0]

    last_status = None  # Track whether AODs were on or off last time

    logging.basicConfig(level=logging.INFO)
    logging.info("🟢🔴 AOD background runner started.")

    while True:
        try:
            experiment_running = os.path.exists(FLAG_PATH)

            if not experiment_running and last_status != "on":
                opx.constant_ac([], analog_chans, analog_volts, analog_freqs)
                logging.info("🟢🔴 AODs running (background mode).")
                last_status = "on"

            elif experiment_running and last_status != "off":
                logging.info("Experiment running. AOD background mode paused.")
                last_status = "off"

        except Exception as e:
            logging.error(f"Error running AODs: {e}")
            last_status = None  # Force retry on next loop

        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    run_aods()
