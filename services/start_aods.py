# -*- coding: utf-8 -*-
"""
Created on June 16th, 2023

@author: Saroj B Chand
"""

import logging
import os
import time

from utils import common

FLAG_PATH = r"C:\Users\matth\GitHub\dioptric\experiment_running.flag"
CHECK_INTERVAL = 15  # seconds


def run_aods():
    cxn = common.labrad_connect()
    opx = cxn.QM_opx
    analog_chans = [3, 4, 2, 6]  # Green x/y, Red x/y
    analog_volts = [0.11, 0.11, 0.15, 0.15]
    analog_freqs = [107.0, 107.0, 72.0, 72.0]
    last_status = None

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logging.info("AOD background runner started.")

    while True:
        try:
            experiment_running = os.path.exists(FLAG_PATH)

            if not experiment_running:
                opx.constant_ac([], analog_chans, analog_volts, analog_freqs)
                if last_status != "on":
                    logging.info("AODs running (background mode).")
                    last_status = "on"
            else:
                if last_status != "off":
                    logging.info("Experiment running. AOD background mode paused.")
                    last_status = "off"

        except Exception as e:
            logging.error(f"Error running AODs: {e}")
            last_status = None
            opx = None

        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    run_aods()
