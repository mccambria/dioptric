# -*- coding: utf-8 -*-
"""
Virtual LabRAD server to run AODs continuously.

Created on January 17, 2025

@author: Saroj Chand

### BEGIN NODE INFO
[info]
name = aod_continuous_runs
version = 1.0
description = Virtual LabRAD server to control AODs.

[startup]
cmdline = %PYTHON% %FILE%
timeout = 20

[shutdown]
message = 987654321
timeout = 5
### END NODE INFO
"""

import logging
import socket

from labrad.server import LabradServer, setting
from qm import QuantumMachinesManager, qua

from servers.timing.sequencelibrary.QM_opx import seq_utils
from utils import common
from utils import tool_belt as tb

# Initialize the cache for running AODs
# _cache_macro_run_aods = {}

global _cache_macro_run_aods


class AODContinuousRunsServer(LabradServer):
    """Server for controlling AODs."""

    name = "aod_continuous_runs"
    pc_name = socket.gethostname()

    def initServer(self):
        """
        Initialize the server and AOD states.
        """
        tb.configure_logging(self)
        logging.info("AOD Continuous Runs Server started.")
        default_aod_suffices = None  # Adjust if needed
        default_amps = None  # Adjust if needed
        with qua.program():
            seq_utils.init()
            self.initialize_aods(default_aod_suffices, default_amps)

    def initialize_aods(self, aod_suffices=None, amps=None):
        """Initialize AODs using macro_run_aods."""
        try:
            # Get laser names dynamically from macro_run_aods
            laser_names = self.get_laser_names()
            if not laser_names:
                logging.warning("No lasers found to initialize AODs.")
                return
            logging.info(f"AODs initialized for lasers: {laser_names}")

            seq_utils.macro_run_aods(
                laser_names=laser_names, aod_suffices=aod_suffices, amps=amps
            )
            self.running_aods = {
                laser: {"suffix": aod_suffices, "amps": amps} for laser in laser_names
            }
        except Exception as e:
            logging.error(f"Error initializing AODs: {e}")

    def get_laser_names(self):
        """Retrieve laser names dynamically."""
        config = common.get_config_dict("purcell")
        positioners_dict = config["Positioning"]["Positioners"]
        laser_names = [
            tb.get_physical_laser_name(positioner["opti_virtual_laser_key"])
            for key, positioner in positioners_dict.items()
            if "aod" in positioner and positioner["aod"]
        ]
        return laser_names

    @setting(2, "List Running AODs", returns="*s")
    def list_running_aods(self, c):
        """List all currently running AODs."""
        return list(self.running_aods.keys())

    def stopServer(self):
        """Ensure all AODs are stopped when the server shuts down."""
        logging.info("Stopping all running AODs.")
        self.running_aods.clear()
        logging.info("AOD Continuous Runs Server stopped.")


__server__ = AODContinuousRunsServer()

if __name__ == "__main__":
    from labrad import util

    util.runServer(__server__)
