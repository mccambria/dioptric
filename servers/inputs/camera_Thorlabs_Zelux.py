# -*- coding: utf-8 -*-
"""
Input server for the Thorlabs Zelux CMOS camera.

Created on June 25th, 2024

@author: Your Name

### BEGIN NODE INFO
[info]
name = camera_thorlabs_zelux
version = 1.0
description = Control server for Thorlabs Zelux CMOS camera using Thorlabs TLCameraSDK.
[startup]
cmdline = %PYTHON% %FILE%
timeout = 60
[shutdown]
message = 987654321
timeout = 5
### END NODE INFO
"""

import logging
import socket
import time
import numpy as np
import os
from labrad.server import LabradServer, setting

from slmsuite.hardware.cameras.camera import Camera
from utils import tool_belt as tb

def configure_path():
    absolute_path_to_dlls = "C:\\Users\\Saroj Chand\\Documents\\dioptric\\slmsuite\\hardware\\cameras\\dlls\\Native_64_lib"
    os.environ['PATH'] = absolute_path_to_dlls + os.pathsep + os.environ['PATH']
    try:
        os.add_dll_directory(absolute_path_to_dlls)
    except AttributeError:
        pass

try:
    configure_path()
except ImportError:
    configure_path = None

try:
    from thorlabs_tsi_sdk.tl_camera import TLCameraSDK, ROI
except ImportError:
    print("thorlabs.py: thorlabs_tsi_sdk not installed. Install to use Thorlabs cameras.")

class ThorCam(LabradServer):
    name = "camera_thorlabs_zelux"
    pc_name = socket.gethostname()
    sdk = None

    def initServer(self):
        tb.configure_logging(self)
        if ThorCam.sdk is None:
            try:
                ThorCam.sdk = TLCameraSDK()
            except Exception as e:
                raise RuntimeError(
                    "thorlabs.py: TLCameraSDK() open failed. "
                    "Is thorlabs_tsi_sdk installed? "
                    "Are the .dlls in the directory added by configure_tlcam_dll_path? "
                    "Sometimes adding the .dlls to the working directory can help."
                )
        
        camera_list = ThorCam.sdk.discover_available_cameras()
        if len(camera_list) == 0:
            raise RuntimeError("No cameras found by TLCameraSDK.")
        serial = camera_list[0]
        self.cam = ThorCam.sdk.open_camera(serial)
        self.cam.is_led_on = False
        print(f"Connected to camera with serial number: {serial }")
        
        self.width = self.cam.image_width_pixels
        self.height = self.cam.image_height_pixels
        self.bitdepth = self.cam.bit_depth
        self.dx_um = self.cam.sensor_pixel_width_um
        self.dy_um = self.cam.sensor_pixel_height_um

        self.profile = None
        self.setup("single")
        self.set_binning()

    def stopServer(self):
        self.reset(None)
        self.cam.dispose()

    @setting(9)
    def clear_buffer(self, c):
        self.cam.flushReadQueue()

    @setting(6)
    def get_detector_temp(self, c):
        return self.cam.get_detector_temp()

    @setting(0, num_images="i")
    def arm(self, c, num_images=0):
        self.clear_buffer()
        self.cam.open_shutter()
        self.cam.start(num_images)

    @setting(1)
    def disarm(self, c):
        self.cam.stop()
        self.cam.close_shutter()
        self.clear_buffer()

    @setting(2, returns="y")
    def read(self, c):
        if self.profile == "single":
            self.cam.issue_software_trigger()
        frame = self.cam.get_pending_frame_or_null()
        while frame is None:
            time.sleep(0.001)
            frame = self.cam.get_pending_frame_or_null()
        return np.copy(frame.image_buffer)

    @setting(5)
    def reset(self, c):
        self.disarm(c)

    @setting(8, readout_mode="i")
    def set_readout_mode(self, c, readout_mode):
        self.cam.stop()
        self.cam.set_readout_mode(readout_mode)

    @setting(7, returns="s")
    def get_readout_mode(self, c):
        return self.cam.get_readout_mode()

    @setting(10, returns="i")
    def get_num_readout_modes(self, c):
        return self.cam.get_num_readout_modes()

    @setting(11, exposure_time="v[]")
    def set_exposure_time(self, c, exposure_time):
        self.cam.exposure_time_us = int(exposure_time * 1e6)

    @setting(12, waiting_time="v[]")
    def set_waiting_time(self, c, waiting_time):
        self.cam.setWaitingTime(waiting_time)

    def get_frame_latency(self):
        return self.cam.get_frame_latency()

    def clear_roi(self):
        num_rois = self.cam.get_roi_count()
        if num_rois > 0:
            for ind in range(num_rois):
                self.cam.delete_roi(ind)

    def set_roi(self, offsetX, offsetY, width, height):
        self.cam.add_roi(offsetX, offsetY, width, height)
        self.cam.apply_roi()

__server__ = ThorCam()

if __name__ == "__main__":
    from labrad import util
    util.runServer(__server__)
