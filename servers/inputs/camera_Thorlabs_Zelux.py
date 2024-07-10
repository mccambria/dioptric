# -*- coding: utf-8 -*-
"""
Input server for the Thorlabs Zelux CMOS camera.

Created on June 25th, 2024

@author: Saroj B Chand

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
from labrad.types import Value
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

class ThorCam(LabradServer, Camera):
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
        logging.info(f"Connected to camera with serial number: {serial}")
        
        self.width = self.cam.image_width_pixels
        self.height = self.cam.image_height_pixels
        self.bitdepth = self.cam.bit_depth
        self.dx_um = self.cam.sensor_pixel_width_um
        self.dy_um = self.cam.sensor_pixel_height_um

        self.profile = None

    def stopServer(self):
        self.reset(None)
        self.cam.dispose()

    @setting(2)
    def close(self, c, close_sdk=False):
        """
        Close the camera connection.

        Parameters
        ----------
        close_sdk : bool
            Whether or not to close the TLCameraSDK instance.
        """
        if close_sdk:
            self.close_sdk(c)
            
        self.cam.dispose()

    @setting(3)
    def close_sdk(self, c):
        """
        Close the TLCameraSDK instance.
        """
        ThorCam.sdk.dispose()
        ThorCam.sdk = None

    @setting(4)
    def reset(self):
        """See :meth:`.Camera.reset`."""
        self.close()
        self.__init__()

    @setting(5, returns='s')
    def get_exposure(self):
        """See :meth:`.Camera.get_exposure`."""
        return str(float(self.cam.exposure_time_us) / 1e6)

    # @setting(6, exposure_s='v[s]', returns='')
    # def set_exposure(self, c, exposure_s):
    #     """Set the exposure time in seconds."""
    #     self.cam.exposure_time_us = int(exposure_s['s'] * 1e6)

    @setting(6, exposure_s='v[s]', returns='')
    def set_exposure(self, c, exposure_s):
        """Set the exposure time in seconds."""
        self.cam.exposure_time_us = int(exposure_s['s'] * 1e6)

    @setting(7)
    def set_binning(self, bx=None, by=None):
        """
        Set the binning of the camera. Will error if a certain binning is not supported.

        Parameters
        ----------
        bx : int
            The binning value in the horizontal direction.
        by : int
            The binning value in the vertical direction.
        """
        profile = self.profile
        self.setup(None)

        if bx is None:
            bx = 1
        if by is None:
            by = 1
        self.cam.binx = int(bx)
        self.cam.biny = int(by)

        self.setup(profile)

    @setting(8)
    def set_woi(self, woi=None):
        """See :meth:`.Camera.set_woi`."""
        profile = self.profile
        self.setup(None)

        if woi is None:
            woi = (
                self.cam.roi_range.upper_left_x_pixels_min,
                self.cam.roi_range.lower_right_x_pixels_max
                - self.cam.roi_range.upper_left_x_pixels_min + 1,
                self.cam.roi_range.upper_left_y_pixels_min,
                self.cam.roi_range.lower_right_y_pixels_max
                - self.cam.roi_range.upper_left_y_pixels_min + 1,
            )

        self.woi = woi

        newroi = ROI(
            self.cam.roi_range.lower_right_x_pixels_max - woi[0] - woi[1] + 1,
            woi[2],
            self.cam.roi_range.lower_right_x_pixels_max - woi[0],
            woi[2] + woi[3] - 1,
        )

        assert (
            self.cam.roi_range.upper_left_x_pixels_min
            <= newroi.upper_left_x_pixels
            <= self.cam.roi_range.upper_left_x_pixels_max
        )
        assert (
            self.cam.roi_range.upper_left_y_pixels_min
            <= newroi.upper_left_y_pixels
            <= self.cam.roi_range.upper_left_y_pixels_max
        )
        assert (
            self.cam.roi_range.lower_right_x_pixels_min
            <= newroi.lower_right_x_pixels
            <= self.cam.roi_range.lower_right_x_pixels_max
        )
        assert (
            self.cam.roi_range.lower_right_y_pixels_min
            <= newroi.lower_right_y_pixels
            <= self.cam.roi_range.lower_right_y_pixels_max
        )

        self.cam.roi = newroi
        self.woi = woi

        test = np.zeros((woi[3], woi[1]))
        self.shape = np.shape(self.transform(test))

        self.setup(profile)

        return woi

    @setting(9)
    def setup(self, profile):
        """
        Set operation mode.

        Parameters
        ----------
        profile
            See :attr:`profile`.
        """
        if profile != self.profile:
            if profile is None:
                self.cam.disarm()
            elif profile == "free":
                self.cam.disarm()
                self.cam.frames_per_trigger_zero_for_unlimited = 0
                self.cam.operation_mode = 0  # Software triggered
                self.cam.arm(2)
                self.cam.issue_software_trigger()
            elif profile == "single":
                self.cam.disarm()
                self.cam.frames_per_trigger_zero_for_unlimited = 1
                self.cam.operation_mode = 0  # Software triggered
                self.cam.arm(2)
            elif profile == "single_hardware":
                self.cam.disarm()
                self.cam.frames_per_trigger_zero_for_unlimited = 1
                self.cam.operation_mode = 1  # Hardware triggered
                self.cam.arm(2)
            else:
                raise ValueError("Profile {} not recognized".format(profile))

            self.profile = profile

    @setting(10, returns='*s')
    def info(self, c, verbose=True):
        """
        Retrieves information about the camera.

        Args:
            c (Context): The LabRAD context object.
            verbose (bool): Whether to print verbose information (default True).

        Returns:
            list: List of strings containing information about the camera.
        """
        info = [
            f"PC Name: {self.pc_name}",
            f"Image Width: {self.width}",
            f"Image Height: {self.height}",
            f"Bit Depth: {self.bitdepth}",
            f"Pixel Width (um): {self.dx_um}",
            f"Pixel Height (um): {self.dy_um}",
        ]
        if verbose:
            for line in info:
                print(line)
        return info

    
    @setting(11, timeout_s='v[s]', trigger='b', grab='b', attempts='i', returns='*v[s]')
    def get_image(self, c, timeout_s=0.1, trigger=True, grab=True, attempts=1):
        """
        See :meth:`.Camera.get_image`. By default ``trigger=True`` and ``grab=True`` which
        will result in blocking image acquisition.
        For non-blocking acquisition,
        set ``trigger=True`` and ``grab=False`` to issue a software trigger;
        then, call the method again with ``trigger=False`` and ``grab=True``
        to grab the resulting frame.

        Parameters
        ----------
        c : Context
            The LabRAD context object.
        timeout_s : float, optional
            Timeout in seconds for acquiring the image.
        trigger : bool, optional
            Whether or not to issue a software trigger.
        grab : bool, optional
            Whether or not to grab the frame (blocking).
        attempts : int, optional
            Number of attempts to try acquiring the image.

        Returns
        -------
        numpy.ndarray or None
            Array of shape :attr:`shape` if ``grab=True``, else ``None``.
        """
        should_trigger = trigger and self.profile == "single"

        for _ in range(attempts):
            if should_trigger:
                t = time.time()
                self.cam.issue_software_trigger()  # Issue software trigger via LabRAD command

            ret = None
            if grab:
                if not should_trigger:
                    t = time.time()

                frame = None

                while time.time() - t < timeout_s and frame is None:
                    frame = self.cam.get_pending_frame_or_null()  # Get frame via LabRAD command

                if frame is not None:
                    ret = self.transform(np.copy(frame.image_buffer))  # Process frame data
                    break  # Exit loop if frame is successfully acquired

        return ret


    @setting(12)
    def flush(self, timeout_s=1, verbose=False):
        """
        See :meth:`.Camera.flush`.

        Parameters
        ----------
        verbose : bool
            Whether or not to print extra information.
        """
        t = time.perf_counter()

        ii = 0
        frame = self.cam.get_pending_frame_or_null()
        frametime = 0

        while (
            time.perf_counter() - t < timeout_s
            and frame is not None
            and frametime < 0.003
        ):
            t2 = time.perf_counter()
            frame = self.cam.get_pending_frame_or_null()
            frametime = time.perf_counter() - t2
            ii += 1

        if verbose:
            print(
                "Flushed {} frames in {:.2f} ms".format(
                    ii, 1e3 * (time.perf_counter() - t)
                )
            )

    @setting(13, returns='b')
    def is_capturing(self):
        """
        Determine whether or not the camera is currently capturing images.

        Returns
        -------
        bool
            Whether or not the camera is actively capturing images.
        """
        return self.profile == "free"

    # @setting(14, returns='w')
    # def get_width(self, c):
    #     return self.width

__server__ = ThorCam()

if __name__ == "__main__":
    from labrad import util
    util.runServer(__server__)
