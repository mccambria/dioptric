# -*- coding: utf-8 -*-
"""
Output server for the PI nanocube P606.3C piezo.
Sending commands over usb

Created on Wed July  3 15:58:30 2024

@author: saroj b chand

### BEGIN NODE INFO
[info]
name = pos_xyz_PI_p616_3c
version = 1.0
description =

[startup]
cmdline = %PYTHON% %FILE%
timeout = 30

[shutdown]
message = 987654321
timeout =
### END NODE INFO
"""

import logging
import socket
import time
from pathlib import Path

import nidaqmx
import nidaqmx.stream_writers as stream_writers
import numpy
from labrad.server import LabradServer, setting
from numpy.polynomial.polynomial import Polynomial
from pipython import GCSDevice, GCSError
from twisted.internet.defer import ensureDeferred

from utils import common
from utils import tool_belt as tb


class PosXyzPiP6163c(LabradServer):
    name = "pos_xyz_PI_p616_3c"
    pc_name = socket.gethostname()

    def initServer(self):
        tb.configure_logging(self)
        self.task = None
        self.sub_init_server_xyz()

    def sub_init_server_xyz(self):
        """Sub-routine to be called by xyz server"""
        self.x_last_position = None
        self.x_current_direction = None
        self.x_last_turning_position = None
        self.y_last_position = None
        self.y_current_direction = None
        self.y_last_turning_position = None
        self.z_last_position = None
        self.z_current_direction = None
        self.z_last_turning_position = None

        config = common.get_config_dict()
        gcs_dll_path = config["DeviceIDs"]["gcs_dll_path"]
        model = config["DeviceIDs"]["piezo_controller_E727_model"]
        serial = config["DeviceIDs"]["piezo_controller_E727_serial"]

        self.piezo = GCSDevice(devname=model, gcsdll=gcs_dll_path)
        self.piezo.ConnectUSB(serial)

        # Set command level with password 'advanced'
        try:
            command_level_password = "advanced"
            self.piezo.CCL(1, command_level_password)
        except GCSError as gcse:
            logging.error(f"GCSError during setting command level: {gcse}")
            raise
        self.axis_x = self.piezo.axes[0]
        self.axis_y = self.piezo.axes[1]
        self.axis_z = self.piezo.axes[2]
        axes_list = [self.axis_x, self.axis_y, self.axis_z]

        # Log responses from specific commands with separators
        # try:
        #     idn = self.piezo.qIDN()
        #     logging.info("----- DEVICE IDENTIFICATION -----")
        #     logging.info(f"IDN: {idn}")

        #     ver = self.piezo.qVER()
        #     logging.info("----- DEVICE VERSION -----")
        #     logging.info(f"VER: {ver}")

        #     cst = self.piezo.qCST()
        #     logging.info("----- CONTROLLER STATUS -----")
        #     logging.info(f"CST: {cst}")

        #     hpa = self.piezo.qHPA()
        #     logging.info("----- HARDWARE PARAMETERS -----")
        #     logging.info(f"HPA: {hpa}")

        #     spa = self.piezo.qSPA()
        #     logging.info("----- SYSTEM PARAMETERS -----")
        #     logging.info(f"SPA: {spa}")

        # except GCSError as gcse:
        #     logging.error(f"GCSError during command logging: {gcse}")

        # Select the control algorithm for closed-loop operation
        # control_mode = 1  # 0 = None, 1 = PID, 2 = APC (if licensed)
        # for ax in [self.axis_x, self.axis_y, self.axis_z]:
        #     self.piezo.SVO(ax, control_mode)

        # self.get_servo_state(self.axis_x)
        # self.get_servo_state(self.axis_y)
        # self.get_servo_state(self.axis_z)
        #
        # for ax in axes_list:
        #     input_mode = self.piezo.qSPA(ax, 0x02010000)
        #     logging.info(
        #         f"Axis {ax} input mode: {'Analog' if input_mode == 1 else 'Digital'}"
        #     )

        #     analog_channel = self.piezo.qSPA(ax, 0x06000500)
        #     logging.info(f"Axis {ax} is using analog input channel: {analog_channel}")
        # # for ax in axes_list:
        #     self.check_overflow_state(ax)

        # self.perform_auto_zero()

        # for ax in axes_list:
        #     self.check_overflow_state(ax)

        config_wiring_daq = config["Wiring"]["Piezo_Controller_E727"]
        self.daq_voltage_range_factor = config_wiring_daq["voltage_range_factor"]

        config_wiring_piezo = config["Wiring"]["Piezo_Controller_E727"]
        self.piezo_stage_voltage_range_factor = config_wiring_piezo[
            "voltage_range_factor"
        ]
        self.piezo_stage_scaling_offset = config_wiring_piezo["scaling_offset"]
        self.piezo_stage_scaling_gain = config_wiring_piezo["scaling_gain"]
        self.piezo_stage_channel_x = config_wiring_piezo["piezo_controller_channel_x"]
        self.piezo_stage_channel_y = config_wiring_piezo["piezo_controller_channel_y"]
        self.piezo_stage_channel_z = config_wiring_piezo["piezo_controller_channel_z"]
        channels_list = [
            self.piezo_stage_channel_x,
            self.piezo_stage_channel_y,
            self.piezo_stage_channel_z,
        ]

        # Determine the psvrf_value based on the voltage range factor
        if self.piezo_stage_voltage_range_factor == 5.0:
            psvrf_value = 1
        elif self.piezo_stage_voltage_range_factor == 10.0:
            psvrf_value = 2
        else:
            logging.debug("Piezo stage voltage range factor must be either 5.0 or 10.0")
            raise ValueError(
                "Piezo stage voltage range factor must be either 5.0 or 10.0"
            )

        for ax in axes_list:
            self.piezo.SPA(ax, 0x02000100, psvrf_value)
        logging.debug(
            f"Piezo stage voltage range factor set to: {self.piezo_stage_voltage_range_factor}"
        )

        # Set scaling parameters for the piezo stage
        for ax in axes_list:
            self.piezo.SPA(ax, 0x02000200, self.piezo_stage_scaling_offset)  # offset
            self.piezo.SPA(ax, 0x02000300, self.piezo_stage_scaling_gain)  # gain

        logging.debug(
            f"Piezo stage scaling OFFSET set to: {self.piezo_stage_scaling_offset}"
        )
        logging.debug(
            f"Piezo stage scaling GAIN set to: {self.piezo_stage_scaling_gain}"
        )

        # Connect the controller's axes to the incoming signals
        for ax, channel in zip(axes_list, channels_list):
            self.piezo.SPA(ax, 0x06000500, channel)
            logging.debug(f"Piezo axis {ax} connected to channel {channel}")

        self.daq_ao_piezo_stage_x = config["Wiring"]["Daq"]["ao_piezo_stage_P616_3c_x"]
        self.daq_ao_piezo_stage_y = config["Wiring"]["Daq"]["ao_piezo_stage_P616_3c_y"]
        self.daq_ao_piezo_stage_z = config["Wiring"]["Daq"]["ao_piezo_stage_P616_3c_z"]
        self.daq_di_clock = config["Wiring"]["Daq"]["di_clock"]

        # # hysteris parameters xy
        self.xy_hysteresis_b = 0.9410070861592449
        self.xy_hysteresis_a = 1 - self.xy_hysteresis_b
        # for z
        linearity = 1
        self.z_hysteresis_b = linearity
        self.z_hysteresis_a = 1 - self.z_hysteresis_b

        logging.info("Initialization Complete")

    def get_servo_state(self, axis):
        """
        Gets the current servo state for the given axis.
        Args:
            axis: The axis ID to query.
        Returns:
            The current servo state.
        """
        servo_state = self.piezo.qSVO(axis)
        logging.info(
            f"Servo state for axis {axis} is {'enabled' if servo_state else 'disabled'}."
        )
        return servo_state

    def check_overflow_state(self, axis):
        """
        Checks if the given axis is in overflow state.
        Args:
            axis: The axis ID to query.
        Returns:
            Boolean indicating whether the axis is in overflow state.
        """
        overflow_state = self.piezo.qOVF(axis)
        logging.info(
            f"Overflow state for axis {axis} is {'detected' if overflow_state else 'not detected'}."
        )
        return overflow_state

    def perform_auto_zero(self):
        """Perform AutoZero procedure for all axes."""
        axes = [self.axis_x, self.axis_y, self.axis_z]
        try:
            for ax in axes:
                self.piezo.ATZ(ax, 0.0)
                time.sleep(5)
            # Save parameters to non-volatile memory (example command, adapt as needed)
            # self.piezo.send("SAV 0x02000200")
            # self.piezo.send("SAV 0x02000102")

            logging.info("AutoZero procedure completed and parameters saved.")

        except GCSError as gcse:
            logging.error(f"GCSError occurred: {gcse}")
        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}")

        # %%

    def compensate_hysteresis(self, position, axis, apply_compensation=False):
        """
        Compensate for hysteresis using a quadratic model on the specified axis (X, Y, or Z).

        Parameters
        ----------
        position : float or ndarray(float)
            Position (in this case the nominal voltage) the user intends
            to move to for a linear response without hysteresis.

        axis : str
            The axis to compensate ('x', 'y', or 'z').

        apply_compensation : bool, optional
            Flag to determine whether to apply hysteresis compensation. Default is True.

        Returns
        -------
        float or ndarray(float)
            Compensated voltage to set.
        """

        if not apply_compensation:
            return position

        # Define hysteresis coefficients and state variables for each axis
        if axis == "x":
            a = self.xy_hysteresis_a
            b = self.xy_hysteresis_b
            last_position = self.x_last_position
            current_direction = self.x_current_direction
            last_turning_position = self.x_last_turning_position
        elif axis == "y":
            a = self.xy_hysteresis_a
            b = self.xy_hysteresis_b
            last_position = self.y_last_position
            current_direction = self.y_current_direction
            last_turning_position = self.y_last_turning_position
        elif axis == "z":
            a = self.z_hysteresis_a
            b = self.z_hysteresis_b
            last_position = self.z_last_position
            current_direction = self.z_current_direction
            last_turning_position = self.z_last_turning_position
        else:
            raise ValueError("Invalid axis. Choose 'x', 'y', or 'z'.")

        # If coefficients are not provided, assume no hysteresis compensation is needed
        if a == 0 and b == 0:
            return position

        single_value = False
        if not isinstance(position, (numpy.ndarray, list)):
            single_value = True
            position = [position]

        # Initialize state variables if they are not set
        if None in [last_position, current_direction, last_turning_position]:
            last_position = position[0]
            current_direction = +1
            last_turning_position = position[0]

        compensated_voltage = []
        for val in position:
            movement_direction = numpy.sign(val - last_position)
            if movement_direction == 0:
                movement_direction = current_direction
            elif movement_direction == -current_direction:
                last_turning_position = last_position
                current_direction = movement_direction

            abs_p = abs(val - last_turning_position)
            # Calculate compensated voltage based on the quadratic model
            if a != 0:
                discriminant = b**2 - 4 * a * (0 - abs_p)
                if discriminant < 0:
                    raise ValueError(
                        "No real roots found for quadratic equation. Adjust hysteresis parameters."
                    )
                v = (-b + numpy.sqrt(discriminant)) / (2 * a)
            else:
                v = (val - last_turning_position) / b

            result = last_turning_position + (movement_direction * v)
            compensated_voltage.append(result)

            last_position = val

        # Update state variables
        if axis == "x":
            self.x_last_position = last_position
            self.x_current_direction = movement_direction
            self.x_last_turning_position = last_turning_position
        elif axis == "y":
            self.y_last_position = last_position
            self.y_current_direction = movement_direction
            self.y_last_turning_position = last_turning_position
        elif axis == "z":
            self.z_last_position = last_position
            self.z_current_direction = movement_direction
            self.z_last_turning_position = last_turning_position

        return (
            compensated_voltage[0] if single_value else numpy.array(compensated_voltage)
        )

    def load_stream_writer_xy(self, c, task_name, voltages, period):
        # Close the existing task if there is one
        if self.task is not None:
            self.close_task_internal()

        # Write the initial voltages and stream the rest
        num_voltages = voltages.shape[1]
        self.write_xy(c, voltages[0, 0], voltages[1, 0])
        stream_voltages = voltages[:, 1:num_voltages]
        # Compensate hysteresis for both x and y axes
        compensated_voltages = numpy.vstack(
            self.compensate_hysteresis(stream_voltages[0], "x"),
            self.compensate_hysteresis(stream_voltages[1], "y"),
        )
        stream_voltages = numpy.ascontiguousarray(compensated_voltages)
        num_stream_voltages = num_voltages - 1
        # Create a new task
        task = nidaqmx.Task(task_name)
        self.task = task

        # Set up the output channels
        # task.ao_channels.add_ao_voltage_chan(
        #     self.daq_ao_piezo_stage_x, min_val=-self.daq_voltage_range_factor,
        #                                 max_val=self.daq_voltage_range_factor
        # )
        # task.ao_channels.add_ao_voltage_chan(
        #     self.daq_ao_piezo_stage_y, min_val=-self.daq_voltage_range_factor,
        #                                 max_val=self.daq_voltage_range_factor
        # )

        channel_0 = task.ao_channels.add_ao_voltage_chan(
            self.daq_ao_piezo_stage_x,
            min_val=-self.daq_voltage_range_factor,
            max_val=self.daq_voltage_range_factor,
        )
        channel_1 = task.ao_channels.add_ao_voltage_chan(
            self.daq_ao_piezo_stage_y,
            min_val=-self.daq_voltage_range_factor,
            max_val=self.daq_voltage_range_factor,
        )

        # Set the daq reference value to either 5 or 10 V
        channel_0.ao_dac_ref_val = self.daq_voltage_range_factor
        channel_1.ao_dac_ref_val = self.daq_voltage_range_factor
        # Set up the output stream
        output_stream = nidaqmx.task.OutStream(task)
        writer = stream_writers.AnalogMultiChannelWriter(output_stream)

        # Configure the sample to advance on the rising edge of the PFI input.
        # The frequency specified is just the max expected rate in this case.
        # We'll stop once we've run all the samples.
        freq = float(1 / (period * (10**-9)))  # freq in seconds as a float

        task.timing.cfg_samp_clk_timing(
            freq, source=self.daq_di_clock, samps_per_chan=num_stream_voltages
        )

        writer.write_many_sample(stream_voltages)

        # Close the task once we've written all the samples
        task.register_done_event(self.close_task_internal)

        task.start()

    def load_stream_writer_xyz(self, c, task_name, voltages, period):
        # Close the existing task if there is one
        if self.task is not None:
            self.close_task_internal()

        # Write the initial voltages and stream the rest
        num_voltages = voltages.shape[1]
        self.write_xyz(c, voltages[0, 0], voltages[1, 0], voltages[2, 0])
        stream_voltages = voltages[:, 1:num_voltages]
        # Compensate hysteresis for both x and y axes
        compensated_voltages = numpy.vstack(
            self.compensate_hysteresis(stream_voltages[0], "x"),
            self.compensate_hysteresis(stream_voltages[1], "y"),
            self.compensate_hysteresis(stream_voltages[2], "z"),
        )
        stream_voltages = numpy.ascontiguousarray(compensated_voltages)
        num_stream_voltages = num_voltages - 1
        # Create a new task
        task = nidaqmx.Task(task_name)
        self.task = task

        channel_0 = task.ao_channels.add_ao_voltage_chan(
            self.daq_ao_piezo_stage_x,
            min_val=-self.daq_voltage_range_factor,
            max_val=self.daq_voltage_range_factor,
        )
        channel_1 = task.ao_channels.add_ao_voltage_chan(
            self.daq_ao_piezo_stage_y,
            min_val=-self.daq_voltage_range_factor,
            max_val=self.daq_voltage_range_factor,
        )
        channel_2 = task.ao_channels.add_ao_voltage_chan(
            self.daq_ao_piezo_stage_z,
            min_val=-self.daq_voltage_range_factor,
            max_val=self.daq_voltage_range_factor,
        )

        # Set the daq reference value to either 5 or 10 V
        channel_0.ao_dac_ref_val = self.daq_voltage_range_factor
        channel_1.ao_dac_ref_val = self.daq_voltage_range_factor
        channel_2.ao_dac_ref_val = self.daq_voltage_range_factor

        # Set up the output stream
        output_stream = nidaqmx.task.OutStream(task)
        writer = stream_writers.AnalogMultiChannelWriter(output_stream)

        # Configure the sample to advance on the rising edge of the PFI input.
        # The frequency specified is just the max expected rate in this case.
        # We'll stop once we've run all the samples.
        freq = float(1 / (period * (10**-9)))  # freq in seconds as a float

        task.timing.cfg_samp_clk_timing(
            freq, source=self.daq_di_clock, samps_per_chan=num_stream_voltages
        )

        writer.write_many_sample(stream_voltages)

        # Close the task once we've written all the samples
        task.register_done_event(self.close_task_internal)

        task.start()

    def close_task_internal(self, task_handle=None, status=None, callback_data=None):
        task = self.task
        if task is not None:
            task.close()
            self.task = None
        return 0

    @setting(70, xVoltage="v[]")  # Voltage value for X-axis
    def write_x(self, c, xVoltage):
        """Write the specified x voltage to the piezo stage"""

        # Close the existing task if there is one
        if self.task is not None:
            self.close_task_internal()
        # Adjust voltage turn for hysteresis
        xVoltage = self.compensate_hysteresis(xVoltage, "x")
        # Create a new task to write the voltage
        with nidaqmx.Task() as task:
            # Set up the output channel for X
            channel_0 = task.ao_channels.add_ao_voltage_chan(
                self.daq_ao_piezo_stage_x,
                min_val=-self.daq_voltage_range_factor,
                max_val=self.daq_voltage_range_factor,
            )
            channel_0.ao_dac_ref_val = self.daq_voltage_range_factor

            # Write the specified voltage to the channel
            task.write([xVoltage])
            # logging.debug(f"X-axis voltage set to {xVoltage} V")

    @setting(71, yVoltage="v[]")  # Voltage value for Y-axis
    def write_y(self, c, yVoltage):
        """Write the specified y voltage to the piezo stage"""

        # Close the existing task if there is one
        if self.task is not None:
            self.close_task_internal()
        # Adjust voltage turn for hysteresis
        yVoltage = self.compensate_hysteresis(yVoltage, "y")
        # Create a new task to write the voltage
        with nidaqmx.Task() as task:
            # Set up the output channel for Y
            channel_0 = task.ao_channels.add_ao_voltage_chan(
                self.daq_ao_piezo_stage_y,
                min_val=-self.daq_voltage_range_factor,
                max_val=self.daq_voltage_range_factor,
            )
            channel_0.ao_dac_ref_val = self.daq_voltage_range_factor

            # Write the specified voltage to the channel
            task.write([yVoltage])
            # logging.debug(f"Y-axis voltage set to {yVoltage} V")

    @setting(72, zVoltage="v[]")  # Voltage value for Z-axis
    def write_z(self, c, zVoltage):
        """Write the specified z voltage to the piezo stage"""

        # Close the existing task if there is one
        if self.task is not None:
            self.close_task_internal()
        # Adjust voltage turn for hysteresis
        zVoltage = self.compensate_hysteresis(zVoltage, "z")
        # Create a new task to write the voltage
        with nidaqmx.Task() as task:
            # Set up the output channel for Z
            channel_2 = task.ao_channels.add_ao_voltage_chan(
                self.daq_ao_piezo_stage_z,
                min_val=-self.daq_voltage_range_factor,
                max_val=self.daq_voltage_range_factor,
            )
            channel_2.ao_dac_ref_val = self.daq_voltage_range_factor

            task.write([zVoltage])
            # logging.debug(f"Z-axis voltage set to {zVoltage} V")

    @setting(32, xVoltage="v", yVoltage="v")
    def write_xy(self, c, xVoltage, yVoltage):
        """Write the specified x and y voltages to the piezo stage"""

        # Close the stream task if it exists
        if self.task is not None:
            self.close_task_internal()
        # Compensate hysteresis for both x and y axes
        xVoltage = self.compensate_hysteresis(xVoltage, "x")
        yVoltage = self.compensate_hysteresis(yVoltage, "y")

        with nidaqmx.Task() as task:
            # Set up the output channels
            task.ao_channels.add_ao_voltage_chan(
                self.daq_ao_piezo_stage_x,
                min_val=-self.daq_voltage_range_factor,
                max_val=self.daq_voltage_range_factor,
            )

            task.ao_channels.add_ao_voltage_chan(
                self.daq_ao_piezo_stage_y,
                min_val=-self.daq_voltage_range_factor,
                max_val=self.daq_voltage_range_factor,
            )

            task.write([xVoltage, yVoltage])

    @setting(42, xVoltage="v[]", yVoltage="v[]", zVoltage="v[]")
    def write_xyz(self, c, xVoltage, yVoltage, zVoltage):
        """Write the specified x and y and z voltages to the piezo stage"""

        # Close the stream task if it exists
        # This can happen if we quit out early
        if self.task is not None:
            self.close_task_internal()
        # Compensate hysteresis for both x and y axes
        xVoltage = self.compensate_hysteresis(xVoltage, "x")
        yVoltage = self.compensate_hysteresis(yVoltage, "y")
        zVoltage = self.compensate_hysteresis(yVoltage, "z")

        with nidaqmx.Task() as task:
            # Set up the output channels
            channel_0 = task.ao_channels.add_ao_voltage_chan(
                self.daq_ao_piezo_stage_x,
                min_val=-self.daq_voltage_range_factor,
                max_val=self.daq_voltage_range_factor,
            )

            channel_1 = task.ao_channels.add_ao_voltage_chan(
                self.daq_ao_piezo_stage_y,
                min_val=-self.daq_voltage_range_factor,
                max_val=self.daq_voltage_range_factor,
            )
            channel_2 = task.ao_channels.add_ao_voltage_chan(
                self.daq_ao_piezo_stage_z,
                min_val=-self.daq_voltage_range_factor,
                max_val=self.daq_voltage_range_factor,
            )

            channel_0.ao_dac_ref_val = self.daq_voltage_range_factor
            channel_1.ao_dac_ref_val = self.daq_voltage_range_factor
            channel_2.ao_dac_ref_val = self.daq_voltage_range_factor

            task.write([xVoltage, yVoltage, zVoltage])

    @setting(31, returns="*v[]")
    def read_xy(self, c):
        """Return the current voltages on the piezo's DAQ channels"""
        with nidaqmx.Task() as task:
            # Set up the internal channels - to do: actual parsing...
            if self.daq_ao_piezo_stage_x == "dev1/AO0":
                chan_name = "dev1/_ao0_vs_aognd"
            task.ai_channels.add_ai_voltage_chan(
                chan_name,
                min_val=-self.daq_voltage_range_factor,
                max_val=self.daq_voltage_range_factor,
            )
            if self.daq_ao_piezo_stage_y == "dev1/AO1":
                chan_name = "dev1/_ao1_vs_aognd"
            task.ai_channels.add_ai_voltage_chan(
                chan_name,
                min_val=-self.daq_voltage_range_factor,
                max_val=self.daq_voltage_range_factor,
            )
            voltages = task.read()

        return voltages[0], voltages[1]

    @setting(
        2,
        x_center="v[]",
        y_center="v[]",
        z_center="v[]",
        x_range="v[]",
        y_range="v[]",
        z_range="v[]",
        num_steps="i",
        period="i",
        returns="*v[]*v[]",
    )
    def read_xyz(
        self,
        c,
        x_center,
        y_center,
        z_center,
        x_range,
        y_range,
        z_range,
        num_steps,
        period,
    ):
        """Return the current voltages on the piezo's DAQ channels"""
        with nidaqmx.Task() as task:
            # Set up the internal channels - to do: actual parsing...
            if self.daq_ao_piezo_stage_x == "dev1/AO0":
                chan_name = "dev1/_ao0_vs_aognd"
            task.ai_channels.add_ai_voltage_chan(
                chan_name,
                min_val=-self.daq_voltage_range_factor,
                max_val=self.daq_voltage_range_factor,
            )
            if self.daq_ao_piezo_stage_y == "dev1/AO1":
                chan_name = "dev1/_ao1_vs_aognd"
            task.ai_channels.add_ai_voltage_chan(
                chan_name,
                min_val=-self.daq_voltage_range_factor,
                max_val=self.daq_voltage_range_factor,
            )
            if self.daq_ao_piezo_stage_z == "dev1/AO2":
                chan_name = "dev1/_ao2_vs_aognd"
            task.ai_channels.add_ai_voltage_chan(
                chan_name,
                min_val=-self.daq_voltage_range_factor,
                max_val=self.daq_voltage_range_factor,
            )
            voltages = task.read()

        return voltages[0], voltages[1], voltages[2]

    @setting(
        52,
        x_center="v[]",
        y_center="v[]",
        x_range="v[]",
        y_range="v[]",
        num_steps="i",
        period="i",
        returns="*v[]*v[]",
    )
    def load_sweep_scan_xy(
        self, c, x_center, y_center, x_range, y_range, num_steps, period
    ):
        """Load a scan that will wind through the grid defined by the passed
        parameters. Samples are advanced by the clock. Currently x_range
        must equal y_range.

        Normal scan performed, starts in bottom right corner, and starts
        heading left

        Params
            x_center: float
                Center x voltage of the scan
            y_center: float
                Center y voltage of the scan
            x_range: float
                Full scan range in x
            y_range: float
                Full scan range in y
            num_steps: int
                Number of steps the break the ranges into
            period: int
                Expected period between clock signals in ns

        Returns
            list(float)
                The x voltages that make up the scan
            list(float)
                The y voltages that make up the scan
        """

        ######### Assumes x_range == y_range #########

        if x_range != y_range:
            raise ValueError("x_range must equal y_range for now")

        x_num_steps = num_steps
        y_num_steps = num_steps

        # Force the scan to have square pixels by only applying num_steps
        # to the shorter axis
        half_x_range = x_range / 2
        half_y_range = y_range / 2

        x_low = x_center - half_x_range
        x_high = x_center + half_x_range
        y_low = y_center - half_y_range
        y_high = y_center + half_y_range

        # Apply scale and offset to get the voltages we'll apply to the stage
        # Note that the polar/azimuthal angles, not the actual x/y positions
        # are linear in these voltages. For a small range, however, we don't
        # really care.
        x_voltages_1d = numpy.linspace(x_low, x_high, num_steps)
        y_voltages_1d = numpy.linspace(y_low, y_high, num_steps)

        ######### Works for any x_range, y_range #########

        # Winding cartesian product
        # The x values are repeated and the y values are mirrored and tiled
        # The comments below shows what happens for [1, 2, 3], [4, 5, 6]

        # [1, 2, 3] => [1, 2, 3, 3, 2, 1]
        x_inter = numpy.concatenate((x_voltages_1d, numpy.flipud(x_voltages_1d)))
        # [1, 2, 3, 3, 2, 1] => [1, 2, 3, 3, 2, 1, 1, 2, 3]
        if y_num_steps % 2 == 0:  # Even x size
            x_voltages = numpy.tile(x_inter, int(y_num_steps / 2))
        else:  # Odd x size
            x_voltages = numpy.tile(x_inter, int(numpy.floor(y_num_steps / 2)))
            x_voltages = numpy.concatenate((x_voltages, x_voltages_1d))

        # [4, 5, 6] => [4, 4, 4, 5, 5, 5, 6, 6, 6]
        y_voltages = numpy.repeat(y_voltages_1d, x_num_steps)

        voltages = numpy.vstack((x_voltages, y_voltages))

        # logging.debug(voltages)
        self.load_stream_writer_xy(
            c, "Piezo_stage-load_sweep_scan_xy", voltages, period
        )

        return x_voltages_1d, y_voltages_1d

    @setting(
        62,
        x_center="v[]",
        y_center="v[]",
        z_center="v[]",
        x_range="v[]",
        z_range="v[]",
        num_steps="i",
        period="i",
        returns="*v[]*v[]",
    )
    def load_sweep_scan_xz(
        self, c, x_center, y_center, z_center, x_range, z_range, num_steps, period
    ):
        """Load a scan that will wind through the grid defined by the passed
        parameters. Samples are advanced by the clock.

        Normal scan performed, starts in bottom right corner, and starts
        heading left

        Params
            x_center: float
                Center x voltage of the scan
            y_center: float
                Center y voltage of the scan
            z_center: float
                Center z voltage of the scan
            x_range: float
                Full scan range in x
            z_range: float
                Full scan range in z
            num_steps: int
                Number of steps the break the ranges into
            period: int
                Expected period between clock signals in ns

        Returns
            list(float)
                The x voltages that make up the scan
            list(float)
                The z voltages that make up the scan
        """

        x_num_steps = num_steps
        z_num_steps = num_steps

        # Force the scan to have square pixels by only applying num_steps
        # to the shorter axis
        half_x_range = x_range / 2
        half_z_range = z_range / 2

        x_low = x_center - half_x_range
        x_high = x_center + half_x_range
        z_low = z_center - half_z_range
        z_high = z_center + half_z_range

        # Apply scale and offset to get the voltages we'll apply to the stage
        # Note that the polar/azimuthal angles, not the actual x/z positions
        # are linear in these voltages. For a small range, however, we don't
        # really care.
        x_voltages_1d = numpy.linspace(x_low, x_high, num_steps)
        z_voltages_1d = numpy.linspace(z_low, z_high, num_steps)

        ######### Works for any x_range, z_range #########

        # Winding cartesian product
        # The x values are repeated and the z values are mirrored and tiled
        # The comments below shows what happens for [1, 2, 3], [4, 5, 6]

        # [1, 2, 3] => [1, 2, 3, 3, 2, 1]
        x_inter = numpy.concatenate((x_voltages_1d, numpy.flipud(x_voltages_1d)))
        # [1, 2, 3, 3, 2, 1] => [1, 2, 3, 3, 2, 1, 1, 2, 3]
        if z_num_steps % 2 == 0:  # Even x size
            x_voltages = numpy.tile(x_inter, int(z_num_steps / 2))
        else:  # Odd x size
            x_voltages = numpy.tile(x_inter, int(numpy.floor(z_num_steps / 2)))
            x_voltages = numpy.concatenate((x_voltages, x_voltages_1d))

        # [4, 5, 6] => [4, 4, 4, 5, 5, 5, 6, 6, 6]
        z_voltages = numpy.repeat(z_voltages_1d, x_num_steps)
        y_voltages = numpy.empty(len(z_voltages))
        y_voltages.fill(y_center)
        voltages = numpy.vstack((x_voltages, y_voltages, z_voltages))

        # logging.debug(voltages)
        self.load_stream_writer_xyz(
            c, "Piezo_stage-load_sweep_scan_xz", voltages, period
        )

        return x_voltages_1d, z_voltages_1d

    @setting(
        3,
        x_center="v[]",
        y_center="v[]",
        xy_range="v[]",
        num_steps="i",
        period="i",
        returns="*v[]*v[]",
    )
    def load_cross_scan_xy(self, c, x_center, y_center, xy_range, num_steps, period):
        """Load a scan that will first step through xy_range in x keeping y
        constant at its center, then step through xy_range in y keeping x
        constant at its center.

        Params
            x_center: float
                Center x voltage of the scan
            y_center: float
                Center y voltage of the scan
            xy_range: float
                Full scan range in x/y
            num_steps: int
                Number of steps the break the x/y range into
            period: int
                Expected period between clock signals in ns

        Returns
            list(float)
                The x voltages that make up the scan
            list(float)
                The y voltages that make up the scan
        """

        half_xy_range = xy_range / 2

        x_low = x_center - half_xy_range
        x_high = x_center + half_xy_range
        y_low = y_center - half_xy_range
        y_high = y_center + half_xy_range

        x_voltages_1d = numpy.linspace(x_low, x_high, num_steps)
        y_voltages_1d = numpy.linspace(y_low, y_high, num_steps)

        x_voltages = numpy.concatenate([x_voltages_1d, numpy.full(num_steps, x_center)])
        y_voltages = numpy.concatenate([numpy.full(num_steps, y_center), y_voltages_1d])

        voltages = numpy.vstack((x_voltages, y_voltages))

        self.load_stream_writer_xy(
            c, "Piezo_stage-load_cross_scan_xy", voltages, period
        )

        return x_voltages_1d, y_voltages_1d

    @setting(
        7,
        radius="v[]",
        num_steps="i",
        period="i",
        returns="*v[]*v[]",
    )
    def load_circle_scan_xy(self, c, radius, num_steps, period):
        """Load a circle scan centered about 0,0. Useful for testing cat's eye
        stationary point. For this reason, the scan runs continuously, not
        just until it makes it through all the samples once.

        Params
            radius: float
                Radius of the circle in V
            num_steps: int
                Number of steps the break the x/y range into
            period: int
                Expected period between clock signals in ns

        Returns
            list(float)
                The x voltages that make up the scan
            list(float)
                The y voltages that make up the scan
        """

        angles = numpy.linspace(0, 2 * numpy.pi, num_steps)

        x_voltages = radius * numpy.sin(angles)

        y_voltages = radius * numpy.cos(angles)
        # y_voltages = numpy.zeros(len(angles))

        voltages = numpy.vstack((x_voltages, y_voltages))

        self.load_stream_writer_xy(
            c, "Piezo_stage-load_circle_scan_xy", voltages, period, True
        )

        return x_voltages, y_voltages

    @setting(
        4,
        x_center="v[]",
        y_center="v[]",
        scan_range="v[]",
        num_steps="i",
        period="i",
        returns="*v[]",
    )
    def load_stream_x(
        self, c, x_center, y_center, z_center, scan_range, num_steps, period
    ):
        """Load a scan that will step through scan_range in x keeping y and z
        constant at its center.

        Params
            x_center: float
                Center x voltage of the scan
            y_center: float
                Center y voltage of the scan
            z_center: float
                Center z voltage of the scan
            scan_range: float
                Full scan range in x/y
            num_steps: int
                Number of steps the break the x/y range into
            period: int
                Expected period between clock signals in ns

        Returns
            list(float)
                The x voltages that make up the scan
        """

        half_scan_range = scan_range / 2

        x_low = x_center - half_scan_range
        x_high = x_center + half_scan_range

        x_voltages = numpy.linspace(x_low, x_high, num_steps)
        y_voltages = numpy.full(num_steps, y_center)
        z_voltages = numpy.full(num_steps, z_center)

        voltages = numpy.vstack((x_voltages, y_voltages, z_voltages))

        self.load_stream_writer_xyz(c, "Piezo_stage-load_scan_x", voltages, period)

        return x_voltages

    @setting(
        5,
        x_center="v[]",
        y_center="v[]",
        z_center="v[]",
        scan_range="v[]",
        num_steps="i",
        period="i",
        returns="*v[]",
    )
    def load_stream_y(
        self, c, x_center, y_center, z_center, scan_range, num_steps, period
    ):
        """Load a scan that will step through scan_range in y keeping x and y
        constant at its center.

        Params
            x_center: float
                Center x voltage of the scan
            y_center: float
                Center y voltage of the scan
            z_center: float
                Center z voltage of the scan
            scan_range: float
                Full scan range in x/y
            num_steps: int
                Number of steps the break the x/y range into
            period: int
                Expected period between clock signals in ns

        Returns
            list(float)
                The y voltages that make up the scan
        """

        half_scan_range = scan_range / 2

        y_low = y_center - half_scan_range
        y_high = y_center + half_scan_range

        x_voltages = numpy.full(num_steps, x_center)
        y_voltages = numpy.linspace(y_low, y_high, num_steps)
        z_voltages = numpy.full(num_steps, z_center)

        voltages = numpy.vstack((x_voltages, y_voltages, z_voltages))

        self.load_stream_writer_xyz(c, "Piezo_stage-load_scan_y", voltages, period)

        return y_voltages

    @setting(
        55,
        x_center="v[]",
        y_center="v[]",
        z_center="v[]",
        scan_range="v[]",
        num_steps="i",
        period="i",
        returns="*v[]",
    )
    def load_stream_z(
        self, c, x_center, y_center, z_center, scan_range, num_steps, period
    ):
        """Load a scan that will step through scan_range in z keeping x and y
        constant at its center.

        Params
            x_center: float
                Center x voltage of the scan
            y_center: float
                Center y voltage of the scan
            z_center: float
                Center z voltage of the scan
            scan_range: float
                Full scan range in z
            num_steps: int
                Number of steps the break the x/y range into
            period: int
                Expected period between clock signals in ns

        Returns
            list(float)
                The z voltages that make up the scan
        """

        half_scan_range = scan_range / 2

        z_low = z_center - half_scan_range
        z_high = z_center + half_scan_range

        x_voltages = numpy.full(num_steps, x_center)
        y_voltages = numpy.full(num_steps, y_center)
        z_voltages = numpy.linspace(z_low, z_high, num_steps)

        voltages = numpy.vstack((x_voltages, y_voltages, z_voltages))

        self.load_stream_writer_xyz(c, "Piezo_stage-load_scan_y", voltages, period)

        return z_voltages

    @setting(6, x_points="*v[]", y_points="*v[]", period="i")
    def load_arb_scan_xy(self, c, x_points, y_points, period):
        """Load a scan that goes between points. E.i., starts at [1,1] and
        then on a clock pulse, moves to [2,1]. Can work for arbitrarily large
        number of points
        (previously load_two_point_xy_scan)

        Params
            x_points: list(float)
                X values correspnding to positions in x
                y_points: list(float)
                Y values correspnding to positions in y
            period: int
                Expected period between clock signals in ns

        """

        voltages = numpy.vstack((x_points, y_points))

        self.load_stream_writer_xy(c, "Piezo_stage-load_arb_scan_xy", voltages, period)

        return


__server__ = PosXyzPiP6163c()

if __name__ == "__main__":
    from labrad import util

    util.runServer(__server__)
