# -*- coding: utf-8 -*-
"""
Base class for LaserGlow laser servers, specifically to control analog voltage
to the AOM via the DAQ and gated by the Pulsestreamer

Created on January 24th, 2023

@author: gardill
"""

from labrad.server import LabradServer
from labrad.server import setting
from twisted.internet.defer import ensureDeferred
import nidaqmx
import nidaqmx.stream_writers as stream_writers
import numpy
import logging
import socket
from utils import common


class LaserLgloBase(LabradServer):
    pc_name = socket.gethostname()

    def initServer(self):
        filename = (
            "E:/Shared drives/Kolkowitz Lab Group/nvdata/pc_{}/labrad_logging/{}.log"
        )
        filename = filename.format(self.pc_name, self.name)
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s %(levelname)-8s %(message)s",
            datefmt="%y-%m-%d_%H-%M-%S",
            filename=filename,
        )
        self.task = None
        config = common.get_config_dict()
        wiring = config["Wiring"]["Daq"]
        self.do_feedthrough = wiring["do_{}_feedthrough".format(self.name)]
        self.di_feedthrough = wiring["di_{}_feedthrough".format(self.name)]
        self.true_zero_value = config["Optics"][self.name]
        # Load the feedthrough and just leave it running
        try:
            self.load_feedthrough(None, 1.0)
        except Exception as e:
            logging.debug(e)
        logging.debug("Init complete")

    def stopServer(self):
        self.close_task_internal()

    def load_stream_writer(self, c, HIGH_voltage):
        stream_bools = [HIGH_voltage, self.true_zero_value]
        # Close the existing task if there is one
        if self.task is not None:
            self.close_task_internal()

        # Create a new task
        task = nidaqmx.Task(f"{self.name}-load_feedthrough")
        self.task = task

        # Set up the output channel
        task.ao_channels.add_ao_voltage_chan(
            self.ao_feedthrough, min_val=-0.1, max_val=1.0
        )
        # task.write([write_voltage])

        # Set up the output stream
        #        output_stream = nidaqmx.task.OutStream(task)
        #        writer = stream_writers.DigitalSingleChannelWriter(output_stream)

        # Configure the sample to advance on the rising edge of the PFI input.
        # The frequency specified is just the max expected rate in this case.
        freq = 10e6  # 10 MHz, every 100 ns. Thjs is the max freq for PCIE6738
        clock = self.di_feedthrough
        sample_mode = nidaqmx.constants.AcquisitionType.CONTINUOUS
        task.timing.cfg_samp_clk_timing(freq, source=clock, sample_mode=sample_mode)

        task.write(stream_bools)

        # Close the task once we've written all the samples
        task.register_done_event(self.close_task_internal)

        task.start()

    def close_task_internal(self, task_handle=None, status=None, callback_data=None):
        task = self.task
        if task is not None:
            task.close()
            self.task = None
        return 0

    @setting(0)
    def load_feedthrough(self, c, HIGH_voltage):
        # Just flip the TTL out on the rising edge of the TTL in
        # stream_bools = numpy.array([True, False], dtype=bool)

        self.load_stream_writer(c, HIGH_voltage)

    @setting(1)
    def reset(self, c):
        # Make sure the laser is off
        self.laser_off(c)

    @setting(2)
    def laser_on(self, c, voltage):
        self.close_task_internal()
        with nidaqmx.Task() as task:
            task.ao_channels.add_ao_voltage_chan(
                self.ao_feedthrough, min_val=-0.1, max_val=1.0
            )
            task.write([voltage])

    @setting(3)
    def laser_off(self, c):
        self.close_task_internal()
        with nidaqmx.Task() as task:
            task.ao_channels.add_ao_voltage_chan(
                self.ao_feedthrough, min_val=-0.1, max_val=1.0
            )
            task.write([self.true_zero_value])
