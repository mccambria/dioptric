# -*- coding: utf-8 -*-
"""
Combined server for galvo and objective_piezo

Created on July 2021

@author: mccambria

### BEGIN NODE INFO
[info]
name = galvo_and_objective_piezo
version = 1.0
description =

[startup]
cmdline = %PYTHON% %FILE%
timeout = 20

[shutdown]
message = 987654321
timeout = 5
### END NODE INFO
"""


from labrad.server import LabradServer
from labrad.server import setting
from twisted.internet.defer import ensureDeferred
import nidaqmx
import nidaqmx.stream_writers as stream_writers
import numpy
import logging
import socket
from pipython import GCSDevice
from pathlib import Path
from servers.outputs.galvo import Galvo
from servers.outputs.objective_piezo import ObjectivePiezo


class GalvoAndObjectivePiezo(Galvo, ObjectivePiezo):
    name = 'galvo_and_objective_piezo'
    pc_name = socket.gethostname()

    def initServer(self):
        filename = 'E:/Shared drives/Kolkowitz Lab Group/nvdata/pc_{}/labrad_logging/{}.log'
        filename = filename.format(self.pc_name, self.name)
        logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    datefmt='%y-%m-%d_%H-%M-%S', filename=filename)
        self.task = None
        self.z_last_position = None
        self.z_current_direction = None
        self.z_last_turning_position = None
        config = ensureDeferred(self.get_config())
        config.addCallback(self.on_get_config)

    async def get_config(self):
        p = self.client.registry.packet()
        p.cd(['', 'Config', 'DeviceIDs'])
        p.get('objective_piezo_model')
        p.get('objective_piezo_serial')
        p.cd(['', 'Config', 'Wiring', 'Daq'])
        p.get('ao_galvo_x')
        p.get('ao_galvo_y')
        p.get('ao_objective_piezo')
        p.get('di_clock')
        p.cd(['', 'Config', 'Positioning'])
        p.get('z_hysteresis_a')
        p.get('z_hysteresis_b')
        result = await p.send()
        return result['get']

    def on_get_config(self, config):
        
        # Objective piezo setup
        # Load the generic device
        gcs_dll_path = str(Path.home())
        gcs_dll_path += '\\Documents\\GitHub\\kolkowitz-nv-experiment-v1.0'
        gcs_dll_path += '\\servers\\outputs\\GCSTranslator\\PI_GCS2_DLL_x64.dll'
        self.piezo = GCSDevice(devname=config[0], gcsdll=gcs_dll_path)
        # Connect the specific device with the serial number
        self.piezo.ConnectUSB(config[1])
        # Just one axis for this device
        self.axis = self.piezo.axes[0]
        self.piezo.SPA(self.axis, 0x06000500, 2)  # External control mode
        
        # DAQ setup
        self.daq_ao_galvo_x = config[2]
        self.daq_ao_galvo_y = config[3]
        self.daq_ao_objective_piezo = config[4]
        self.daq_di_clock = config[5]
        self.z_hysteresis_a = config[6]
        self.z_hysteresis_b = config[7]
        logging.debug('Init complete')

    def stopServer(self):
        self.close_task_internal()

    def load_stream_writer_xyz(self, c, task_name, voltages, period):

        # Close the existing task if there is one
        if self.task is not None:
            self.close_task_internal()

        # Write the initial voltages and stream the rest
        num_voltages = voltages.shape[1]
        self.write_xy(c, voltages[0, 0], voltages[1, 0])
        self.write_z(c, voltages[2, 0])
        stream_voltages = voltages[:, 1:num_voltages]
        stream_voltages = numpy.ascontiguousarray(stream_voltages)
        num_stream_voltages = num_voltages - 1

        # Create a new task
        task = nidaqmx.Task(task_name)
        self.task = task

        # Set up the output channels
        task.ao_channels.add_ao_voltage_chan(self.daq_ao_galvo_x,
                                             min_val=-10.0, max_val=10.0)
        task.ao_channels.add_ao_voltage_chan(self.daq_ao_galvo_y,
                                             min_val=-10.0, max_val=10.0)
        task.ao_channels.add_ao_voltage_chan(self.daq_ao_objective_piezo,
                                             min_val=3.0, max_val=7.0)

        # Set up the output stream
        output_stream = nidaqmx.task.OutStream(task)
        writer = stream_writers.AnalogMultiChannelWriter(output_stream)

        # Configure the sample to advance on the rising edge of the PFI input.
        # The frequency specified is just the max expected rate in this case.
        # We'll stop once we've run all the samples.
        freq = float(1/(period*(10**-9)))  # freq in seconds as a float
        task.timing.cfg_samp_clk_timing(freq, source=self.daq_di_clock,
                                        samps_per_chan=num_stream_voltages)

        writer.write_many_sample(stream_voltages)

        # Close the task once we've written all the samples
        task.register_done_event(self.close_task_internal)

        task.start()

    @setting(11, x_points='*v[]', y_points='*v[]', z_points='*v[]', period='i')
    def load_arb_xyz_scan(self, c, x_points, y_points, z_points, period):
        """
        Load a scan around a seuqence of arbitrary xyz points 

        Params
            x_points: list(float)
                X values correspnding to positions in x
            y_points: list(float)
                Y values correspnding to positions in y
            z_points: list(float)
                Z values correspnding to positions in z
            period: int
                Expected period between clock signals in ns

        """

        voltages = numpy.vstack((x_points, y_points, z_points))

        self.load_stream_writer_xyz(c, 'GalvoAndObjectivePiezo-load_arb_xyz_scan', voltages, period)

        return
    
    
__server__ = GalvoAndObjectivePiezo()

if __name__ == '__main__':
    from labrad import util
    util.runServer(__server__)
