# -*- coding: utf-8 -*-
"""
Output server for the Thorlabs GVS212 galvanometer. Controlled by the DAQ.

Created on Mon Apr  8 19:50:12 2019

@author: mccambria

### BEGIN NODE INFO
[info]
name = galvo
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


class Galvo(LabradServer):
    name = 'galvo'
    logging.basicConfig(level=logging.DEBUG, 
                format='%(asctime)s %(levelname)-8s %(message)s',
                datefmt='%y-%m-%d_%H-%M-%S',
                filename='E:/Shared drives/Kolkowitz Lab Group/nvdata/labrad_logging/{}.log'.format(name))

    def initServer(self):
        self.task = None
        config = ensureDeferred(self.get_config())
        config.addCallback(self.on_get_config)

    async def get_config(self):
        p = self.client.registry.packet()
        p.cd(['Config', 'Wiring', 'Daq'])
        p.get('ao_galvo_x')
        p.get('ao_galvo_y')
        p.get('di_clock')
        result = await p.send()
        return result['get']

    def on_get_config(self, config):
        self.daq_ao_galvo_x = config[0]
        self.daq_ao_galvo_y = config[1]
        self.daq_di_clock = config[2]
        # logging.debug(self.daq_di_clock)
        

    def stopServer(self):
        self.close_task_internal()

    def load_stream_writer(self, c, task_name, voltages, period):

        # Close the existing task if there is one
        if self.task is not None:
            self.close_task_internal()

        # Write the initial voltages and stream the rest
        num_voltages = voltages.shape[1]
        self.write(c, voltages[0, 0], voltages[1, 0])
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

    def close_task_internal(self, task_handle=None, status=None,
                            callback_data=None):
        task = self.task
        if task is not None:
            task.close()
            self.task = None
        return 0

    @setting(0, xVoltage='v[]', yVoltage='v[]')
    def write_xy(self, c, xVoltage, yVoltage):
        """Write the specified voltages to the galvo.

        Params
            xVoltage: float
                Voltage to write to the x channel
            yVoltage: float
                Voltage to write to the y channel
        """

        # Close the stream task if it exists
        # This can happen if we quit out early
        if self.task is not None:
            self.close_task_internal()

        with nidaqmx.Task() as task:
            # Set up the output channels
            task.ao_channels.add_ao_voltage_chan(self.daq_ao_galvo_x,
                                                 min_val=-10.0, max_val=10.0)
            task.ao_channels.add_ao_voltage_chan(self.daq_ao_galvo_y,
                                                 min_val=-10.0, max_val=10.0)
            task.write([xVoltage, yVoltage])

    @setting(1, returns='*v[]')
    def read_xy(self, c):
        """Return the current voltages on the x and y channels.

        Returns
            list(float)
                Current voltages on the x and y channels

        """
        with nidaqmx.Task() as task:
            # Set up the internal channels - to do: actual parsing...
            if self.daq_ao_galvo_x == 'dev1/AO0':
                chan_name = 'dev1/_ao0_vs_aognd'
            task.ai_channels.add_ai_voltage_chan(chan_name,
                                                 min_val=-10.0, max_val=10.0)
            if self.daq_ao_galvo_y == 'dev1/AO1':
                chan_name = 'dev1/_ao1_vs_aognd'
            task.ai_channels.add_ai_voltage_chan(chan_name,
                                                 min_val=-10.0, max_val=10.0)
            voltages = task.read()

        return voltages[0], voltages[1]

    @setting(2, x_center='v[]', y_center='v[]',
             x_range='v[]', y_range='v[]', num_steps='i', period='i',
             returns='*v[]*v[]')
    def load_sweep_xy_scan(self, c, x_center, y_center,
                        x_range, y_range, num_steps, period):
        """Load a scan that will wind through the grid defined by the passed
        parameters. Samples are advanced by the clock. Currently x_range
        must equal y_range.

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
            raise ValueError('x_range must equal y_range for now')

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

        # Apply scale and offset to get the voltages we'll apply to the galvo
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
        x_inter = numpy.concatenate((x_voltages_1d,
                                     numpy.flipud(x_voltages_1d)))
        # [1, 2, 3, 3, 2, 1] => [1, 2, 3, 3, 2, 1, 1, 2, 3]
        if y_num_steps % 2 == 0:  # Even x size
            x_voltages = numpy.tile(x_inter, int(y_num_steps/2))
        else:  # Odd x size
            x_voltages = numpy.tile(x_inter, int(numpy.floor(y_num_steps/2)))
            x_voltages = numpy.concatenate((x_voltages, x_voltages_1d))

        # [4, 5, 6] => [4, 4, 4, 5, 5, 5, 6, 6, 6]
        y_voltages = numpy.repeat(y_voltages_1d, x_num_steps)

        voltages = numpy.vstack((x_voltages, y_voltages))

        self.load_stream_writer(c, 'Galvo-load_sweep_scan', voltages, period)

        return x_voltages_1d, y_voltages_1d

    @setting(3, x_center='v[]', y_center='v[]', xy_range='v[]',
             num_steps='i', period='i', returns='*v[]*v[]')
    def load_cross_xy_scan(self, c, x_center, y_center,
                        xy_range, num_steps, period):
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

        x_voltages = numpy.concatenate([x_voltages_1d,
                                        numpy.full(num_steps, x_center)])
        y_voltages = numpy.concatenate([numpy.full(num_steps, y_center),
                                        y_voltages_1d])

        voltages = numpy.vstack((x_voltages, y_voltages))

        self.load_stream_writer(c, 'Galvo-load_cross_scan', voltages, period)

        return x_voltages_1d, y_voltages_1d

    @setting(4, x_center='v[]', y_center='v[]', scan_range='v[]',
             num_steps='i', period='i', returns='*v[]')
    def load_x_scan(self, c, x_center, y_center,
                    scan_range, num_steps, period):
        """Load a scan that will step through scan_range in x keeping y
        constant at its center.

        Params
            x_center: float
                Center x voltage of the scan
            y_center: float
                Center y voltage of the scan
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

        voltages = numpy.vstack((x_voltages, y_voltages))

        self.load_stream_writer(c, 'Galvo-load_x_scan', voltages, period)

        return x_voltages

    @setting(5, x_center='v[]', y_center='v[]', scan_range='v[]',
             num_steps='i', period='i', returns='*v[]')
    def load_y_scan(self, c, x_center, y_center,
                    scan_range, num_steps, period):
        """Load a scan that will step through scan_range in y keeping x
        constant at its center.

        Params
            x_center: float
                Center x voltage of the scan
            y_center: float
                Center y voltage of the scan
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

        voltages = numpy.vstack((x_voltages, y_voltages))

        self.load_stream_writer(c, 'Galvo-load_y_scan', voltages, period)

        return y_voltages
    
    @setting(6, x_points='*v[]', y_points='*v[]', period='i')
    def load_two_point_xy_scan(self, c, x_points, y_points, period):
        """Load a scan that goes between two points. E.i., starts at [1,1] and 
        then on a clock pulse, moves to [2,1].

        Params
            x_points: list(float)
                X values correspnding to the initial and final points
            y_points: list(float)
                Y values correspnding to the initial and final points
            period: int
                Expected period between clock signals in ns

        """

        voltages = numpy.vstack((x_points, y_points))

        self.load_stream_writer(c, 'Galvo-load_two_point_scan', voltages, period)

        return
__server__ = Galvo()

if __name__ == '__main__':
    from labrad import util
    util.runServer(__server__)
