# -*- coding: utf-8 -*-
"""
Output server for the PI E709 objective piezo. 

Created on Thu Apr  4 15:58:30 2019

@author: mccambria

### BEGIN NODE INFO
[info]
name = objective_piezo
version = 1.0
description =

[startup]
cmdline = %PYTHON% %FILE%
timeout = 20

[shutdown]
message = 987654321
timeout = 
### END NODE INFO
"""


from labrad.server import LabradServer
from labrad.server import setting
from twisted.internet.defer import ensureDeferred
from pipython import GCSDevice
import nidaqmx
import logging
import numpy
import nidaqmx.stream_writers as stream_writers
import socket
from pathlib import Path


class ObjectivePiezo(LabradServer):
    name = 'objective_piezo'
    pc_name = socket.gethostname()

    def initServer(self):
        filename = 'E:/Shared drives/Kolkowitz Lab Group/nvdata/pc_{}/labrad_logging/{}.log'
        filename = filename.format(self.pc_name, self.name)
        logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    datefmt='%y-%m-%d_%H-%M-%S', filename=filename)
        self.task = None
        
    def sub_init_server_z(self):
        """Sub-routine to be called by xyz server"""
        self.z_last_position = None
        self.z_current_direction = None
        self.z_last_turning_position = None
        config = ensureDeferred(self.get_config_z())
        config.addCallback(self.on_get_config_z)

    async def get_config_z(self):
        p = self.client.registry.packet()
        p.cd(['', 'Config', 'DeviceIDs'])
        p.get('objective_piezo_model')
        p.get('objective_piezo_serial')
        p.cd(['', 'Config', 'Wiring', 'Daq'])
        p.get('ao_objective_piezo')
        p.get('di_clock')
        p.cd(['', 'Config', 'Positioning'])
        p.get('z_hysteresis_a')
        p.get('z_hysteresis_b')
        result = await p.send()
        return result['get']

    def on_get_config_z(self, config):
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
        self.daq_ao_objective_piezo = config[2]
        self.daq_di_clock = config[3]
        self.z_hysteresis_a = config[4]
        self.z_hysteresis_b = config[5]
        logging.debug('Init complete')
        
    def compensate_hysteresis_z(self, position):
        """
        The hysteresis curve is p(v) = a * v**2 + b * v.
        We want to feedforward using this curve to set the piezo voltage
        such that the nominal voltage passed by the user functions
        linearly and without hysteresis. The goal is to prevent the
        accumulation of small errors until active feedback (eg 
        optimizing on an NV) can be performed

        Parameters
        ----------          
        position : float
            Position (in this case the nominal voltage) the user intends 
            to move to for a linear response without hysteresis

        Returns
        -------
        float
            Compensated voltage to set
        """
        
        single_value = False
        if type(position) not in [numpy.ndarray, list]:
            single_value = True
            position = [position]
        
        # Pull everything out of self to save some lookup time
        last_position = self.z_last_position
        current_direction = self.z_current_direction
        last_turning_position = self.z_last_turning_position
        # If values are uninitialized, assume we just started heading up
        # (ie don't adjust the first position at all)
        if None in [last_position, current_direction, last_turning_position]:
            last_position = position[0]
            current_direction = +1
            last_turning_position = position[0]
        a = self.z_hysteresis_a
        b = self.z_hysteresis_b
        
        # We'll have to for loop because the (n+1)th value depends 
        # on the nth value
        compensated_voltage = []
        for val in position:
            
            # compensated_voltage.append(val)
            # continue
            
            # First determine if we're turning around
            movement_direction = numpy.sign(val - last_position)
            if movement_direction == -current_direction:
                last_turning_position = last_position
                current_direction = movement_direction
            
            # The adjustment voltage we need is obtained by inverting p(v)
            abs_p = abs(val - last_turning_position)
            # v = (-b + numpy.sqrt(b**2 + 4 * a * abs_p)) / (2 * a)
            v = abs_p
            result = last_turning_position + (movement_direction * v)
            compensated_voltage.append(result)
            
            # Cache the last position
            last_position = val
            
        self.z_last_position = val
        self.z_current_direction = movement_direction
        self.z_last_turning_position = last_turning_position
        
        if single_value:
            return compensated_voltage[0]
        else:
            return numpy.array(compensated_voltage)

    def load_stream_writer_z(self, c, task_name, voltages, period):

        # Close the existing task if there is one
        if self.task is not None:
            self.close_task_internal()
            
        # Make sure the voltages are an array
        voltages = numpy.array(voltages)
            
        compensated_voltages = self.compensate_hysteresis_z(voltages)

        # Write the initial voltages and stream the rest
        num_voltages = len(compensated_voltages)
        self.write_z(c, compensated_voltages[0])
        stream_voltages = compensated_voltages[1:num_voltages]
        stream_voltages = numpy.ascontiguousarray(stream_voltages)
        num_stream_voltages = num_voltages - 1

        # Create a new task
        task = nidaqmx.Task(task_name)
        self.task = task

        # Set up the output channels
        task.ao_channels.add_ao_voltage_chan(self.daq_ao_objective_piezo,
                                             min_val=3.0, max_val=7.0)

        # Set up the output stream
        output_stream = nidaqmx.task.OutStream(task)
        writer = stream_writers.AnalogSingleChannelWriter(output_stream)

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

    @setting(22, voltage='v[]')
    def write_z(self, c, voltage):
        """Write the specified voltage to the piezo"""

        # Close the stream task if it exists
        # This can happen if we quit out early
        if self.task is not None:
            self.close_task_internal()

        # Adjust voltage turn for hysteresis
        compensated_voltage = self.compensate_hysteresis_z(voltage)

        with nidaqmx.Task() as task:
            # Set up the output channels
            task.ao_channels.add_ao_voltage_chan(self.daq_ao_objective_piezo,
                                                 min_val=3.0, max_val=7.0)
            task.write(compensated_voltage)

    @setting(21, returns='v[]')
    def read_z(self, c):
        """Return the current voltages on the piezo's DAQ channel"""
        with nidaqmx.Task() as task:
            # Set up the internal channels - to do: actual parsing...
            if self.daq_ao_objective_piezo == 'dev1/AO2':
                chan_name = 'dev1/_ao2_vs_aognd'
            task.ai_channels.add_ai_voltage_chan(chan_name,
                                                 min_val=3.0, max_val=7.0)
            voltage = task.read()
        return voltage
    
    
    @setting(23, center='v[]', scan_range='v[]',
             num_steps='i', period='i', returns='*v[]')
    def load_scan_z(self, c, center, scan_range, num_steps, period):
        """Load a linear sweep with the DAQ"""

        half_scan_range = scan_range / 2
        low = center - half_scan_range
        high = center + half_scan_range
        voltages = numpy.linspace(low, high, num_steps)
        self.load_stream_writer_z(c, 'ObjectivePiezo-load_scan_z',
                                  voltages, period)
        return voltages
    
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    @setting(24, z_voltages='*v[]', period='i', returns='*v[]')
    def load_arb_scan_z(self, c, z_voltages, period):
        """Load a list of voltages with the DAQ"""

        self.load_stream_writer_z(c, 'ObjectivePiezo-load_arb_scan_z',
                                  numpy.array(z_voltages), period)
        return z_voltages
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++

__server__ = ObjectivePiezo()

if __name__ == '__main__':
    from labrad import util
    util.runServer(__server__)
