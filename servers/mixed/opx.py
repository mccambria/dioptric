# -*- coding: utf-8 -*-
"""
server for the Quantum Machines OPX

Created on August 29th, 2022

@author: carter fox

### BEGIN NODE INFO
[info]
name = opx
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

from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig
import matplotlib.pyplot as plt
from configuration import *
from labrad.server import LabradServer
from labrad.server import setting
from twisted.internet.defer import ensureDeferred
from numpy import count_nonzero, nonzero, concatenate
import numpy as np
import importlib
import numpy
import logging
import re
import sys
import time
import socket
from numpy import pi
root2_on_2 = numpy.sqrt(2) / 2
from inputs.interfaces.tagger import Tagger
from timing.interfaces.pulse_gen import PulseGen


class OPX(LabradServer, Tagger, PulseGen):
    name = "opx"
    pc_name = socket.gethostname()
    steady_state_program_file = 'steady_state_program.py'
    opx_configuration_file_path = "/Users/carterfox/Documents/GitHub/kolkowitz-nv-experiment-v1.0"
    sys.path.append(opx_configuration_file_path)
    from opx_configuration_file import *
    

    def initServer(self):
        filename = 'E:/Shared drives/Kolkowitz Lab Group/nvdata/pc_{}/labrad_logging/{}.log'
        filename = filename.format(self.pc_name, self.name)
        logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    datefmt='%y-%m-%d_%H-%M-%S', filename=filename)
        config = ensureDeferred(self.get_config())
        config.addCallback(self.on_get_config)
        
    async def get_config(self):  ### async???
        p = self.client.registry.packet()
        p.cd(['', 'Config', 'Microwaves'])
        p.get('iq_comp_amp') #this is needed for arbitrary waveform generator functions
        result = await p.send()
        return result['get']

    def on_get_config(self, config):
        resource_manager = visa.ResourceManager()
        self.iq_comp_amp = config[0]
        logging.info('Init complete')
        self.qmm = QuantumMachinesManager(qop_ip) # opens communication with the QOP so we can make a quantum machine
        self.qm = qmm.open_qm(config_opx)             # makes a quantum machine with the specific configuration
        ####in the future we could make this start the infinite loop program. But we can always just run the program too.
        
    def stopServer(self):
        try:
            self.pending_steady_state_job.halt()
            self.qmm.close_all_quantum_machines()
            self.qmm.close()
        except:
            self.qmm.close_all_quantum_machines()
            self.qmm.close()
    
   
    #%% sequence loading and executing functions ###
    ### looks good. It's ready to queue up both the experiment job and the infinite loop job. I can test it with simple programs and config
    
        
    def get_seq(self, seq_file, seq_args_string): # this returns the qua sequence without repeats. from pulse streamer. DONE
        """For the OPX, this will grab a single iteration of the desired sequence
            seq_file: str
                A qua sequence file from the sequence library
            seq_args_string: list(any)
                Arbitrary list used to modulate a sequence from the sequence
                library - see simple_readout.py for an example. Default is
                None
                
        Returns
            seq: Qua program 
                A sequence written in Qua for the OPX
            final: unsure. 
            list(any)
                Arbitrary list returned by the sequence file
        """    
        seq = None
        file_name, file_ext = os.path.splitext(seq_file)
        if file_ext == ".py":  # py: import as a module
            seq_module = importlib.import_module(file_name)
            args = tool_belt.decode_seq_args(seq_args_string)
            
            seq, final, ret_vals = seq_module.get_seq(        # here seq is the qua program. refer to opx sequence template      
                self, self.config_dict, args
            )
        
        self.qua_program = seq
        self.seq_file = seq_file
        self.seq_args_string = seq_args_string
        
        return seq, final, ret_vals

    def get_full_seq(self, seq_file, seq_args_string, num_repeat): # this one returns the full qua sequence with repeats, such as rabi with nreps=1e5. DONE
        """For the OPX, this will grab the desired sequence with the desired number of repeats
            seq_file: str
                A qua sequence file from the sequence library
            seq_args_string: list(any)
                Arbitrary list used to modulate a sequence from the sequence
                library - see simple_readout.py for an example. Default is
                None
            num_repeat: int
                The number of times to repeat the sequence, such as the number of reps in a rabi routine
                
        Returns
            seq: Qua program 
                A sequence written in Qua for the OPX
            final: unsure. 
            list(any)
                Arbitrary list returned by the sequence file
        """    
        seq = None
        file_name, file_ext = os.path.splitext(seq_file)
        if file_ext == ".py":  # py: import as a module
            seq_module = importlib.import_module(file_name)
            args = tool_belt.decode_seq_args(seq_args_string)
            
            seq, final, ret_vals = seq_module.get_full_seq(        # here seq is the qua program. refer to opx sequence template      
                self, self.config_dict, args, num_repeat
            )
        
        self.qua_program = seq
        self.seq_file = seq_file
        self.seq_args_string = seq_args_string
        
        file_name_steady_state, file_ext_steady_state = os.path.splitext(steady_state_program_file)
        if file_ext_steady_state == ".py":  # py: import as a module
            seq_module_steady_state = importlib.import_module(file_name_steady_state)
            
            seq_steady_state = seq_module_steady_state.get_steady_state_seq(self)
        
        self.steady_state_program = seq_steady_state
        
        return seq, final, ret_vals
    
    @setting(1, seq_file="s", seq_args_string="s", returns="*?")
    def stream_load(self, c, seq_file, seq_args_string=""): #from pulse streamer. for the opx all this function does is return back some property(s) of the sequence you want. DONE
        """For the OPX, this will just grab a single iteration of the desired sequence and return the parameter of interest
        Params
            seq_file: str
                A qua sequence file from the sequence library
            args: list(any)
                Arbitrary list used to modulate a sequence from the sequence
                library - see simple_readout.py for an example. Default is
                None
                
        Returns
            list(any)
                Arbitrary list returned by the sequence file
        """
        seq, final, ret_vals = self.get_seq(seq_file, seq_args_string)
        self.x_voltages_1d = []
        self.y_voltages_1d = []
        self.z_voltages_1d = []
        
        return ret_vals

    @setting(2, num_repeat="i")
    def stream_start(self, c, num_repeat=1): # from pulse streamer. It will start the full sequence. DONE
        """For the OPX, this will run the qua program already grabbed by stream_load(). It opens communication with the QOP and executes the program, creating the job 
        Params
            num_repeat: int
                The number of times the sequence is repeated, such as the number of reps in a rabi routine
        """
        
        seq, final, ret_vals = self.get_full_seq(self.seq_file, self.seq_args_string, num_repeat, self.x_voltages_1d, self.y_voltages_1d, self.z_voltages_1d) #gets the full sequence
        
        if len(self.x_voltages_1d) >0 :
            self.write_x(c, self.x_voltages_1d[0])
        if len(self.y_voltages_1d) >0 :
            self.write_y(c, self.y_voltages_1d[0])
        if len(self.z_voltages_1d) >0 :
            self.write_z(c, self.z_voltages_1d[0])
            
        try:
            self.pending_steady_state_job.halt()
            self.pending_experiment_job = qm.queue.add_compiled(self.experiment_program_id)
            self.pending_steady_state_job = qm.queue.add_compiled(self.steady_state_program_id) # executes the qua program on the quantum machine  
            
        except:
            self.pending_experiment_job = qm.queue.add_compiled(self.experiment_program_id)
            self.pending_steady_state_job = qm.queue.add_compiled(self.steady_state_program_id) # executes the qua program on the quantum machine   
           
        return 
    
    @setting(3, seq_file="s", num_repeat="i", seq_args_string="s", returns="*?")
    def stream_immediate(self, c, seq_file, num_repeat=1, seq_args_string=""): # from pulse streamer. For the OPX this will get the full sequence we want and run it. DONE
        """Load the sequence from seq_file and immediately run it for
        the specified number of repitions. End in the specified
        final output state.

        Params
            seq_file: str
                A sequence file from the sequence library
            num_repeat: int
                Number of times to repeat the sequence. Default is 1
            args: list(any)
                Arbitrary list used to modulate a sequence from the sequence
                library - see simple_readout.py for an example. Default is
                None. All values in list must have same type.

        Returns
            list(any)
                Arbitrary list returned by the sequence file
        """
        
        seq, final, ret_vals = self.get_full_seq(seq_file, seq_args_string, num_repeat, self.x_voltages_1d, self.y_voltages_1d, self.z_voltages_1d) #gets the full sequence
        if len(self.x_voltages_1d) >0 :
            self.write_x(c, self.x_voltages_1d[0])
        if len(self.y_voltages_1d) >0 :
            self.write_y(c, self.y_voltages_1d[0])
        if len(self.z_voltages_1d) >0 :
            self.write_z(c, self.z_voltages_1d[0])
            
        self.experiment_program_id = qm.compile(self.qua_program)
        self.steady_state_program_id = qm.compile(self.steady_state_program)
        
        try:
            self.pending_steady_state_job.halt()
            self.pending_experiment_job = qm.queue.add_compiled(self.experiment_program_id)
            self.pending_steady_state_job = qm.queue.add_compiled(self.steady_state_program_id) # executes the qua program on the quantum machine  
            
        except:
            self.pending_experiment_job = qm.queue.add_compiled(self.experiment_program_id)
            self.pending_steady_state_job = qm.queue.add_compiled(self.steady_state_program_id) # executes the qua program on the quantum machine   
        
        return ret_vals
    
    #%% counting and time tagging functions ### 
    ### currently in good shape. Waiting to see if the time tag processing function I made will work with how timetags are saved
    ### these also need to be tested once I get the opx up and running again
    
        
    @setting(47, num_to_read="i", returns="*3w")
    def read_counter_complete(self, c, num_to_read=None):
        return self.read_counter_setting_internal(num_to_read)
    
    def read_counter_setting_internal(self, c, num_to_read=None): #from apd tagger. for the opx it fetches the results from the job. Don't think num_to_read has to do anything
        """This is the core function that any tagger we have needs. 
        For the OPX this fetches the data from the job that was created when the program was executed. 
        Assumes "counts" is one of the data streams
        The count stream should be a three level list. First level is the sample, second is the gates, third is the different apds. 
        first index gives the sample. next level gives the gate. next level gives which apd
        [  [ [],[] ] , [ [],[] ], [ [],[] ]  ]
        ##### This may be slightly wrong. It may be apds then gate, in which I need to slightly change the sequence code
        
        Params
            num_to_read: int
                This is not needed for the OPX
        Returns
            return_counts: array
                This is an array of the counts 
        """
        results = fetching_tool(self.pending_experiment_job, data_list = ["counts"], mode="wait_for_all")
        
        while results.is_processing():
            # Fetch results
            return_counts = results.fetch_all() #just not sure if its gonna put it into the list structure we want
            return_counts = return_counts.tolist()
            
        return return_counts

    @setting(5, num_to_read="i", returns="*w")
    def read_counter_simple(self, c, num_to_read=None):

        complete_counts = self.read_counter_setting_internal(num_to_read)

        # To combine APDs we assume all the APDs have the same gate
        gate_channels = list(self.tagger_di_gate.values())
        first_gate_channel = gate_channels[0]
        if not all(val == first_gate_channel for val in gate_channels):
            logging.critical("Combined counts from APDs with different gates.")

        # Just find the sum of each sample in complete_counts
        return_counts = [
            np.sum(sample, dtype=int) for sample in complete_counts ]

        return return_counts

    @setting(6, num_to_read="i", returns="*2w")
    def read_counter_separate_gates(self, c, num_to_read=None):

        complete_counts = self.read_counter_setting_internal(num_to_read)
        # logging.info(complete_counts)

        # To combine APDs we assume all the APDs have the same gate
        gate_channels = list(self.tagger_di_gate.values())
        first_gate_channel = gate_channels[0]
        if not all(val == first_gate_channel for val in gate_channels):
            logging.critical("Combined counts from APDs with different gates.")

        # Add the APD counts as vectors for each sample in complete_counts
        return_counts = [
            np.sum(sample, 0, dtype=int).tolist() for sample in complete_counts ]

        return return_counts
    
    @setting(11, modulus="i", num_to_read="i", returns="*2w")
    def read_counter_modulo_gates(self, c, modulus, num_to_read=None):

        complete_counts = self.read_counter_setting_internal(num_to_read)
        # logging.info(complete_counts)

        # To combine APDs we assume all the APDs have the same gate
        gate_channels = list(self.tagger_di_gate.values())
        first_gate_channel = gate_channels[0]
        if not all(val == first_gate_channel for val in gate_channels):
            logging.critical("Combined counts from APDs with different gates.")

        # Add the APD counts as vectors for each sample in complete_counts
        # sum_lambda = lambda arg: np.sum(arg, 0, dtype=int).tolist()
        # with Pool() as p:
        #     separate_gate_counts = p.map(sum_lambda, complete_counts)
        separate_gate_counts = [np.sum(el, 0, dtype=int).tolist() for el in complete_counts]

        # Run the modulus
        return_counts = []
        for sample in separate_gate_counts:
            sample_list = []
            for ind in range(modulus):
                sample_list.append(np.sum(sample[ind::modulus]))
            return_counts.append(sample_list)

        return return_counts

    
    @setting(7, num_to_read="i", returns="*s*i")
    def read_tag_stream(self, c, num_to_read=None): #from apd tagger ###need to update
        """This will take in the same three level list as read_counter_internal, but the third level is not a number of counts, but actually a list of time tags
            It will return a list of samples. Each sample is a list of gates. Each gates is a list of time tags
            It will also return a list of channels. Each channel is a list of gates. In each gate is a list of the channel associated to each time tag
        """
        results = fetching_tool(self.pending_experiment_job, data_list=["counts","times"], mode="wait_for_all")
        
        all_channels_list = []
        all_times_list = []
        
        while results.is_processing():
            # Fetch results
            counts, times = results.fetch_all()
            num_samples = len(counts)
            num_apds = len(counts[0])
            num_gates = len(counts[0][0])
            last_ind = 0
            
            ordered_time_tags = [] #this will be a list of sample. each sample is a list of gates. each gate is a list of sorted time tags
            ordered_channels = [] #this will be a list of samples. each sample is a list of gates. each gate is a list of channels for the corresponding time tags
            
            for i in range(num_samples):
                sample_counts = counts[i]
                all_gate_time_tags = []
                all_gate_channels = []
                
                for j in range(num_gates):
                    gate_counts = sample_counts[i]
                    all_apd_time_tags = []
                    channels = []
                    
                    for k in range(num_apd):
                        apd_counts = gate_counts[k] #gate_counts will be a single number
                        apd_time_tags = times[last_ind:last_ind+apd_counts]
                        all_apd_time_tags = all_apd_time_tags + apd_time_tags
                        channels = channels+np.full(len(apd_time_tags),apd_indices[k]).tolist()
                        last_ind = last_ind+apd_counts
                    
                    sorting = np.argsort(np.array(all_apd_time_tags))
                    all_apd_time_tags = np.array(all_apd_time_tags)[sorting]
                    all_apd_time_tags = all_apd_time_tags.tolist()
                    channels = np.array(channels)[sorting]
                    channels = channels.tolist()
                    
                    all_gate_time_tags.append(all_apd_time_tags)
                    all_gate_channels.append(channels)
                
                ordered_time_tags.append(all_gate_time_tags)
                ordered_channels.append(all_gate_channels)
            
        return ordered_time_tags, ordered_channels 

    
    @setting(8, apd_indices="*i", gate_indices="*i", clock="b") # from apd tagger. 
    def start_tag_stream(self, c, apd_indices, gate_indices=None, clock=True):
        """opx doesn't need it"""
        pass
    
    @setting(9, apd_indices="*i", gate_indices="*i", clock="b") # from apd tagger. 
    def stop_tag_stream(self, c, apd_indices, gate_indices=None, clock=True):
        """OPX doesn't need it"""
        pass
    
    @setting(10)
    def clear_buffer(self, c): # from apd tagger
        """OPX doesn't need it"""
        pass

    #%% xy server functions
    ### looking into piezo controller to see if it can be given a stream of voltages, like we do with the daq, and triggered digitally
    @setting(41, xVoltage="v[]")
    def write_x(self, c, xVoltage):
        """Write the specified voltages to the xy positioning outputs.

        Params
            xVoltage: float
                Voltage to write to the x channel
        """
        # need this to write a constant output to the x channel that gets overwritten by its next instruction
        
    @setting(51, yVoltage="v[]")
    def write_y(self, c, yVoltage):
        """Write the specified voltages to the xy positioning outputs.

        Params
            yVoltage: float
                Voltage to write to the y channel
        """
        # need this to write a constant output to the y channel that gets overwritten by its next instruction
        
    @setting(11, xVoltage="v[]", yVoltage="v[]")
    def write_xy(self, c, xVoltage, yVoltage):
        """Write the specified voltages to the xy positioning outputs.

        Params
            xVoltage: float
                Voltage to write to the x channel
            yVoltage: float
                Voltage to write to the y channel
        """
        self.write_x(c, xVoltage)
        self.write_y(c, xVoltage)

        # Close the stream task if it exists
        # This can happen if we quit out early
        # want this to make the z and y analog outputs set to a constant voltage of choice and it to stay at that voltage indefinitely. How do I do that? ask kevin

    @setting(12, returns="*v[]")
    def read_xy(self, c):
        """Return the current voltages on the x and y channels.

        Returns
            list(float)
                Current voltages on the x and y channels

        """
       # want this to just report the current output at these two channels. how do I do that?
        return voltages[0], voltages[1]
    

    @setting(13,x_center="v[]",y_center="v[]",x_range="v[]",y_range="v[]",num_steps="i",period="i",returns="*v[]*v[]",)
    def load_sweep_scan_xy(self, c, x_center, y_center, x_range, y_range, num_steps, period):
        """Load a scan that will wind through the grid defined by the passed
        parameters. x and y voltage streams are saved to the class and can be used in the qua program. Currently x_range
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
                Expected period between clock signals in ns. doesn't matter for opx

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
        x_inter = numpy.concatenate((x_voltages_1d, numpy.flipud(x_voltages_1d)))
        # [1, 2, 3, 3, 2, 1] => [1, 2, 3, 3, 2, 1, 1, 2, 3]
        if y_num_steps % 2 == 0:  # Even x size
            x_voltages = numpy.tile(x_inter, int(y_num_steps / 2))
        else:  # Odd x size
            x_voltages = numpy.tile(x_inter, int(numpy.floor(y_num_steps / 2)))
            x_voltages = numpy.concatenate((x_voltages, x_voltages_1d))

        # [4, 5, 6] => [4, 4, 4, 5, 5, 5, 6, 6, 6]
        y_voltages = numpy.repeat(y_voltages_1d, x_num_steps)
        
        self.x_voltages_1d = x_voltages_1d
        self.y_voltages_1d = y_voltages_1d
        self.z_voltages_1d = []
        
        # need a command here that will tell the opx to put the position outputs at the first in the series, 

        return x_voltages_1d, y_voltages_1d
    
    @setting(14,x_center="v[]",y_center="v[]",scan_range="v[]",num_steps="i",period="i",returns="*v[]",)
    def load_scan_x(self, c, x_center, y_center, scan_range, num_steps, period):
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
                Expected period between clock signals in ns. doesn't matter for opx

        Returns
            list(float)
                The x voltages that make up the scan
        """

        half_scan_range = scan_range / 2

        x_low = x_center - half_scan_range
        x_high = x_center + half_scan_range

        x_voltages = numpy.linspace(x_low, x_high, num_steps)
        y_voltages = numpy.full(num_steps, y_center)

        self.x_voltages_1d = x_voltages_1d
        self.y_voltages_1d = y_voltages_1d
        self.z_voltages_1d = []
        # need a command here that will tell the opx to put the position outputs at the first in the series, 

        return x_voltages

    @setting(15,x_center="v[]",y_center="v[]",scan_range="v[]",num_steps="i",period="i",returns="*v[]",)
    def load_scan_y(self, c, x_center, y_center, scan_range, num_steps, period):
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
                Expected period between clock signals in ns. doesn't matter for opx

        Returns
            list(float)
                The y voltages that make up the scan
        """

        half_scan_range = scan_range / 2

        y_low = y_center - half_scan_range
        y_high = y_center + half_scan_range

        x_voltages = numpy.full(num_steps, x_center)
        y_voltages = numpy.linspace(y_low, y_high, num_steps)

        self.x_voltages_1d = x_voltages_1d
        self.y_voltages_1d = y_voltages_1d
        self.z_voltages_1d = []
        # need a command here that will tell the opx to put the position outputs at the first in the series, 

        return y_voltages

    @setting(16, x_points="*v[]", y_points="*v[]", period="i")
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

        x_voltages_1d = x_points
        y_voltages_1d = y_points

        self.x_voltages_1d = x_voltages_1d
        self.y_voltages_1d = y_voltages_1d
        self.z_voltages_1d = []
        
        # need a command here that will tell the opx to put the position outputs at the first in the series, 
        
        return
    
    #%% z positioning functions
    ### same as above. In the long term we should get a daq so we can use that with the z piezo in the new setup
    
    @setting(17, voltage="v[]")
    def write_z(self, c, voltage):
        """Write the specified voltage to the z channel"""

        # same problem as xy
        
        return

    @setting(18, returns="v[]")
    def read_z(self, c):
        """Return the current voltages on the z channel"""
        
        # same problem as xy
        
        return voltage

    @setting(19, center="v[]", scan_range="v[]", num_steps="i", period="i", returns="*v[]")
    def load_scan_z(self, c, center, scan_range, num_steps, period):
        """Load a linear sweep in z"""

        half_scan_range = scan_range / 2
        low = center - half_scan_range
        high = center + half_scan_range
        voltages = numpy.linspace(low, high, num_steps)
        self.z_voltages_1d = voltages
        self.x_voltages_1d = []
        self.y_voltages_1d = []
        # need a command here that will tell the opx to put the position outputs at the first in the series, 
        
        return voltages

    @setting(20, z_voltages="*v[]", period="i", returns="*v[]")
    def load_arb_scan_z(self, c, z_voltages, period):
        """Load a list of voltages with the DAQ"""
        self.z_voltages_1d = z_voltages
        self.x_voltages_1d = []
        self.y_voltages_1d = []
        # need a command here that will tell the opx to put the position outputs at the first in the series, 
        
        return z_voltages
    
    
    @setting(12,x_center="v[]",z_center="v[]",x_range="v[]",z_range="v[]",num_steps="i",period="i",returns="*v[]*v[]",)
    def load_sweep_scan_xz(self, c, x_center, y_center, z_center, x_range, z_range, num_steps, period):
        """Load a scan that will wind through the grid defined by the passed
        parameters. 

        Normal scan performed, starts in bottom right corner, and starts
        heading left
        

        Params
            x_center: float
                Center x voltage of the scan
            y_center: float
                Center position y voltage (won't change in y)
            z_center: float
                Center z voltage of the scan
            x_range: float
                Full scan range in x
            z_range: float
                Full scan range in z
            num_steps: int
                Number of steps the break the ranges into
            period: int
                Expected period between clock signals in ns, doesn't matter for opx

        Returns
            list(float)
                The x voltages that make up the scan
            list(float)
                The z voltages that make up the scan
        """

        # Must use same number of steps right now
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

        # Apply scale and offset to get the voltages we'll apply.
        x_voltages_1d = numpy.linspace(x_low, x_high, num_steps)
        z_voltages_1d = numpy.linspace(z_low, z_high, num_steps)
    

        ######### Works for any x_range, y_range #########

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
        x_voltages.fill(x_center)

        self.x_voltages_1d = x_voltages
        self.y_voltages_1d = y_voltages
        self.z_voltages_1d = z_voltages
        
        # need a command here that will tell the opx to put the position outputs at the first in the series, 

        return x_voltages_1d, z_voltages_1d
    
    
    #%% arbitary waveform generator functions. 
    ### mostly just pass

    #all the 'load' functions are not necessary on the OPX
    #the pulses need to exist in the configuration file and they are used in the qua sequence
    
    def iq_comps(phase, amp):
        if type(phase) is list:
            ret_vals = []
            for val in phase:
                ret_vals.append(numpy.round(amp * numpy.exp((0+1j) * val), 5))
            return (numpy.real(ret_vals).tolist(), numpy.imag(ret_vals).tolist())
        else:
            ret_val = numpy.round(amp * numpy.exp((0+1j) * phase), 5)
            return (numpy.real(ret_val), numpy.imag(ret_val))
        
    @setting(13, amp="v[]")
    def set_i_full(self, c, amp):
        pass
    
    @setting(14)
    def load_knill(self, c):
        pass
        
    @setting(15, phases="*v[]")
    def load_arb_phases(self, c, phases):
        pass
        
    @setting(16, num_dd_reps="i")
    def load_xy4n(self, c, num_dd_reps):
        pass
        
    @setting(17, num_dd_reps="i")
    def load_xy8n(self, c, num_dd_reps):
        pass

    @setting(18, num_dd_reps="i")
    def load_cpmg(self, c, num_dd_reps):
        pass
    
    
    #%% reset the opx. this is called in reset_cfm in the tool_belt. we don't need it to do anything
    
    def reset(self, c):
        pass
        
    
    
    
    #%%


__server__ = OPX()

if __name__ == "__main__":
    from labrad import util

    util.runServer(__server__)
