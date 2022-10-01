# -*- coding: utf-8 -*-
"""
server for the Quantum Machines OPX

Created on August 29th, 2022

@author: carter fox

### BEGIN NODE INFO
[info]
name = qm_opx
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
from qualang_tools.results import fetching_tool, progress_counter
import matplotlib.pyplot as plt
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
import os
import utils.tool_belt as tool_belt
import socket
from numpy import pi
root2_on_2 = numpy.sqrt(2) / 2
from qualang_tools.units import unit
from pathlib import Path
# u = unit()
from servers.inputs.interfaces.tagger import Tagger
from servers.timing.interfaces.pulse_gen import PulseGen
from opx_configuration_file import *


class OPX(Tagger, PulseGen, LabradServer):

    name = "qm_opx"
    pc_name = socket.gethostname()
    # steady_state_program_file = 'steady_state_program_test_opx.py'
    

    def initServer(self):
        filename = 'E:/Shared drives/Kolkowitz Lab Group/nvdata/pc_{}/labrad_logging/{}.log'
        filename = filename.format(self.pc_name, self.name)
        logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    datefmt='%y-%m-%d_%H-%M-%S', filename=filename)
        self.get_config_dict()
        logging.info(qop_ip)
        self.qmm = QuantumMachinesManager(qop_ip) # opens communication with the QOP so we can make a quantum machine
        self.qm = self.qmm.open_qm(config_opx)
        self.steady_state_program_file = 'steady_state_program_test_opx.py'
        
        opx_sequence_library_path = (
            Path.home()
            / "Documents/GitHub/kolkowitz-nv-experiment-v1.0/servers/timing/sequencelibrary/OPX_sequences"
        )
        sys.path.append(str(opx_sequence_library_path))
        self.steady_state_option = False
        
    
    def get_config_dict(self):
        """
        Get the config dictionary on the registry recursively. Very similar
        to the function of the same name in tool_belt.
        """
        config_dict = {}
        _ = ensureDeferred(
            self.populate_config_dict(["", "Config"], config_dict)
        )
        _.addCallback(self.on_get_config_dict, config_dict)

    async def populate_config_dict(self, reg_path, dict_to_populate):
        """Populate the config dictionary recursively"""

        # Sub-folders
        p = self.client.registry.packet()
        p.cd(reg_path)
        p.dir()
        result = await p.send()
        sub_folders, keys = result["dir"]
        for el in sub_folders:
            sub_dict = {}
            sub_path = reg_path + [el]
            await self.populate_config_dict(sub_path, sub_dict)
            dict_to_populate[el] = sub_dict

        # Keys
        if len(keys) == 1:
            p = self.client.registry.packet()
            p.cd(reg_path)
            key = keys[0]
            p.get(key)
            result = await p.send()
            val = result["get"]
            dict_to_populate[key] = val

        elif len(keys) > 1:
            p = self.client.registry.packet()
            p.cd(reg_path)
            for key in keys:
                p.get(key)
            result = await p.send()
            vals = result["get"]

            for ind in range(len(keys)):
                key = keys[ind]
                val = vals[ind]
                dict_to_populate[key] = val

    def on_get_config_dict(self, _, config_dict):
        self.config_dict = config_dict
        self.apd_indices = config_dict["apd_indices"]
        logging.info("Init complete")
        
        
        
    def stopServer(self):
        try:
            self.qmm.close_all_quantum_machines()
            self.qmm.close()
        except:
            pass
    
    def set_steady_state_option_on_off(self, selection): #selection should be true or false
        self.steady_state_option = selection
        
    #%% sequence loading and executing functions ###
    ### looks good. It's ready to queue up both the experiment job and the infinite loop job. I can test it with simple programs and config
    
    # compiles the program ands adds it to the desired quantum machine 
    def compile_qua_sequence(self,quantum_machine,program):
        compilied_program_id = quantum_machine.compile(program)
        return compilied_program_id
    
    def add_qua_sequence_to_qm_queue(self,quantum_machine,compilied_program_id):
        program_job = quantum_machine.queue.add_compiled(compilied_program_id)
        return program_job
        
    def get_seq(self, seq_file, seq_args_string, num_repeat): # this returns the qua sequence without repeats. from pulse streamer. DONE
        """For the OPX, this will grab the desired sequence with the desired number of repetitions
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
            seq_module = importlib.import_module(file_name+"_opx")
            args = tool_belt.decode_seq_args(seq_args_string)
                        
            seq, final, ret_vals = seq_module.get_seq(self,self.config_dict, args, num_repeat )
        
        return seq, final, ret_vals


    
    @setting(13, seq_file="s", seq_args_string="s", returns="*?")
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
        self.qmm.close_all_quantum_machines()
        self.qm = self.qmm.open_qm(config_opx)
        self.seq_file = seq_file
        self.seq_args_string = seq_args_string
        seq, final, ret_vals = self.get_seq(self.seq_file, seq_args_string, 1)
        self.pending_experiment_compiled_program_id = self.compile_qua_sequence(self.qm, seq)
        
        # if self.steady_state_option:
        #     self.pending_steady_state_compiled_program_id = self.compile_qua_sequence(self.qm,self.steady_state_program)
        
        return ret_vals

    @setting(14, num_repeat="i")
    def stream_start(self, c, num_repeat=1): # from pulse streamer. It will start the full sequence. DONE
        """For the OPX, this will run the qua program already grabbed by stream_load(). If num_repeat is greater than 1, it will get the sequence again, this time with the
            number of repetitions and then run it.
            num_repeat: int
                The number of times the sequence is repeated, such as the number of reps in a rabi routine
        """
        
        if num_repeat >= 2:   # if we want to repeat it, get the sequence again but with all the repetitions.
            self.qmm.close_all_quantum_machines()
            self.qm = self.qmm.open_qm(config_opx)
            
            seq, final, ret_vals = self.get_seq(self.seq_file, self.seq_args_string, num_repeat) #gets the full sequence
            self.pending_experiment_compiled_program_id = self.compile_qua_sequence(self.qm, seq)
            
        self.pending_experiment_job = self.add_qua_sequence_to_qm_queue(self.qm,self.pending_experiment_compiled_program_id)
        self.experiment_job = self.pending_experiment_job.wait_for_execution()
        self.counter_index = 0
        # if self.steady_state_option:
        #     self.pending_steady_state_compiled_program_id = self.compile_qua_sequence(self.qm,self.steady_state_program)
        
        return 
    
    
    @setting(15, digital_channels="*i", analog_0_voltage="v[]", analog_1_voltage="v[]")
    def constant(self,c,digital_channels=[],analog_0_voltage=0.0,analog_1_voltage=0.0):
        """Set the OPX to an infinite loop."""
        """ This function is not finished. I made a starting file (constant_program_opx.py) but it needs to be finished """
        high_digital_channels = digital_channels
        analog_channels_to_set = [0, 1]
        analog_channel_values = [analog_0_voltage, analog_1_voltage]
        args = [high_digital_channels,analog_channels_to_set,analog_channel_values]
        
        args_string = tool_belt.encode_seq_args(args)
        self.stream_immediate(seq_file='constant_program_opx.py', num_repeat=1, seq_args_string=args_string)
        
    #%% counting and time tagging functions ### 
    ### currently in good shape. Waiting to see if the time tag processing function I made will work with how timetags are saved
    ### these also need to be tested once I get the opx up and running again
    
    def read_counter_internal(self): #from apd tagger. for the opx it fetches the results from the job. Don't think num_to_read has to do anything
        """This is the core function that any tagger we have needs. 
        For the OPX this fetches the data from the job that was created when the program was executed. 
        Assumes "counts" is one of the data streams
        The count stream should be a three level list. First level is the sample, second is the apds, third is the different gates. 
        first index gives the sample. next level gives the gate. next level gives which apd
        [  [ [],[] ] , [ [],[] ], [ [],[] ]  ]
        
        Params
            num_to_read: int
                This is not needed for the OPX
        Returns
            return_counts: array
                This is an array of the counts 
        """
        

        results = fetching_tool(self.experiment_job, data_list = ["counts_apd0","counts_apd1"], mode="live")
    
        counts_apd0, counts_apd1 = results.fetch_all() #just not sure if its gonna put it into the list structure we want
        
        #now we need to combine into our data structure. they have different lengths because the fpga may 
        #save one faster than the other. So just go as far as we have samples on both
        max_length = min(len(counts_apd0),len(counts_apd1))
        
        counts_apd0 = counts_apd0[self.counter_index:max_length]
        counts_apd1 = counts_apd1[self.counter_index:max_length]
        
        #now we need to sum over all the iterative readouts that occur if the readout time is longer than 1ms
        counts_apd0 = np.sum(counts_apd0,2).tolist()
        counts_apd1 = np.sum(counts_apd1,2).tolist()

        return_counts = []
        
        if len(self.apd_indices)==2:
            for i in range(len(counts_apd0)):
                return_counts.append([counts_apd0[i],counts_apd1[i]])
                
        elif len(self.apd_indices)==1:
            for i in range(len(counts_apd0)):
                return_counts.append([counts_apd0[i]])
            
        self.counter_index = max_length #make the counter indix the new max length (-1) so the samples start there

        return return_counts
    
   
    
    def read_raw_stream(self):
        # if self.stream is None:
        #     logging.error("read_raw_stream attempted while stream is None.")
        #     return
        # # buffer = self.stream.getData()
        # results = fetching_tool(job, data_list = ["counts_apd0","counts_apd1","times_apd0","times_apd1","rep_readout_delay"], mode="live")
        # counts_apd0, counts_apd1, times_apd0, times_apd1, rep_readout_delay = results.fetch_all() 
        # times_apd0 = times_apd0[1:1+np.sum(counts_apd0)]
        # times_apd1 = times_apd1[1:1+np.sum(counts_apd1)]
        
        
        # for i in range(0,len(counts_apd0[0][0])):
        #     start_index = np.sum(counts_apd0[0][0][0:i])
        #     end_index = np.sum(counts_apd0[0][0][0:i+1])
        #     times_apd0[start_index:end_index] = ( times_apd0[start_index:end_index] + ( i* (rep_readout_delay + max_readout_time)) )
            
        # for i in range(0,len(counts_apd1[0][0])):
        #     start_index = np.sum(counts_apd1[0][0][0:i])
        #     end_index = np.sum(counts_apd1[0][0][0:i+1])
        #     times_apd1[start_index:end_index] = ( times_apd1[start_index:end_index] + ( i* (rep_readout_delay + max_readout_time)) )
        logging.info('TIMETAGGING NOT SET UP YET FOR OPX')
        raise RuntimeError
        return timestamps, channels
        
    
    
    @setting(16, num_to_read="i", returns="*s*i")
    def read_tag_stream(self, c, num_to_read=None): #from apd tagger ###need to update
        """This will take in the same three level list as read_counter_internal, but the third level is not a number of counts, but actually a list of time tags
            It will return a list of samples. Each sample is a list of gates. Each gates is a list of time tags
            It will also return a list of channels. Each channel is a list of gates. In each gate is a list of the channel associated to each time tag
        """
        # results = fetching_tool(self.experiment_job, data_list=["counts","times"], mode="wait_for_all")
        
        # res_handles_tagstream = self.experiment_job.result_handles
        # res_handles_tagstream.wait_for_all_values()
        # counts_data = res_handles_tagstream.get("counts").fetch_all()
        # times_data = res_handles_tagstream.get("times").fetch_all()
        
        # # counts_data = results.res_handles.counts.fetch_all()
        # # times_data = results.res_handles.times.fetch_all()
        
        # counts_data = counts_data[0][0].tolist()

        # times_data = (times_data).tolist()
        # times_data = np.asarray(times_data)*1e3
        # # times_data = times_data.astype(int)
        # times_data = times_data.tolist()
        # times_data = [np.int64(val) for val in times_data]
        # times_data = [t[0] for t in times_data]
        
        # logging.info('test')
        # logging.info(counts_data)
        # logging.info(times_data)
        
        # num_samples = len(counts_data)
        # num_apds = len(counts_data[0])
        # last_ind = 0
        
        # ordered_time_tags = [] #this will be a list of sample. each sample is a list of gates. each gate is a list of sorted time tags
        # ordered_channels = [] #this will be a list of samples. each sample is a list of gates. each gate is a list of channels for the corresponding time tags

        # for i in range(num_samples):
        #     sample_counts = counts_data[i]
        #     all_apd_time_tags = []
        #     channels = []
                
        #     for k in range(num_apds):
                
        #         apd_counts = sample_counts[k][0] #gate_counts will be a single number
        #         apd_time_tags = times_data[last_ind:last_ind+apd_counts]
        #         all_apd_time_tags = all_apd_time_tags + apd_time_tags
        #         channels = channels+np.full(len(apd_time_tags),self.apd_indices[k]).tolist()
        #         last_ind = last_ind+apd_counts
            
        #     sorting = np.argsort(np.array(all_apd_time_tags))
        #     all_apd_time_tags = np.array(all_apd_time_tags)[sorting]
        #     all_apd_time_tags = all_apd_time_tags.tolist()
        #     channels = np.array(channels)[sorting]
        #     channels = channels.tolist()
            
        #     ordered_time_tags.append(all_apd_time_tags)
        #     ordered_channels.append(channels)
            
        # ordered_time_tags = np.asarray(ordered_time_tags).astype('str').tolist()
        
            
        # flat_ordered_channels_list = [item for sublist in ordered_channels for item in sublist]
        # flat_ordered_time_tags_list = [item for sublist in ordered_time_tags for item in sublist]

        # return flat_ordered_time_tags_list, flat_ordered_channels_list 
        # return ordered_time_tags, ordered_channels
        logging.info('TIMETAGGING NOT SET UP YET FOR OPX')
        raise RuntimeError
        return flat_ordered_time_tags_list, flat_ordered_channels_list 

    
    @setting(17, apd_indices="*i", gate_indices="*i", clock="b") # from apd tagger. 
    def start_tag_stream(self, c, apd_indices, gate_indices=None, clock=True):
        self.stream = True
        pass        
    
    @setting(18, apd_indices="*i", gate_indices="*i", clock="b") # from apd tagger. 
    def stop_tag_stream(self, c, apd_indices, gate_indices=None, clock=True):
        self.stream = None
        pass
    
    @setting(19)
    def clear_buffer(self, c): # from apd tagger
        """OPX doesn't need it"""
        pass

    
    #%% arbitary waveform generator functions. 
    ### mostly just pass

    #all the 'load' functions are not necessary on the OPX
    #the pulses need to exist in the configuration file and they are used in the qua sequence
    
    # def iq_comps(phase, amp):
    #     if type(phase) is list:
    #         ret_vals = []
    #         for val in phase:
    #             ret_vals.append(numpy.round(amp * numpy.exp((0+1j) * val), 5))
    #         return (numpy.real(ret_vals).tolist(), numpy.imag(ret_vals).tolist())
    #     else:
    #         ret_val = numpy.round(amp * numpy.exp((0+1j) * phase), 5)
    #         return (numpy.real(ret_val), numpy.imag(ret_val))
        
    # @setting(20, amp="v[]")
    # def set_i_full(self, c, amp):
    #     pass
    
    # @setting(21)
    # def load_knill(self, c):
    #     pass
        
    # @setting(22, phases="*v[]")
    # def load_arb_phases(self, c, phases):
    #     pass
        
    # @setting(23, num_dd_reps="i")
    # def load_xy4n(self, c, num_dd_reps):
    #     pass
        
    # @setting(24, num_dd_reps="i")
    # def load_xy8n(self, c, num_dd_reps):
    #     pass

    # @setting(25, num_dd_reps="i")
    # def load_cpmg(self, c, num_dd_reps):
    #     pass
    
    
    # %% reset the opx. this is called in reset_cfm in the tool_belt. we don't need it to do anything
    @setting(26)
    def reset(self, c):
        pass
        

    #%%


__server__ = OPX()

if __name__ == "__main__":
    
    # opx_configuration_file_path = "C:/Users/kolkowitz/Documents/GitHub/kolkowitz-nv-experiment-v1.0"
    # sys.path.append(opx_configuration_file_path)

    # from opx_configuration_file import *
    # sys.path.append("C:/Users/kolkowitz/Documents/GitHub/kolkowitz-nv-experiment-v1.0/servers/timing/sequencelibrary/OPX_sequences")
    
    from labrad import util

    util.runServer(__server__)
    
    # opx = OPX()
    # opx.initServer()
    # # meas_len=1000
    # # opx.set_steady_state_option_on_off(True)
    # # delay, readout_time, apd_index, laser_name, laser_power = args
    # readout_time = 120000
    # args_test = [200,readout_time,0,'green_laser_do',1]
    # args_test_str = tool_belt.encode_seq_args(args_test)
    # opx.stream_load('test_program_opx.py',args_test_str)
    # opx.stream_start(1)
    # all_counts = opx.read_counter_complete()
    # counts_simple = opx.read_counter_simple()
    # counts_gates = opx.read_counter_separate_gates()
    # # times, channels = opx.read_tag_stream()
    # print('')
    # print('')
    # print(all_counts)
    # print('')
    # print(counts_simple)
    # print('')
    # print(counts_gates)
    # print('')
    # # print(times)
    # print('')
    # # print(channels)
    # # opx.stream_immediate('test_program_opx.py')
    # # print('hi')