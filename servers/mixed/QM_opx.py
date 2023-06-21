# -*- coding: utf-8 -*-
"""
server for the Quantum Machines OPX

Created on August 29th, 2022

@author: carter fox

### BEGIN NODE INFO
[info]
name = QM_opx
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
from qm import qua
from qm import SimulationConfig
from qualang_tools.results import fetching_tool, progress_counter
from labrad.server import LabradServer
from labrad.server import setting
from utils import common
from utils import tool_belt as tb
import numpy as np
import importlib
import numpy
import logging
import sys
import os
import socket
from servers.inputs.interfaces.tagger import Tagger
from servers.timing.interfaces.pulse_gen import PulseGen

class QmOpx(Tagger, PulseGen, LabradServer):
    # region Setup

    name = "QM_opx"
    pc_name = socket.gethostname()
    # steady_state_program_file = 'steady_state_program_test_opx.py'

    def initServer(self):
        
        nvdata_path = common.get_nvdata_path()
        filename = nvdata_path / f"pc_{self.pc_name}/labrad_logging/{self.name}.log"
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)-8s %(message)s",
            datefmt="%y-%m-%d_%H-%M-%S",
            filename=filename,
        )
        
        config_module = common.get_config_module()
        opx_config = config_module.opx_config
        config = common.get_config_dict()
        self.config = config
        self.opx_config = opx_config
        
        ip_address = config["DeviceIDs"]["QM_opx_ip"]
        logging.info(ip_address)
        self.qmm = QuantumMachinesManager(ip_address)
        self.qm = self.qmm.open_qm(opx_config)
        self.steady_state_program_file = "constant.py"

        repo_path = common.get_repo_path()
        opx_sequence_library_path = (
            repo_path / f"servers/timing/sequencelibrary/{self.name}"
        )
        sys.path.append(str(opx_sequence_library_path))
        self.steady_state_option = False
        # logging.info(tb.get_mod_type('cobolt_515'))

        # steady_state_seq, final_ss, period_ss = get_seq(self, self.steady_state_program_file, self.steady_state_seq_args_string, 1)
        # self.pending_steady_state_compiled_program_id = self.compile_qua_sequence(self.qm,steady_state_seq)

        self.apd_indices = config["apd_indices"]
        ss_params = config["SteadyStateParameters"]["QmOpx"]
        self.steady_state_digital_on = ss_params["steady_state_digital_on"]
        self.steady_state_analog_on = ss_params["steady_state_analog_on"]
        val = ss_params["steady_state_analog_freqs"]
        self.steady_state_analog_freqs = np.array(val).astype(float).tolist()
        val = ss_params["steady_state_analog_amps"]
        self.steady_state_analog_amps = np.array(val).astype(float).tolist()

        self.steady_state_seq_args = [
            self.steady_state_digital_on,
            self.steady_state_analog_on,
            self.steady_state_analog_freqs,
            self.steady_state_analog_amps,
        ]
        self.steady_state_seq_args_string = tb.encode_seq_args(
            self.steady_state_seq_args
        )

        self.steady_state_seq, final_ss, period_ss = self.get_seq(
            self.steady_state_program_file, self.steady_state_seq_args_string, 1
        )
        logging.info("Init complete")
        self.tagger_di_clock = int(config["Wiring"]["Tagger"]["di_apd_gate"])

    def stopServer(self):
        try:
            self.qmm.close_all_quantum_machines()
            self.qmm.close()
        except:
            pass

    # endregion

    # region Sequencing

    # compiles the program ands adds it to the desired quantum machine
    def compile_qua_sequence(self, quantum_machine, program):
        compilied_program_id = quantum_machine.compile(program)
        return compilied_program_id

    def set_steady_state_option_on_off(
        self, selection
    ):  # selection should be true or false
        self.steady_state_option = selection

    def add_qua_sequence_to_qm_queue(self, quantum_machine, compilied_program_id):
        # logging.info('start here')
        # a = time.time()
        program_job = quantum_machine.queue.add_compiled(compilied_program_id)
        # logging.info(time.time()-a)
        # logging.info('end here')
        return program_job

    def get_seq(self, seq_file, seq_args_string, num_reps):
        """
        For the OPX, this will grab the desired sequence with the desired number of repetitions
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
            args = tb.decode_seq_args(seq_args_string)
            ret_vals = seq_module.get_seq(self, self.config, args, num_reps)
            seq, final, ret_vals, self.num_gates_per_rep, self.sample_size = ret_vals

        return seq, final, ret_vals

    @setting(13, seq_file="s", seq_args_string="s", returns="*?")
    def stream_load(self, c, seq_file, seq_args_string=""):  
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
        # logging.info('made it here')
        self.qmm.close_all_quantum_machines()
        self.qm = self.qmm.open_qm(self.opx_config)
        self.seq_file = seq_file
        self.seq_args_string = seq_args_string
        # logging.info(seq_args_string)
        seq, final, ret_vals = self.get_seq(self.seq_file, seq_args_string, 1)
        # logging.info('hi')
        self.pending_experiment_compiled_program_id = self.compile_qua_sequence(
            self.qm, seq
        )

        return ret_vals

    @setting(14, num_reps="i")
    def stream_start(self, c, num_reps=1):
        """
        For the OPX, this will run the qua program already grabbed by 
        stream_load(). If num_reps is greater than 1, it will get the 
        sequence again, this time with the number of repetitions and then run it.
        num_reps: int
            The number of times the sequence is repeated, such as the number of reps in a rabi routine
        """
        
        self.num_reps = num_reps

        # if num_reps == -1:
        #     num_reps = 10000  # just make it go a bunch of times for now. canceling it will just kill the operation
        #     self.num_reps = 1
        #     logging.info(f"repeating {num_reps} times instead of indefinitely")
        # # if we want to repeat it, get the sequence again but with all the repetitions.
        # elif num_reps >= 2:  
        #     self.qmm.close_all_quantum_machines()
        #     self.qm = self.qmm.open_qm(self.opx_config)

        #     seq, final, ret_vals = self.get_seq(
        #         self.seq_file, self.seq_args_string, num_reps
        #     )  # gets the full sequence
        #     self.pending_experiment_compiled_program_id = self.compile_qua_sequence(
        #         self.qm, seq
        #     )
            
        seq, final, ret_vals = self.get_seq(
            self.seq_file, self.seq_args_string, num_reps
        )

        # logging.info('starting here')
        # st = time.time()
        self.pending_experiment_job = self.add_qua_sequence_to_qm_queue(
            self.qm, self.pending_experiment_compiled_program_id
        )
        # logging.info(time.time() - st)

        self.experiment_job = self.pending_experiment_job.wait_for_execution()
        # logging.info(time.time() - st)
        # logging.info('ending')

        self.counter_index = 0

        return

    @setting(15, high_digital_channels="*s", analog_elements_to_set="*s",
             analog_frequencies="*v[]", analog_amplitudes="*v[]")
    def constant(self, c, high_digital_channels=[], analog_elements_to_set=[],
                 analog_frequencies=[], analog_amplitudes=[]):
        """
        Set the OPX to an infinite loop, ouputing the desired things on 
        each channel.
        """

        high_digital_channels = np.asarray(high_digital_channels)
        analog_amplitudes = np.asarray(analog_amplitudes)
        analog_elements_to_set = np.asarray(analog_elements_to_set)
        analog_frequencies = np.asarray(analog_frequencies)

        args = [
            high_digital_channels.tolist(),
            analog_elements_to_set.tolist(),
            analog_frequencies.tolist(),
            analog_amplitudes.tolist(),
        ]
        # logging.info(args)

        args_string = tb.encode_seq_args(args)
        self.stream_immediate(
            c, seq_file="constant.py", num_reps=1, seq_args_string=args_string
        )

    # endregion
    # region Time tagging
    # from apd tagger. for the opx it fetches the results from the job. Don't think num_to_read has to do anything
    def read_counter_internal(self):  
        """
        This is the core function that any tagger needs in order to function 
        as a counter. 
        For the OPX this fetches the data from the job that was created when 
        the program was executed. Assumes "counts" is one of the data streams
        The count stream should be a three level list. First level is the 
        sample, second is the apds, third is the different gates. First index 
        gives the sample. next level gives the gate. next level gives which apd
        [  [ [],[] ] , [ [],[] ], [ [],[] ]  ]

        Params
            num_to_read: int
                This is not needed for the OPX
        Returns
            return_counts: array
                This is an array of the counts
        """
        # st = time.time()
        if self.sample_size == "one_rep":
            num_gates_per_sample = self.num_gates_per_rep
            results = fetching_tool(
                self.experiment_job,
                data_list=["counts_apd0", "counts_apd1"],
                mode="live",
            )

        elif self.sample_size == "all_reps":
            # logging.info('waiting for all')
            num_gates_per_sample = self.num_reps * self.num_gates_per_rep
            results = fetching_tool(
                self.experiment_job,
                data_list=["counts_apd0", "counts_apd1"],
                mode="wait_for_all",
            )
            # logging.info('got them')
            # logging.info(time.time()-st)

        (
            counts_apd0,
            counts_apd1,
        ) = (
            results.fetch_all()
        )  # just not sure if its gonna put it into the list structure we want
        # logging.info('fetched them')
        # logging.info(time.time()-st)
        # logging.info('checkpoint')
        # logging.info(counts_apd0)
        # now we need to combine into our data structure. they have different lengths because the fpga may
        # save one faster than the other. So just go as far as we have samples on both
        num_new_samples_both = min(
            int(len(counts_apd0) / num_gates_per_sample),
            int(len(counts_apd1) / num_gates_per_sample),
        )
        max_length = num_new_samples_both * num_gates_per_sample

        # get only the number of samples that both have
        counts_apd0 = counts_apd0[self.counter_index : max_length]
        counts_apd1 = counts_apd1[self.counter_index : max_length]
        # logging.info(counts_apd0)
        # now we need to sum over all the iterative readouts that occur if the readout time is longer than 1ms
        counts_apd0 = np.sum(counts_apd0, 1).tolist()
        counts_apd1 = np.sum(counts_apd1, 1).tolist()

        ### now I buffer the list
        n = num_gates_per_sample

        counts_apd0 = [
            counts_apd0[i * n : (i + 1) * n]
            for i in range((len(counts_apd0) + n - 1) // n)
        ]
        counts_apd1 = [
            counts_apd1[i * n : (i + 1) * n]
            for i in range((len(counts_apd1) + n - 1) // n)
        ]

        return_counts = []

        if len(self.apd_indices) == 2:
            for i in range(len(counts_apd0)):
                return_counts.append([counts_apd0[i], counts_apd1[i]])

        elif len(self.apd_indices) == 1:
            for i in range(len(counts_apd0)):
                return_counts.append([counts_apd0[i]])

        # logging.info('checkpoint1')
        # logging.info(return_counts)
        self.counter_index = max_length  # make the counter indix the new max length (-1) so the samples start there
        # logging.info('done processing counts')
        # logging.info(time.time()-st)
        return return_counts

    def read_raw_stream(self):
        """
        Read the raw stream. currently it waits for all data in the job to 
        come in and reports it all. Ideally it would do it live
        """
        # logging.info('at read raw stream')
        results = fetching_tool(
            self.experiment_job,
            data_list=["counts_apd0", "counts_apd1", "times_apd0", "times_apd1"],
            mode="wait_for_all",
        )

        counts_apd0, counts_apd1, times_apd0, times_apd1 = results.fetch_all()
        # logging.info(np.shape(counts_apd0))
        c1 = counts_apd0.tolist()
        c2 = counts_apd1.tolist()

        # *1000 to convert ps to ns
        t1 = (times_apd0[1::] * 1000).tolist()
        t2 = (times_apd1[1::] * 1000).tolist()

        config = self.config
        max_readout_time = config["PhotonCollection"]["qm_opx_max_readout_time"]
        max_readout_time *= 1000  # To ns
        gate_open_channel = config["Wiring"]["Tagger"]["di_apd_gate"]
        gate_close_channel = int(-1 * gate_open_channel)

        all_time_tags = []
        all_channels = []
        # logging.info('made it to loops')
        running_sum_c1 = 0
        running_sum_c2 = 0

        for gate_ind in range(len(c1)):
            cur_gate_counts_list = c1[gate_ind]
            all_channels.append(gate_open_channel)
            all_time_tags.append(0)

            for sub_gate_ind in range(len(cur_gate_counts_list)):
                sub_gate_counts1 = c1[gate_ind][sub_gate_ind]
                sub_gate_counts2 = c2[gate_ind][sub_gate_ind]

                all_past_counts1 = int(
                    running_sum_c1 + np.sum(c1[gate_ind][0:sub_gate_ind])
                )
                all_past_counts2 = int(
                    running_sum_c2 + np.sum(c2[gate_ind][0:sub_gate_ind])
                )

                sub_gate_time_tags1 = t1[
                    all_past_counts1 : all_past_counts1 + sub_gate_counts1
                ]
                sub_gate_time_tags2 = t1[
                    all_past_counts2 : all_past_counts2 + sub_gate_counts2
                ]

                sub_gate_time_tags1 = np.array(sub_gate_time_tags1) + (
                    sub_gate_ind * max_readout_time
                )
                sub_gate_time_tags2 = np.array(sub_gate_time_tags2) + (
                    sub_gate_ind * max_readout_time
                )

                total_sub_gate_tags = np.append(
                    sub_gate_time_tags1, sub_gate_time_tags2
                )
                total_sub_gate_channels = np.append(
                    np.full(len(sub_gate_time_tags1), 0),
                    np.full(len(sub_gate_time_tags2), 1),
                )

                sort_inds = np.argsort(total_sub_gate_tags)

                all_time_tags.extend((total_sub_gate_tags[sort_inds]).tolist())
                all_channels.extend((total_sub_gate_channels[sort_inds]).tolist())
                # all_time_tags = all_time_tags + (total_sub_gate_tags[sort_inds]).tolist()
                # all_channels = all_channels + (total_sub_gate_channels[sort_inds]).tolist()

            running_sum_c1 = running_sum_c1 + np.sum(c1[gate_ind])
            running_sum_c2 = running_sum_c2 + np.sum(c2[gate_ind])

            all_channels.append(gate_close_channel)
            all_time_tags.append(0)

        t_return = np.array(all_time_tags).astype(np.int64).astype(str)

        return t_return, np.array(all_channels)

    @setting(17, gate_indices="*i", clock="b")  # from apd tagger.
    def start_tag_stream(self, c, gate_indices=None, clock=True):
        self.stream = True
        pass

    @setting(18)  # from apd tagger.
    def stop_tag_stream(self, c):
        self.stream = None
        pass

    @setting(19)
    def clear_buffer(self, c):
        """OPX does not need this - used by Swabian Time Tagger"""
        pass

    # endregion
    # region Conditional logic

    @setting(10805, num_streams="i")
    def get_cond_logic_num_ops(self, c, num_streams):
        """
        This function assumes you are trying to save a num_ops stream in the sequence to keep track of how many conditional logic
        operations you do in each run. For instance, how many readouts did you do in each iteration of the sequence, where you stop reading
        out if you met some condition.
        """

        data_list = ["num_ops_{}".format(1 + i) for i in range(num_streams)]
        logging.info("at cond logic")
        results = fetching_tool(self.experiment_job, data_list, mode="wait_for_all")
        return_streams = results.fetch_all()
        return return_streams

    # endregion
    # region AWG and RF signal generator

    # all the 'load' functions are not necessary on the OPX
    # the pulses need to exist in the configuration file and they are used in the qua sequence

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
    # endregion

    @setting(26)
    def reset(self, c):
        self.qmm.close_all_quantum_machines()
        self.qm = self.qmm.open_qm(self.opx_config)

        if self.steady_state_option:
            self.pending_steady_state_compiled_program_id = self.compile_qua_sequence(
                self.qm, self.steady_state_seq
            )
            self.pending_steady_state_job = self.add_qua_sequence_to_qm_queue(
                self.qm, self.pending_steady_state_compiled_program_id
            )
            self.steady_state_job = self.pending_steady_state_job.wait_for_execution()


__server__ = QmOpx()

if __name__ == "__main__":
    from labrad import util

    util.runServer(__server__)
