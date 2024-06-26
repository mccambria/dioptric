#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test sequence for driving AODs with the OPX

Created on June 21st, 2023

@author: mccambria
"""


import numpy
import qm
from qm import qua
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.simulate import SimulationConfig
from qm.qua import program, declare, declare_stream, stream_processing
from qm.qua import measure, wait, save, play, align, fixed, assign
from qm.qua import infinite_loop_, while_
from utils.tool_belt import States
import servers.timing.sequencelibrary.QM_opx.seq_utils as seq_utils
import utils.common as common
import utils.tool_belt as tb
import utils.kplotlib as kpl
import matplotlib.pyplot as plt
from qm import generate_qua_script


def qua_program(element, freq, amp, duration, num_reps=1):
    with program() as seq:
        ### Non-repeated stuff here
        qua.update_frequency(element, freq * 1e6)
        a = declare(fixed, value=amp)
        clock_cycles = round(duration / 4)

        ### Define one rep here
        def one_rep():
            qua.play("cw" * qua.amp(a), element, duration=clock_cycles)

        ### Handle the reps in the utils code
        seq_utils.handle_reps(one_rep, num_reps)

    return seq


def get_seq(opx_config, config, args, num_reps=1):
    element, freq, amp, duration = args
    seq = qua_program(element, freq, amp, duration, num_reps)
    final = ""
    # specify what one 'sample' means for  readout
    sample_size = "all_reps"
    num_gates = 0
    return seq, final, [], num_gates, sample_size


if __name__ == "__main__":
    kpl.init_kplotlib(font_size=kpl.Size.SMALL)

    config_module = common.get_config_module()
    config = config_module.config
    opx_config = config_module.opx_config

    ip_address = config["DeviceIDs"]["QM_opx_ip"]
    qmm = QuantumMachinesManager(ip_address)
    opx = qmm.open_qm(opx_config)

    try:
        seq_args = ["ao1", 10, 0.4, 100]
        seq = qua_program(*seq_args, 10)
        
        # Serialize to file
        # sourceFile = open('debug.py', 'w')
        # print(generate_qua_script(seq, opx_config), file=sourceFile)
        # sourceFile.close()

        sim_config = SimulationConfig(duration=5000 // 4)
        sim = opx.simulate(seq, sim_config)
        samples = sim.get_simulated_samples()
        samples.con1.plot(analog_ports=["1"])

    except Exception as exc:
        print(exc)
    finally:
        qmm.close_all_quantum_machines()
        qmm.close()
