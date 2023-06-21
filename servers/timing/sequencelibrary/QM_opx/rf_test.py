#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test sequence for driving AODs with the OPX

Created on June 21st, 2023

@author: mccambria
"""


import numpy
import utils.tool_belt as tool_belt
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm import qua
from qm import SimulationConfig
from qm.qua import program, declare, declare_stream, stream_processing
from qm.qua import measure, wait, save, play, align, fixed
from utils.tool_belt import States


def qua_program(element, freq, amp, duration):

    with program() as seq:
        qua.update_frequency(element, freq * 1e6)
        a = declare(fixed, value=amp)
        qua.play("continuous" * qua.amp(a), element, duration=duration)

    return seq


def get_seq(opx, config, args, num_repeat):
    element, freq, amp, duration = args
    seq = qua_program(element, freq, amp, duration)
    final = ""
    # specify what one 'sample' means for  readout
    sample_size = "all_reps"
    num_gates = 0
    return seq, final, [], num_gates, sample_size


if __name__ == "__main__":b
    from qualang_tools.results import fetching_tool, progress_counter
    import matplotlib.pylab as plt
    import time

    config = tool_belt.get_config_dict()
    qmm = QuantumMachinesManager(host="128.104.160.117", port="80")
    qm = qmm.open_qm(config_opx)

    simulation_duration = 35000 // 4  # clock cycle units - 4ns

    num_repeat = 3

    args = [100, 1000.0, 350, 100, 3, "cobolt_515", 1]
    seq, f, p, ns, ss = get_seq([], config, args, num_repeat)

    plt.figure()

    job_sim = qm.simulate(seq, SimulationConfig(simulation_duration))
    job_sim.get_simulated_samples().con1.plot()
    plt.show()
