#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Constant sequence for the QM OPX

Created on June 21st, 2023

@author: mccambria
"""


import numpy
import qm
from qm import qua
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.simulate import SimulationConfig
from qm.qua import program, declare, declare_stream, stream_processing
from qm.qua import measure, wait, save, play, align, fixed, assign, for_each_
import servers.timing.sequencelibrary.QM_opx.seq_utils as seq_utils
from utils.tool_belt import States
import utils.common as common
import utils.tool_belt as tb
import utils.positioning as positioning
import utils.kplotlib as kpl
import matplotlib.pyplot as plt
from qm import generate_qua_script


def qua_program(
    x_center,
    y_center,
    x_range,
    y_range,
    x_num_steps,
    y_num_steps,
    readout,
    laser_name,
    readout_power,
    num_reps=1,
):
    x_freqs, y_freqs, _, _, _ = positioning.get_scan_grid_2d(
        x_center, y_center, x_range, y_range, x_num_steps, y_num_steps, dtype=int
    )
    x_freqs = x_freqs.tolist()
    y_freqs = y_freqs.tolist()
    readout_qua = round(readout / 4)  # * 4 ns / clock_cycle
    # x_element = f"ao1"
    x_element = f"{laser_name}_x"
    y_element = f"{laser_name}_y"

    with program() as seq:
        ### Non-repeated stuff here
        readout_power_qua = declare(fixed, value=readout_power)

        x_freq = declare(int)
        y_freq = declare(int)

        ### Define one rep here
        def one_rep():
            with for_each_((x_freq, y_freq), (x_freqs, y_freqs)):
                qua.update_frequency(x_element, x_freq)
                qua.update_frequency(y_element, y_freq)
                qua.play("readout", x_element)
                qua.play("readout", y_element)
                align()

        ### Handle the reps in the utils code
        seq_utils.handle_reps(one_rep, num_reps)

    return seq


def get_seq(opx_config, config, args, num_reps=1):
    (
        x_center,
        y_center,
        x_range,
        y_range,
        x_num_steps,
        y_num_steps,
        readout,
        laser_name,
        readout_power,
    ) = args
    seq = qua_program(
        x_center,
        y_center,
        x_range,
        y_range,
        x_num_steps,
        y_num_steps,
        readout,
        laser_name,
        readout_power,
        num_reps,
    )
    final = ""
    # specify what one 'sample' means for  readout
    sample_size = "all_reps"
    num_gates = 0
    return seq, final, [], num_gates, sample_size


if __name__ == "__main__":
    kpl.init_kplotlib()
    # fig, ax = plt.subplots()
    # ax.plot([1, 2, 3, 5])
    # plt.show(block=True)

    config_module = common.get_config_module()
    config = config_module.config
    opx_config = config_module.opx_config

    ip_address = config["DeviceIDs"]["QM_opx_ip"]
    qmm = QuantumMachinesManager(ip_address)
    opx = qmm.open_qm(opx_config)

    try:
        args = [
            75000000.0,
            75000000.0,
            10000000.0,
            10000000.0,
            3,
            3,
            1000.0,
            "cobolt_638",
            0.45,
        ]
        seq = qua_program(*args, -1)  # num_reps=1
        # Serialize to file
        # sourceFile = open('debug2.py', 'w')
        # print(generate_qua_script(seq, opx_config), file=sourceFile)
        # sourceFile.close()

        sim_config = SimulationConfig(duration=50000 // 4)
        sim = opx.simulate(seq, sim_config)
        samples = sim.get_simulated_samples()
        samples.con1.plot()
        plt.show(block=True)

    except Exception as exc:
        print(exc)
    finally:
        qmm.close_all_quantum_machines()
        qmm.close()
