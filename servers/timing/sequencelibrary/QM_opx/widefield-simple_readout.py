# -*- coding: utf-8 -*-
"""
Widefield illumination and collection

Created on October 5th, 2023

@author: mccambria
"""


import numpy
from qm import qua
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.simulate import SimulationConfig
from qm.qua import program, declare, declare_stream, stream_processing
from qm.qua import measure, wait, save, play, align, fixed, assign
import servers.timing.sequencelibrary.QM_opx.seq_utils as seq_utils
import utils.common as common
import utils.tool_belt as tb
import utils.kplotlib as kpl
import matplotlib.pyplot as plt
from qm import generate_qua_script


def qua_program(readout, readout_laser):
    laser_element = f"do_{readout_laser}_dm"
    camera_element = f"do_camera_trigger"
    digital_elements = [laser_element, camera_element]
    num_reps = readout / 1000  # Num of us cycles
    clock_cycles = 250  # * 4 ns / clock_cycle = 1 us
    with program() as seq:
        ### Define one rep here
        def one_rep():
            for el in digital_elements:
                qua.play("on", el, duration=clock_cycles)

        ### Handle the reps in the utils code
        seq_utils.handle_reps(one_rep, num_reps)
        qua.play("off", camera_element, duration=clock_cycles)

    return seq


def get_seq(opx_config, config, args, num_reps=-1):
    readout, readout_laser = args
    seq = qua_program(readout, readout_laser)
    final = ""
    # specify what one 'sample' means for  readout
    sample_size = "all_reps"
    num_gates = 0
    return seq, final, [], num_gates, sample_size


if __name__ == "__main__":
    config_module = common.get_config_module()
    config = config_module.config
    opx_config = config_module.opx_config

    ip_address = config["DeviceIDs"]["QM_opx_ip"]
    qmm = QuantumMachinesManager(ip_address)
    # qmm.close_all_quantum_machines()
    # print(qmm.list_open_quantum_machines())
    opx = qmm.open_qm(opx_config)

    try:
        args = [1e6, "laser_OPTO_589"]
        ret_vals = get_seq(opx_config, config, args)
        seq, final, ret_vals, _, _ = ret_vals

        sim_config = SimulationConfig(duration=round(1.5e6 / 4))
        sim = opx.simulate(seq, sim_config)
        samples = sim.get_simulated_samples()
        samples.con1.plot()
        plt.show(block=True)

    except Exception as exc:
        print(exc)
    finally:
        qmm.close_all_quantum_machines()
        qmm.close()
