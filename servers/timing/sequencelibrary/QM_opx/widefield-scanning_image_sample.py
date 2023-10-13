# -*- coding: utf-8 -*-
"""
Scanning illumination and widefield collection

Created on October 13th, 2023

@author: mccambria
"""


import numpy
import qm
from qm import qua
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.simulate import SimulationConfig
from qm.qua import program, declare, declare_stream, stream_processing
from qm.qua import wait, update_frequency, play, align, fixed, assign, for_each_
import servers.timing.sequencelibrary.QM_opx.seq_utils as seq_utils
import utils.common as common
import utils.tool_belt as tb
import utils.kplotlib as kpl
import matplotlib.pyplot as plt
from qm import generate_qua_script


def qua_program(coords_1, coords_2, readout, readout_laser, readout_power):
    laser_element = f"do_{readout_laser}_dm"
    camera_element = f"do_camera_trigger"
    x_element = f"ao_{readout_laser}_x"
    y_element = f"ao_{readout_laser}_y"
    clock_cycles = readout / 4  # * 4 ns / clock_cycle = 1 us
    x_freq = declare(fixed)
    y_freq = declare(fixed)
    with program() as seq:
        with for_each_((x_freq, y_freq), (coords_1, coords_2)):
            update_frequency(x_element, x_freq * 10**6)
            update_frequency(y_element, y_freq * 10**6)
            play("cw", x_element, duration=clock_cycles)
            play("cw", y_element, duration=clock_cycles)
            play("on", laser_element, duration=clock_cycles)
            play("on", camera_element, duration=clock_cycles)

    return seq


def get_seq(opx_config, config, args, num_reps=-1):
    coords_1, coords_2, readout, readout_laser, readout_power = args
    seq = qua_program(coords_1, coords_2, readout, readout_laser, readout_power)
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
        args = [0, 1e6, "laser_OPTO_589"]
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
