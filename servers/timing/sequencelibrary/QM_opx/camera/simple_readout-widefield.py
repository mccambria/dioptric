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
from servers.timing.sequencelibrary.QM_opx import seq_utils
import utils.common as common
import utils.tool_belt as tb
import utils.kplotlib as kpl
from utils.constants import ModMode
import matplotlib.pyplot as plt
from qm import generate_qua_script


def get_seq(args, num_reps):
    readout_duration, readout_laser = args
    if num_reps == None:
        num_reps = 1

    laser_element = seq_utils.get_laser_mod_element(readout_laser)
    camera_element = f"do_camera_trigger"
    readout_duration_cc = round(readout_duration / 4)
    with qua.program() as seq:
        ### Define one rep here
        def one_rep():
            qua.play("on", laser_element, duration=readout_duration_cc)
            qua.play("on", camera_element)
            qua.align()
            qua.play("off", camera_element)
            # qua.align()

        ### Handle the reps in the utils code
        seq_utils.handle_reps(one_rep, num_reps)

    seq_ret_vals = []
    return seq, seq_ret_vals


if __name__ == "__main__":
    config_module = common.get_config_module()
    config = config_module.config
    opx_config = config_module.opx_config

    ip_address = config["DeviceIDs"]["QM_opx_ip"]
    qmm = QuantumMachinesManager(ip_address)
    opx = qmm.open_qm(opx_config)

    try:
        args = [3500.0, "laser_OPTO_589"]
        seq, seq_ret_vals = get_seq(args, 5)

        sim_config = SimulationConfig(duration=round(10e3 / 4))
        sim = opx.simulate(seq, sim_config)
        samples = sim.get_simulated_samples()
        samples.con1.plot()
        plt.show(block=True)

    except Exception as exc:
        print(exc)
    finally:
        qmm.close_all_quantum_machines()
        qmm.close()
