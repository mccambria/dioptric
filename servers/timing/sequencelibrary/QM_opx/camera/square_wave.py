#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Constant sequence for the QM OPX

Created on June 21st, 2023

@author: mccambria
"""


import matplotlib.pyplot as plt
import numpy
import qm
from qm import generate_qua_script, qua
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.simulate import SimulationConfig

import utils.common as common
import utils.kplotlib as kpl
import utils.tool_belt as tb
from servers.timing.sequencelibrary.QM_opx import seq_utils


def get_seq(args, num_reps=None):
    digital_channels, analog_channels, analog_voltages, period = args
    if num_reps == None:
        num_reps = -1

    half_period_cc = seq_utils.convert_ns_to_cc(period / 2, allow_rounding=True)
    with qua.program() as seq:
        ### Non-repeated stuff here
        num_analog_channels = len(analog_channels)
        amps = [None] * num_analog_channels
        for ind in range(len(analog_channels)):
            chan = analog_channels[ind]
            element = f"ao{chan}"
            qua.update_frequency(element, 0)
            # Declare amplitudes
            amp = analog_voltages[ind]
            amps[ind] = qua.declare(qua.fixed, value=amp)

        ### Define one rep here
        def one_rep():
            for chan in digital_channels:
                element = f"do{chan}"
                qua.play("on", element, duration=half_period_cc)
                qua.play("off", element, duration=half_period_cc)
            for ind in range(len(analog_channels)):
                chan = analog_channels[ind]
                element = f"ao{chan}"
                amp = amps[ind]
                qua.play("cw" * qua.amp(amp), element, duration=half_period_cc)
                qua.play("off", element, duration=half_period_cc)
            # qua.play("on", "ao_laser_OPTO_589_am", duration=clock_cycles)
            # qua.play("cw" * qua.amp(0), element, duration=clock_cycles)

        ### Handle the reps in the utils code
        seq_utils.handle_reps(one_rep, num_reps, wait_for_trigger=False)
        # one_rep()

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
        args = [[], [1], [1.0], 1000]
        ret_vals = get_seq(args)
        seq, seq_ret_vals = ret_vals

        # Serialize to file
        # sourceFile = open('debug2.py', 'w')
        # print(generate_qua_script(seq, opx_config), file=sourceFile)
        # sourceFile.close()

        sim_config = SimulationConfig(duration=10000 // 4)
        sim = opx.simulate(seq, sim_config)
        samples = sim.get_simulated_samples()
        samples.con1.plot()
        plt.show(block=True)

    except Exception as exc:
        print(exc)
    finally:
        qmm.close_all_quantum_machines()
        qmm.close()
