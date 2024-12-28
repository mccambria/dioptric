# -*- coding: utf-8 -*-
"""
Widefield ESR

Created on October 13th, 2023

@author: mccambria
"""

import time

import matplotlib.pyplot as plt
import numpy as np
from qm import QuantumMachinesManager, qua
from qm.simulate import SimulationConfig

import utils.common as common
from servers.timing.sequencelibrary.QM_opx import seq_utils
from servers.timing.sequencelibrary.QM_opx.camera import base_scc_sequence


def get_seq(base_scc_seq_args, num_reps=1):
    with qua.program() as seq:
        seq_utils.init()
        seq_utils.macro_run_aods()
        qua_random = qua.Random()

        def uwave_macro_sig(uwave_ind_list, step_val):
            qua.assign(step_val, qua_random.rand_int(2))
            with qua.if_(step_val == 1):
                seq_utils.macro_pi_pulse(uwave_ind_list)

        def uwave_macro_ref(uwave_ind_list, step_val):
            pass

        base_scc_sequence.macro(
            base_scc_seq_args,
            [uwave_macro_sig, uwave_macro_ref],
            num_reps=num_reps,
            reference=False,
        )

    seq_ret_vals = []
    return seq, seq_ret_vals


if __name__ == "__main__":
    config_module = common.get_config_module()
    config = config_module.config
    opx_config = config_module.opx_config

    qm_opx_args = config["DeviceIDs"]["QM_opx_args"]
    qmm = QuantumMachinesManager(**qm_opx_args)
    opx = qmm.open_qm(opx_config)

    try:
        seq, seq_ret_vals = get_seq(
            [
                [
                    [107.10254616156483, 109.48113965080046],
                    [108.73154616156484, 111.06213965080046],
                    [109.54654616156483, 109.73813965080046],
                    [109.48954616156483, 111.64113965080047],
                ],
                [
                    [72.15897428830978, 74.83507649222732],
                    [73.53597428830977, 76.09007649222731],
                    [74.23797428830977, 75.06807649222732],
                    [74.17097428830978, 76.61007649222732],
                ],
                [156, 192, 176, 144],
                [0, 1],
                [0, 1],
            ],
            10,
        )

        sim_config = SimulationConfig(duration=int(500e3 / 4))
        sim = opx.simulate(seq, sim_config)
        samples = sim.get_simulated_samples()
        samples.con1.plot()
        plt.show(block=True)

    except Exception as exc:
        raise exc
    finally:
        qmm.close_all_quantum_machines()
