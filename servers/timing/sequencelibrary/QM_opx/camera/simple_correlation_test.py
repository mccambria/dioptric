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
        qua_random = qua.Random()

        def uwave_macro_sig(uwave_ind_list, step_val):
            qua.assign(step_val, qua_random.rand_int(2))
            with qua.if_(step_val == 1):
                seq_utils.macro_pi_pulse(uwave_ind_list)

        base_scc_sequence.macro(base_scc_seq_args, uwave_macro_sig, num_reps=num_reps)

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
                    [108.31887532965924, 110.19598961196738],
                    [108.75887532965923, 110.44698961196738],
                    [109.01087532965924, 110.79498961196738],
                    [108.43987532965923, 110.84898961196738],
                ],
                [
                    [73.03248366966189, 75.43273396827259],
                    [73.30848366966188, 75.57973396827258],
                    [73.56748366966188, 75.84673396827257],
                    [73.05748366966189, 75.83073396827258],
                ],
                [
                    124,
                    104,
                    116,
                    208,
                ],
                [0, 2],
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
