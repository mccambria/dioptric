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
from servers.timing.sequencelibrary.QM_opx.camera import base_sequence


def get_seq(
    pol_coords_list,
    ion_coords_list,
    anticorrelation_ind_list,
    uwave_ind,
    num_reps=1,
):
    sig_gen_el = seq_utils.get_sig_gen_element(uwave_ind)
    buffer = seq_utils.get_widefield_operation_buffer()

    with qua.program() as seq:
        qua_random = qua.Random()

        def uwave_macro_sig(step_val):
            qua.assign(step_val, qua_random.rand_int(2))
            # qua.assign(step_val, 1)
            qua.align()
            with qua.if_(step_val == 1):
                qua.play("pi_pulse", sig_gen_el)
                qua.wait(buffer, sig_gen_el)

        base_sequence.macro(
            pol_coords_list,
            ion_coords_list,
            anticorrelation_ind_list,
            uwave_ind,
            uwave_macro_sig,
            num_reps=num_reps,
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
                    [108.20663774042235, 109.18919887824842],
                    [108.63163774042235, 109.42319887824841],
                ],
                [
                    [108.20663774042235, 109.18919887824842],
                ],
                [
                    [73.2138344723166, 74.44585432573876],
                    [73.4068344723166, 74.57285432573876],
                ],
                0,
            ],
            10,
        )

        sim_config = SimulationConfig(duration=int(200e3 / 4))
        sim = opx.simulate(seq, sim_config)
        samples = sim.get_simulated_samples()
        samples.con1.plot()
        plt.show(block=True)

    except Exception as exc:
        raise exc
    finally:
        qmm.close_all_quantum_machines()
