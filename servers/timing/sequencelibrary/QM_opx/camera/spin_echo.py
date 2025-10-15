# -*- coding: utf-8 -*-
"""
Widefield ESR

Created on October 13th, 2023

@author: mccambria
"""

import matplotlib.pyplot as plt
import numpy as np
from qm import QuantumMachinesManager, qua
from qm.simulate import SimulationConfig

import utils.common as common
from servers.timing.sequencelibrary.QM_opx import seq_utils
from servers.timing.sequencelibrary.QM_opx.camera import base_scc_sequence


def get_seq(base_scc_seq_args, step_vals, num_reps=1):
    buffer = seq_utils.get_widefield_operation_buffer()
    # uwave_ind_list = base_scc_seq_args[-1]
    # macro_pi_pulse_duration = seq_utils.get_macro_pi_pulse_duration(uwave_ind_list)
    # macro_pi_on_2_pulse_duration = seq_utils.get_macro_pi_on_2_pulse_duration(
    #     uwave_ind_list
    # )

    step_vals = [
        seq_utils.convert_ns_to_cc(el) for el in step_vals
    ]
    if np.any(np.less(step_vals, 4)):
        raise RuntimeError("Negative wait duration")

    with qua.program() as seq:
        seq_utils.init()
        seq_utils.macro_run_aods()
        step_val = qua.declare(int)

        def uwave_macro_sig(uwave_ind_list, step_val):
            qua.align()
            seq_utils.macro_pi_on_2_pulse(uwave_ind_list)
            qua.wait(step_val)
            seq_utils.macro_pi_pulse(uwave_ind_list)
            qua.wait(step_val)
            seq_utils.macro_pi_on_2_pulse(uwave_ind_list)
            qua.wait(buffer)

    seq_ret_vals = []
    return seq, seq_ret_vals


if __name__ == "__main__":
    config_module = common.get_config_module()
    config = config_module.config
    opx_config = config_module.opx_config
    opx_config["pulses"]["yellow_spin_pol"]["length"] = 2e3

    qm_opx_args = config["DeviceIDs"]["QM_opx_args"]
    qmm = QuantumMachinesManager(**qm_opx_args)
    opx = qmm.open_qm(opx_config)

    try:
        seq, seq_ret_vals = get_seq(
            [
                [[109.062, 107.003], [110.183, 105.383], [110.417, 108.653]],
                [10000, 10000, 10000],
                [1.0, 1.0, 1.0],
                [[73.477, 72.33], [74.328, 70.992], [74.594, 73.662]],
                [168, 184, 220],
                [1.0, 1.0, 1.0],
                [False, False, False],
                [0, 1],
            ],
            [
                27200,
                41832,
                41500,
                78500,
                74500,
                78000,
                39668,
            ],
            10,
        )

        sim_config = SimulationConfig(duration=int(100e3 / 4))
        sim = opx.simulate(seq, sim_config)
        samples = sim.get_simulated_samples()
        samples.con1.plot()
        plt.show(block=True)

    except Exception as exc:
        raise exc
    finally:
        qmm.close_all_quantum_machines()
