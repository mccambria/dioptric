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


def get_seq(
    base_scc_seq_args,
    step_inds=None,
    num_reps=1,
):
    reference = False  # References for this sequence are handled routine-side

    # MCC
    total_num_steps = len(step_inds)
    half_num_steps = total_num_steps // 2
    esr_pulse_duration = seq_utils.convert_ns_to_cc(68)

    with qua.program() as seq:
        seq_utils.init()
        seq_utils.macro_run_aods()

        step_ind = qua.declare(int)

        def uwave_macro(uwave_ind_list, step_ind):
            # MCC
            with qua.if_(step_ind < half_num_steps):
                seq_utils.macro_pi_pulse(uwave_ind_list, duration_cc=esr_pulse_duration)
            with qua.else_():
                seq_utils.macro_pi_pulse(uwave_ind_list)

        with qua.for_each_(step_ind, step_inds):
            base_scc_sequence.macro(
                base_scc_seq_args,
                uwave_macro,
                step_ind,
                num_reps=num_reps,
                reference=reference,
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
                    [110.31633152241405, 109.89129171060787],
                    [109.78333152241404, 109.92829171060788],
                ],
                [
                    [74.96497948092767, 75.05933281333888],
                    [74.41497948092767, 75.28333281333887],
                ],
                [140, 140],
                [1, 1],
                [],
                [0, 1],
            ],
            [0, 1, 2],
            2,
        )

        sim_config = SimulationConfig(duration=int(300e3 / 4))
        sim = opx.simulate(seq, sim_config)
        samples = sim.get_simulated_samples()
        samples.con1.plot()
        plt.show(block=True)

    except Exception as exc:
        raise exc
    finally:
        qmm.close_all_quantum_machines()
