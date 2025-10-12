# -*- coding: utf-8 -*-
"""
Widefield ESR

Created on October 3th, 2025

@author: schand
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
    buffer = seq_utils.get_widefield_operation_buffer()
    revival = 19.6e4 # in ns
    step_val = seq_utils.convert_ns_to_cc(revival)
    with qua.program() as seq:
        seq_utils.init()
        seq_utils.macro_run_aods()
        step_ind = qua.declare(int)

        def uwave_macro(uwave_ind_list, step_ind):
            MW_NV = [uwave_ind_list[1]]  # NV microwave chain (~2.87 GHz)
            RF = [uwave_ind_list[0]]  # RF chain (~133 MHz)
            qua.align()
            seq_utils.macro_pi_on_2_pulse(MW_NV, phase=9)
            qua.wait(step_val)
            seq_utils.macro_pi_pulse(RF, phase=9)
            seq_utils.macro_pi_pulse(MW_NV, phase=9)
            qua.wait(step_val)
            seq_utils.macro_pi_on_2_pulse(MW_NV, phase=99)
            qua.wait(buffer)

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
    opx_config["pulses"]["yellow_spin_pol"]["length"] = 10e3

    qm_opx_args = config["DeviceIDs"]["QM_opx_args"]
    qmm = QuantumMachinesManager(**qm_opx_args)
    opx = qmm.open_qm(opx_config)

    try:
        seq, seq_ret_vals = get_seq(
            [
                [[108.477, 107.282], [109.356, 108.789]],
                [220, 220],
                [1.0, 1.0],
                [[73.558, 71.684], [74.227, 72.947]],
                [124, 124],
                [1.0, 1.0],
                [False, False],
                [0,1],
            ],
            [70, 219],
            1,
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
