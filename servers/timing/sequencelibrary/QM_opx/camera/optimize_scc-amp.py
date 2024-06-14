# -*- coding: utf-8 -*-
"""
Widefield ESR

Created on October 13th, 2023

@author: mccambria
"""

import matplotlib.pyplot as plt
from qm import QuantumMachinesManager, qua
from qm.simulate import SimulationConfig

import utils.common as common
from servers.timing.sequencelibrary.QM_opx import seq_utils
from servers.timing.sequencelibrary.QM_opx.camera import base_scc_sequence


def get_seq(base_scc_seq_args, scc_amp_steps, num_reps):
    with qua.program() as seq:
        scc_amp_override = qua.declare(qua.fixed)

        def uwave_macro_sig(uwave_ind_list, step_val):
            seq_utils.macro_pi_pulse(uwave_ind_list)

        def uwave_macro_ref(uwave_ind_list, step_val):
            pass

        base_scc_sequence.macro(
            base_scc_seq_args,
            [uwave_macro_sig, uwave_macro_ref],
            step_vals=scc_amp_steps,
            num_reps=num_reps,
            scc_amp_override=scc_amp_override,
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
                    [108.73558077916097, 109.58849887552114],
                    [109.41358077916098, 110.19749887552115],
                    [108.86258077916098, 110.22749887552115],
                    [108.21258077916097, 110.48949887552115],
                ],
                [
                    [73.71607634356549, 75.00994679683262],
                    [74.26107634356549, 75.48294679683262],
                    [73.74607634356549, 75.46294679683263],
                    [73.2590763435655, 75.72694679683262],
                ],
                [120, 120, 120, 120],
                [1.0, 1.0, 1.0, 1.0],
                [],
                [0, 1],
            ],
            [
                0.7,
                0.74,
                1.1400000000000001,
                0.86,
                1.2200000000000002,
                0.78,
                1.1,
                1.02,
                0.9,
                1.3,
                1.1800000000000002,
                0.82,
                1.26,
                0.94,
                0.98,
                1.06,
            ],
            5,
        )

        sim_config = SimulationConfig(duration=int(250e3 / 4))
        sim = opx.simulate(seq, sim_config)
        samples = sim.get_simulated_samples()
        samples.con1.plot()
        plt.show(block=True)

    except Exception as exc:
        raise exc
    finally:
        qmm.close_all_quantum_machines()
