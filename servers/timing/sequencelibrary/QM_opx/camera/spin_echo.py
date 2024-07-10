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
    uwave_ind_list = base_scc_seq_args[-1]
    macro_pi_pulse_duration = seq_utils.get_macro_pi_pulse_duration(uwave_ind_list)
    step_vals = [
        seq_utils.convert_ns_to_cc(el) - macro_pi_pulse_duration for el in step_vals
    ]
    if np.any(np.less(step_vals, 4)):
        raise RuntimeError("Negative wait duration")

    with qua.program() as seq:

        def uwave_macro_sig(uwave_ind_list, step_val):
            # for uwave_ind in uwave_ind_list:
            qua.align()
            seq_utils.macro_pi_on_2_pulse(uwave_ind_list)
            qua.wait(step_val)
            seq_utils.macro_pi_pulse(uwave_ind_list)
            qua.wait(step_val)
            seq_utils.macro_pi_on_2_pulse(uwave_ind_list)
            qua.wait(buffer)

        # def uwave_macro_sig(uwave_ind_list, step_val):
        #     for uwave_ind in uwave_ind_list:
        #         qua.align()
        #         seq_utils.macro_pi_on_2_pulse([uwave_ind])
        #         qua.wait(step_val)
        #         # seq_utils.macro_pi_pulse([uwave_ind])
        #         # seq_utils.macro_pi_on_2_pulse_b([uwave_ind])
        #         # qua.wait(step_val)
        #         seq_utils.macro_pi_on_2_pulse([uwave_ind])
        #         qua.wait(buffer)

        base_scc_sequence.macro(base_scc_seq_args, uwave_macro_sig, step_vals, num_reps)

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
                    [109.22251952469692, 108.47143630712519],
                    [109.90051952469693, 109.0804363071252],
                    [109.34951952469693, 109.1104363071252],
                    [108.69951952469692, 109.3724363071252],
                ],
                [
                    [74.10743467866433, 74.03473932250022],
                    [74.65243467866433, 74.50773932250021],
                    [74.13743467866433, 74.48773932250022],
                    [73.65043467866434, 74.75173932250021],
                ],
                [140, 140, 140, 140],
                [0.942, 0.91, 0.87, 0.94],
                [],
                [0, 1],
            ],
            [36000, 35500, 41832, 37832],
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
