# -*- coding: utf-8 -*-
"""
Widefield ESR

Created on October 13th, 2023

@author: mccambria
@author: Saroj Chand
"""

import time

import matplotlib.pyplot as plt
import numpy as np
from qm import QuantumMachinesManager, qua
from qm.simulate import SimulationConfig

import utils.tool_belt as tb
import utils.common as common
from servers.timing.sequencelibrary.QM_opx import seq_utils
from servers.timing.sequencelibrary.QM_opx.camera import base_scc_sequence

def get_seq(base_scc_seq_args, tau, num_reps=1):
    tau_cc = seq_utils.convert_ns_to_cc(tau)
    with qua.program() as seq:
        seq_utils.init()
        seq_utils.macro_run_aods()

        def uwave_macro_sig(uwave_ind_list, step_val):
            seq_utils.macro_pi_on_2_pulse(uwave_ind_list, phase=0)
            qua.wait(tau_cc)
            seq_utils.macro_pi_pulse(uwave_ind_list, phase=0)
            qua.wait(tau_cc)
            seq_utils.macro_pi_on_2_pulse(uwave_ind_list, phase=99)

        def uwave_macro_ref(uwave_ind_list, step_val):
            return True

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
    tb.set_delays_to_zero(opx_config)
    opx_config["pulses"]["yellow_spin_pol"]["length"] = 2e3

    qm_opx_args = config["DeviceIDs"]["QM_opx_args"]
    qmm = QuantumMachinesManager(**qm_opx_args)
    opx = qmm.open_qm(opx_config)


    try:
        seq, seq_ret_vals = get_seq(
            [
                [[109.114, 107.084], [110.468, 108.724]],
                [1000, 1000],
                [1.0, 1.0],
                [[73.686, 72.605], [74.759, 73.921]],
                [116, 108],
                [1.0, 1.0],
                [False, False],
                [0, 1],
            ],
            19.6e3,
            5,
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
