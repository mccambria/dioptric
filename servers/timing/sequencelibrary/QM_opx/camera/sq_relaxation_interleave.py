# -*- coding: utf-8 -*-
"""
Widefield ESR

Created on October 13th, 2023

@author: mccambria
"""

import matplotlib.pyplot as plt
from qm import QuantumMachinesManager, qua
from qm.simulate import SimulationConfig

from servers.timing.sequencelibrary.QM_opx import seq_utils
from servers.timing.sequencelibrary.QM_opx.camera import base_scc_sequence
from utils import common
from utils import tool_belt as tb
from utils.constants import NVSpinState


def get_seq(
    base_scc_seq_args,
    step_vals,
    num_reps=1,
):
    step_vals = [seq_utils.convert_ns_to_cc(el) for el in step_vals]
    # print(step_vals)
    with qua.program() as seq:
        ### init
        seq_utils.init()
        seq_utils.macro_run_aods()

        step_val = qua.declare(int)

        def uwave_macro_ref(uwave_ind_list, step_val):
            qua.align()
            qua.wait(step_val)

        def uwave_macro_sig(uwave_ind_list, step_val):
            qua.align()
            qua.wait(step_val)
            seq_utils.macro_pi_pulse(uwave_ind_list)

        with qua.for_each_(step_val, step_vals):
            base_scc_sequence.macro(
                base_scc_seq_args,
                [uwave_macro_sig, uwave_macro_ref],
                step_val,
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
                [[107.689, 107.586], [107.405, 105.849]],
                [164, 144],
                [1.0, 1.0],
                [[72.414, 73.124], [72.145, 71.713]],
                [88, 80],
                [1.0, 1.0],
                [False, False],
                [0, 1],
            ],
            [
                2692.0,
                19512.0,
                52528.0,
                1640.0,
                11892.0,
                7244.0,
                380740.0,
                86192.0,
                4416.0,
                232048.0,
                141424.0,
                1000.0,
                1025028.0,
                12189868.0,
            ],
        )

        sim_config = SimulationConfig(duration=int(400e3 / 4))
        sim = opx.simulate(seq, sim_config)
        samples = sim.get_simulated_samples()
        samples.con1.plot()
        plt.show(block=True)

    except Exception as exc:
        raise exc
    finally:
        qmm.close_all_quantum_machines()
